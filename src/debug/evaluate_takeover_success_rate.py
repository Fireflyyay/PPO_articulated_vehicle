import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import configs as cfg
import env.car_parking_base as car_parking_module
from env.car_parking_base import CarParking
from env.vehicle import Status
from model.agent.parking_agent import ParkingAgent, PrimitivePlanner
from model.agent.ppo_agent import PPOAgent as PPO


def _find_checkpoint(default_path: str) -> Optional[str]:
    if os.path.exists(default_path):
        return default_path
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ckpt'))
    for root, _, files in os.walk(ckpt_dir):
        if 'PPO_best.pt' in files:
            return os.path.join(root, 'PPO_best.pt')
    return None


def _load_checkpoint(path: str, map_location: str = 'cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location)


def _extract_checkpoint_configs(checkpoint: object) -> dict:
    out = {}
    if not isinstance(checkpoint, dict):
        return out
    cfg_obj = checkpoint.get('configs', None)
    if cfg_obj is None:
        return out
    for key in ('discrete', 'observation_shape', 'action_dim', 'gamma', 'dist_type'):
        if hasattr(cfg_obj, key):
            out[key] = getattr(cfg_obj, key)
    for key in ('actor_layers', 'critic_layers'):
        if hasattr(cfg_obj, key):
            value = getattr(cfg_obj, key)
            out[key] = dict(value) if isinstance(value, dict) else value
    return out


def _infer_actor_output_size(checkpoint: object) -> Optional[int]:
    if not isinstance(checkpoint, dict):
        return None
    actor_sd = checkpoint.get('actor_net')
    if not isinstance(actor_sd, dict):
        return None
    weight = actor_sd.get('net.4.weight')
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    weight_tensors = [
        value for key, value in actor_sd.items()
        if key.endswith('weight') and isinstance(value, torch.Tensor) and value.ndim == 2
    ]
    if not weight_tensors:
        return None
    return int(weight_tensors[-1].shape[0])


def _infer_primitive_size(npz_path: str) -> Optional[int]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        actions = data['actions']
        return int(actions.shape[0])
    except Exception:
        return None


def _resolve_adaptive_library_from_checkpoint_dir(checkpoint_path: str) -> Optional[str]:
    active_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), 'adaptive_primitives', 'active_version.json')
    if not os.path.exists(active_path):
        return None
    try:
        with open(active_path, 'r', encoding='utf-8') as handle:
            obj = json.load(handle)
        version_id = str(obj.get('version_id', '')).strip()
        if not version_id:
            return None
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(checkpoint_path)),
            'adaptive_primitives',
            'versions',
            f'primitives_v{version_id}.npz',
        )
        return candidate if os.path.exists(candidate) else None
    except Exception:
        return None


def _find_matching_primitive_library(expected_size: int, preferred_path: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if preferred_path and os.path.exists(preferred_path):
        candidates.append(os.path.abspath(preferred_path))

    default_lib = os.path.normpath(os.path.join(SRC_DIR, cfg.PRIMITIVE_LIBRARY_PATH))
    if os.path.exists(default_lib):
        candidates.append(os.path.abspath(default_lib))

    exp_root = os.path.join(SRC_DIR, 'log', 'exp')
    for root, _, files in os.walk(exp_root):
        for name in files:
            if name.endswith('.npz'):
                candidates.append(os.path.abspath(os.path.join(root, name)))

    seen = set()
    matched = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        size = _infer_primitive_size(path)
        if size == int(expected_size):
            matched.append(path)
    if not matched:
        return None
    matched.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matched[0]


def _set_global_seed(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))


def _build_agent_and_library(checkpoint_path: str):
    checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
    ckpt_cfg = _extract_checkpoint_configs(checkpoint)
    inferred_actor_out = _infer_actor_output_size(checkpoint)
    use_macro_actions = bool(inferred_actor_out is not None and inferred_actor_out > 2)

    primitive_lib = None
    primitive_h = cfg.PRIMITIVE_H

    actor_params = dict(ckpt_cfg.get('actor_layers', cfg.ACTOR_CONFIGS))
    critic_params = dict(ckpt_cfg.get('critic_layers', cfg.CRITIC_CONFIGS))
    obs_shape = (cfg.LIDAR_NUM + 7 + 2 + (cfg.GUIDANCE_FEATURE_DIM if cfg.ENABLE_GLOBAL_SOFT_GUIDANCE else 0),)
    actor_params['input_dim'] = int(obs_shape[0])
    critic_params['input_dim'] = int(obs_shape[0])

    if use_macro_actions:
        from primitives.library import load_library

        preferred_lib = _resolve_adaptive_library_from_checkpoint_dir(checkpoint_path)
        expected_action_dim = int(ckpt_cfg.get('action_dim', inferred_actor_out))
        lib_path = _find_matching_primitive_library(expected_action_dim, preferred_path=preferred_lib)
        if lib_path is None:
            raise RuntimeError(f'Failed to locate primitive library with size {expected_action_dim}')
        primitive_lib = load_library(lib_path)
        primitive_h = getattr(primitive_lib, 'horizon', cfg.PRIMITIVE_H)
        actor_params['output_size'] = int(primitive_lib.size)
        actor_params['use_tanh_output'] = False
        action_dim = int(primitive_lib.size)
    else:
        actor_params['output_size'] = 2
        actor_params['use_tanh_output'] = True
        action_dim = 2

    configs = {
        'discrete': use_macro_actions,
        'observation_shape': obs_shape,
        'action_dim': action_dim,
        'hidden_size': 64,
        'activation': 'tanh',
        'dist_type': ckpt_cfg.get('dist_type', 'gaussian'),
        'save_params': False,
        'actor_layers': actor_params,
        'critic_layers': critic_params,
        'load_params': True,
        'gamma': float(ckpt_cfg.get('gamma', (cfg.GAMMA_BASE ** primitive_h) if use_macro_actions else cfg.GAMMA)),
    }

    rl_agent = PPO(configs, discrete=use_macro_actions, load_params=True)
    rl_agent.load(checkpoint_path, params_only=True)
    planner = PrimitivePlanner() if use_macro_actions else None
    parking_agent = ParkingAgent(rl_agent, planner=planner)
    return parking_agent, primitive_lib, use_macro_actions, primitive_h


def _build_env(level: str, primitive_lib, primitive_h: int, takeover_enabled: bool):
    cfg.TAKEOVER_ENABLE = bool(takeover_enabled)
    car_parking_module.NAVIGATION_PRELOAD_ALL_LEVEL_MAPS = False
    car_parking_module.NAVIGATION_SCENE_POOL_ENABLE = True
    car_parking_module.NAVIGATION_SCENE_POOL_PREFILL_ON_INIT = True

    base_env = CarParking(render_mode='rgb_array', fps=100, verbose=False)
    if primitive_lib is None:
        return base_env

    from env.wrappers.macro_action_wrapper import MacroActionWrapper

    env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
    return env


@dataclass
class EpisodeMetrics:
    seed: int
    level: str
    takeover_enabled: bool
    success: bool
    takeover_triggered: bool
    takeover_used: bool
    takeover_steps: int
    total_steps: int
    final_status: str


def _run_episode(parking_agent, primitive_lib, primitive_h: int, use_macro_actions: bool, level: str, seed: int, takeover_enabled: bool) -> EpisodeMetrics:
    _set_global_seed(seed)
    env = _build_env(level, primitive_lib, primitive_h, takeover_enabled=takeover_enabled)
    obs, _ = env.reset(options={'level': level})
    parking_agent.reset()

    done = False
    step_num = 0
    takeover_steps = 0
    takeover_triggered = False
    takeover_used = False
    final_info: Dict = {}

    while not done:
        step_num += 1
        action_mask = None
        if use_macro_actions and getattr(cfg, 'USE_ACTION_MASK', False) and hasattr(env, 'get_action_mask'):
            action_mask = env.get_action_mask(obs)

        action, _ = parking_agent.choose_action(obs, deterministic=True, action_mask=action_mask)
        next_obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        if bool(info.get('takeover_active', False)):
            takeover_steps += 1
            takeover_used = True
        if bool(info.get('takeover_triggered', False)):
            takeover_triggered = True
        if use_macro_actions and info.get('path_to_dest', None) is not None:
            parking_agent.set_planner_path(info['path_to_dest'])

        final_info = info if isinstance(info, dict) else {}
        obs = next_obs

    try:
        env.close()
    except Exception:
        pass

    success = final_info.get('status', None) == Status.ARRIVED
    final_status = str(final_info.get('status', 'UNKNOWN'))
    return EpisodeMetrics(
        seed=int(seed),
        level=str(level),
        takeover_enabled=bool(takeover_enabled),
        success=bool(success),
        takeover_triggered=bool(takeover_triggered),
        takeover_used=bool(takeover_used),
        takeover_steps=int(takeover_steps),
        total_steps=int(step_num),
        final_status=final_status,
    )


def _aggregate(episodes: List[EpisodeMetrics]) -> Dict:
    total = len(episodes)
    success_count = sum(1 for ep in episodes if ep.success)
    triggered = [ep for ep in episodes if ep.takeover_triggered]
    used = [ep for ep in episodes if ep.takeover_used]
    success_when_used = (sum(1 for ep in used if ep.success) / float(len(used))) if used else None
    success_when_triggered = (sum(1 for ep in triggered if ep.success) / float(len(triggered))) if triggered else None
    avg_step_ratio = float(np.mean([
        (ep.takeover_steps / float(max(1, ep.total_steps))) for ep in episodes
    ])) if episodes else 0.0
    return {
        'episodes': int(total),
        'success_rate': float(success_count / float(total)) if total else 0.0,
        'success_count': int(success_count),
        'triggered_episode_count': int(len(triggered)),
        'used_episode_count': int(len(used)),
        'trigger_rate': float(len(triggered) / float(total)) if total else 0.0,
        'used_rate': float(len(used) / float(total)) if total else 0.0,
        'success_when_used': success_when_used,
        'success_when_triggered': success_when_triggered,
        'avg_takeover_step_ratio': float(avg_step_ratio),
    }


def _pairwise_delta(disabled_eps: List[EpisodeMetrics], enabled_eps: List[EpisodeMetrics]) -> Dict:
    by_key_disabled = {(ep.level, ep.seed): ep for ep in disabled_eps}
    by_key_enabled = {(ep.level, ep.seed): ep for ep in enabled_eps}
    shared = sorted(set(by_key_disabled.keys()) & set(by_key_enabled.keys()))
    deltas = []
    for key in shared:
        base = by_key_disabled[key]
        take = by_key_enabled[key]
        deltas.append(int(take.success) - int(base.success))
    if not deltas:
        return {'paired_count': 0, 'mean_success_delta': None, 'improved': 0, 'worsened': 0, 'unchanged': 0}
    return {
        'paired_count': int(len(deltas)),
        'mean_success_delta': float(np.mean(deltas)),
        'improved': int(sum(1 for v in deltas if v > 0)),
        'worsened': int(sum(1 for v in deltas if v < 0)),
        'unchanged': int(sum(1 for v in deltas if v == 0)),
    }


def evaluate(levels: List[str], episodes_per_level: int, base_seed: int, checkpoint_path: str) -> Dict:
    parking_agent, primitive_lib, use_macro_actions, primitive_h = _build_agent_and_library(checkpoint_path)
    all_results: Dict[str, Dict[str, List[EpisodeMetrics]]] = {
        'disabled': {level: [] for level in levels},
        'enabled': {level: [] for level in levels},
    }

    for level_idx, level in enumerate(levels):
        for episode_idx in range(int(episodes_per_level)):
            seed = int(base_seed + level_idx * 10000 + episode_idx)
            disabled_metrics = _run_episode(
                parking_agent,
                primitive_lib,
                primitive_h,
                use_macro_actions,
                level,
                seed,
                takeover_enabled=False,
            )
            enabled_metrics = _run_episode(
                parking_agent,
                primitive_lib,
                primitive_h,
                use_macro_actions,
                level,
                seed,
                takeover_enabled=True,
            )
            all_results['disabled'][level].append(disabled_metrics)
            all_results['enabled'][level].append(enabled_metrics)

    summary = {
        'checkpoint_path': os.path.abspath(checkpoint_path),
        'episodes_per_level': int(episodes_per_level),
        'levels': list(levels),
        'base_seed': int(base_seed),
        'modes': {},
        'pairwise_delta': {},
    }

    for mode in ('disabled', 'enabled'):
        summary['modes'][mode] = {}
        for level in levels:
            summary['modes'][mode][level] = _aggregate(all_results[mode][level])

    for level in levels:
        summary['pairwise_delta'][level] = _pairwise_delta(all_results['disabled'][level], all_results['enabled'][level])

    summary['episodes'] = {
        mode: {
            level: [asdict(ep) for ep in eps]
            for level, eps in level_map.items()
        }
        for mode, level_map in all_results.items()
    }
    return summary


def _write_markdown(summary: Dict, path: str) -> None:
    lines: List[str] = []
    lines.append('# Takeover Success Rate Evaluation')
    lines.append('')
    lines.append(f"- checkpoint: {summary['checkpoint_path']}")
    lines.append(f"- levels: {', '.join(summary['levels'])}")
    lines.append(f"- episodes per level: {summary['episodes_per_level']}")
    lines.append(f"- base seed: {summary['base_seed']}")
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append('| Level | Mode | Success rate | Trigger rate | Used rate | Success when used | Avg takeover step ratio |')
    lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: |')
    for level in summary['levels']:
        for mode in ('disabled', 'enabled'):
            item = summary['modes'][mode][level]
            swu = item['success_when_used']
            swu_text = 'N/A' if swu is None else f'{swu:.4f}'
            lines.append(
                f"| {level} | {mode} | {item['success_rate']:.4f} | {item['trigger_rate']:.4f} | {item['used_rate']:.4f} | {swu_text} | {item['avg_takeover_step_ratio']:.4f} |"
            )
        delta = summary['pairwise_delta'][level]
        mean_delta = delta['mean_success_delta']
        mean_delta_text = 'N/A' if mean_delta is None else f'{mean_delta:.4f}'
        lines.append(
            f"| {level} | enabled-disabled | {mean_delta_text} | improved={delta['improved']} | worsened={delta['worsened']} | unchanged={delta['unchanged']} | paired={delta['paired_count']} |"
        )
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- `success_when_used` only counts episodes where takeover became active for at least one step.')
    lines.append('- The disabled/enabled comparison uses the same seed schedule per level to keep scene generation aligned as much as possible.')
    lines.append('- Current evaluation uses deterministic policy actions from the provided PPO checkpoint.')

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-per-level', type=int, default=15)
    parser.add_argument('--levels', nargs='+', default=['Normal', 'Complex', 'Extrem'])
    parser.add_argument('--base-seed', type=int, default=20260409)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'log', 'analysis', 'takeover_success_rate_20260409')))
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = _find_checkpoint(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ckpt', 'PPO_best.pt')))
    if checkpoint_path is None:
        raise RuntimeError('Failed to locate PPO checkpoint')

    os.makedirs(args.output_dir, exist_ok=True)
    summary = evaluate(
        levels=[str(level) for level in args.levels],
        episodes_per_level=int(args.episodes_per_level),
        base_seed=int(args.base_seed),
        checkpoint_path=checkpoint_path,
    )

    json_path = os.path.join(args.output_dir, 'summary.json')
    md_path = os.path.join(args.output_dir, 'report.md')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _write_markdown(summary, md_path)

    print(json.dumps({
        'summary_json': json_path,
        'report_md': md_path,
        'levels': summary['levels'],
        'episodes_per_level': summary['episodes_per_level'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()