import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import configs as cfg
import env.car_parking_base as car_parking_module
from env.car_parking_base import CarParking
from env.vehicle import Status
from model.agent.parking_agent import ParkingAgent, PrimitivePlanner
from model.agent.ppo_agent import PPOAgent as PPO


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


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
        return int(data['actions'].shape[0])
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

    if not use_macro_actions:
        raise RuntimeError('This evaluation expects a macro-action checkpoint with a primitive library.')

    from primitives.library import load_library

    preferred_lib = _resolve_adaptive_library_from_checkpoint_dir(checkpoint_path)
    expected_action_dim = int(ckpt_cfg.get('action_dim', inferred_actor_out))
    lib_path = _find_matching_primitive_library(expected_action_dim, preferred_path=preferred_lib)
    if lib_path is None:
        raise RuntimeError(f'Failed to locate primitive library with size {expected_action_dim}')

    primitive_lib = load_library(lib_path)
    primitive_h = getattr(primitive_lib, 'horizon', cfg.PRIMITIVE_H)

    actor_params = dict(ckpt_cfg.get('actor_layers', cfg.ACTOR_CONFIGS))
    critic_params = dict(ckpt_cfg.get('critic_layers', cfg.CRITIC_CONFIGS))
    obs_shape = (cfg.LIDAR_NUM + 7 + 2 + (cfg.GUIDANCE_FEATURE_DIM if cfg.ENABLE_GLOBAL_SOFT_GUIDANCE else 0),)
    actor_params['input_dim'] = int(obs_shape[0])
    critic_params['input_dim'] = int(obs_shape[0])
    actor_params['output_size'] = int(primitive_lib.size)
    actor_params['use_tanh_output'] = False

    configs = {
        'discrete': True,
        'observation_shape': obs_shape,
        'action_dim': int(primitive_lib.size),
        'hidden_size': 64,
        'activation': 'tanh',
        'dist_type': ckpt_cfg.get('dist_type', 'gaussian'),
        'save_params': False,
        'actor_layers': actor_params,
        'critic_layers': critic_params,
        'load_params': True,
        'gamma': float(ckpt_cfg.get('gamma', (cfg.GAMMA_BASE ** primitive_h))),
    }

    rl_agent = PPO(configs, discrete=True, load_params=True)
    rl_agent.load(checkpoint_path, params_only=True)
    planner = PrimitivePlanner()
    parking_agent = ParkingAgent(rl_agent, planner=planner)
    return parking_agent, primitive_lib, primitive_h


def _build_env(level: str, primitive_lib, primitive_h: int, refinement_enabled: bool, planning_dist: float):
    cfg.TAKEOVER_ENABLE = False
    cfg.USE_PRIMITIVE_REFINEMENT = bool(refinement_enabled)
    car_parking_module.NAVIGATION_PRELOAD_ALL_LEVEL_MAPS = False
    car_parking_module.NAVIGATION_SCENE_POOL_ENABLE = False
    car_parking_module.NAVIGATION_SCENE_POOL_PREFILL_ON_INIT = False

    base_env = CarParking(render_mode='rgb_array', fps=100, verbose=False)

    from env.wrappers.macro_action_wrapper import MacroActionWrapper

    env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
    env.takeover_dist = float(max(float(env.takeover_dist), float(planning_dist)))
    return env


def _flatten_low_level_actions(step_infos: Sequence[dict]) -> np.ndarray:
    chunks = []
    for info in step_infos:
        if not isinstance(info, dict):
            continue
        tr = info.get('macro_exec_trace', None)
        if not isinstance(tr, dict):
            continue
        sub_actions = tr.get('sub_actions_phys', None)
        if sub_actions is None:
            continue
        arr = np.asarray(sub_actions, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
            chunks.append(arr)
    if len(chunks) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _path_length_from_trajectory(trajectory) -> float:
    if trajectory is None or len(trajectory) < 2:
        return 0.0
    total = 0.0
    for prev_state, curr_state in zip(trajectory[:-1], trajectory[1:]):
        dx = float(curr_state.loc.x - prev_state.loc.x)
        dy = float(curr_state.loc.y - prev_state.loc.y)
        total += math.hypot(dx, dy)
    return float(total)


def _reverse_ratio(actions_phys: np.ndarray) -> float:
    if actions_phys.shape[0] == 0:
        return 0.0
    return float(np.mean(actions_phys[:, 1] < 0.0))


def _steering_change_rate(actions_phys: np.ndarray) -> float:
    if actions_phys.shape[0] < 2:
        return 0.0
    steer = np.asarray(actions_phys[:, 0], dtype=np.float64)
    return float(np.mean(np.abs(np.diff(steer))))


def _curvature_change_rate(trajectory) -> float:
    if trajectory is None or len(trajectory) < 3:
        return 0.0
    curvatures = []
    for prev_state, curr_state in zip(trajectory[:-1], trajectory[1:]):
        ds = math.hypot(float(curr_state.loc.x - prev_state.loc.x), float(curr_state.loc.y - prev_state.loc.y))
        if ds < 1e-6:
            curvatures.append(0.0)
            continue
        dtheta = _wrap_pi(float(curr_state.heading - prev_state.heading))
        curvatures.append(float(dtheta / ds))
    if len(curvatures) < 2:
        return 0.0
    curvatures = np.asarray(curvatures, dtype=np.float64)
    return float(np.mean(np.abs(np.diff(curvatures))))


def _terminal_overlap_metrics(final_state, goal_state) -> Tuple[float, float, float]:
    final_boxes = final_state.create_box()
    goal_boxes = goal_state.create_box()
    overlaps = []
    for final_box, goal_box in zip(final_boxes, goal_boxes):
        poly_final = Polygon(final_box)
        poly_goal = Polygon(goal_box)
        area_goal = float(poly_goal.area) + 1e-9
        inter = float(poly_final.intersection(poly_goal).area)
        overlaps.append(float(inter / area_goal))
    if len(overlaps) == 0:
        return 0.0, 0.0, 0.0
    front_overlap = float(overlaps[0])
    rear_overlap = float(overlaps[1]) if len(overlaps) > 1 else float(overlaps[0])
    return front_overlap, rear_overlap, float(np.mean(overlaps))


def _terminal_overlap_ratio(final_state, goal_state) -> float:
    _, _, mean_overlap = _terminal_overlap_metrics(final_state, goal_state)
    return float(mean_overlap)


def _terminal_heading_error_deg(final_state, goal_state) -> float:
    return float(np.degrees(abs(_wrap_pi(float(final_state.heading - goal_state.heading)))))


def _terminal_position_error(final_state, goal_state) -> float:
    return float(final_state.loc.distance(goal_state.loc))


def _terminal_articulation_error_deg(final_state, goal_state) -> float:
    final_articulation = _wrap_pi(float(final_state.heading) - float(final_state.rear_heading))
    goal_articulation = _wrap_pi(float(goal_state.heading) - float(goal_state.rear_heading))
    return float(np.degrees(abs(_wrap_pi(final_articulation - goal_articulation))))


def _precise_terminal_ok(final_state, goal_state) -> bool:
    heading_tol_deg = float(getattr(cfg, 'PRIMITIVE_REFINEMENT_FINAL_HEADING_TOL_DEG', 5.0))
    overlap_target = float(getattr(cfg, 'PRIMITIVE_REFINEMENT_FINAL_OVERLAP_TARGET', 0.75))
    heading_error = _terminal_heading_error_deg(final_state, goal_state)
    _, _, mean_overlap = _terminal_overlap_metrics(final_state, goal_state)
    return bool(heading_error <= heading_tol_deg and mean_overlap >= overlap_target)


def _front_terminal_ok(final_state, goal_state) -> bool:
    heading_tol_deg = float(getattr(cfg, 'PRIMITIVE_REFINEMENT_FRONT_TERMINAL_OK_HEADING_TOL_DEG', 3.0))
    overlap_target = float(getattr(cfg, 'PRIMITIVE_REFINEMENT_FRONT_TERMINAL_OK_OVERLAP_TARGET', 0.80))
    heading_error = _terminal_heading_error_deg(final_state, goal_state)
    front_overlap, _, _ = _terminal_overlap_metrics(final_state, goal_state)
    return bool(heading_error <= heading_tol_deg and front_overlap >= overlap_target)


@dataclass
class EpisodeMetrics:
    seed: int
    level: str
    refinement_enabled: bool
    success: bool
    step_count_macro: int
    plan_trigger_count: int
    path_length: float
    reverse_ratio: float
    steering_change_rate: float
    curvature_change_rate: float
    terminal_position_error: float
    terminal_heading_error_deg: float
    terminal_articulation_error_deg: float
    terminal_front_overlap_ratio: float
    terminal_rear_overlap_ratio: float
    terminal_overlap_ratio: float
    front_terminal_ok: bool
    precise_terminal_ok: bool
    final_status: str


def _maybe_prepare_plan(env, parking_agent: ParkingAgent, obs: np.ndarray, planning_dist: float, max_plan_len: int):
    if parking_agent.executing_plan:
        return None
    if not hasattr(env, 'plan_to_dest') or not hasattr(env, 'prepare_plan_execution'):
        return None

    try:
        goal = env._parse_goal_repr_from_obs(obs)
        dist = float(goal.get('dist', 1e9))
    except Exception:
        dist = 1e9
    if dist > float(planning_dist):
        return None

    plan = env.plan_to_dest(max_len=int(max_plan_len))
    if plan is None or len(plan) == 0:
        return None

    debug = env.prepare_plan_execution(plan, prefix_steps=None, source='external')
    parking_agent.set_planner_path(plan, forced=True)
    return debug


def _run_episode(
    checkpoint_path: str,
    level: str,
    seed: int,
    refinement_enabled: bool,
    planning_dist: float,
    plan_max_len: int,
) -> EpisodeMetrics:
    _set_global_seed(seed)
    parking_agent, primitive_lib, primitive_h = _build_agent_and_library(checkpoint_path)
    env = _build_env(level, primitive_lib, primitive_h, refinement_enabled=refinement_enabled, planning_dist=planning_dist)
    obs, _ = env.reset(seed=int(seed), options={'level': str(level)})
    parking_agent.reset()

    done = False
    step_num = 0
    plan_trigger_count = 0
    step_infos: List[dict] = []
    final_info: Dict = {}

    while not done:
        plan_debug = _maybe_prepare_plan(env, parking_agent, obs, planning_dist=planning_dist, max_plan_len=plan_max_len)
        if isinstance(plan_debug, dict):
            plan_trigger_count += 1

        step_num += 1
        action_mask = None
        if getattr(cfg, 'USE_ACTION_MASK', False) and hasattr(env, 'get_action_mask'):
            action_mask = env.get_action_mask(obs)

        action, _ = parking_agent.choose_action(obs, deterministic=True, action_mask=action_mask)
        next_obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_infos.append(info if isinstance(info, dict) else {})
        final_info = info if isinstance(info, dict) else {}
        obs = next_obs

    base_env = env.unwrapped
    actions_phys = _flatten_low_level_actions(step_infos)
    trajectory = list(getattr(base_env.vehicle, 'trajectory', []) or [])
    final_state = getattr(base_env.vehicle, 'state', None)
    goal_state = getattr(base_env.map, 'dest', None)

    path_length = _path_length_from_trajectory(trajectory)
    reverse_ratio = _reverse_ratio(actions_phys)
    steering_change_rate = _steering_change_rate(actions_phys)
    curvature_change_rate = _curvature_change_rate(trajectory)
    position_error = _terminal_position_error(final_state, goal_state) if (final_state is not None and goal_state is not None) else float('nan')
    heading_error = _terminal_heading_error_deg(final_state, goal_state) if (final_state is not None and goal_state is not None) else float('nan')
    articulation_error = _terminal_articulation_error_deg(final_state, goal_state) if (final_state is not None and goal_state is not None) else float('nan')
    front_overlap, rear_overlap, overlap_ratio = _terminal_overlap_metrics(final_state, goal_state) if (final_state is not None and goal_state is not None) else (float('nan'), float('nan'), float('nan'))
    front_terminal_ok = _front_terminal_ok(final_state, goal_state) if (final_state is not None and goal_state is not None) else False
    precise_terminal_ok = _precise_terminal_ok(final_state, goal_state) if (final_state is not None and goal_state is not None) else False
    success = final_info.get('status', None) == Status.ARRIVED

    try:
        env.close()
    except Exception:
        pass

    return EpisodeMetrics(
        seed=int(seed),
        level=str(level),
        refinement_enabled=bool(refinement_enabled),
        success=bool(success),
        step_count_macro=int(step_num),
        plan_trigger_count=int(plan_trigger_count),
        path_length=float(path_length),
        reverse_ratio=float(reverse_ratio),
        steering_change_rate=float(steering_change_rate),
        curvature_change_rate=float(curvature_change_rate),
        terminal_position_error=float(position_error),
        terminal_heading_error_deg=float(heading_error),
        terminal_articulation_error_deg=float(articulation_error),
        terminal_front_overlap_ratio=float(front_overlap),
        terminal_rear_overlap_ratio=float(rear_overlap),
        terminal_overlap_ratio=float(overlap_ratio),
        front_terminal_ok=bool(front_terminal_ok),
        precise_terminal_ok=bool(precise_terminal_ok),
        final_status=str(final_info.get('status', 'UNKNOWN')),
    )


def _aggregate(episodes: Sequence[EpisodeMetrics]) -> Dict[str, Optional[float]]:
    episodes = list(episodes)
    return {
        'episodes': int(len(episodes)),
        'success_rate': _safe_mean([1.0 if ep.success else 0.0 for ep in episodes]),
        'avg_path_length': _safe_mean([ep.path_length for ep in episodes]),
        'avg_reverse_ratio': _safe_mean([ep.reverse_ratio for ep in episodes]),
        'avg_steering_change_rate': _safe_mean([ep.steering_change_rate for ep in episodes]),
        'avg_curvature_change_rate': _safe_mean([ep.curvature_change_rate for ep in episodes]),
        'avg_terminal_position_error': _safe_mean([ep.terminal_position_error for ep in episodes]),
        'avg_terminal_heading_error_deg': _safe_mean([ep.terminal_heading_error_deg for ep in episodes]),
        'avg_terminal_articulation_error_deg': _safe_mean([ep.terminal_articulation_error_deg for ep in episodes]),
        'avg_terminal_front_overlap_ratio': _safe_mean([ep.terminal_front_overlap_ratio for ep in episodes]),
        'avg_terminal_rear_overlap_ratio': _safe_mean([ep.terminal_rear_overlap_ratio for ep in episodes]),
        'avg_terminal_overlap_ratio': _safe_mean([ep.terminal_overlap_ratio for ep in episodes]),
        'front_terminal_ok_rate': _safe_mean([1.0 if ep.front_terminal_ok else 0.0 for ep in episodes]),
        'precise_terminal_ok_rate': _safe_mean([1.0 if ep.precise_terminal_ok else 0.0 for ep in episodes]),
        'avg_plan_trigger_count': _safe_mean([ep.plan_trigger_count for ep in episodes]),
    }


def _paired_delta(off_eps: Sequence[EpisodeMetrics], on_eps: Sequence[EpisodeMetrics]) -> Dict[str, Optional[float]]:
    off_map = {(ep.level, ep.seed): ep for ep in off_eps}
    on_map = {(ep.level, ep.seed): ep for ep in on_eps}
    shared_keys = sorted(set(off_map.keys()) & set(on_map.keys()))
    if len(shared_keys) == 0:
        return {'paired_count': 0}

    def collect(metric_getter):
        vals = []
        for key in shared_keys:
            vals.append(float(metric_getter(on_map[key]) - metric_getter(off_map[key])))
        return float(np.mean(vals)) if len(vals) > 0 else None

    return {
        'paired_count': int(len(shared_keys)),
        'delta_success_rate': collect(lambda ep: 1.0 if ep.success else 0.0),
        'delta_avg_path_length': collect(lambda ep: ep.path_length),
        'delta_avg_reverse_ratio': collect(lambda ep: ep.reverse_ratio),
        'delta_avg_steering_change_rate': collect(lambda ep: ep.steering_change_rate),
        'delta_avg_curvature_change_rate': collect(lambda ep: ep.curvature_change_rate),
        'delta_avg_terminal_position_error': collect(lambda ep: ep.terminal_position_error),
        'delta_avg_terminal_heading_error_deg': collect(lambda ep: ep.terminal_heading_error_deg),
        'delta_avg_terminal_articulation_error_deg': collect(lambda ep: ep.terminal_articulation_error_deg),
        'delta_avg_terminal_front_overlap_ratio': collect(lambda ep: ep.terminal_front_overlap_ratio),
        'delta_avg_terminal_rear_overlap_ratio': collect(lambda ep: ep.terminal_rear_overlap_ratio),
        'delta_avg_terminal_overlap_ratio': collect(lambda ep: ep.terminal_overlap_ratio),
        'delta_front_terminal_ok_rate': collect(lambda ep: 1.0 if ep.front_terminal_ok else 0.0),
        'delta_precise_terminal_ok_rate': collect(lambda ep: 1.0 if ep.precise_terminal_ok else 0.0),
    }


def _write_markdown(summary: dict, path: str) -> None:
    def fmt(value):
        if value is None:
            return 'N/A'
        return f'{float(value):.4f}'

    level = summary['level']
    lines = []
    lines.append('# Refinement Quality Evaluation')
    lines.append('')
    lines.append(f"- checkpoint: {summary['checkpoint_path']}")
    lines.append(f"- level: {level}")
    lines.append(f"- episodes: {summary['episodes']}")
    lines.append(f"- base seed: {summary['base_seed']}")
    lines.append(f"- planning_dist: {summary['planning_dist']}")
    lines.append(f"- plan_max_len: {summary['plan_max_len']}")
    lines.append('')
    lines.append('## Metrics')
    lines.append('')
    lines.append('| Mode | Success rate | Avg path length | Reverse ratio | Steering change rate | Curvature change rate | Terminal heading err (deg) | Terminal overlap | Avg plan triggers |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for mode_key in ('refinement_off', 'refinement_on'):
        item = summary['modes'][mode_key]
        lines.append(
            f"| {mode_key} | {fmt(item['success_rate'])} | {fmt(item['avg_path_length'])} | {fmt(item['avg_reverse_ratio'])} | {fmt(item['avg_steering_change_rate'])} | {fmt(item['avg_curvature_change_rate'])} | {fmt(item['avg_terminal_heading_error_deg'])} | {fmt(item['avg_terminal_overlap_ratio'])} | {fmt(item['avg_plan_trigger_count'])} |"
        )
    lines.append('')
    lines.append('## Terminal Precision')
    lines.append('')
    lines.append('| Mode | Terminal pos err | Terminal articulation err (deg) | Front overlap | Rear overlap | Mean overlap | Front terminal ok | Precise terminal ok |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for mode_key in ('refinement_off', 'refinement_on'):
        item = summary['modes'][mode_key]
        lines.append(
            f"| {mode_key} | {fmt(item['avg_terminal_position_error'])} | {fmt(item['avg_terminal_articulation_error_deg'])} | {fmt(item['avg_terminal_front_overlap_ratio'])} | {fmt(item['avg_terminal_rear_overlap_ratio'])} | {fmt(item['avg_terminal_overlap_ratio'])} | {fmt(item['front_terminal_ok_rate'])} | {fmt(item['precise_terminal_ok_rate'])} |"
        )
    lines.append('')
    lines.append('## Paired Delta')
    lines.append('')
    delta = summary['paired_delta']
    lines.append(f"- paired_count: {delta.get('paired_count', 0)}")
    lines.append(f"- delta_success_rate: {fmt(delta.get('delta_success_rate'))}")
    lines.append(f"- delta_avg_path_length: {fmt(delta.get('delta_avg_path_length'))}")
    lines.append(f"- delta_avg_reverse_ratio: {fmt(delta.get('delta_avg_reverse_ratio'))}")
    lines.append(f"- delta_avg_steering_change_rate: {fmt(delta.get('delta_avg_steering_change_rate'))}")
    lines.append(f"- delta_avg_curvature_change_rate: {fmt(delta.get('delta_avg_curvature_change_rate'))}")
    lines.append(f"- delta_avg_terminal_position_error: {fmt(delta.get('delta_avg_terminal_position_error'))}")
    lines.append(f"- delta_avg_terminal_heading_error_deg: {fmt(delta.get('delta_avg_terminal_heading_error_deg'))}")
    lines.append(f"- delta_avg_terminal_articulation_error_deg: {fmt(delta.get('delta_avg_terminal_articulation_error_deg'))}")
    lines.append(f"- delta_avg_terminal_front_overlap_ratio: {fmt(delta.get('delta_avg_terminal_front_overlap_ratio'))}")
    lines.append(f"- delta_avg_terminal_rear_overlap_ratio: {fmt(delta.get('delta_avg_terminal_rear_overlap_ratio'))}")
    lines.append(f"- delta_avg_terminal_overlap_ratio: {fmt(delta.get('delta_avg_terminal_overlap_ratio'))}")
    lines.append(f"- delta_front_terminal_ok_rate: {fmt(delta.get('delta_front_terminal_ok_rate'))}")
    lines.append(f"- delta_precise_terminal_ok_rate: {fmt(delta.get('delta_precise_terminal_ok_rate'))}")

    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def evaluate(
    level: str,
    episodes: int,
    base_seed: int,
    checkpoint_path: str,
    planning_dist: float,
    plan_max_len: int,
) -> dict:
    off_eps: List[EpisodeMetrics] = []
    on_eps: List[EpisodeMetrics] = []

    for idx in range(int(episodes)):
        seed = int(base_seed + idx)
        off_eps.append(
            _run_episode(
                checkpoint_path=checkpoint_path,
                level=level,
                seed=seed,
                refinement_enabled=False,
                planning_dist=planning_dist,
                plan_max_len=plan_max_len,
            )
        )
        on_eps.append(
            _run_episode(
                checkpoint_path=checkpoint_path,
                level=level,
                seed=seed,
                refinement_enabled=True,
                planning_dist=planning_dist,
                plan_max_len=plan_max_len,
            )
        )

    return {
        'checkpoint_path': os.path.abspath(checkpoint_path),
        'level': str(level),
        'episodes': int(episodes),
        'base_seed': int(base_seed),
        'planning_dist': float(planning_dist),
        'plan_max_len': int(plan_max_len),
        'modes': {
            'refinement_off': _aggregate(off_eps),
            'refinement_on': _aggregate(on_eps),
        },
        'paired_delta': _paired_delta(off_eps, on_eps),
        'episode_metrics': {
            'refinement_off': [asdict(ep) for ep in off_eps],
            'refinement_on': [asdict(ep) for ep in on_eps],
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Compare path quality with refinement enabled vs disabled.')
    parser.add_argument('--level', type=str, default='Normal', choices=['Normal', 'Complex', 'Extrem'])
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--base-seed', type=int, default=20260410)
    parser.add_argument('--planning-dist', type=float, default=6.0)
    parser.add_argument('--plan-max-len', type=int, default=6)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = _find_checkpoint(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ckpt', 'PPO_best.pt')))
    if checkpoint_path is None:
        raise RuntimeError('Failed to locate PPO checkpoint')

    if args.output_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'log', 'analysis', f'refinement_quality_{stamp}'))
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary = evaluate(
        level=str(args.level),
        episodes=int(args.episodes),
        base_seed=int(args.base_seed),
        checkpoint_path=checkpoint_path,
        planning_dist=float(args.planning_dist),
        plan_max_len=int(args.plan_max_len),
    )

    json_path = os.path.join(output_dir, 'summary.json')
    md_path = os.path.join(output_dir, 'report.md')
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _write_markdown(summary, md_path)

    print(json.dumps({
        'summary_json': json_path,
        'report_md': md_path,
        'level': summary['level'],
        'episodes': summary['episodes'],
        'metrics': summary['modes'],
        'paired_delta': summary['paired_delta'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()