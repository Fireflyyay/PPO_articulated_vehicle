import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from shapely.geometry import LinearRing, Polygon, MultiPolygon
from typing import List, Optional, Tuple
import argparse
import glob

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vehicle import Status
import configs as cfg


def _normalize_status(status_obj) -> Optional[Status]:
    """Best-effort normalization for status values coming from env info."""
    if isinstance(status_obj, Status):
        return status_obj
    if isinstance(status_obj, str):
        # Accept: "Status.OUTTIME", "OUTTIME", etc.
        raw = status_obj.strip()
        if raw.startswith("Status."):
            raw = raw.split("Status.", 1)[1]
        try:
            return Status[raw]
        except Exception:
            return None
    return None


def _episode_result_label(
    last_info: Optional[dict],
    terminated: bool,
    truncated: bool,
    forced_break_reason: Optional[str] = None,
) -> Tuple[bool, str]:
    """Return (success, label_text) to be drawn on the figure."""
    if forced_break_reason:
        return False, f"FAILURE: {forced_break_reason}"

    status = None
    if isinstance(last_info, dict):
        status = _normalize_status(last_info.get('status'))

    if status == Status.ARRIVED:
        return True, "SUCCESS"
    if status == Status.COLLIDED:
        return False, "FAILURE: collision"
    if status == Status.OUTBOUND:
        return False, "FAILURE: out of bounds"
    if status == Status.OUTTIME:
        return False, "FAILURE: timeout"

    # Fallbacks when status is missing/unexpected
    if truncated:
        return False, "FAILURE: truncated"
    if terminated:
        return False, "FAILURE: terminated"
    return False, "FAILURE: unknown"


def _find_checkpoint(default_path: str) -> Optional[str]:
    if os.path.exists(default_path):
        return default_path
    exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ckpt'))
    if not os.path.exists(exp_dir):
        return None
    for root, dirs, files in os.walk(exp_dir):
        if 'PPO_best.pt' in files:
            return os.path.join(root, 'PPO_best.pt')
    return None


def _load_checkpoint(path: str, map_location: str = 'cpu'):
    """Load checkpoint with safe-first fallback for mixed checkpoints."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def _find_latest_checkpoint(default_path: str) -> Optional[str]:
    search_roots = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../log/exp')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../ckpt')),
    ]
    candidates = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for candidate in glob.glob(os.path.join(root, '**', 'PPO_best.pt'), recursive=True):
            candidates.append(candidate)
    if not candidates:
        return default_path if os.path.exists(default_path) else None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def _extract_checkpoint_configs(checkpoint: object) -> dict:
    """Best-effort extraction of training configs from checkpoint['configs']."""
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
            v = getattr(cfg_obj, key)
            out[key] = dict(v) if isinstance(v, dict) else v
    return out


def _infer_primitive_size(npz_path: str) -> Optional[int]:
    try:
        data = np.load(npz_path, allow_pickle=True)
        actions = data['actions']
        if actions.ndim >= 1:
            return int(actions.shape[0])
    except Exception:
        return None
    return None


def _resolve_current_primitive_library(src_dir: str) -> str:
    candidate = os.path.normpath(os.path.join(src_dir, cfg.PRIMITIVE_LIBRARY_PATH))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(cfg.PRIMITIVE_LIBRARY_PATH):
        return cfg.PRIMITIVE_LIBRARY_PATH
    raise FileNotFoundError(f"Current primitive library not found: {cfg.PRIMITIVE_LIBRARY_PATH}")


def _infer_actor_output_size(checkpoint: object) -> Optional[int]:
    """Infer actor output size from checkpoint state_dict.

    Works with checkpoints saved by PPOAgent.save(params_only=True).
    """
    if not isinstance(checkpoint, dict):
        return None
    actor_sd = checkpoint.get('actor_net')
    if not isinstance(actor_sd, dict):
        return None
    # MultiObsEmbedding last linear is net.4
    w = actor_sd.get('net.4.weight')
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    # Fallback: last weight-like tensor
    weight_tensors = [v for k, v in actor_sd.items() if k.endswith('weight') and isinstance(v, torch.Tensor) and v.ndim == 2]
    if not weight_tensors:
        return None
    return int(weight_tensors[-1].shape[0])


def _resolve_visualization_mode(use_macro_actions: bool, refinement_enabled: bool) -> Tuple[str, str]:
    if not bool(use_macro_actions):
        return "continuous", "Continuous Policy"
    if bool(refinement_enabled):
        return "refined", "Refined Primitive Execution"
    return "raw", "Raw Primitive Execution"


def _get_visualization_mode(use_macro_actions: bool) -> dict:
    refinement_enabled = bool(getattr(cfg, 'USE_PRIMITIVE_REFINEMENT', False))
    mode_slug, mode_label = _resolve_visualization_mode(use_macro_actions, refinement_enabled)
    return {
        "slug": mode_slug,
        "label": mode_label,
        "policy_label": "macro-actions" if bool(use_macro_actions) else "continuous",
        "refinement_enabled": refinement_enabled,
        "refinement_applicable": bool(use_macro_actions),
    }


def _build_output_filename(episode_index: int, mode_slug: str) -> str:
    return f"path_planning_{mode_slug}_{int(episode_index)}.png"


def _build_figure_title(episode_index: int, mode_label: str) -> str:
    return f"Articulated Vehicle Path Planning - Episode {int(episode_index)} | {mode_label}"


def _count_takeover_triggers(step_infos: List[dict]) -> int:
    return int(sum(1 for info in step_infos if isinstance(info, dict) and bool(info.get('takeover_triggered', False))))


def _build_run_summary_lines(
    mode_info: dict,
    checkpoint_path: str,
    primitive_library_path: Optional[str],
    level: Optional[str],
    takeover_trigger_count: int,
) -> List[str]:
    refinement_text = "n/a"
    if bool(mode_info.get('refinement_applicable', False)):
        refinement_text = "on" if bool(mode_info.get('refinement_enabled', False)) else "off"

    lines = [
        f"mode={mode_info.get('slug', 'unknown')}",
        f"policy={mode_info.get('policy_label', 'unknown')}",
        f"refinement={refinement_text}",
        f"level={level if level is not None else getattr(cfg, 'MAP_LEVEL', 'unknown')}",
        f"takeover_triggers={int(takeover_trigger_count)}",
        f"ckpt={os.path.basename(checkpoint_path)}",
    ]
    if primitive_library_path:
        lines.append(f"library={os.path.basename(primitive_library_path)}")
    return lines

def plot_vehicle(ax, state, alpha=0.3, is_final=False):
    front_box, rear_box = state.create_box()

    # Plot front box
    x, y = front_box.xy
    color_front = 'blue' if not is_final else 'darkblue'
    ax.plot(x, y, color=color_front, alpha=alpha, linewidth=1)
    ax.fill(x, y, color=color_front, alpha=alpha/2)

    # Plot rear box
    x, y = rear_box.xy
    color_rear = 'red' if not is_final else 'darkred'
    ax.plot(x, y, color=color_rear, alpha=alpha, linewidth=1)
    ax.fill(x, y, color=color_rear, alpha=alpha/2)

def visualize(episodes: int = 10, level: Optional[str] = None, checkpoint_path: Optional[str] = None, primitive_library_path: Optional[str] = None):
    from env.car_parking_base import CarParking
    from model.agent.ppo_agent import PPOAgent as PPO

    # Setup base environment
    base_env = CarParking(render_mode='rgb_array')

    # Locate checkpoint
    default_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ckpt/PPO_best.pt'))
    if checkpoint_path is None:
        checkpoint_path = _find_latest_checkpoint(default_ckpt)
    if checkpoint_path is None:
        print(f"No checkpoint found under {os.path.dirname(default_ckpt)}. Exiting.")
        return

    # Peek checkpoint to infer whether it is discrete (macro-actions) and the actor output size.
    ckpt_obj = _load_checkpoint(checkpoint_path, map_location='cpu')
    inferred_actor_out = _infer_actor_output_size(ckpt_obj)
    ckpt_cfg = _extract_checkpoint_configs(ckpt_obj)

    # Decide whether we should use macro-action wrapper.
    # If actor outputs more than 2 dims, it's almost certainly discrete primitives.
    use_macro_actions = (inferred_actor_out is not None and inferred_actor_out > 2)

    env = base_env
    primitive_h = cfg.PRIMITIVE_H
    primitive_library_path = None
    if use_macro_actions:
        from primitives.library import load_library
        from env.wrappers.macro_action_wrapper import MacroActionWrapper

        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        lib_full_path = os.path.abspath(primitive_library_path) if primitive_library_path is not None else _resolve_current_primitive_library(src_dir)

        primitive_lib = load_library(lib_full_path)
        primitive_h = getattr(primitive_lib, 'horizon', cfg.PRIMITIVE_H)
        primitive_library_path = lib_full_path
        env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
        print(f"Using MacroActionWrapper: action_space.n={env.action_space.n}, H={primitive_h}, lib={lib_full_path}")

        if inferred_actor_out is not None and int(env.action_space.n) != int(inferred_actor_out):
            raise RuntimeError(
                f"Checkpoint actor output ({inferred_actor_out}) and primitive library size ({env.action_space.n}) mismatch. "
                f"Please provide a matched checkpoint/library pair."
            )

    # For plotting/trajectory access, always use the underlying base env.
    plot_env = base_env
    mode_info = _get_visualization_mode(use_macro_actions)
    print(
        "Visualization mode: "
        f"{mode_info['label']} "
        f"(refinement={'on' if mode_info['refinement_enabled'] else 'off'}, "
        f"applicable={'yes' if mode_info['refinement_applicable'] else 'no'})"
    )

    # Setup agent (match training-time logic)
    actor_params = dict(ckpt_cfg.get('actor_layers', cfg.ACTOR_CONFIGS))
    critic_params = dict(ckpt_cfg.get('critic_layers', cfg.CRITIC_CONFIGS))
    obs_shape = env.observation_shape if hasattr(env, 'observation_shape') else base_env.observation_shape
    actor_params['input_dim'] = int(obs_shape[0])
    critic_params['input_dim'] = int(obs_shape[0])

    if use_macro_actions:
        actor_params['output_size'] = env.action_space.n
        actor_params['use_tanh_output'] = False
    else:
        actor_params['output_size'] = env.action_space.shape[0]

    configs = {
        "discrete": bool(ckpt_cfg.get('discrete', use_macro_actions)),
        "observation_shape": obs_shape,
        "action_dim": env.action_space.n if use_macro_actions else env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": ckpt_cfg.get('dist_type', "gaussian"),
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
        "load_params": True,
        "gamma": float(ckpt_cfg.get('gamma', (cfg.GAMMA_BASE ** primitive_h) if use_macro_actions else cfg.GAMMA)),
    }

    agent = PPO(configs, discrete=use_macro_actions, load_params=True)
    agent.load(checkpoint_path, params_only=True)
    print(f"Loaded checkpoint from {checkpoint_path}")

    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../img'))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if level is None:
        print(f"Scene level: {cfg.MAP_LEVEL}")
    else:
        print(f"Scene level (override): {level}")

    for i in range(int(episodes)):
        if level is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(options={"level": level})
        done = False
        last_info = {}
        last_terminated = False
        last_truncated = False
        forced_break_reason = None
        step_infos = []

        # Run episode
        while not done:
            action, _ = agent.choose_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info
            step_infos.append(info if isinstance(info, dict) else {})
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            if len(plot_env.vehicle.trajectory) > 2000: # Safety break
                forced_break_reason = "max steps reached"
                break

        states = plot_env.vehicle.trajectory
        if len(states) == 0:
            print(f"Episode {i + 1}: empty trajectory, skipping figure generation.")
            continue

        takeover_trigger_count = _count_takeover_triggers(step_infos)
        summary_lines = _build_run_summary_lines(
            mode_info=mode_info,
            checkpoint_path=checkpoint_path,
            primitive_library_path=primitive_library_path,
            level=level,
            takeover_trigger_count=takeover_trigger_count,
        )

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot obstacles (support LinearRing / Polygon / MultiPolygon)
        for area in plot_env.map.obstacles:
            geom = area.shape
            if isinstance(geom, LinearRing):
                x, y = geom.xy
                ax.plot(x, y, color='black', linewidth=2)
                ax.fill(x, y, color='gray', alpha=0.4)
            elif isinstance(geom, Polygon):
                x, y = geom.exterior.xy
                ax.plot(x, y, color='black', linewidth=2)
                ax.fill(x, y, color='gray', alpha=0.4)
                # holes = drivable area; paint as white for clarity
                for interior in list(geom.interiors):
                    xi, yi = interior.xy
                    ax.plot(xi, yi, color='black', linewidth=1, alpha=0.7)
                    ax.fill(xi, yi, color='white', alpha=1.0)
            elif isinstance(geom, MultiPolygon):
                for g in geom.geoms:
                    x, y = g.exterior.xy
                    ax.plot(x, y, color='black', linewidth=2)
                    ax.fill(x, y, color='gray', alpha=0.4)
                    for interior in list(g.interiors):
                        xi, yi = interior.xy
                        ax.plot(xi, yi, color='black', linewidth=1, alpha=0.7)
                        ax.fill(xi, yi, color='white', alpha=1.0)

        # Plot target (destination)
        dest_front, dest_rear = plot_env.map.dest.create_box()
        xf, yf = dest_front.xy
        ax.plot(xf, yf, color='green', linestyle='--', linewidth=2, label='Target Front')
        xr, yr = dest_rear.xy
        ax.plot(xr, yr, color='darkgreen', linestyle='--', linewidth=2, label='Target Rear')

        # Plot path (trajectory of the front center)
        path_x = [s.loc.x for s in states]
        path_y = [s.loc.y for s in states]
        ax.plot(path_x, path_y, color='cyan', linestyle='-', alpha=0.6, linewidth=1, label='Path')

        # Plot vehicle at intervals
        # We want to show about 10 intermediate states
        num_intermediate = 10
        interval = max(1, len(states) // num_intermediate)
        for j in range(0, len(states), interval):
            plot_vehicle(ax, states[j], alpha=0.2)

        # Plot start state
        plot_vehicle(ax, states[0], alpha=0.5)

        # Plot final state
        plot_vehicle(ax, states[-1], alpha=1.0, is_final=True)

        ax.set_aspect('equal')
        ax.set_xlim(plot_env.map.xmin, plot_env.map.xmax)
        ax.set_ylim(plot_env.map.ymin, plot_env.map.ymax)
        ax.set_title(_build_figure_title(i + 1, mode_info['label']))

        # Annotate success / failure (and reason in English)
        success, label = _episode_result_label(
            last_info=last_info,
            terminated=last_terminated,
            truncated=last_truncated,
            forced_break_reason=forced_break_reason,
        )
        label_color = 'green' if success else 'red'
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=14,
            fontweight='bold',
            color=label_color,
            bbox=dict(facecolor='white', edgecolor=label_color, alpha=0.9, boxstyle='round,pad=0.3'),
        )
        ax.text(
            0.02,
            0.02,
            " | ".join(summary_lines),
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=9,
            color='black',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round,pad=0.25'),
        )
        ax.grid(True, linestyle=':', alpha=0.5)

        save_path = os.path.join(img_dir, _build_output_filename(i + 1, mode_info['slug']))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--level', type=str, default=None, choices=["Warmup", "Normal", "Complex", "Extrem"])
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint path to visualize')
    parser.add_argument('--primitive-library', type=str, default=None, help='Optional primitive library npz path')
    args = parser.parse_args()
    visualize(
        episodes=args.episodes,
        level=args.level,
        checkpoint_path=args.checkpoint,
        primitive_library_path=args.primitive_library,
    )
