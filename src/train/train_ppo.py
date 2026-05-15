
import sys
import os
# Ensure src is in path regardless of CWD
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import time
from shutil import copyfile
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

import configs as cfg
from train.lr_schedule import build_scaled_learning_rates, compute_post_expand_restore_scale
from train.adaptive_mining import build_mining_schedule, build_proxy_eval_contexts, select_replay_episodes, wrap_pi as adaptive_wrap_pi

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent
from env.car_parking_base import CarParking
from env.vehicle import VALID_SPEED, Status
from configs import *

# Primitives imports
if USE_MOTION_PRIMITIVES:
    from primitives.library import load_library
    from env.wrappers.macro_action_wrapper import MacroActionWrapper

# Adaptive primitive expansion imports (kept optional)
if USE_MOTION_PRIMITIVES:
    try:
        from primitives.adaptive_library_manager import AdaptivePrimitiveLibraryManager
        from primitives.trajectory_miner import EpisodeTrace, TrajectoryMiner
        from primitives.primitive_pruner import PrimitivePruner
        from train.adaptive_primitive_scheduler import AdaptivePrimitiveScheduler
        from reward.shaping_from_discovered_primitives import DiscoveredPrimitiveShaping
    except Exception:
        AdaptivePrimitiveLibraryManager = None
        EpisodeTrace = None
        TrajectoryMiner = None
        PrimitivePruner = None
        AdaptivePrimitiveScheduler = None
        DiscoveredPrimitiveShaping = None


def _scene_is_hard(scene_name: str) -> bool:
    return str(scene_name) in ("Complex", "Extrem", "Extreme")


def _safe_mean(xs):
    if xs is None or len(xs) == 0:
        return 0.0
    return float(np.mean(xs))


def _to_scalar(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        pass
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size > 0:
            return float(arr[0])
    except Exception:
        pass
    return float(default)


def _maybe_print_episode_heartbeat(
    verbose: bool,
    episode_idx: int,
    total_episodes: int,
    step_num: int,
    total_reward: float,
    update_count: int,
    memory_len: int,
    batch_size: int,
    heartbeat_steps: int,
    last_report_step: int,
) -> int:
    if not verbose:
        return int(last_report_step)
    if heartbeat_steps <= 0:
        return int(last_report_step)
    if step_num - last_report_step < heartbeat_steps:
        return int(last_report_step)

    print(
        f"Episode {episode_idx}/{total_episodes} in progress | "
        f"Steps: {step_num} | Reward: {total_reward:.2f} | "
        f"Updates: {update_count} | Buffer: {memory_len}/{batch_size}"
    )
    sys.stdout.flush()
    return int(step_num)


def _run_eval_episodes(env, parking_agent, n_episodes: int, scene_schedule: list, deterministic: bool = True):
    """Lightweight evaluation (no learning). Returns dict metrics."""
    succ = []
    succ_extreme = []
    lengths = []
    for k in range(int(n_episodes)):
        scene = scene_schedule[k % len(scene_schedule)]
        obs, _ = env.reset(options={'level': scene})
        parking_agent.reset()
        done = False
        step_num = 0
        while not done:
            step_num += 1
            action_mask = None
            if USE_MOTION_PRIMITIVES and USE_ACTION_MASK and hasattr(env, 'get_action_mask'):
                action_mask = env.get_action_mask(obs)
            action, _ = parking_agent.choose_action(obs, deterministic=deterministic, action_mask=action_mask)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        lengths.append(step_num)
        s = 1 if info.get('status', None) == Status.ARRIVED else 0
        succ.append(s)
        if str(scene) in ("Extrem", "Extreme"):
            succ_extreme.append(s)
    return {
        "success": _safe_mean(succ),
        "success_extreme": _safe_mean(succ_extreme),
        "avg_len": _safe_mean(lengths),
    }


def _collect_rollouts_for_mining(env, parking_agent, n_episodes: int, scene_schedule: list, deterministic: bool, start_episode_id: int = 0):
    """Collect EpisodeTrace list for mining (no learning)."""
    episodes = []
    for k in range(int(n_episodes)):
        scene = scene_schedule[k % len(scene_schedule)]
        obs, _ = env.reset(options={'level': scene})
        parking_agent.reset()

        done = False
        ep_obs = []
        ep_actions = []
        ep_low = []
        ep_rewards = []
        ep_dones = []
        ep_infos = []
        total_reward = 0.0

        while not done:
            ep_obs.append(np.asarray(obs, dtype=np.float64))
            action_mask = None
            if USE_MOTION_PRIMITIVES and USE_ACTION_MASK and hasattr(env, 'get_action_mask'):
                action_mask = env.get_action_mask(obs)
            action, _ = parking_agent.choose_action(obs, deterministic=deterministic, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_actions.append(int(info.get('resolved_family_id', action)))
            tr = info.get('macro_exec_trace', {}) if isinstance(info, dict) else {}
            sub_u = tr.get('sub_actions_phys', None)
            if sub_u is None:
                # fallback: use library primitive actions (may over-estimate executed steps)
                try:
                    sub_u = env.primitive_lib.get_actions(int(info.get('primitive_id', -1)))
                except Exception:
                    sub_u = np.zeros((1, 2), dtype=np.float64)
            ep_low.append(np.asarray(sub_u, dtype=np.float64))

            ep_rewards.append(float(reward))
            ep_dones.append(bool(done))
            ep_infos.append(info if isinstance(info, dict) else {})
            total_reward += float(reward)
            obs = next_obs

        success = bool(ep_infos[-1].get('status', None) == Status.ARRIVED) if len(ep_infos) > 0 else False
        episodes.append(
            EpisodeTrace(
                episode_id=int(start_episode_id + k),
                scene_type=str(scene),
                success=bool(success),
                total_reward=float(total_reward),
                step_count_macro=int(len(ep_actions)),
                takeover_used=False,
                observations=ep_obs,
                actions_primitive=ep_actions,
                actions_low_level=ep_low,
                rewards=ep_rewards,
                dones=ep_dones,
                infos=ep_infos,
                states_optional=None,
            )
        )
    return episodes


def _apply_learning_rate_scale(parking_agent, base_actor_lr: float, base_critic_lr: float, scale: float):
    actor_lr, critic_lr = build_scaled_learning_rates(base_actor_lr, base_critic_lr, scale)
    return parking_agent.agent.set_learning_rates(actor_lr, critic_lr)


def _log_learning_rate_state(writer, episode_idx: int, parking_agent, scale: float, restore_progress: float):
    actor_lr, critic_lr = parking_agent.agent.get_learning_rates()
    writer.add_scalar("lr/actor", float(actor_lr), episode_idx)
    writer.add_scalar("lr/critic", float(critic_lr), episode_idx)
    writer.add_scalar("lr/post_expand_scale", float(scale), episode_idx)
    writer.add_scalar("lr/post_expand_restore_progress", float(restore_progress), episode_idx)

def _get_current_library_size(env, ap_lib_mgr=None) -> int:
    if ap_lib_mgr is not None:
        try:
            return int(ap_lib_mgr.library_size)
        except Exception:
            pass
    try:
        primitive_lib = getattr(env, "primitive_lib", None)
        if primitive_lib is not None:
            return int(getattr(primitive_lib, "size"))
    except Exception:
        pass
    try:
        return int(env.action_space.n)
    except Exception:
        return 0


def _log_adaptive_library_state(writer, episode_idx: int, env, ap_lib_mgr=None, base_size: int = None, max_size: int = None):
    current_size = _get_current_library_size(env, ap_lib_mgr=ap_lib_mgr)
    writer.add_scalar("adaptive/library_size", float(current_size), episode_idx)
    writer.add_scalar("adaptive/library_size_absolute", float(current_size), episode_idx)

    if base_size is not None:
        growth = max(0, int(current_size) - int(base_size))
        writer.add_scalar("adaptive/library_size_from_base", float(growth), episode_idx)

    if max_size is not None and int(max_size) > 0:
        writer.add_scalar(
            "adaptive/library_capacity_utilization",
            float(current_size) / float(max_size),
            episode_idx,
        )


def _windowed_success_rates(scene_chooser, succ_record, window: int):
    n = min(len(scene_chooser.scene_record), len(succ_record))
    if n <= 0 or window <= 0:
        return 0.0, 0.0, 0.0, 0.0

    aligned = list(zip(scene_chooser.scene_record[-n:], succ_record[-n:]))
    recent = aligned[-int(window) :]
    previous = aligned[-2 * int(window) : -int(window)] if len(aligned) > int(window) else []

    def _rate(records, hard_only: bool) -> float:
        values = []
        for sid, success in records:
            scene_name = scene_chooser.scene_types.get(int(sid), 'Normal')
            if hard_only and not _scene_is_hard(scene_name):
                continue
            values.append(int(success))
        return _safe_mean(values)

    return _rate(recent, False), _rate(recent, True), _rate(previous, False), _rate(previous, True)


def _build_proxy_pruning_hooks(wrapper_env, proxy_contexts, config):
    contexts = list(proxy_contexts)
    if len(contexts) == 0:
        return None, None

    base_env = getattr(wrapper_env, 'env', wrapper_env)
    vehicle = getattr(base_env, 'vehicle', None)
    is_valid = getattr(wrapper_env, '_is_state_valid', None)
    if vehicle is None or getattr(vehicle, 'kinetic_model', None) is None or is_valid is None:
        return None, None

    try:
        from env.vehicle import State
    except Exception:
        return None, None

    try:
        step_time = int(NUM_STEP)
    except Exception:
        step_time = None

    progress_scale = float(getattr(config, 'AP_D_TERM', 10.0))

    def env_sampler():
        return list(contexts)

    def planner_eval_fn(context, actions_H):
        start_state = context.start_state
        try:
            s = State([
                float(start_state.get('x', 0.0)),
                float(start_state.get('y', 0.0)),
                float(start_state.get('heading', 0.0)),
                float(start_state.get('speed', 0.0)),
                float(start_state.get('steering', 0.0)),
                float(start_state.get('rear_heading', start_state.get('heading', 0.0))),
            ])
        except Exception:
            return 0.0

        goal_x, goal_y, goal_heading = context.goal_world
        initial_dist = max(float(context.initial_goal_dist), 1e-6)
        best_dist = initial_dist
        final_dist = initial_dist
        initial_heading_err = abs(adaptive_wrap_pi(goal_heading - float(start_state.get('heading', 0.0))))
        final_heading_err = initial_heading_err

        try:
            actions = np.asarray(actions_H, dtype=np.float64)
            for action in actions:
                if step_time is None:
                    s = vehicle.kinetic_model.step(s, action)
                else:
                    s = vehicle.kinetic_model.step(s, action, step_time=step_time)
                if not bool(is_valid(s)):
                    return -1.0

                cur_dist = float(np.hypot(float(goal_x) - float(s.loc.x), float(goal_y) - float(s.loc.y)))
                best_dist = min(best_dist, cur_dist)
                final_dist = cur_dist
                final_heading_err = abs(adaptive_wrap_pi(goal_heading - float(s.heading)))
        except Exception:
            return 0.0

        progress = max(0.0, initial_dist - final_dist) / max(progress_scale, 1e-6)
        best_progress = max(0.0, initial_dist - best_dist) / max(progress_scale, 1e-6)
        heading_gain = max(0.0, initial_heading_err - final_heading_err) / np.pi
        return float(context.scene_weight) * (0.55 * progress + 0.35 * best_progress + 0.10 * heading_gain)

    return env_sampler, planner_eval_fn


def _log_adaptive_round_uplift(
    writer,
    episode_idx: int,
    before_metrics: dict,
    after_metrics: dict,
    old_library_size: int,
    new_library_size: int,
):
    success_gain = float(after_metrics.get("success", 0.0) - before_metrics.get("success", 0.0))
    extreme_success_gain = float(
        after_metrics.get("success_extreme", 0.0) - before_metrics.get("success_extreme", 0.0)
    )
    added = max(0, int(new_library_size) - int(old_library_size))
    denom = float(max(1, added))

    writer.add_scalar("adaptive/validation_success_gain", success_gain, episode_idx)
    writer.add_scalar("adaptive/validation_extreme_success_gain", extreme_success_gain, episode_idx)
    writer.add_scalar("adaptive/validation_success_gain_per_added_primitive", success_gain / denom, episode_idx)
    writer.add_scalar(
        "adaptive/validation_extreme_success_gain_per_added_primitive",
        extreme_success_gain / denom,
        episode_idx,
    )


def _new_refinement_episode_stats() -> dict:
    return {
        "plan_count": 0,
        "applied_plans": 0,
        "feasible_plans": 0,
        "cost_deltas": [],
        "runtime_ms": [],
        "terminal_scales": [],
        "plan_lengths": [],
        "terminal_window_cost_after": [],
        "final_barrier_cost_after": [],
        "final_position_error_before": [],
        "final_position_error_after": [],
        "final_heading_error_deg_before": [],
        "final_heading_error_deg_after": [],
        "final_mean_overlap_before": [],
        "final_mean_overlap_after": [],
        "final_front_overlap_after": [],
        "final_rear_overlap_after": [],
    }


def _update_refinement_episode_stats(stats: dict, debug_info) -> None:
    if not isinstance(debug_info, dict):
        return

    stats["plan_count"] += 1
    if bool(debug_info.get("applied", False)):
        stats["applied_plans"] += 1
    if bool(debug_info.get("feasible", False)):
        stats["feasible_plans"] += 1

    cost_before = _to_scalar(debug_info.get("cost_before", 0.0), 0.0)
    cost_after = _to_scalar(debug_info.get("cost_after", 0.0), 0.0)
    stats["cost_deltas"].append(float(max(0.0, cost_before - cost_after)))

    runtime_ms = _to_scalar(debug_info.get("elapsed_ms", 0.0), 0.0)
    terminal_scale = _to_scalar(debug_info.get("terminal_scale", 1.0), 1.0)
    plan_length = _to_scalar(debug_info.get("plan_length", 0.0), 0.0)

    stats["runtime_ms"].append(float(max(0.0, runtime_ms)))
    stats["terminal_scales"].append(float(max(0.0, terminal_scale)))
    stats["plan_lengths"].append(float(max(0.0, plan_length)))
    stats["terminal_window_cost_after"].append(_to_scalar(debug_info.get("terminal_window_cost_after", 0.0), 0.0))
    stats["final_barrier_cost_after"].append(_to_scalar(debug_info.get("final_barrier_cost_after", 0.0), 0.0))
    stats["final_position_error_before"].append(_to_scalar(debug_info.get("final_position_error_before", 0.0), 0.0))
    stats["final_position_error_after"].append(_to_scalar(debug_info.get("final_position_error_after", 0.0), 0.0))
    stats["final_heading_error_deg_before"].append(_to_scalar(debug_info.get("final_heading_error_deg_before", 0.0), 0.0))
    stats["final_heading_error_deg_after"].append(_to_scalar(debug_info.get("final_heading_error_deg_after", 0.0), 0.0))
    stats["final_mean_overlap_before"].append(_to_scalar(debug_info.get("final_mean_overlap_before", 0.0), 0.0))
    stats["final_mean_overlap_after"].append(_to_scalar(debug_info.get("final_mean_overlap_after", 0.0), 0.0))
    stats["final_front_overlap_after"].append(_to_scalar(debug_info.get("final_front_overlap_after", 0.0), 0.0))
    stats["final_rear_overlap_after"].append(_to_scalar(debug_info.get("final_rear_overlap_after", 0.0), 0.0))


def _log_refinement_episode_stats(writer, episode_idx: int, step_num: int, enabled: bool, stats: dict) -> None:
    plan_count = int(stats.get("plan_count", 0))
    applied_plans = int(stats.get("applied_plans", 0))
    feasible_plans = int(stats.get("feasible_plans", 0))

    writer.add_scalar("refinement/enabled", float(bool(enabled)), episode_idx)
    writer.add_scalar("refinement/triggered", float(plan_count > 0), episode_idx)
    writer.add_scalar("refinement/plan_count", float(plan_count), episode_idx)
    writer.add_scalar("refinement/plan_rate", float(plan_count) / float(max(1, step_num)), episode_idx)
    writer.add_scalar("refinement/attempted_steps", float(plan_count), episode_idx)
    writer.add_scalar("refinement/applied_steps", float(applied_plans), episode_idx)
    writer.add_scalar("refinement/attempted_ratio", float(plan_count) / float(max(1, step_num)), episode_idx)
    writer.add_scalar("refinement/applied_ratio", float(applied_plans) / float(max(1, plan_count)), episode_idx)
    writer.add_scalar("refinement/applied_plan_ratio", float(applied_plans) / float(max(1, plan_count)), episode_idx)
    writer.add_scalar("refinement/feasible_ratio", float(feasible_plans) / float(max(1, plan_count)), episode_idx)
    writer.add_scalar("refinement/feasible_plan_ratio", float(feasible_plans) / float(max(1, plan_count)), episode_idx)
    writer.add_scalar("refinement/plan_length_mean", _safe_mean(stats.get("plan_lengths", [])), episode_idx)
    writer.add_scalar("refinement/cost_delta_mean", _safe_mean(stats.get("cost_deltas", [])), episode_idx)
    writer.add_scalar("refinement/cost_delta_sum", float(np.sum(stats.get("cost_deltas", []) or [0.0])), episode_idx)
    writer.add_scalar("refinement/prefix_shrink_ratio", 0.0, episode_idx)
    writer.add_scalar("refinement/prefix_shrink_steps_mean", 0.0, episode_idx)
    writer.add_scalar("refinement/runtime_ms_mean", _safe_mean(stats.get("runtime_ms", [])), episode_idx)
    writer.add_scalar("refinement/terminal_scale_mean", _safe_mean(stats.get("terminal_scales", [])), episode_idx)
    writer.add_scalar("refinement/terminal_window_cost_after_mean", _safe_mean(stats.get("terminal_window_cost_after", [])), episode_idx)
    writer.add_scalar("refinement/final_barrier_cost_after_mean", _safe_mean(stats.get("final_barrier_cost_after", [])), episode_idx)
    writer.add_scalar("refinement/final_position_error_before_mean", _safe_mean(stats.get("final_position_error_before", [])), episode_idx)
    writer.add_scalar("refinement/final_position_error_after_mean", _safe_mean(stats.get("final_position_error_after", [])), episode_idx)
    writer.add_scalar("refinement/final_heading_error_deg_before_mean", _safe_mean(stats.get("final_heading_error_deg_before", [])), episode_idx)
    writer.add_scalar("refinement/final_heading_error_deg_after_mean", _safe_mean(stats.get("final_heading_error_deg_after", [])), episode_idx)
    writer.add_scalar("refinement/final_mean_overlap_before_mean", _safe_mean(stats.get("final_mean_overlap_before", [])), episode_idx)
    writer.add_scalar("refinement/final_mean_overlap_after_mean", _safe_mean(stats.get("final_mean_overlap_after", [])), episode_idx)
    writer.add_scalar("refinement/final_front_overlap_after_mean", _safe_mean(stats.get("final_front_overlap_after", [])), episode_idx)
    writer.add_scalar("refinement/final_rear_overlap_after_mean", _safe_mean(stats.get("final_rear_overlap_after", [])), episode_idx)

class SceneChoose:
    """Failure-driven curriculum sampler (ported from HOPE).

    Strategy:
    - Warm-up: pick scenes to balance coverage (uniform by count).
    - After enough history: with 50% probability, sample scenes biased toward
      those whose recent success rate lags behind a target.
    """

    def __init__(self) -> None:
        self.scene_types = {
            0: 'Normal',
            1: 'Complex',
            2: 'Extrem',
        }

        # target success rates (can be tuned)
        self.target_success_rate = np.array([0.95, 0.95, 0.90], dtype=np.float64)

        # success_record indexed by scene_id
        self.success_record = {sid: [] for sid in self.scene_types.keys()}
        self.scene_record = []

        # curriculum parameters
        self.history_horizon = 200
        self.recent_window = 250
        self.extrem_success_band = tuple(float(x) for x in EXTREM_SUCCESS_BAND)
        self.extrem_focus_prob = float(EXTREM_SUCCESS_BAND_FOCUS_PROB)
        self.extrem_bridge_prob = float(EXTREM_SUCCESS_BAND_BRIDGE_PROB)

    def choose_case(self):
        if len(self.scene_record) < self.history_horizon:
            scene_id = self._choose_case_uniform()
        else:
            scene_id = self._choose_case_success_band()
            if scene_id is None and np.random.random() > 0.5:
                scene_id = self._choose_case_worst_perform()
            elif scene_id is None:
                scene_id = self._choose_case_uniform()

        self.scene_record.append(int(scene_id))
        return self.scene_types[int(scene_id)]

    def update_success_record(self, success: int):
        if len(self.scene_record) == 0:
            return
        sid = int(self.scene_record[-1])
        self.success_record[sid].append(int(success))

    def _choose_case_uniform(self):
        case_count = np.zeros(len(self.scene_types), dtype=np.int64)
        for i in range(min(len(self.scene_record), self.history_horizon)):
            sid = int(self.scene_record[-(i + 1)])
            case_count[sid] += 1
        return int(np.argmin(case_count))

    def _choose_case_worst_perform(self):
        success_rate = []
        for sid in sorted(self.scene_types.keys()):
            recent = self.success_record[sid][-min(self.recent_window, len(self.success_record[sid])) :]
            if len(recent) == 0:
                success_rate.append(0.0)
            else:
                success_rate.append(float(np.mean(recent)))

        fail_rate = self.target_success_rate - np.array(success_rate, dtype=np.float64)
        fail_rate = np.clip(fail_rate, 0.01, 1.0)
        fail_rate = fail_rate / np.sum(fail_rate)
        return int(np.random.choice(np.arange(len(fail_rate)), p=fail_rate))

    def _recent_success_rate(self, sid: int) -> float:
        rec = self.success_record[int(sid)]
        if len(rec) == 0:
            return 0.0
        recent = rec[-min(self.recent_window, len(rec)) :]
        return float(np.mean(recent)) if len(recent) > 0 else 0.0

    def _choose_case_success_band(self):
        extrem_sid = 2
        complex_sid = 1
        extrem_sr = self._recent_success_rate(extrem_sid)
        low, high = self.extrem_success_band

        if extrem_sr < low:
            if np.random.random() < self.extrem_bridge_prob:
                return int(complex_sid)
            return None

        if extrem_sr <= high:
            if np.random.random() < self.extrem_focus_prob:
                return int(extrem_sid)
            return None

        return None

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None)
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    verbose = args.verbose

    if args.visualize:
        base_env = CarParking(fps=100, verbose=verbose,)
    else:
        base_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')

    # Use Motion Primitives Wrapper
    env = base_env
    if USE_MOTION_PRIMITIVES:
        print(f"Using Motion Primitives from {PRIMITIVE_LIBRARY_PATH}")
        # Resolve path: assume PRIMITIVE_LIBRARY_PATH is relative to src/configs.py
        # We need absolute path or relative to CWD.
        # If running from root, and path is "../data/...", it might be wrong if logic assumes relative to src.
        # Let's try to find it.
        # If path starts with .., assume relative to src folder
        current_dir = os.path.dirname(os.path.abspath(__file__)) # src/train
        src_dir = os.path.dirname(current_dir) # src
        root_dir = os.path.dirname(src_dir) # root
        project_root = root_dir

        # In configs.py, path is "../data/..." relative to configs.py location (src/)
        # So it points to root/data/...

        # Let's resolve relative to src_dir
        lib_full_path = os.path.normpath(os.path.join(src_dir, PRIMITIVE_LIBRARY_PATH))

        if not os.path.exists(lib_full_path):
             # Try relative to CWD if failed
             if os.path.exists(PRIMITIVE_LIBRARY_PATH):
                 lib_full_path = PRIMITIVE_LIBRARY_PATH
             else:
                 # Try typical location
                 lib_full_path = os.path.join(project_root, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))

        primitive_lib = load_library(lib_full_path)
        primitive_h = getattr(primitive_lib, 'horizon', PRIMITIVE_H)
        env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
        print(f"Wrapped env with MacroActionWrapper. Action space: {env.action_space.n} families. H={primitive_h}")

    scene_chooser = SceneChoose()

    # the path to log and save model
    # Use src/log/exp directory
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/train
    src_dir = os.path.dirname(current_dir) # src
    log_exp_dir = os.path.join(src_dir, 'log', 'exp')

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = os.path.join(log_exp_dir, 'ppo_%s/' % timestamp)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    # configs log
    if os.path.exists('./src/configs.py'):
        copyfile('./src/configs.py', save_path+'configs.txt')
    elif os.path.exists('./configs.py'):
        copyfile('./configs.py', save_path+'configs.txt')

    # More robust tensorboard command for Python 3.8 environments
    print(f"You can track the training process with:\n  python -m tensorboard --logdir {os.path.abspath(save_path)}\nThen open http://localhost:6006 in your browser.")

    seed = SEED
    # env.seed(seed)

    # Fix for gym seeding
    # env.action_space.seed(seed)
    # Wrapper might not forward logic or attribute
    if hasattr(env.action_space, 'seed'):
        env.action_space.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Update Output Size for Discrete
    # NOTE: keep per-run copies to avoid mutating global configs.
    actor_params = dict(ACTOR_CONFIGS)
    critic_params = dict(CRITIC_CONFIGS)

    motion_action_dim = int(getattr(primitive_lib, 'action_dim', env.action_space.n)) if USE_MOTION_PRIMITIVES else None

    if USE_MOTION_PRIMITIVES:
        actor_params['output_size'] = int(motion_action_dim)
        # Discrete policy uses logits; do NOT tanh-clip.
        actor_params['use_tanh_output'] = False
        # Critic input dim doesn't change (observation same)
    else:
        actor_params['output_size'] = env.action_space.shape[0]
        actor_params['use_tanh_output'] = True

    configs = {
        "discrete": USE_MOTION_PRIMITIVES,
        "observation_shape": env.observation_shape if hasattr(env, 'observation_shape') else base_env.observation_shape,
        # env.observation_shape might not be exposed by wrapper.
        # CarParking has observation_shape attribute.
        # Gym Wrapper forwards getattr usually, but let's be safe.

        "action_dim": int(motion_action_dim) if USE_MOTION_PRIMITIVES else env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian", # This might be ignored if discrete is True in Agent
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
        "action_std_init": 1.5, # Increased from 0.6
        "action_std_decay_rate": 0.001, # Decreased from 0.001 to slow down decay
        "min_action_std": 0.1,
        "use_imitation_loss": False,
        "imitation_buffer_size": int(IMITATION_BUFFER_SIZE),
        "imitation_batch_size": int(IMITATION_BATCH_SIZE),
        "imitation_min_buffer": int(IMITATION_MIN_BUFFER),
        "imitation_loss_weight": float(IMITATION_LOSS_WEIGHT),
        "soft_mask_logit_lambda": float(SOFT_MASK_LOGIT_LAMBDA),
        "soft_mask_small_value": float(SOFT_MASK_SMALL_VALUE),
        # Ensure gamma is consistent with macro-action horizon
        "gamma": (GAMMA_BASE ** primitive_h) if USE_MOTION_PRIMITIVES else GAMMA,
    }

    rl_agent = PPO(configs, discrete=USE_MOTION_PRIMITIVES)
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    parking_agent = ParkingAgent(rl_agent, planner=None)

    # Adaptive primitive expansion components
    adaptive_enabled = bool(USE_MOTION_PRIMITIVES and USE_ADAPTIVE_PRIMITIVE_EXPANSION)
    ap_scheduler = None
    ap_lib_mgr = None
    ap_miner = None
    ap_pruner = None
    ap_shaping = None
    ap_round_id = 0
    ap_last_good_ckpt = None
    ap_last_good_version = None
    ap_base_library_size = _get_current_library_size(env)
    ap_last_expand_baseline = None
    ap_last_round_metrics = None
    post_expand_lr_restore = None

    # mining buffer
    ap_trace_buffer = []
    ap_trace_next_id = 0

    if adaptive_enabled:
        if AdaptivePrimitiveLibraryManager is None:
            print("[adaptive] imports failed; disabling adaptive primitive expansion")
            adaptive_enabled = False
        else:
            ap_scheduler = AdaptivePrimitiveScheduler(cfg)
            ap_lib_mgr = AdaptivePrimitiveLibraryManager(verbose=True)
            ap_lib_mgr.load(base_path=lib_full_path, save_dir=save_path)
            ap_miner = TrajectoryMiner(verbose=True)
            ap_pruner = PrimitivePruner(verbose=True)
            ap_shaping = DiscoveredPrimitiveShaping(cfg) if DiscoveredPrimitiveShaping is not None else None
            ap_last_good_version = ap_lib_mgr.active_version_id
            ap_base_library_size = _get_current_library_size(env, ap_lib_mgr=ap_lib_mgr)
            ap_last_good_ckpt = os.path.join(save_path, "adaptive_primitives", "last_good_agent.pt")
            parking_agent.agent.save(ap_last_good_ckpt, params_only=True)

            print(f"[adaptive] enabled. base_version={ap_last_good_version}, lib_size={ap_lib_mgr.library_size}")

    def run_adaptive_primitive_round(ep_idx: int, trigger_stats: dict = None):
        global env, primitive_lib, primitive_h
        global ap_round_id, ap_last_good_ckpt, ap_last_good_version, ap_last_expand_baseline, ap_last_round_metrics, post_expand_lr_restore

        if not adaptive_enabled:
            return

        # round id + bookkeeping
        ap_round_id = ap_scheduler.on_round_started(ep_idx)
        writer.add_scalar("adaptive/round_id", float(ap_round_id), ep_idx)
        writer.add_scalar("adaptive/triggered", 1.0, ep_idx)

        old_version = ap_lib_mgr.active_version_id
        old_lib_size = _get_current_library_size(env, ap_lib_mgr=ap_lib_mgr)
        _log_adaptive_library_state(
            writer,
            ep_idx,
            env,
            ap_lib_mgr=ap_lib_mgr,
            base_size=ap_base_library_size,
            max_size=int(AP_MAX_LIBRARY_SIZE),
        )
        old_actor_lr, old_critic_lr = parking_agent.agent.get_learning_rates()
        post_expand_lr_restore = {
            "actor_lr": float(old_actor_lr),
            "critic_lr": float(old_critic_lr),
            "restore_start_episode": None,
        }

        # Save rollback checkpoint
        ckpt_before = os.path.join(save_path, "adaptive_primitives", f"before_round_{ap_round_id}.pt")
        os.makedirs(os.path.dirname(ckpt_before), exist_ok=True)
        parking_agent.agent.save(ckpt_before, params_only=True)

        # Validation before
        val_schedule = ["Complex", "Extrem"]
        val_before = _run_eval_episodes(env, parking_agent, int(AP_VALIDATION_EPISODES), val_schedule, deterministic=True)
        writer.add_scalar("adaptive/validation_success_before", float(val_before["success"]), ep_idx)
        writer.add_scalar("adaptive/validation_extreme_success_before", float(val_before["success_extreme"]), ep_idx)

        replay_budget = int(max(0, round(int(AP_MINING_ROLLOUTS) * float(getattr(cfg, 'AP_MINING_REPLAY_RATIO', 0.35)))))
        replay_rollouts = select_replay_episodes(ap_trace_buffer, replay_budget, cfg)
        mining_schedule = build_mining_schedule(trigger_stats or {}, cfg)
        fresh_budget = max(0, int(AP_MINING_ROLLOUTS) - len(replay_rollouts))
        fresh_rollouts = _collect_rollouts_for_mining(
            env,
            parking_agent,
            n_episodes=int(fresh_budget),
            scene_schedule=mining_schedule,
            deterministic=bool(AP_MINING_DETERMINISTIC),
            start_episode_id=int(1000000 + ap_round_id * 10000),
        )
        rollouts = list(replay_rollouts) + list(fresh_rollouts)
        writer.add_scalar('adaptive/mining_replay_count', float(len(replay_rollouts)), ep_idx)
        writer.add_scalar('adaptive/mining_fresh_count', float(len(fresh_rollouts)), ep_idx)

        # Mine
        cands = ap_miner.mine_from_episodes(rollouts, ap_lib_mgr.get_active_library(), cfg)
        writer.add_scalar("adaptive/candidates_raw_count", float(len(cands)), ep_idx)

        # Dedup
        c_dedup = ap_pruner.deduplicate(cands, ap_lib_mgr.get_active_library(), cfg)
        writer.add_scalar("adaptive/candidates_after_dedup", float(len(c_dedup)), ep_idx)

        proxy_contexts = build_proxy_eval_contexts(replay_rollouts if len(replay_rollouts) > 0 else ap_trace_buffer, cfg)
        env_sampler, planner_eval_fn = _build_proxy_pruning_hooks(env, proxy_contexts, cfg)
        writer.add_scalar('adaptive/proxy_context_count', float(len(proxy_contexts)), ep_idx)
        c_proxy = ap_pruner.prune_by_proxy_value(c_dedup, env_sampler=env_sampler, planner_eval_fn=planner_eval_fn, config=cfg)
        writer.add_scalar("adaptive/candidates_after_prune", float(len(c_proxy)), ep_idx)

        # Feasibility checks (best-effort)
        c_feas = ap_pruner.validate_feasibility(c_proxy, env, cfg)
        writer.add_scalar("adaptive/candidates_after_feasibility", float(len(c_feas)), ep_idx)

        # Select top-K to add
        remaining = int(AP_MAX_LIBRARY_SIZE) - int(old_lib_size)
        max_add_this_round = int(AP_MAX_ADD_PER_ROUND)
        if isinstance(trigger_stats, dict):
            hard_delta = float(trigger_stats.get('hard_success_rate_recent_delta', 0.0))
            recent_uplift = trigger_stats.get('post_expand_hard_success_uplift_per_added_primitive_recent', None)
            validation_uplift = trigger_stats.get('last_validation_extreme_success_gain_per_added_primitive', None)
            if hard_delta <= 0.0:
                max_add_this_round = max(1, int(np.ceil(float(max_add_this_round) * 0.5)))
            if recent_uplift is not None and float(recent_uplift) <= 0.0:
                max_add_this_round = max(1, int(np.ceil(float(max_add_this_round) * 0.5)))
            if validation_uplift is not None and float(validation_uplift) <= 0.0:
                max_add_this_round = max(1, int(np.ceil(float(max_add_this_round) * 0.5)))
        writer.add_scalar('adaptive/max_add_this_round', float(max_add_this_round), ep_idx)
        k_add = int(min(max_add_this_round, max(0, remaining)))
        add_list = c_feas[:k_add]

        if k_add <= 0 or len(add_list) == 0:
            if ap_scheduler is not None:
                ap_scheduler.state.post_expand_freeze_remaining = 0
            post_expand_lr_restore = None
            writer.add_scalar("adaptive/added_count", 0.0, ep_idx)
            _log_adaptive_library_state(
                writer,
                ep_idx,
                env,
                ap_lib_mgr=ap_lib_mgr,
                base_size=ap_base_library_size,
                max_size=int(AP_MAX_LIBRARY_SIZE),
            )
            return

        # Add to manager and persist a new version
        added = ap_lib_mgr.add_candidates(add_list, round_id=int(ap_round_id), config=cfg)
        info = ap_lib_mgr.save_version(save_dir=save_path)
        new_lib = ap_lib_mgr.get_active_library()

        writer.add_scalar("adaptive/added_count", float(added), ep_idx)
        _log_adaptive_library_state(
            writer,
            ep_idx,
            env,
            ap_lib_mgr=ap_lib_mgr,
            base_size=ap_base_library_size,
            max_size=int(AP_MAX_LIBRARY_SIZE),
        )

        writer.add_scalar(
            "adaptive/avg_complexity_added",
            float(np.mean([c.complexity_score for c in add_list])) if len(add_list) > 0 else 0.0,
            ep_idx,
        )
        writer.add_scalar(
            "adaptive/avg_utility_added",
            float(np.mean([c.utility_score for c in add_list])) if len(add_list) > 0 else 0.0,
            ep_idx,
        )
        writer.add_scalar(
            "adaptive/avg_novelty_added",
            float(np.mean([c.novelty_score for c in add_list])) if len(add_list) > 0 else 0.0,
            ep_idx,
        )

        # The PPO-visible action space is fixed at family_count; only clear the on-policy buffer.
        parking_agent.agent.memory.clear()

        # Post-expansion stabilization: lower LR + freeze backbone + logit bias for new actions
        try:
            lr_scale = float(AP_POST_EXPAND_LR_SCALE)
            _apply_learning_rate_scale(parking_agent, old_actor_lr, old_critic_lr, lr_scale)
        except Exception:
            pass
        try:
            parking_agent.agent.freeze_actor_backbone(True)
        except Exception:
            pass
        # Update wrapper/library reference (rebuild wrapper is safest)
        primitive_lib = new_lib
        primitive_h = getattr(primitive_lib, 'horizon', primitive_h)
        env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)

        # Update shaping centroids from discovered segments (use end macro obs)
        if ap_shaping is not None and bool(USE_DISCOVERED_PRIMITIVE_SHAPING):
            try:
                by_ep = {int(ep.episode_id): ep for ep in rollouts}
                feats = []
                for c in add_list:
                    eid = int(c.source_metadata.get('episode_id'))
                    ep = by_ep.get(eid, None)
                    if ep is None:
                        continue
                    mt1 = int(c.source_metadata.get('macro_t1', min(len(ep.observations) - 1, 0)))
                    mt1 = max(0, min(mt1, len(ep.observations) - 1))
                    feats.append(ap_shaping.extract_feature_from_obs(ep.observations[mt1]))
                ap_shaping.add_centroids(feats)
            except Exception:
                pass

        # Validation after
        val_after = _run_eval_episodes(env, parking_agent, int(AP_VALIDATION_EPISODES), val_schedule, deterministic=True)
        writer.add_scalar("adaptive/validation_success_after", float(val_after["success"]), ep_idx)
        writer.add_scalar("adaptive/validation_extreme_success_after", float(val_after["success_extreme"]), ep_idx)
        _log_adaptive_round_uplift(
            writer,
            ep_idx,
            before_metrics=val_before,
            after_metrics=val_after,
            old_library_size=old_lib_size,
            new_library_size=int(new_lib.size),
        )
        added_count = max(0, int(new_lib.size) - int(old_lib_size))
        gain_denom = float(max(1, added_count))
        ap_last_round_metrics = {
            'added_count': int(added_count),
            'validation_success_gain': float(val_after['success'] - val_before['success']),
            'validation_extreme_success_gain': float(val_after['success_extreme'] - val_before['success_extreme']),
            'validation_success_gain_per_added_primitive': float(val_after['success'] - val_before['success']) / gain_denom,
            'validation_extreme_success_gain_per_added_primitive': float(val_after['success_extreme'] - val_before['success_extreme']) / gain_denom,
        }

        # Rollback if regressed
        rollback = False
        if bool(AP_ENABLE_ROLLBACK):
            drop = float(val_before["success"] - val_after["success"])
            drop_ext = float(val_before["success_extreme"] - val_after["success_extreme"])
            if drop > float(AP_ROLLBACK_DROP_TOL) or drop_ext > float(AP_ROLLBACK_DROP_TOL):
                rollback = True

        if rollback:
            writer.add_scalar("adaptive/rollback", 1.0, ep_idx)
            try:
                ap_lib_mgr.rollback_to(old_version, save_dir=save_path)
            except Exception:
                pass
            # restore wrapper
            primitive_lib = ap_lib_mgr.get_active_library()
            primitive_h = getattr(primitive_lib, 'horizon', primitive_h)
            env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)

            # restore agent params and action dim
            try:
                parking_agent.agent.load(ckpt_before, params_only=True)
            except Exception:
                pass
            parking_agent.agent.memory.clear()
            try:
                parking_agent.agent.freeze_actor_backbone(False)
                parking_agent.agent.clear_action_logit_bias()
                parking_agent.agent.set_learning_rates(old_actor_lr, old_critic_lr)
            except Exception:
                pass
            _log_adaptive_library_state(
                writer,
                ep_idx,
                env,
                ap_lib_mgr=ap_lib_mgr,
                base_size=ap_base_library_size,
                max_size=int(AP_MAX_LIBRARY_SIZE),
            )
            if ap_scheduler is not None:
                ap_scheduler.state.post_expand_freeze_remaining = 0
            post_expand_lr_restore = None
            ap_last_expand_baseline = None
        else:
            writer.add_scalar("adaptive/rollback", 0.0, ep_idx)
            ap_last_good_version = ap_lib_mgr.active_version_id
            ap_last_expand_baseline = {
                "episode": int(ep_idx),
                "success_rate_recent": float((trigger_stats or {}).get("success_rate_recent", 0.0)),
                "hard_success_rate_recent": float((trigger_stats or {}).get("hard_success_rate_recent", 0.0)),
                "library_size": int(old_lib_size),
            }
            try:
                parking_agent.agent.save(ap_last_good_ckpt, params_only=True)
            except Exception:
                pass

    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    succ_record = []
    best_success_rate = [0.0, 0.0, 0.0]
    progress_heartbeat_steps = max(250, int(parking_agent.agent.configs.batch_size) // 4)

    for i in range(args.train_episode):
        scene_chosen = scene_chooser.choose_case()
        obs, _ = env.reset(options={'level': scene_chosen})
        parking_agent.reset()

        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        episode_update_count = 0
        last_progress_report_step = 0

        # ---- EpisodeTrace buffers (macro-step aligned) ----
        ep_obs_trace = []
        ep_actions_trace = []
        ep_low_actions_trace = []
        ep_rewards_trace = []
        ep_dones_trace = []
        ep_infos_trace = []

        ep_refinement_stats = _new_refinement_episode_stats()
        # action distributions
        n_actions = env.action_space.n if USE_MOTION_PRIMITIVES else None
        ep_action_counts = np.zeros((n_actions,), dtype=np.int64) if n_actions is not None else None

        while not done:
            step_num += 1
            ep_obs_trace.append(np.asarray(obs, dtype=np.float64))
            action_mask = None
            if USE_MOTION_PRIMITIVES and USE_ACTION_MASK and hasattr(env, 'get_action_mask'):
                action_mask = env.get_action_mask(obs)
            action, log_prob = parking_agent.choose_action(obs, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # weak shaping reward from discovered primitives (optional)
            shaping_r = 0.0
            if adaptive_enabled and ap_shaping is not None and bool(USE_DISCOVERED_PRIMITIVE_SHAPING):
                try:
                    shaping_r = float(ap_shaping.reward(obs, next_obs))
                except Exception:
                    shaping_r = 0.0
            reward = float(reward) + float(shaping_r)
            if shaping_r != 0.0:
                writer.add_scalar("adaptive/shaping_reward", float(shaping_r), i)

            _update_refinement_episode_stats(ep_refinement_stats, info.get('refinement_plan_debug', None))

            if ep_action_counts is not None:
                try:
                    family_id = int(info.get('resolved_family_id', action))
                    ep_action_counts[family_id] += 1
                except Exception:
                    pass

            if 'reward_info' in info:
                ri = info.get('reward_info', {})
                if isinstance(ri, dict):
                    row = [_to_scalar(ri.get(k, 0.0), 0.0) for k in REWARD_WEIGHT.keys()]
                    reward_info.append(row)

            total_reward += reward
            reward_per_state_list.append(reward)

            # ---- EpisodeTrace step record ----
            try:
                ep_actions_trace.append(int(info.get('resolved_family_id', action)))
            except Exception:
                ep_actions_trace.append(int(action) if USE_MOTION_PRIMITIVES else -1)
            tr = info.get('macro_exec_trace', {}) if isinstance(info, dict) else {}
            sub_u = tr.get('sub_actions_phys', None)
            if sub_u is None:
                try:
                    sub_u = env.primitive_lib.get_actions(int(info.get('primitive_id', -1)))
                except Exception:
                    sub_u = np.zeros((1, 2), dtype=np.float64)
            ep_low_actions_trace.append(np.asarray(sub_u, dtype=np.float64))
            ep_rewards_trace.append(float(reward))
            ep_dones_trace.append(bool(done))
            ep_infos_trace.append(info if isinstance(info, dict) else {})

            # Store transition in memory
            # obs, action, reward, done, log_prob, next_obs
            if USE_MOTION_PRIMITIVES:
                parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs, action_mask))
            else:
                parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs))

            last_progress_report_step = _maybe_print_episode_heartbeat(
                verbose=verbose,
                episode_idx=i,
                total_episodes=args.train_episode,
                step_num=step_num,
                total_reward=float(total_reward),
                update_count=episode_update_count,
                memory_len=len(parking_agent.agent.memory),
                batch_size=int(parking_agent.agent.configs.batch_size),
                heartbeat_steps=progress_heartbeat_steps,
                last_report_step=last_progress_report_step,
            )

            obs = next_obs

            # Update agent
            if len(parking_agent.agent.memory) % parking_agent.agent.configs.batch_size == 0 and len(parking_agent.agent.memory) >= parking_agent.agent.configs.batch_size:
                if verbose and i % 10 == 0 and step_num == 1: # Print less frequently
                    print("Updating the agent.")
                actor_loss, critic_loss = parking_agent.agent.update()

                # Decay action std
                parking_agent.agent.decay_action_std(
                    parking_agent.agent.configs.action_std_decay_rate,
                    parking_agent.agent.configs.min_action_std,
                    verbose=False,
                )
                episode_update_count += 1

                writer.add_scalar("actor_loss", actor_loss, i)
                writer.add_scalar("critic_loss", critic_loss, i)

            if done:
                if info['status'] == Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)

        writer.add_scalar("total_reward", total_reward, i)
        if len(reward_per_state_list) > 0:
            writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)

        # Log std
        writer.add_scalar("action_std", parking_agent.agent.action_std, i)

        if USE_MOTION_PRIMITIVES:
            _log_refinement_episode_stats(
                writer,
                i,
                step_num,
                bool(USE_PRIMITIVE_REFINEMENT),
                ep_refinement_stats,
            )

        for type_id, scene_name in scene_chooser.scene_types.items():
            rec = scene_chooser.success_record[int(type_id)]
            if len(rec) > 0:
                writer.add_scalar(
                    "success_rate_%s" % scene_name,
                    float(np.mean(rec[-100:])),
                    i,
                )

        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)

        if len(reward_info) > 0:
            reward_info_arr = np.array(reward_info, dtype=np.float64)
            reward_info_sum = np.round(np.sum(reward_info_arr, axis=0), 4)
            reward_info_list.append(list(reward_info_sum))

            # Log reward components dynamically (HOPE-style keys)
            reward_keys = list(REWARD_WEIGHT.keys())

            for idx, name in enumerate(reward_keys):
                if idx >= len(reward_info_sum):
                    break
                writer.add_scalar(f"reward_component/{name}", float(reward_info_sum[idx]), i)

        if verbose and i%10==0 and i>0:
            print('success rate:',np.sum(succ_record[-100:]),'/',len(succ_record[-100:]))
            print('std:', parking_agent.agent.action_std)
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            if len(parking_agent.agent.actor_loss_list) > 0:
                print('loss:', np.mean(parking_agent.agent.actor_loss_list[-100:]),np.mean(parking_agent.agent.critic_loss_list[-100:]))
            # Print reward component summary if available
            if len(reward_info_list) > 0:
                try:
                    keys = list(info.get('reward_info', {}).keys())
                    vals = reward_info_list[-1]
                    msg = ', '.join([f"{k}={vals[j]:.4f}" for j, k in enumerate(keys) if j < len(vals)])
                    print('reward components:', msg)
                except Exception:
                    pass
            print("")

        # save best model (scene-wise, HOPE-style): only save when each scene is not worse.
        success_rates = []
        for sid in sorted(scene_chooser.scene_types.keys()):
            rec = scene_chooser.success_record[int(sid)]
            success_rates.append(float(np.mean(rec[-100:])) if len(rec) > 0 else 0.0)

        if i > 100:
            improved_all = True
            for k in range(len(best_success_rate)):
                if success_rates[k] + 1e-12 < float(best_success_rate[k]):
                    improved_all = False
                    break

            if improved_all:
                best_success_rate = list(success_rates)
                parking_agent.agent.save("%s/PPO_best.pt" % (save_path), params_only=True)
                with open(save_path + 'best.txt', 'w') as f_best_log:
                    f_best_log.write('epoch: %s, success rate: %s' % (i + 1, success_rates))

        if (i+1) % 2000 == 0:
            parking_agent.agent.save("%s/PPO2_%s.pt" % (save_path, i),params_only=True)

        if verbose:
            print(f"Episode {i}/{args.train_episode} | Reward: {total_reward:.2f} | Steps: {step_num} | Success Rate: {np.mean(succ_record[-100:]):.2f} | Updates: {episode_update_count}")
            sys.stdout.flush()

        if verbose and i%10==0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
            plt.figure()
            plt.plot(episodes,reward_list)
            plt.plot(episodes,mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.title(f'Training Reward (Episode {i})')
            plt.savefig('%s/reward.png'%save_path)
            plt.close()

        # ---- Build and store EpisodeTrace for mining ----
        if adaptive_enabled and EpisodeTrace is not None:
            try:
                success = 1 if (len(ep_infos_trace) > 0 and ep_infos_trace[-1].get('status', None) == Status.ARRIVED) else 0
                takeover_used = bool(ep_takeover_used)
                ep_trace = EpisodeTrace(
                    episode_id=int(ap_trace_next_id),
                    scene_type=str(scene_chosen),
                    success=bool(success),
                    total_reward=float(total_reward),
                    step_count_macro=int(len(ep_actions_trace)),
                    takeover_used=False,
                    observations=ep_obs_trace,
                    actions_primitive=ep_actions_trace,
                    actions_low_level=ep_low_actions_trace,
                    rewards=ep_rewards_trace,
                    dones=ep_dones_trace,
                    infos=ep_infos_trace,
                    states_optional=None,
                )
                ap_trace_next_id += 1

                keep = True
                if bool(AP_TRACE_KEEP_SUCCESS_ONLY) and not bool(ep_trace.success):
                    keep = False
                    if bool(AP_TRACE_KEEP_NEAR_SUCCESS):
                        try:
                            # near-success: last obs dist
                            lidar_n = int(LIDAR_NUM)
                            dist_last = float(ep_trace.observations[-1][lidar_n]) * float(MAX_DIST_TO_DEST)
                            if dist_last <= float(AP_NEAR_SUCCESS_DIST_THR):
                                keep = True
                        except Exception:
                            pass
                if keep:
                    ap_trace_buffer.append(ep_trace)
                    # cap buffer
                    if len(ap_trace_buffer) > int(AP_TRACE_BUFFER_MAX_EPISODES):
                        ap_trace_buffer = ap_trace_buffer[-int(AP_TRACE_BUFFER_MAX_EPISODES) :]
            except Exception:
                pass

        # ---- Trigger adaptive round between episodes ----
        if adaptive_enabled:
            _log_adaptive_library_state(
                writer,
                i,
                env,
                ap_lib_mgr=ap_lib_mgr,
                base_size=ap_base_library_size,
                max_size=int(AP_MAX_LIBRARY_SIZE),
            )
            try:
                # recent success rates
                w = int(AP_TRIGGER_WINDOW)
                sr_recent, sr_hard, sr_prev, sr_hard_prev = _windowed_success_rates(scene_chooser, succ_record, w)
                current_size = _get_current_library_size(env, ap_lib_mgr=ap_lib_mgr)
                capacity_remaining = max(0, int(AP_MAX_LIBRARY_SIZE) - int(current_size))

                stats = {
                    "success_rate_recent": sr_recent,
                    "hard_success_rate_recent": sr_hard,
                    "success_rate_recent_delta": float(sr_recent - sr_prev),
                    "hard_success_rate_recent_delta": float(sr_hard - sr_hard_prev),
                    "capacity_remaining": int(capacity_remaining),
                }
                writer.add_scalar("adaptive/success_rate_recent", float(sr_recent), i)
                writer.add_scalar("adaptive/hard_success_rate_recent", float(sr_hard), i)
                writer.add_scalar("adaptive/success_rate_recent_delta", float(sr_recent - sr_prev), i)
                writer.add_scalar("adaptive/hard_success_rate_recent_delta", float(sr_hard - sr_hard_prev), i)

                if ap_last_expand_baseline is not None:
                    base_recent = float(ap_last_expand_baseline.get("success_rate_recent", 0.0))
                    base_hard = float(ap_last_expand_baseline.get("hard_success_rate_recent", 0.0))
                    base_size = int(ap_last_expand_baseline.get("library_size", ap_base_library_size))
                    size_gain = max(0, int(current_size) - int(base_size))
                    writer.add_scalar("adaptive/post_expand_success_uplift_recent", float(sr_recent - base_recent), i)
                    writer.add_scalar("adaptive/post_expand_hard_success_uplift_recent", float(sr_hard - base_hard), i)
                    writer.add_scalar("adaptive/post_expand_episode_delta", float(i - int(ap_last_expand_baseline.get("episode", i))), i)
                    writer.add_scalar(
                        "adaptive/post_expand_success_uplift_per_added_primitive_recent",
                        float(sr_recent - base_recent) / float(max(1, size_gain)),
                        i,
                    )
                    writer.add_scalar(
                        "adaptive/post_expand_hard_success_uplift_per_added_primitive_recent",
                        float(sr_hard - base_hard) / float(max(1, size_gain)),
                        i,
                    )
                    stats["added_since_baseline"] = float(size_gain)
                    stats["post_expand_success_uplift_per_added_primitive_recent"] = float(sr_recent - base_recent) / float(max(1, size_gain))
                    stats["post_expand_hard_success_uplift_per_added_primitive_recent"] = float(sr_hard - base_hard) / float(max(1, size_gain))

                if ap_last_round_metrics is not None:
                    stats["last_validation_extreme_success_gain_per_added_primitive"] = float(
                        ap_last_round_metrics.get("validation_extreme_success_gain_per_added_primitive", 0.0)
                    )

                if ap_scheduler.should_trigger(stats, episode_idx=int(i)):
                    run_adaptive_primitive_round(ep_idx=int(i), trigger_stats=stats)
            except Exception as e:
                writer.add_scalar("adaptive/triggered", 0.0, i)
                if verbose:
                    print(f"[adaptive] trigger check failed: {e}")

        # ---- Post-expansion bias decay / unfreeze ----
        if adaptive_enabled and ap_scheduler is not None:
            lr_scale = 1.0
            restore_progress = 1.0
            try:
                remaining = ap_scheduler.tick_post_expand_freeze()
                writer.add_scalar("adaptive/post_expand_freeze_remaining", float(remaining), i)

                # decay new-action bias
                if parking_agent.agent.action_logit_bias is not None:
                    decay_ep = int(AP_NEW_ACTION_LOGIT_BIAS_DECAY_EPISODES)
                    if decay_ep > 0:
                        factor = max(0.0, 1.0 - 1.0 / float(decay_ep))
                        b = parking_agent.agent.action_logit_bias.detach().cpu().numpy()
                        b = b * float(factor)
                        if float(np.max(b)) < 1e-3:
                            parking_agent.agent.clear_action_logit_bias()
                        else:
                            parking_agent.agent.set_action_logit_bias(b)

                # restore lr/unfreeze when freeze window ends
                if post_expand_lr_restore is not None:
                    base_actor_lr = float(post_expand_lr_restore["actor_lr"])
                    base_critic_lr = float(post_expand_lr_restore["critic_lr"])
                    if remaining > 0:
                        lr_scale = float(AP_POST_EXPAND_LR_SCALE)
                        restore_progress = 0.0
                        _apply_learning_rate_scale(parking_agent, base_actor_lr, base_critic_lr, lr_scale)
                    else:
                        try:
                            parking_agent.agent.freeze_actor_backbone(False)
                        except Exception:
                            pass

                        if post_expand_lr_restore.get("restore_start_episode") is None:
                            post_expand_lr_restore["restore_start_episode"] = int(i)

                        if bool(AP_POST_EXPAND_LR_WARMUP):
                            lr_scale, restore_progress = compute_post_expand_restore_scale(
                                current_episode=int(i),
                                restore_start_episode=int(post_expand_lr_restore["restore_start_episode"]),
                                restore_episodes=int(AP_POST_EXPAND_LR_RESTORE_EPISODES),
                                start_scale=float(AP_POST_EXPAND_LR_SCALE),
                            )
                        else:
                            lr_scale, restore_progress = 1.0, 1.0

                        _apply_learning_rate_scale(parking_agent, base_actor_lr, base_critic_lr, lr_scale)

                        if restore_progress >= 1.0:
                            post_expand_lr_restore = None
            except Exception:
                pass

            _log_learning_rate_state(writer, i, parking_agent, lr_scale, restore_progress)
        else:
            _log_learning_rate_state(writer, i, parking_agent, 1.0, 1.0)
