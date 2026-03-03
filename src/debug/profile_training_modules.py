import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Callable

import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import configs as cfg
from configs import *

from env.car_parking_base import CarParking
from env.lidar_simulator import LidarSimlator
from env.wrappers.macro_action_wrapper import MacroActionWrapper
from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent, PrimitivePlanner
from primitives.library import load_library


@dataclass
class Stat:
    total_s: float = 0.0
    calls: int = 0


class TimerRegistry:
    def __init__(self):
        self.stats: Dict[str, Stat] = {}

    def add(self, name: str, dt: float):
        st = self.stats.get(name)
        if st is None:
            st = Stat()
            self.stats[name] = st
        st.total_s += float(dt)
        st.calls += 1

    def timed(self, name: str, fn: Callable):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(name, time.perf_counter() - t0)

        return wrapped


def patch_methods(timer: TimerRegistry):
    from model.agent import parking_agent as parking_agent_mod
    from model.agent import ppo_agent as ppo_agent_mod
    from env import car_parking_base as car_parking_mod
    from env.wrappers import macro_action_wrapper as wrapper_mod
    from env import lidar_simulator as lidar_mod
    from env import global_guidance as guidance_mod

    parking_agent_mod.ParkingAgent.choose_action = timer.timed(
        "agent.choose_action_total", parking_agent_mod.ParkingAgent.choose_action
    )
    ppo_agent_mod.PPOAgent._actor_forward = timer.timed(
        "agent.actor_forward", ppo_agent_mod.PPOAgent._actor_forward
    )
    ppo_agent_mod.PPOAgent.push_memory = timer.timed(
        "agent.push_memory", ppo_agent_mod.PPOAgent.push_memory
    )
    ppo_agent_mod.PPOAgent.update = timer.timed(
        "agent.update", ppo_agent_mod.PPOAgent.update
    )

    wrapper_mod.MacroActionWrapper.get_action_mask = timer.timed(
        "wrapper.get_action_mask", wrapper_mod.MacroActionWrapper.get_action_mask
    )
    wrapper_mod.MacroActionWrapper.step = timer.timed(
        "wrapper.step", wrapper_mod.MacroActionWrapper.step
    )
    wrapper_mod.MacroActionWrapper.reset = timer.timed(
        "wrapper.reset", wrapper_mod.MacroActionWrapper.reset
    )
    wrapper_mod.MacroActionWrapper._maybe_plan_terminal_takeover = timer.timed(
        "wrapper.takeover_planner", wrapper_mod.MacroActionWrapper._maybe_plan_terminal_takeover
    )

    car_parking_mod.CarParking.step = timer.timed(
        "env.base_step", car_parking_mod.CarParking.step
    )
    car_parking_mod.CarParking.reset = timer.timed(
        "env.base_reset", car_parking_mod.CarParking.reset
    )
    car_parking_mod.CarParking.get_reward = timer.timed(
        "env.get_reward", car_parking_mod.CarParking.get_reward
    )

    lidar_mod.LidarSimlator.get_observation = timer.timed(
        "env.lidar", lidar_mod.LidarSimlator.get_observation
    )

    guidance_mod.SoftGlobalGuidance.plan_path = timer.timed(
        "guidance.plan_path", guidance_mod.SoftGlobalGuidance.plan_path
    )
    guidance_mod.SoftGlobalGuidance.get_soft_hint = timer.timed(
        "guidance.get_soft_hint", guidance_mod.SoftGlobalGuidance.get_soft_hint
    )


def build_agent_and_env(verbose: bool = False):
    base_env = CarParking(fps=100, verbose=verbose, render_mode="rgb_array")
    env = base_env

    if not USE_MOTION_PRIMITIVES:
        raise RuntimeError("This profiler currently targets motion-primitive PPO training path.")

    lib_full_path = os.path.normpath(os.path.join(src_dir, PRIMITIVE_LIBRARY_PATH))
    if not os.path.exists(lib_full_path):
        if os.path.exists(PRIMITIVE_LIBRARY_PATH):
            lib_full_path = PRIMITIVE_LIBRARY_PATH
        else:
            root_dir = os.path.dirname(src_dir)
            lib_full_path = os.path.join(root_dir, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))

    primitive_lib = load_library(lib_full_path)
    primitive_h = getattr(primitive_lib, "horizon", PRIMITIVE_H)
    env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)

    actor_params = dict(ACTOR_CONFIGS)
    critic_params = dict(CRITIC_CONFIGS)
    actor_params["output_size"] = env.action_space.n
    actor_params["use_tanh_output"] = False

    configs = {
        "discrete": True,
        "observation_shape": base_env.observation_shape,
        "action_dim": env.action_space.n,
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
        "action_std_init": 1.5,
        "action_std_decay_rate": 0.0003,
        "min_action_std": 0.1,
        "gamma": GAMMA_BASE ** primitive_h,
    }

    rl_agent = PPO(configs, discrete=True)
    primitive_planner = PrimitivePlanner()
    parking_agent = ParkingAgent(rl_agent, planner=primitive_planner)
    return env, parking_agent


def run_profile(max_episodes: int, max_macro_steps: int, seed: int):
    timer = TimerRegistry()
    patch_methods(timer)

    np.random.seed(seed)
    torch.manual_seed(seed)

    env, parking_agent = build_agent_and_env(verbose=False)

    scene_cycle = ["Normal", "Complex", "Extrem"]
    succ_record = []

    total_macro_steps = 0
    update_calls = 0
    takeover_plan_ms = []
    takeover_prune_ms = []
    takeover_score_ms = []

    t_wall0 = time.perf_counter()

    for ep in range(int(max_episodes)):
        scene = scene_cycle[ep % len(scene_cycle)]
        obs, _ = env.reset(options={"level": scene})
        parking_agent.reset()
        done = False

        while not done:
            action_mask = None
            if USE_ACTION_MASK and hasattr(env, "get_action_mask"):
                action_mask = env.get_action_mask(obs)

            action, log_prob = parking_agent.choose_action(obs, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            if USE_MOTION_PRIMITIVES:
                parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs, action_mask))
            else:
                parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs))

            m = parking_agent.agent.memory
            bsz = int(parking_agent.agent.configs.batch_size)
            if len(m) % bsz == 0 and len(m) >= bsz:
                parking_agent.agent.update()
                update_calls += 1

            dbg = info.get("takeover_debug", None)
            if isinstance(dbg, dict):
                if "plan_ms" in dbg:
                    takeover_plan_ms.append(float(dbg["plan_ms"]))
                if "fast_prune_ms" in dbg:
                    takeover_prune_ms.append(float(dbg["fast_prune_ms"]))
                if "score_ms" in dbg:
                    takeover_score_ms.append(float(dbg["score_ms"]))

            obs = next_obs
            total_macro_steps += 1
            if total_macro_steps >= int(max_macro_steps):
                break

        succ = 1 if info.get("status", None).name == "ARRIVED" else 0
        succ_record.append(succ)

        if total_macro_steps >= int(max_macro_steps):
            break

    wall_s = time.perf_counter() - t_wall0

    rows = []
    for name, st in timer.stats.items():
        pct_wall = 100.0 * st.total_s / max(wall_s, 1e-9)
        avg_ms = 1000.0 * st.total_s / max(st.calls, 1)
        rows.append((name, st.calls, st.total_s, pct_wall, avg_ms))

    rows.sort(key=lambda x: x[2], reverse=True)

    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else 0.0

    summary = {
        "episodes": len(succ_record),
        "macro_steps": int(total_macro_steps),
        "updates": int(update_calls),
        "success_rate": safe_mean(succ_record),
        "wall_s": float(wall_s),
        "rows": rows,
        "takeover_plan_ms_mean": safe_mean(takeover_plan_ms),
        "takeover_prune_ms_mean": safe_mean(takeover_prune_ms),
        "takeover_score_ms_mean": safe_mean(takeover_score_ms),
    }
    return summary


def print_summary(summary):
    print("=" * 98)
    print("Training module profiling summary")
    print("=" * 98)
    print(
        f"episodes={summary['episodes']} | macro_steps={summary['macro_steps']} | updates={summary['updates']} "
        f"| success_rate={summary['success_rate']:.3f} | wall={summary['wall_s']:.2f}s"
    )
    print("-" * 98)
    print(f"{'module':38s} {'calls':>8s} {'total_s':>12s} {'pct_wall':>10s} {'avg_ms':>10s}")
    print("-" * 98)
    for name, calls, total_s, pct_wall, avg_ms in summary["rows"]:
        print(f"{name:38s} {calls:8d} {total_s:12.4f} {pct_wall:10.2f}% {avg_ms:10.3f}")
    print("-" * 98)
    print(
        "takeover_debug(ms): "
        f"plan_mean={summary['takeover_plan_ms_mean']:.3f}, "
        f"fast_prune_mean={summary['takeover_prune_ms_mean']:.3f}, "
        f"score_mean={summary['takeover_score_ms_mean']:.3f}"
    )
    print("=" * 98)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--macro_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    summary = run_profile(max_episodes=args.episodes, max_macro_steps=args.macro_steps, seed=args.seed)
    print_summary(summary)
