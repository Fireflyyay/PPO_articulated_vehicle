import argparse
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import configs as cfg
from configs import *

from env.car_parking_base import CarParking
from env.wrappers.macro_action_wrapper import MacroActionWrapper
from model.agent.parking_agent import ParkingAgent, PrimitivePlanner
from model.agent.ppo_agent import PPOAgent as PPO
from primitives.library import load_library


@dataclass
class Stat:
    total_s: float = 0.0
    calls: int = 0


class TimerRegistry:
    def __init__(self):
        self.stats: Dict[str, Stat] = {}

    def add(self, name: str, dt: float, count: int = 1):
        if not name:
            return
        st = self.stats.get(name)
        if st is None:
            st = Stat()
            self.stats[name] = st
        st.total_s += float(max(0.0, dt))
        st.calls += int(count)

    def total(self, name: str) -> float:
        st = self.stats.get(name)
        return float(st.total_s) if st is not None else 0.0

    def calls(self, name: str) -> int:
        st = self.stats.get(name)
        return int(st.calls) if st is not None else 0

    def timed(self, name: str, fn: Callable):
        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(name, time.perf_counter() - t0)

        return wrapped

    @contextmanager
    def scope(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)


class GenerateCaseSectionTracer:
    def __init__(self, timer: TimerRegistry, target_code):
        self.timer = timer
        self.target_code = target_code
        self.target_filename = str(target_code.co_filename)
        self.stack = []

    @staticmethod
    def _target_section_for_line(lineno: int) -> Optional[str]:
        if 570 <= lineno < 1800:
            return "reset.scene_generate.geometry_build"
        if 1800 <= lineno < 1889:
            return "reset.scene_generate.pair_sampling"
        if 1889 <= lineno < 2133:
            return "reset.scene_generate.divider_walls"
        if 2133 <= lineno <= 2142:
            return "reset.scene_generate.divider_walls"
        return None

    @staticmethod
    def _special_group_for_name(name: str) -> Optional[str]:
        if name in {"_scene_metrics_for_pair", "_scene_metrics_pass", "_grid_path_stats", "_grid_reachable"}:
            return "reset.scene_generate.metrics_check"
        if name == "_plan_guidance_path_for_scene":
            return "reset.scene_generate.guidance_validate"
        return None

    def _flush(self, now: float):
        if not self.stack:
            return
        current = self.stack[-1]
        if current["group"] is not None and current["last_ts"] is not None:
            self.timer.add(current["group"], now - current["last_ts"], count=0)

    def _mark_target_section(self, section: Optional[str], seen_sections):
        if not section:
            return
        if section not in seen_sections:
            self.timer.add(section, 0.0, count=1)
            seen_sections.add(section)

    def trace(self, frame, event, arg):
        now = time.perf_counter()
        if event == "call":
            if frame.f_code is self.target_code:
                section = self._target_section_for_line(frame.f_lineno)
                seen_sections = set()
                self._mark_target_section(section, seen_sections)
                self.stack.append(
                    {
                        "frame": frame,
                        "group": section,
                        "last_ts": now,
                        "target": True,
                        "seen_sections": seen_sections,
                    }
                )
                return self.trace

            if self.stack and str(frame.f_code.co_filename) == self.target_filename:
                self._flush(now)
                parent = self.stack[-1]
                special_group = self._special_group_for_name(frame.f_code.co_name)
                group = special_group or parent["group"]
                if special_group:
                    self.timer.add(special_group, 0.0, count=1)
                self.stack.append(
                    {
                        "frame": frame,
                        "group": group,
                        "last_ts": now,
                        "target": False,
                        "seen_sections": parent["seen_sections"],
                    }
                )
                return self.trace

            return None

        if not self.stack or frame is not self.stack[-1]["frame"]:
            return self.trace if self.stack else None

        if event == "line":
            self._flush(now)
            current = self.stack[-1]
            if current["target"]:
                current["group"] = self._target_section_for_line(frame.f_lineno)
                self._mark_target_section(current["group"], current["seen_sections"])
            current["last_ts"] = now
            return self.trace

        if event == "return":
            self._flush(now)
            self.stack.pop()
            if self.stack:
                self.stack[-1]["last_ts"] = now
            return self.trace

        return self.trace


class ProfileRuntime:
    def __init__(self):
        self.stage_timer = TimerRegistry()
        self.method_timer = TimerRegistry()
        self.event_counts = defaultdict(int)
        self.retry_reason_counts = defaultdict(int)
        self.guidance_phase_stack = []
        self._installed = False

    @contextmanager
    def guidance_phase(self, prefix: str):
        self.guidance_phase_stack.append(str(prefix))
        try:
            yield
        finally:
            self.guidance_phase_stack.pop()

    def guidance_child_name(self, suffix: str) -> str:
        prefix = self.guidance_phase_stack[-1] if self.guidance_phase_stack else "guidance.plan_path"
        return f"{prefix}.{suffix}"

    def note_retry_reasons(self, scene_meta):
        if not isinstance(scene_meta, dict):
            return
        self.event_counts["scene_generate.generated_snapshots"] += 1
        self.event_counts["scene_generate.attempts_used_sum"] += int(scene_meta.get("generation_attempts_used", 1))
        self.event_counts["scene_generate.retry_count_sum"] += int(scene_meta.get("generation_retry_count", 0))
        for key, value in dict(scene_meta.get("generation_retry_reasons", {}) or {}).items():
            self.retry_reason_counts[str(key)] += int(value)

    def install(self):
        if self._installed:
            return

        from env import car_parking_base as car_parking_mod
        from env import global_guidance as guidance_mod
        from env import lidar_simulator as lidar_mod
        from env import parking_map_normal as parking_map_mod
        from env.wrappers import macro_action_wrapper as wrapper_mod
        from model.agent import parking_agent as parking_agent_mod
        from model.agent import ppo_agent as ppo_agent_mod

        self.car_parking_mod = car_parking_mod
        self.guidance_mod = guidance_mod
        self.lidar_mod = lidar_mod
        self.parking_map_mod = parking_map_mod
        self.wrapper_mod = wrapper_mod
        self.parking_agent_mod = parking_agent_mod
        self.ppo_agent_mod = ppo_agent_mod

        parking_agent_mod.ParkingAgent.choose_action = self.method_timer.timed(
            "agent.choose_action_total", parking_agent_mod.ParkingAgent.choose_action
        )
        ppo_agent_mod.PPOAgent._actor_forward = self.method_timer.timed(
            "agent.actor_forward", ppo_agent_mod.PPOAgent._actor_forward
        )
        ppo_agent_mod.PPOAgent.push_memory = self.method_timer.timed(
            "agent.push_memory", ppo_agent_mod.PPOAgent.push_memory
        )
        ppo_agent_mod.PPOAgent.update = self.method_timer.timed(
            "agent.update", ppo_agent_mod.PPOAgent.update
        )

        wrapper_mod.MacroActionWrapper.get_action_mask = self.method_timer.timed(
            "wrapper.get_action_mask", wrapper_mod.MacroActionWrapper.get_action_mask
        )
        wrapper_mod.MacroActionWrapper.step = self.method_timer.timed(
            "wrapper.step", wrapper_mod.MacroActionWrapper.step
        )
        wrapper_mod.MacroActionWrapper._maybe_plan_terminal_takeover = self.method_timer.timed(
            "wrapper.takeover_planner", wrapper_mod.MacroActionWrapper._maybe_plan_terminal_takeover
        )

        car_parking_mod.CarParking.step = self.method_timer.timed(
            "env.base_step", car_parking_mod.CarParking.step
        )
        car_parking_mod.CarParking.get_reward = self.method_timer.timed(
            "env.get_reward", car_parking_mod.CarParking.get_reward
        )

        lidar_mod.LidarSimlator.get_observation = self.method_timer.timed(
            "env.lidar", lidar_mod.LidarSimlator.get_observation
        )
        guidance_mod.SoftGlobalGuidance.get_soft_hint = self.method_timer.timed(
            "guidance.get_soft_hint", guidance_mod.SoftGlobalGuidance.get_soft_hint
        )

        wrapper_mod.MacroActionWrapper.reset = self._wrap_wrapper_reset(wrapper_mod.MacroActionWrapper.reset)
        car_parking_mod.CarParking.reset = self._wrap_car_reset(car_parking_mod.CarParking.reset)
        parking_map_mod.ParkingMapNormal.reset = self._wrap_map_reset(parking_map_mod.ParkingMapNormal.reset)
        parking_map_mod.ParkingMapNormal._acquire_scene_snapshot = self._wrap_acquire_scene_snapshot(
            parking_map_mod.ParkingMapNormal._acquire_scene_snapshot
        )
        parking_map_mod.ParkingMapNormal._generate_scene_snapshot = self._wrap_generate_scene_snapshot(
            parking_map_mod.ParkingMapNormal._generate_scene_snapshot
        )
        parking_map_mod.generate_navigation_case = self._wrap_generate_navigation_case(
            parking_map_mod.generate_navigation_case
        )
        generate_case_once_name = "_generate_block_mixing_navigation_case_once"
        generate_case_once = getattr(parking_map_mod, generate_case_once_name, None)
        if generate_case_once is None:
            generate_case_once_name = "_generate_navigation_case_once"
            generate_case_once = parking_map_mod._generate_navigation_case_once
        setattr(
            parking_map_mod,
            generate_case_once_name,
            self._wrap_generate_case_once(generate_case_once),
        )
        parking_map_mod._plan_guidance_path_for_scene = self._wrap_scene_guidance_validate(
            parking_map_mod._plan_guidance_path_for_scene
        )
        guidance_mod.SoftGlobalGuidance.plan_path = self._wrap_guidance_plan_path(
            guidance_mod.SoftGlobalGuidance.plan_path
        )

        self._installed = True

    def _wrap_wrapper_reset(self, original):
        stage_timer = self.stage_timer

        def wrapped(wrapper_self, **kwargs):
            out = wrapper_self.env.reset(**kwargs)
            with stage_timer.scope("reset.wrapper_reset"):
                wrapper_self._takeover_active = False
                wrapper_self._takeover_prev_choice = None
                wrapper_self._takeover_mode = "auto"
                wrapper_self._takeover_fail_count = 0
                wrapper_self._prefix_steps_queue.clear()
                wrapper_self._mask_obstacles_prepared = None
                wrapper_self._mask_obstacles_bounds = None
                wrapper_self._action_mask_cached = None
                wrapper_self._action_mask_calls_since_update = 0
            return out

        return wrapped

    def _wrap_car_reset(self, original):
        stage_timer = self.stage_timer
        car_parking_mod = self.car_parking_mod

        def wrapped(env_self, seed: int = None, options: dict = None):
            with stage_timer.scope("reset.total"):
                super(car_parking_mod.CarParking, env_self).reset(seed=seed)

                options = options or {}
                case_id = options.get("case_id")
                data_dir = options.get("data_dir")
                level = options.get("level")

                env_self.reward = 0.0
                env_self.accum_arrive_reward = 0.0
                env_self.t = 0.0
                env_self.accum_turn_count = 0
                env_self.accum_turn_degree = 0.0
                env_self.accum_dist = 0.0

                if level is not None:
                    env_self.set_level(level)

                guidance_ready = False
                last_reset_error = None
                max_scene_retries = 1
                if ENABLE_GLOBAL_SOFT_GUIDANCE and env_self.global_guidance is not None and bool(NAVIGATION_REQUIRE_GUIDANCE_SUCCESS):
                    max_scene_retries = int(max(1, NAVIGATION_RESET_MAX_SCENE_RETRIES))

                for _ in range(max_scene_retries):
                    try:
                        with stage_timer.scope("reset.map_reset"):
                            initial_state = env_self.map.reset(case_id, data_dir)

                        with stage_timer.scope("reset.vehicle_reset"):
                            env_self.vehicle.reset(initial_state)
                            env_self.matrix = env_self.coord_transform_matrix()

                        if ENABLE_GLOBAL_SOFT_GUIDANCE and env_self.global_guidance is not None:
                            guidance_ready = False
                            precomputed_path = getattr(env_self.map, "guidance_path_points", None)
                            if precomputed_path is not None:
                                with stage_timer.scope("reset.guidance.precomputed_path"):
                                    guidance_ready = bool(env_self.global_guidance.set_precomputed_path(precomputed_path))
                                self.event_counts["guidance.precomputed_attempts"] += 1
                                if guidance_ready:
                                    self.event_counts["guidance.precomputed_hits"] += 1
                                else:
                                    self.event_counts["guidance.precomputed_misses"] += 1

                            if not guidance_ready:
                                dest_center = np.mean(env_self.map.dest_box.coords[:-1], axis=0)
                                with self.guidance_phase("reset.guidance.plan_path"):
                                    with stage_timer.scope("reset.guidance.plan_path"):
                                        guidance_ready = bool(
                                            env_self.global_guidance.plan_path(
                                                env_self.map,
                                                start_xy=(
                                                    float(env_self.vehicle.state.loc.x),
                                                    float(env_self.vehicle.state.loc.y),
                                                ),
                                                goal_xy=(float(dest_center[0]), float(dest_center[1])),
                                            )
                                        )
                                self.event_counts["guidance.plan_path_calls"] += 1

                            if bool(NAVIGATION_REQUIRE_GUIDANCE_SUCCESS) and (not guidance_ready):
                                continue
                        else:
                            guidance_ready = True

                        break
                    except Exception as exc:
                        last_reset_error = exc
                        guidance_ready = False
                        continue

                if ENABLE_GLOBAL_SOFT_GUIDANCE and env_self.global_guidance is not None and bool(NAVIGATION_REQUIRE_GUIDANCE_SUCCESS):
                    if not guidance_ready:
                        if last_reset_error is not None:
                            raise RuntimeError("Failed to reset environment with a valid guidance path") from last_reset_error
                        raise RuntimeError("Failed to reset environment with a valid guidance path")

                env_self.initial_dist = float(env_self.vehicle.state.loc.distance(car_parking_mod.Point(env_self.map.dest.loc))) + 1e-6

                with stage_timer.scope("reset.observation_build"):
                    obs = env_self._build_observation()

                info = {
                    "status": car_parking_mod.Status.CONTINUE,
                    "reward_info": car_parking_mod.OrderedDict({k: 0.0 for k in car_parking_mod.REWARD_WEIGHT.keys()}),
                    "path_to_dest": None,
                }
                return obs, info

        return wrapped

    def _wrap_map_reset(self, original):
        stage_timer = self.stage_timer
        parking_map_mod = self.parking_map_mod

        def wrapped(map_self, case_id: int = None, path: str = None):
            with stage_timer.scope("reset.scene_snapshot.acquire"):
                snapshot = map_self._acquire_scene_snapshot()

            start = snapshot["start"]
            dest = snapshot["dest"]
            obstacles = snapshot["obstacles"]
            scene_meta = snapshot["scene_meta"]

            with stage_timer.scope("reset.scene_snapshot.materialize_state"):
                map_self.case_id = 2
                map_self.scene_regions = scene_meta
                map_self.scene_metrics = scene_meta.get("divider_scene_metrics")
                map_self.guidance_path_points = scene_meta.get("guidance_path_points")

                start_articulation = parking_map_mod.random_uniform_num(np.radians(-10), np.radians(10))
                start_rear_heading = start[2] - start_articulation
                map_self.start = parking_map_mod.State(start + [0, 0, start_rear_heading])
                map_self.start_box = map_self.start.create_box()[0]

                dest_articulation = parking_map_mod.random_uniform_num(np.radians(-10), np.radians(10))
                dest_rear_heading = dest[2] - dest_articulation
                map_self.dest = parking_map_mod.State(dest + [0, 0, dest_rear_heading])
                map_self.dest_box = map_self.dest.create_box()[0]

                map_self.xmin = -40.0
                map_self.xmax = 40.0
                map_self.ymin = -40.0
                map_self.ymax = 40.0

                map_self.obstacles = [
                    parking_map_mod.Area(shape=obs, subtype="obstacle", color=(150, 150, 150, 255))
                    for obs in obstacles
                ]
                map_self.n_obstacle = len(map_self.obstacles)

            return map_self.start

        return wrapped

    def _wrap_acquire_scene_snapshot(self, original):
        def wrapped(map_self):
            if not map_self._scene_pool_enabled():
                map_self._scene_pool_stats["direct_generations"] += 1
                map_self._scene_pool_stats["generated_scenes"] += 1
                self.event_counts["scene_pool.direct_generations"] += 1
                return map_self._generate_scene_snapshot()

            if len(map_self._scene_pool) == 0:
                map_self._scene_pool_stats["pool_misses"] += 1
                self.event_counts["scene_pool.pool_misses"] += 1
                with self.stage_timer.scope("reset.scene_snapshot.pool_fill"):
                    map_self._fill_scene_pool(map_self.scene_pool_size, reason="miss")
            else:
                map_self._scene_pool_stats["pool_hits"] += 1
                self.event_counts["scene_pool.pool_hits"] += 1

            snapshot = map_self._scene_pool.popleft()
            map_self._scene_pool_stats["consumed_scenes"] += 1
            self.event_counts["scene_pool.consumed_scenes"] += 1

            if map_self._should_top_up_scene_pool():
                with self.stage_timer.scope("reset.scene_snapshot.top_up"):
                    before_generated = int(map_self._scene_pool_stats.get("generated_scenes", 0))
                    map_self._fill_scene_pool(reason="top_up")
                    generated_delta = int(map_self._scene_pool_stats.get("generated_scenes", 0)) - before_generated
                    if generated_delta > 0:
                        self.event_counts["scene_pool.top_up_generated"] += int(generated_delta)

            return snapshot

        return wrapped

    def _wrap_generate_scene_snapshot(self, original):
        def wrapped(map_self):
            with self.stage_timer.scope("reset.scene_snapshot.direct_generate"):
                snapshot = original(map_self)
            return snapshot

        return wrapped

    def _wrap_generate_navigation_case(self, original):
        def wrapped(*args, **kwargs):
            with self.stage_timer.scope("reset.scene_generate.retry_total"):
                result = original(*args, **kwargs)
            if kwargs.get("return_regions") and isinstance(result, tuple) and len(result) == 4:
                self.note_retry_reasons(result[3])
            return result

        return wrapped

    def _wrap_generate_case_once(self, original):
        tracer = GenerateCaseSectionTracer(self.stage_timer, original.__code__)

        def wrapped(*args, **kwargs):
            previous_trace = sys.gettrace()
            sys.settrace(tracer.trace)
            try:
                return original(*args, **kwargs)
            finally:
                sys.settrace(previous_trace)

        return wrapped

    def _wrap_scene_guidance_validate(self, original):
        def wrapped(*args, **kwargs):
            with self.guidance_phase("reset.scene_generate.guidance_validate"):
                with self.stage_timer.scope("reset.scene_generate.guidance_validate"):
                    return original(*args, **kwargs)

        return wrapped

    def _wrap_guidance_plan_path(self, original):
        method_timer = self.method_timer
        stage_timer = self.stage_timer

        def wrapped(planner_self, world_map, start_xy, goal_xy):
            phase_total_name = self.guidance_child_name("total")
            t0_total = time.perf_counter()
            try:
                with stage_timer.scope(self.guidance_child_name("occupancy_build")):
                    occ, bounds = planner_self._build_occupancy(world_map)
                cache_status = "unknown"
                if hasattr(planner_self, "get_last_occupancy_cache_status"):
                    cache_status = str(planner_self.get_last_occupancy_cache_status())
                self.event_counts[f"guidance.occupancy_cache.{cache_status}"] += 1
                if hasattr(planner_self, "get_last_occupancy_cache_details"):
                    cache_details = planner_self.get_last_occupancy_cache_details() or {}
                    for layer_name, layer_status in cache_details.items():
                        self.event_counts[f"guidance.occupancy_cache_detail.{layer_name}.{layer_status}"] += 1
                    builder_mode = str(cache_details.get("combined_builder", ""))
                    if builder_mode == "raster":
                        self.event_counts["guidance.occupancy_build.raster_hit"] += 1
                    elif builder_mode == "intersects":
                        self.event_counts["guidance.occupancy_build.intersects_fallback"] += 1

                start = planner_self._world_to_cell(float(start_xy[0]), float(start_xy[1]), bounds, occ.shape)
                goal = planner_self._world_to_cell(float(goal_xy[0]), float(goal_xy[1]), bounds, occ.shape)
                if start is None or goal is None:
                    planner_self.clear_path()
                    return False

                occ[start[0], start[1]] = 0
                occ[goal[0], goal[1]] = 0

                with stage_timer.scope(self.guidance_child_name("astar")):
                    cell_path = planner_self._astar(occ, start, goal)
                if cell_path is None or len(cell_path) == 0:
                    planner_self.clear_path()
                    return False

                with stage_timer.scope(self.guidance_child_name("path_materialize")):
                    pts = np.array([planner_self._cell_to_world(i, j, bounds) for i, j in cell_path], dtype=np.float64)
                    planner_self.path_points_world = pts
                    planner_self.path_s = planner_self._polyline_arc_length(pts)
                    planner_self.progress_idx = 0
                return True
            finally:
                elapsed = time.perf_counter() - t0_total
                method_timer.add("guidance.plan_path", elapsed)
                stage_timer.add(phase_total_name, elapsed)

        return wrapped


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
    runtime = ProfileRuntime()
    runtime.install()
    stage_timer = runtime.stage_timer
    method_timer = runtime.method_timer

    np.random.seed(seed)
    torch.manual_seed(seed)

    with stage_timer.scope("setup.build_env_agent"):
        env, parking_agent = build_agent_and_env(verbose=False)

    scene_cycle = ["Normal", "Complex", "Extrem"]
    succ_record = []
    total_macro_steps = 0
    update_calls = 0
    takeover_plan_ms = []
    takeover_prune_ms = []
    takeover_score_ms = []

    t_wall0 = time.perf_counter()
    loop_wall_s = 0.0

    for ep in range(int(max_episodes)):
        scene = scene_cycle[ep % len(scene_cycle)]
        with stage_timer.scope("episode.reset_env"):
            obs, _ = env.reset(options={"level": scene})
        with stage_timer.scope("episode.reset_agent"):
            parking_agent.reset()
        done = False

        while not done:
            step_t0 = time.perf_counter()
            accounted_s = 0.0

            action_mask = None
            t0 = time.perf_counter()
            if USE_ACTION_MASK and hasattr(env, "get_action_mask"):
                with stage_timer.scope("policy.action_mask"):
                    action_mask = env.get_action_mask(obs)
            accounted_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            with stage_timer.scope("policy.choose_action"):
                action, log_prob = parking_agent.choose_action(obs, action_mask=action_mask)
            accounted_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            with stage_timer.scope("env.step"):
                next_obs, reward, terminated, truncated, info = env.step(action)
            accounted_s += time.perf_counter() - t0
            done = bool(terminated or truncated)

            t0 = time.perf_counter()
            if USE_MOTION_PRIMITIVES:
                with stage_timer.scope("buffer.push_memory"):
                    parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs, action_mask))
            else:
                with stage_timer.scope("buffer.push_memory"):
                    parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            accounted_s += time.perf_counter() - t0

            memory = parking_agent.agent.memory
            batch_size = int(parking_agent.agent.configs.batch_size)
            t0 = time.perf_counter()
            if len(memory) % batch_size == 0 and len(memory) >= batch_size:
                with stage_timer.scope("learn.update"):
                    parking_agent.agent.update()
                    update_calls += 1
            accounted_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            takeover_debug = info.get("takeover_debug")
            if isinstance(takeover_debug, dict):
                if "plan_ms" in takeover_debug:
                    takeover_plan_ms.append(float(takeover_debug["plan_ms"]))
                if "fast_prune_ms" in takeover_debug:
                    takeover_prune_ms.append(float(takeover_debug["fast_prune_ms"]))
                if "score_ms" in takeover_debug:
                    takeover_score_ms.append(float(takeover_debug["score_ms"]))
            accounted_s += time.perf_counter() - t0

            obs = next_obs
            total_macro_steps += 1
            step_elapsed = time.perf_counter() - step_t0
            loop_wall_s += step_elapsed
            stage_timer.add("macro_step.total", step_elapsed)
            stage_timer.add("macro_step.python_overhead", max(0.0, step_elapsed - accounted_s))
            if total_macro_steps >= int(max_macro_steps):
                break

        status = info.get("status") if isinstance(info, dict) else None
        succ_record.append(1 if getattr(status, "name", None) == "ARRIVED" else 0)
        if total_macro_steps >= int(max_macro_steps):
            break

    wall_s = time.perf_counter() - t_wall0

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    training_stage_names = [
        "setup.build_env_agent",
        "episode.reset_env",
        "episode.reset_agent",
        "policy.action_mask",
        "policy.choose_action",
        "env.step",
        "buffer.push_memory",
        "learn.update",
        "macro_step.python_overhead",
    ]
    training_rows = []
    for name in training_stage_names:
        total_s = stage_timer.total(name)
        calls = stage_timer.calls(name)
        if calls == 0 and total_s <= 0.0:
            continue
        training_rows.append((name, calls, total_s, 100.0 * total_s / max(wall_s, 1e-9), 1000.0 * total_s / max(calls, 1)))
    training_rows.sort(key=lambda item: item[2], reverse=True)

    method_rows = []
    for name, stat in method_timer.stats.items():
        method_rows.append((name, stat.calls, stat.total_s, 100.0 * stat.total_s / max(wall_s, 1e-9), 1000.0 * stat.total_s / max(stat.calls, 1)))
    method_rows.sort(key=lambda item: item[2], reverse=True)

    summary = {
        "episodes": len(succ_record),
        "macro_steps": int(total_macro_steps),
        "updates": int(update_calls),
        "success_rate": safe_mean(succ_record),
        "wall_s": float(wall_s),
        "loop_wall_s": float(loop_wall_s),
        "training_rows": training_rows,
        "method_rows": method_rows,
        "stage_stats": stage_timer.stats,
        "event_counts": dict(runtime.event_counts),
        "retry_reason_counts": dict(runtime.retry_reason_counts),
        "takeover_plan_ms_mean": safe_mean(takeover_plan_ms),
        "takeover_prune_ms_mean": safe_mean(takeover_prune_ms),
        "takeover_score_ms_mean": safe_mean(takeover_score_ms),
    }
    return summary


def _build_rows(stage_stats, names, parent_total: Optional[float] = None, residual_name: Optional[str] = None):
    rows = []
    running_total = 0.0
    for name in names:
        stat = stage_stats.get(name)
        if stat is None:
            continue
        running_total += stat.total_s
        pct_parent = 100.0 * stat.total_s / max(parent_total, 1e-9) if parent_total and parent_total > 0.0 else 0.0
        rows.append((name, stat.calls, stat.total_s, pct_parent, 1000.0 * stat.total_s / max(stat.calls, 1)))
    if residual_name and parent_total is not None:
        residual = max(0.0, float(parent_total) - running_total)
        if residual > 1e-9:
            rows.append((residual_name, 1, residual, 100.0 * residual / max(parent_total, 1e-9), 1000.0 * residual))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows


def _print_table(title: str, rows, name_header: str, overlap_note: Optional[str] = None):
    if not rows:
        return
    print("-" * 110)
    print(title)
    if overlap_note:
        print(overlap_note)
    print("-" * 110)
    print(f"{name_header:46s} {'calls':>8s} {'total_s':>12s} {'pct_parent':>12s} {'avg_ms':>12s}")
    print("-" * 110)
    for name, calls, total_s, pct_parent, avg_ms in rows:
        print(f"{name:46s} {calls:8d} {total_s:12.4f} {pct_parent:11.2f}% {avg_ms:12.3f}")


def print_summary(summary):
    stage_stats = summary["stage_stats"]
    reset_env_total = float(stage_stats.get("episode.reset_env", Stat()).total_s)
    reset_core_total = float(stage_stats.get("reset.total", Stat()).total_s)
    map_reset_total = float(stage_stats.get("reset.map_reset", Stat()).total_s)
    snapshot_acquire_total = float(stage_stats.get("reset.scene_snapshot.acquire", Stat()).total_s)
    direct_generate_total = float(stage_stats.get("reset.scene_snapshot.direct_generate", Stat()).total_s)
    guidance_plan_total = float(stage_stats.get("reset.guidance.plan_path", Stat()).total_s)

    print("=" * 110)
    print("Training module profiling summary")
    print("=" * 110)
    print(
        f"episodes={summary['episodes']} | macro_steps={summary['macro_steps']} | updates={summary['updates']} "
        f"| success_rate={summary['success_rate']:.3f} | wall={summary['wall_s']:.2f}s | loop_wall={summary['loop_wall_s']:.2f}s"
    )
    print(
        f"reset_env_total={reset_env_total:.2f}s | reset_core_total={reset_core_total:.2f}s "
        f"| scene_generation_total={direct_generate_total:.2f}s"
    )

    _print_table("Training-loop stage breakdown", summary["training_rows"], "stage")

    reset_primary_rows = _build_rows(
        stage_stats,
        [
            "reset.map_reset",
            "reset.vehicle_reset",
            "reset.guidance.precomputed_path",
            "reset.guidance.plan_path",
            "reset.observation_build",
            "reset.wrapper_reset",
        ],
        parent_total=reset_env_total,
        residual_name="reset.other_overhead",
    )
    _print_table(
        "Reset primary breakdown (exclusive children under episode.reset_env)",
        reset_primary_rows,
        "reset_stage",
    )

    map_rows = _build_rows(
        stage_stats,
        [
            "reset.scene_snapshot.acquire",
            "reset.scene_snapshot.materialize_state",
        ],
        parent_total=map_reset_total,
        residual_name="reset.map_reset.other_overhead",
    )
    _print_table("reset.map_reset breakdown", map_rows, "map_stage")

    snapshot_rows = _build_rows(
        stage_stats,
        [
            "reset.scene_snapshot.pool_fill",
            "reset.scene_snapshot.direct_generate",
        ],
        parent_total=snapshot_acquire_total,
        residual_name="reset.scene_snapshot.acquire.other_overhead",
    )
    _print_table(
        "reset.scene_snapshot.acquire breakdown",
        snapshot_rows,
        "snapshot_stage",
        overlap_note="note: pool_fill contains direct_generate calls when the pool refills",
    )

    scene_generate_rows = _build_rows(
        stage_stats,
        [
            "reset.scene_generate.retry_total",
            "reset.scene_generate.geometry_build",
            "reset.scene_generate.pair_sampling",
            "reset.scene_generate.divider_walls",
            "reset.scene_generate.metrics_check",
            "reset.scene_generate.guidance_validate",
        ],
        parent_total=direct_generate_total,
    )
    _print_table(
        "Scene generation breakdown",
        scene_generate_rows,
        "scene_stage",
        overlap_note="note: retry_total is parent total; other rows are internal traced subphases under direct_generate",
    )

    guidance_rows = _build_rows(
        stage_stats,
        [
            "reset.guidance.plan_path.occupancy_build",
            "reset.guidance.plan_path.astar",
            "reset.guidance.plan_path.path_materialize",
        ],
        parent_total=guidance_plan_total,
        residual_name="reset.guidance.plan_path.other_overhead",
    )
    _print_table("reset.guidance.plan_path breakdown", guidance_rows, "guidance_stage")

    method_rows = summary["method_rows"]
    if method_rows:
        print("-" * 110)
        print("Internal hotspots (inclusive method timing)")
        print("-" * 110)
        print(f"{'module':46s} {'calls':>8s} {'total_s':>12s} {'pct_wall':>12s} {'avg_ms':>12s}")
        print("-" * 110)
        for name, calls, total_s, pct_wall, avg_ms in method_rows:
            print(f"{name:46s} {calls:8d} {total_s:12.4f} {pct_wall:11.2f}% {avg_ms:12.3f}")

    event_counts = summary.get("event_counts", {})
    retry_reason_counts = summary.get("retry_reason_counts", {})
    print("-" * 110)
    print("Reset counters")
    print("-" * 110)
    print(
        "scene_pool: "
        f"hits={event_counts.get('scene_pool.pool_hits', 0)}, "
        f"misses={event_counts.get('scene_pool.pool_misses', 0)}, "
        f"direct_generations={event_counts.get('scene_pool.direct_generations', 0)}, "
        f"consumed={event_counts.get('scene_pool.consumed_scenes', 0)}"
    )
    print(
        "guidance: "
        f"precomputed_attempts={event_counts.get('guidance.precomputed_attempts', 0)}, "
        f"precomputed_hits={event_counts.get('guidance.precomputed_hits', 0)}, "
        f"precomputed_misses={event_counts.get('guidance.precomputed_misses', 0)}, "
        f"runtime_plan_calls={event_counts.get('guidance.plan_path_calls', 0)}, "
        f"occupancy_payload_hits={event_counts.get('guidance.occupancy_cache.payload_hit', 0)}, "
        f"occupancy_instance_hits={event_counts.get('guidance.occupancy_cache.instance_hit', 0)}, "
        f"occupancy_layered_hits={event_counts.get('guidance.occupancy_cache.layered_hit', 0)}, "
        f"occupancy_misses={event_counts.get('guidance.occupancy_cache.miss', 0)}"
    )
    print(
        "guidance_layers: "
        f"static_hits={event_counts.get('guidance.occupancy_cache_detail.static.hit', 0)}, "
        f"static_misses={event_counts.get('guidance.occupancy_cache_detail.static.miss', 0)}, "
        f"dynamic_hits={event_counts.get('guidance.occupancy_cache_detail.dynamic.hit', 0)}, "
        f"dynamic_misses={event_counts.get('guidance.occupancy_cache_detail.dynamic.miss', 0)}, "
        f"dynamic_empty={event_counts.get('guidance.occupancy_cache_detail.dynamic.empty', 0)}"
    )
    print(
        "guidance_builders: "
        f"raster_builds={event_counts.get('guidance.occupancy_build.raster_hit', 0)}, "
        f"intersects_fallbacks={event_counts.get('guidance.occupancy_build.intersects_fallback', 0)}, "
        f"static_raster={event_counts.get('guidance.occupancy_cache_detail.static_builder.raster', 0)}, "
        f"static_intersects={event_counts.get('guidance.occupancy_cache_detail.static_builder.intersects', 0)}, "
        f"dynamic_raster={event_counts.get('guidance.occupancy_cache_detail.dynamic_builder.raster', 0)}, "
        f"dynamic_intersects={event_counts.get('guidance.occupancy_cache_detail.dynamic_builder.intersects', 0)}"
    )
    print(
        "scene_generation: "
        f"snapshots={event_counts.get('scene_generate.generated_snapshots', 0)}, "
        f"attempts_used_sum={event_counts.get('scene_generate.attempts_used_sum', 0)}, "
        f"retry_count_sum={event_counts.get('scene_generate.retry_count_sum', 0)}"
    )
    if retry_reason_counts:
        print("retry_reasons:")
        for key, value in sorted(retry_reason_counts.items(), key=lambda item: item[1], reverse=True):
            print(f"  {key}: {value}")

    print("-" * 110)
    print(
        "takeover_debug(ms): "
        f"plan_mean={summary['takeover_plan_ms_mean']:.3f}, "
        f"fast_prune_mean={summary['takeover_prune_ms_mean']:.3f}, "
        f"score_mean={summary['takeover_score_ms_mean']:.3f}"
    )
    print("=" * 110)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--macro_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    summary = run_profile(max_episodes=args.episodes, max_macro_steps=args.macro_steps, seed=args.seed)
    print_summary(summary)
