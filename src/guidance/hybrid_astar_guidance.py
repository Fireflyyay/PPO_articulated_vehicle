from __future__ import annotations

import copy
import heapq
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon

from env.vehicle import State


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _sanitize_reason(reason: str) -> str:
    text = str(reason or "unknown").strip().lower()
    if not text:
        return "unknown"
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in text)


@dataclass
class GuidanceResult:
    guidance_logits: np.ndarray
    guidance_weight: float = 0.0
    guidance_valid: bool = False
    planner_success: bool = False
    teacher_family_id: int = -1
    family_path: List[int] = field(default_factory=list)
    fail_reason: str = "disabled"
    plan_time_ms: float = 0.0
    expand_nodes: int = 0
    path_cost: float = 0.0
    reference_state_available: bool = False
    subgoal_index: int = -1
    average_progress_along_reference: float = 0.0
    teacher_action_mask_value: float = 0.0
    guidance_dropout_active: bool = False
    planner_cache_hit: bool = False
    primitive_mode: str = "normal"
    guidance_confidence: float = 0.0
    jackknife_margin_min: float = 0.0
    rear_body_near_collision_count: int = 0
    reference_state_count: int = 0


@dataclass(order=True)
class _QueueNode:
    priority: float
    counter: int
    state: State = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    depth: int = field(compare=False)
    direction: int = field(compare=False)
    family_path: Tuple[int, ...] = field(compare=False)
    jackknife_margin_min: float = field(compare=False)
    rear_body_near_collision_count: int = field(compare=False)


class HybridAStarGuidance:
    def __init__(self, cfg_module) -> None:
        self.cfg = cfg_module
        self.cache: "OrderedDict[Tuple, GuidanceResult]" = OrderedDict()
        self._episode_failure_streak = 0
        self._episode_guidance_disabled = False
        self._last_runtime_result: Optional[GuidanceResult] = None
        self._rng = np.random.default_rng(int(getattr(cfg_module, "SEED", 42)))

    def clear_cache(self) -> None:
        self.cache.clear()
        self._last_runtime_result = None

    def reset_episode(self, wrapper, scene_level: str) -> None:
        self._episode_failure_streak = 0
        self._episode_guidance_disabled = False
        self._last_runtime_result = None

    def _by_level(self, value, scene_level: str, default):
        if isinstance(value, dict):
            return value.get(str(scene_level), default)
        return default if value is None else value

    def _enabled_for_level(self, scene_level: str) -> bool:
        if not bool(getattr(self.cfg, "ENABLE_HYBRID_ASTAR_GUIDANCE", False)):
            return False
        enabled = getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_CURRICULUM_ENABLE", None)
        return bool(self._by_level(enabled, scene_level, True))

    def _lambda_for_level(self, scene_level: str) -> float:
        return float(self._by_level(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_LAMBDA_BY_CURRICULUM", None), scene_level, 0.0))

    def _dropout_for_level(self, scene_level: str) -> float:
        return float(np.clip(self._by_level(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_DROPOUT_BY_CURRICULUM", None), scene_level, 0.0), 0.0, 1.0))

    def _replan_every_for_level(self, scene_level: str) -> int:
        return max(1, int(self._by_level(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_REPLAN_EVERY_BY_CURRICULUM", None), scene_level, 1)))

    def _subgoal_horizon_for_level(self, scene_level: str) -> int:
        return max(1, int(self._by_level(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_LOCAL_SUBGOAL_HORIZON_BY_CURRICULUM", None), scene_level, 8)))

    def _empty_result(self, wrapper, scene_level: str, fail_reason: str = "disabled") -> GuidanceResult:
        action_dim = 0
        if wrapper is not None:
            action_dim = int(getattr(getattr(wrapper, "action_space", None), "n", 0) or getattr(getattr(wrapper, "primitive_lib", None), "action_dim", 0) or 0)
        return GuidanceResult(
            guidance_logits=np.zeros((max(0, int(action_dim)),), dtype=np.float32),
            guidance_weight=0.0,
            guidance_valid=False,
            planner_success=False,
            teacher_family_id=-1,
            family_path=[],
            fail_reason=str(fail_reason),
            plan_time_ms=0.0,
            expand_nodes=0,
            path_cost=0.0,
            reference_state_available=False,
            subgoal_index=-1,
            average_progress_along_reference=0.0,
            teacher_action_mask_value=0.0,
            guidance_dropout_active=False,
            planner_cache_hit=False,
            primitive_mode=str(getattr(wrapper, "_current_primitive_mode", "normal")) if wrapper is not None else "normal",
            guidance_confidence=0.0,
            jackknife_margin_min=0.0,
            rear_body_near_collision_count=0,
            reference_state_count=0,
        )

    def compute_guidance(self, wrapper, obs, action_mask=None, scene_level: str = "Warmup", episode_step: int = 0) -> GuidanceResult:
        result = self._empty_result(wrapper, scene_level=scene_level, fail_reason="disabled")
        if not self._enabled_for_level(scene_level):
            return result
        if wrapper is None or not hasattr(wrapper, "primitive_lib") or not hasattr(wrapper, "_current_vehicle_state"):
            result.fail_reason = "missing_wrapper_context"
            return result
        if self._episode_guidance_disabled:
            result.fail_reason = "episode_guidance_disabled"
            return result

        replan_every = self._replan_every_for_level(scene_level)
        if (
            self._last_runtime_result is not None
            and int(episode_step) > 0
            and int(episode_step) % int(replan_every) != 0
        ):
            return self._apply_runtime_controls(
                copy.deepcopy(self._last_runtime_result),
                action_mask=action_mask,
                scene_level=scene_level,
                planner_cache_hit=bool(self._last_runtime_result.planner_cache_hit),
            )

        planned = self._plan(wrapper, obs=obs, action_mask=action_mask, scene_level=scene_level)
        if planned.planner_success:
            self._episode_failure_streak = 0
        else:
            self._episode_failure_streak += 1
            if self._episode_failure_streak >= int(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_FAIL_DISABLE_STREAK", 3)):
                self._episode_guidance_disabled = True

        runtime = self._apply_runtime_controls(planned, action_mask=action_mask, scene_level=scene_level, planner_cache_hit=bool(planned.planner_cache_hit))
        self._last_runtime_result = copy.deepcopy(runtime)
        return runtime

    def _apply_runtime_controls(self, planned: GuidanceResult, action_mask=None, scene_level: str = "Warmup", planner_cache_hit: bool = False) -> GuidanceResult:
        result = copy.deepcopy(planned)
        result.planner_cache_hit = bool(planner_cache_hit)
        base_lambda = self._lambda_for_level(scene_level)
        dropout_prob = self._dropout_for_level(scene_level)
        dropout_active = bool(base_lambda > 0.0 and self._rng.random() < dropout_prob)
        result.guidance_dropout_active = dropout_active

        guidance_logits = np.asarray(result.guidance_logits, dtype=np.float32).reshape(-1)
        teacher_mask_value = 0.0
        if action_mask is not None and guidance_logits.size > 0:
            mask_arr = np.asarray(action_mask).reshape(-1)
            if mask_arr.size == guidance_logits.size:
                if np.issubdtype(mask_arr.dtype, np.integer) or mask_arr.dtype == np.bool_:
                    mask_values = mask_arr.astype(np.float32)
                else:
                    mask_values = np.clip(mask_arr.astype(np.float32), 0.0, 1.0)
                if 0 <= int(result.teacher_family_id) < mask_values.size:
                    teacher_mask_value = float(mask_values[int(result.teacher_family_id)])
                zero_thr = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_MASK_ZERO_THRESHOLD", 0.0))
                guidance_logits[mask_values <= zero_thr] = 0.0
        result.teacher_action_mask_value = float(teacher_mask_value)
        result.guidance_logits = guidance_logits.astype(np.float32, copy=False)
        result.guidance_valid = bool(result.planner_success and np.any(np.abs(result.guidance_logits) > 1e-8))
        result.guidance_weight = float(base_lambda if (result.guidance_valid and not dropout_active) else 0.0)
        return result

    def _scene_meta(self, wrapper) -> dict:
        base_env = getattr(wrapper, "env", wrapper)
        world_map = getattr(base_env, "map", None)
        meta = getattr(world_map, "scene_regions", None)
        return dict(meta) if isinstance(meta, dict) else {}

    def _scene_key(self, wrapper, scene_level: str) -> tuple:
        meta = self._scene_meta(wrapper)
        return (
            str(scene_level),
            meta.get("attempt_seed"),
            meta.get("generation_attempt_index"),
            meta.get("seed"),
            int(getattr(getattr(wrapper, "primitive_lib", None), "action_dim", 0)),
        )

    def _cache_get(self, key: tuple) -> Optional[GuidanceResult]:
        hit = self.cache.get(key)
        if hit is None:
            return None
        self.cache.move_to_end(key)
        return copy.deepcopy(hit)

    def _cache_put(self, key: tuple, value: GuidanceResult) -> None:
        self.cache[key] = copy.deepcopy(value)
        self.cache.move_to_end(key)
        limit = max(1, int(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_CACHE_SIZE", 2048)))
        while len(self.cache) > limit:
            self.cache.popitem(last=False)

    def _state_hash(self, state: State, direction: int) -> tuple:
        pos_res = max(1e-3, float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_POSITION_RESOLUTION_M", 0.75)))
        heading_res = max(1e-3, math.radians(float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_HEADING_RESOLUTION_DEG", 12.0))))
        gamma_res = max(1e-3, math.radians(float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_GAMMA_RESOLUTION_DEG", 6.0))))
        gamma = _wrap_pi(float(state.heading) - float(state.rear_heading))
        return (
            int(round(float(state.loc.x) / pos_res)),
            int(round(float(state.loc.y) / pos_res)),
            int(round(_wrap_pi(float(state.heading)) / heading_res)),
            int(round(gamma / gamma_res)),
            int(np.sign(direction)),
        )

    def _gamma_limit(self, wrapper) -> float:
        primitive_lib = getattr(wrapper, "primitive_lib", None)
        if primitive_lib is not None:
            try:
                gamma_bins = np.asarray(getattr(primitive_lib, "gamma_bin_values", np.asarray([])), dtype=np.float64)
                if gamma_bins.size > 0:
                    return float(np.max(np.abs(gamma_bins)))
            except Exception:
                pass
        return float(np.deg2rad(36.0))

    def _reference_states(self, wrapper) -> list:
        meta = self._scene_meta(wrapper)
        if not bool(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_REFERENCE_ENABLE", True)):
            return []
        return list(meta.get("warmup_reference_states") or [])

    def _guidance_points(self, wrapper) -> np.ndarray:
        base_env = getattr(wrapper, "env", wrapper)
        world_map = getattr(base_env, "map", None)
        points = getattr(world_map, "guidance_path_points", None)
        if points is None:
            points = self._scene_meta(wrapper).get("guidance_path_points")
        pts = np.asarray(points or [], dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            return np.zeros((0, 2), dtype=np.float64)
        return pts

    def _nearest_reference_index(self, x: float, y: float, ref_points: np.ndarray) -> int:
        if ref_points.shape[0] == 0:
            return -1
        delta = ref_points - np.asarray([[float(x), float(y)]], dtype=np.float64)
        d2 = np.sum(delta * delta, axis=1)
        return int(np.argmin(d2))

    def _select_subgoal(self, wrapper, state: State, scene_level: str) -> dict:
        ref_states = self._reference_states(wrapper)
        if bool(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_LOCAL_SUBGOAL_ENABLE", True)) and len(ref_states) > 0:
            ref_xy = np.asarray([[float(item.get("x", 0.0)), float(item.get("y", 0.0))] for item in ref_states], dtype=np.float64)
            nearest = self._nearest_reference_index(float(state.loc.x), float(state.loc.y), ref_xy)
            horizon = self._subgoal_horizon_for_level(scene_level)
            subgoal_idx = min(len(ref_states) - 1, max(0, nearest) + horizon)
            item = dict(ref_states[subgoal_idx])
            return {
                "x": float(item.get("x", state.loc.x)),
                "y": float(item.get("y", state.loc.y)),
                "heading": float(item.get("theta_front", state.heading)),
                "rear_heading": float(item.get("theta_rear", item.get("theta_front", state.rear_heading))),
                "is_final": bool(subgoal_idx >= len(ref_states) - 1),
                "subgoal_index": int(subgoal_idx),
                "reference_state_available": True,
                "reference_state_count": int(len(ref_states)),
                "average_progress_along_reference": float(max(0, nearest) / max(1, len(ref_states) - 1)),
            }

        guide_points = self._guidance_points(wrapper)
        if bool(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_LOCAL_SUBGOAL_ENABLE", True)) and guide_points.shape[0] > 1:
            nearest = self._nearest_reference_index(float(state.loc.x), float(state.loc.y), guide_points)
            horizon = self._subgoal_horizon_for_level(scene_level)
            subgoal_idx = min(int(guide_points.shape[0] - 1), max(0, nearest) + horizon)
            if subgoal_idx < int(guide_points.shape[0] - 1):
                dx = float(guide_points[subgoal_idx + 1, 0] - guide_points[subgoal_idx, 0])
                dy = float(guide_points[subgoal_idx + 1, 1] - guide_points[subgoal_idx, 1])
                heading = math.atan2(dy, dx)
            else:
                base_env = getattr(wrapper, "env", wrapper)
                heading = float(getattr(getattr(getattr(base_env, "map", None), "dest", None), "heading", state.heading))
            return {
                "x": float(guide_points[subgoal_idx, 0]),
                "y": float(guide_points[subgoal_idx, 1]),
                "heading": float(heading),
                "rear_heading": float(heading),
                "is_final": bool(subgoal_idx >= int(guide_points.shape[0] - 1)),
                "subgoal_index": int(subgoal_idx),
                "reference_state_available": False,
                "reference_state_count": 0,
                "average_progress_along_reference": float(max(0, nearest) / max(1, int(guide_points.shape[0] - 1))),
            }

        base_env = getattr(wrapper, "env", wrapper)
        world_map = getattr(base_env, "map", None)
        dest = getattr(world_map, "dest", None)
        if dest is None:
            return {
                "x": float(state.loc.x),
                "y": float(state.loc.y),
                "heading": float(state.heading),
                "rear_heading": float(state.rear_heading),
                "is_final": True,
                "subgoal_index": -1,
                "reference_state_available": False,
                "reference_state_count": 0,
                "average_progress_along_reference": 0.0,
            }
        return {
            "x": float(dest.loc.x),
            "y": float(dest.loc.y),
            "heading": float(dest.heading),
            "rear_heading": float(getattr(dest, "rear_heading", dest.heading)),
            "is_final": True,
            "subgoal_index": -1,
            "reference_state_available": False,
            "reference_state_count": 0,
            "average_progress_along_reference": 0.0,
        }

    def _target_state_from_subgoal(self, subgoal: dict) -> State:
        return State([
            float(subgoal.get("x", 0.0)),
            float(subgoal.get("y", 0.0)),
            float(subgoal.get("heading", 0.0)),
            0.0,
            0.0,
            float(subgoal.get("rear_heading", subgoal.get("heading", 0.0))),
        ])

    def _goal_repr_from_state(self, state: State, subgoal: dict) -> dict:
        dx = float(subgoal.get("x", state.loc.x)) - float(state.loc.x)
        dy = float(subgoal.get("y", state.loc.y)) - float(state.loc.y)
        c = math.cos(float(state.heading))
        s = math.sin(float(state.heading))
        goal_x = c * dx + s * dy
        goal_y = -s * dx + c * dy
        goal_heading = _wrap_pi(float(subgoal.get("heading", state.heading)) - float(state.heading))
        articulation = _wrap_pi(float(subgoal.get("heading", state.heading)) - float(subgoal.get("rear_heading", state.rear_heading)))
        return {
            "goal_x": float(goal_x),
            "goal_y": float(goal_y),
            "goal_heading": float(goal_heading),
            "articulation": float(articulation),
            "dist": float(math.hypot(goal_x, goal_y)),
            "rel_angle": float(math.atan2(goal_y, goal_x)),
        }

    def _heuristic(self, state: State, subgoal: dict) -> float:
        dist = float(math.hypot(float(subgoal.get("x", state.loc.x)) - float(state.loc.x), float(subgoal.get("y", state.loc.y)) - float(state.loc.y)))
        heading_err = abs(_wrap_pi(float(subgoal.get("heading", state.heading)) - float(state.heading)))
        gamma_abs = abs(_wrap_pi(float(state.heading) - float(state.rear_heading)))
        return float(dist + 0.35 * heading_err + 0.15 * gamma_abs)

    def _goal_reached(self, state: State, subgoal: dict) -> bool:
        target_state = self._target_state_from_subgoal(subgoal)
        if bool(subgoal.get("is_final", False)):
            metrics = self._terminal_metrics(state, target_state)
            overlap_thr = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_FINAL_FRONT_OVERLAP_THR", 0.70))
            heading_thr = math.radians(float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_FINAL_HEADING_TOL_DEG", 15.0)))
            return bool(metrics["front_overlap"] >= overlap_thr and metrics["heading_error"] <= heading_thr)

        pos_tol = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_LOCAL_GOAL_POS_TOL_M", 1.0))
        heading_tol = math.radians(float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_LOCAL_GOAL_HEADING_TOL_DEG", 25.0)))
        dist = float(state.loc.distance(target_state.loc))
        heading_err = abs(_wrap_pi(float(target_state.heading) - float(state.heading)))
        return bool(dist <= pos_tol and heading_err <= heading_tol)

    def _terminal_metrics(self, state: State, target_state: State) -> dict:
        front_box_ego = Polygon(state.create_box()[0])
        front_box_tgt = Polygon(target_state.create_box()[0])
        rear_box_ego = Polygon(state.create_box()[1])
        rear_box_tgt = Polygon(target_state.create_box()[1])
        front_overlap = float(front_box_ego.intersection(front_box_tgt).area) / (float(front_box_tgt.area) + 1e-9)
        rear_overlap = float(rear_box_ego.intersection(rear_box_tgt).area) / (float(rear_box_tgt.area) + 1e-9)
        heading_error = abs(_wrap_pi(float(target_state.heading) - float(state.heading)))
        return {
            "front_overlap": float(front_overlap),
            "rear_overlap": float(rear_overlap),
            "heading_error": float(heading_error),
        }

    def _interp_angle(self, lhs: float, rhs: float, t: float) -> float:
        delta = _wrap_pi(float(rhs) - float(lhs))
        return _wrap_pi(float(lhs) + float(t) * delta)

    def _interp_state(self, prev_state: State, next_state: State, t: float) -> State:
        x = float(prev_state.loc.x) + (float(next_state.loc.x) - float(prev_state.loc.x)) * float(t)
        y = float(prev_state.loc.y) + (float(next_state.loc.y) - float(prev_state.loc.y)) * float(t)
        heading = self._interp_angle(float(prev_state.heading), float(next_state.heading), float(t))
        rear_heading = self._interp_angle(float(prev_state.rear_heading), float(next_state.rear_heading), float(t))
        speed = float(getattr(prev_state, "speed", 0.0)) + (float(getattr(next_state, "speed", 0.0)) - float(getattr(prev_state, "speed", 0.0))) * float(t)
        steering = float(getattr(prev_state, "steering", 0.0)) + (float(getattr(next_state, "steering", 0.0)) - float(getattr(prev_state, "steering", 0.0))) * float(t)
        return State([x, y, heading, speed, steering, rear_heading])

    def _densify_states(self, world_states: Sequence[State]) -> List[State]:
        if world_states is None or len(world_states) == 0:
            return []
        max_translation = max(1e-3, float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_COLLISION_MAX_TRANSLATION_M", 0.55)))
        max_heading = max(1e-3, math.radians(float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_COLLISION_MAX_HEADING_DEG", 8.0))))
        max_gamma = max(1e-3, math.radians(float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_COLLISION_MAX_GAMMA_DEG", 5.0))))

        densified = [copy.deepcopy(world_states[0])]
        for prev_state, next_state in zip(world_states[:-1], world_states[1:]):
            translation = float(prev_state.loc.distance(next_state.loc))
            heading_delta = abs(_wrap_pi(float(next_state.heading) - float(prev_state.heading)))
            prev_gamma = _wrap_pi(float(prev_state.heading) - float(prev_state.rear_heading))
            next_gamma = _wrap_pi(float(next_state.heading) - float(next_state.rear_heading))
            gamma_delta = abs(_wrap_pi(next_gamma - prev_gamma))
            steps = max(
                1,
                int(math.ceil(translation / max_translation)),
                int(math.ceil(heading_delta / max_heading)),
                int(math.ceil(gamma_delta / max_gamma)),
            )
            for k in range(1, steps + 1):
                densified.append(self._interp_state(prev_state, next_state, float(k) / float(steps)))
        return densified

    def _validate_rollout(self, wrapper, start_state: State, flat_index: int) -> tuple[bool, List[State], str, float, int]:
        primitive_lib = getattr(wrapper, "primitive_lib", None)
        if primitive_lib is None:
            return False, [], "missing_primitive_library", 0.0, 0
        try:
            canonical_rollout = primitive_lib.get_rollout_states(int(flat_index))
        except Exception:
            return False, [], "invalid_rollout", 0.0, 0
        if canonical_rollout is None or len(canonical_rollout) < 2:
            return False, [], "invalid_rollout", 0.0, 0

        try:
            world_states = [wrapper._canonical_state_to_world(start_state, row) for row in canonical_rollout]
        except Exception:
            return False, [], "invalid_rollout_transform", 0.0, 0
        densified = self._densify_states(world_states)
        if len(densified) == 0:
            return False, [], "invalid_rollout", 0.0, 0

        base_env = getattr(wrapper, "env", wrapper)
        world_map = getattr(base_env, "map", None)
        if world_map is None:
            return False, [], "missing_scene_metadata", 0.0, 0

        xmin = float(getattr(world_map, "xmin", -np.inf))
        xmax = float(getattr(world_map, "xmax", np.inf))
        ymin = float(getattr(world_map, "ymin", -np.inf))
        ymax = float(getattr(world_map, "ymax", np.inf))
        obstacles = list(getattr(world_map, "obstacles", []) or [])
        collision_margin = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_COLLISION_MARGIN_M", 0.0))
        rear_near_margin = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_REAR_NEAR_COLLISION_MARGIN_M", 0.35))
        gamma_limit = self._gamma_limit(wrapper)
        gamma_limit_margin = math.radians(float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_ARTICULATION_LIMIT_MARGIN_DEG", 0.0)))
        effective_limit = max(1e-6, gamma_limit - gamma_limit_margin)
        min_jackknife_margin = float(gamma_limit)
        rear_near_collision_count = 0

        for state in densified[1:]:
            x = float(state.loc.x)
            y = float(state.loc.y)
            if x < xmin or x > xmax or y < ymin or y > ymax:
                return False, densified, "boundary_collision", min_jackknife_margin, rear_near_collision_count

            gamma_abs = abs(_wrap_pi(float(state.heading) - float(state.rear_heading)))
            min_jackknife_margin = min(min_jackknife_margin, float(gamma_limit - gamma_abs))
            if gamma_abs > effective_limit:
                return False, densified, "articulation_limit", min_jackknife_margin, rear_near_collision_count

            front_box, rear_box = state.create_box()
            front_poly = Polygon(front_box)
            rear_poly = Polygon(rear_box)
            if collision_margin > 0.0:
                front_check = front_poly.buffer(collision_margin, join_style=2)
                rear_check = rear_poly.buffer(collision_margin, join_style=2)
            else:
                front_check = front_poly
                rear_check = rear_poly

            rear_near = False
            for obst in obstacles:
                geom = getattr(obst, "shape", obst)
                if front_check.intersects(geom):
                    return False, densified, "front_body_collision", min_jackknife_margin, rear_near_collision_count
                if rear_check.intersects(geom):
                    return False, densified, "rear_body_collision", min_jackknife_margin, rear_near_collision_count
                try:
                    if rear_poly.distance(geom) <= rear_near_margin:
                        rear_near = True
                except Exception:
                    pass
            if rear_near:
                rear_near_collision_count += 1

        return True, densified, "", min_jackknife_margin, rear_near_collision_count

    def _ordered_families(self, wrapper, action_mask) -> List[int]:
        action_dim = int(getattr(getattr(wrapper, "action_space", None), "n", getattr(getattr(wrapper, "primitive_lib", None), "action_dim", 0)))
        order = list(range(max(0, action_dim)))
        if action_mask is None:
            return order
        mask_arr = np.asarray(action_mask).reshape(-1)
        if mask_arr.size != action_dim:
            return order
        try:
            score = mask_arr.astype(np.float32)
        except Exception:
            return order
        order.sort(key=lambda idx: float(score[idx]), reverse=True)
        return order

    def _build_guidance_logits(self, wrapper, teacher_family_id: int) -> np.ndarray:
        action_dim = int(getattr(getattr(wrapper, "action_space", None), "n", getattr(getattr(wrapper, "primitive_lib", None), "action_dim", 0)))
        logits = np.zeros((max(0, action_dim),), dtype=np.float32)
        if teacher_family_id < 0 or teacher_family_id >= logits.shape[0]:
            return logits
        logits[int(teacher_family_id)] = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_TEACHER_LOGIT", 2.5))

        primitive_lib = getattr(wrapper, "primitive_lib", None)
        teacher_motion_family = None
        try:
            family_to_motion = np.asarray(getattr(primitive_lib, "family_to_motion_family", np.asarray([])), dtype=np.int64)
            if family_to_motion.size > int(teacher_family_id):
                teacher_motion_family = int(family_to_motion[int(teacher_family_id)])
        except Exception:
            teacher_motion_family = None

        if teacher_motion_family is None:
            return logits

        neighbor_logit = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_NEIGHBOR_LOGIT", 0.75))
        for family_id in range(logits.shape[0]):
            if family_id == int(teacher_family_id):
                continue
            try:
                motion_family = int(np.asarray(primitive_lib.family_to_motion_family, dtype=np.int64)[family_id])
            except Exception:
                continue
            if motion_family == teacher_motion_family:
                logits[family_id] = neighbor_logit
        return logits

    def _edge_cost(self, wrapper, densified_states: Sequence[State], direction: int, prev_direction: int, subgoal: dict, rear_near_collision_count: int, min_jackknife_margin: float) -> float:
        travel = 0.0
        abs_gamma = []
        for prev_state, next_state in zip(densified_states[:-1], densified_states[1:]):
            travel += float(prev_state.loc.distance(next_state.loc))
            abs_gamma.append(abs(_wrap_pi(float(next_state.heading) - float(next_state.rear_heading))))
        mean_gamma = float(np.mean(abs_gamma)) if len(abs_gamma) > 0 else 0.0
        reverse_penalty = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_REVERSE_PENALTY", 0.25)) if int(direction) < 0 else 0.0
        switch_penalty = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_DIRECTION_SWITCH_PENALTY", 0.20)) if int(prev_direction) != 0 and int(direction) != int(prev_direction) else 0.0
        heading_err = abs(_wrap_pi(float(subgoal.get("heading", densified_states[-1].heading)) - float(densified_states[-1].heading)))
        gamma_limit = self._gamma_limit(wrapper)
        jackknife_penalty = 0.20 * max(0.0, 1.0 - float(min_jackknife_margin) / max(gamma_limit, 1e-6))
        return float(travel + 0.35 * heading_err + reverse_penalty + switch_penalty + 0.25 * mean_gamma + jackknife_penalty + 0.02 * float(rear_near_collision_count))

    def _resolve_flat_index(self, wrapper, family_id: int, node_state: State, subgoal: dict, primitive_mode: str, selection_context) -> Optional[int]:
        try:
            if hasattr(wrapper.primitive_lib, "resolve_family_variant"):
                ref = wrapper.primitive_lib.resolve_family_variant(
                    int(family_id),
                    gamma=float(_wrap_pi(float(node_state.heading) - float(node_state.rear_heading))),
                    primitive_mode=primitive_mode,
                    goal_repr=self._goal_repr_from_state(node_state, subgoal),
                    selection_context=selection_context,
                )
                return int(ref.flat_index)
            return int(family_id)
        except Exception:
            return None

    def _greedy_fallback(self, wrapper, state0: State, action_mask, subgoal: dict, primitive_mode: str, heuristic_weight: float) -> Optional[_QueueNode]:
        if not bool(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_GREEDY_FALLBACK_ENABLE", True)):
            return None

        selection_context = None
        if hasattr(wrapper, "_variant_selection_context"):
            try:
                selection_context = wrapper._variant_selection_context({"selected_mode": primitive_mode})
            except Exception:
                selection_context = None

        best_candidate: Optional[_QueueNode] = None
        prev_direction = int(np.sign(float(getattr(state0, "speed", 0.0))))
        for family_id in self._ordered_families(wrapper, action_mask=action_mask):
            flat_index = self._resolve_flat_index(wrapper, family_id, state0, subgoal, primitive_mode, selection_context)
            if flat_index is None:
                continue

            valid, densified_states, _reject_reason, min_jackknife_margin, rear_near_collision_count = self._validate_rollout(wrapper, state0, flat_index)
            if not valid or len(densified_states) == 0:
                continue

            actions = np.asarray(wrapper.primitive_lib.get_actions(int(flat_index)), dtype=np.float64)
            direction = int(np.sign(float(np.mean(actions[:, 1])))) if actions.size > 0 else prev_direction
            edge_cost = self._edge_cost(
                wrapper,
                densified_states,
                direction=direction,
                prev_direction=prev_direction,
                subgoal=subgoal,
                rear_near_collision_count=rear_near_collision_count,
                min_jackknife_margin=min_jackknife_margin,
            )
            state1 = copy.deepcopy(densified_states[-1])
            h_cost = float(self._heuristic(state1, subgoal))
            candidate = _QueueNode(
                priority=float(h_cost + 0.1 * edge_cost),
                counter=int(family_id),
                state=state1,
                g_cost=float(edge_cost),
                h_cost=h_cost,
                depth=1,
                direction=int(direction),
                family_path=(int(family_id),),
                jackknife_margin_min=float(min_jackknife_margin),
                rear_body_near_collision_count=int(rear_near_collision_count),
            )
            if best_candidate is None or float(candidate.priority) < float(best_candidate.priority):
                best_candidate = candidate

        return best_candidate

    def _plan(self, wrapper, obs, action_mask=None, scene_level: str = "Warmup") -> GuidanceResult:
        t0 = time.perf_counter()
        result = self._empty_result(wrapper, scene_level=scene_level, fail_reason="planner_unavailable")

        state0 = wrapper._current_vehicle_state()
        if state0 is None:
            result.fail_reason = "missing_state"
            return result

        primitive_mode = str(getattr(wrapper, "_current_primitive_mode", "normal"))
        subgoal = self._select_subgoal(wrapper, state0, scene_level=scene_level)
        result.reference_state_available = bool(subgoal.get("reference_state_available", False))
        result.reference_state_count = int(subgoal.get("reference_state_count", 0))
        result.subgoal_index = int(subgoal.get("subgoal_index", -1))
        result.average_progress_along_reference = float(subgoal.get("average_progress_along_reference", 0.0))
        result.primitive_mode = primitive_mode

        cache_key = (
            self._scene_key(wrapper, scene_level),
            self._state_hash(state0, int(np.sign(float(getattr(state0, "speed", 0.0))))),
            int(result.subgoal_index),
            str(primitive_mode),
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            cached.planner_cache_hit = True
            return cached

        open_heap: List[_QueueNode] = []
        counter = 0
        h0 = self._heuristic(state0, subgoal)
        heapq.heappush(
            open_heap,
            _QueueNode(
                priority=float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_HEURISTIC_WEIGHT", 1.35)) * h0,
                counter=counter,
                state=copy.deepcopy(state0),
                g_cost=0.0,
                h_cost=h0,
                depth=0,
                direction=int(np.sign(float(getattr(state0, "speed", 0.0)))),
                family_path=tuple(),
                jackknife_margin_min=float(self._gamma_limit(wrapper)),
                rear_body_near_collision_count=0,
            ),
        )
        best_g: Dict[Tuple, float] = {}
        expand_nodes = 0
        max_expand_nodes = max(1, int(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_MAX_EXPAND_NODES", 96)))
        max_depth = max(1, int(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_MAX_SEARCH_DEPTH", 5)))
        max_time_ms = max(1.0, float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_MAX_PLAN_TIME_MS", 20.0)))
        heuristic_weight = float(getattr(self.cfg, "HYBRID_ASTAR_GUIDANCE_HEURISTIC_WEIGHT", 1.35))
        ordered_families = self._ordered_families(wrapper, action_mask=action_mask)
        best_node: Optional[_QueueNode] = None
        fail_reason = "open_set_exhausted"

        while len(open_heap) > 0:
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            if elapsed_ms > max_time_ms:
                fail_reason = "timeout"
                break
            node = heapq.heappop(open_heap)
            state_key = self._state_hash(node.state, node.direction)
            if float(best_g.get(state_key, np.inf)) <= float(node.g_cost):
                continue
            best_g[state_key] = float(node.g_cost)

            if self._goal_reached(node.state, subgoal):
                best_node = node
                fail_reason = ""
                break
            if int(node.depth) >= max_depth:
                continue
            if expand_nodes >= max_expand_nodes:
                fail_reason = "max_expand_nodes"
                break

            selection_context = None
            if hasattr(wrapper, "_variant_selection_context"):
                try:
                    selection_context = wrapper._variant_selection_context({"selected_mode": primitive_mode})
                except Exception:
                    selection_context = None

            expand_nodes += 1
            for family_id in ordered_families:
                flat_index = self._resolve_flat_index(wrapper, family_id, node.state, subgoal, primitive_mode, selection_context)
                if flat_index is None:
                    continue

                valid, densified_states, reject_reason, min_jackknife_margin, rear_near_collision_count = self._validate_rollout(wrapper, node.state, flat_index)
                if not valid or len(densified_states) == 0:
                    continue
                actions = np.asarray(wrapper.primitive_lib.get_actions(int(flat_index)), dtype=np.float64)
                direction = int(np.sign(float(np.mean(actions[:, 1])))) if actions.size > 0 else int(node.direction)
                edge_cost = self._edge_cost(
                    wrapper,
                    densified_states,
                    direction=direction,
                    prev_direction=int(node.direction),
                    subgoal=subgoal,
                    rear_near_collision_count=rear_near_collision_count,
                    min_jackknife_margin=min_jackknife_margin,
                )
                state1 = copy.deepcopy(densified_states[-1])
                g_cost = float(node.g_cost + edge_cost)
                h_cost = float(self._heuristic(state1, subgoal))
                successor_key = self._state_hash(state1, direction)
                if float(best_g.get(successor_key, np.inf)) <= g_cost:
                    continue
                counter += 1
                heapq.heappush(
                    open_heap,
                    _QueueNode(
                        priority=float(g_cost + heuristic_weight * h_cost),
                        counter=counter,
                        state=state1,
                        g_cost=g_cost,
                        h_cost=h_cost,
                        depth=int(node.depth + 1),
                        direction=int(direction),
                        family_path=tuple(list(node.family_path) + [int(family_id)]),
                        jackknife_margin_min=float(min(node.jackknife_margin_min, min_jackknife_margin)),
                        rear_body_near_collision_count=int(node.rear_body_near_collision_count + rear_near_collision_count),
                    ),
                )

        result.plan_time_ms = float(1000.0 * (time.perf_counter() - t0))
        result.expand_nodes = int(expand_nodes)
        result.fail_reason = str(fail_reason)

        if best_node is None or len(best_node.family_path) == 0:
            if best_node is not None and len(best_node.family_path) == 0 and result.fail_reason == "":
                result.fail_reason = "already_at_subgoal"
            elif result.fail_reason != "already_at_subgoal":
                best_node = self._greedy_fallback(
                    wrapper,
                    state0=state0,
                    action_mask=action_mask,
                    subgoal=subgoal,
                    primitive_mode=primitive_mode,
                    heuristic_weight=heuristic_weight,
                )
            if best_node is None or len(best_node.family_path) == 0:
                self._cache_put(cache_key, result)
                return result

        teacher_family = int(best_node.family_path[0])
        result.planner_success = True
        result.teacher_family_id = teacher_family
        result.family_path = list(best_node.family_path)
        result.path_cost = float(best_node.g_cost)
        result.guidance_logits = self._build_guidance_logits(wrapper, teacher_family)
        result.guidance_valid = bool(np.any(np.abs(result.guidance_logits) > 1e-8))
        result.fail_reason = ""
        result.guidance_confidence = float(1.0 / (1.0 + best_node.g_cost))
        result.jackknife_margin_min = float(best_node.jackknife_margin_min)
        result.rear_body_near_collision_count = int(best_node.rear_body_near_collision_count)
        self._cache_put(cache_key, result)
        return copy.deepcopy(result)