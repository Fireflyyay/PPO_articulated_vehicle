from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class RefinementResult:
    actions: np.ndarray
    original_horizon: int
    effective_horizon: int
    applied: bool
    feasible: bool
    cost_before: float
    cost_after: float
    iterations: int
    cost_evaluations: int
    terminal_scale: float
    terminal_polish_triggered: bool = False
    terminal_polish_applied: bool = False
    terminal_polish_tail_steps: int = 0
    terminal_polish_passes: int = 0
    cost_breakdown_before: Dict[str, float] = field(default_factory=dict)
    cost_breakdown_after: Dict[str, float] = field(default_factory=dict)
    reason: str = "ok"
    elapsed_ms: float = 0.0

    def to_debug_dict(self) -> Dict[str, float]:
        shrink_steps = max(0, int(self.original_horizon) - int(self.effective_horizon))
        shrink_ratio = float(shrink_steps) / float(max(1, int(self.original_horizon)))
        debug = {
            "attempted": True,
            "applied": bool(self.applied),
            "feasible": bool(self.feasible),
            "original_horizon": int(self.original_horizon),
            "cost_before": float(self.cost_before),
            "cost_after": float(self.cost_after),
            "effective_horizon": int(self.effective_horizon),
            "shrink_steps": int(shrink_steps),
            "shrink_ratio": float(shrink_ratio),
            "iterations": int(self.iterations),
            "cost_evaluations": int(self.cost_evaluations),
            "terminal_scale": float(self.terminal_scale),
            "terminal_polish_triggered": bool(self.terminal_polish_triggered),
            "terminal_polish_applied": bool(self.terminal_polish_applied),
            "terminal_polish_tail_steps": int(self.terminal_polish_tail_steps),
            "terminal_polish_passes": int(self.terminal_polish_passes),
            "elapsed_ms": float(self.elapsed_ms),
            "reason": str(self.reason),
        }
        for key, value in self.cost_breakdown_before.items():
            debug[f"{key}_before"] = float(value)
        for key, value in self.cost_breakdown_after.items():
            debug[f"{key}_after"] = float(value)
        return debug


@dataclass
class PlanRefinementResult:
    primitive_ids: List[int]
    phase_actions: List[np.ndarray]
    plan_length: int
    total_steps: int
    applied: bool
    feasible: bool
    cost_before: float
    cost_after: float
    iterations: int
    cost_evaluations: int
    terminal_scale: float
    terminal_polish_triggered: bool = False
    terminal_polish_applied: bool = False
    terminal_polish_tail_steps: int = 0
    terminal_polish_passes: int = 0
    cost_breakdown_before: Dict[str, float] = field(default_factory=dict)
    cost_breakdown_after: Dict[str, float] = field(default_factory=dict)
    reason: str = "ok"
    elapsed_ms: float = 0.0

    def to_debug_dict(self) -> Dict[str, float]:
        debug = {
            "attempted": True,
            "applied": bool(self.applied),
            "feasible": bool(self.feasible),
            "plan_length": int(self.plan_length),
            "total_steps": int(self.total_steps),
            "cost_before": float(self.cost_before),
            "cost_after": float(self.cost_after),
            "cost_delta": float(max(0.0, self.cost_before - self.cost_after)),
            "iterations": int(self.iterations),
            "cost_evaluations": int(self.cost_evaluations),
            "terminal_scale": float(self.terminal_scale),
            "terminal_polish_triggered": bool(self.terminal_polish_triggered),
            "terminal_polish_applied": bool(self.terminal_polish_applied),
            "terminal_polish_tail_steps": int(self.terminal_polish_tail_steps),
            "terminal_polish_passes": int(self.terminal_polish_passes),
            "elapsed_ms": float(self.elapsed_ms),
            "reason": str(self.reason),
        }
        for key, value in self.cost_breakdown_before.items():
            debug[f"{key}_before"] = float(value)
        for key, value in self.cost_breakdown_after.items():
            debug[f"{key}_after"] = float(value)
        return debug


class PrimitiveTrajectoryRefiner:
    """Planning-level continuous improvement layer for motion primitive plans.

    The refiner preserves phase order and drive modes and performs one local
    shooting-style refinement on a whole primitive plan before execution.
    A single-primitive refine() API is kept as a compatibility wrapper.
    """

    def __init__(self, config=None):
        if config is None:
            import configs as config

        self.cfg = config

    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "USE_PRIMITIVE_REFINEMENT", False))

    def refine(self, env, actions: np.ndarray) -> RefinementResult:
        return self._refine_actions(env, actions, allow_prefix_shrink=True, terminal_polish_start_idx=None)

    def refine_plan(self, env, primitive_lib, primitive_ids: Sequence[int]) -> PlanRefinementResult:
        t0 = time.perf_counter()
        primitive_ids = [int(pid) for pid in list(primitive_ids or [])]
        if len(primitive_ids) == 0:
            return PlanRefinementResult(
                primitive_ids=[],
                phase_actions=[],
                plan_length=0,
                total_steps=0,
                applied=False,
                feasible=True,
                cost_before=0.0,
                cost_after=0.0,
                iterations=0,
                cost_evaluations=0,
                terminal_scale=1.0,
                terminal_polish_triggered=False,
                terminal_polish_applied=False,
                terminal_polish_tail_steps=0,
                terminal_polish_passes=0,
                reason="empty_plan",
                elapsed_ms=1000.0 * (time.perf_counter() - t0),
            )

        max_primitives = int(max(1, getattr(self.cfg, "PRIMITIVE_REFINEMENT_MAX_PLAN_PRIMITIVES", 6)))
        if len(primitive_ids) > max_primitives:
            phase_actions = [np.asarray(primitive_lib.get_actions(pid), dtype=np.float64).copy() for pid in primitive_ids]
            total_steps = int(sum(int(a.shape[0]) for a in phase_actions))
            return PlanRefinementResult(
                primitive_ids=primitive_ids,
                phase_actions=phase_actions,
                plan_length=int(len(primitive_ids)),
                total_steps=total_steps,
                applied=False,
                feasible=True,
                cost_before=0.0,
                cost_after=0.0,
                iterations=0,
                cost_evaluations=0,
                terminal_scale=1.0,
                terminal_polish_triggered=False,
                terminal_polish_applied=False,
                terminal_polish_tail_steps=0,
                terminal_polish_passes=0,
                reason="plan_too_long",
                elapsed_ms=1000.0 * (time.perf_counter() - t0),
            )

        phase_actions = [np.asarray(primitive_lib.get_actions(pid), dtype=np.float64).copy() for pid in primitive_ids]
        phase_lengths = [int(actions.shape[0]) for actions in phase_actions]
        actions = np.concatenate(phase_actions, axis=0)
        tail_steps = self._terminal_polish_tail_steps()
        last_phase_len = phase_lengths[-1] if len(phase_lengths) > 0 else 0
        terminal_polish_start_idx = max(0, int(actions.shape[0]) - int(min(last_phase_len, tail_steps)))
        scalar_result = self._refine_actions(
            env,
            actions,
            allow_prefix_shrink=False,
            terminal_polish_start_idx=terminal_polish_start_idx,
        )

        split_actions: List[np.ndarray] = []
        cursor = 0
        for length in phase_lengths:
            split_actions.append(np.asarray(scalar_result.actions[cursor : cursor + length], dtype=np.float64).copy())
            cursor += length

        return PlanRefinementResult(
            primitive_ids=primitive_ids,
            phase_actions=split_actions,
            plan_length=int(len(primitive_ids)),
            total_steps=int(sum(phase_lengths)),
            applied=bool(scalar_result.applied),
            feasible=bool(scalar_result.feasible),
            cost_before=float(scalar_result.cost_before),
            cost_after=float(scalar_result.cost_after),
            iterations=int(scalar_result.iterations),
            cost_evaluations=int(scalar_result.cost_evaluations),
            terminal_scale=float(scalar_result.terminal_scale),
            terminal_polish_triggered=bool(scalar_result.terminal_polish_triggered),
            terminal_polish_applied=bool(scalar_result.terminal_polish_applied),
            terminal_polish_tail_steps=int(scalar_result.terminal_polish_tail_steps),
            terminal_polish_passes=int(scalar_result.terminal_polish_passes),
            cost_breakdown_before=dict(scalar_result.cost_breakdown_before),
            cost_breakdown_after=dict(scalar_result.cost_breakdown_after),
            reason=str(scalar_result.reason),
            elapsed_ms=float(1000.0 * (time.perf_counter() - t0)),
        )

    def _refine_actions(
        self,
        env,
        actions: np.ndarray,
        allow_prefix_shrink: bool,
        terminal_polish_start_idx: Optional[int],
    ) -> RefinementResult:
        t0 = time.perf_counter()
        actions = np.asarray(actions, dtype=np.float64)
        if (not self.enabled()) or actions.ndim != 2 or actions.shape[0] == 0:
            return RefinementResult(
                actions=actions.copy(),
                original_horizon=int(actions.shape[0]) if actions.ndim == 2 else 0,
                effective_horizon=int(actions.shape[0]) if actions.ndim == 2 else 0,
                applied=False,
                feasible=True,
                cost_before=0.0,
                cost_after=0.0,
                iterations=0,
                cost_evaluations=0,
                terminal_scale=1.0,
                terminal_polish_triggered=False,
                terminal_polish_applied=False,
                terminal_polish_tail_steps=0,
                terminal_polish_passes=0,
                reason="disabled",
                elapsed_ms=1000.0 * (time.perf_counter() - t0),
            )

        base_env = getattr(env, "unwrapped", env)
        vehicle = getattr(base_env, "vehicle", None)
        world_map = getattr(base_env, "map", None)
        state0 = getattr(vehicle, "state", None)
        if vehicle is None or world_map is None or state0 is None:
            return RefinementResult(
                actions=actions.copy(),
                original_horizon=int(actions.shape[0]),
                effective_horizon=int(actions.shape[0]),
                applied=False,
                feasible=True,
                cost_before=0.0,
                cost_after=0.0,
                iterations=0,
                cost_evaluations=0,
                terminal_scale=1.0,
                terminal_polish_triggered=False,
                terminal_polish_applied=False,
                terminal_polish_tail_steps=0,
                terminal_polish_passes=0,
                reason="missing_env_context",
                elapsed_ms=1000.0 * (time.perf_counter() - t0),
            )

        start_state = copy.deepcopy(state0)
        mode_sign = self._infer_mode_sign(actions)
        step_mode_signs = self._infer_step_mode_signs(actions, fallback_sign=mode_sign)
        terminal_scale = self._terminal_scale(start_state, world_map)
        base_rollout = self._rollout(base_env, start_state, actions)
        cost_before_breakdown = self._trajectory_cost_details(base_env, base_rollout, actions, mode_sign, terminal_scale)
        cost_before = float(cost_before_breakdown["total"])
        cost_evals = 1

        best_actions = actions.copy()
        best_rollout = base_rollout
        best_cost = cost_before
        improved = False

        max_passes = int(max(1, getattr(self.cfg, "PRIMITIVE_REFINEMENT_MAX_PASSES", 2)))
        steer_delta = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_STEER_DELTA", np.deg2rad(4.0)))
        speed_delta = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_SPEED_DELTA", 0.25))

        for _ in range(max_passes):
            updated_in_pass = False
            for step_idx in range(int(best_actions.shape[0])):
                candidate_actions, candidate_rollout, candidate_cost = self._optimize_single_step(
                    base_env=base_env,
                    reference_actions=best_actions,
                    reference_cost=best_cost,
                    start_state=start_state,
                    step_idx=step_idx,
                    steer_delta=steer_delta,
                    speed_delta=speed_delta,
                    mode_sign=mode_sign,
                    step_mode_sign=float(step_mode_signs[step_idx]),
                    terminal_scale=terminal_scale,
                )
                cost_evals += 8
                if candidate_cost + 1e-9 < best_cost:
                    best_actions = candidate_actions
                    best_rollout = candidate_rollout
                    best_cost = candidate_cost
                    updated_in_pass = True
                    improved = True
            if not updated_in_pass:
                break

        effective_horizon = int(best_actions.shape[0])
        if allow_prefix_shrink and bool(getattr(self.cfg, "PRIMITIVE_REFINEMENT_ALLOW_PREFIX_SHRINK", True)):
            best_prefix_rollout = best_rollout
            for horizon in range(1, int(best_actions.shape[0]) + 1):
                prefix_actions = best_actions[:horizon]
                prefix_rollout = self._rollout(base_env, start_state, prefix_actions)
                prefix_cost = self._trajectory_cost(base_env, prefix_rollout, prefix_actions, mode_sign, terminal_scale)
                cost_evals += 1
                if prefix_cost + 1e-9 < best_cost:
                    best_cost = prefix_cost
                    best_prefix_rollout = prefix_rollout
                    effective_horizon = int(horizon)
                    improved = True
            best_rollout = best_prefix_rollout

        scored_actions = best_actions[:effective_horizon] if effective_horizon < int(best_actions.shape[0]) else best_actions
        scored_step_mode_signs = step_mode_signs[:effective_horizon]
        terminal_polish_triggered = False
        terminal_polish_applied = False
        terminal_polish_tail_steps = 0
        terminal_polish_passes = 0

        if self._should_terminal_polish(start_state, world_map) and scored_actions.shape[0] > 0:
            terminal_polish_triggered = True
            polish_actions, polish_rollout, polish_cost, polish_applied, polish_evals, polish_tail_steps, polish_passes = self._terminal_polish_actions(
                base_env=base_env,
                start_state=start_state,
                reference_actions=np.asarray(scored_actions, dtype=np.float64).copy(),
                reference_rollout=best_rollout,
                reference_cost=best_cost,
                step_mode_signs=scored_step_mode_signs,
                mode_sign=mode_sign,
                terminal_scale=terminal_scale,
                tail_start_idx=terminal_polish_start_idx,
            )
            cost_evals += int(polish_evals)
            terminal_polish_applied = bool(polish_applied)
            terminal_polish_tail_steps = int(polish_tail_steps)
            terminal_polish_passes = int(polish_passes)
            if polish_applied:
                best_cost = float(polish_cost)
                best_rollout = polish_rollout
                if effective_horizon < int(best_actions.shape[0]):
                    best_actions = np.asarray(best_actions, dtype=np.float64).copy()
                    best_actions[:effective_horizon] = polish_actions
                else:
                    best_actions = np.asarray(polish_actions, dtype=np.float64).copy()
                scored_actions = polish_actions
                improved = True

        cost_after_breakdown = self._trajectory_cost_details(base_env, best_rollout, scored_actions, mode_sign, terminal_scale)

        elapsed_ms = 1000.0 * (time.perf_counter() - t0)
        return RefinementResult(
            actions=best_actions,
            original_horizon=int(actions.shape[0]),
            effective_horizon=effective_horizon,
            applied=bool(improved),
            feasible=bool(best_rollout["feasible"]),
            cost_before=float(cost_before),
            cost_after=float(cost_after_breakdown["total"]),
            iterations=int(max_passes),
            cost_evaluations=int(cost_evals),
            terminal_scale=float(terminal_scale),
            terminal_polish_triggered=bool(terminal_polish_triggered),
            terminal_polish_applied=bool(terminal_polish_applied),
            terminal_polish_tail_steps=int(terminal_polish_tail_steps),
            terminal_polish_passes=int(terminal_polish_passes),
            cost_breakdown_before=cost_before_breakdown,
            cost_breakdown_after=cost_after_breakdown,
            reason="ok" if improved else "no_improvement",
            elapsed_ms=float(elapsed_ms),
        )

    def _optimize_single_step(
        self,
        base_env,
        reference_actions: np.ndarray,
        reference_cost: float,
        start_state,
        step_idx: int,
        steer_delta: float,
        speed_delta: float,
        mode_sign: float,
        step_mode_sign: float,
        terminal_scale: float,
    ) -> Tuple[np.ndarray, Dict[str, object], float]:
        action = np.asarray(reference_actions[step_idx], dtype=np.float64)
        speed = float(action[1])
        steer = float(action[0])

        speed_candidates = [
            speed,
            self._clamp_speed(speed - speed_delta, step_mode_sign),
            self._clamp_speed(speed + speed_delta, step_mode_sign),
        ]
        steer_candidates = [
            steer,
            steer - steer_delta,
            steer + steer_delta,
        ]

        best_actions = reference_actions
        best_rollout = self._rollout(base_env, copy.deepcopy(start_state), reference_actions)
        best_cost = reference_cost

        for candidate_steer in steer_candidates:
            for candidate_speed in speed_candidates:
                cand_actions = np.asarray(reference_actions, dtype=np.float64).copy()
                cand_actions[step_idx, 0] = float(np.clip(candidate_steer, *self.cfg.VALID_STEER))
                cand_actions[step_idx, 1] = self._clamp_speed(candidate_speed, step_mode_sign)
                rollout = self._rollout(base_env, copy.deepcopy(start_state), cand_actions)
                cost = self._trajectory_cost(base_env, rollout, cand_actions, mode_sign, terminal_scale)
                if cost + 1e-9 < best_cost:
                    best_cost = cost
                    best_actions = cand_actions
                    best_rollout = rollout

        return best_actions, best_rollout, best_cost

    def _terminal_polish_actions(
        self,
        base_env,
        start_state,
        reference_actions: np.ndarray,
        reference_rollout: Dict[str, object],
        reference_cost: float,
        step_mode_signs: np.ndarray,
        mode_sign: float,
        terminal_scale: float,
        tail_start_idx: Optional[int],
    ) -> Tuple[np.ndarray, Dict[str, object], float, bool, int, int, int]:
        tail_steps = self._terminal_polish_tail_steps()
        polish_passes = self._terminal_polish_passes()
        steer_delta = self._terminal_polish_steer_delta()
        speed_delta = self._terminal_polish_speed_delta()
        if reference_actions.shape[0] == 0:
            return reference_actions, reference_rollout, reference_cost, False, 0, 0, 0

        default_start_idx = max(0, int(reference_actions.shape[0]) - tail_steps)
        if tail_start_idx is None:
            start_idx = default_start_idx
        else:
            start_idx = max(int(tail_start_idx), default_start_idx)
        best_actions = np.asarray(reference_actions, dtype=np.float64).copy()
        best_rollout = reference_rollout
        best_cost = float(reference_cost)
        improved = False
        actual_passes = 0
        cost_evals = 0
        use_front_body_first = self._front_body_first_terminal_polish_enabled()
        best_summary = self._terminal_polish_candidate_summary(
            base_env=base_env,
            rollout=best_rollout,
            actions=best_actions,
            step_mode_signs=step_mode_signs,
            terminal_scale=terminal_scale,
        )

        for _ in range(polish_passes):
            actual_passes += 1
            updated_in_pass = False
            for step_idx in range(start_idx, int(best_actions.shape[0])):
                action = np.asarray(best_actions[step_idx], dtype=np.float64)
                speed = float(action[1])
                steer = float(action[0])
                speed_candidates = [
                    speed,
                    self._clamp_speed(speed - speed_delta, float(step_mode_signs[step_idx])),
                    self._clamp_speed(speed + speed_delta, float(step_mode_signs[step_idx])),
                ]
                steer_candidates = [
                    steer,
                    steer - steer_delta,
                    steer + steer_delta,
                ]
                candidate_actions = best_actions
                candidate_rollout = best_rollout
                candidate_cost = best_cost
                candidate_summary = best_summary
                for candidate_steer in steer_candidates:
                    for candidate_speed in speed_candidates:
                        cand_actions = np.asarray(best_actions, dtype=np.float64).copy()
                        cand_actions[step_idx, 0] = float(np.clip(candidate_steer, *self.cfg.VALID_STEER))
                        cand_actions[step_idx, 1] = self._clamp_speed(candidate_speed, float(step_mode_signs[step_idx]))
                        rollout = self._rollout(base_env, copy.deepcopy(start_state), cand_actions)
                        cost = self._trajectory_cost(base_env, rollout, cand_actions, mode_sign, terminal_scale)
                        cost_evals += 1
                        if use_front_body_first:
                            summary = self._terminal_polish_candidate_summary(
                                base_env=base_env,
                                rollout=rollout,
                                actions=cand_actions,
                                step_mode_signs=step_mode_signs,
                                terminal_scale=terminal_scale,
                            )
                            if self._is_terminal_polish_candidate_better(summary, candidate_summary):
                                candidate_actions = cand_actions
                                candidate_rollout = rollout
                                candidate_cost = cost
                                candidate_summary = summary
                        elif cost + 1e-9 < candidate_cost:
                            candidate_actions = cand_actions
                            candidate_rollout = rollout
                            candidate_cost = cost
                if use_front_body_first:
                    if self._is_terminal_polish_candidate_better(candidate_summary, best_summary):
                        best_actions = candidate_actions
                        best_rollout = candidate_rollout
                        best_cost = candidate_cost
                        best_summary = candidate_summary
                        updated_in_pass = True
                        improved = True
                elif candidate_cost + 1e-9 < best_cost:
                    best_actions = candidate_actions
                    best_rollout = candidate_rollout
                    best_cost = candidate_cost
                    updated_in_pass = True
                    improved = True
            if not updated_in_pass:
                break

        return best_actions, best_rollout, best_cost, improved, cost_evals, int(best_actions.shape[0] - start_idx), actual_passes

    def _rollout(self, base_env, state0, actions: np.ndarray) -> Dict[str, object]:
        model = base_env.vehicle.kinetic_model
        num_step = int(getattr(self.cfg, "NUM_STEP", 1))

        states: List = [copy.deepcopy(state0)]
        clearances: List[float] = [self._state_clearance(base_env, state0)]
        goal_metrics: List[Dict[str, float]] = [self._state_goal_metrics(base_env, state0)]
        feasible = True
        collision = False

        state = copy.deepcopy(state0)
        for action in np.asarray(actions, dtype=np.float64):
            state = model.step(state, action, step_time=num_step)
            states.append(copy.deepcopy(state))
            clearance = self._state_clearance(base_env, state)
            clearances.append(float(clearance))
            goal_metrics.append(self._state_goal_metrics(base_env, state))
            if not self._is_state_valid(base_env, state):
                feasible = False
                collision = True
                break

        return {
            "states": states,
            "clearances": clearances,
            "goal_metrics": goal_metrics,
            "feasible": feasible,
            "collision": collision,
        }

    def _trajectory_cost(
        self,
        base_env,
        rollout: Dict[str, object],
        actions: np.ndarray,
        mode_sign: float,
        terminal_scale: float,
    ) -> float:
        return float(self._trajectory_cost_details(base_env, rollout, actions, mode_sign, terminal_scale)["total"])

    def _front_body_first_terminal_polish_enabled(self) -> bool:
        return bool(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_ENABLE", True))

    def _terminal_polish_tail_steps(self) -> int:
        default_value = int(max(1, getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_TAIL_STEPS", 6)))
        return int(max(1, getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_TAIL_STEPS", default_value)))

    def _terminal_polish_passes(self) -> int:
        default_value = int(max(1, getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_PASSES", 2)))
        return int(max(1, getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_PASSES", default_value)))

    def _terminal_polish_steer_delta(self) -> float:
        default_value = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_STEER_DELTA", np.deg2rad(1.25)))
        return float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_STEER_DELTA", default_value))

    def _terminal_polish_speed_delta(self) -> float:
        default_value = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_SPEED_DELTA", 0.06))
        return float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_SPEED_DELTA", default_value))

    def _terminal_polish_candidate_summary(
        self,
        base_env,
        rollout: Dict[str, object],
        actions: np.ndarray,
        step_mode_signs: np.ndarray,
        terminal_scale: float,
    ) -> Dict[str, float]:
        goal_metrics: Sequence[Dict[str, float]] = rollout.get("goal_metrics") or []
        if len(goal_metrics) > 0:
            final_metric = dict(goal_metrics[-1])
        else:
            states = rollout.get("states") or []
            final_state = states[-1] if len(states) > 0 else None
            final_metric = self._state_goal_metrics(base_env, final_state) if final_state is not None else {}
        front_overlap_target = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_FRONT_OVERLAP_TARGET", 0.85))
        rear_overlap_min = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_REAR_OVERLAP_MIN", 0.45))
        articulation_safe = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_ARTICULATION_SAFE", np.deg2rad(28.0)))
        desired_clearance = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_COST", {}).get("desired_clearance", 0.75))
        states = rollout.get("states") or []
        final_state = states[-1] if len(states) > 0 else None
        clearances = [float(value) for value in (rollout.get("clearances") or [])]
        finite_clearances = [value for value in clearances if not math.isinf(float(value))]
        min_clearance = min(finite_clearances) if len(finite_clearances) > 0 else float("inf")
        clearance_deficit = self._clearance_penalty(min_clearance, desired_clearance)
        control_energy = float(np.sum(np.square(np.asarray(actions, dtype=np.float64))))
        control_delta = 0.0
        if int(actions.shape[0]) > 1:
            control_delta = float(np.sum(np.square(np.diff(np.asarray(actions, dtype=np.float64), axis=0))))
        front_heading_error = abs(float(final_metric.get("front_heading_error", final_metric.get("heading_error", 0.0))))
        front_position_error = float(final_metric.get("front_position_error", final_metric.get("position_error", 0.0)))
        front_overlap = float(final_metric.get("front_overlap", final_metric.get("mean_overlap", 0.0)))
        rear_overlap = float(final_metric.get("rear_overlap", final_metric.get("mean_overlap", 0.0)))
        front_overlap_deficit = max(0.0, front_overlap_target - front_overlap)
        rear_overlap_deficit = max(0.0, rear_overlap_min - rear_overlap)
        articulation_abs = abs(self._articulation(final_state)) if final_state is not None else 0.0
        articulation_safe_penalty = float(max(0.0, articulation_abs - articulation_safe) ** 2)
        rear_soft_cost = (
            float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_ARTICULATION_SAFE_WEIGHT", 1.0)) * articulation_safe_penalty
            + float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_REAR_SOFT_WEIGHT", 1.0)) * rear_overlap_deficit
            + float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_CLEARANCE_WEIGHT", 1.0)) * clearance_deficit
            + float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_RUNNING_WEIGHT", 1.0)) * (control_energy + control_delta)
        )
        front_scale = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_NEAR_GOAL_SCALE", terminal_scale))
        scalar_fallback = (
            front_scale
            * (
                float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_SCALAR_HEADING_WEIGHT", 100.0)) * front_heading_error
                + float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_SCALAR_FRONT_OVERLAP_WEIGHT", 60.0)) * front_overlap_deficit
                + float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_SCALAR_FRONT_POSITION_WEIGHT", 20.0)) * front_position_error
            )
            + float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_SCALAR_SOFT_WEIGHT", 1.0)) * rear_soft_cost
        )
        return {
            "valid": 1.0 if (bool(rollout.get("feasible", True)) and self._actions_preserve_step_mode(actions, step_mode_signs)) else 0.0,
            "front_heading_error": float(front_heading_error),
            "front_overlap_deficit": float(front_overlap_deficit),
            "front_position_error": float(front_position_error),
            "rear_soft_cost": float(rear_soft_cost),
            "scalar_fallback": float(scalar_fallback),
            "front_overlap": float(front_overlap),
            "rear_overlap": float(rear_overlap),
            "articulation_safe_penalty": float(articulation_safe_penalty),
            "clearance_deficit": float(clearance_deficit),
        }

    def _is_terminal_polish_candidate_better(
        self,
        candidate: Dict[str, float],
        incumbent: Dict[str, float],
    ) -> bool:
        cand_valid = bool(candidate.get("valid", 0.0) > 0.5)
        inc_valid = bool(incumbent.get("valid", 0.0) > 0.5)
        if cand_valid != inc_valid:
            return cand_valid
        if not cand_valid:
            return False
        heading_eps = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_HEADING_EPS", np.deg2rad(0.5)))
        overlap_eps = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_OVERLAP_EPS", 0.01))
        position_eps = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_POSITION_EPS", 0.03))
        for key, eps in (
            ("front_heading_error", heading_eps),
            ("front_overlap_deficit", overlap_eps),
            ("front_position_error", position_eps),
        ):
            cand_value = float(candidate.get(key, 0.0))
            inc_value = float(incumbent.get(key, 0.0))
            if cand_value < inc_value - eps:
                return True
            if cand_value > inc_value + eps:
                return False
        cand_soft = float(candidate.get("rear_soft_cost", 0.0))
        inc_soft = float(incumbent.get("rear_soft_cost", 0.0))
        if cand_soft + 1e-9 < inc_soft:
            return True
        if cand_soft > inc_soft + 1e-9:
            return False
        return float(candidate.get("scalar_fallback", 0.0)) + 1e-9 < float(incumbent.get("scalar_fallback", 0.0))

    def _actions_preserve_step_mode(self, actions: np.ndarray, step_mode_signs: np.ndarray) -> bool:
        speeds = np.asarray(actions, dtype=np.float64)[:, 1]
        for speed, required_sign in zip(speeds, np.asarray(step_mode_signs, dtype=np.float64)):
            if required_sign > 1e-6 and float(speed) < -1e-6:
                return False
            if required_sign < -1e-6 and float(speed) > 1e-6:
                return False
        return True

    def _trajectory_cost_details(
        self,
        base_env,
        rollout: Dict[str, object],
        actions: np.ndarray,
        mode_sign: float,
        terminal_scale: float,
    ) -> Dict[str, float]:
        objective = str(getattr(self.cfg, "PRIMITIVE_REFINEMENT_OBJECTIVE", "terminal_window_v1"))
        if objective == "legacy":
            return self._trajectory_cost_legacy(base_env, rollout, actions, mode_sign, terminal_scale)
        return self._trajectory_cost_terminal_window(base_env, rollout, actions, mode_sign, terminal_scale)

    def _trajectory_cost_terminal_window(
        self,
        base_env,
        rollout: Dict[str, object],
        actions: np.ndarray,
        mode_sign: float,
        terminal_scale: float,
    ) -> Dict[str, float]:
        cfg_cost = dict(getattr(self.cfg, "PRIMITIVE_REFINEMENT_COST", {}))
        goal_state = getattr(base_env.map, "dest", None)
        if goal_state is None:
            penalty = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_INVALID_PENALTY", 5000.0))
            return {
                "running_cost": 0.0,
                "terminal_window_cost": 0.0,
                "final_barrier_cost": 0.0,
                "invalid_penalty": penalty,
                "total": penalty,
                "final_position_error": 0.0,
                "final_front_position_error": 0.0,
                "final_heading_error_deg": 0.0,
                "final_front_heading_error_deg": 0.0,
                "final_articulation_error_deg": 0.0,
                "final_front_overlap": 0.0,
                "final_rear_overlap": 0.0,
                "final_mean_overlap": 0.0,
                "final_front_overlap_deficit": 0.0,
                "final_rear_overlap_deficit": 0.0,
                "final_articulation_safe_penalty": 0.0,
                "final_heading_barrier": 0.0,
                "final_overlap_barrier": 0.0,
            }

        states: Sequence = rollout["states"]
        clearances: Sequence[float] = rollout["clearances"]
        goal_metrics: Sequence[Dict[str, float]] = rollout.get("goal_metrics") or [
            self._state_goal_metrics(base_env, state) for state in states
        ]
        desired_clearance = float(cfg_cost.get("desired_clearance", 0.75))

        running_cost = 0.0
        prev_action = None
        for idx, _state in enumerate(states[1:], start=0):
            action = np.asarray(actions[min(idx, actions.shape[0] - 1)], dtype=np.float64)
            clearance_pen = self._clearance_penalty(float(clearances[min(idx + 1, len(clearances) - 1)]), desired_clearance)
            control_cost = float(np.dot(action, action))
            delta_cost = 0.0 if prev_action is None else float(np.dot(action - prev_action, action - prev_action))
            running_cost += float(cfg_cost.get("running_clearance", 1000.0)) * float(clearance_pen)
            running_cost += float(cfg_cost.get("running_control", 0.02)) * float(control_cost)
            running_cost += float(cfg_cost.get("running_control_delta", 0.05)) * float(delta_cost)
            prev_action = action

        terminal_window = int(max(1, getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_WINDOW", 8)))
        alpha_scale = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_ALPHA_SCALE", 4.0))
        window_overlap_target = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_WINDOW_OVERLAP_TARGET", 0.72))
        window_front_overlap_target = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_WINDOW_FRONT_OVERLAP_TARGET", 0.82))
        state_window = min(int(len(goal_metrics)), terminal_window)
        window_start = max(0, int(len(goal_metrics)) - state_window)

        terminal_window_cost_raw = 0.0
        for offset, metric in enumerate(goal_metrics[window_start:]):
            alpha = 1.0 + alpha_scale * (float(offset) / float(max(1, state_window - 1)))
            overlap_deficit = max(0.0, window_overlap_target - float(metric["mean_overlap"]))
            front_overlap_deficit = max(0.0, window_front_overlap_target - float(metric["front_overlap"]))
            terminal_window_cost_raw += alpha * (
                float(cfg_cost.get("window_pos", 8.0)) * float(metric["position_error"] ** 2)
                + float(cfg_cost.get("window_heading", 30.0)) * float(metric["heading_error"] ** 2)
                + float(cfg_cost.get("window_articulation", 24.0)) * float(metric["articulation_error"] ** 2)
                + float(cfg_cost.get("window_overlap_deficit", 80.0)) * float(overlap_deficit ** 2)
                + float(cfg_cost.get("window_front_overlap_deficit", 220.0)) * float(front_overlap_deficit ** 2)
            )

        final_metric = goal_metrics[-1]
        final_overlap_target = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FINAL_OVERLAP_TARGET", 0.75))
        final_front_overlap_target = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FINAL_FRONT_OVERLAP_TARGET", 0.88))
        rear_overlap_floor = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_REAR_OVERLAP_MIN", 0.45))
        articulation_safe = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_ARTICULATION_SAFE", np.deg2rad(28.0)))
        overlap_barrier = max(0.0, final_overlap_target - float(final_metric["mean_overlap"]))
        front_overlap_barrier = max(0.0, final_front_overlap_target - float(final_metric["front_overlap"]))
        rear_overlap_deficit = max(0.0, rear_overlap_floor - float(final_metric.get("rear_overlap", final_metric["mean_overlap"])))
        articulation_safe_penalty = max(
            0.0,
            abs(float(self._articulation(states[-1]))) - articulation_safe,
        ) ** 2
        final_barrier_cost_raw = (
            float(cfg_cost.get("final_position", 28.0)) * float(final_metric["position_error"] ** 2)
            + float(cfg_cost.get("final_heading", 160.0)) * float(final_metric["heading_error"] ** 2)
            + float(cfg_cost.get("final_articulation", 120.0)) * float(final_metric["articulation_error"] ** 2)
            + float(cfg_cost.get("final_overlap_deficit", 320.0)) * float(overlap_barrier ** 2)
            + float(cfg_cost.get("final_front_overlap_deficit", 520.0)) * float(front_overlap_barrier ** 2)
        )

        terminal_window_cost = float(terminal_scale) * float(terminal_window_cost_raw)
        final_barrier_cost = float(terminal_scale) * float(final_barrier_cost_raw)
        invalid_penalty = 0.0
        if not bool(rollout.get("feasible", True)):
            invalid_penalty = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_INVALID_PENALTY", 5000.0))

        total = running_cost + terminal_window_cost + final_barrier_cost + invalid_penalty
        return {
            "running_cost": float(running_cost),
            "terminal_window_cost": float(terminal_window_cost),
            "final_barrier_cost": float(final_barrier_cost),
            "invalid_penalty": float(invalid_penalty),
            "total": float(total),
            "final_position_error": float(final_metric["position_error"]),
            "final_front_position_error": float(final_metric.get("front_position_error", final_metric["position_error"])),
            "final_heading_error_deg": float(np.degrees(abs(float(final_metric["heading_error"])))),
            "final_front_heading_error_deg": float(np.degrees(abs(float(final_metric.get("front_heading_error", final_metric["heading_error"]))))),
            "final_articulation_error_deg": float(np.degrees(abs(float(final_metric["articulation_error"])))),
            "final_front_overlap": float(final_metric["front_overlap"]),
            "final_rear_overlap": float(final_metric["rear_overlap"]),
            "final_mean_overlap": float(final_metric["mean_overlap"]),
            "final_front_overlap_deficit": float(front_overlap_barrier),
            "final_rear_overlap_deficit": float(rear_overlap_deficit),
            "final_articulation_safe_penalty": float(articulation_safe_penalty),
            "final_heading_barrier": float(abs(float(final_metric["heading_error"]))),
            "final_overlap_barrier": float(max(overlap_barrier, front_overlap_barrier)),
        }

    def _trajectory_cost_legacy(
        self,
        base_env,
        rollout: Dict[str, object],
        actions: np.ndarray,
        mode_sign: float,
        terminal_scale: float,
    ) -> Dict[str, float]:
        cfg_cost = dict(getattr(self.cfg, "PRIMITIVE_REFINEMENT_COST", {}))
        goal_state = getattr(base_env.map, "dest", None)
        if goal_state is None:
            penalty = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_INVALID_PENALTY", 5000.0))
            return {
                "running_cost": 0.0,
                "terminal_window_cost": 0.0,
                "final_barrier_cost": 0.0,
                "invalid_penalty": penalty,
                "total": penalty,
                "final_position_error": 0.0,
                "final_front_position_error": 0.0,
                "final_heading_error_deg": 0.0,
                "final_front_heading_error_deg": 0.0,
                "final_articulation_error_deg": 0.0,
                "final_front_overlap": 0.0,
                "final_rear_overlap": 0.0,
                "final_mean_overlap": 0.0,
                "final_front_overlap_deficit": 0.0,
                "final_rear_overlap_deficit": 0.0,
                "final_articulation_safe_penalty": 0.0,
                "final_heading_barrier": 0.0,
                "final_overlap_barrier": 0.0,
            }

        states: Sequence = rollout["states"]
        clearances: Sequence[float] = rollout["clearances"]
        goal_metrics: Sequence[Dict[str, float]] = rollout.get("goal_metrics") or [
            self._state_goal_metrics(base_env, state) for state in states
        ]
        start_dist = max(1e-6, float(states[0].loc.distance(goal_state.loc)))

        running_cost = 0.0
        prev_dist = float(states[0].loc.distance(goal_state.loc))
        prev_steer = float(getattr(states[0], "steering", 0.0))
        prev_speed_sign = mode_sign
        desired_clearance = float(cfg_cost.get("desired_clearance", 0.75))

        for idx, state in enumerate(states[1:], start=0):
            action = np.asarray(actions[min(idx, actions.shape[0] - 1)], dtype=np.float64)
            dist = float(state.loc.distance(goal_state.loc))
            progress = max(0.0, prev_dist - dist)
            beta = self._articulation_error(state, goal_state)
            steer = float(action[0])
            speed = float(action[1])
            steer_rate = float(steer - prev_steer)
            clearance_pen = self._clearance_penalty(float(clearances[min(idx + 1, len(clearances) - 1)]), desired_clearance)
            speed_sign = self._signed_mode(speed)
            mode_switch = 1.0 if (prev_speed_sign != 0.0 and speed_sign != 0.0 and speed_sign != prev_speed_sign) else 0.0

            running_cost += float(cfg_cost.get("step", 0.08))
            running_cost += float(cfg_cost.get("progress", 0.55)) * float(dist / start_dist)
            running_cost -= 0.25 * float(progress / start_dist)
            running_cost += float(cfg_cost.get("clearance", 2.50)) * float(clearance_pen)
            running_cost += float(cfg_cost.get("steer", 0.08)) * float(steer * steer)
            running_cost += float(cfg_cost.get("steer_rate", 0.18)) * float(steer_rate * steer_rate)
            running_cost += float(cfg_cost.get("articulation", 0.65)) * float(beta * beta)
            running_cost += float(cfg_cost.get("control", 0.02)) * float(steer * steer + speed * speed)
            running_cost += float(cfg_cost.get("mode_switch", 1.50)) * float(mode_switch)

            prev_dist = dist
            prev_steer = steer
            prev_speed_sign = speed_sign if speed_sign != 0.0 else prev_speed_sign

        final_metric = goal_metrics[-1]
        terminal_window_cost = float(terminal_scale) * (
            float(cfg_cost.get("terminal_pos", 1.60)) * float(final_metric["position_error"] ** 2)
            + float(cfg_cost.get("terminal_heading", 1.20)) * float(final_metric["heading_error"] ** 2)
            + float(cfg_cost.get("terminal_articulation", 1.10)) * float(final_metric["articulation_error"] ** 2)
            + float(cfg_cost.get("terminal_steer", 0.25)) * float(getattr(states[-1], "steering", 0.0) ** 2)
        )
        invalid_penalty = 0.0
        if not bool(rollout.get("feasible", True)):
            invalid_penalty = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_INVALID_PENALTY", 5000.0))

        total = running_cost + terminal_window_cost + invalid_penalty
        rear_overlap_floor = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_REAR_OVERLAP_MIN", 0.45))
        articulation_safe = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_ARTICULATION_SAFE", np.deg2rad(28.0)))
        rear_overlap_deficit = max(0.0, rear_overlap_floor - float(final_metric.get("rear_overlap", final_metric["mean_overlap"])))
        articulation_safe_penalty = max(
            0.0,
            abs(float(self._articulation(states[-1]))) - articulation_safe,
        ) ** 2
        return {
            "running_cost": float(running_cost),
            "terminal_window_cost": float(terminal_window_cost),
            "final_barrier_cost": 0.0,
            "invalid_penalty": float(invalid_penalty),
            "total": float(total),
            "final_position_error": float(final_metric["position_error"]),
            "final_front_position_error": float(final_metric.get("front_position_error", final_metric["position_error"])),
            "final_heading_error_deg": float(np.degrees(abs(float(final_metric["heading_error"])))),
            "final_front_heading_error_deg": float(np.degrees(abs(float(final_metric.get("front_heading_error", final_metric["heading_error"]))))),
            "final_articulation_error_deg": float(np.degrees(abs(float(final_metric["articulation_error"])))),
            "final_front_overlap": float(final_metric["front_overlap"]),
            "final_rear_overlap": float(final_metric["rear_overlap"]),
            "final_mean_overlap": float(final_metric["mean_overlap"]),
            "final_front_overlap_deficit": float(max(0.0, float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_FRONT_OVERLAP_TARGET", 0.85)) - float(final_metric["front_overlap"]))),
            "final_rear_overlap_deficit": float(rear_overlap_deficit),
            "final_articulation_safe_penalty": float(articulation_safe_penalty),
            "final_heading_barrier": 0.0,
            "final_overlap_barrier": 0.0,
        }

    def _is_state_valid(self, base_env, state) -> bool:
        world_map = base_env.map
        x = float(state.loc.x)
        y = float(state.loc.y)
        if x < float(world_map.xmin) or x > float(world_map.xmax) or y < float(world_map.ymin) or y > float(world_map.ymax):
            return False

        obstacles = getattr(world_map, "obstacles", []) or []
        try:
            boxes = state.create_box()
        except Exception:
            return False

        for box in boxes:
            for obst in obstacles:
                shape = getattr(obst, "shape", obst)
                if box.intersects(shape):
                    return False
        return True

    def _state_clearance(self, base_env, state) -> float:
        world_map = base_env.map
        obstacles = getattr(world_map, "obstacles", []) or []
        if len(obstacles) == 0:
            return float("inf")

        min_clearance = float("inf")
        for box in state.create_box():
            for obst in obstacles:
                shape = getattr(obst, "shape", obst)
                try:
                    dist = float(box.distance(shape))
                except Exception:
                    dist = 0.0
                if dist < min_clearance:
                    min_clearance = dist
        return float(min_clearance)

    def _clearance_penalty(self, clearance: float, desired_clearance: float) -> float:
        if math.isinf(clearance):
            return 0.0
        deficit = max(0.0, float(desired_clearance) - float(clearance))
        return float(deficit * deficit)

    def _state_goal_metrics(self, base_env, state) -> Dict[str, float]:
        goal_state = getattr(base_env.map, "dest", None)
        if goal_state is None:
            return {
                "position_error": 0.0,
                "front_position_error": 0.0,
                "heading_error": 0.0,
                "front_heading_error": 0.0,
                "articulation_error": 0.0,
                "front_overlap": 0.0,
                "rear_overlap": 0.0,
                "mean_overlap": 0.0,
            }

        front_overlap, rear_overlap, mean_overlap = self._overlap_ratios(state, goal_state)
        front_position_error = float(state.loc.distance(goal_state.loc))
        front_heading_error = float(_wrap_pi(float(state.heading) - float(goal_state.heading)))
        return {
            "position_error": float(front_position_error),
            "front_position_error": float(front_position_error),
            "heading_error": float(front_heading_error),
            "front_heading_error": float(front_heading_error),
            "articulation_error": float(self._articulation_error(state, goal_state)),
            "front_overlap": float(front_overlap),
            "rear_overlap": float(rear_overlap),
            "mean_overlap": float(mean_overlap),
        }

    def _overlap_ratios(self, state, goal_state) -> Tuple[float, float, float]:
        try:
            state_boxes = state.create_box()
            goal_boxes = goal_state.create_box()
        except Exception:
            return 0.0, 0.0, 0.0

        overlaps: List[float] = []
        for state_box, goal_box in zip(state_boxes, goal_boxes):
            try:
                poly_state = Polygon(state_box)
                poly_goal = Polygon(goal_box)
                area_goal = float(poly_goal.area) + 1e-9
                overlap = float(poly_state.intersection(poly_goal).area) / area_goal
            except Exception:
                overlap = 0.0
            overlaps.append(float(overlap))

        if len(overlaps) == 0:
            return 0.0, 0.0, 0.0
        front_overlap = float(overlaps[0])
        rear_overlap = float(overlaps[1]) if len(overlaps) > 1 else float(overlaps[0])
        return front_overlap, rear_overlap, float(np.mean(overlaps))

    def _articulation(self, state) -> float:
        rear_heading = float(getattr(state, "rear_heading", state.heading))
        return float(_wrap_pi(float(state.heading) - rear_heading))

    def _articulation_error(self, state, goal_state) -> float:
        return float(_wrap_pi(self._articulation(state) - self._articulation(goal_state)))

    def _should_terminal_polish(self, start_state, world_map) -> bool:
        if not bool(getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_ENABLE", True)):
            return False
        goal_state = getattr(world_map, "dest", None)
        if goal_state is None:
            return False
        trigger_dist = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_TRIGGER_DIST", 6.0))
        front_trigger_dist = getattr(self.cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_TRIGGER_DIST", None)
        if front_trigger_dist is not None:
            trigger_dist = float(min(trigger_dist, float(front_trigger_dist)))
        dist = float(start_state.loc.distance(goal_state.loc))
        return bool(dist <= trigger_dist)

    def _terminal_scale(self, start_state, world_map) -> float:
        terminal_dist = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_DIST", 6.0))
        dist = float(start_state.loc.distance(world_map.dest.loc))
        if dist <= terminal_dist:
            return float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_TERMINAL_SCALE", 4.0))
        return 1.0

    def _infer_mode_sign(self, actions: np.ndarray) -> float:
        speeds = np.asarray(actions, dtype=np.float64)[:, 1]
        nz = speeds[np.abs(speeds) > 1e-6]
        if nz.size == 0:
            return 0.0
        return self._signed_mode(float(np.mean(nz)))

    def _infer_step_mode_signs(self, actions: np.ndarray, fallback_sign: float) -> np.ndarray:
        speeds = np.asarray(actions, dtype=np.float64)[:, 1]
        signs = np.asarray([self._signed_mode(float(speed)) for speed in speeds], dtype=np.float64)
        if signs.size == 0:
            return signs
        last_sign = float(fallback_sign)
        for idx in range(int(signs.shape[0])):
            if abs(float(signs[idx])) > 1e-6:
                last_sign = float(signs[idx])
            else:
                signs[idx] = float(last_sign)
        next_sign = float(fallback_sign)
        for idx in range(int(signs.shape[0]) - 1, -1, -1):
            if abs(float(signs[idx])) > 1e-6:
                next_sign = float(signs[idx])
            else:
                signs[idx] = float(next_sign)
        if abs(float(fallback_sign)) > 1e-6:
            zero_mask = np.abs(signs) <= 1e-6
            signs[zero_mask] = float(fallback_sign)
        return signs

    def _signed_mode(self, value: float) -> float:
        if value > 1e-6:
            return 1.0
        if value < -1e-6:
            return -1.0
        return 0.0

    def _clamp_speed(self, speed: float, mode_sign: float) -> float:
        lo, hi = map(float, getattr(self.cfg, "VALID_SPEED", [-2.5, 2.5]))
        min_abs = float(getattr(self.cfg, "PRIMITIVE_REFINEMENT_MIN_SPEED_ABS", 0.20))
        speed = float(np.clip(speed, lo, hi))
        if mode_sign > 0.0:
            return float(max(min_abs, speed))
        if mode_sign < 0.0:
            return float(min(-min_abs, speed))
        return float(speed)