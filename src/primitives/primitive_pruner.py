from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from primitives.trajectory_miner import CandidatePrimitive


def _action_l2_normalized(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / np.sqrt(float(a.size) + 1e-9))


@dataclass
class PruneReport:
    raw: int
    after_dedup: int
    after_proxy_prune: int
    after_feasibility: int


class PrimitivePruner:
    """Dedup / prune / validate candidates before inserting into library."""

    def __init__(self, verbose: bool = False):
        self.verbose = bool(verbose)

    def deduplicate(self, candidates: Sequence[CandidatePrimitive], current_library, config) -> List[CandidatePrimitive]:
        """Near-duplicate filtering against current library and within candidates."""
        cands = list(candidates)
        if len(cands) == 0:
            return []

        tau = float(getattr(config, "AP_DEDUP_ACTION_L2_TAU", 0.35))

        # library vectors
        lib_actions = None
        try:
            lib_actions = np.asarray(getattr(current_library, "actions"), dtype=np.float64)
        except Exception:
            lib_actions = None

        kept: List[CandidatePrimitive] = []
        for cand in cands:
            a = np.asarray(cand.actions_resampled, dtype=np.float64)

            # against library
            if lib_actions is not None and lib_actions.ndim == 3 and lib_actions.shape[1:] == a.shape:
                vec = a.reshape(-1)
                lib_vec = lib_actions.reshape(lib_actions.shape[0], -1)
                dists = np.linalg.norm(lib_vec - vec[None, :], axis=1) / np.sqrt(float(vec.size) + 1e-9)
                if dists.size > 0 and float(np.min(dists)) < tau:
                    continue

            # within kept
            dup = False
            for prev in kept:
                if _action_l2_normalized(prev.actions_resampled, a) < tau:
                    dup = True
                    break
            if dup:
                continue

            kept.append(cand)

        return kept

    def prune_by_proxy_value(
        self,
        candidates: Sequence[CandidatePrimitive],
        env_sampler=None,
        planner_eval_fn=None,
        config=None,
    ) -> List[CandidatePrimitive]:
        """Optional proxy pruning hook.

        For Phase1 we keep a no-op fallback unless user supplies env_sampler/planner_eval_fn.
        """
        cands = list(candidates)
        if len(cands) == 0:
            return []

        if not bool(getattr(config, "AP_ENABLE_PROXY_PRUNING", True)):
            return cands
        if env_sampler is None or planner_eval_fn is None:
            # no proxy evaluator available: keep top by final_score
            return cands

        # Expected interface:
        # - env_sampler(): yields a list of "hard" start states
        # - planner_eval_fn(state, actions_H)-> float proxy value
        try:
            states = list(env_sampler())
        except Exception:
            return cands

        topn = int(getattr(config, "AP_PROXY_PRUNE_TOPN", 20))
        cands = sorted(cands, key=lambda x: float(x.final_score), reverse=True)[: max(1, topn)]

        kept: List[CandidatePrimitive] = []
        for cand in cands:
            improved_any = False
            for s in states:
                try:
                    v = float(planner_eval_fn(s, cand.actions_resampled))
                    if v > 0.0:
                        improved_any = True
                        break
                except Exception:
                    continue
            if improved_any:
                kept.append(cand)

        return kept

    def validate_feasibility(self, candidates: Sequence[CandidatePrimitive], base_env_or_wrapper, config) -> List[CandidatePrimitive]:
        """Safety checks before adding to library.

        - Always checks action bounds (physical units).
        - Best-effort open-loop kinematic replay from the candidate's source start_state.
        """
        cands = list(candidates)
        if len(cands) == 0:
            return []

        try:
            steer_min, steer_max = float(getattr(config, "VALID_STEER")[0]), float(getattr(config, "VALID_STEER")[1])
            v_min, v_max = float(getattr(config, "VALID_SPEED")[0]), float(getattr(config, "VALID_SPEED")[1])
        except Exception:
            steer_min, steer_max = -np.deg2rad(36), np.deg2rad(36)
            v_min, v_max = -2.5, 2.5

        kept: List[CandidatePrimitive] = []
        for cand in cands:
            u = np.asarray(cand.actions_resampled, dtype=np.float64)
            if u.ndim != 2 or u.shape[1] != 2:
                continue
            if np.any(u[:, 0] < steer_min - 1e-6) or np.any(u[:, 0] > steer_max + 1e-6):
                continue
            if np.any(u[:, 1] < v_min - 1e-6) or np.any(u[:, 1] > v_max + 1e-6):
                continue

            if not self._open_loop_replay_ok(cand, base_env_or_wrapper, config):
                continue

            kept.append(cand)

        return kept

    def _open_loop_replay_ok(self, cand: CandidatePrimitive, base_env_or_wrapper, config) -> bool:
        """Kinematic replay feasibility from recorded start state (best-effort).

        Returns True when replay succeeds or when replay is unavailable.
        """
        start_state = None
        try:
            start_state = cand.source_metadata.get("start_state", None)
        except Exception:
            start_state = None
        if not isinstance(start_state, dict):
            return True

        # Need vehicle model + map validity checker
        wrapper = base_env_or_wrapper
        base_env = getattr(wrapper, "env", wrapper)
        vehicle = getattr(base_env, "vehicle", None)
        world_map = getattr(base_env, "map", None)

        if vehicle is None or getattr(vehicle, "kinetic_model", None) is None or world_map is None:
            return True

        # Build State object
        try:
            from env.vehicle import State

            raw = [
                float(start_state.get("x", 0.0)),
                float(start_state.get("y", 0.0)),
                float(start_state.get("heading", 0.0)),
                float(start_state.get("speed", 0.0)),
                float(start_state.get("steering", 0.0)),
                float(start_state.get("rear_heading", float(start_state.get("heading", 0.0)))),
            ]
            s = State(raw)
        except Exception:
            return True

        # validity check helper (prefer wrapper's method)
        is_valid = getattr(wrapper, "_is_state_valid", None)
        if is_valid is None:
            return True

        try:
            from configs import NUM_STEP

            step_time = int(NUM_STEP)
        except Exception:
            step_time = None

        try:
            u = np.asarray(cand.actions_resampled, dtype=np.float64)
            for t in range(u.shape[0]):
                if step_time is None:
                    s = vehicle.kinetic_model.step(s, u[t])
                else:
                    s = vehicle.kinetic_model.step(s, u[t], step_time=step_time)
                if not bool(is_valid(s)):
                    return False
        except Exception:
            return True

        return True

    def report(self, raw: int, after_dedup: int, after_proxy: int, after_feas: int) -> PruneReport:
        return PruneReport(raw=raw, after_dedup=after_dedup, after_proxy_prune=after_proxy, after_feasibility=after_feas)
