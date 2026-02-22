import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from primitives.primitive_index import PrimitiveGridIndex


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class PlanResult:
    primitive_ids: List[int]
    prefix_steps: Optional[int] = None
    debug: Dict = None


class RecedingHorizonTakeoverPlanner:
    """Online Receding Horizon (RHP) motion-primitive planner for terminal takeover.

    This follows the paper "Motion Primitives Planning For Center-Articulated Vehicles":
    - Offline primitive sets: we assume primitives are in ego frame; grouping is approximated
      via speed bins (control groups) and a shared prefix length per group.
    - Two-step collision detection: use PrimitiveGridIndex (offline cell index) + online occupancy
      from lidar to prune candidates without per-primitive rollout.
    - Online receding horizon: every call selects ONE short prefix (typically 1 primitive with
      prefix_steps << H). Wrapper re-plans at high frequency.
    """

    def __init__(
        self,
        primitive_actions: np.ndarray,
        primitive_deltas: np.ndarray,
        grid_index: PrimitiveGridIndex,
        lidar_num: int,
        lidar_range: float,
        score_weights: Dict[str, float],
        occupancy_inflation_radius: float,
        group_score_topk: int = 5,
        max_prefix_steps: Optional[int] = None,
        min_clearance_hit_penalty: float = 0.1,
    ) -> None:
        self.actions = primitive_actions
        self.deltas = primitive_deltas
        self.index = grid_index
        self.lidar_num = int(lidar_num)
        self.lidar_range = float(lidar_range)
        self.score_weights = dict(score_weights or {})
        self.occupancy_inflation_radius = float(occupancy_inflation_radius)
        self.group_score_topk = int(group_score_topk)
        self.max_prefix_steps = None if max_prefix_steps is None else int(max_prefix_steps)
        self.min_clearance_hit_penalty = float(min_clearance_hit_penalty)

        self._lidar_angles = np.linspace(0.0, 2.0 * math.pi, self.lidar_num, endpoint=False)

        # Precompute inflation offsets in cell coordinates
        r = max(0.0, self.occupancy_inflation_radius)
        res = max(1e-6, self.index.grid_resolution)
        rad = int(math.ceil(r / res))
        offsets = []
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                if (dx * dx + dy * dy) * (res * res) <= r * r + 1e-9:
                    offsets.append((dx, dy))
        self._inflation_offsets = offsets

    def _build_occupied_cells_from_lidar(self, lidar_norm: np.ndarray) -> Tuple[set, Dict]:
        """Build a sparse occupied-cell set from normalized lidar observation."""
        t0 = time.perf_counter()
        lidar_norm = np.asarray(lidar_norm, dtype=np.float64).reshape(-1)
        if lidar_norm.size != self.lidar_num:
            # best-effort: clip/resize
            lidar_norm = lidar_norm[: self.lidar_num]

        dist = np.clip(lidar_norm, 0.0, 1.0) * self.lidar_range

        occupied = set()
        hits = 0
        # Consider beams with a real hit (not max range)
        # Leave margin for noise.
        hit_mask = dist < (0.98 * self.lidar_range)

        for i in np.nonzero(hit_mask)[0]:
            d = float(dist[i])
            a = float(self._lidar_angles[i])
            x = d * math.cos(a)
            y = d * math.sin(a)
            cell = self.index.world_to_cell(x, y)
            if cell is None:
                continue
            hits += 1
            ix, iy = cell
            for dx, dy in self._inflation_offsets:
                occupied.add((ix + dx, iy + dy))

        dt_ms = 1000.0 * (time.perf_counter() - t0)
        debug = {
            "lidar_hit_beams": int(hits),
            "occupied_cells": int(len(occupied)),
            "occupancy_ms": float(dt_ms),
        }
        return occupied, debug

    def plan(self, state, obs, lidar, goal_repr, prev_choice=None, mode: str = "auto") -> PlanResult:
        """Plan one short prefix.

        Args:
            state: env.vehicle.State (only used for articulation continuity; optional)
            obs: full observation vector (unused except for debug)
            lidar: lidar normalized vector (shape [lidar_num])
            goal_repr: dict with keys {"goal_x","goal_y","goal_heading","articulation"} in ego frame
            prev_choice: previous primitive id (for smooth term)
            mode: "auto"/"forward"/"reverse" (simple bi-directional mode)
        """
        t_plan0 = time.perf_counter()

        debug: Dict = {}
        occupied_cells, occ_dbg = self._build_occupied_cells_from_lidar(lidar)
        debug.update(occ_dbg)

        # Fast prune by occupancy
        t0 = time.perf_counter()
        candidate_mask = self.index.fast_prune_primitives(occupied_cells)
        prune_ms = 1000.0 * (time.perf_counter() - t0)
        debug["fast_prune_ms"] = float(prune_ms)
        debug["candidates"] = int(candidate_mask.sum())

        if candidate_mask.sum() == 0:
            return PlanResult(primitive_ids=[], prefix_steps=None, debug={**debug, "reason": "no_candidate"})

        # Optional directional restriction
        if mode in ("forward", "reverse"):
            speeds = self.actions[:, 0, 1]
            if mode == "forward":
                candidate_mask &= (speeds > 0)
            else:
                candidate_mask &= (speeds < 0)
            debug["candidates_dir"] = int(candidate_mask.sum())
            if candidate_mask.sum() == 0:
                return PlanResult(primitive_ids=[], prefix_steps=None, debug={**debug, "reason": "no_candidate_dir"})

        # Near-hit counts for clearance penalty (also via inverted index)
        t0 = time.perf_counter()
        near_counts = self.index.count_near_hits(occupied_cells)
        debug["near_hits_ms"] = float(1000.0 * (time.perf_counter() - t0))

        # Goal quantities (ego frame)
        gx = float(goal_repr.get("goal_x", 0.0))
        gy = float(goal_repr.get("goal_y", 0.0))
        g_heading = float(goal_repr.get("goal_heading", 0.0))
        articulation = float(goal_repr.get("articulation", 0.0))

        # Scoring each primitive (heuristics)
        t0 = time.perf_counter()

        w_dist = float(self.score_weights.get("dist", 1.0))
        w_dir = float(self.score_weights.get("dir", 0.5))
        w_state = float(self.score_weights.get("state", 0.2))
        w_smooth = float(self.score_weights.get("smooth", 0.2))
        w_speed = float(self.score_weights.get("speed", 0.1))
        w_clear = float(self.score_weights.get("clearance", 0.3))

        # predicted end displacement in ego frame (canonical)
        dx = self.deltas[:, 0]
        dy = self.deltas[:, 1]
        dtheta = self.deltas[:, 2]
        dgamma = self.deltas[:, 3]

        # distance-to-goal after primitive
        nx = gx - dx
        ny = gy - dy
        dist_after = np.sqrt(nx * nx + ny * ny)
        dist_before = math.hypot(gx, gy) + 1e-6
        dist_improve = (dist_before - dist_after)

        # heading error after primitive (goal heading is relative to ego)
        hd_after = np.vectorize(_wrap_pi)(g_heading - dtheta)
        hd_cost = np.abs(hd_after)

        # articulation continuity (penalize big mismatch)
        gamma_cost = np.abs(np.vectorize(_wrap_pi)(dgamma - articulation))

        # speed preference
        speeds = self.actions[:, 0, 1]
        speed_score = np.abs(speeds)  # prefer non-zero

        # clearance penalty
        clear_pen = np.clip(near_counts.astype(np.float64), 0.0, 10.0)

        # smooth penalty by previous choice
        smooth_pen = np.zeros_like(dist_after)
        if prev_choice is not None:
            try:
                p = int(prev_choice)
                smooth_pen = np.sqrt((dx - float(dx[p])) ** 2 + (dy - float(dy[p])) ** 2)
            except Exception:
                pass

        primitive_score = (
            w_dist * dist_improve
            - w_dir * hd_cost
            - w_state * gamma_cost
            - w_smooth * smooth_pen
            + w_speed * speed_score
            - w_clear * self.min_clearance_hit_penalty * clear_pen
        )

        # mask infeasible
        primitive_score = np.where(candidate_mask, primitive_score, -1e9)
        debug["score_ms"] = float(1000.0 * (time.perf_counter() - t0))

        # Group aggregation: score group by mean of top-k primitives
        t0 = time.perf_counter()
        best_group = None
        best_group_score = None
        best_group_best_pid = None

        for gid, pids in enumerate(self.index.group_to_primitive_ids):
            if pids.size == 0:
                continue
            s = primitive_score[pids]
            # filter invalid
            s = s[s > -1e8]
            if s.size == 0:
                continue
            k = min(self.group_score_topk, int(s.size))
            # top-k mean
            topk = np.partition(s, -k)[-k:]
            gscore = float(np.mean(topk))
            if best_group_score is None or gscore > best_group_score:
                best_group_score = gscore
                best_group = gid
                # choose best primitive in this group
                pbest = int(pids[int(np.argmax(primitive_score[pids]))])
                best_group_best_pid = pbest

        debug["group_ms"] = float(1000.0 * (time.perf_counter() - t0))

        if best_group is None or best_group_best_pid is None:
            return PlanResult(primitive_ids=[], prefix_steps=None, debug={**debug, "reason": "no_group"})

        prefix = int(self.index.group_prefix_steps[int(best_group)])
        if self.max_prefix_steps is not None:
            prefix = min(prefix, self.max_prefix_steps)
        prefix = max(1, prefix)

        debug["chosen_group"] = int(best_group)
        debug["chosen_pid"] = int(best_group_best_pid)
        debug["prefix_steps"] = int(prefix)

        debug["plan_ms"] = float(1000.0 * (time.perf_counter() - t_plan0))

        return PlanResult(primitive_ids=[int(best_group_best_pid)], prefix_steps=prefix, debug=debug)
