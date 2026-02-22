import os
import pickle
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


Cell = Tuple[int, int]


@dataclass
class PrimitiveGridIndex:
    """Offline grid index for fast online pruning.

    This mirrors the paper's two-step collision detection idea:
    - Offline: precompute which grid cells each primitive trajectory visits.
    - Online: build an occupancy grid from lidar, then use a cell->primitive inverted index
      to mark occluded/colliding candidates in ~O(#occupied_cells + #hits).

    Notes for this repo:
    - Primitives are stored as action sequences in local ego frame (start at x=y=0, heading=0).
    - We only index sparse centerline points (degraded but fast); collision corridor is approximated
      by online inflation in occupancy construction.
    """

    # geometry
    grid_resolution: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # inverted indices
    primitive_to_cells: List[np.ndarray]  # list of int64 [K_i, 2] cell coords
    cell_to_primitives: Dict[Cell, np.ndarray]  # (ix,iy)->np.int64 primitive ids

    # control-group metadata
    primitive_to_group_id: np.ndarray  # int64 [N]
    group_to_primitive_ids: List[np.ndarray]  # list of int64 arrays
    group_prefix_steps: np.ndarray  # int64 [G]

    @property
    def num_primitives(self) -> int:
        return int(self.primitive_to_group_id.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.group_prefix_steps.shape[0])

    def world_to_cell(self, x: float, y: float) -> Optional[Cell]:
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return None
        ix = int(np.floor((x - self.x_min) / self.grid_resolution))
        iy = int(np.floor((y - self.y_min) / self.grid_resolution))
        return (ix, iy)

    def fast_prune_primitives(self, occupied_cells: Iterable[Cell]) -> np.ndarray:
        """Return a boolean mask (True=candidate) using inverted index."""
        n = self.num_primitives
        blocked = np.zeros(n, dtype=np.bool_)
        for cell in occupied_cells:
            pids = self.cell_to_primitives.get(cell)
            if pids is None:
                continue
            blocked[pids] = True
        return ~blocked

    def count_near_hits(self, occupied_cells: Iterable[Cell]) -> np.ndarray:
        """Count how many occupied cells each primitive intersects.

        Implemented via the same inverted index, so complexity follows hits.
        """
        counts = np.zeros(self.num_primitives, dtype=np.int32)
        for cell in occupied_cells:
            pids = self.cell_to_primitives.get(cell)
            if pids is None:
                continue
            counts[pids] += 1
        return counts


def _default_index_path(npz_path: str) -> str:
    base, _ = os.path.splitext(npz_path)
    return base + ".grid_index.npz"


def load_primitive_grid_index(index_path: str) -> PrimitiveGridIndex:
    if index_path.endswith(".pkl"):
        with open(index_path, "rb") as f:
            return pickle.load(f)

    data = np.load(index_path, allow_pickle=True)

    primitive_to_cells = data["primitive_to_cells"].tolist()
    # stored as list of arrays of shape [K_i,2]

    cell_to_primitives = data["cell_to_primitives"].item()
    primitive_to_group_id = np.asarray(data["primitive_to_group_id"], dtype=np.int64)
    group_to_primitive_ids = data["group_to_primitive_ids"].tolist()
    group_prefix_steps = np.asarray(data["group_prefix_steps"], dtype=np.int64)

    return PrimitiveGridIndex(
        grid_resolution=float(data["grid_resolution"]),
        x_min=float(data["x_min"]),
        y_min=float(data["y_min"]),
        x_max=float(data["x_max"]),
        y_max=float(data["y_max"]),
        primitive_to_cells=primitive_to_cells,
        cell_to_primitives=cell_to_primitives,
        primitive_to_group_id=primitive_to_group_id,
        group_to_primitive_ids=group_to_primitive_ids,
        group_prefix_steps=group_prefix_steps,
    )


def try_load_index_for_library(npz_path: str, explicit_index_path: Optional[str] = None) -> Optional[PrimitiveGridIndex]:
    candidates: List[str] = []
    if explicit_index_path:
        candidates.append(explicit_index_path)
    candidates.append(_default_index_path(npz_path))

    for p in candidates:
        if p and os.path.exists(p):
            try:
                return load_primitive_grid_index(p)
            except Exception:
                continue
    return None


def build_approx_index_from_deltas(
    actions: np.ndarray,
    deltas: np.ndarray,
    grid_resolution: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    group_prefix_steps: int,
    max_samples_per_primitive: int = 64,
) -> PrimitiveGridIndex:
    """Build a degraded-but-fast index using only delta endpoints.

    This is a fallback when an offline index file is not present yet.
    It samples points along the straight segment from (0,0) -> (dx,dy) in ego frame.

    Online complexity still follows O(#occupied_cells + #hits).
    """
    actions = np.asarray(actions)
    deltas = np.asarray(deltas)
    n = int(actions.shape[0])

    def world_to_cell(x: float, y: float) -> Optional[Cell]:
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return None
        ix = int(np.floor((x - x_min) / grid_resolution))
        iy = int(np.floor((y - y_min) / grid_resolution))
        return (ix, iy)

    # control-group approximation by speed bin
    speeds = actions[:, 0, 1]
    uniq = sorted(list({float(np.round(v, 3)) for v in speeds}))
    speed_to_gid = {v: i for i, v in enumerate(uniq)}
    primitive_to_group_id = np.array([speed_to_gid[float(np.round(v, 3))] for v in speeds], dtype=np.int64)

    group_to_primitive_ids: List[List[int]] = [[] for _ in range(len(uniq))]
    for pid, gid in enumerate(primitive_to_group_id.tolist()):
        group_to_primitive_ids[gid].append(pid)

    primitive_to_cells: List[np.ndarray] = []
    cell_to_prims: Dict[Cell, List[int]] = {}

    for pid in range(n):
        dx = float(deltas[pid, 0])
        dy = float(deltas[pid, 1])
        length = math.hypot(dx, dy)
        # choose sample count roughly by resolution
        k = max(2, int(math.ceil(length / max(1e-6, grid_resolution))))
        k = min(k, int(max_samples_per_primitive))
        xs = np.linspace(0.0, dx, k)
        ys = np.linspace(0.0, dy, k)

        visited: Set[Cell] = set()
        for x, y in zip(xs, ys):
            c = world_to_cell(float(x), float(y))
            if c is not None:
                visited.add(c)

        arr = np.array(sorted(list(visited)), dtype=np.int64) if len(visited) > 0 else np.zeros((0, 2), dtype=np.int64)
        primitive_to_cells.append(arr)
        for cell in visited:
            if cell not in cell_to_prims:
                cell_to_prims[cell] = []
            cell_to_prims[cell].append(pid)

    cell_to_primitives = {k: np.array(v, dtype=np.int64) for k, v in cell_to_prims.items()}
    group_to_primitive_ids_arr = [np.array(v, dtype=np.int64) for v in group_to_primitive_ids]
    group_prefix_steps_arr = np.full((len(group_to_primitive_ids_arr),), int(group_prefix_steps), dtype=np.int64)

    return PrimitiveGridIndex(
        grid_resolution=float(grid_resolution),
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_max),
        y_max=float(y_max),
        primitive_to_cells=primitive_to_cells,
        cell_to_primitives=cell_to_primitives,
        primitive_to_group_id=primitive_to_group_id,
        group_to_primitive_ids=group_to_primitive_ids_arr,
        group_prefix_steps=group_prefix_steps_arr,
    )
