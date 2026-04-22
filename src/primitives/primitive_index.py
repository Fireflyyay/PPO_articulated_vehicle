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

    # optional metadata for richer offline indices
    index_kind: str = "approx_centerline"
    primitive_envelope_bounds: Optional[np.ndarray] = None  # float64 [N,4]

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


def _default_mask_index_path(npz_path: str) -> str:
    base, _ = os.path.splitext(npz_path)
    return base + ".mask_index.npz"


def _normalize_cell_arrays(raw_cells: Sequence[np.ndarray]) -> List[np.ndarray]:
    normalized: List[np.ndarray] = []
    for cells in raw_cells:
        arr = np.asarray(cells, dtype=np.int64)
        if arr.size == 0:
            arr = np.zeros((0, 2), dtype=np.int64)
        else:
            arr = arr.reshape(-1, 2)
        normalized.append(arr)
    return normalized


def _build_group_metadata(actions: np.ndarray, group_prefix_steps: int) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    speeds = actions[:, 0, 1]
    uniq = sorted(list({float(np.round(v, 3)) for v in speeds}))
    speed_to_gid = {v: i for i, v in enumerate(uniq)}
    primitive_to_group_id = np.array([speed_to_gid[float(np.round(v, 3))] for v in speeds], dtype=np.int64)

    group_to_primitive_ids: List[List[int]] = [[] for _ in range(len(uniq))]
    for pid, gid in enumerate(primitive_to_group_id.tolist()):
        group_to_primitive_ids[gid].append(pid)

    group_to_primitive_ids_arr = [np.array(v, dtype=np.int64) for v in group_to_primitive_ids]
    group_prefix_steps_arr = np.full((len(group_to_primitive_ids_arr),), int(group_prefix_steps), dtype=np.int64)
    return primitive_to_group_id, group_to_primitive_ids_arr, group_prefix_steps_arr


def _build_cell_to_primitives(primitive_to_cells: Sequence[np.ndarray]) -> Dict[Cell, np.ndarray]:
    cell_to_prims: Dict[Cell, List[int]] = {}
    for pid, cells in enumerate(primitive_to_cells):
        for ix, iy in np.asarray(cells, dtype=np.int64).reshape(-1, 2):
            cell = (int(ix), int(iy))
            if cell not in cell_to_prims:
                cell_to_prims[cell] = []
            cell_to_prims[cell].append(int(pid))
    return {cell: np.asarray(pids, dtype=np.int64) for cell, pids in cell_to_prims.items()}


def _coord_to_cell_index(value: float, coord_min: float, coord_max: float, grid_resolution: float) -> int:
    clipped = min(max(float(value), float(coord_min)), float(np.nextafter(coord_max, coord_min)))
    return int(math.floor((clipped - coord_min) / grid_resolution))


def _cell_bounds(ix: int, iy: int, grid_resolution: float, x_min: float, y_min: float) -> Tuple[float, float, float, float]:
    left = float(x_min + ix * grid_resolution)
    bottom = float(y_min + iy * grid_resolution)
    return (left, bottom, left + grid_resolution, bottom + grid_resolution)


def _state_from_values(x: float, y: float, heading: float, speed: float, steering: float, rear_heading: float):
    from env.vehicle import State

    return State([float(x), float(y), float(heading), float(speed), float(steering), float(rear_heading)])


def simulate_primitive_states_canonical(
    actions: np.ndarray,
    num_step: int,
    step_len: Optional[float] = None,
    mini_iter: int = 20,
    speed_range: Optional[Sequence[float]] = None,
    angle_range: Optional[Sequence[float]] = None,
    wheel_base: Optional[float] = None,
    trailer_length: Optional[float] = None,
    hitch_offset: Optional[float] = None,
) -> List:
    from configs import HITCH_OFFSET, STEP_LENGTH, TRAILER_LENGTH, VALID_SPEED, VALID_STEER, WHEEL_BASE

    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    if actions.size == 0:
        return [_state_from_values(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]

    speed_range = VALID_SPEED if speed_range is None else speed_range
    angle_range = VALID_STEER if angle_range is None else angle_range
    step_len = STEP_LENGTH if step_len is None else float(step_len)
    wheel_base = WHEEL_BASE if wheel_base is None else float(wheel_base)
    trailer_length = TRAILER_LENGTH if trailer_length is None else float(trailer_length)
    hitch_offset = HITCH_OFFSET if hitch_offset is None else float(hitch_offset)

    x = 0.0
    y = 0.0
    heading = 0.0
    rear_heading = 0.0
    phi_max = float(np.deg2rad(36.0))
    dt = float(step_len) / max(1, int(mini_iter))

    states = [_state_from_values(x, y, heading, 0.0, 0.0, rear_heading)]

    for raw_action in actions:
        steer, speed_cmd = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        speed = float(np.clip(speed_cmd, speed_range[0], speed_range[1]))
        steering = float(np.clip(steer, angle_range[0], angle_range[1]))

        if abs(trailer_length) > 1e-9 or abs(hitch_offset) > 1e-9:
            for _ in range(max(1, int(num_step))):
                for _ in range(max(1, int(mini_iter))):
                    phi = (heading - rear_heading + math.pi) % (2.0 * math.pi) - math.pi
                    effective_omega = steering
                    if phi >= phi_max and steering > 0.0:
                        effective_omega = 0.0
                    elif phi <= -phi_max and steering < 0.0:
                        effective_omega = 0.0

                    denom = hitch_offset * math.cos(phi) + trailer_length
                    if abs(denom) < 1e-6:
                        denom = 1e-6

                    theta1_dot = (speed * math.sin(phi) + trailer_length * effective_omega) / denom
                    theta2_dot = theta1_dot - effective_omega
                    x += speed * math.cos(heading) * dt
                    y += speed * math.sin(heading) * dt
                    heading += theta1_dot * dt
                    rear_heading += theta2_dot * dt
                    states.append(_state_from_values(x, y, heading, speed, steering, rear_heading))
        else:
            heading_dot = speed * math.tan(steering) / max(wheel_base, 1e-6)
            for _ in range(max(1, int(num_step))):
                for _ in range(max(1, int(mini_iter))):
                    x += speed * math.cos(heading) * dt
                    y += speed * math.sin(heading) * dt
                    heading += heading_dot * dt
                    rear_heading = heading
                    states.append(_state_from_values(x, y, heading, speed, steering, rear_heading))

    return states


def create_articulated_boxes_canonical(state) -> Tuple:
    from shapely.geometry import Polygon

    front_box, rear_box = state.create_box()
    return (Polygon(front_box), Polygon(rear_box))


def rasterize_boxes_to_cells(
    boxes: Sequence,
    grid_resolution: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    from shapely.geometry import box as shapely_box

    visited: Set[Cell] = set()
    for poly in boxes:
        if poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        if maxx < x_min or minx > x_max or maxy < y_min or miny > y_max:
            continue

        start_ix = _coord_to_cell_index(minx, x_min, x_max, grid_resolution)
        end_ix = _coord_to_cell_index(maxx, x_min, x_max, grid_resolution)
        start_iy = _coord_to_cell_index(miny, y_min, y_max, grid_resolution)
        end_iy = _coord_to_cell_index(maxy, y_min, y_max, grid_resolution)

        for ix in range(start_ix, end_ix + 1):
            for iy in range(start_iy, end_iy + 1):
                left, bottom, right, top = _cell_bounds(ix, iy, grid_resolution, x_min, y_min)
                cell_poly = shapely_box(left, bottom, right, top)
                if poly.intersects(cell_poly):
                    visited.add((int(ix), int(iy)))

    if not visited:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(visited), dtype=np.int64)


def primitive_grid_index_to_payload(index: PrimitiveGridIndex) -> Dict[str, np.ndarray]:
    payload = dict(
        grid_resolution=float(index.grid_resolution),
        x_min=float(index.x_min),
        x_max=float(index.x_max),
        y_min=float(index.y_min),
        y_max=float(index.y_max),
        primitive_to_cells=np.asarray(index.primitive_to_cells, dtype=object),
        cell_to_primitives=np.asarray(index.cell_to_primitives, dtype=object),
        primitive_to_group_id=np.asarray(index.primitive_to_group_id, dtype=np.int64),
        group_to_primitive_ids=np.asarray(index.group_to_primitive_ids, dtype=object),
        group_prefix_steps=np.asarray(index.group_prefix_steps, dtype=np.int64),
        index_kind=np.asarray(index.index_kind, dtype=object),
    )
    if index.primitive_envelope_bounds is not None:
        payload["primitive_envelope_bounds"] = np.asarray(index.primitive_envelope_bounds, dtype=np.float64)
    return payload


def load_primitive_grid_index(index_path: str) -> PrimitiveGridIndex:
    if index_path.endswith(".pkl"):
        with open(index_path, "rb") as f:
            return pickle.load(f)

    data = np.load(index_path, allow_pickle=True)

    primitive_to_cells = _normalize_cell_arrays(data["primitive_to_cells"].tolist())
    # stored as list of arrays of shape [K_i,2]

    cell_to_primitives = data["cell_to_primitives"].item()
    primitive_to_group_id = np.asarray(data["primitive_to_group_id"], dtype=np.int64)
    group_to_primitive_ids = [np.asarray(item, dtype=np.int64).reshape(-1) for item in data["group_to_primitive_ids"].tolist()]
    group_prefix_steps = np.asarray(data["group_prefix_steps"], dtype=np.int64)
    index_kind = str(data["index_kind"].item()) if "index_kind" in data else "approx_centerline"
    primitive_envelope_bounds = None
    if "primitive_envelope_bounds" in data:
        primitive_envelope_bounds = np.asarray(data["primitive_envelope_bounds"], dtype=np.float64).reshape(-1, 4)

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
        index_kind=index_kind,
        primitive_envelope_bounds=primitive_envelope_bounds,
    )


def try_load_index_for_library(npz_path: str, explicit_index_path: Optional[str] = None) -> Optional[PrimitiveGridIndex]:
    candidates: List[str] = []
    if explicit_index_path:
        candidates.append(explicit_index_path)
    candidates.append(_default_mask_index_path(npz_path))
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

    primitive_to_group_id, group_to_primitive_ids_arr, group_prefix_steps_arr = _build_group_metadata(
        actions=actions,
        group_prefix_steps=group_prefix_steps,
    )

    primitive_to_cells: List[np.ndarray] = []

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

    cell_to_primitives = _build_cell_to_primitives(primitive_to_cells)

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
        index_kind="approx_centerline",
    )


def build_primitive_swept_cells(
    actions: np.ndarray,
    grid_resolution: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sample_stride: int,
    num_step: int,
    group_prefix_steps: int,
) -> PrimitiveGridIndex:
    actions = np.asarray(actions, dtype=np.float64)
    n = int(actions.shape[0])
    sample_stride = max(1, int(sample_stride))

    primitive_to_group_id, group_to_primitive_ids_arr, group_prefix_steps_arr = _build_group_metadata(
        actions=actions,
        group_prefix_steps=group_prefix_steps,
    )

    primitive_to_cells: List[np.ndarray] = []
    primitive_envelope_bounds = np.zeros((n, 4), dtype=np.float64)

    for pid in range(n):
        states = simulate_primitive_states_canonical(actions[pid], num_step=num_step)
        if sample_stride > 1:
            sampled_states = states[::sample_stride]
            if sampled_states[-1] is not states[-1]:
                sampled_states = list(sampled_states) + [states[-1]]
        else:
            sampled_states = states

        visited: Set[Cell] = set()
        env_minx = math.inf
        env_miny = math.inf
        env_maxx = -math.inf
        env_maxy = -math.inf

        for state in sampled_states:
            boxes = create_articulated_boxes_canonical(state)
            env_minx = min(env_minx, *(poly.bounds[0] for poly in boxes))
            env_miny = min(env_miny, *(poly.bounds[1] for poly in boxes))
            env_maxx = max(env_maxx, *(poly.bounds[2] for poly in boxes))
            env_maxy = max(env_maxy, *(poly.bounds[3] for poly in boxes))
            cells = rasterize_boxes_to_cells(
                boxes=boxes,
                grid_resolution=grid_resolution,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
            for ix, iy in cells:
                visited.add((int(ix), int(iy)))

        primitive_to_cells.append(np.asarray(sorted(visited), dtype=np.int64) if visited else np.zeros((0, 2), dtype=np.int64))

        if env_minx is math.inf:
            primitive_envelope_bounds[pid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            primitive_envelope_bounds[pid] = np.array([env_minx, env_miny, env_maxx, env_maxy], dtype=np.float64)

    cell_to_primitives = _build_cell_to_primitives(primitive_to_cells)

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
        index_kind="swept_cells",
        primitive_envelope_bounds=primitive_envelope_bounds,
    )


def build_mask_index_from_library(
    actions: np.ndarray,
    grid_resolution: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sample_stride: int,
    num_step: int,
    group_prefix_steps: int,
) -> PrimitiveGridIndex:
    return build_primitive_swept_cells(
        actions=actions,
        grid_resolution=grid_resolution,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        sample_stride=sample_stride,
        num_step=num_step,
        group_prefix_steps=group_prefix_steps,
    )
