from __future__ import annotations

import hashlib
import heapq
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from configs import (
    BLOCK_MIXING_PLANT_CONFIG,
    FRONT_HANG,
    GUIDANCE_GRID_RESOLUTION,
    GUIDANCE_MAP_MARGIN,
    GUIDANCE_OBS_INFLATION,
    LENGTH,
    NAVIGATION_MIN_ENDPOINT_CLEARANCE_BY_LEVEL,
    NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL,
    NAVIGATION_PATH_RATIO_LIMIT_BY_LEVEL,
    NAVIGATION_TIGHT_TURN_HEADING_DEG_BY_LEVEL,
    NAVIGATION_TIGHT_TURN_MIN_ENDPOINT_CLEARANCE_BY_LEVEL,
    WIDTH,
)
from env.vehicle import State


WORLD_MIN = -40.0
WORLD_MAX = 40.0
_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS_8 = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass
class ParkingBay:
    center: tuple[float, float]
    heading: float
    length: float
    depth: float
    grid_cells: list[tuple[int, int]]
    access_cells: list[tuple[int, int]] | None = None
    polygon: Any | None = None


@dataclass
class BlockMixingPlantScene:
    occupancy_grid: np.ndarray
    free_grid: np.ndarray
    obstacle_polygons: list
    parking_bays: list[ParkingBay]
    metadata: dict


@dataclass
class _CorridorSegment:
    start: tuple[int, int]
    end: tuple[int, int]
    width: int
    kind: str
    orientation: str
    direction: int
    bbox: tuple[int, int, int, int]
    cell_length: int

    def to_dict(self) -> dict:
        return {
            "start": tuple(int(v) for v in self.start),
            "end": tuple(int(v) for v in self.end),
            "width": int(self.width),
            "kind": str(self.kind),
            "orientation": str(self.orientation),
            "direction": int(self.direction),
            "bbox": tuple(int(v) for v in self.bbox),
            "cell_length": int(self.cell_length),
        }


@dataclass
class _PoseCandidate:
    pose: list[float]
    bay_index: int
    bay_heading: float
    anchor_cell: tuple[int, int]
    reverse: bool


def _wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _abs_angle_diff(lhs: float, rhs: float) -> float:
    return abs(_wrap_pi(float(lhs) - float(rhs)))


def _range_sample(rng: np.random.Generator, value) -> int:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        low = int(value[0])
        high = int(value[1])
        if high < low:
            low, high = high, low
        return int(rng.integers(low, high + 1))
    return int(value)


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _grid_origin(config: dict) -> tuple[float, float]:
    width_m = float(config["grid_width"]) * float(config["block_size"])
    height_m = float(config["grid_height"]) * float(config["block_size"])
    return (-0.5 * width_m, -0.5 * height_m)


def _cell_bounds_world(config: dict, x0: int, y0: int, x1: int, y1: int) -> tuple[float, float, float, float]:
    origin_x, origin_y = _grid_origin(config)
    block_size = float(config["block_size"])
    return (
        origin_x + float(x0) * block_size,
        origin_y + float(y0) * block_size,
        origin_x + float(x1) * block_size,
        origin_y + float(y1) * block_size,
    )


def _cell_center_world(config: dict, x: int, y: int) -> tuple[float, float]:
    origin_x, origin_y = _grid_origin(config)
    block_size = float(config["block_size"])
    return (
        origin_x + (float(x) + 0.5) * block_size,
        origin_y + (float(y) + 0.5) * block_size,
    )


def _world_to_grid_cell(config: dict, x: float, y: float) -> Optional[tuple[int, int]]:
    origin_x, origin_y = _grid_origin(config)
    block_size = float(config["block_size"])
    gx = int(math.floor((float(x) - origin_x) / block_size))
    gy = int(math.floor((float(y) - origin_y) / block_size))
    if gx < 0 or gx >= int(config["grid_width"]) or gy < 0 or gy >= int(config["grid_height"]):
        return None
    return gx, gy


def carve_rect(occupancy: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    height, width = occupancy.shape
    xx0 = max(0, min(int(x0), int(x1)))
    yy0 = max(0, min(int(y0), int(y1)))
    xx1 = min(width, max(int(x0), int(x1)))
    yy1 = min(height, max(int(y0), int(y1)))
    if xx0 >= xx1 or yy0 >= yy1:
        return
    occupancy[yy0:yy1, xx0:xx1] = 0


def carve_corridor(occupancy: np.ndarray, start: tuple[int, int], end: tuple[int, int], width: int) -> None:
    sx, sy = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])
    if sx == ex or sy == ey:
        _carve_axis_segment(occupancy, (sx, sy), (ex, ey), int(width))
        return
    pivot = (ex, sy)
    _carve_axis_segment(occupancy, (sx, sy), pivot, int(width))
    _carve_axis_segment(occupancy, pivot, (ex, ey), int(width))


def _carve_axis_segment(occupancy: np.ndarray, start: tuple[int, int], end: tuple[int, int], width: int) -> Optional[_CorridorSegment]:
    sx, sy = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])
    half_low = (int(width) - 1) // 2
    half_high = int(width) // 2
    if sx == ex:
        y0 = min(sy, ey)
        y1 = max(sy, ey) + 1
        carve_rect(occupancy, sx - half_low, y0, sx + half_high + 1, y1)
        bbox = (sx - half_low, y0, sx + half_high + 1, y1)
        return _CorridorSegment(
            start=(sx, sy),
            end=(ex, ey),
            width=int(width),
            kind="unknown",
            orientation="vertical",
            direction=1 if ey >= sy else -1,
            bbox=bbox,
            cell_length=abs(ey - sy) + 1,
        )
    if sy == ey:
        x0 = min(sx, ex)
        x1 = max(sx, ex) + 1
        carve_rect(occupancy, x0, sy - half_low, x1, sy + half_high + 1)
        bbox = (x0, sy - half_low, x1, sy + half_high + 1)
        return _CorridorSegment(
            start=(sx, sy),
            end=(ex, ey),
            width=int(width),
            kind="unknown",
            orientation="horizontal",
            direction=1 if ex >= sx else -1,
            bbox=bbox,
            cell_length=abs(ex - sx) + 1,
        )
    return None


def _copy_segment_with_kind(segment: _CorridorSegment, kind: str) -> _CorridorSegment:
    return _CorridorSegment(
        start=segment.start,
        end=segment.end,
        width=segment.width,
        kind=str(kind),
        orientation=segment.orientation,
        direction=segment.direction,
        bbox=segment.bbox,
        cell_length=segment.cell_length,
    )


def _carve_pad(occupancy: np.ndarray, point: tuple[int, int], size: int) -> None:
    radius = max(1, int(size) // 2)
    x, y = int(point[0]), int(point[1])
    carve_rect(occupancy, x - radius, y - radius, x + radius + 1, y + radius + 1)


def _compress_waypoints(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return []
    compact: list[tuple[int, int]] = [tuple(int(v) for v in points[0])]
    for point in points[1:]:
        pt = tuple(int(v) for v in point)
        if pt != compact[-1]:
            compact.append(pt)
    idx = 1
    while idx < len(compact) - 1:
        prev_pt = compact[idx - 1]
        cur_pt = compact[idx]
        next_pt = compact[idx + 1]
        if (prev_pt[0] == cur_pt[0] == next_pt[0]) or (prev_pt[1] == cur_pt[1] == next_pt[1]):
            compact.pop(idx)
            continue
        idx += 1
    return compact


def _sample_bridge_value(rng: np.random.Generator, start_v: int, end_v: int, low: int, high: int) -> int:
    lo = min(int(start_v), int(end_v)) + 1
    hi = max(int(start_v), int(end_v)) - 1
    lo = max(int(low), lo)
    hi = min(int(high), hi)
    if lo <= hi:
        return int(rng.integers(lo, hi + 1))
    return int(rng.integers(int(low), int(high) + 1))


def _build_manhattan_waypoints(
    start: tuple[int, int],
    end: tuple[int, int],
    config: dict,
    rng: np.random.Generator,
    prefer_dogleg: bool,
) -> list[tuple[int, int]]:
    if tuple(start) == tuple(end):
        return [tuple(int(v) for v in start)]

    margin = int(config["boundary_margin"]) + max(2, int(config["corridor_width_range"][1]))
    x_low = margin
    x_high = int(config["grid_width"]) - margin - 1
    y_low = margin
    y_high = int(config["grid_height"]) - margin - 1
    dogleg = bool(prefer_dogleg and rng.random() < float(config["turn_probability"]))
    order = "hv" if rng.random() < 0.5 else "vh"

    sx, sy = int(start[0]), int(start[1])
    ex, ey = int(end[0]), int(end[1])
    if order == "hv":
        if dogleg:
            mid_x = _sample_bridge_value(rng, sx, ex, x_low, x_high)
            mid_y = _sample_bridge_value(rng, sy, ey, y_low, y_high)
            return _compress_waypoints([(sx, sy), (mid_x, sy), (mid_x, mid_y), (ex, mid_y), (ex, ey)])
        return _compress_waypoints([(sx, sy), (ex, sy), (ex, ey)])
    if dogleg:
        mid_y = _sample_bridge_value(rng, sy, ey, y_low, y_high)
        mid_x = _sample_bridge_value(rng, sx, ex, x_low, x_high)
        return _compress_waypoints([(sx, sy), (sx, mid_y), (mid_x, mid_y), (mid_x, ey), (ex, ey)])
    return _compress_waypoints([(sx, sy), (sx, ey), (ex, ey)])


def _distance_to_margin(point: tuple[int, int], direction: tuple[int, int], config: dict) -> int:
    margin = int(config["boundary_margin"]) + int(config["corridor_width_range"][1])
    x_low = margin
    x_high = int(config["grid_width"]) - margin - 1
    y_low = margin
    y_high = int(config["grid_height"]) - margin - 1
    x, y = int(point[0]), int(point[1])
    dx, dy = int(direction[0]), int(direction[1])
    if dx > 0:
        return max(0, x_high - x)
    if dx < 0:
        return max(0, x - x_low)
    if dy > 0:
        return max(0, y_high - y)
    return max(0, y - y_low)


def _perpendicular_directions(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dx, dy = int(direction[0]), int(direction[1])
    if dx != 0:
        return (0, 1), (0, -1)
    return (1, 0), (-1, 0)


def _random_walk_waypoints(
    start: tuple[int, int],
    direction: tuple[int, int],
    config: dict,
    rng: np.random.Generator,
    max_segments: int,
) -> list[tuple[int, int]]:
    low_len, high_len = tuple(int(v) for v in config["segment_length_range"])
    current = (int(start[0]), int(start[1]))
    current_dir = (int(direction[0]), int(direction[1]))
    points = [current]
    for _ in range(int(max_segments)):
        max_len = _distance_to_margin(current, current_dir, config)
        if max_len < 2:
            break
        length = min(_range_sample(rng, (low_len, high_len)), max_len)
        if length < 2:
            break
        next_pt = (current[0] + current_dir[0] * length, current[1] + current_dir[1] * length)
        points.append(next_pt)
        current = next_pt
        if rng.random() >= float(config["turn_probability"]):
            continue
        current_dir = _choice(rng, _perpendicular_directions(current_dir))
    return _compress_waypoints(points)


def _carve_waypoint_path(
    occupancy: np.ndarray,
    waypoints: list[tuple[int, int]],
    width: int,
    kind: str,
    carved_segments: list[_CorridorSegment],
) -> None:
    if len(waypoints) < 2:
        return
    for idx in range(len(waypoints) - 1):
        segment = _carve_axis_segment(occupancy, waypoints[idx], waypoints[idx + 1], int(width))
        if segment is None:
            continue
        carved_segments.append(_copy_segment_with_kind(segment, kind))
        if idx < len(waypoints) - 2:
            _carve_pad(occupancy, waypoints[idx + 1], int(width) + 1)
    _carve_pad(occupancy, waypoints[0], int(width))
    _carve_pad(occupancy, waypoints[-1], int(width))


def _collect_frontier_choices(occupancy: np.ndarray, config: dict) -> list[tuple[int, int, tuple[int, int]]]:
    margin = int(config["boundary_margin"]) + int(config["corridor_width_range"][1])
    height, width = occupancy.shape
    choices: list[tuple[int, int, tuple[int, int]]] = []
    for y in range(margin, height - margin):
        for x in range(margin, width - margin):
            if int(occupancy[y, x]) != 0:
                continue
            for dx, dy in _CARDINALS:
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                if int(occupancy[ny, nx]) != 1:
                    continue
                choices.append((x, y, (dx, dy)))
    return choices


def _sample_center_anchor(config: dict, rng: np.random.Generator) -> tuple[int, int]:
    jitter = 2 if str(config.get("grid_width")) != "72" else 3
    center_x = int(config["grid_width"]) // 2 + int(rng.integers(-jitter, jitter + 1))
    center_y = int(config["grid_height"]) // 2 + int(rng.integers(-jitter, jitter + 1))
    margin = int(config["boundary_margin"]) + max(2, int(config["corridor_width_range"][1]))
    center_x = int(np.clip(center_x, margin, int(config["grid_width"]) - margin - 1))
    center_y = int(np.clip(center_y, margin, int(config["grid_height"]) - margin - 1))
    return center_x, center_y


def _sample_boundary_anchor(config: dict, rng: np.random.Generator, side: int) -> tuple[int, int]:
    margin = int(config["boundary_margin"]) + max(2, int(config["corridor_width_range"][1]))
    x_low = margin
    x_high = int(config["grid_width"]) - margin - 1
    y_low = margin
    y_high = int(config["grid_height"]) - margin - 1
    if side == 0:
        return x_low, int(rng.integers(y_low, y_high + 1))
    if side == 1:
        return x_high, int(rng.integers(y_low, y_high + 1))
    if side == 2:
        return int(rng.integers(x_low, x_high + 1)), y_low
    return int(rng.integers(x_low, x_high + 1)), y_high


def _largest_component_mask(free_grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = free_grid.shape
    visited = np.zeros((height, width), dtype=bool)
    best_mask = np.zeros((height, width), dtype=bool)
    best_size = 0
    total_free = int(np.count_nonzero(free_grid))
    for y in range(height):
        for x in range(width):
            if not bool(free_grid[y, x]) or bool(visited[y, x]):
                continue
            queue: deque[tuple[int, int]] = deque([(x, y)])
            component: list[tuple[int, int]] = []
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                component.append((cx, cy))
                for dx, dy in _CARDINALS:
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if bool(visited[ny, nx]) or not bool(free_grid[ny, nx]):
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
            if len(component) <= best_size:
                continue
            best_size = len(component)
            best_mask.fill(False)
            for cx, cy in component:
                best_mask[cy, cx] = True
    return best_mask, int(best_size), int(total_free)


def _cleanup_free_islands(occupancy: np.ndarray) -> tuple[int, float]:
    free_grid = occupancy == 0
    keep_mask, component_size, total_free = _largest_component_mask(free_grid)
    if total_free > 0:
        occupancy[np.logical_and(free_grid, np.logical_not(keep_mask))] = 1
    ratio = float(component_size) / float(total_free) if total_free > 0 else 0.0
    removed = max(0, int(total_free) - int(component_size))
    return removed, ratio


def _grid_free_ratio(occupancy: np.ndarray) -> float:
    return float(np.count_nonzero(occupancy == 0)) / float(occupancy.size)


def _filter_bays_against_grid(parking_bays: list[ParkingBay], occupancy: np.ndarray) -> list[ParkingBay]:
    filtered: list[ParkingBay] = []
    for bay in parking_bays:
        if len(bay.grid_cells) == 0:
            continue
        if any(int(occupancy[gy, gx]) != 0 for gx, gy in bay.grid_cells):
            continue
        filtered.append(bay)
    return filtered


def _merge_occupied_runs(occupancy: np.ndarray) -> list[tuple[int, int, int, int]]:
    active: dict[tuple[int, int], list[int]] = {}
    merged: list[tuple[int, int, int, int]] = []
    height, width = occupancy.shape
    for y in range(height):
        row_runs: list[tuple[int, int]] = []
        x = 0
        while x < width:
            if int(occupancy[y, x]) != 1:
                x += 1
                continue
            x0 = x
            while x < width and int(occupancy[y, x]) == 1:
                x += 1
            row_runs.append((x0, x))

        next_active: dict[tuple[int, int], list[int]] = {}
        for run in row_runs:
            rect = active.pop(run, None)
            if rect is None:
                next_active[run] = [run[0], y, run[1], y + 1]
            else:
                rect[3] = y + 1
                next_active[run] = rect
        for rect in active.values():
            merged.append((rect[0], rect[1], rect[2], rect[3]))
        active = next_active

    for rect in active.values():
        merged.append((rect[0], rect[1], rect[2], rect[3]))
    return merged


def _grid_obstacle_polygons(occupancy: np.ndarray, config: dict) -> list[Polygon]:
    polygons: list[Polygon] = []
    merge_mode = str(config.get("obstacle_merge_mode", "rect_merge"))
    if merge_mode == "cell_polygons":
        ys, xs = np.where(occupancy == 1)
        for y, x in zip(ys.tolist(), xs.tolist()):
            wx0, wy0, wx1, wy1 = _cell_bounds_world(config, x, y, x + 1, y + 1)
            polygons.append(box(wx0, wy0, wx1, wy1))
        return polygons

    for x0, y0, x1, y1 in _merge_occupied_runs(occupancy):
        wx0, wy0, wx1, wy1 = _cell_bounds_world(config, x0, y0, x1, y1)
        polygons.append(box(wx0, wy0, wx1, wy1))
    return polygons


def _outer_world_obstacle_polygons(config: dict) -> list[Polygon]:
    origin_x, origin_y = _grid_origin(config)
    width_m = float(config["grid_width"]) * float(config["block_size"])
    height_m = float(config["grid_height"]) * float(config["block_size"])
    grid_min_x = origin_x
    grid_max_x = origin_x + width_m
    grid_min_y = origin_y
    grid_max_y = origin_y + height_m
    strips = [
        box(WORLD_MIN, WORLD_MIN, grid_min_x, WORLD_MAX),
        box(grid_max_x, WORLD_MIN, WORLD_MAX, WORLD_MAX),
        box(grid_min_x, WORLD_MIN, grid_max_x, grid_min_y),
        box(grid_min_x, grid_max_y, grid_max_x, WORLD_MAX),
    ]
    return [poly for poly in strips if (not poly.is_empty) and float(poly.area) > 1e-9]


def _build_guidance_payload(occupancy: np.ndarray, config: dict, difficulty: str) -> dict:
    raise RuntimeError("_build_guidance_payload requires obstacle lists; call the updated overload")


def _obstacle_signature(obstacles) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for obstacle in obstacles:
        try:
            hasher.update(obstacle.wkb)
        except Exception:
            fallback = (getattr(obstacle, "geom_type", type(obstacle).__name__), tuple(round(float(v), 4) for v in getattr(obstacle, "bounds", ())))
            hasher.update(repr(fallback).encode("utf-8"))
    return hasher.hexdigest()


def _build_guidance_payload_from_obstacles(
    occupancy: np.ndarray,
    config: dict,
    difficulty: str,
    static_obstacles: list,
    dynamic_obstacles: list | None = None,
) -> dict:
    grid_resolution = float(GUIDANCE_GRID_RESOLUTION)
    map_margin = float(GUIDANCE_MAP_MARGIN)
    obstacle_inflation = float(GUIDANCE_OBS_INFLATION)
    bounds = (WORLD_MIN - map_margin, WORLD_MAX + map_margin, WORLD_MIN - map_margin, WORLD_MAX + map_margin)
    nx = int(math.ceil((bounds[1] - bounds[0]) / grid_resolution)) + 1
    ny = int(math.ceil((bounds[3] - bounds[2]) / grid_resolution)) + 1
    occ = np.ones((nx, ny), dtype=np.uint8)

    free_y, free_x = np.where(occupancy == 0)
    for gy, gx in zip(free_y.tolist(), free_x.tolist()):
        world_x, world_y = _cell_center_world(config, gx, gy)
        ix = int(round((world_x - bounds[0]) / grid_resolution))
        iy = int(round((world_y - bounds[2]) / grid_resolution))
        if 0 <= ix < nx and 0 <= iy < ny:
            occ[ix, iy] = 0

    static_signature = _obstacle_signature(static_obstacles)
    dynamic_signature = _obstacle_signature(dynamic_obstacles or [])
    obstacle_signature = f"{static_signature}|{dynamic_signature}"
    grid_key = (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
        float(grid_resolution),
        float(obstacle_inflation),
        int(nx),
        int(ny),
    )
    static_cache_key = tuple(grid_key) + ("static", str(static_signature))
    dynamic_cache_key = tuple(grid_key) + ("dynamic", str(dynamic_signature))
    cache_key = tuple(grid_key) + (str(obstacle_signature),)
    dynamic_occ = np.zeros_like(occ, dtype=np.uint8)
    return {
        "cache_key": tuple(cache_key),
        "bounds": tuple(float(v) for v in bounds),
        "occ": np.array(occ, copy=True),
        "static_occ": np.array(occ, copy=True),
        "dynamic_occ": np.array(dynamic_occ, copy=True),
        "obstacle_signature": str(obstacle_signature),
        "static_cache_key": tuple(static_cache_key),
        "dynamic_cache_key": tuple(dynamic_cache_key),
        "static_obstacle_signature": str(static_signature),
        "dynamic_obstacle_signature": str(dynamic_signature),
    }


def _validate_scene_data(
    occupancy: np.ndarray,
    parking_bays: list[ParkingBay],
    config: dict,
    metadata: dict,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if occupancy.ndim != 2:
        reasons.append("occupancy_grid must be 2D")
        return False, reasons

    boundary_margin = int(config["boundary_margin"])
    if np.any(occupancy[:boundary_margin, :] == 0) or np.any(occupancy[-boundary_margin:, :] == 0):
        reasons.append("boundary rows carved open")
    if np.any(occupancy[:, :boundary_margin] == 0) or np.any(occupancy[:, -boundary_margin:] == 0):
        reasons.append("boundary cols carved open")

    free_ratio = _grid_free_ratio(occupancy)
    if free_ratio < float(config["min_free_ratio"]) - 1e-6:
        reasons.append(f"free ratio below minimum: {free_ratio:.3f}")
    if free_ratio > float(config["max_free_ratio"]) + 1e-6:
        reasons.append(f"free ratio above maximum: {free_ratio:.3f}")

    _, component_ratio = metadata.get("island_cleanup", (0, 0.0))
    if float(component_ratio) < 0.90:
        reasons.append(f"largest component ratio too small: {component_ratio:.3f}")

    min_bays = max(2, int(config["parking_bay_count_range"][0]))
    if len(parking_bays) < min_bays:
        reasons.append(f"insufficient parking bays: {len(parking_bays)} < {min_bays}")

    if metadata.get("largest_component_free_cells", 0) <= 0:
        reasons.append("scene has no free cells")

    return len(reasons) == 0, reasons


def validate_scene(scene: BlockMixingPlantScene, config: dict) -> bool:
    valid, _ = _validate_scene_data(scene.occupancy_grid, scene.parking_bays, config, dict(scene.metadata))
    return bool(valid)


def _generate_main_corridors(
    occupancy: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    carved_segments: list[_CorridorSegment],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    hub = _sample_center_anchor(config, rng)
    endpoints: list[tuple[int, int]] = []
    side_cycle = list(rng.permutation(np.arange(4, dtype=np.int64)).tolist())
    while len(side_cycle) < int(config["main_corridor_count"]):
        side_cycle.extend(list(rng.permutation(np.arange(4, dtype=np.int64)).tolist()))

    for idx in range(int(config["main_corridor_count"])):
        endpoint = _sample_boundary_anchor(config, rng, int(side_cycle[idx]))
        width = _range_sample(rng, config["corridor_width_range"])
        waypoints = _build_manhattan_waypoints(hub, endpoint, config, rng, prefer_dogleg=True)
        _carve_waypoint_path(occupancy, waypoints, width, kind="main", carved_segments=carved_segments)
        endpoints.append(endpoint)
    return hub, endpoints


def _generate_branches_or_deadends(
    occupancy: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    carved_segments: list[_CorridorSegment],
    target_count: int,
    kind: str,
    stop_free_ratio: float | None = None,
) -> None:
    if target_count <= 0:
        return
    min_width = int(config["corridor_width_range"][0])
    if kind == "branch":
        width_penalty = 2
        max_segments = 1 if str(config["grid_width"]) == "42" else 2
        len_scale = 0.55
    else:
        width_penalty = 3
        max_segments = 1
        len_scale = 0.40

    branch_config = dict(config)
    seg_low = int(config["segment_length_range"][0])
    seg_high = int(config["segment_length_range"][1])
    branch_config["segment_length_range"] = (
        max(3, int(round(seg_low * len_scale))),
        max(4, int(round(seg_high * len_scale))),
    )
    for _ in range(int(target_count)):
        if stop_free_ratio is not None and _grid_free_ratio(occupancy) >= float(stop_free_ratio):
            return
        choices = _collect_frontier_choices(occupancy, branch_config)
        if not choices:
            return
        start_x, start_y, direction = _choice(rng, choices)
        width = max(min_width, _range_sample(rng, config["corridor_width_range"]) - width_penalty)
        waypoints = _random_walk_waypoints((start_x, start_y), direction, branch_config, rng, max_segments=max_segments)
        if len(waypoints) < 2:
            continue
        _carve_waypoint_path(occupancy, waypoints, width, kind=kind, carved_segments=carved_segments)
        if kind == "branch" and rng.random() < float(config["turn_probability"]) * 0.20:
            _carve_pad(occupancy, waypoints[-1], width + 1)


def _generate_loops(
    occupancy: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    carved_segments: list[_CorridorSegment],
    target_count: int,
    stop_free_ratio: float | None = None,
) -> None:
    if target_count <= 0:
        return
    min_width = int(config["corridor_width_range"][0])
    loop_config = dict(config)
    seg_low = int(config["segment_length_range"][0])
    seg_high = int(config["segment_length_range"][1])
    loop_config["segment_length_range"] = (
        max(4, int(round(seg_low * 0.65))),
        max(5, int(round(seg_high * 0.65))),
    )
    for _ in range(int(target_count)):
        if stop_free_ratio is not None and _grid_free_ratio(occupancy) >= float(stop_free_ratio):
            return
        choices = _collect_frontier_choices(occupancy, loop_config)
        if len(choices) < 2:
            return
        start_x, start_y, _ = _choice(rng, choices)
        candidates = [it for it in choices if abs(it[0] - start_x) + abs(it[1] - start_y) >= max(6, int(config["segment_length_range"][0]))]
        if not candidates:
            continue
        end_x, end_y, _ = _choice(rng, candidates)
        width = max(2, min_width)
        waypoints = _build_manhattan_waypoints((start_x, start_y), (end_x, end_y), loop_config, rng, prefer_dogleg=False)
        if len(waypoints) < 2:
            continue
        _carve_waypoint_path(occupancy, waypoints, width, kind="loop", carved_segments=carved_segments)


def _generate_parking_bays(
    occupancy: np.ndarray,
    carved_segments: list[_CorridorSegment],
    config: dict,
    rng: np.random.Generator,
    difficulty: str,
) -> list[ParkingBay]:
    parking_bays: list[ParkingBay] = []
    target_count = _range_sample(rng, config["parking_bay_count_range"])
    max_attempts = int(config.get("max_parking_bay_attempts", 240))
    boundary_margin = int(config["boundary_margin"])
    far_wall_buffer_cells = int(max(1, config.get("parking_bay_far_wall_buffer_cells", 1)))
    min_spacing = float(config["parking_bay_min_spacing"]) * float(config["block_size"])
    min_bay_length = int(config["parking_bay_length_range"][0])
    eligible_segments = [
        segment
        for segment in carved_segments
        if segment.kind in {"main", "branch", "loop"} and segment.cell_length >= (min_bay_length + 1)
    ]
    if not eligible_segments:
        return parking_bays
    eligible_segments.sort(key=lambda segment: (segment.kind != "main", -int(segment.cell_length)))

    def _is_far_enough(center_xy: tuple[float, float]) -> bool:
        for bay in parking_bays:
            if math.hypot(float(center_xy[0]) - float(bay.center[0]), float(center_xy[1]) - float(bay.center[1])) < min_spacing:
                return False
        return True

    def _keeps_far_side_walled(bay_x0: int, bay_y0: int, bay_x1: int, bay_y1: int, side: int, orientation: str) -> bool:
        if orientation == "horizontal":
            if side < 0:
                strip_y0 = bay_y0 - far_wall_buffer_cells
                strip_y1 = bay_y0
            else:
                strip_y0 = bay_y1
                strip_y1 = bay_y1 + far_wall_buffer_cells
            if strip_y0 < 0 or strip_y1 > occupancy.shape[0]:
                return False
            return bool(np.all(occupancy[strip_y0:strip_y1, bay_x0:bay_x1] == 1))

        if side < 0:
            strip_x0 = bay_x0 - far_wall_buffer_cells
            strip_x1 = bay_x0
        else:
            strip_x0 = bay_x1
            strip_x1 = bay_x1 + far_wall_buffer_cells
        if strip_x0 < 0 or strip_x1 > occupancy.shape[1]:
            return False
        return bool(np.all(occupancy[bay_y0:bay_y1, strip_x0:strip_x1] == 1))

    attempts = 0
    pass_count = max(3, int(math.ceil(max_attempts / max(1, len(eligible_segments)))))
    for _ in range(pass_count):
        if len(parking_bays) >= target_count or attempts >= max_attempts:
            break
        segment_order = list(eligible_segments)
        rng.shuffle(segment_order)
        for segment in segment_order:
            if len(parking_bays) >= target_count or attempts >= max_attempts:
                break
            length_cells = _range_sample(rng, config["parking_bay_length_range"])
            depth_cells = _range_sample(rng, config["parking_bay_depth_range"])
            x0, y0, x1, y1 = (int(v) for v in segment.bbox)
            side_order = [-1, 1]
            rng.shuffle(side_order)
            for side in side_order:
                if len(parking_bays) >= target_count or attempts >= max_attempts:
                    break
                attempts += 1
                if segment.orientation == "horizontal":
                    seg_start = min(int(segment.start[0]), int(segment.end[0]))
                    seg_end = max(int(segment.start[0]), int(segment.end[0]))
                    if seg_end - seg_start + 1 < length_cells + 2:
                        continue
                    x_candidates = list(range(seg_start + 1, seg_end - length_cells + 2))
                    rng.shuffle(x_candidates)
                    bay_rect = None
                    for bay_x0 in x_candidates:
                        bay_x1 = bay_x0 + length_cells
                        if side < 0:
                            bay_y1 = y0
                            bay_y0 = bay_y1 - depth_cells
                        else:
                            bay_y0 = y1
                            bay_y1 = bay_y0 + depth_cells
                        if bay_x0 < boundary_margin or bay_y0 < boundary_margin:
                            continue
                        if bay_x1 > int(config["grid_width"]) - boundary_margin or bay_y1 > int(config["grid_height"]) - boundary_margin:
                            continue
                        if np.any(occupancy[bay_y0:bay_y1, bay_x0:bay_x1] == 0):
                            continue
                        if not _keeps_far_side_walled(bay_x0, bay_y0, bay_x1, bay_y1, side, segment.orientation):
                            continue
                        world_x0, world_y0, world_x1, world_y1 = _cell_bounds_world(config, bay_x0, bay_y0, bay_x1, bay_y1)
                        center_xy = (0.5 * (world_x0 + world_x1), 0.5 * (world_y0 + world_y1))
                        if not _is_far_enough(center_xy):
                            continue
                        bay_rect = (bay_x0, bay_y0, bay_x1, bay_y1)
                        break
                else:
                    seg_start = min(int(segment.start[1]), int(segment.end[1]))
                    seg_end = max(int(segment.start[1]), int(segment.end[1]))
                    if seg_end - seg_start + 1 < length_cells + 2:
                        continue
                    y_candidates = list(range(seg_start + 1, seg_end - length_cells + 2))
                    rng.shuffle(y_candidates)
                    bay_rect = None
                    for bay_y0 in y_candidates:
                        bay_y1 = bay_y0 + length_cells
                        if side < 0:
                            bay_x1 = x0
                            bay_x0 = bay_x1 - depth_cells
                        else:
                            bay_x0 = x1
                            bay_x1 = bay_x0 + depth_cells
                        if bay_x0 < boundary_margin or bay_y0 < boundary_margin:
                            continue
                        if bay_x1 > int(config["grid_width"]) - boundary_margin or bay_y1 > int(config["grid_height"]) - boundary_margin:
                            continue
                        if np.any(occupancy[bay_y0:bay_y1, bay_x0:bay_x1] == 0):
                            continue
                        if not _keeps_far_side_walled(bay_x0, bay_y0, bay_x1, bay_y1, side, segment.orientation):
                            continue
                        world_x0, world_y0, world_x1, world_y1 = _cell_bounds_world(config, bay_x0, bay_y0, bay_x1, bay_y1)
                        center_xy = (0.5 * (world_x0 + world_x1), 0.5 * (world_y0 + world_y1))
                        if not _is_far_enough(center_xy):
                            continue
                        bay_rect = (bay_x0, bay_y0, bay_x1, bay_y1)
                        break
                if bay_rect is None:
                    continue
                bay_x0, bay_y0, bay_x1, bay_y1 = bay_rect
                world_x0, world_y0, world_x1, world_y1 = _cell_bounds_world(config, bay_x0, bay_y0, bay_x1, bay_y1)
                center_xy = (0.5 * (world_x0 + world_x1), 0.5 * (world_y0 + world_y1))
                carve_rect(occupancy, bay_x0, bay_y0, bay_x1, bay_y1)
                if segment.orientation == "horizontal":
                    anchor_y = (y0 + y1 - 1) // 2
                    access_cells = [(gx, anchor_y) for gx in range(bay_x0, bay_x1)]
                else:
                    anchor_x = (x0 + x1 - 1) // 2
                    access_cells = [(anchor_x, gy) for gy in range(bay_y0, bay_y1)]

                anchor_cell = access_cells[len(access_cells) // 2]
                access_center = _cell_center_world(config, int(anchor_cell[0]), int(anchor_cell[1]))
                heading = math.atan2(
                    float(center_xy[1]) - float(access_center[1]),
                    float(center_xy[0]) - float(access_center[0]),
                )

                grid_cells = [(gx, gy) for gy in range(bay_y0, bay_y1) for gx in range(bay_x0, bay_x1)]
                parking_bays.append(
                    ParkingBay(
                        center=center_xy,
                        heading=float(_wrap_pi(heading)),
                        length=float(length_cells) * float(config["block_size"]),
                        depth=float(depth_cells) * float(config["block_size"]),
                        grid_cells=grid_cells,
                        access_cells=access_cells,
                        polygon=box(world_x0, world_y0, world_x1, world_y1),
                    )
                )
                break

    return parking_bays


def _generate_scene_once(config: dict, difficulty: str, rng: np.random.Generator, attempt_seed: int) -> BlockMixingPlantScene:
    occupancy = np.ones((int(config["grid_height"]), int(config["grid_width"])), dtype=np.uint8)
    carved_segments: list[_CorridorSegment] = []
    hub, _endpoints = _generate_main_corridors(occupancy, config, rng, carved_segments)

    branch_count = _range_sample(rng, config["branch_count_range"])
    loop_count = _range_sample(rng, config["loop_count_range"])
    dead_end_count = _range_sample(rng, config["dead_end_count_range"])

    min_free_ratio = float(config["min_free_ratio"])
    max_free_ratio = float(config["max_free_ratio"])
    pre_bay_target = min(max_free_ratio - 0.03, max(min_free_ratio, 0.5 * (min_free_ratio + max_free_ratio) - 0.02))
    branch_target = max(min_free_ratio * 0.80, pre_bay_target - 0.08)
    loop_target = max(min_free_ratio * 0.90, pre_bay_target - 0.04)
    dead_end_target = pre_bay_target

    _generate_branches_or_deadends(occupancy, config, rng, carved_segments, branch_count, kind="branch", stop_free_ratio=branch_target)
    _generate_loops(occupancy, config, rng, carved_segments, loop_count, stop_free_ratio=loop_target)
    _generate_branches_or_deadends(occupancy, config, rng, carved_segments, dead_end_count, kind="dead_end", stop_free_ratio=dead_end_target)

    # Top up tight scenes before bay carving if they remain too dense.
    extra_corridors = 0
    while _grid_free_ratio(occupancy) < pre_bay_target and extra_corridors < 24:
        _generate_branches_or_deadends(occupancy, config, rng, carved_segments, 1, kind="branch", stop_free_ratio=pre_bay_target)
        if rng.random() < 0.5:
            _generate_loops(occupancy, config, rng, carved_segments, 1, stop_free_ratio=pre_bay_target)
        extra_corridors += 1

    parking_bays = _generate_parking_bays(occupancy, carved_segments, config, rng, difficulty)
    while _grid_free_ratio(occupancy) < float(config["min_free_ratio"]) and len(parking_bays) < int(config["parking_bay_count_range"][1]):
        newly_added = _generate_parking_bays(occupancy, carved_segments, config, rng, difficulty)
        if len(newly_added) == 0:
            break
        parking_bays.extend(newly_added)

    removed_islands, component_ratio = _cleanup_free_islands(occupancy)
    parking_bays = _filter_bays_against_grid(parking_bays, occupancy)
    free_grid = occupancy == 0
    keep_mask, component_size, total_free = _largest_component_mask(free_grid)

    obstacle_polygons = _grid_obstacle_polygons(occupancy, config)
    obstacle_polygons.extend(_outer_world_obstacle_polygons(config))
    static_obstacles = list(obstacle_polygons)
    dynamic_obstacles = []

    corridor_widths = [segment.width for segment in carved_segments]
    guidance_payload = _build_guidance_payload_from_obstacles(
        occupancy,
        config,
        difficulty,
        static_obstacles=static_obstacles,
        dynamic_obstacles=dynamic_obstacles,
    )
    metadata = {
        "scene_type": "block_mixing_plant",
        "difficulty": str(difficulty),
        "seed": None if config.get("seed") is None else int(config["seed"]),
        "attempt_seed": int(attempt_seed),
        "world_bounds": (WORLD_MIN, WORLD_MAX, WORLD_MIN, WORLD_MAX),
        "grid_origin": _grid_origin(config),
        "grid_width": int(config["grid_width"]),
        "grid_height": int(config["grid_height"]),
        "block_size": float(config["block_size"]),
        "hub_cell": tuple(int(v) for v in hub),
        "corridor_segments": [segment.to_dict() for segment in carved_segments],
        "corridor_width_mean": float(np.mean(corridor_widths)) if corridor_widths else 0.0,
        "corridor_width_min": int(min(corridor_widths)) if corridor_widths else 0,
        "corridor_width_max": int(max(corridor_widths)) if corridor_widths else 0,
        "main_corridor_count": int(config["main_corridor_count"]),
        "branch_count": int(branch_count),
        "loop_count": int(loop_count),
        "dead_end_count": int(dead_end_count),
        "extra_corridor_topups": int(extra_corridors),
        "parking_bay_count": int(len(parking_bays)),
        "free_ratio": float(_grid_free_ratio(occupancy)),
        "largest_component_free_cells": int(component_size),
        "total_free_cells": int(total_free),
        "island_cleanup": (int(removed_islands), float(component_ratio)),
        "guidance_occupancy_payload": guidance_payload,
        "guidance_static_obstacle_signature": str(guidance_payload["static_obstacle_signature"]),
        "guidance_dynamic_obstacle_signature": str(guidance_payload["dynamic_obstacle_signature"]),
        "guidance_static_obstacles": static_obstacles,
        "guidance_dynamic_obstacles": dynamic_obstacles,
        "block_rotation_deg_max": float(config["block_rotation_deg_max"]),
        "block_position_jitter_ratio": float(config["block_position_jitter_ratio"]),
    }

    return BlockMixingPlantScene(
        occupancy_grid=occupancy,
        free_grid=free_grid,
        obstacle_polygons=obstacle_polygons,
        parking_bays=parking_bays,
        metadata=metadata,
    )


def get_block_mixing_plant_config(difficulty: str, seed: int | None = None) -> dict:
    key = str(difficulty)
    if key not in BLOCK_MIXING_PLANT_CONFIG:
        raise KeyError(f"Unknown block mixing plant difficulty: {difficulty}")
    config = dict(BLOCK_MIXING_PLANT_CONFIG[key])
    if seed is not None:
        config["seed"] = int(seed)
    return config


def generate_block_mixing_plant_scene(difficulty: str = "Normal", seed: int | None = None) -> BlockMixingPlantScene:
    config = get_block_mixing_plant_config(difficulty, seed=seed)
    base_seed = config.get("seed")
    rng = np.random.default_rng(base_seed)
    reasons: Counter[str] = Counter()
    max_attempts = int(config.get("max_generation_attempts", 50))
    last_reasons: list[str] = []
    for attempt_idx in range(max_attempts):
        attempt_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        scene = _generate_scene_once(config, str(difficulty), np.random.default_rng(attempt_seed), attempt_seed)
        valid, attempt_reasons = _validate_scene_data(scene.occupancy_grid, scene.parking_bays, config, scene.metadata)
        if valid:
            try:
                sample_navigation_case_from_scene(scene, str(difficulty), seed=attempt_seed)
            except RuntimeError as exc:
                attempt_reasons = [f"navigation pair sampling failed: {exc}"]
            else:
                scene.metadata["generation_attempt_index"] = int(attempt_idx)
                scene.metadata["generation_attempts_used"] = int(attempt_idx) + 1
                scene.metadata["generation_retry_reasons"] = dict(reasons)
                return scene
        last_reasons = attempt_reasons
        for reason in attempt_reasons:
            reasons[str(reason)] += 1

    detail = ", ".join(last_reasons) if last_reasons else "unknown failure"
    raise RuntimeError(f"Failed to generate block mixing plant scene for {difficulty}: {detail}")


def _blocking_poly(obstacles: list) -> Polygon:
    geom = unary_union(obstacles).buffer(0)
    if isinstance(geom, Polygon):
        return geom
    if hasattr(geom, "buffer"):
        return geom.buffer(0)
    raise RuntimeError("Unable to build blocking polygon from obstacles")


def _pose_is_collision_free(pose_xyz: list[float], blocking_poly) -> bool:
    state = State([float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2]), 0.0, 0.0])
    try:
        return all(not blocking_poly.intersects(body) for body in state.create_box())
    except Exception:
        return False


def _candidate_pose_from_bay(
    scene: BlockMixingPlantScene,
    bay: ParkingBay,
    bay_index: int,
    blocking_poly,
    reverse: bool,
) -> Optional[_PoseCandidate]:
    if bay.polygon is None:
        return None
    bay_cfg = BLOCK_MIXING_PLANT_CONFIG.get(str(scene.metadata.get("difficulty")), {})
    desired_head_clearance = float(bay_cfg.get("parking_head_wall_clearance", 1.0))
    heading = _wrap_pi(float(bay.heading) + (math.pi if bool(reverse) else 0.0))
    bx0, by0, bx1, by1 = bay.polygon.bounds
    center_x = 0.5 * (float(bx0) + float(bx1))
    center_y = 0.5 * (float(by0) + float(by1))
    axis = (math.cos(heading), math.sin(heading))
    best_pose = None
    best_clearance = -1.0

    def _build_candidate(pose_xyz: list[float]) -> _PoseCandidate:
        if bay.access_cells:
            anchor = bay.access_cells[len(bay.access_cells) // 2]
        else:
            anchor = _world_to_grid_cell(scene.metadata, pose_xyz[0], pose_xyz[1]) or bay.grid_cells[len(bay.grid_cells) // 2]
        return _PoseCandidate(
            pose=list(pose_xyz),
            bay_index=int(bay_index),
            bay_heading=float(bay.heading),
            anchor_cell=(int(anchor[0]), int(anchor[1])),
            reverse=bool(reverse),
        )

    def _maybe_accept_pose(x: float, y: float) -> None:
        nonlocal best_pose, best_clearance
        pose = [float(x), float(y), float(heading)]
        if not _pose_is_collision_free(pose, blocking_poly):
            return
        state = State([float(x), float(y), float(heading), 0.0, 0.0])
        clearance = float(min(float(body.distance(blocking_poly)) for body in state.create_box()))
        if clearance <= best_clearance:
            return
        best_clearance = clearance
        best_pose = pose

    if not bool(reverse):
        bay_coords = np.asarray(bay.polygon.exterior.coords[:-1], dtype=np.float64)
        wall_projection = float(np.max(bay_coords @ np.asarray(axis, dtype=np.float64)))
        center_projection = float(center_x * axis[0] + center_y * axis[1])
        desired_pose_projection = wall_projection - desired_head_clearance - float(FRONT_HANG)
        preferred_offset = desired_pose_projection - center_projection
        for delta in [0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75]:
            offset = preferred_offset + float(delta)
            pose = [
                center_x + axis[0] * offset,
                center_y + axis[1] * offset,
                float(heading),
            ]
            if _pose_is_collision_free(pose, blocking_poly):
                return _build_candidate(pose)

    usable_longitudinal_margin = max(0.0, 0.5 * max(float(bay.length) - float(LENGTH), 0.0) - 0.15)
    offsets = [0.0]
    if usable_longitudinal_margin > 1e-6:
        probe = min(usable_longitudinal_margin, 0.75)
        offsets.extend([probe, -probe])
        if usable_longitudinal_margin > probe + 0.25:
            offsets.extend([usable_longitudinal_margin, -usable_longitudinal_margin])

    for offset in offsets:
        _maybe_accept_pose(
            center_x + axis[0] * float(offset),
            center_y + axis[1] * float(offset),
        )

    # Short nominal bays cannot contain the full articulated vehicle length.
    # Fall back to a pose on the access lane so the scene remains navigable
    # while keeping the requested bay footprint unchanged.
    if best_pose is None and bay.access_cells:
        anchor = bay.access_cells[len(bay.access_cells) // 2]
        lane_center_x, lane_center_y = _cell_center_world(scene.metadata, int(anchor[0]), int(anchor[1]))
        lane_offsets = [0.0, 0.75, -0.75, 1.5, -1.5, 2.25, -2.25]
        for offset in lane_offsets:
            _maybe_accept_pose(
                lane_center_x + axis[0] * float(offset),
                lane_center_y + axis[1] * float(offset),
            )

    if best_pose is None:
        return None
    return _build_candidate(best_pose)


def _difficulty_constraints(level: str):
    cfg = BLOCK_MIXING_PLANT_CONFIG.get(str(level), {})
    dist_range = cfg.get("pair_distance_range", (0.0, None))
    heading_range = cfg.get("pair_heading_diff_range_deg", (0.0, 180.0))
    dmin = float(dist_range[0])
    dmax = None if dist_range[1] is None else float(dist_range[1])
    amin = float(heading_range[0])
    amax = float(heading_range[1])
    return (dmin, dmax), (amin, amax)


def _pair_satisfies_constraints(level: str, start_pose: list[float], dest_pose: list[float]) -> bool:
    (dmin, dmax), (amin, amax) = _difficulty_constraints(str(level))
    dist = float(math.hypot(float(start_pose[0]) - float(dest_pose[0]), float(start_pose[1]) - float(dest_pose[1])))
    heading_diff = float(math.degrees(_abs_angle_diff(float(start_pose[2]), float(dest_pose[2]))))
    if dist < float(dmin) - 1e-6:
        return False
    if dmax is not None and dist > float(dmax) + 1e-6:
        return False
    if heading_diff < float(amin) - 1e-6:
        return False
    if heading_diff > float(amax) + 1e-6:
        return False
    return True


def _obstacle_distance_steps(occupancy: np.ndarray) -> np.ndarray:
    height, width = occupancy.shape
    dist = np.full((height, width), np.inf, dtype=np.float64)
    queue: deque[tuple[int, int]] = deque()
    occ_y, occ_x = np.where(occupancy == 1)
    for y, x in zip(occ_y.tolist(), occ_x.tolist()):
        dist[y, x] = 0.0
        queue.append((x, y))
    while queue:
        cx, cy = queue.popleft()
        base = float(dist[cy, cx])
        for dx, dy in _CARDINALS:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if base + 1.0 >= float(dist[ny, nx]) - 1e-12:
                continue
            dist[ny, nx] = base + 1.0
            queue.append((nx, ny))
    return dist


def _astar_path(
    free_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacle_distance_steps: np.ndarray | None = None,
) -> Optional[list[tuple[int, int]]]:
    height, width = free_grid.shape
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if sx < 0 or sx >= width or sy < 0 or sy >= height:
        return None
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
        return None
    if not bool(free_grid[sy, sx]) or not bool(free_grid[gy, gx]):
        return None

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic((sx, sy), (gx, gy)), 0.0, (sx, sy)))
    parent = {(sx, sy): None}
    g_cost = {(sx, sy): 0.0}

    while open_heap:
        _, g_now, current = heapq.heappop(open_heap)
        if current == (gx, gy):
            path: list[tuple[int, int]] = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        if g_now > g_cost.get(current, 1e18) + 1e-12:
            continue
        for dx, dy, step_cost in _NEIGHBORS_8:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if not bool(free_grid[ny, nx]):
                continue
            nxt = (nx, ny)
            clearance_penalty = 0.0
            if obstacle_distance_steps is not None:
                clearance_steps = max(1.0, float(obstacle_distance_steps[ny, nx]))
                clearance_penalty = 0.35 / clearance_steps
            new_cost = float(g_now) + float(step_cost) + float(clearance_penalty)
            if new_cost + 1e-12 >= g_cost.get(nxt, 1e18):
                continue
            g_cost[nxt] = new_cost
            parent[nxt] = current
            heapq.heappush(open_heap, (new_cost + heuristic(nxt, (gx, gy)), new_cost, nxt))
    return None


def _scene_metrics_for_pair(
    scene: BlockMixingPlantScene,
    level: str,
    blocking_poly,
    free_region,
    start_pose: list[float],
    dest_pose: list[float],
    start_cell: tuple[int, int],
    dest_cell: tuple[int, int],
    obstacle_distance_steps: np.ndarray | None = None,
) -> Optional[dict]:
    path = _astar_path(scene.free_grid, start_cell, dest_cell, obstacle_distance_steps=obstacle_distance_steps)
    if path is None or len(path) < 2:
        return None

    block_size = float(scene.metadata["block_size"])
    path_points_world = np.array([_cell_center_world(scene.metadata, x, y) for x, y in path], dtype=np.float64)
    seg_lengths = np.linalg.norm(path_points_world[1:] - path_points_world[:-1], axis=1)
    path_length = float(np.sum(seg_lengths))
    direct_distance = float(math.hypot(float(start_pose[0]) - float(dest_pose[0]), float(start_pose[1]) - float(dest_pose[1])))
    if direct_distance <= 1e-6:
        return None

    free_boundary = free_region.boundary
    block_size = float(scene.metadata["block_size"])
    endpoint_skip = max(1, int(math.ceil(float(LENGTH) / max(block_size, 1e-6))))
    endpoint_skip = min(endpoint_skip, max(1, (len(path_points_world) - 3) // 2))
    if len(path_points_world) > (2 * endpoint_skip + 1):
        clearance_points = path_points_world[endpoint_skip:-endpoint_skip]
    else:
        clearance_points = path_points_world
    min_clearance = min(float(Point(float(px), float(py)).distance(free_boundary)) for px, py in clearance_points)

    def _pose_box_clearance(pose_xyz: list[float]) -> float:
        state = State([float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2]), 0.0, 0.0])
        return float(min(float(body.distance(blocking_poly)) for body in state.create_box()))

    start_clearance = _pose_box_clearance(start_pose)
    dest_clearance = _pose_box_clearance(dest_pose)
    heading_diff_deg = float(math.degrees(_abs_angle_diff(float(start_pose[2]), float(dest_pose[2]))))
    return {
        "level": str(level),
        "path_length": float(path_length),
        "direct_distance": float(direct_distance),
        "path_ratio": float(path_length / max(direct_distance, 1e-6)),
        "path_min_clearance": float(min_clearance),
        "bottleneck_width": float(2.0 * min_clearance),
        "heading_diff_deg": float(heading_diff_deg),
        "start_endpoint_clearance": float(start_clearance),
        "dest_endpoint_clearance": float(dest_clearance),
        "min_endpoint_clearance": float(min(start_clearance, dest_clearance)),
        "path_cell_count": int(len(path)),
        "path_block_size": float(block_size),
        "path_endpoint_skip": int(endpoint_skip),
        "path_points_world": [tuple(float(v) for v in pt) for pt in path_points_world.tolist()],
    }


def _scene_metrics_pass(level: str, metrics: dict) -> bool:
    if metrics is None:
        return False
    cfg = BLOCK_MIXING_PLANT_CONFIG.get(str(level), {})
    max_path_ratio = float(cfg.get("scene_metric_max_path_ratio", NAVIGATION_PATH_RATIO_LIMIT_BY_LEVEL.get(str(level), 3.0)))
    min_path_clearance = float(cfg.get("scene_metric_min_path_clearance", NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL.get(str(level), 1.2)))
    min_endpoint_clearance = float(cfg.get("scene_metric_min_endpoint_clearance", NAVIGATION_MIN_ENDPOINT_CLEARANCE_BY_LEVEL.get(str(level), 0.7)))
    tight_turn_heading = float(cfg.get("scene_metric_tight_turn_heading_deg", NAVIGATION_TIGHT_TURN_HEADING_DEG_BY_LEVEL.get(str(level), 140.0)))
    tight_turn_endpoint_clearance = float(cfg.get("scene_metric_tight_turn_min_endpoint_clearance", NAVIGATION_TIGHT_TURN_MIN_ENDPOINT_CLEARANCE_BY_LEVEL.get(str(level), 0.9)))
    if float(metrics["path_ratio"]) > max_path_ratio:
        return False
    if float(metrics["path_min_clearance"]) < min_path_clearance:
        return False
    if float(metrics["min_endpoint_clearance"]) < min_endpoint_clearance:
        return False
    if (
        float(metrics["heading_diff_deg"]) >= tight_turn_heading
        and float(metrics["min_endpoint_clearance"]) < tight_turn_endpoint_clearance
    ):
        return False
    return True


def sample_navigation_case_from_scene(
    scene: BlockMixingPlantScene,
    difficulty: str,
    seed: int | None = None,
) -> tuple[list[float], list[float], dict]:
    if len(scene.parking_bays) < 2:
        raise RuntimeError("Block mixing plant scene has fewer than two parking bays")

    rng = np.random.default_rng(seed if seed is not None else scene.metadata.get("attempt_seed"))
    blocking_poly = _blocking_poly(list(scene.obstacle_polygons))
    world_poly = box(WORLD_MIN, WORLD_MIN, WORLD_MAX, WORLD_MAX)
    free_region = world_poly.difference(blocking_poly).buffer(0)
    obstacle_distance_steps = _obstacle_distance_steps(scene.occupancy_grid)

    start_candidates: list[_PoseCandidate] = []
    dest_candidates: list[_PoseCandidate] = []
    dest_fallback_candidates: list[_PoseCandidate] = []
    for bay_index, bay in enumerate(scene.parking_bays):
        canonical = _candidate_pose_from_bay(scene, bay, bay_index, blocking_poly, reverse=False)
        if canonical is not None:
            start_candidates.append(canonical)
            dest_candidates.append(canonical)
        reversed_pose = _candidate_pose_from_bay(scene, bay, bay_index, blocking_poly, reverse=True)
        if reversed_pose is not None:
            start_candidates.append(reversed_pose)
            dest_fallback_candidates.append(reversed_pose)

    if len(dest_candidates) < 2 or len(start_candidates) < 2:
        raise RuntimeError("Unable to sample enough collision-free parking bay poses")

    max_attempts = int(BLOCK_MIXING_PLANT_CONFIG[str(difficulty)].get("max_pose_sample_attempts", 240))

    def _candidate_score(start_candidate: _PoseCandidate, dest_candidate: _PoseCandidate) -> tuple[float, float]:
        distance = float(math.hypot(float(start_candidate.pose[0]) - float(dest_candidate.pose[0]), float(start_candidate.pose[1]) - float(dest_candidate.pose[1])))
        heading_diff = float(math.degrees(_abs_angle_diff(float(start_candidate.pose[2]), float(dest_candidate.pose[2]))))
        return distance, heading_diff

    def _make_nav_meta(start_candidate: _PoseCandidate, dest_candidate: _PoseCandidate, metrics: dict) -> tuple[list[float], list[float], dict]:
        path_points_world = list(metrics.get("path_points_world", []))
        nav_meta = {
            "scene_metrics": metrics,
            "divider_scene_metrics": dict(metrics),
            "start_bay_index": int(start_candidate.bay_index),
            "dest_bay_index": int(dest_candidate.bay_index),
            "start_support_edge": {
                "poly_label": "parking_bay",
                "bay_index": int(start_candidate.bay_index),
                "heading": float(start_candidate.bay_heading),
                "reverse": bool(start_candidate.reverse),
            },
            "dest_support_edge": {
                "poly_label": "parking_bay",
                "bay_index": int(dest_candidate.bay_index),
                "heading": float(dest_candidate.bay_heading),
                "reverse": bool(dest_candidate.reverse),
            },
            "start_divider_wall_count": 1,
            "dest_divider_wall_count": 1,
            "divider_wall_count": int(scene.metadata.get("parking_bay_count", len(scene.parking_bays))),
            "parking_bays": list(scene.parking_bays),
            "free_grid": np.array(scene.free_grid, copy=True),
            "occupancy_grid": np.array(scene.occupancy_grid, copy=True),
            "plaza": None,
            "corridors": list(scene.metadata.get("corridor_segments", [])),
            "free_region_polygon": free_region,
            "blocking_polygon": blocking_poly,
            "guidance_path_points": path_points_world,
        }
        return list(start_candidate.pose), list(dest_candidate.pose), nav_meta

    def _search_pairs(start_pool: list[_PoseCandidate], dest_pool: list[_PoseCandidate]):
        pair_candidates = []
        for start_candidate in start_pool:
            for dest_candidate in dest_pool:
                if int(start_candidate.bay_index) == int(dest_candidate.bay_index):
                    continue
                if not _pair_satisfies_constraints(str(difficulty), start_candidate.pose, dest_candidate.pose):
                    continue
                distance, heading_diff = _candidate_score(start_candidate, dest_candidate)
                pair_candidates.append((distance, heading_diff, start_candidate, dest_candidate))
        pair_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, _, start_candidate, dest_candidate in pair_candidates[:max_attempts]:
            metrics = _scene_metrics_for_pair(
                scene,
                str(difficulty),
                blocking_poly,
                free_region,
                start_candidate.pose,
                dest_candidate.pose,
                start_candidate.anchor_cell,
                dest_candidate.anchor_cell,
                obstacle_distance_steps=obstacle_distance_steps,
            )
            if not _scene_metrics_pass(str(difficulty), metrics):
                continue
            return _make_nav_meta(start_candidate, dest_candidate, metrics)
        return None

    result = _search_pairs(start_candidates, dest_candidates)
    if result is not None:
        return result
    if len(dest_fallback_candidates) > 0:
        result = _search_pairs(start_candidates, dest_candidates + dest_fallback_candidates)
        if result is not None:
            return result

    raise RuntimeError(f"Failed to sample valid navigation pair for block mixing plant scene ({difficulty})")


def render_scene(
    scene: BlockMixingPlantScene,
    show_parking_bays: bool = False,
    start_pose: list[float] | None = None,
    dest_pose: list[float] | None = None,
    start_bay_index: int | None = None,
    dest_bay_index: int | None = None,
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D

    from env.vehicle import State

    figure, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f1f0ea")
    ax.set_xlim(WORLD_MIN, WORLD_MAX)
    ax.set_ylim(WORLD_MIN, WORLD_MAX)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    def _draw_vehicle(pose_xyz: list[float], front_color: str, rear_color: str, label: str | None = None) -> None:
        state = State([float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2]), 0.0, 0.0])
        front_box, rear_box = state.create_box()
        front_patch = MplPolygon(
            np.asarray(front_box.coords[:-1]),
            closed=True,
            facecolor=front_color,
            edgecolor="#1f1f1f",
            linewidth=1.0,
            alpha=0.85,
            zorder=5,
            label=label,
        )
        rear_patch = MplPolygon(
            np.asarray(rear_box.coords[:-1]),
            closed=True,
            facecolor=rear_color,
            edgecolor="#1f1f1f",
            linewidth=1.0,
            alpha=0.85,
            zorder=5,
        )
        ax.add_patch(front_patch)
        ax.add_patch(rear_patch)

    highlight_bays = {}
    if start_bay_index is not None:
        highlight_bays[int(start_bay_index)] = ("#84a59d", "#52796f", "Start bay")
    if dest_bay_index is not None:
        highlight_bays[int(dest_bay_index)] = ("#f6bd60", "#bc6c25", "Goal bay")

    outer_strips = _outer_world_obstacle_polygons(scene.metadata)
    for poly in outer_strips:
        ax.add_patch(MplPolygon(np.asarray(poly.exterior.coords[:-1]), closed=True, facecolor="#b6b3ad", edgecolor="none", zorder=0))

    occ_y, occ_x = np.where(scene.occupancy_grid == 1)
    rng = np.random.default_rng(scene.metadata.get("attempt_seed"))
    block_size = float(scene.metadata["block_size"])
    jitter_ratio = float(scene.metadata.get("block_position_jitter_ratio", 0.0))
    rotation_max_deg = float(scene.metadata.get("block_rotation_deg_max", 0.0))
    origin_x, origin_y = _grid_origin(scene.metadata)
    for y_idx, x_idx in zip(occ_y.tolist(), occ_x.tolist()):
        x0 = origin_x + float(x_idx) * block_size
        y0 = origin_y + float(y_idx) * block_size
        jitter = jitter_ratio * block_size
        dx = float(rng.uniform(-jitter, jitter))
        dy = float(rng.uniform(-jitter, jitter))
        angle = float(rng.uniform(-rotation_max_deg, rotation_max_deg))
        rect = Rectangle((x0 + dx, y0 + dy), block_size, block_size, facecolor="#57544f", edgecolor="#4b4843", linewidth=0.15, zorder=1)
        rect.set_transform(Affine2D().rotate_deg_around(x0 + dx + 0.5 * block_size, y0 + dy + 0.5 * block_size, angle) + ax.transData)
        ax.add_patch(rect)

    if show_parking_bays:
        for bay_index, bay in enumerate(scene.parking_bays):
            if bay.polygon is None:
                continue
            facecolor = "none"
            edgecolor = "#d98565"
            linewidth = 1.0
            label = None
            if bay_index in highlight_bays:
                facecolor, edgecolor, label = highlight_bays[bay_index]
                linewidth = 1.4
            ax.add_patch(
                MplPolygon(
                    np.asarray(bay.polygon.exterior.coords[:-1]),
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=0.35 if facecolor != "none" else 1.0,
                    zorder=3,
                    label=label,
                )
            )

    if start_pose is not None:
        _draw_vehicle(start_pose, front_color="#2a9d8f", rear_color="#1d6f63", label="Start vehicle")
    if dest_pose is not None:
        _draw_vehicle(dest_pose, front_color="#f4a261", rear_color="#c46d28", label="Goal vehicle")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True, framealpha=0.92, fontsize=8)

    figure.tight_layout(pad=0.1)
    if save_path:
        figure.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
        plt.close(figure)
        return
    plt.show()
