import hashlib
import heapq
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely import intersects_xy
from shapely.geometry import LinearRing, Point, Polygon, box
from shapely.prepared import prep


def _to_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, LinearRing):
        return Polygon(geom)
    return geom


class SoftGlobalGuidance:
    """Low-overhead global guidance:
    - Per reset: coarse grid A* path planning
    - Per step: soft directional hint extraction from lookahead waypoint
    """

    def __init__(
        self,
        grid_resolution: float = 1.0,
        obstacle_inflation: float = 1.2,
        map_margin: float = 0.5,
        lookahead_base: float = 6.0,
        lookahead_speed_gain: float = 1.5,
        lookahead_min: float = 3.0,
        lookahead_max: float = 12.0,
        progress_search_window: int = 40,
        min_clearance_m: float = 1.2,
        full_clearance_m: float = 4.0,
        near_obs_dist_m: float = 2.0,
        max_dense_ratio: float = 0.35,
    ) -> None:
        self.grid_resolution = float(grid_resolution)
        self.obstacle_inflation = float(obstacle_inflation)
        self.map_margin = float(map_margin)

        self.lookahead_base = float(lookahead_base)
        self.lookahead_speed_gain = float(lookahead_speed_gain)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_max = float(lookahead_max)

        self.progress_search_window = int(progress_search_window)

        self.min_clearance_m = float(min_clearance_m)
        self.full_clearance_m = float(full_clearance_m)
        self.near_obs_dist_m = float(near_obs_dist_m)
        self.max_dense_ratio = float(max_dense_ratio)

        self.path_points_world = None
        self.path_s = None
        self.progress_idx = 0

        # Grid cache for occupancy construction (same bounds/resolution).
        self._grid_cache_key = None
        self._grid_cell_boxes = None
        self._occupancy_cache: "OrderedDict[Tuple[float, ...], Dict[str, object]]" = OrderedDict()
        self._static_occupancy_cache: "OrderedDict[Tuple[float, ...], np.ndarray]" = OrderedDict()
        self._dynamic_occupancy_cache: "OrderedDict[Tuple[float, ...], np.ndarray]" = OrderedDict()
        self._occupancy_cache_limit = 32
        self._last_occupancy_payload = None
        self._last_occupancy_cache_status = "miss"
        self._last_occupancy_cache_details = {
            "combined": "miss",
            "static": "miss",
            "dynamic": "miss",
            "combined_builder": "unknown",
            "static_builder": "unknown",
            "dynamic_builder": "unknown",
        }

    def clear_path(self) -> None:
        self.path_points_world = None
        self.path_s = None
        self.progress_idx = 0

    def clear_occupancy_cache(self, keep_layer_caches: bool = False) -> None:
        self._occupancy_cache.clear()
        self._last_occupancy_payload = None
        self._last_occupancy_cache_status = "miss"
        self._last_occupancy_cache_details = {
            "combined": "miss",
            "static": "miss",
            "dynamic": "miss",
            "combined_builder": "unknown",
            "static_builder": "unknown",
            "dynamic_builder": "unknown",
        }
        if not keep_layer_caches:
            self._static_occupancy_cache.clear()
            self._dynamic_occupancy_cache.clear()

    def set_precomputed_path(self, path_points: Sequence[Sequence[float]]) -> bool:
        pts = np.asarray(path_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
            self.clear_path()
            return False

        self.path_points_world = np.array(pts, dtype=np.float64, copy=True)
        self.path_s = self._polyline_arc_length(self.path_points_world)
        self.progress_idx = 0
        return True

    def get_last_occupancy_payload(self, copy: bool = True):
        if self._last_occupancy_payload is None:
            return None
        return self._copy_occupancy_payload(self._last_occupancy_payload, copy=copy)

    def get_last_occupancy_cache_status(self) -> str:
        return str(self._last_occupancy_cache_status)

    def get_last_occupancy_cache_details(self):
        return dict(self._last_occupancy_cache_details)

    def _make_grid_cache_key(self, bounds, nx: int, ny: int):
        xmin, xmax, ymin, ymax = bounds
        return (
            float(xmin),
            float(xmax),
            float(ymin),
            float(ymax),
            float(self.grid_resolution),
            float(self.obstacle_inflation),
            int(nx),
            int(ny),
        )

    def _make_layer_cache_key(self, bounds, nx: int, ny: int, layer_name: str, obstacle_signature: str):
        return self._make_grid_cache_key(bounds, nx, ny) + (str(layer_name), str(obstacle_signature))

    def _combine_obstacle_signatures(self, static_signature: str, dynamic_signature: str) -> str:
        return f"{static_signature}|{dynamic_signature}"

    def _make_obstacle_signature(self, obstacles) -> str:
        hasher = hashlib.blake2b(digest_size=16)
        for obst in obstacles:
            geom = _to_polygon(getattr(obst, "shape", obst))
            if geom is None:
                continue
            try:
                hasher.update(geom.wkb)
            except Exception:
                fallback = (
                    getattr(geom, "geom_type", type(geom).__name__),
                    tuple(round(float(v), 4) for v in getattr(geom, "bounds", ())),
                )
                hasher.update(repr(fallback).encode("utf-8"))
        return hasher.hexdigest()

    def _make_occupancy_cache_key(self, bounds, nx: int, ny: int, obstacle_signature: str):
        return self._make_grid_cache_key(bounds, nx, ny) + (str(obstacle_signature),)

    def _copy_occupancy_payload(self, payload, copy: bool = True):
        cloned = dict(payload)
        occ = np.asarray(cloned.get("occ"), dtype=np.uint8)
        cloned["occ"] = np.array(occ, copy=copy)
        static_occ = cloned.get("static_occ")
        dynamic_occ = cloned.get("dynamic_occ")
        if static_occ is not None:
            cloned["static_occ"] = np.array(np.asarray(static_occ, dtype=np.uint8), copy=copy)
        if dynamic_occ is not None:
            cloned["dynamic_occ"] = np.array(np.asarray(dynamic_occ, dtype=np.uint8), copy=copy)
        cloned["bounds"] = tuple(float(v) for v in cloned.get("bounds", ()))
        cloned["cache_key"] = tuple(cloned.get("cache_key", ()))
        cloned["static_cache_key"] = tuple(cloned.get("static_cache_key", ()))
        cloned["dynamic_cache_key"] = tuple(cloned.get("dynamic_cache_key", ()))
        cloned["obstacle_signature"] = str(cloned.get("obstacle_signature", ""))
        cloned["static_obstacle_signature"] = str(cloned.get("static_obstacle_signature", ""))
        cloned["dynamic_obstacle_signature"] = str(cloned.get("dynamic_obstacle_signature", ""))
        return cloned

    def _normalize_occupancy_payload(
        self,
        payload,
        cache_key,
        bounds,
        nx: int,
        ny: int,
        static_cache_key=None,
        dynamic_cache_key=None,
    ):
        if not isinstance(payload, dict):
            return None

        payload_key = tuple(payload.get("cache_key", ()))
        payload_bounds = tuple(float(v) for v in payload.get("bounds", ()))
        if payload_key != tuple(cache_key):
            return None
        if payload_bounds != tuple(float(v) for v in bounds):
            return None

        occ = np.asarray(payload.get("occ"), dtype=np.uint8)
        if occ.shape != (nx, ny):
            return None

        static_occ = payload.get("static_occ")
        if static_occ is None:
            static_occ = np.array(occ, copy=True)
        static_occ = np.asarray(static_occ, dtype=np.uint8)
        if static_occ.shape != (nx, ny):
            return None

        dynamic_occ = payload.get("dynamic_occ")
        if dynamic_occ is None:
            dynamic_occ = np.zeros((nx, ny), dtype=np.uint8)
        dynamic_occ = np.asarray(dynamic_occ, dtype=np.uint8)
        if dynamic_occ.shape != (nx, ny):
            return None

        normalized_static_cache_key = tuple(payload.get("static_cache_key", static_cache_key or ()))
        normalized_dynamic_cache_key = tuple(payload.get("dynamic_cache_key", dynamic_cache_key or ()))
        if static_cache_key is not None and normalized_static_cache_key != tuple(static_cache_key):
            return None
        if dynamic_cache_key is not None and normalized_dynamic_cache_key != tuple(dynamic_cache_key):
            return None

        return {
            "cache_key": tuple(cache_key),
            "bounds": payload_bounds,
            "occ": np.array(occ, copy=True),
            "static_occ": np.array(static_occ, copy=True),
            "dynamic_occ": np.array(dynamic_occ, copy=True),
            "obstacle_signature": str(payload.get("obstacle_signature", "")),
            "static_cache_key": normalized_static_cache_key,
            "dynamic_cache_key": normalized_dynamic_cache_key,
            "static_obstacle_signature": str(payload.get("static_obstacle_signature", "")),
            "dynamic_obstacle_signature": str(payload.get("dynamic_obstacle_signature", "")),
        }

    def _build_occupancy_payload(
        self,
        cache_key,
        bounds,
        occ: np.ndarray,
        obstacle_signature: str,
        static_occ: np.ndarray,
        dynamic_occ: np.ndarray,
        static_cache_key,
        dynamic_cache_key,
        static_signature: str,
        dynamic_signature: str,
    ):
        return {
            "cache_key": tuple(cache_key),
            "bounds": tuple(float(v) for v in bounds),
            "occ": np.array(occ, copy=True),
            "static_occ": np.array(static_occ, copy=True),
            "dynamic_occ": np.array(dynamic_occ, copy=True),
            "obstacle_signature": str(obstacle_signature),
            "static_cache_key": tuple(static_cache_key),
            "dynamic_cache_key": tuple(dynamic_cache_key),
            "static_obstacle_signature": str(static_signature),
            "dynamic_obstacle_signature": str(dynamic_signature),
        }

    def _cache_occupancy_payload(self, payload) -> None:
        cache_key = tuple(payload["cache_key"])
        if cache_key in self._occupancy_cache:
            self._occupancy_cache.pop(cache_key)
        self._occupancy_cache[cache_key] = self._copy_occupancy_payload(payload, copy=True)
        while len(self._occupancy_cache) > int(self._occupancy_cache_limit):
            self._occupancy_cache.popitem(last=False)

    def _cache_layer_occupancy(self, cache_store, cache_key, occ: np.ndarray) -> None:
        cache_key = tuple(cache_key)
        if cache_key in cache_store:
            cache_store.pop(cache_key)
        cache_store[cache_key] = np.array(occ, dtype=np.uint8, copy=True)
        while len(cache_store) > int(self._occupancy_cache_limit):
            cache_store.popitem(last=False)

    def _inflate_obstacle_geometry(self, obstacle):
        geom = _to_polygon(getattr(obstacle, "shape", obstacle))
        if geom is None:
            return None
        if self.obstacle_inflation > 1e-9:
            geom = geom.buffer(self.obstacle_inflation)
        if geom.is_empty:
            return None
        return geom

    def _expand_geometry_for_grid_fill(self, geom):
        half_res = 0.5 * float(self.grid_resolution)
        if half_res <= 1e-9:
            return geom
        half_diag = math.sqrt(2.0) * half_res
        expanded = geom.buffer(half_diag + 1e-9)
        if expanded.is_empty:
            return geom
        return expanded

    def _world_to_grid_bounds(self, geom_bounds, bounds, nx: int, ny: int, padding_cells: int = 0):
        if nx <= 0 or ny <= 0:
            return None

        xmin, _, ymin, _ = bounds
        gxmin, gymin, gxmax, gymax = geom_bounds
        res = float(self.grid_resolution)

        i0 = int(math.floor((gxmin - xmin) / res)) - int(padding_cells)
        i1 = int(math.ceil((gxmax - xmin) / res)) + int(padding_cells)
        j0 = int(math.floor((gymin - ymin) / res)) - int(padding_cells)
        j1 = int(math.ceil((gymax - ymin) / res)) + int(padding_cells)

        i0 = max(0, i0)
        i1 = min(nx - 1, i1)
        j0 = max(0, j0)
        j1 = min(ny - 1, j1)
        if i0 > i1 or j0 > j1:
            return None
        return i0, i1, j0, j1

    def _grid_index_to_world_center(self, indices: np.ndarray, origin: float) -> np.ndarray:
        return float(origin) + np.asarray(indices, dtype=np.float64) * float(self.grid_resolution)

    def _rasterize_obstacle_to_grid(self, occ: np.ndarray, bounds, obstacle) -> None:
        if occ.size == 0:
            return

        inflated = self._inflate_obstacle_geometry(obstacle)
        if inflated is None:
            return

        coverage_geom = self._expand_geometry_for_grid_fill(inflated)
        nx, ny = occ.shape
        grid_bounds = self._world_to_grid_bounds(coverage_geom.bounds, bounds, nx, ny)
        if grid_bounds is None:
            return

        i0, i1, j0, j1 = grid_bounds
        xmin, _, ymin, _ = bounds

        x_idx = np.arange(i0, i1 + 1, dtype=np.int32)
        y_idx = np.arange(j0, j1 + 1, dtype=np.int32)
        x_coords = self._grid_index_to_world_center(x_idx, xmin)
        y_coords = self._grid_index_to_world_center(y_idx, ymin)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="ij")
        mask = np.asarray(intersects_xy(coverage_geom, grid_x, grid_y), dtype=bool)
        if not np.any(mask):
            return

        occ_slice = occ[i0 : i1 + 1, j0 : j1 + 1]
        occ_slice[mask] = 1

    def _build_occupancy_rasterized(self, bounds, nx: int, ny: int, obstacles) -> np.ndarray:
        occ = np.zeros((nx, ny), dtype=np.uint8)
        if occ.size == 0 or len(obstacles) == 0:
            return occ

        for obstacle in obstacles:
            self._rasterize_obstacle_to_grid(occ, bounds, obstacle)
        return occ

    def _build_occupancy_intersects(self, bounds, nx: int, ny: int, obstacles) -> np.ndarray:
        occ = np.zeros((nx, ny), dtype=np.uint8)
        if occ.size == 0 or len(obstacles) == 0:
            return occ

        self._ensure_grid_cell_boxes(bounds, nx, ny)
        cell_boxes = self._grid_cell_boxes

        for obstacle in obstacles:
            geom = self._inflate_obstacle_geometry(obstacle)
            if geom is None:
                continue

            pg = prep(geom)
            grid_bounds = self._world_to_grid_bounds(geom.bounds, bounds, nx, ny)
            if grid_bounds is None:
                continue

            i0, i1, j0, j1 = grid_bounds
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    if pg.intersects(cell_boxes[i][j]):
                        occ[i, j] = 1
        return occ

    def _build_layer_occupancy(self, bounds, nx: int, ny: int, obstacles, cache_store, cache_key):
        if len(obstacles) == 0:
            return np.zeros((nx, ny), dtype=np.uint8), "empty", "empty"

        cached = cache_store.get(tuple(cache_key))
        if cached is not None:
            cache_store.move_to_end(tuple(cache_key))
            return np.array(cached, copy=True), "hit", "cache"

        builder_mode = "raster"
        try:
            occ = self._build_occupancy_rasterized(bounds, nx, ny, obstacles)
        except Exception:
            occ = self._build_occupancy_intersects(bounds, nx, ny, obstacles)
            builder_mode = "intersects"
        self._cache_layer_occupancy(cache_store, cache_key, occ)
        return occ, "miss", builder_mode

    def _split_guidance_obstacles(self, world_map):
        static_obstacles = getattr(world_map, "guidance_static_obstacles", None)
        dynamic_obstacles = getattr(world_map, "guidance_dynamic_obstacles", None)
        if static_obstacles is None and dynamic_obstacles is None:
            return list(getattr(world_map, "obstacles", []) or []), []
        return list(static_obstacles or []), list(dynamic_obstacles or [])

    def _ensure_grid_cell_boxes(self, bounds, nx: int, ny: int):
        key = self._make_grid_cache_key(bounds, nx, ny)
        if self._grid_cache_key == key and self._grid_cell_boxes is not None:
            return

        xmin, _, ymin, _ = bounds
        res = float(self.grid_resolution)

        cell_boxes = [[None for _ in range(ny)] for _ in range(nx)]
        half = 0.5 * res
        for i in range(nx):
            cx = xmin + i * res
            x0 = cx - half
            x1 = cx + half
            for j in range(ny):
                cy = ymin + j * res
                cell_boxes[i][j] = box(x0, cy - half, x1, cy + half)

        self._grid_cache_key = key
        self._grid_cell_boxes = cell_boxes

    def _build_occupancy(self, world_map) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        xmin = float(world_map.xmin) - self.map_margin
        xmax = float(world_map.xmax) + self.map_margin
        ymin = float(world_map.ymin) - self.map_margin
        ymax = float(world_map.ymax) + self.map_margin

        res = self.grid_resolution
        nx = int(math.ceil((xmax - xmin) / res)) + 1
        ny = int(math.ceil((ymax - ymin) / res)) + 1
        bounds = (xmin, xmax, ymin, ymax)
        has_layer_split = hasattr(world_map, "guidance_static_obstacles") or hasattr(world_map, "guidance_dynamic_obstacles")
        static_obstacles, dynamic_obstacles = self._split_guidance_obstacles(world_map)
        raw_payload = getattr(world_map, "guidance_occupancy_payload", None)
        obstacle_signature = getattr(world_map, "guidance_obstacle_signature", None)
        static_signature = getattr(world_map, "guidance_static_obstacle_signature", None)
        dynamic_signature = getattr(world_map, "guidance_dynamic_obstacle_signature", None)
        if (not has_layer_split) and (not obstacle_signature) and isinstance(raw_payload, dict):
            obstacle_signature = raw_payload.get("obstacle_signature")
        if not static_signature:
            static_signature = self._make_obstacle_signature(static_obstacles)
        if not dynamic_signature:
            dynamic_signature = self._make_obstacle_signature(dynamic_obstacles)
        if not obstacle_signature:
            obstacle_signature = self._combine_obstacle_signatures(static_signature, dynamic_signature)
        static_cache_key = self._make_layer_cache_key(bounds, nx, ny, "static", static_signature)
        dynamic_cache_key = self._make_layer_cache_key(bounds, nx, ny, "dynamic", dynamic_signature)
        cache_key = self._make_occupancy_cache_key(bounds, nx, ny, obstacle_signature)

        self._last_occupancy_cache_details = {
            "combined": "miss",
            "static": "miss",
            "dynamic": "miss",
            "combined_builder": "unknown",
            "static_builder": "unknown",
            "dynamic_builder": "unknown",
        }

        payload = self._normalize_occupancy_payload(
            raw_payload,
            cache_key,
            bounds,
            nx,
            ny,
            static_cache_key=static_cache_key if has_layer_split else None,
            dynamic_cache_key=dynamic_cache_key if has_layer_split else None,
        )
        if payload is not None:
            self._last_occupancy_cache_status = "payload_hit"
            self._last_occupancy_cache_details = {
                "combined": "payload_hit",
                "static": "payload",
                "dynamic": "payload",
                "combined_builder": "payload",
                "static_builder": "payload",
                "dynamic_builder": "payload",
            }
            self._last_occupancy_payload = self._copy_occupancy_payload(payload, copy=True)
            self._cache_occupancy_payload(payload)
            self._cache_layer_occupancy(self._static_occupancy_cache, static_cache_key, payload["static_occ"])
            self._cache_layer_occupancy(self._dynamic_occupancy_cache, dynamic_cache_key, payload["dynamic_occ"])
            try:
                setattr(world_map, "guidance_occupancy_payload", self._copy_occupancy_payload(payload, copy=True))
                setattr(world_map, "guidance_obstacle_signature", str(payload.get("obstacle_signature", obstacle_signature)))
                setattr(
                    world_map,
                    "guidance_static_obstacle_signature",
                    str(payload.get("static_obstacle_signature", static_signature)),
                )
                setattr(
                    world_map,
                    "guidance_dynamic_obstacle_signature",
                    str(payload.get("dynamic_obstacle_signature", dynamic_signature)),
                )
            except Exception:
                pass
            return np.array(payload["occ"], copy=True), bounds

        cached = self._occupancy_cache.get(cache_key)
        if cached is not None:
            self._occupancy_cache.move_to_end(cache_key)
            self._last_occupancy_cache_status = "instance_hit"
            self._last_occupancy_cache_details = {
                "combined": "instance_hit",
                "static": "instance",
                "dynamic": "instance",
                "combined_builder": "instance",
                "static_builder": "instance",
                "dynamic_builder": "instance",
            }
            self._last_occupancy_payload = self._copy_occupancy_payload(cached, copy=True)
            return np.array(cached["occ"], copy=True), bounds

        static_occ, static_status, static_builder = self._build_layer_occupancy(
            bounds,
            nx,
            ny,
            static_obstacles,
            self._static_occupancy_cache,
            static_cache_key,
        )
        dynamic_occ, dynamic_status, dynamic_builder = self._build_layer_occupancy(
            bounds,
            nx,
            ny,
            dynamic_obstacles,
            self._dynamic_occupancy_cache,
            dynamic_cache_key,
        )
        occ = np.maximum(static_occ, dynamic_occ)

        if static_status == "miss" and dynamic_status in ("miss", "empty"):
            combined_status = "miss"
        else:
            combined_status = "layered_hit"

        if static_builder == "intersects" or dynamic_builder == "intersects":
            combined_builder = "intersects"
        elif static_builder == "raster" or dynamic_builder == "raster":
            combined_builder = "raster"
        elif static_builder == "cache" or dynamic_builder == "cache":
            combined_builder = "cache"
        else:
            combined_builder = static_builder

        payload = self._build_occupancy_payload(
            cache_key,
            bounds,
            occ,
            obstacle_signature,
            static_occ,
            dynamic_occ,
            static_cache_key,
            dynamic_cache_key,
            static_signature,
            dynamic_signature,
        )
        self._last_occupancy_cache_status = combined_status
        self._last_occupancy_cache_details = {
            "combined": combined_status,
            "static": static_status,
            "dynamic": dynamic_status,
            "combined_builder": combined_builder,
            "static_builder": static_builder,
            "dynamic_builder": dynamic_builder,
        }
        self._last_occupancy_payload = self._copy_occupancy_payload(payload, copy=True)
        self._cache_occupancy_payload(payload)
        try:
            setattr(world_map, "guidance_occupancy_payload", self._copy_occupancy_payload(payload, copy=True))
            setattr(world_map, "guidance_obstacle_signature", str(obstacle_signature))
            setattr(world_map, "guidance_static_obstacle_signature", str(static_signature))
            setattr(world_map, "guidance_dynamic_obstacle_signature", str(dynamic_signature))
        except Exception:
            pass

        return occ, bounds

    def _world_to_cell(self, x: float, y: float, bounds, shape) -> Optional[Tuple[int, int]]:
        xmin, xmax, ymin, ymax = bounds
        nx, ny = shape
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return None
        i = int(round((x - xmin) / self.grid_resolution))
        j = int(round((y - ymin) / self.grid_resolution))
        i = min(max(i, 0), nx - 1)
        j = min(max(j, 0), ny - 1)
        return i, j

    def _cell_to_world(self, i: int, j: int, bounds) -> Tuple[float, float]:
        xmin, _, ymin, _ = bounds
        x = xmin + float(i) * self.grid_resolution
        y = ymin + float(j) * self.grid_resolution
        return x, y

    def _astar(self, occ: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        nx, ny = occ.shape

        def h(a, b):
            return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

        moves = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        open_heap = []
        heapq.heappush(open_heap, (h(start, goal), 0.0, start))
        parent = {start: None}
        g_cost = {start: 0.0}

        while open_heap:
            _, g_now, cur = heapq.heappop(open_heap)
            if cur == goal:
                out = []
                t = cur
                while t is not None:
                    out.append(t)
                    t = parent[t]
                out.reverse()
                return out

            if g_now > g_cost.get(cur, 1e18) + 1e-12:
                continue

            ci, cj = cur
            for di, dj, w in moves:
                ni, nj = ci + di, cj + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                if occ[ni, nj] != 0:
                    continue
                ng = g_now + w
                nxt = (ni, nj)
                if ng + 1e-12 < g_cost.get(nxt, 1e18):
                    g_cost[nxt] = ng
                    parent[nxt] = cur
                    f = ng + h(nxt, goal)
                    heapq.heappush(open_heap, (f, ng, nxt))

        return None

    def _polyline_arc_length(self, pts: np.ndarray) -> np.ndarray:
        if pts is None or len(pts) == 0:
            return np.zeros((0,), dtype=np.float64)
        if len(pts) == 1:
            return np.zeros((1,), dtype=np.float64)
        d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        s = np.zeros((len(pts),), dtype=np.float64)
        s[1:] = np.cumsum(d)
        return s

    def _interp_on_path(self, s_query: float) -> Optional[np.ndarray]:
        if self.path_points_world is None or self.path_s is None or len(self.path_points_world) == 0:
            return None
        s_arr = self.path_s
        pts = self.path_points_world

        if s_query <= s_arr[0]:
            return pts[0].copy()
        if s_query >= s_arr[-1]:
            return pts[-1].copy()

        k = int(np.searchsorted(s_arr, s_query))
        k0 = max(0, k - 1)
        k1 = min(len(s_arr) - 1, k)
        s0, s1 = float(s_arr[k0]), float(s_arr[k1])
        if s1 - s0 < 1e-9:
            return pts[k1].copy()
        t = (s_query - s0) / (s1 - s0)
        return (1.0 - t) * pts[k0] + t * pts[k1]

    def _hint_strength(self, lidar_norm: Optional[np.ndarray], lidar_range: float) -> float:
        if lidar_norm is None or len(lidar_norm) == 0:
            return 1.0
        d = np.clip(np.asarray(lidar_norm, dtype=np.float64).reshape(-1), 0.0, 1.0) * float(lidar_range)
        min_lidar = float(np.min(d))
        dense_ratio = float(np.mean(d < self.near_obs_dist_m))

        clear_span = max(1e-6, self.full_clearance_m - self.min_clearance_m)
        clear_factor = np.clip((min_lidar - self.min_clearance_m) / clear_span, 0.0, 1.0)
        dense_factor = 1.0 - np.clip(dense_ratio / max(1e-6, self.max_dense_ratio), 0.0, 1.0)

        return float(np.clip(clear_factor * dense_factor, 0.0, 1.0))

    def plan_path(self, world_map, start_xy: Sequence[float], goal_xy: Sequence[float]) -> bool:
        occ, bounds = self._build_occupancy(world_map)
        start = self._world_to_cell(float(start_xy[0]), float(start_xy[1]), bounds, occ.shape)
        goal = self._world_to_cell(float(goal_xy[0]), float(goal_xy[1]), bounds, occ.shape)
        if start is None or goal is None:
            self.clear_path()
            return False

        occ[start[0], start[1]] = 0
        occ[goal[0], goal[1]] = 0

        cell_path = self._astar(occ, start, goal)
        if cell_path is None or len(cell_path) == 0:
            self.clear_path()
            return False

        pts = np.array([self._cell_to_world(i, j, bounds) for i, j in cell_path], dtype=np.float64)
        self.path_points_world = pts
        self.path_s = self._polyline_arc_length(pts)
        self.progress_idx = 0
        return True

    def get_soft_hint(
        self,
        state_x: float,
        state_y: float,
        heading: float,
        speed: float,
        lidar_norm: Optional[np.ndarray] = None,
        lidar_range: float = 30.0,
    ) -> np.ndarray:
        """Return 4-dim soft guidance feature:
        [u_x_soft, u_y_soft, lateral_err_soft, hint_strength]
        """
        if self.path_points_world is None or self.path_s is None or len(self.path_points_world) < 2:
            return np.zeros((4,), dtype=np.float64)

        p = np.array([float(state_x), float(state_y)], dtype=np.float64)

        lo = int(max(0, self.progress_idx - 2))
        hi = int(min(len(self.path_points_world), self.progress_idx + self.progress_search_window + 1))
        seg = self.path_points_world[lo:hi]
        if len(seg) == 0:
            seg = self.path_points_world
            lo = 0

        d2 = np.sum((seg - p) ** 2, axis=1)
        local_best = int(np.argmin(d2))
        best_idx = int(lo + local_best)
        self.progress_idx = max(self.progress_idx, best_idx)

        lookahead = self.lookahead_base + self.lookahead_speed_gain * abs(float(speed))
        lookahead = float(np.clip(lookahead, self.lookahead_min, self.lookahead_max))

        s_now = float(self.path_s[self.progress_idx])
        wp = self._interp_on_path(s_now + lookahead)
        if wp is None:
            return np.zeros((4,), dtype=np.float64)

        dx = float(wp[0] - p[0])
        dy = float(wp[1] - p[1])
        c = math.cos(float(heading))
        s = math.sin(float(heading))

        x_e = c * dx + s * dy
        y_e = -s * dx + c * dy

        norm = math.hypot(x_e, y_e)
        if norm < 1e-6:
            return np.zeros((4,), dtype=np.float64)

        ux = x_e / norm
        uy = y_e / norm
        lat_err = float(np.clip(y_e / max(lookahead, 1e-6), -1.0, 1.0))

        hint_strength = self._hint_strength(lidar_norm, lidar_range)
        soft = np.array([ux, uy, lat_err], dtype=np.float64) * hint_strength

        return np.array([soft[0], soft[1], soft[2], hint_strength], dtype=np.float64)
