import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
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

    def _make_grid_cache_key(self, bounds, nx: int, ny: int):
        xmin, xmax, ymin, ymax = bounds
        return (
            float(xmin),
            float(xmax),
            float(ymin),
            float(ymax),
            float(self.grid_resolution),
            int(nx),
            int(ny),
        )

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
        occ = np.zeros((nx, ny), dtype=np.uint8)
        bounds = (xmin, xmax, ymin, ymax)

        self._ensure_grid_cell_boxes(bounds, nx, ny)
        cell_boxes = self._grid_cell_boxes

        obstacles = getattr(world_map, "obstacles", []) or []
        for obst in obstacles:
            g = _to_polygon(getattr(obst, "shape", obst))
            if g is None:
                continue
            if self.obstacle_inflation > 1e-9:
                g = g.buffer(self.obstacle_inflation)
            if g.is_empty:
                continue

            pg = prep(g)
            gxmin, gymin, gxmax, gymax = g.bounds
            i0 = max(0, int(math.floor((gxmin - xmin) / res)))
            i1 = min(nx - 1, int(math.ceil((gxmax - xmin) / res)))
            j0 = max(0, int(math.floor((gymin - ymin) / res)))
            j1 = min(ny - 1, int(math.ceil((gymax - ymin) / res)))

            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cbox = cell_boxes[i][j]
                    if pg.intersects(cbox):
                        occ[i, j] = 1

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
            self.path_points_world = None
            self.path_s = None
            self.progress_idx = 0
            return False

        occ[start[0], start[1]] = 0
        occ[goal[0], goal[1]] = 0

        cell_path = self._astar(occ, start, goal)
        if cell_path is None or len(cell_path) == 0:
            self.path_points_world = None
            self.path_s = None
            self.progress_idx = 0
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
