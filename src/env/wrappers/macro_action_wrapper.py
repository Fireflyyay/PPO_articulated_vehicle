import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import math
import time
from collections import deque
from types import SimpleNamespace
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
try:
    from shapely.prepared import prep
except Exception:  # pragma: no cover
    prep = None

try:
    from primitives.primitive_refinement import PrimitiveTrajectoryRefiner
except Exception:  # pragma: no cover
    PrimitiveTrajectoryRefiner = None

from env.vehicle import Status

class MacroActionWrapper(gym.Wrapper):
    """
    Gym wrapper that converts discrete primitive family IDs into low-level action sequences.
    """
    def __init__(self, env, primitive_lib, H=None, takeover_dist: float = None, takeover_max_len: int = 3, normalize_before_step: bool = True):
        """
        Args:
            env: The base environment.
            primitive_lib: Family-based primitive library. PPO-visible action IDs are
                           family IDs; the wrapper resolves them to concrete variants.
            H (int): Horizon length of primitives (must match lib).
        """
        super().__init__(env)
        self.primitive_lib = primitive_lib
        # NOTE: CarParking.step() expects actions in [-1, 1] and internally scales to
        # physical ranges. Motion primitive libraries are typically stored in physical
        # units. If we pass physical actions directly, they'll be scaled *again*.
        self.normalize_before_step = bool(normalize_before_step)
        if H is None:
            # Prefer library metadata, then fallback to action array shape.
            H = getattr(primitive_lib, 'horizon', None)
            if H is None:
                actions = getattr(primitive_lib, 'actions', None)
                if actions is not None:
                    H = int(actions.shape[1])
        self.H = int(H)
        self.action_space = spaces.Discrete(int(getattr(primitive_lib, 'action_dim', primitive_lib.size)))
        self._family_resolution_cache = {}
        # Terminal takeover settings (motion-primitive planner)
        if takeover_dist is None:
            try:
                from configs import RS_MAX_DIST
                takeover_dist = min(float(RS_MAX_DIST), 4.0)
            except Exception:
                takeover_dist = 4.0
        self.takeover_dist = float(takeover_dist)
        self.takeover_max_len = int(takeover_max_len)

        # Terminal takeover (RHP) runtime state
        self._takeover_active = False
        self._takeover_prev_choice = None
        self._takeover_mode = "auto"
        self._takeover_fail_count = 0
        self._prefix_steps_queue = []  # next primitive(s) prefix steps, aligned with info['path_to_dest']
        self._soft_prefix_family_steps = {}
        self._last_obs = None
        self._last_base_obs = None

        base_obs_shape = getattr(getattr(self.env, 'observation_space', None), 'shape', None)
        self._base_obs_dim = int(np.prod(base_obs_shape)) if base_obs_shape is not None else 0
        try:
            from configs import (
                PRIMITIVE_ALL_INVALID_FALLBACK_ENABLE,
                PRIMITIVE_MODE_ARTICULATION_MARGIN,
                PRIMITIVE_MODE_CLEARANCE_FREE,
                PRIMITIVE_MODE_CLEARANCE_SAFE,
                PRIMITIVE_MODE_HYSTERESIS_STEPS,
                PRIMITIVE_MODE_NAMES,
                PRIMITIVE_MODE_OBS_ENABLE,
                PRIMITIVE_MODE_OBS_INCLUDE_ONE_HOT,
                PRIMITIVE_MODE_OBS_INCLUDE_SCORES,
                PRIMITIVE_MODE_PROGRESS_WINDOW,
                PRIMITIVE_MODE_SELECTOR_HIGH,
                PRIMITIVE_MODE_SELECTOR_LOW,
                PRIMITIVE_MODE_SELECTOR_WEIGHTS,
                PRIMITIVE_MODE_STUCK_STEPS,
                PRIMITIVE_PREFIX_COMPOUND_RATIO,
                PRIMITIVE_PREFIX_MIN_BY_MODE,
                PRIMITIVE_TERMINAL_DIST,
                PRIMITIVE_TERMINAL_HEADING_DEG,
                PRIMITIVE_TERMINAL_OVERLAP,
            )
        except Exception:
            PRIMITIVE_MODE_NAMES = ("normal", "narrow_escape", "terminal")
            PRIMITIVE_MODE_OBS_ENABLE = True
            PRIMITIVE_MODE_OBS_INCLUDE_ONE_HOT = True
            PRIMITIVE_MODE_OBS_INCLUDE_SCORES = True
            PRIMITIVE_MODE_SELECTOR_WEIGHTS = {
                "clearance": 0.30,
                "valid_action": 0.28,
                "occupancy": 0.10,
                "stuck": 0.18,
                "articulation": 0.14,
            }
            PRIMITIVE_MODE_SELECTOR_HIGH = 0.58
            PRIMITIVE_MODE_SELECTOR_LOW = 0.36
            PRIMITIVE_MODE_HYSTERESIS_STEPS = 3
            PRIMITIVE_MODE_PROGRESS_WINDOW = 6
            PRIMITIVE_MODE_STUCK_STEPS = 5
            PRIMITIVE_TERMINAL_DIST = 4.0
            PRIMITIVE_TERMINAL_OVERLAP = 0.58
            PRIMITIVE_TERMINAL_HEADING_DEG = 20.0
            PRIMITIVE_MODE_CLEARANCE_SAFE = 1.2
            PRIMITIVE_MODE_CLEARANCE_FREE = 4.0
            PRIMITIVE_MODE_ARTICULATION_MARGIN = 0.85
            PRIMITIVE_PREFIX_MIN_BY_MODE = {"normal": 1, "narrow_escape": 2, "terminal": 1}
            PRIMITIVE_PREFIX_COMPOUND_RATIO = 0.55
            PRIMITIVE_ALL_INVALID_FALLBACK_ENABLE = True
        self._primitive_mode_names = [str(name) for name in list(PRIMITIVE_MODE_NAMES)]
        self._mode_obs_enable = bool(PRIMITIVE_MODE_OBS_ENABLE)
        self._mode_obs_include_one_hot = bool(PRIMITIVE_MODE_OBS_INCLUDE_ONE_HOT)
        self._mode_obs_include_scores = bool(PRIMITIVE_MODE_OBS_INCLUDE_SCORES)
        self._mode_selector_weights = dict(PRIMITIVE_MODE_SELECTOR_WEIGHTS)
        self._mode_selector_high = float(PRIMITIVE_MODE_SELECTOR_HIGH)
        self._mode_selector_low = float(PRIMITIVE_MODE_SELECTOR_LOW)
        self._mode_hysteresis_steps = max(1, int(PRIMITIVE_MODE_HYSTERESIS_STEPS))
        self._mode_selector_progress_window = max(2, int(PRIMITIVE_MODE_PROGRESS_WINDOW))
        self._mode_selector_stuck_steps = max(2, int(PRIMITIVE_MODE_STUCK_STEPS))
        self._terminal_dist = float(PRIMITIVE_TERMINAL_DIST)
        self._terminal_overlap = float(PRIMITIVE_TERMINAL_OVERLAP)
        self._terminal_heading = float(np.deg2rad(PRIMITIVE_TERMINAL_HEADING_DEG))
        self._mode_clearance_safe = float(PRIMITIVE_MODE_CLEARANCE_SAFE)
        self._mode_clearance_free = float(PRIMITIVE_MODE_CLEARANCE_FREE)
        self._mode_articulation_margin = float(PRIMITIVE_MODE_ARTICULATION_MARGIN)
        self._mode_prefix_min_by_mode = {str(k): max(1, int(v)) for k, v in dict(PRIMITIVE_PREFIX_MIN_BY_MODE).items()}
        self._mode_prefix_compound_ratio = float(PRIMITIVE_PREFIX_COMPOUND_RATIO)
        self._all_invalid_fallback_enabled = bool(PRIMITIVE_ALL_INVALID_FALLBACK_ENABLE)
        self._current_primitive_mode = str(self._primitive_mode_names[0])
        self._pending_mode_debug = None
        self._last_mode_debug = {}
        self._mode_transition_count = 0
        self._mode_hold_steps = 0
        self._progress_history = deque(maxlen=self._mode_selector_progress_window)
        self._last_progress_metrics = None
        self._no_progress_steps = 0
        self._last_family_mask = None
        self._all_invalid_fallback_count = 0
        self._mode_feature_dim = 0
        if self._mode_obs_enable:
            if self._mode_obs_include_one_hot:
                self._mode_feature_dim += len(self._primitive_mode_names)
            if self._mode_obs_include_scores:
                self._mode_feature_dim += 4
        if self._mode_obs_enable and self._base_obs_dim > 0:
            self.observation_shape = (int(self._base_obs_dim + self._mode_feature_dim),)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.observation_shape,
                dtype=np.float64,
            )
        else:
            self.observation_shape = getattr(self.env, 'observation_shape', (self._base_obs_dim,))
            self.observation_space = getattr(
                self.env,
                'observation_space',
                spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float64),
            )

        # Terminal takeover remains available, but this project should not invoke RHP.
        self._takeover_planner = None
        self._takeover_use_rhp = False
        try:
            from configs import (
                TAKEOVER_ENABLE,
                LIDAR_NUM,
                LIDAR_RANGE,
                OCCUPANCY_INFLATION_RADIUS,
            )

            self._takeover_enabled = bool(TAKEOVER_ENABLE)
        except Exception:
            self._takeover_enabled = True
            LIDAR_NUM = None
            LIDAR_RANGE = None
            OCCUPANCY_INFLATION_RADIUS = 1.5
        self._mask_lidar_num = int(LIDAR_NUM) if LIDAR_NUM is not None else 120
        self._mask_lidar_range = float(LIDAR_RANGE) if LIDAR_RANGE is not None else 30.0
        self._mask_occupancy_inflation_radius = float(OCCUPANCY_INFLATION_RADIUS)
        self._mask_grid_index = getattr(self.primitive_lib, 'grid_index', None)
        if self._mask_grid_index is None:
            try:
                from primitives.primitive_index import build_approx_index_from_deltas
                from configs import GRID_RESOLUTION

                self._mask_grid_index = build_approx_index_from_deltas(
                    actions=getattr(self.primitive_lib, 'actions'),
                    deltas=getattr(self.primitive_lib, 'deltas'),
                    grid_resolution=float(GRID_RESOLUTION),
                    x_min=-6.0,
                    x_max=12.0,
                    y_min=-9.0,
                    y_max=9.0,
                    group_prefix_steps=max(1, int(round(self.H * 0.3))),
                )
            except Exception:
                self._mask_grid_index = None
        self._mask_lidar_angles = np.linspace(0.0, 2.0 * math.pi, self._mask_lidar_num, endpoint=False)
        if self._mask_grid_index is not None:
            r = max(0.0, self._mask_occupancy_inflation_radius)
            res = max(1e-6, float(self._mask_grid_index.grid_resolution))
            rad = int(math.ceil(r / res))
            offsets = []
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if (dx * dx + dy * dy) * (res * res) <= r * r + 1e-9:
                        offsets.append((dx, dy))
            self._mask_inflation_offsets = offsets
        else:
            self._mask_inflation_offsets = []

        # Planner pruning (depth-2 search can be expensive). We use primitive library
        # deltas (precomputed under a canonical start) to rank candidates cheaply.
        self._plan_topk1 = 12
        self._plan_topk2 = 8
        self._primitive_deltas = getattr(primitive_lib, 'deltas', None)

        # Cached obstacles for faster masking (rebuilt on reset)
        self._mask_obstacles_prepared = None
        self._mask_obstacles_bounds = None
        # Optional fast-prune before precise mask simulation (can be toggled in configs)
        try:
            from configs import ACTION_MASK_USE_FAST_PRUNE

            self._mask_use_fast_prune = bool(ACTION_MASK_USE_FAST_PRUNE)
        except Exception:
            self._mask_use_fast_prune = True

        try:
            from configs import ACTION_MASK_MODE

            mode = str(ACTION_MASK_MODE).strip().lower()
        except Exception:
            mode = "hybrid"
        if mode in ("soft", "ray_soft"):
            mode = "soft_ray"
        if mode not in ("fast_only", "hybrid", "full", "soft_ray"):
            mode = "hybrid"
        self._action_mask_mode = mode

        try:
            from configs import ACTION_MASK_UPDATE_EVERY_K

            self._action_mask_update_every_k = max(1, int(ACTION_MASK_UPDATE_EVERY_K))
        except Exception:
            self._action_mask_update_every_k = 1

        self._action_mask_cached = None
        self._action_mask_calls_since_update = 0
        self._last_action_mask_debug = {}
        self._ray_safety_index = getattr(self.primitive_lib, "ray_safety_index", None)
        try:
            from configs import (
                SOFT_MASK_EPS,
                SOFT_MASK_GAMMA,
                SOFT_MASK_LOGIT_LAMBDA,
                SOFT_MASK_MIN_ACTION_COUNT,
                SOFT_MASK_SMALL_VALUE,
                SOFT_MASK_TERMINAL_ARTICULATION_SCALE,
                SOFT_MASK_TERMINAL_EPS,
                SOFT_MASK_TERMINAL_GAMMA,
                SOFT_MASK_TERMINAL_HEADING_SCALE,
                SOFT_MASK_TERMINAL_RADIUS,
                SOFT_MASK_TERMINAL_WEIGHT_MAX,
                SOFT_MASK_TERMINAL_WEIGHT_MIN,
            )
        except Exception:
            SOFT_MASK_GAMMA = 1.5
            SOFT_MASK_EPS = 0.01
            SOFT_MASK_SMALL_VALUE = 1e-8
            SOFT_MASK_LOGIT_LAMBDA = 1.0
            SOFT_MASK_TERMINAL_RADIUS = 4.0
            SOFT_MASK_TERMINAL_GAMMA = 0.5
            SOFT_MASK_TERMINAL_EPS = 0.05
            SOFT_MASK_MIN_ACTION_COUNT = 6
            SOFT_MASK_TERMINAL_HEADING_SCALE = math.radians(35.0)
            SOFT_MASK_TERMINAL_ARTICULATION_SCALE = math.radians(35.0)
            SOFT_MASK_TERMINAL_WEIGHT_MIN = 0.60
            SOFT_MASK_TERMINAL_WEIGHT_MAX = 1.25
        self._soft_mask_gamma = float(SOFT_MASK_GAMMA)
        self._soft_mask_eps = float(SOFT_MASK_EPS)
        self._soft_mask_small_value = float(SOFT_MASK_SMALL_VALUE)
        self._soft_mask_logit_lambda = float(SOFT_MASK_LOGIT_LAMBDA)
        self._soft_mask_terminal_radius = float(SOFT_MASK_TERMINAL_RADIUS)
        self._soft_mask_terminal_gamma = float(SOFT_MASK_TERMINAL_GAMMA)
        self._soft_mask_terminal_eps = float(SOFT_MASK_TERMINAL_EPS)
        self._soft_mask_min_action_count = int(SOFT_MASK_MIN_ACTION_COUNT)
        self._soft_mask_terminal_heading_scale = float(SOFT_MASK_TERMINAL_HEADING_SCALE)
        self._soft_mask_terminal_articulation_scale = float(SOFT_MASK_TERMINAL_ARTICULATION_SCALE)
        self._soft_mask_terminal_weight_min = float(SOFT_MASK_TERMINAL_WEIGHT_MIN)
        self._soft_mask_terminal_weight_max = float(SOFT_MASK_TERMINAL_WEIGHT_MAX)
        try:
            from configs import (
                BLOCKED_ACTION_REWARD,
                MAX_CONSECUTIVE_BLOCKED_ACTIONS,
                MIN_SAFE_PREFIX_STEPS,
                PPO_DYNAMIC_PREFIX_CRITICAL_ACTION_COUNT,
                PPO_DYNAMIC_PREFIX_CRITICAL_MIN_LIDAR,
                PPO_DYNAMIC_PREFIX_DENSE_OBS_HIGH,
                PPO_DYNAMIC_PREFIX_DENSE_OBS_LOW,
                PPO_DYNAMIC_PREFIX_ENABLE,
                PPO_DYNAMIC_PREFIX_LOW_ACTION_COUNT,
                PPO_DYNAMIC_PREFIX_MAX_STEPS,
                PPO_DYNAMIC_PREFIX_MIN_STEPS,
                PPO_DYNAMIC_PREFIX_NARROW_MIN_LIDAR,
                PPO_DYNAMIC_PREFIX_NARROW_STEPS,
            )
        except Exception:
            MIN_SAFE_PREFIX_STEPS = 1
            BLOCKED_ACTION_REWARD = -0.1
            MAX_CONSECUTIVE_BLOCKED_ACTIONS = 20
            PPO_DYNAMIC_PREFIX_ENABLE = True
            PPO_DYNAMIC_PREFIX_MIN_STEPS = 1
            PPO_DYNAMIC_PREFIX_NARROW_STEPS = 2
            PPO_DYNAMIC_PREFIX_MAX_STEPS = None
            PPO_DYNAMIC_PREFIX_NARROW_MIN_LIDAR = 2.5
            PPO_DYNAMIC_PREFIX_CRITICAL_MIN_LIDAR = 1.5
            PPO_DYNAMIC_PREFIX_DENSE_OBS_LOW = 0.20
            PPO_DYNAMIC_PREFIX_DENSE_OBS_HIGH = 0.35
            PPO_DYNAMIC_PREFIX_LOW_ACTION_COUNT = 8
            PPO_DYNAMIC_PREFIX_CRITICAL_ACTION_COUNT = 3
        self._ppo_dynamic_prefix_enabled = bool(PPO_DYNAMIC_PREFIX_ENABLE)
        self._ppo_dynamic_prefix_min_steps = max(1, int(PPO_DYNAMIC_PREFIX_MIN_STEPS))
        self._ppo_dynamic_prefix_narrow_steps = max(1, int(PPO_DYNAMIC_PREFIX_NARROW_STEPS))
        self._ppo_dynamic_prefix_max_steps = None if PPO_DYNAMIC_PREFIX_MAX_STEPS is None else max(1, int(PPO_DYNAMIC_PREFIX_MAX_STEPS))
        self._ppo_dynamic_prefix_narrow_min_lidar = float(PPO_DYNAMIC_PREFIX_NARROW_MIN_LIDAR)
        self._ppo_dynamic_prefix_critical_min_lidar = float(PPO_DYNAMIC_PREFIX_CRITICAL_MIN_LIDAR)
        self._ppo_dynamic_prefix_dense_obs_low = float(PPO_DYNAMIC_PREFIX_DENSE_OBS_LOW)
        self._ppo_dynamic_prefix_dense_obs_high = float(PPO_DYNAMIC_PREFIX_DENSE_OBS_HIGH)
        self._ppo_dynamic_prefix_low_action_count = max(1, int(PPO_DYNAMIC_PREFIX_LOW_ACTION_COUNT))
        self._ppo_dynamic_prefix_critical_action_count = max(1, int(PPO_DYNAMIC_PREFIX_CRITICAL_ACTION_COUNT))
        self._min_safe_prefix_steps = max(1, int(MIN_SAFE_PREFIX_STEPS))
        self._blocked_action_reward = float(BLOCKED_ACTION_REWARD)
        self._max_consecutive_blocked_actions = max(1, int(MAX_CONSECUTIVE_BLOCKED_ACTIONS))
        self._primitive_refiner = PrimitiveTrajectoryRefiner() if PrimitiveTrajectoryRefiner is not None else None
        self._planned_action_queue = []
        self._planned_primitive_queue = []
        self._planned_phase_debug_queue = []
        self._planned_cache_source = None
        self._consecutive_blocked_actions = 0

    def _ensure_mask_obstacle_cache(self):
        if self._mask_obstacles_prepared is not None and self._mask_obstacles_bounds is not None:
            return
        try:
            world_map = getattr(self.env, 'map', None)
            obstacles = getattr(world_map, 'obstacles', []) or []
            if prep is not None:
                self._mask_obstacles_prepared = [prep(o.shape) for o in obstacles]
            else:
                self._mask_obstacles_prepared = [o.shape for o in obstacles]
            self._mask_obstacles_bounds = [o.shape.bounds for o in obstacles]
        except Exception:
            self._mask_obstacles_prepared = []
            self._mask_obstacles_bounds = []

    def _build_mask_occupied_cells_from_lidar(self, lidar_norm: np.ndarray):
        grid_index = getattr(self, '_mask_grid_index', None)
        if grid_index is None:
            return set()

        lidar_norm = np.asarray(lidar_norm, dtype=np.float64).reshape(-1)
        if lidar_norm.size > int(self._mask_lidar_num):
            lidar_norm = lidar_norm[: int(self._mask_lidar_num)]

        dist = np.clip(lidar_norm, 0.0, 1.0) * float(self._mask_lidar_range)
        occupied = set()
        hit_mask = dist < (0.98 * float(self._mask_lidar_range))
        for i in np.nonzero(hit_mask)[0]:
            d = float(dist[i])
            a = float(self._mask_lidar_angles[i])
            x = d * math.cos(a)
            y = d * math.sin(a)
            cell = grid_index.world_to_cell(x, y)
            if cell is None:
                continue
            ix, iy = cell
            for dx, dy in self._mask_inflation_offsets:
                occupied.add((ix + dx, iy + dy))
        return occupied

    def _get_mask_candidate_ids(self, obs_vec, n_actions: int):
        """Best-effort fast pruning via takeover planner's grid index.

        Returns:
            np.ndarray[int] candidate primitive ids, or None if unavailable.
        """
        if not bool(getattr(self, '_mask_use_fast_prune', True)):
            return None

        grid_index = getattr(self, '_mask_grid_index', None)
        if grid_index is None:
            return None
        if obs_vec is None:
            return None

        try:
            obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
            lidar_n = int(self._mask_lidar_num)
            lidar = obs_vec[:lidar_n]
            occupied_cells = self._build_mask_occupied_cells_from_lidar(lidar)
            candidate_mask = grid_index.fast_prune_primitives(occupied_cells)
            candidate_mask = np.asarray(candidate_mask, dtype=np.bool_).reshape(-1)

            if candidate_mask.shape[0] != int(n_actions):
                return None

            candidate_ids = np.flatnonzero(candidate_mask)
            return candidate_ids.astype(np.int64)
        except Exception:
            return None

    def _terminal_weights_from_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        n_actions = int(self.action_space.n)
        deltas = getattr(self.primitive_lib, "deltas", None)
        if deltas is None:
            return np.ones((n_actions,), dtype=np.float32)
        deltas = np.asarray(deltas, dtype=np.float64)
        if deltas.shape[0] != n_actions or deltas.shape[1] < 3:
            return np.ones((n_actions,), dtype=np.float32)

        goal = self._parse_goal_repr_from_obs(obs_vec)
        gx = float(goal.get("goal_x", 0.0))
        gy = float(goal.get("goal_y", 0.0))
        gh = float(goal.get("goal_heading", 0.0))
        art = float(goal.get("articulation", 0.0))
        dist_before = max(float(goal.get("dist", 0.0)), 1e-6)
        radius = max(float(self._soft_mask_terminal_radius), 1e-6)
        heading_scale = max(float(self._soft_mask_terminal_heading_scale), 1e-6)
        art_scale = max(float(self._soft_mask_terminal_articulation_scale), 1e-6)

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dtheta = deltas[:, 2]
        dgamma = deltas[:, 3] if deltas.shape[1] > 3 else np.zeros_like(dtheta)
        pos_after = np.sqrt((gx - dx) * (gx - dx) + (gy - dy) * (gy - dy))
        heading_err = np.abs((gh - dtheta + np.pi) % (2.0 * np.pi) - np.pi)
        art_err = np.abs((art - dgamma + np.pi) % (2.0 * np.pi) - np.pi)
        progress = np.clip((dist_before - pos_after) / radius, -1.0, 1.0)
        pos_score = np.exp(-np.square(pos_after / radius))
        heading_score = np.exp(-np.square(heading_err / heading_scale))
        art_score = np.exp(-np.square(art_err / art_scale))
        weights = (
            0.72
            + 0.28 * pos_score
            + 0.22 * heading_score
            + 0.08 * art_score
            + 0.18 * np.maximum(progress, 0.0)
            - 0.10 * np.maximum(-progress, 0.0)
        )
        return np.clip(
            weights,
            float(self._soft_mask_terminal_weight_min),
            float(self._soft_mask_terminal_weight_max),
        ).astype(np.float32)

    def _compute_soft_ray_action_mask(self, obs_vec=None, primitive_mode=None, mode_debug=None, write_debug: bool = True, return_debug: bool = False) -> np.ndarray:
        t0 = time.perf_counter()
        n_actions = int(self.action_space.n)
        eps = float(self._soft_mask_eps)
        index = getattr(self, "_ray_safety_index", None)
        debug = {
            "mode": "soft_ray",
            "ray_safety_available": index is not None,
            "terminal_reweight_applied": False,
            "fallback": None,
            "selected_mode": str(self._mode_name(primitive_mode)),
        }

        base_obs = self._extract_base_obs(obs_vec)
        if base_obs is None or index is None:
            hard_mask, hard_debug = self._compute_hard_action_mask(base_obs, mode_override="full", primitive_mode=primitive_mode, mode_debug=mode_debug, write_debug=False, return_debug=True)
            mask = np.clip(np.asarray(hard_mask, dtype=np.float32), eps, 1.0).astype(np.float32)
            debug.update(hard_debug or {})
            debug["fallback"] = "hard_family_mask"
            self._soft_prefix_family_steps = {}
        else:
            try:
                from configs import LIDAR_NUM, LIDAR_RANGE

                lidar_num = int(LIDAR_NUM)
                lidar_range = float(LIDAR_RANGE)
            except Exception:
                lidar_num = int(getattr(index, "lidar_num", 120))
                lidar_range = float(getattr(index, "lidar_range", 30.0))
            lidar = np.asarray(base_obs, dtype=np.float64).reshape(-1)[:lidar_num]
            min_lidar = float(np.min(lidar)) * lidar_range if lidar.size > 0 else float(lidar_range)
            obs_density = float(np.mean((lidar * lidar_range) < 3.0)) if lidar.size > 0 else 0.0
            variant_mask, mask_debug = index.compute_soft_mask(
                lidar,
                gamma=float(self._soft_mask_gamma),
                eps=eps,
                lidar_range=lidar_range,
            )
            debug.update(mask_debug)
            debug["min_lidar_m"] = float(min_lidar)
            debug["obs_density"] = float(obs_density)
            safe_step_lens = np.asarray(
                mask_debug.get("safe_step_lens", np.zeros((variant_mask.shape[0],), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            mask = np.full((n_actions,), eps, dtype=np.float32)
            self._family_resolution_cache = {}
            self._soft_prefix_family_steps = {}
            selection_context = self._variant_selection_context(mode_debug)
            for family_id in range(n_actions):
                ref = self._resolve_family_ref(family_id, obs_vec=base_obs, primitive_mode=primitive_mode, selection_context=selection_context)
                mask[int(family_id)] = float(variant_mask[int(ref.flat_index)])
                safe_prefix_steps = 0
                if int(ref.flat_index) < safe_step_lens.shape[0]:
                    safe_prefix_steps = int(np.clip(safe_step_lens[int(ref.flat_index)], 0, int(self.H)))
                self._soft_prefix_family_steps[int(family_id)] = int(safe_prefix_steps)
            if len(self._soft_prefix_family_steps) > 0:
                family_safe_prefixes = np.asarray(list(self._soft_prefix_family_steps.values()), dtype=np.int64)
                debug["family_safe_prefix_mean"] = float(np.mean(family_safe_prefixes))
                debug["family_safe_prefix_min"] = int(np.min(family_safe_prefixes))
                debug["family_safe_prefix_max"] = int(np.max(family_safe_prefixes))

        debug["soft_mask_ms"] = float((time.perf_counter() - t0) * 1000.0)
        debug["soft_mask_min"] = float(np.min(mask)) if mask.size else 0.0
        debug["soft_mask_max"] = float(np.max(mask)) if mask.size else 0.0
        debug["soft_mask_mean"] = float(np.mean(mask)) if mask.size else 0.0
        debug["effective_action_count"] = int(np.count_nonzero(mask > max(eps, 0.05)))
        # ---- Diagnostic: per-family mask value distribution (A-class) ----
        debug["family_mask_zero_count"] = int(np.sum(mask <= eps))
        debug["family_mask_low_count"] = int(np.sum((mask > eps) & (mask < 0.10)))
        debug["family_mask_mid_count"] = int(np.sum((mask >= 0.10) & (mask < 0.50)))
        debug["family_mask_high_count"] = int(np.sum(mask >= 0.50))
        # ---- End diagnostic ----
        if write_debug:
            self._last_action_mask_debug = debug
        if return_debug:
            return mask.astype(np.float32), debug
        return mask.astype(np.float32)

    def _physical_to_normalized_action(self, action_phys: np.ndarray) -> np.ndarray:
        """Convert a physical action (steer, speed) into env-expected [-1, 1]."""
        action_phys = np.asarray(action_phys, dtype=np.float64)

        # Prefer the wrapped env's Box bounds when available.
        low = getattr(getattr(self.env, 'action_space', None), 'low', None)
        high = getattr(getattr(self.env, 'action_space', None), 'high', None)

        if low is None or high is None:
            try:
                from configs import VALID_STEER, VALID_SPEED
                low = np.array([VALID_STEER[0], VALID_SPEED[0]], dtype=np.float64)
                high = np.array([VALID_STEER[1], VALID_SPEED[1]], dtype=np.float64)
            except Exception:
                # Last resort: assume already normalized.
                return np.clip(action_phys, -1.0, 1.0)

        low = np.asarray(low, dtype=np.float64).reshape(-1)
        high = np.asarray(high, dtype=np.float64).reshape(-1)
        denom = (high - low)
        denom = np.where(np.abs(denom) < 1e-9, 1.0, denom)
        action_norm = 2.0 * (action_phys - low) / denom - 1.0
        return np.clip(action_norm, -1.0, 1.0)

    def _simulate_mask_primitive_end_state(self, state0, actions, xmin, xmax, ymin, ymax, num_step: int):
        """Fast scalar rollout for action-mask feasibility checks."""
        model = getattr(self.env.vehicle, 'kinetic_model', None)
        if model is None:
            return None, False

        x = float(state0.loc.x)
        y = float(state0.loc.y)
        heading = float(state0.heading)
        rear_heading = float(getattr(state0, 'rear_heading', heading))

        mini_iter = max(1, int(getattr(model, 'mini_iter', 1)))
        step_len = float(getattr(model, 'step_len', 0.0))
        speed_range = getattr(model, 'speed_range', None)
        angle_range = getattr(model, 'angle_range', None)
        if speed_range is None or angle_range is None:
            return None, False

        dt = step_len / float(mini_iter)
        is_articulated = hasattr(model, 'trailer_length') and hasattr(model, 'hitch_offset')

        if is_articulated:
            l1 = float(model.hitch_offset)
            l2 = float(model.trailer_length)
            phi_max = float(np.deg2rad(36.0))
        else:
            wheel_base = float(getattr(model, 'wheel_base', 1.0))

        for action in actions:
            steer, speed_cmd = np.asarray(action, dtype=np.float64).reshape(-1)
            speed = float(np.clip(speed_cmd, *speed_range))
            steering = float(np.clip(steer, *angle_range))

            if is_articulated:
                omega = steering
                for _ in range(int(num_step)):
                    for _ in range(mini_iter):
                        phi = (heading - rear_heading + np.pi) % (2.0 * np.pi) - np.pi
                        effective_omega = omega
                        if phi >= phi_max and omega > 0.0:
                            effective_omega = 0.0
                        elif phi <= -phi_max and omega < 0.0:
                            effective_omega = 0.0

                        denom = l1 * math.cos(phi) + l2
                        if abs(denom) < 1e-6:
                            denom = 1e-6

                        theta1_dot = (speed * math.sin(phi) + l2 * effective_omega) / denom
                        theta2_dot = theta1_dot - effective_omega
                        x += speed * math.cos(heading) * dt
                        y += speed * math.sin(heading) * dt
                        heading += theta1_dot * dt
                        rear_heading += theta2_dot * dt
            else:
                heading_dot = speed * math.tan(steering) / max(wheel_base, 1e-6)
                for _ in range(int(num_step) * mini_iter):
                    x += speed * math.cos(heading) * dt
                    y += speed * math.sin(heading) * dt
                    heading += heading_dot * dt
                rear_heading = heading

            if x < xmin or x > xmax or y < ymin or y > ymax:
                return None, False

        return (x, y, heading, rear_heading), True

    def _create_mask_boxes(self, x: float, y: float, heading: float, rear_heading: float):
        from configs import FrontVehicleBox, RearVehicleBox, HITCH_OFFSET, TRAILER_LENGTH

        cos_theta = math.cos(heading)
        sin_theta = math.sin(heading)
        front_box = affine_transform(
            FrontVehicleBox,
            [cos_theta, -sin_theta, sin_theta, cos_theta, x, y],
        )

        hx = x - float(HITCH_OFFSET) * math.cos(heading)
        hy = y - float(HITCH_OFFSET) * math.sin(heading)
        tx = hx - float(TRAILER_LENGTH) * math.cos(rear_heading)
        ty = hy - float(TRAILER_LENGTH) * math.sin(rear_heading)
        cos_theta_r = math.cos(rear_heading)
        sin_theta_r = math.sin(rear_heading)
        rear_box = affine_transform(
            RearVehicleBox,
            [cos_theta_r, -sin_theta_r, sin_theta_r, cos_theta_r, tx, ty],
        )

        return (front_box, rear_box)

    def _current_vehicle_state(self):
        base_env = self.env
        vehicle = getattr(base_env, 'vehicle', None)
        return getattr(vehicle, 'state', None)

    def _current_articulation(self) -> float:
        state = self._current_vehicle_state()
        if state is None:
            return 0.0
        return self._wrap_pi(float(state.heading) - float(getattr(state, 'rear_heading', state.heading)))

    def _mode_name(self, primitive_mode=None) -> str:
        if primitive_mode is None:
            return str(self._current_primitive_mode)
        if isinstance(primitive_mode, (int, np.integer)):
            idx = int(primitive_mode)
            if 0 <= idx < len(self._primitive_mode_names):
                return str(self._primitive_mode_names[idx])
        mode_name = str(primitive_mode)
        for name in self._primitive_mode_names:
            if str(name).lower() == mode_name.lower():
                return str(name)
        return str(self._primitive_mode_names[0])

    def _mode_id(self, primitive_mode=None) -> int:
        name = self._mode_name(primitive_mode)
        for idx, candidate in enumerate(self._primitive_mode_names):
            if str(candidate) == name:
                return int(idx)
        return 0

    def _extract_base_obs(self, obs_vec):
        if obs_vec is None:
            return None
        arr = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
        base_obs_dim = int(getattr(self, '_base_obs_dim', 0) or 0)
        mode_obs_dim = int(getattr(self, '_mode_obs_dim', 0) or 0)
        if base_obs_dim <= 0 or arr.size <= base_obs_dim:
            return arr
        if mode_obs_dim > 0 and arr.size == (base_obs_dim + mode_obs_dim):
            return arr[:base_obs_dim]
        return arr

    def _mode_one_hot(self, primitive_mode: str) -> np.ndarray:
        one_hot = np.zeros((len(self._primitive_mode_names),), dtype=np.float64)
        one_hot[self._mode_id(primitive_mode)] = 1.0
        return one_hot

    def _goal_alignment_metrics(self):
        state = self._current_vehicle_state()
        base_env = self.env
        world_map = getattr(base_env, 'map', None)
        dest = getattr(world_map, 'dest', None) if world_map is not None else None
        if state is None or dest is None:
            return {
                "dist": 0.0,
                "heading_error": 0.0,
                "front_overlap": 0.0,
                "rear_overlap": 0.0,
                "mean_overlap": 0.0,
            }

        heading_error = abs(self._wrap_pi(float(state.heading) - float(dest.heading)))
        ego_boxes = state.create_box()
        dest_boxes = dest.create_box()
        front_box_ego = Polygon(ego_boxes[0])
        rear_box_ego = Polygon(ego_boxes[1])
        front_box_dest = Polygon(dest_boxes[0])
        rear_box_dest = Polygon(dest_boxes[1])
        front_overlap = float(front_box_ego.intersection(front_box_dest).area) / (float(front_box_dest.area) + 1e-9)
        rear_overlap = float(rear_box_ego.intersection(rear_box_dest).area) / (float(rear_box_dest.area) + 1e-9)
        return {
            "dist": float(state.loc.distance(dest.loc)),
            "heading_error": float(heading_error),
            "front_overlap": float(front_overlap),
            "rear_overlap": float(rear_overlap),
            "mean_overlap": float(0.5 * (front_overlap + rear_overlap)),
        }

    def _lidar_clearance_metrics(self, obs_vec=None):
        base_obs = self._extract_base_obs(obs_vec)
        if base_obs is None:
            return {
                "front_clearance": float(self._mask_lidar_range),
                "rear_clearance": float(self._mask_lidar_range),
                "obs_density": 0.0,
            }

        lidar = np.asarray(base_obs, dtype=np.float64).reshape(-1)[: int(self._mask_lidar_num)]
        if lidar.size == 0:
            return {
                "front_clearance": float(self._mask_lidar_range),
                "rear_clearance": float(self._mask_lidar_range),
                "obs_density": 0.0,
            }

        angles = self._mask_lidar_angles[: lidar.size]
        dist = np.clip(lidar, 0.0, 1.0) * float(self._mask_lidar_range)
        front_mask = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= float(np.deg2rad(35.0))
        rear_angles = (angles - np.pi + np.pi) % (2.0 * np.pi) - np.pi
        rear_mask = np.abs(np.arctan2(np.sin(rear_angles), np.cos(rear_angles))) <= float(np.deg2rad(35.0))
        front_clearance = float(np.min(dist[front_mask])) if np.any(front_mask) else float(np.min(dist))
        rear_clearance = float(np.min(dist[rear_mask])) if np.any(rear_mask) else float(np.min(dist))
        obs_density = float(np.mean(dist < 3.0)) if dist.size else 0.0
        return {
            "front_clearance": float(front_clearance),
            "rear_clearance": float(rear_clearance),
            "obs_density": float(obs_density),
        }

    def _selector_progress_ok(self) -> bool:
        if len(self._progress_history) < max(2, self._mode_selector_progress_window // 2):
            return False
        start = self._progress_history[0]
        end = self._progress_history[-1]
        dist_gain = float(start.get("dist", 0.0) - end.get("dist", 0.0))
        overlap_gain = float(end.get("front_overlap", 0.0) - start.get("front_overlap", 0.0))
        heading_gain = float(start.get("heading_error", 0.0) - end.get("heading_error", 0.0))
        return bool(dist_gain > 0.25 or overlap_gain > 0.02 or heading_gain > float(np.deg2rad(4.0)))

    def _update_progress_tracking(self):
        metrics = self._goal_alignment_metrics()
        self._progress_history.append(dict(metrics))
        if self._last_progress_metrics is None:
            self._last_progress_metrics = dict(metrics)
            self._no_progress_steps = 0
            return metrics

        prev = self._last_progress_metrics
        dist_gain = float(prev.get("dist", 0.0) - metrics.get("dist", 0.0))
        overlap_gain = float(metrics.get("front_overlap", 0.0) - prev.get("front_overlap", 0.0))
        heading_gain = float(prev.get("heading_error", 0.0) - metrics.get("heading_error", 0.0))
        if dist_gain < 0.10 and overlap_gain < 0.01 and heading_gain < float(np.deg2rad(2.0)):
            self._no_progress_steps += 1
        else:
            self._no_progress_steps = 0
        self._last_progress_metrics = dict(metrics)
        return metrics

    def _mode_observation_features(self, mode_debug=None) -> np.ndarray:
        if not self._mode_obs_enable:
            return np.zeros((0,), dtype=np.float64)
        debug = dict(mode_debug or self._last_mode_debug or {})
        feats = []
        if self._mode_obs_include_one_hot:
            feats.append(self._mode_one_hot(debug.get("selected_mode", self._current_primitive_mode)))
        if self._mode_obs_include_scores:
            feats.append(
                np.asarray(
                    [
                        float(debug.get("congestion_score", 0.0)),
                        float(debug.get("valid_action_ratio", 0.0)),
                        float(debug.get("mean_soft_mask", 0.0)),
                        float(debug.get("abs_phi_ratio", 0.0)),
                    ],
                    dtype=np.float64,
                )
            )
        if len(feats) == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(feats, axis=0)

    def _augment_observation(self, obs_vec, mode_debug=None):
        base_obs = self._extract_base_obs(obs_vec)
        if base_obs is None:
            return None
        if not self._mode_obs_enable:
            return np.asarray(base_obs, dtype=np.float64).copy()
        debug = mode_debug if isinstance(mode_debug, dict) else self._estimate_mode_state(base_obs, update_state=False)
        extras = self._mode_observation_features(debug)
        return np.concatenate([np.asarray(base_obs, dtype=np.float64), extras], axis=0)

    def _variant_selection_context(self, mode_debug=None):
        primitive_mode = self._mode_name((mode_debug or {}).get("selected_mode")) if isinstance(mode_debug, dict) else self._mode_name()
        if primitive_mode == "terminal":
            return {
                "progress_bias": 0.10,
                "safety_bias": 0.35,
                "articulation_bias": 0.30,
                "terminal_bias": 0.75,
            }
        if primitive_mode == "narrow_escape":
            return {
                "progress_bias": 0.15,
                "safety_bias": 0.60,
                "articulation_bias": 0.55,
                "terminal_bias": 0.10,
            }
        return {
            "progress_bias": 0.50,
            "safety_bias": 0.25,
            "articulation_bias": 0.15,
            "terminal_bias": 0.05,
        }

    def _estimate_mode_state(self, obs_vec=None, update_state: bool = False):
        base_obs = self._extract_base_obs(obs_vec)
        safety_mask = None
        if self._ray_safety_index is not None and base_obs is not None:
            try:
                lidar = np.asarray(base_obs, dtype=np.float64).reshape(-1)[: int(self._mask_lidar_num)]
                safety_mask, _ = self._ray_safety_index.compute_soft_mask(
                    lidar,
                    gamma=float(self._soft_mask_gamma),
                    eps=float(self._soft_mask_eps),
                    lidar_range=float(self._mask_lidar_range),
                )
                safety_mask = np.asarray(safety_mask, dtype=np.float32).reshape(-1)
            except Exception:
                safety_mask = None
        if safety_mask is None:
            n_actions = int(self.action_space.n)
            safety_mask = np.ones((n_actions,), dtype=np.float32)

        goal_metrics = self._goal_alignment_metrics()
        clearance_metrics = self._lidar_clearance_metrics(base_obs)
        valid_action_ratio = float(np.mean(safety_mask > max(float(self._soft_mask_eps), 0.05))) if safety_mask.size else 0.0
        mean_soft_mask = float(np.mean(safety_mask)) if safety_mask.size else 0.0
        phi_abs = abs(self._current_articulation())
        phi_max = float(max(1e-6, np.max(np.abs(getattr(self.primitive_lib, 'gamma_bin_values', np.asarray([np.deg2rad(36.0)]))))))
        abs_phi_ratio = float(np.clip(phi_abs / phi_max, 0.0, 1.5))
        min_clearance = float(min(clearance_metrics["front_clearance"], clearance_metrics["rear_clearance"]))
        clearance_score = 1.0 - float(np.clip((min_clearance - self._mode_clearance_safe) / max(1e-6, self._mode_clearance_free - self._mode_clearance_safe), 0.0, 1.0))
        valid_score = 1.0 - float(np.clip(valid_action_ratio, 0.0, 1.0))
        occupancy_score = float(np.clip(clearance_metrics["obs_density"], 0.0, 1.0))
        stuck_score = float(np.clip(self._no_progress_steps / max(1.0, float(self._mode_selector_stuck_steps)), 0.0, 1.0))
        phi_score = float(np.clip(abs_phi_ratio / max(self._mode_articulation_margin, 1e-6), 0.0, 1.25))
        congestion_score = float(np.clip(
            self._mode_selector_weights.get("clearance", 0.0) * clearance_score
            + self._mode_selector_weights.get("valid_action", 0.0) * valid_score
            + self._mode_selector_weights.get("occupancy", 0.0) * occupancy_score
            + self._mode_selector_weights.get("stuck", 0.0) * stuck_score
            + self._mode_selector_weights.get("articulation", 0.0) * phi_score,
            0.0,
            1.5,
        ))
        near_goal = bool(
            goal_metrics["dist"] <= self._terminal_dist
            or goal_metrics["front_overlap"] >= self._terminal_overlap
            or (
                goal_metrics["heading_error"] <= self._terminal_heading
                and goal_metrics["front_overlap"] >= max(0.45, self._terminal_overlap - 0.15)
            )
        )
        prev_mode = str(self._current_primitive_mode)
        selected_mode = prev_mode
        if prev_mode == "terminal":
            if near_goal or goal_metrics["dist"] <= (self._terminal_dist + 1.0):
                selected_mode = "terminal"
            elif congestion_score > self._mode_selector_high:
                selected_mode = "narrow_escape"
            else:
                selected_mode = "normal"
        elif near_goal:
            selected_mode = "terminal"
        elif stuck_score >= 1.0 or congestion_score > self._mode_selector_high:
            selected_mode = "narrow_escape"
        elif prev_mode == "narrow_escape":
            if congestion_score < self._mode_selector_low and self._selector_progress_ok():
                selected_mode = "normal"
            else:
                selected_mode = "narrow_escape"
        else:
            selected_mode = "normal"

        mode_transitioned = bool(selected_mode != prev_mode)
        transition_count = int(self._mode_transition_count)
        if update_state:
            if mode_transitioned:
                self._mode_transition_count += 1
                self._mode_hold_steps = 0
            else:
                self._mode_hold_steps += 1
            self._current_primitive_mode = str(selected_mode)
            transition_count = int(self._mode_transition_count)

        return {
            "selected_mode": str(selected_mode),
            "selected_mode_id": int(self._mode_id(selected_mode)),
            "previous_mode": str(prev_mode),
            "mode_transitioned": bool(mode_transitioned),
            "mode_transition_count": int(transition_count),
            "congestion_score": float(congestion_score),
            "valid_action_ratio": float(valid_action_ratio),
            "mean_soft_mask": float(mean_soft_mask),
            "abs_phi_ratio": float(np.clip(abs_phi_ratio, 0.0, 1.5)),
            "stuck_score": float(stuck_score),
            "front_clearance": float(clearance_metrics["front_clearance"]),
            "rear_clearance": float(clearance_metrics["rear_clearance"]),
            "goal_dist": float(goal_metrics["dist"]),
            "goal_heading_error": float(goal_metrics["heading_error"]),
            "front_overlap": float(goal_metrics["front_overlap"]),
            "rear_overlap": float(goal_metrics["rear_overlap"]),
            "mean_overlap": float(goal_metrics["mean_overlap"]),
            "terminal_triggered": bool(near_goal),
            # ---- Diagnostic: hypothetical approach / extended-terminal conditions (A-class) ----
            "diag_would_trigger_approach": bool(
                float(goal_metrics["dist"]) <= 10.0
                and float(goal_metrics["dist"]) > 4.0
                and float(goal_metrics["front_overlap"]) < 0.58
            ),
            "diag_would_trigger_terminal_extended": bool(float(goal_metrics["dist"]) <= 8.0),
            # ---- End diagnostic ----
        }

    def _goal_repr_from_env_state(self, obs_vec=None) -> dict:
        if obs_vec is not None:
            try:
                return self._parse_goal_repr_from_obs(self._extract_base_obs(obs_vec))
            except Exception:
                pass

        state = self._current_vehicle_state()
        base_env = self.env
        world_map = getattr(base_env, 'map', None)
        dest = getattr(world_map, 'dest', None) if world_map is not None else None
        if state is None or dest is None:
            return {
                "goal_x": 0.0,
                "goal_y": 0.0,
                "goal_heading": 0.0,
                "articulation": self._current_articulation(),
                "dist": 0.0,
                "rel_angle": 0.0,
            }

        dx = float(dest.loc.x) - float(state.loc.x)
        dy = float(dest.loc.y) - float(state.loc.y)
        c = math.cos(float(state.heading))
        s = math.sin(float(state.heading))
        goal_x = c * dx + s * dy
        goal_y = -s * dx + c * dy
        goal_heading = self._wrap_pi(float(dest.heading) - float(state.heading))
        articulation = self._current_articulation()
        rel_angle = math.atan2(goal_y, goal_x)
        return {
            "goal_x": float(goal_x),
            "goal_y": float(goal_y),
            "goal_heading": float(goal_heading),
            "articulation": float(articulation),
            "dist": float(math.hypot(goal_x, goal_y)),
            "rel_angle": float(rel_angle),
        }

    def _legacy_flat_variant_ref(self, primitive_id: int):
        return SimpleNamespace(
            flat_index=int(primitive_id),
            gamma_bin_id=0,
            family_id=int(primitive_id),
            variant_id=0,
        )

    def _resolved_variant_debug(self, resolved_ref) -> dict:
        if hasattr(self.primitive_lib, 'resolved_variant_debug'):
            return self.primitive_lib.resolved_variant_debug(resolved_ref)

        primitive_id = int(getattr(resolved_ref, 'flat_index', -1))
        actions = np.asarray(self.primitive_lib.get_actions(primitive_id), dtype=np.float64)
        return {
            "flat_index": int(primitive_id),
            "gamma_bin_id": int(getattr(resolved_ref, 'gamma_bin_id', 0)),
            "gamma_bin_value": float(self._current_articulation()),
            "family_id": int(getattr(resolved_ref, 'family_id', primitive_id)),
            "family_name": f"primitive-{primitive_id}",
            "family_type": "legacy_flat",
            "variant_id": int(getattr(resolved_ref, 'variant_id', 0)),
            "mode": "legacy_flat",
            "speed_sign": int(np.sign(np.mean(actions[:, 1]))) if actions.size else 0,
            "duration": float(actions.shape[0]),
            "effective_horizon": int(actions.shape[0]),
            "is_compound": False,
            "switch_index": -1,
        }

    def _resolve_family_ref(self, family_id: int, obs_vec=None, primitive_mode=None, selection_context=None):
        cache_key = (int(family_id), str(self._mode_name(primitive_mode)))
        if cache_key in self._family_resolution_cache:
            return self._family_resolution_cache[cache_key]
        if not hasattr(self.primitive_lib, 'resolve_family_variant'):
            ref = self._legacy_flat_variant_ref(family_id)
            self._family_resolution_cache[cache_key] = ref
            return ref

        goal_repr = self._goal_repr_from_env_state(obs_vec=obs_vec)
        gamma = self._current_articulation()
        ref = self.primitive_lib.resolve_family_variant(
            int(family_id),
            gamma=float(gamma),
            primitive_mode=self._mode_name(primitive_mode),
            goal_repr=goal_repr,
            selection_context=selection_context,
        )
        self._family_resolution_cache[cache_key] = ref
        return ref

    def _current_obs_snapshot(self):
        if self._last_base_obs is not None:
            try:
                return np.asarray(self._last_base_obs).copy()
            except Exception:
                return copy.deepcopy(self._last_base_obs)
        try:
            obs = self.env._build_observation()
        except Exception:
            obs = None
        if obs is None:
            return None
        try:
            return np.asarray(obs).copy()
        except Exception:
            return copy.deepcopy(obs)

    def _mode_fallback_candidates(self, primary_mode: str):
        primary_mode = self._mode_name(primary_mode)
        unique_modes = []
        for mode in self._primitive_mode_names:
            name = self._mode_name(mode)
            if name not in unique_modes:
                unique_modes.append(name)
        if primary_mode not in unique_modes:
            unique_modes.insert(0, primary_mode)

        def _mode_score(mode_name: str) -> float:
            n = str(mode_name).lower()
            score = 0.0
            if any(k in n for k in ("fine", "small", "short", "micro")):
                score += 3.0
            if any(k in n for k in ("narrow", "escape")):
                score += 2.8
            if any(k in n for k in ("terminal", "align")):
                score += 2.2
            if any(k in n for k in ("recover", "recovery", "stuck", "articulation", "straighten")):
                score += 2.6
            if "normal" in n or "coarse" in n:
                score -= 0.4
            return float(score)

        tail = [m for m in unique_modes if m != primary_mode]
        tail = sorted(tail, key=lambda m: _mode_score(m), reverse=True)
        return [primary_mode] + tail

    def _compute_effective_safe_prefix_steps(self, safe_prefix_steps: int, resolved_debug=None, primitive_mode=None):
        primitive_mode = self._mode_name((resolved_debug or {}).get('mode', primitive_mode or self._last_mode_debug.get('selected_mode', self._current_primitive_mode)))
        prefix_steps = max(0, min(int(self.H), int(safe_prefix_steps)))
        source = None

        if self._ppo_dynamic_prefix_max_steps is not None:
            prefix_steps = min(prefix_steps, int(self._ppo_dynamic_prefix_max_steps))

        min_mode_prefix = int(self._mode_prefix_min_by_mode.get(primitive_mode, 1))
        if bool((resolved_debug or {}).get('is_compound', False)):
            compound_min = int(max(min_mode_prefix, round(float((resolved_debug or {}).get('effective_horizon', self.H)) * self._mode_prefix_compound_ratio)))
            if int((resolved_debug or {}).get('switch_index', -1)) > 0:
                compound_min = max(compound_min, int((resolved_debug or {}).get('switch_index', -1)))
            min_mode_prefix = compound_min

        if prefix_steps > 0 and self._ppo_dynamic_prefix_enabled:
            debug = dict(getattr(self, '_last_action_mask_debug', {}) or {})
            min_lidar = float(debug.get('min_lidar_m', np.inf))
            obs_density = float(debug.get('obs_density', 0.0))
            feasible_count = int(debug.get('effective_action_count', debug.get('family_feasible_count', int(self.action_space.n))))

            # ---- Diagnostic: log context when prefix gets reduced ----
            _was_positive = prefix_steps > 0
            _old_prefix = prefix_steps
            # ---- End diagnostic ----

            if (
                min_lidar <= self._ppo_dynamic_prefix_critical_min_lidar
                or feasible_count <= self._ppo_dynamic_prefix_critical_action_count
                or obs_density >= self._ppo_dynamic_prefix_dense_obs_high
            ):
                prefix_steps = min(prefix_steps, int(self._ppo_dynamic_prefix_min_steps))
                source = 'soft_ray_auto_critical'
            elif (
                min_lidar <= self._ppo_dynamic_prefix_narrow_min_lidar
                or feasible_count <= self._ppo_dynamic_prefix_low_action_count
                or obs_density >= self._ppo_dynamic_prefix_dense_obs_low
            ):
                prefix_steps = min(prefix_steps, int(self._ppo_dynamic_prefix_narrow_steps))
                source = 'soft_ray_auto_narrow'

            # ---- Diagnostic: record if prefix was reduced to zero by dynamic logic ----
            if _was_positive and prefix_steps <= 0:
                _diag = {
                    "diag_prefix_reduced_to_zero": 1,
                    "diag_prefix_old": _old_prefix,
                    "diag_prefix_min_lidar": min_lidar,
                    "diag_prefix_obs_density": obs_density,
                    "diag_prefix_feasible_count": feasible_count,
                    "diag_prefix_min_mode": min_mode_prefix,
                    "diag_prefix_source": str(source),
                    "diag_prefix_primitive_mode": str(primitive_mode),
                }
                self._last_diag_prefix_zero = _diag
            # ---- End diagnostic ----

        if prefix_steps > 0 and prefix_steps < int(min_mode_prefix):
            prefix_steps = 0
            source = 'below_mode_min_prefix'

        if prefix_steps <= 0:
            return 0, 'soft_ray_auto_blocked' if source is None else str(source)
        if prefix_steps < int(self.H):
            return int(prefix_steps), 'soft_ray_auto' if source is None else str(source)
        return int(prefix_steps), source

    def _compute_soft_auto_prefix(self, family_id: int, resolved_debug=None):
        safe_prefix_steps = self._soft_prefix_family_steps.get(int(family_id), None)
        if safe_prefix_steps is None:
            safe_prefix_steps = int(self.H)
        prefix_steps, source = self._compute_effective_safe_prefix_steps(
            safe_prefix_steps,
            resolved_debug=resolved_debug,
        )
        return int(prefix_steps), source

    def _family_safe_steps_for_mode(self, family_id: int, base_obs, primitive_mode: str, mode_debug=None):
        selection_context = self._variant_selection_context(mode_debug)
        ref = self._resolve_family_ref(
            int(family_id),
            obs_vec=base_obs,
            primitive_mode=primitive_mode,
            selection_context=selection_context,
        )
        resolved_debug = self._resolved_variant_debug(ref)

        safe_prefix_steps = self._soft_prefix_family_steps.get(int(family_id), None)
        if safe_prefix_steps is None:
            if self._action_mask_mode == 'soft_ray':
                safe_prefix_steps = int(self.H)
            else:
                safe_prefix_steps = int(self.H) if self._variant_rollout_is_safe(int(ref.flat_index)) else 0

        effective_steps, prefix_source = self._compute_effective_safe_prefix_steps(
            safe_prefix_steps,
            resolved_debug=resolved_debug,
            primitive_mode=primitive_mode,
        )
        return int(effective_steps), prefix_source, ref, resolved_debug

    def _build_blocked_transition(self, last_obs, mode_debug: dict, reason: str, selected_info: dict):
        self._consecutive_blocked_actions += 1
        is_stuck = bool(self._consecutive_blocked_actions >= self._max_consecutive_blocked_actions)
        status = Status.STUCK if is_stuck else Status.BLOCKED_ACTION

        # ---- Diagnostic: lidar snapshot at blocked transition ----
        _diag_lidar_blocked = None
        if last_obs is not None:
            try:
                _obs_arr = np.asarray(last_obs, dtype=np.float64).reshape(-1)
                _lidar_num = min(120, int(_obs_arr.shape[0]))
                _lidar_m = np.asarray(_obs_arr[:_lidar_num], dtype=np.float64) * float(getattr(self, '_lidar_range', 30.0))
                _diag_lidar_blocked = {
                    "diag_blocked_lidar_min": float(np.min(_lidar_m)) if _lidar_m.size else -1.0,
                    "diag_blocked_lidar_mean": float(np.mean(_lidar_m)) if _lidar_m.size else -1.0,
                    "diag_blocked_lidar_lt1m": float(np.mean(_lidar_m < 1.0)) if _lidar_m.size else -1.0,
                    "diag_blocked_lidar_lt3m": float(np.mean(_lidar_m < 3.0)) if _lidar_m.size else -1.0,
                }
            except Exception:
                _diag_lidar_blocked = None
        # ---- End diagnostic ----

        # ---- Diagnostic: attach zero-prefix context from previous computation ----
        _diag_prefix_zero = getattr(self, '_last_diag_prefix_zero', None)
        # ---- End diagnostic ----

        # ---- Diagnostic: stuck snapshot (A-class, no behaviour change) ----
        _diag_stuck_snapshot = {}
        try:
            _goal = self._goal_alignment_metrics()
            _clear = self._lidar_clearance_metrics(last_obs)
            _diag_stuck_snapshot = {
                "diag_stuck_goal_dist": float(_goal.get("dist", -1.0)),
                "diag_stuck_phi_abs": float(abs(self._current_articulation())),
                "diag_stuck_valid_actions": int((self._last_action_mask_debug or {}).get("effective_action_count", -1)),
                "diag_stuck_mode": str(self._current_primitive_mode),
                "diag_stuck_front_clearance": float(_clear.get("front_clearance", -1.0)),
                "diag_stuck_rear_clearance": float(_clear.get("rear_clearance", -1.0)),
                "diag_stuck_front_overlap": float(_goal.get("front_overlap", -1.0)),
                "diag_stuck_heading_error_deg": float(np.rad2deg(_goal.get("heading_error", -1.0))),
                "diag_stuck_consecutive_blocked": int(self._consecutive_blocked_actions),
            }
        except Exception:
            pass
        # ---- End diagnostic ----

        info = {
            'status': status,
            'blocked_action': True,
            'blocked_reason': str(reason),
            'soft_ray_blocked': True,
            'terminal_metrics_valid': False,
            'planning_zero_prefix_blocked': 1.0,
            'planning_consecutive_blocked_actions': int(self._consecutive_blocked_actions),
            'planning_all_modes_invalid': float(bool((self._last_action_mask_debug or {}).get('all_modes_invalid', False))),
            'planning_fallback_to_finer_mode': float(bool((self._last_action_mask_debug or {}).get('fallback_to_finer_mode', False))),
            'planning_effective_primitive_mode': str(mode_debug.get('selected_mode', self._current_primitive_mode)),
            'planning_selected_action_safe_steps': int(selected_info.get('selected_action_safe_steps', 0)),
            'planning_min_safe_steps': int((self._last_action_mask_debug or {}).get('min_safe_steps', 0)),
            'planning_max_safe_steps': int((self._last_action_mask_debug or {}).get('max_safe_steps', 0)),
            'planning_mean_safe_steps': float((self._last_action_mask_debug or {}).get('mean_safe_steps', 0.0)),
            'mask_valid_action_count_after_fallback': int((self._last_action_mask_debug or {}).get('valid_action_count_after_fallback', 0)),
            'mask_valid_action_ratio_after_fallback': float((self._last_action_mask_debug or {}).get('valid_action_ratio_after_fallback', 0.0)),
        }
        if _diag_lidar_blocked is not None:
            info.update(_diag_lidar_blocked)
        if _diag_prefix_zero is not None:
            info.update(_diag_prefix_zero)
        if _diag_stuck_snapshot:
            info.update(_diag_stuck_snapshot)
        info.update(selected_info)

        returned_obs = self._augment_observation(last_obs, mode_debug=mode_debug)
        self._last_base_obs = None if last_obs is None else copy.deepcopy(last_obs)
        self._last_obs = None if returned_obs is None else copy.deepcopy(returned_obs)

        reward = float(self._blocked_action_reward)
        terminated = False
        truncated = bool(is_stuck)
        return returned_obs, reward, terminated, truncated, info

    def _mode_preference_weights(self, primitive_mode: str) -> np.ndarray:
        weights = np.ones((int(self.action_space.n),), dtype=np.float32)
        primitive_mode = self._mode_name(primitive_mode)
        for family_id in range(int(self.action_space.n)):
            family_name = str(getattr(self.primitive_lib, 'family_name', lambda fid: f"family-{fid}")(family_id))
            family_type = str(getattr(self.primitive_lib, 'family_type', lambda fid: 'normal')(family_id))
            weight = 1.0
            is_escape = family_name.startswith('escape-')
            is_terminal = family_name.startswith('terminal-') or family_type == 'terminal'
            is_recovery = family_type == 'straighten' or family_name.startswith('articulation-') or 'phi' in family_name or 'jackknife' in family_name
            is_reverse = family_name.startswith('reverse') or '-reverse-' in family_name or family_name.startswith('escape-reverse') or family_name.endswith('-reverse')
            is_large = 'large' in family_name or 'tight' in family_name
            if primitive_mode == 'normal':
                if is_escape:
                    weight *= 0.58
                if is_terminal:
                    weight *= 0.72
                if is_recovery:
                    weight *= 0.86
                if is_reverse:
                    weight *= 0.90
            elif primitive_mode == 'narrow_escape':
                if is_escape:
                    weight *= 1.28
                if is_recovery:
                    weight *= 1.18
                if is_reverse:
                    weight *= 1.10
                if is_large and family_name.startswith('forward'):
                    weight *= 0.82
                if is_terminal:
                    weight *= 0.80
            elif primitive_mode == 'terminal':
                if is_terminal:
                    weight = 1.80
                elif is_recovery:
                    weight = 0.45
                elif is_escape:
                    weight = 0.18
                elif is_reverse:
                    weight = 0.15
                else:
                    weight = 0.08
                if is_large:
                    weight *= 0.50
            weights[family_id] = float(np.clip(weight, 0.05, 2.00))
        return weights

    def _articulation_preference_weights(self, primitive_mode: str, obs_vec=None) -> np.ndarray:
        phi = self._current_articulation()
        phi_abs = abs(phi)
        phi_max = float(max(1e-6, np.max(np.abs(getattr(self.primitive_lib, 'gamma_bin_values', np.asarray([np.deg2rad(36.0)]))))))
        ratio = float(np.clip(phi_abs / phi_max, 0.0, 1.5))
        weights = np.ones((int(self.action_space.n),), dtype=np.float32)
        if ratio < float(self._mode_articulation_margin):
            return weights

        severity = float(np.clip((ratio - self._mode_articulation_margin) / max(1e-6, 1.0 - self._mode_articulation_margin), 0.0, 1.0))
        selection_context = self._variant_selection_context({"selected_mode": primitive_mode})
        for family_id in range(int(self.action_space.n)):
            ref = self._resolve_family_ref(family_id, obs_vec=obs_vec, primitive_mode=primitive_mode, selection_context=selection_context)
            delta = np.asarray(self.primitive_lib.get_delta(int(ref.flat_index)), dtype=np.float64).reshape(-1)
            delta_gamma = float(delta[3]) if delta.size > 3 else 0.0
            family_name = str(getattr(self.primitive_lib, 'family_name', lambda fid: f"family-{fid}")(family_id))
            family_type = str(getattr(self.primitive_lib, 'family_type', lambda fid: 'normal')(family_id))
            weight = 1.0
            if phi > 0.0 and delta_gamma > 0.04:
                weight *= max(0.12, 1.0 - 1.05 * severity)
            elif phi < 0.0 and delta_gamma < -0.04:
                weight *= max(0.12, 1.0 - 1.05 * severity)
            else:
                if abs(delta_gamma) > 0.02:
                    weight *= 1.0 + 0.32 * severity
            if family_type == 'straighten' or family_name.startswith('articulation-') or 'phi' in family_name or 'jackknife' in family_name:
                weight *= 1.0 + 0.42 * severity
            weights[family_id] = float(np.clip(weight, 0.08, 1.65))
        return weights

    def _choose_all_invalid_fallback(self, primitive_mode: str, safety_mask: np.ndarray, mode_weights: np.ndarray, articulation_weights: np.ndarray, obs_vec=None):
        best_family = 0
        best_score = None
        selection_context = self._variant_selection_context({"selected_mode": primitive_mode})
        for family_id in range(int(self.action_space.n)):
            ref = self._resolve_family_ref(family_id, obs_vec=obs_vec, primitive_mode=primitive_mode, selection_context=selection_context)
            family_name = str(getattr(self.primitive_lib, 'family_name', lambda fid: f"family-{fid}")(family_id))
            family_type = str(getattr(self.primitive_lib, 'family_type', lambda fid: 'normal')(family_id))
            variant_horizons = np.asarray(getattr(self.primitive_lib, 'variant_horizons', np.full((int(self.action_space.n),), int(self.H), dtype=np.int64)), dtype=np.int64).reshape(-1)
            horizon = int(variant_horizons[int(ref.flat_index)]) if int(ref.flat_index) < variant_horizons.size else int(self.H)
            bonus = 0.0
            if family_type == 'straighten' or family_name.startswith('articulation-'):
                bonus += 0.55
            if family_name in ('forward-straight', 'reverse-straight'):
                bonus += 0.25
            if primitive_mode == 'terminal' and family_name.startswith('terminal-'):
                bonus += 0.35
            score = float(safety_mask[family_id]) * 1.4 + float(mode_weights[family_id]) * 0.4 + float(articulation_weights[family_id]) * 0.4 + bonus - 0.03 * float(horizon)
            if best_score is None or score > best_score:
                best_score = score
                best_family = int(family_id)
        return int(best_family)

    def _canonical_state_to_world(self, state0, canonical_state: np.ndarray):
        from env.vehicle import State

        row = np.asarray(canonical_state, dtype=np.float64).reshape(-1)
        x_c, y_c, heading_c, rear_heading_c = map(float, row[:4])
        speed = float(row[4]) if row.size > 4 else 0.0
        steering = float(row[5]) if row.size > 5 else 0.0

        c = math.cos(float(state0.heading))
        s = math.sin(float(state0.heading))
        x_w = float(state0.loc.x) + c * x_c - s * y_c
        y_w = float(state0.loc.y) + s * x_c + c * y_c
        heading_w = self._wrap_pi(float(state0.heading) + heading_c)
        rear_heading_w = self._wrap_pi(float(state0.heading) + rear_heading_c)
        return State([x_w, y_w, heading_w, speed, steering, rear_heading_w])

    def _variant_rollout_is_safe(self, flat_index: int) -> bool:
        state0 = self._current_vehicle_state()
        base_env = self.env
        world_map = getattr(base_env, 'map', None)
        if state0 is None or world_map is None:
            return True

        self._ensure_mask_obstacle_cache()
        prepared = self._mask_obstacles_prepared if self._mask_obstacles_prepared is not None else []
        obst_bounds = self._mask_obstacles_bounds if self._mask_obstacles_bounds is not None else []
        xmin, xmax = float(world_map.xmin), float(world_map.xmax)
        ymin, ymax = float(world_map.ymin), float(world_map.ymax)

        def bounds_overlap(a, b):
            return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

        rollout_states = self.primitive_lib.get_rollout_states(int(flat_index))
        for canonical_state in rollout_states[1:]:
            world_state = self._canonical_state_to_world(state0, canonical_state)
            if world_state.loc.x < xmin or world_state.loc.x > xmax or world_state.loc.y < ymin or world_state.loc.y > ymax:
                return False
            boxes = world_state.create_box()
            for box in boxes:
                bb = box.bounds
                for pg, ob in zip(prepared, obst_bounds):
                    if not bounds_overlap(bb, ob):
                        continue
                    try:
                        hit = pg.intersects(box)
                    except Exception:
                        hit = box.intersects(pg)
                    if hit:
                        return False
        return True

    def step(self, family_id):
        """
        Execute the variant resolved from a PPO-selected primitive family.
        accumulate reward, check done at each step.
        """
        if isinstance(family_id, np.ndarray):
            family_id = family_id.item()

        family_id = int(family_id)
        base_obs_for_mode = self._current_obs_snapshot()
        mode_debug = dict(self._pending_mode_debug or self._estimate_mode_state(base_obs_for_mode, update_state=True))
        self._last_mode_debug = dict(mode_debug)
        self._pending_mode_debug = None
        mask_debug = dict(self._last_action_mask_debug or {})
        primitive_mode = str(mask_debug.get('effective_primitive_mode', mode_debug.get('selected_mode', self._current_primitive_mode)))
        mode_debug['selected_mode'] = str(primitive_mode)
        self._current_primitive_mode = str(primitive_mode)
        resolved_ref = self._resolve_family_ref(
            family_id,
            obs_vec=base_obs_for_mode,
            primitive_mode=primitive_mode,
            selection_context=self._variant_selection_context(mode_debug),
        )
        resolved_debug = self._resolved_variant_debug(resolved_ref)
        primitive_id = int(resolved_ref.flat_index)
        actions = self.primitive_lib.get_actions(primitive_id)
        cached_plan_debug = None
        cached_plan = self._pop_planned_actions(primitive_id)
        if cached_plan is not None:
            actions, cached_plan_debug = cached_plan
        actions = np.asarray(actions, dtype=np.float64)

        total_reward = 0.0
        done = False
        info = {}
        steps_executed = 0
        last_obs = self._current_obs_snapshot()

        # Optional execution trace for mining (low-level actions & states).
        # Stored in `info['macro_exec_trace']` as best-effort; callers can ignore.
        macro_exec_trace = {
            "sub_actions_phys": [],   # list[np.ndarray(2,)] in primitive physical units
            "sub_actions_norm": [],   # list[np.ndarray(2,)] normalized to env step range
            "sub_states": [],         # list[dict] vehicle state snapshots (best-effort)
            "status_seq": [],         # list[str]
        }

        # Record initial state snapshot if available.
        try:
            base_env = self.env
            if hasattr(base_env, 'vehicle') and getattr(base_env.vehicle, 'state', None) is not None:
                st = base_env.vehicle.state
                macro_exec_trace["sub_states"].append({
                    "x": float(st.loc.x),
                    "y": float(st.loc.y),
                    "heading": float(st.heading),
                    "rear_heading": float(getattr(st, 'rear_heading', st.heading)),
                    "speed": float(getattr(st, 'speed', 0.0)),
                    "steering": float(getattr(st, 'steering', 0.0)),
                })
        except Exception:
            pass

        # We need to handle potential 'truncated' from gymnasium if base env uses it.
        terminated = False
        truncated = False

        selected_info = {
            'family_id': int(family_id),
            'resolved_family_id': int(getattr(resolved_ref, 'family_id', family_id)),
            'primitive_id': int(primitive_id),
            'resolved_variant_id': int(resolved_ref.variant_id),
            'resolved_gamma_bin_id': int(resolved_ref.gamma_bin_id),
            'resolved_mode': str(resolved_debug.get('mode', primitive_mode)),
            'selected_mode': str(primitive_mode),
            'mode_transitioned': bool(mode_debug.get('mode_transitioned', False)),
            'mode_transition_count': int(mode_debug.get('mode_transition_count', self._mode_transition_count)),
            'congestion_score': float(mode_debug.get('congestion_score', 0.0)),
            'valid_action_ratio': float(mode_debug.get('valid_action_ratio', 0.0)),
            'mean_soft_mask': float(mode_debug.get('mean_soft_mask', 0.0)),
            'abs_phi_ratio': float(mode_debug.get('abs_phi_ratio', 0.0)),
            'front_clearance_m': float(mode_debug.get('front_clearance', 0.0)),
            'rear_clearance_m': float(mode_debug.get('rear_clearance', 0.0)),
            'goal_heading_error': float(mode_debug.get('goal_heading_error', 0.0)),
            'front_overlap_ratio': float(mode_debug.get('front_overlap', 0.0)),
            'rear_overlap_ratio': float(mode_debug.get('rear_overlap', 0.0)),
            'all_invalid_fallback_count': int(self._all_invalid_fallback_count),
            'resolved_is_compound': bool(resolved_debug.get('is_compound', False)),
            'resolved_switch_index': int(resolved_debug.get('switch_index', -1)),
            'resolved_family_name': str(resolved_debug.get('family_name', '')),
            'resolved_family_type': str(resolved_debug.get('family_type', 'normal')),
            'planning_effective_primitive_mode': str(primitive_mode),
            'planning_fallback_to_finer_mode': float(bool(mask_debug.get('fallback_to_finer_mode', False))),
            'planning_all_modes_invalid': float(bool(mask_debug.get('all_modes_invalid', False))),
            'planning_min_safe_steps': int(mask_debug.get('min_safe_steps', 0)),
            'planning_max_safe_steps': int(mask_debug.get('max_safe_steps', 0)),
            'planning_mean_safe_steps': float(mask_debug.get('mean_safe_steps', 0.0)),
            'mask_valid_action_count_after_fallback': int(mask_debug.get('valid_action_count_after_fallback', 0)),
            'mask_valid_action_ratio_after_fallback': float(mask_debug.get('valid_action_ratio_after_fallback', 0.0)),
            'terminal_metrics_valid': False,
            # ---- Diagnostic: per-step selection details (A-class, no behaviour change) ----
            'diag_gamma': float(self._current_articulation()),
            'diag_flat_index': int(primitive_id),
            'diag_variant_horizon': int(getattr(self.primitive_lib, 'variant_horizons', [self.H])[primitive_id] if primitive_id < len(getattr(self.primitive_lib, 'variant_horizons', [])) else self.H),
            'diag_step_seconds': float(getattr(self.primitive_lib, 'step_seconds', 0.2)),
        }
        try:
            selected_info['diag_gamma_bin'] = int(self.primitive_lib.gamma_to_bin(selected_info['diag_gamma']))
        except Exception:
            selected_info['diag_gamma_bin'] = -1
        if self._last_family_mask is not None and int(family_id) < len(self._last_family_mask):
            selected_info['selected_action_mask_value'] = float(self._last_family_mask[int(family_id)])

        if bool(mask_debug.get('all_modes_invalid', False)):
            selected_info['selected_action_safe_steps'] = 0
            selected_info['prefix_steps_used'] = 0
            selected_info['soft_safe_prefix_steps'] = 0
            selected_info['executed_steps'] = 0
            selected_info['macro_exec_trace'] = macro_exec_trace
            selected_info['takeover_active'] = False
            selected_info['takeover_triggered'] = False
            selected_info['path_to_dest'] = None
            selected_info['planning_zero_prefix_blocked'] = 1.0
            return self._build_blocked_transition(
                last_obs,
                mode_debug=mode_debug,
                reason='all_modes_invalid',
                selected_info=selected_info,
            )

        max_steps = min(self.H, int(actions.shape[0]))
        used_prefix_steps = None
        used_prefix_source = None
        if len(self._prefix_steps_queue) > 0:
            try:
                used_prefix_steps = self._prefix_steps_queue.pop(0)
                if used_prefix_steps is not None:
                    max_steps = min(max_steps, int(used_prefix_steps))
                    used_prefix_source = 'planner'
            except Exception:
                used_prefix_steps = None
                used_prefix_source = None
        elif self._action_mask_mode == 'soft_ray':
            auto_prefix_steps, auto_prefix_source = self._compute_soft_auto_prefix(family_id, resolved_debug=resolved_debug)
            if auto_prefix_steps is not None:
                used_prefix_steps = int(auto_prefix_steps)
                used_prefix_source = str(auto_prefix_source) if auto_prefix_source is not None else 'safe_prefix'
                max_steps = min(max_steps, int(auto_prefix_steps))

        if max_steps <= 0:
            selected_info['selected_action_safe_steps'] = int(used_prefix_steps) if used_prefix_steps is not None else 0
            selected_info['prefix_steps_used'] = int(used_prefix_steps) if used_prefix_steps is not None else 0
            if used_prefix_source is not None:
                selected_info['prefix_steps_source'] = str(used_prefix_source)
            selected_info['soft_safe_prefix_steps'] = int(self._soft_prefix_family_steps.get(int(family_id), 0))
            selected_info['executed_steps'] = 0
            selected_info['macro_exec_trace'] = macro_exec_trace
            selected_info['takeover_active'] = False
            selected_info['takeover_triggered'] = False
            selected_info['path_to_dest'] = None
            selected_info['planning_zero_prefix_blocked'] = 1.0
            return self._build_blocked_transition(
                last_obs,
                mode_debug=mode_debug,
                reason=str(used_prefix_source or 'zero_prefix_blocked'),
                selected_info=selected_info,
            )

        for t in range(max_steps):
            # Execute one low-level step
            action_phys = np.asarray(actions[t], dtype=np.float64).reshape(-1)
            action_norm = action_phys
            if self.normalize_before_step:
                action_norm = self._physical_to_normalized_action(action_phys)

            # trace: actions
            try:
                macro_exec_trace["sub_actions_phys"].append(action_phys.copy())
                macro_exec_trace["sub_actions_norm"].append(np.asarray(action_norm, dtype=np.float64).reshape(-1).copy())
            except Exception:
                pass

            step_result = self.env.step(action_norm)

            # Check return signature
            if len(step_result) == 5:
                obs, reward, terminated, truncated, step_info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                obs, reward, done, step_info = step_result
                terminated = done
                truncated = False # Assume not truncated if 4 args
            else:
                raise ValueError(f"Unexpected step result length: {len(step_result)}")

            total_reward += reward
            steps_executed += 1
            last_obs = obs

            # Merge info
            info.update(step_info)

            # trace: status and state
            try:
                st = getattr(getattr(self.env, 'vehicle', None), 'state', None)
                if st is not None:
                    macro_exec_trace["sub_states"].append({
                        "x": float(st.loc.x),
                        "y": float(st.loc.y),
                        "heading": float(st.heading),
                        "rear_heading": float(getattr(st, 'rear_heading', st.heading)),
                        "speed": float(getattr(st, 'speed', 0.0)),
                        "steering": float(getattr(st, 'steering', 0.0)),
                    })
            except Exception:
                pass
            try:
                st_status = step_info.get('status', None)
                macro_exec_trace["status_seq"].append(str(st_status))
            except Exception:
                pass

            if done:
                break

        if steps_executed > 0:
            self._consecutive_blocked_actions = 0

        info.update(selected_info)
        info['planning_selected_action_safe_steps'] = int(max_steps)
        info['planning_consecutive_blocked_actions'] = int(self._consecutive_blocked_actions)
        info['planning_zero_prefix_blocked'] = 0.0
        info['terminal_metrics_valid'] = float(bool(done))
        info['blocked_action'] = False
        info['executed_steps'] = steps_executed
        if cached_plan_debug is not None:
            info['refinement_plan_step'] = True
            info['refinement_plan_debug'] = cached_plan_debug
        if used_prefix_steps is not None:
            info['prefix_steps_used'] = int(used_prefix_steps)
        if used_prefix_source is not None:
            info['prefix_steps_source'] = str(used_prefix_source)
        info['soft_safe_prefix_steps'] = int(self._soft_prefix_family_steps.get(int(family_id), self.H)) if len(self._soft_prefix_family_steps) > 0 else int(self.H)

        # attach trace (convert lists of arrays to stacked arrays when possible)
        try:
            if len(macro_exec_trace.get("sub_actions_phys", [])) > 0 and isinstance(macro_exec_trace["sub_actions_phys"][0], np.ndarray):
                macro_exec_trace["sub_actions_phys"] = np.stack(macro_exec_trace["sub_actions_phys"], axis=0)
            if len(macro_exec_trace.get("sub_actions_norm", [])) > 0 and isinstance(macro_exec_trace["sub_actions_norm"][0], np.ndarray):
                macro_exec_trace["sub_actions_norm"] = np.stack(macro_exec_trace["sub_actions_norm"], axis=0)
        except Exception:
            pass
        info['macro_exec_trace'] = macro_exec_trace

        info['takeover_active'] = False
        info['takeover_triggered'] = False
        info['path_to_dest'] = None
        progress_metrics = self._update_progress_tracking()
        info['terminal_mean_overlap'] = float(progress_metrics.get('mean_overlap', 0.0))
        returned_obs = self._augment_observation(last_obs, mode_debug=mode_debug)
        self._last_base_obs = None if last_obs is None else copy.deepcopy(last_obs)
        self._last_obs = None if returned_obs is None else copy.deepcopy(returned_obs)

        # Return consistent with Gymnasium
        return returned_obs, total_reward, terminated, truncated, info

    def _parse_goal_repr_from_obs(self, obs_vec: np.ndarray) -> dict:
        """Decode goal representation in ego frame from CarParking observation vector."""
        obs_vec = np.asarray(self._extract_base_obs(obs_vec), dtype=np.float64).reshape(-1)
        try:
            from configs import LIDAR_NUM, MAX_DIST_TO_DEST

            lidar_n = int(LIDAR_NUM)
            max_dist = float(MAX_DIST_TO_DEST)
        except Exception:
            lidar_n = 120
            max_dist = 70.0

        # target_obs layout (CarParking.step):
        # [dist_norm, cos(rel_angle), sin(rel_angle), cos(rel_heading), sin(rel_heading), cos(art), sin(art)]
        target = obs_vec[lidar_n : lidar_n + 7]
        if target.size < 7:
            return {
                "dist": float(max_dist),
                "rel_angle": 0.0,
                "rel_heading": 0.0,
                "articulation": 0.0,
            }
        dist = float(target[0]) * max_dist
        rel_angle = math.atan2(float(target[2]), float(target[1]))
        rel_heading = math.atan2(float(target[4]), float(target[3]))
        articulation = math.atan2(float(target[6]), float(target[5]))

        return {
            "goal_x": dist * math.cos(rel_angle),
            "goal_y": dist * math.sin(rel_angle),
                "goal_heading": self._wrap_pi(rel_heading),
                "articulation": self._wrap_pi(articulation),
            "dist": dist,
            "rel_angle": rel_angle,
        }

    def _takeover_enabled_now(self) -> bool:
        try:
            from configs import TAKEOVER_ENABLE

            self._takeover_enabled = bool(TAKEOVER_ENABLE)
        except Exception:
            pass
        return bool(getattr(self, '_takeover_enabled', True))

    def _primitive_refinement_enabled_now(self) -> bool:
        try:
            from configs import USE_PRIMITIVE_REFINEMENT

            enabled = bool(USE_PRIMITIVE_REFINEMENT)
        except Exception:
            enabled = False
        return bool(enabled and self._primitive_refiner is not None)

    def _clear_planned_action_cache(self):
        self._planned_action_queue.clear()
        self._planned_primitive_queue.clear()
        self._planned_phase_debug_queue.clear()
        self._planned_cache_source = None

    def _clear_takeover_plan_cache_if_owned(self):
        if str(getattr(self, '_planned_cache_source', '')) == 'takeover':
            self._clear_planned_action_cache()

    def _pop_planned_actions(self, primitive_id: int):
        if len(self._planned_action_queue) == 0:
            return None
        if len(self._planned_primitive_queue) == 0:
            self._clear_planned_action_cache()
            return None

        expected_pid = int(self._planned_primitive_queue[0])
        if int(primitive_id) != expected_pid:
            self._clear_planned_action_cache()
            return None

        self._planned_primitive_queue.pop(0)
        actions = np.asarray(self._planned_action_queue.pop(0), dtype=np.float64)
        debug = self._planned_phase_debug_queue.pop(0) if len(self._planned_phase_debug_queue) > 0 else None
        return actions, debug

    def prepare_plan_execution(self, primitive_ids, prefix_steps: int = None, source: str = "external"):
        primitive_ids = [int(pid) for pid in list(primitive_ids or [])]
        self._clear_planned_action_cache()
        self._prefix_steps_queue.clear()
        if len(primitive_ids) == 0:
            return None

        phase_actions = [np.asarray(self.primitive_lib.get_actions(pid), dtype=np.float64).copy() for pid in primitive_ids]
        plan_debug = None

        if self._primitive_refinement_enabled_now():
            try:
                result = self._primitive_refiner.refine_plan(self.env, self.primitive_lib, primitive_ids)
                phase_actions = [np.asarray(actions, dtype=np.float64).copy() for actions in result.phase_actions]
                plan_debug = result.to_debug_dict()
            except Exception as exc:
                plan_debug = {
                    "attempted": True,
                    "applied": False,
                    "feasible": False,
                    "plan_length": int(len(primitive_ids)),
                    "total_steps": int(sum(int(a.shape[0]) for a in phase_actions)),
                    "reason": "error",
                    "error": str(exc),
                }

        self._planned_action_queue = [np.asarray(actions, dtype=np.float64).copy() for actions in phase_actions]
        self._planned_primitive_queue = list(primitive_ids)
        self._planned_phase_debug_queue = [None] * len(primitive_ids)
        self._planned_cache_source = str(source)

        if prefix_steps is not None:
            self._prefix_steps_queue = [int(prefix_steps)] + [None] * max(0, len(primitive_ids) - 1)
        else:
            self._prefix_steps_queue = [None] * len(primitive_ids)

        return plan_debug

    def _should_takeover(self, obs_vec: np.ndarray) -> bool:
        """Dynamic trigger + hysteresis for terminal takeover."""
        if not self._takeover_enabled_now():
            return False

        goal = self._parse_goal_repr_from_obs(obs_vec)
        dist = float(goal.get("dist", 1e9))
        try:
            from configs import (
                TAKEOVER_DIST_BASE,
                TAKEOVER_DIST_HYSTERESIS,
                TAKEOVER_NEAR_GOAL_ONLY,
                TAKEOVER_NEAR_GOAL_DIST,
                TAKEOVER_EARLY_HEADING_ERR,
                TAKEOVER_EARLY_ARTICULATION,
                TAKEOVER_EARLY_MIN_LIDAR,
                TAKEOVER_DIST_SPEED_GAIN,
                TAKEOVER_DIST_OBS_DENSITY_GAIN,
                LIDAR_NUM,
                LIDAR_RANGE,
            )

            base = float(TAKEOVER_DIST_BASE)
            hyst = float(TAKEOVER_DIST_HYSTERESIS)
            near_goal_only = bool(TAKEOVER_NEAR_GOAL_ONLY)
            near_goal_dist = float(TAKEOVER_NEAR_GOAL_DIST)
            heading_thr = float(TAKEOVER_EARLY_HEADING_ERR)
            art_thr = float(TAKEOVER_EARLY_ARTICULATION)
            min_lidar_thr = float(TAKEOVER_EARLY_MIN_LIDAR)
            speed_gain = float(TAKEOVER_DIST_SPEED_GAIN)
            dens_gain = float(TAKEOVER_DIST_OBS_DENSITY_GAIN)
            lidar_n = int(LIDAR_NUM)
            lidar_r = float(LIDAR_RANGE)
        except Exception:
            base, hyst = 10.0, 2.0
            near_goal_only, near_goal_dist = True, 4.0
            heading_thr, art_thr, min_lidar_thr = math.radians(35), math.radians(25), 2.0
            speed_gain, dens_gain = 0.0, 0.0
            lidar_n, lidar_r = 120, 30.0

        obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
        lidar = obs_vec[:lidar_n]
        min_lidar = float(np.min(lidar)) * lidar_r
        obs_density = float(np.mean((lidar * lidar_r) < 3.0))

        # goal heading/articulation (ego frame)
        rel_heading = float(goal.get("goal_heading", 0.0))
        articulation = float(goal.get("articulation", 0.0))

        # dynamic takeover distance
        # speed is encoded in last 2 dims; first is normalized speed in [-1,1]
        try:
            speed_norm = float(obs_vec[lidar_n + 7])
            speed_mps = abs(speed_norm) * 2.5
        except Exception:
            speed_mps = 0.0
        takeover_dist = base + speed_gain * speed_mps + dens_gain * obs_density

        if near_goal_only:
            if self._takeover_active:
                if dist > (near_goal_dist + hyst):
                    return False
            elif dist > near_goal_dist:
                return False

        if self._takeover_active:
            # hysteresis exit
            return dist <= (takeover_dist + hyst)

        # enter takeover if within dist OR difficulty high
        if dist <= takeover_dist:
            return True
        if abs(rel_heading) >= heading_thr:
            return True
        if abs(articulation) >= art_thr:
            return True
        if min_lidar <= min_lidar_thr:
            return True

        return False

    def _maybe_plan_terminal_takeover(self, obs_vec, done: bool, info: dict):
        if done:
            self._takeover_active = False
            self._prefix_steps_queue.clear()
            self._clear_planned_action_cache()
            return

        if not self._takeover_enabled_now():
            self._takeover_active = False
            self._takeover_fail_count = 0
            self._prefix_steps_queue.clear()
            self._clear_takeover_plan_cache_if_owned()
            info['takeover_active'] = False
            info['takeover_triggered'] = False
            return

        base_env = self.env
        if not hasattr(base_env, 'vehicle') or not hasattr(base_env, 'map'):
            return
        if base_env.vehicle is None or getattr(base_env.vehicle, 'state', None) is None:
            return

        obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
        goal = self._parse_goal_repr_from_obs(obs_vec)
        was_active = bool(self._takeover_active)
        self._takeover_active = bool(self._should_takeover(obs_vec))
        info['takeover_active'] = bool(self._takeover_active)
        info['takeover_triggered'] = bool((not was_active) and self._takeover_active)

        if not self._takeover_active:
            self._takeover_fail_count = 0
            self._prefix_steps_queue.clear()
            self._clear_takeover_plan_cache_if_owned()
            return

        # Build lidar slice
        try:
            from configs import LIDAR_NUM

            lidar_n = int(LIDAR_NUM)
        except Exception:
            lidar_n = 120
        lidar = obs_vec[:lidar_n]

        # Bi-directional mode heuristic: if goal is mostly behind, prefer reverse.
        # (Paper uses unreachable-zone reasoning; we use a minimal mode machine.)
        if self._takeover_mode == "auto":
            if float(goal.get("goal_x", 0.0)) < -1.0 and float(goal.get("dist", 0.0)) > 1.0:
                mode = "reverse"
            else:
                mode = "forward"
        else:
            mode = self._takeover_mode

        # Plan using RHP planner if enabled; otherwise keep the original depth-2 planner behavior.
        plan_ids = None
        prefix_steps = None
        debug = {}

        if not getattr(self, "_takeover_use_rhp", False):
            plan_ids = self.plan_to_dest(max_len=self.takeover_max_len)
            if plan_ids is not None and len(plan_ids) > 0:
                refinement_plan_debug = self.prepare_plan_execution(plan_ids, prefix_steps=None, source="takeover")
                info['path_to_dest'] = list(map(int, plan_ids))
                if refinement_plan_debug is not None:
                    info['refinement_plan_debug'] = refinement_plan_debug
            return

        if self._takeover_planner is not None:
            res = self._takeover_planner.plan(
                state=base_env.vehicle.state,
                obs=obs_vec,
                lidar=lidar,
                goal_repr=goal,
                prev_choice=self._takeover_prev_choice,
                mode=mode,
            )
            plan_ids = res.primitive_ids
            prefix_steps = res.prefix_steps
            debug = res.debug or {}

        if plan_ids is None or len(plan_ids) == 0:
            self._takeover_fail_count += 1
            debug = {**debug, "fallback": "old_depth2"}
            try:
                from configs import TAKEOVER_FALLBACK_OLD_PLANNER

                allow_old = bool(TAKEOVER_FALLBACK_OLD_PLANNER)
            except Exception:
                allow_old = False
            if allow_old:
                plan_ids = self.plan_to_dest(max_len=1)
                prefix_steps = None
            else:
                plan_ids = []

        if plan_ids is not None and len(plan_ids) > 0:
            refinement_plan_debug = self.prepare_plan_execution(plan_ids, prefix_steps=prefix_steps, source="takeover")
            info['path_to_dest'] = list(map(int, plan_ids))
            info['takeover_expert_primitive'] = int(plan_ids[0])
            info['takeover_plan_length'] = int(len(plan_ids))
            if prefix_steps is not None:
                info['takeover_prefix_steps'] = int(prefix_steps)
            if refinement_plan_debug is not None:
                info['refinement_plan_debug'] = refinement_plan_debug

            self._takeover_prev_choice = int(plan_ids[0])
        else:
            self._clear_takeover_plan_cache_if_owned()
            info['takeover_no_path'] = True

        # Light profiling hooks
        if len(debug) > 0:
            info['takeover_debug'] = debug


    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _success_metrics(self, state):
        base_env = self.env
        dest = base_env.map.dest

        heading_diff = state.heading - dest.heading
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        heading_diff_abs = abs(heading_diff)

        front_box_ego = Polygon(state.create_box()[0])
        front_box_dest = Polygon(dest.create_box()[0])
        intersection_area = front_box_ego.intersection(front_box_dest).area
        overlap_ratio = intersection_area / (front_box_dest.area + 1e-9)
        return overlap_ratio, heading_diff_abs

    def _is_state_valid(self, state) -> bool:
        base_env = self.env
        world_map = base_env.map

        x, y = state.loc.x, state.loc.y
        if x < world_map.xmin or x > world_map.xmax or y < world_map.ymin or y > world_map.ymax:
            return False

        obstacles = getattr(world_map, 'obstacles', []) or []
        boxes = state.create_box()
        for box in boxes:
            for obst in obstacles:
                if box.intersects(obst.shape):
                    return False
        return True

    def _simulate_primitive(self, state0, primitive_id):
        """Simulate executing one primitive from a given state; returns (state1, feasible)."""
        base_env = self.env
        vehicle = base_env.vehicle

        try:
            from configs import NUM_STEP
        except Exception:
            NUM_STEP = None

        actions = self.primitive_lib.get_actions(int(primitive_id))
        steps = min(self.H, int(actions.shape[0]))
        state = copy.deepcopy(state0)

        for t in range(steps):
            action = actions[t]
            if NUM_STEP is None:
                state = vehicle.kinetic_model.step(state, action)
            else:
                state = vehicle.kinetic_model.step(state, action, step_time=NUM_STEP)
            if not self._is_state_valid(state):
                return state, False

        return state, True

    def _rank_primitives_approx(self, state, k: int):
        """Cheaply rank primitives by approximate progress toward destination.

        Uses library deltas (dx, dy, dtheta, gamma) defined in the canonical frame.
        The approximation rotates (dx, dy) by current heading and adds dtheta.

        If deltas are unavailable, falls back to returning all primitives.
        """
        if self._primitive_deltas is None:
            return list(range(self.action_space.n))

        base_env = self.env
        dest = base_env.map.dest
        deltas = self._primitive_deltas
        n = min(self.action_space.n, int(deltas.shape[0]))

        c = math.cos(state.heading)
        s = math.sin(state.heading)

        scores = []
        for pid in range(n):
            dx, dy, dtheta = float(deltas[pid, 0]), float(deltas[pid, 1]), float(deltas[pid, 2])
            # rotate delta into world frame
            pred_x = state.loc.x + c * dx - s * dy
            pred_y = state.loc.y + s * dx + c * dy
            pred_heading = state.heading + dtheta
            # distance + heading alignment heuristic
            dist = math.hypot(pred_x - dest.loc.x, pred_y - dest.loc.y)
            hd = pred_heading - dest.heading
            hd = (hd + math.pi) % (2 * math.pi) - math.pi
            score = -dist - 0.75 * abs(hd)
            scores.append((score, pid))

        scores.sort(reverse=True, key=lambda x: x[0])
        k = int(k)
        if k <= 0 or k >= len(scores):
            return [pid for _, pid in scores]
        return [pid for _, pid in scores[:k]]

    def plan_to_dest(self, max_len: int = 6):
        """Terminal planner using motion primitives with depth-2 lookahead.

        Depth-2 tree search (receding horizon):
        - enumerate first primitive -> simulate to state1
        - enumerate second primitive -> simulate to state2
        - score state2 and pick best (pid1, pid2)
        - execute pid1 virtually and repeat

        Returns a list of primitive IDs to execute. Does not modify env state.
        """
        base_env = self.env
        if not hasattr(base_env, 'vehicle') or not hasattr(base_env, 'map'):
            return None
        if base_env.vehicle is None or getattr(base_env.vehicle, 'state', None) is None:
            return None

        state = copy.deepcopy(base_env.vehicle.state)
        dest = base_env.map.dest
        plan = []
        max_len = int(max_len)

        # Safety guard: don't spend time planning if we're not in takeover range.
        try:
            if state.loc.distance(dest.loc) >= float(self.takeover_dist):
                return None
        except Exception:
            pass

        def score_state(s):
            dist = s.loc.distance(dest.loc)
            overlap, heading_diff_abs = self._success_metrics(s)
            # Heuristic score: favor overlap + heading alignment + distance reduction
            return (5.0 * overlap) - (0.25 * dist) - (0.75 * heading_diff_abs)

        for _ in range(max_len):
            # If already satisfies success condition, stop.
            overlap, heading_diff_abs = self._success_metrics(state)
            if heading_diff_abs < math.radians(15) and overlap > 0.7:
                break

            best_pid1 = None
            best_pid2 = None
            best_score = None

            # Depth-2 lookahead with pruning
            pid1_candidates = self._rank_primitives_approx(state, self._plan_topk1)
            for pid1 in pid1_candidates:
                state1, feasible1 = self._simulate_primitive(state, pid1)
                if not feasible1:
                    continue

                # If first step already succeeds, prefer it.
                overlap1, hd1 = self._success_metrics(state1)
                if hd1 < math.radians(15) and overlap1 > 0.7:
                    s1 = score_state(state1) + 10.0
                    if best_score is None or s1 > best_score:
                        best_score = s1
                        best_pid1 = pid1
                        best_pid2 = None
                    continue

                best2_local = None
                best_pid2_local = None
                pid2_candidates = self._rank_primitives_approx(state1, self._plan_topk2)
                for pid2 in pid2_candidates:
                    state2, feasible2 = self._simulate_primitive(state1, pid2)
                    if not feasible2:
                        continue
                    s2 = score_state(state2)
                    if best2_local is None or s2 > best2_local:
                        best2_local = s2
                        best_pid2_local = pid2

                # If no feasible second step, fall back to scoring state1.
                if best2_local is None:
                    best2_local = score_state(state1)

                if best_score is None or best2_local > best_score:
                    best_score = best2_local
                    best_pid1 = pid1
                    best_pid2 = best_pid2_local

            if best_pid1 is None:
                break

            plan.append(int(best_pid1))
            state, _ = self._simulate_primitive(state, best_pid1)

        return plan

    def get_action_mask(self, obs_vec=None):
        """Return a family-level action mask with shape [family_count]."""
        base_obs = self._extract_base_obs(obs_vec)
        mode_debug = self._estimate_mode_state(base_obs, update_state=True)
        primitive_mode = str(mode_debug.get('selected_mode', self._current_primitive_mode))
        self._last_mode_debug = dict(mode_debug)
        self._pending_mode_debug = dict(mode_debug)
        requested_mode = getattr(self, "_action_mask_mode", "hybrid")
        soft_ray_available = not (
            requested_mode == "soft_ray"
            and getattr(self, "_ray_safety_index", None) is None
        )

        effective_mode = requested_mode
        soft_fallback = None
        if requested_mode == "soft_ray" and not soft_ray_available:
            effective_mode = "hybrid"
            soft_fallback = "hybrid_no_ray_safety"

        if (
            self._action_mask_update_every_k > 1
            and self._action_mask_cached is not None
            and self._action_mask_calls_since_update < (self._action_mask_update_every_k - 1)
        ):
            self._action_mask_calls_since_update += 1
            return self._action_mask_cached.copy()

        n_actions = int(self.action_space.n)
        mode_candidates = self._mode_fallback_candidates(primitive_mode)
        selected_mode = str(primitive_mode)
        selected_safety_mask = np.zeros((n_actions,), dtype=np.float32)
        selected_mode_weights = np.ones((n_actions,), dtype=np.float32)
        selected_articulation_weights = np.ones((n_actions,), dtype=np.float32)
        selected_safe_steps = np.zeros((n_actions,), dtype=np.int64)
        selected_mask = np.zeros((n_actions,), dtype=np.float32)
        fallback_to_finer_mode = False
        all_modes_invalid = True
        fallback_family_id = None

        for idx, candidate_mode in enumerate(mode_candidates):
            candidate_mode = str(candidate_mode)
            candidate_mode_debug = dict(mode_debug)
            candidate_mode_debug['selected_mode'] = candidate_mode

            if effective_mode == "soft_ray":
                safety_mask = self._compute_soft_ray_action_mask(base_obs, primitive_mode=candidate_mode, mode_debug=candidate_mode_debug)
            else:
                safety_mask = self._compute_hard_action_mask(base_obs, mode_override=effective_mode, primitive_mode=candidate_mode, mode_debug=candidate_mode_debug)
            safety_mask = np.asarray(safety_mask, dtype=np.float32).reshape(-1)

            mode_weights = self._mode_preference_weights(candidate_mode)
            articulation_weights = self._articulation_preference_weights(candidate_mode, obs_vec=base_obs)
            mask = np.clip(safety_mask * mode_weights * articulation_weights, 0.0, 1.0).astype(np.float32)

            safe_steps = np.zeros((n_actions,), dtype=np.int64)
            for family_id in range(n_actions):
                steps, _prefix_source, _ref, _resolved_debug = self._family_safe_steps_for_mode(
                    family_id,
                    base_obs,
                    primitive_mode=candidate_mode,
                    mode_debug=candidate_mode_debug,
                )
                safe_steps[int(family_id)] = int(max(0, steps))

            hard_valid = safe_steps >= int(self._min_safe_prefix_steps)
            mask = np.where(hard_valid, mask, 0.0).astype(np.float32)
            valid_count = int(np.count_nonzero(hard_valid))

            selected_mode = str(candidate_mode)
            selected_safety_mask = safety_mask
            selected_mode_weights = mode_weights
            selected_articulation_weights = articulation_weights
            selected_safe_steps = safe_steps
            selected_mask = mask

            if valid_count > 0:
                all_modes_invalid = False
                fallback_to_finer_mode = idx > 0
                break

        mask = selected_mask

        # Preserve previous all-invalid fallback behavior as a last resort when enabled,
        # but only among actions that satisfy minimum safe-step execution.
        if all_modes_invalid and self._all_invalid_fallback_enabled:
            hard_valid = selected_safe_steps >= int(self._min_safe_prefix_steps)
            if np.any(hard_valid):
                fallback_family_id = self._choose_all_invalid_fallback(
                    selected_mode,
                    selected_safety_mask,
                    selected_mode_weights,
                    selected_articulation_weights,
                    obs_vec=base_obs,
                )
                if int(fallback_family_id) < n_actions and bool(hard_valid[int(fallback_family_id)]):
                    mask = np.zeros_like(mask, dtype=np.float32)
                    mask[int(fallback_family_id)] = 1.0
                    self._all_invalid_fallback_count += 1
                    all_modes_invalid = False

        self._action_mask_cached = mask.copy()
        self._action_mask_calls_since_update = 0
        self._last_family_mask = mask.copy()
        final_debug = dict(self._last_action_mask_debug or {})
        if soft_fallback is not None:
            final_debug["fallback"] = soft_fallback
            final_debug["effective_mode"] = effective_mode
        final_debug.update(
            {
                "selected_mode": str(selected_mode),
                "effective_primitive_mode": str(selected_mode),
                "fallback_to_finer_mode": bool(fallback_to_finer_mode),
                "all_modes_invalid": bool(all_modes_invalid),
                "mode_transitioned": bool(mode_debug.get('mode_transitioned', False)),
                "mode_transition_count": int(mode_debug.get('mode_transition_count', self._mode_transition_count)),
                "congestion_score": float(mode_debug.get('congestion_score', 0.0)),
                "valid_action_ratio": float(mode_debug.get('valid_action_ratio', 0.0)),
                "mean_soft_mask": float(mode_debug.get('mean_soft_mask', 0.0)),
                "abs_phi_ratio": float(mode_debug.get('abs_phi_ratio', 0.0)),
                "front_clearance_m": float(mode_debug.get('front_clearance', 0.0)),
                "rear_clearance_m": float(mode_debug.get('rear_clearance', 0.0)),
                "min_safe_steps": int(np.min(selected_safe_steps)) if selected_safe_steps.size else 0,
                "max_safe_steps": int(np.max(selected_safe_steps)) if selected_safe_steps.size else 0,
                "mean_safe_steps": float(np.mean(selected_safe_steps)) if selected_safe_steps.size else 0.0,
                "valid_action_count_after_fallback": int(np.count_nonzero(selected_safe_steps >= int(self._min_safe_prefix_steps))),
                "valid_action_ratio_after_fallback": float(np.mean(selected_safe_steps >= int(self._min_safe_prefix_steps))) if selected_safe_steps.size else 0.0,
                "final_mask_min": float(np.min(mask)) if mask.size else 0.0,
                "final_mask_max": float(np.max(mask)) if mask.size else 0.0,
                "final_mask_mean": float(np.mean(mask)) if mask.size else 0.0,
                "effective_action_count": int(np.count_nonzero(mask > 0.0)),
                "all_invalid_fallback": fallback_family_id is not None,
                "all_invalid_fallback_family_id": int(fallback_family_id) if fallback_family_id is not None else None,
                "all_invalid_fallback_count": int(self._all_invalid_fallback_count),
            }
        )
        self._last_action_mask_debug = final_debug

        mode_debug_sync = dict(mode_debug)
        mode_debug_sync['selected_mode'] = str(selected_mode)
        mode_debug_sync['valid_action_ratio'] = float(final_debug.get('valid_action_ratio_after_fallback', 0.0))
        self._current_primitive_mode = str(selected_mode)
        self._last_mode_debug = dict(mode_debug_sync)
        self._pending_mode_debug = dict(mode_debug_sync)

        return mask

    def _compute_hard_action_mask(self, obs_vec=None, mode_override=None, primitive_mode=None, mode_debug=None, write_debug: bool = True, return_debug: bool = False):
        n_actions = self.action_space.n
        mask = np.zeros(n_actions, dtype=np.int8)
        self._soft_prefix_family_steps = {}
        self._family_resolution_cache = {}
        base_obs = self._extract_base_obs(obs_vec)
        selection_context = self._variant_selection_context(mode_debug)
        for family_id in range(n_actions):
            ref = self._resolve_family_ref(family_id, obs_vec=base_obs, primitive_mode=primitive_mode, selection_context=selection_context)
            mask[int(family_id)] = 1 if self._variant_rollout_is_safe(int(ref.flat_index)) else 0
            variant_horizons = np.asarray(getattr(self.primitive_lib, 'variant_horizons', np.full((n_actions,), int(self.H), dtype=np.int64)), dtype=np.int64).reshape(-1)
            horizon = int(variant_horizons[int(ref.flat_index)]) if int(ref.flat_index) < variant_horizons.size else int(self.H)
            self._soft_prefix_family_steps[int(family_id)] = int(horizon if mask[int(family_id)] > 0 else 0)

        debug = {
            "mode": str(mode_override or getattr(self, "_action_mask_mode", "hybrid")),
            "selected_mode": str(self._mode_name(primitive_mode)),
            "family_feasible_count": int(np.count_nonzero(mask)),
        }
        if write_debug:
            self._last_action_mask_debug = debug
        if return_debug:
            return mask, debug
        return mask

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # Reset takeover runtime state
        self._takeover_active = False
        self._takeover_prev_choice = None
        self._takeover_mode = "auto"
        self._takeover_fail_count = 0
        self._prefix_steps_queue.clear()
        self._soft_prefix_family_steps = {}
        self._clear_planned_action_cache()
        self._consecutive_blocked_actions = 0

        # Invalidate obstacle cache; it will be rebuilt lazily on first mask query.
        self._mask_obstacles_prepared = None
        self._mask_obstacles_bounds = None

        self._action_mask_cached = None
        self._action_mask_calls_since_update = 0
        self._family_resolution_cache = {}
        if isinstance(out, tuple) and len(out) > 0:
            base_obs = copy.deepcopy(out[0])
            self._last_base_obs = copy.deepcopy(base_obs)
            self._progress_history.clear()
            self._last_progress_metrics = None
            self._no_progress_steps = 0
            self._mode_transition_count = 0
            self._mode_hold_steps = 0
            self._current_primitive_mode = str(self._primitive_mode_names[0])
            initial_mode_debug = self._estimate_mode_state(base_obs, update_state=False)
            self._last_mode_debug = dict(initial_mode_debug)
            self._pending_mode_debug = None
            self._last_obs = self._augment_observation(base_obs, mode_debug=initial_mode_debug)
            self._update_progress_tracking()
            out = (self._last_obs, out[1]) if len(out) > 1 else (self._last_obs,)
        else:
            self._last_base_obs = None
            self._last_obs = None
        return out


def update_primitive_library(env_or_wrapper, new_lib, H: int = None):
    """Hot-update a MacroActionWrapper (or wrapped env) with a new primitive library.

    This is intended to be called between episodes.

    Args:
        env_or_wrapper: MacroActionWrapper instance (preferred). If a base env is
            passed by mistake, it is returned unchanged.
        new_lib: PrimitiveLibrary-like object
        H: optional horizon override

    Returns:
        updated wrapper (same object if possible)
    """
    w = env_or_wrapper
    if not isinstance(w, MacroActionWrapper):
        return env_or_wrapper

    w.primitive_lib = new_lib
    if H is None:
        H = getattr(new_lib, 'horizon', None)
    if H is not None:
        w.H = int(H)

    # update action space
    try:
        from gymnasium import spaces

        w.action_space = spaces.Discrete(int(getattr(new_lib, 'action_dim', getattr(new_lib, 'size'))))
    except Exception:
        pass

    # refresh cached deltas used by approximate planner ranking
    w._primitive_deltas = getattr(new_lib, 'deltas', None)
    w._ray_safety_index = getattr(new_lib, 'ray_safety_index', None)
    w._mask_grid_index = getattr(new_lib, 'grid_index', None)
    w._family_resolution_cache = {}
    if w._mask_grid_index is not None:
        try:
            r = max(0.0, float(getattr(w, '_mask_occupancy_inflation_radius', 0.0)))
            res = max(1e-6, float(w._mask_grid_index.grid_resolution))
            rad = int(math.ceil(r / res))
            offsets = []
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if (dx * dx + dy * dy) * (res * res) <= r * r + 1e-9:
                        offsets.append((dx, dy))
            w._mask_inflation_offsets = offsets
        except Exception:
            w._mask_inflation_offsets = []
    else:
        w._mask_inflation_offsets = []

    # reset prepared caches to avoid stale references
    try:
        w._mask_obstacles_prepared = None
        w._mask_obstacles_bounds = None
        w._prefix_steps_queue.clear()
        w._clear_planned_action_cache()
    except Exception:
        pass

    try:
        w._action_mask_cached = None
        w._action_mask_calls_since_update = 0
    except Exception:
        pass

    return w
