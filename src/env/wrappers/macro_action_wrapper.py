import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
import math
import time
from shapely.geometry import Polygon
try:
    from shapely.prepared import prep
except Exception:  # pragma: no cover
    prep = None

try:
    from terminal_takeover_rhp import RecedingHorizonTakeoverPlanner
except Exception:  # pragma: no cover
    RecedingHorizonTakeoverPlanner = None

class MacroActionWrapper(gym.Wrapper):
    """
    Gym wrapper that converts discrete primitive IDs into a sequence of low-level actions.
    """
    def __init__(self, env, primitive_lib, H=None, takeover_dist: float = None, takeover_max_len: int = 3, normalize_before_step: bool = True):
        """
        Args:
            env: The base environment.
            primitive_lib: An object with .get_actions(primitive_id) -> np.ndarray[H, 2]
                           and .size property.
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
        self.action_space = spaces.Discrete(primitive_lib.size)
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

        # Optional RHP takeover planner (paper-style fast prune + group scoring)
        self._takeover_planner = None
        try:
            from configs import (
                TAKEOVER_USE_RHP,
                LIDAR_NUM,
                LIDAR_RANGE,
                TAKEOVER_SCORE_WEIGHTS,
                OCCUPANCY_INFLATION_RADIUS,
                TAKEOVER_GROUP_SCORE_TOPK,
                TAKEOVER_MAX_PREFIX_STEPS,
            )

            use_rhp = bool(TAKEOVER_USE_RHP)
        except Exception:
            use_rhp = False
            LIDAR_NUM = None
            LIDAR_RANGE = None
            TAKEOVER_SCORE_WEIGHTS = None
            OCCUPANCY_INFLATION_RADIUS = 1.5
            TAKEOVER_GROUP_SCORE_TOPK = 5
            TAKEOVER_MAX_PREFIX_STEPS = None

        if use_rhp and RecedingHorizonTakeoverPlanner is not None:
            self._takeover_use_rhp = True
            grid_index = getattr(self.primitive_lib, "grid_index", None)
            if grid_index is None:
                # Degraded fallback: build an approximate index once from deltas.
                # This keeps online complexity near O(#occupied_cells + #hits) and avoids per-step rollouts.
                try:
                    from primitives.primitive_index import build_approx_index_from_deltas
                    from configs import GRID_RESOLUTION

                    # same bounds as the offline script defaults (tuned for terminal 12m-ish range)
                    grid_index = build_approx_index_from_deltas(
                        actions=getattr(self.primitive_lib, "actions"),
                        deltas=getattr(self.primitive_lib, "deltas"),
                        grid_resolution=float(GRID_RESOLUTION),
                        x_min=-6.0,
                        x_max=12.0,
                        y_min=-9.0,
                        y_max=9.0,
                        group_prefix_steps=max(1, int(round(self.H * 0.3))),
                    )
                except Exception:
                    grid_index = None

            if grid_index is not None:
                try:
                    self._takeover_planner = RecedingHorizonTakeoverPlanner(
                        primitive_actions=getattr(self.primitive_lib, "actions"),
                        primitive_deltas=getattr(self.primitive_lib, "deltas"),
                        grid_index=grid_index,
                        lidar_num=int(LIDAR_NUM) if LIDAR_NUM is not None else 120,
                        lidar_range=float(LIDAR_RANGE) if LIDAR_RANGE is not None else 30.0,
                        score_weights=TAKEOVER_SCORE_WEIGHTS or {},
                        occupancy_inflation_radius=float(OCCUPANCY_INFLATION_RADIUS),
                        group_score_topk=int(TAKEOVER_GROUP_SCORE_TOPK),
                        max_prefix_steps=TAKEOVER_MAX_PREFIX_STEPS,
                    )
                except Exception:
                    self._takeover_planner = None
        else:
            self._takeover_use_rhp = False

        # Planner pruning (depth-2 search can be expensive). We use primitive library
        # deltas (precomputed under a canonical start) to rank candidates cheaply.
        self._plan_topk1 = 12
        self._plan_topk2 = 8
        self._primitive_deltas = getattr(primitive_lib, 'deltas', None)

        # Cached obstacles for faster masking (rebuilt on reset)
        self._mask_obstacles_prepared = None
        self._mask_obstacles_bounds = None

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
        
    def step(self, primitive_id):
        """
        Execute the primitive corresponding to primitive_id.
        accumulate reward, check done at each step.
        """
        # Get action sequence from library
        # Ensure primitive_id is int
        if isinstance(primitive_id, np.ndarray):
            primitive_id = primitive_id.item()
        
        actions = self.primitive_lib.get_actions(primitive_id)
        # actions shape: [H, 2]

        total_reward = 0.0
        done = False
        info = {}
        steps_executed = 0
        
        last_obs = None
        
        # We need to handle potential 'truncated' from gymnasium if base env uses it.
        terminated = False
        truncated = False

        max_steps = min(self.H, int(actions.shape[0]))
        used_prefix_steps = None
        if len(self._prefix_steps_queue) > 0:
            try:
                used_prefix_steps = self._prefix_steps_queue.pop(0)
                if used_prefix_steps is not None:
                    max_steps = min(max_steps, int(used_prefix_steps))
            except Exception:
                used_prefix_steps = None

        for t in range(max_steps):
            # Execute one low-level step
            action = actions[t]
            if self.normalize_before_step:
                action = self._physical_to_normalized_action(action)
            step_result = self.env.step(action)
            
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
            
            if done:
                break
        
        info['primitive_id'] = primitive_id
        info['executed_steps'] = steps_executed
        if used_prefix_steps is not None:
            info['prefix_steps_used'] = int(used_prefix_steps)

        # Provide a terminal plan (HOPE-compatible key) for the next decision.
        # Online planner is called at high frequency, and only commits a short prefix.
        try:
            self._maybe_plan_terminal_takeover(last_obs, done, info)
        except Exception:
            pass
        
        # Return consistent with Gymnasium
        return last_obs, total_reward, terminated, truncated, info

    def _parse_goal_repr_from_obs(self, obs_vec: np.ndarray) -> dict:
        """Decode goal representation in ego frame from CarParking observation vector."""
        obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
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
        dist = float(target[0]) * max_dist
        rel_angle = math.atan2(float(target[2]), float(target[1]))
        rel_heading = math.atan2(float(target[4]), float(target[3]))
        articulation = math.atan2(float(target[6]), float(target[5]))

        return {
            "goal_x": dist * math.cos(rel_angle),
            "goal_y": dist * math.sin(rel_angle),
            "goal_heading": _wrap_pi(rel_heading),
            "articulation": _wrap_pi(articulation),
            "dist": dist,
            "rel_angle": rel_angle,
        }

    def _should_takeover(self, obs_vec: np.ndarray) -> bool:
        """Dynamic trigger + hysteresis for terminal takeover."""
        goal = self._parse_goal_repr_from_obs(obs_vec)
        dist = float(goal.get("dist", 1e9))
        try:
            from configs import (
                TAKEOVER_DIST_BASE,
                TAKEOVER_DIST_HYSTERESIS,
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
            heading_thr = float(TAKEOVER_EARLY_HEADING_ERR)
            art_thr = float(TAKEOVER_EARLY_ARTICULATION)
            min_lidar_thr = float(TAKEOVER_EARLY_MIN_LIDAR)
            speed_gain = float(TAKEOVER_DIST_SPEED_GAIN)
            dens_gain = float(TAKEOVER_DIST_OBS_DENSITY_GAIN)
            lidar_n = int(LIDAR_NUM)
            lidar_r = float(LIDAR_RANGE)
        except Exception:
            base, hyst = 10.0, 2.0
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
                info['path_to_dest'] = list(map(int, plan_ids))
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
            info['path_to_dest'] = list(map(int, plan_ids))
            if prefix_steps is not None:
                self._prefix_steps_queue = [int(prefix_steps)] + [None] * max(0, len(plan_ids) - 1)
                info['takeover_prefix_steps'] = int(prefix_steps)
            else:
                self._prefix_steps_queue = [None] * len(plan_ids)

            self._takeover_prev_choice = int(plan_ids[0])
        else:
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

    def get_action_mask(self):
        """Return a binary mask over primitives: 1=feasible, 0=infeasible.

        Feasibility is checked by forward simulating each primitive from the
        current vehicle state using the SAME kinematic model, and marking
        primitives infeasible if they collide with obstacles or leave the map.

        This does NOT change environment state.
        """
        # IMPORTANT: during terminal takeover we must NOT forward-simulate all primitives
        # for collision checking (paper's two-step collision detection). We let the
        # takeover planner prune/choose, and keep PPO log_prob consistent by not masking.
        if getattr(self, "_takeover_active", False):
            return np.ones(self.action_space.n, dtype=np.int8)

        # If base env doesn't expose expected attributes, fall back to no mask.
        base_env = self.env
        if not hasattr(base_env, 'vehicle') or not hasattr(base_env, 'map'):
            return np.ones(self.action_space.n, dtype=np.int8)

        vehicle = base_env.vehicle
        world_map = base_env.map
        if vehicle is None or getattr(vehicle, 'state', None) is None:
            return np.ones(self.action_space.n, dtype=np.int8)

        state0 = vehicle.state
        n_actions = self.action_space.n
        mask = np.ones(n_actions, dtype=np.int8)

        # Cache obstacles list for speed (prepared in reset when possible)
        obstacles = getattr(world_map, 'obstacles', []) or []
        xmin, xmax = world_map.xmin, world_map.xmax
        ymin, ymax = world_map.ymin, world_map.ymax

        prepared = self._mask_obstacles_prepared
        obst_bounds = self._mask_obstacles_bounds
        if prepared is None or obst_bounds is None:
            # Fallback: build lightweight bounds cache
            prepared = [o.shape for o in obstacles]
            obst_bounds = [o.shape.bounds for o in obstacles]

        def bounds_overlap(a, b):
            return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

        # Import NUM_STEP from configs (used by the vehicle model)
        try:
            from configs import NUM_STEP
        except Exception:
            NUM_STEP = None

        for pid in range(n_actions):
            actions = self.primitive_lib.get_actions(pid)
            # Defensive: if library H differs, respect wrapper's H
            steps = min(self.H, int(actions.shape[0]))

            state = copy.deepcopy(state0)
            feasible = True

            for t in range(steps):
                action = actions[t]

                # Step with the same kinematic model as the real env.
                if NUM_STEP is None:
                    state = vehicle.kinetic_model.step(state, action)
                else:
                    state = vehicle.kinetic_model.step(state, action, step_time=NUM_STEP)

                # Out-of-map
                x, y = state.loc.x, state.loc.y
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    feasible = False
                    break

            # Collision check only at the end state (much faster; allows rare
            # false-feasible cases where intermediate collision would occur).
            if feasible and len(prepared) > 0:
                boxes = state.create_box()
                collided = False
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
                            collided = True
                            break
                    if collided:
                        break
                if collided:
                    feasible = False

            mask[pid] = 1 if feasible else 0

        # If everything got masked (can happen in tight scenarios), fall back.
        if mask.sum() == 0:
            mask[:] = 1

        return mask

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # Reset takeover runtime state
        self._takeover_active = False
        self._takeover_prev_choice = None
        self._takeover_mode = "auto"
        self._takeover_fail_count = 0
        self._prefix_steps_queue.clear()

        # Rebuild prepared obstacles cache for fast action masking.
        try:
            world_map = getattr(self.env, 'map', None)
            obstacles = getattr(world_map, 'obstacles', []) or []
            if prep is not None:
                self._mask_obstacles_prepared = [prep(o.shape) for o in obstacles]
            else:
                self._mask_obstacles_prepared = [o.shape for o in obstacles]
            self._mask_obstacles_bounds = [o.shape.bounds for o in obstacles]
        except Exception:
            self._mask_obstacles_prepared = None
            self._mask_obstacles_bounds = None
        return out
