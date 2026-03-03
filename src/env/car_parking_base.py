
'''
This is a collision-free env which only includes one parking case with random start position.
'''


import sys
sys.path.append("../")
from typing import Optional, Union
import math
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon, LinearRing
from shapely.affinity import affine_transform
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from env.vehicle import *
from env.map_base import *
from env.lidar_simulator import LidarSimlator
from env.parking_map_normal import ParkingMapNormal
from env.global_guidance import SoftGlobalGuidance
# from env.parking_map_dlp import ParkingMapDLP # Exclude DLP
# import env.reeds_shepp as rsCurve # Exclude RS
# from env.observation_processor import Obs_Processor # Exclude Image
# from model.action_mask import ActionMask # Exclude Action Mask
from configs import *

class CarParking(gym.Env):

    metadata = {
        "render_mode": [
            "human", 
            "rgb_array",
        ]
    }

    def __init__(
        self, 
        render_mode: str = None,
        fps: int = FPS,
        verbose: bool =True, 
        use_lidar_observation: bool =USE_LIDAR,
        use_img_observation: bool=USE_IMG,
        use_action_mask: bool=USE_ACTION_MASK,
    ):
        super().__init__()

        self.verbose = verbose
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        self.use_action_mask = use_action_mask
        self.render_mode = "human" if render_mode is None else render_mode
        self.fps = fps
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.t = 0.0
        self.k = None
        self.level = MAP_LEVEL
        self.tgt_repr_size = 7 # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)
        self._map_cache = {}

        if self.level in ['Normal', 'Complex', 'Extrem']:
            self.map = self._get_or_create_map(self.level)
        # elif self.level == 'dlp':
        #     self.map = ParkingMapDLP()
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH, articulated=True, trailer_length= TRAILER_LENGTH, hitch_offset=HITCH_OFFSET)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.global_guidance = None
        if ENABLE_GLOBAL_SOFT_GUIDANCE:
            self.global_guidance = SoftGlobalGuidance(
                grid_resolution=GUIDANCE_GRID_RESOLUTION,
                obstacle_inflation=GUIDANCE_OBS_INFLATION,
                map_margin=GUIDANCE_MAP_MARGIN,
                lookahead_base=GUIDANCE_LOOKAHEAD_BASE,
                lookahead_speed_gain=GUIDANCE_LOOKAHEAD_SPEED_GAIN,
                lookahead_min=GUIDANCE_LOOKAHEAD_MIN,
                lookahead_max=GUIDANCE_LOOKAHEAD_MAX,
                progress_search_window=GUIDANCE_PROGRESS_WINDOW,
                min_clearance_m=GUIDANCE_MIN_CLEARANCE_M,
                full_clearance_m=GUIDANCE_FULL_CLEARANCE_M,
                near_obs_dist_m=GUIDANCE_NEAR_OBS_DIST_M,
                max_dense_ratio=GUIDANCE_MAX_DENSE_RATIO,
            )
        self.reward = 0.0
        self.accum_arrive_reward = 0.0
        
        # Tracking variables for new penalties
        self.accum_turn_count = 0
        self.accum_turn_degree = 0.0
        self.accum_dist = 0.0

        self.action_space = spaces.Box(
            np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
            np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
        ) # steer, speed
       
        self.observation_space = {}
        # if self.use_action_mask:
        #     self.action_filter = ActionMask()
        #     self.observation_space['action_mask'] = spaces.Box(low=0, high=1, 
        #         shape=(N_DISCRETE_ACTION,), dtype=np.float64
        #     )
        # if self.use_img_observation:
        #     self.img_processor = Obs_Processor()
        #     self.observation_space['img'] = spaces.Box(low=0, high=255, 
        #         shape=(OBS_W//self.img_processor.downsample_rate, OBS_H//self.img_processor.downsample_rate, 
        #         self.img_processor.n_channels), dtype=np.uint8
        #     )
        #     self.raw_img_shape = (OBS_W, OBS_H, 3)
        if self.use_lidar_observation:
            # the observation is composed of lidar points and target representation
            # the target representation is (relative_distance, cos(theta), sin(theta), cos(phi), sin(phi))
            # where the theta indicates the relative angle of parking lot, and phi means the heading of 
            # parking lot in the polar coordinate of the ego car's view
            
            # Widen input state: Flatten everything into a single vector
            # Lidar + Target + Velocity (+ optional soft global guidance hint)
            self.obs_dim = LIDAR_NUM + self.tgt_repr_size + 2
            if ENABLE_GLOBAL_SOFT_GUIDANCE:
                self.obs_dim += GUIDANCE_FEATURE_DIM
            
            low_bound = -np.inf * np.ones(self.obs_dim)
            high_bound = np.inf * np.ones(self.obs_dim)
            
            self.observation_space = spaces.Box(
                low=low_bound, high=high_bound, shape=(self.obs_dim,), dtype=np.float64
            )
            
        self.observation_shape = (self.obs_dim,)
    
    def set_level(self, level:str=None):
        if level is None:
            self.map = self._get_or_create_map('Normal')
            self.level = 'Normal'
            return
        if level == self.level and self.level in self._map_cache:
            self.map = self._map_cache[self.level]
            return
        self.level = level
        if self.level in ['Normal', 'Complex', 'Extrem',]:
            self.map = self._get_or_create_map(self.level)
        # elif self.level == 'dlp':
        #     self.map = ParkingMapDLP()

    def _get_or_create_map(self, level: str):
        level = str(level)
        if level not in self._map_cache:
            self._map_cache[level] = ParkingMapNormal(level)
        return self._map_cache[level]

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        super().reset(seed=seed)
        
        case_id = options.get('case_id') if options else None
        data_dir = options.get('data_dir') if options else None
        level = options.get('level') if options else None

        self.reward = 0.0
        self.accum_arrive_reward = 0.0
        self.t = 0.0
        
        self.accum_turn_count = 0
        self.accum_turn_degree = 0.0
        self.accum_dist = 0.0

        if level is not None:
            self.set_level(level)
        initial_state = self.map.reset(case_id, data_dir)
        self.vehicle.reset(initial_state)
        self.matrix = self.coord_transform_matrix()

        if ENABLE_GLOBAL_SOFT_GUIDANCE and self.global_guidance is not None:
            try:
                dest_center = np.mean(self.map.dest_box.coords[:-1], axis=0)
                self.global_guidance.plan_path(
                    self.map,
                    start_xy=(float(self.vehicle.state.loc.x), float(self.vehicle.state.loc.y)),
                    goal_xy=(float(dest_center[0]), float(dest_center[1])),
                )
            except Exception:
                pass
        
        # For reward normalization
        self.initial_dist = float(self.vehicle.state.loc.distance(Point(self.map.dest.loc))) + 1e-6

        obs = self._build_observation()
        info = {
            'status': Status.CONTINUE,
            'reward_info': OrderedDict({k: 0.0 for k in REWARD_WEIGHT.keys()}),
            'path_to_dest': None,
        }
        return obs, info

    def coord_transform_matrix(self) -> list:
        """Get the transform matrix that convert the real world coordinate to the pygame coordinate.
        """
        win_w, win_h = WIN_W, WIN_H
        # xmin, xmax, ymin, ymax = self.map.xmin, self.map.xmax, self.map.ymin, self.map.ymax
        # scale = min(win_w/(xmax-xmin), win_h/(ymax-ymin))
        # x_off = -xmin*scale + (win_w - (xmax-xmin)*scale)/2
        # y_off = -ymin*scale + (win_h - (ymax-ymin)*scale)/2
        # return [scale, 0, 0, -scale, x_off, win_h-y_off]
        
        # Fixed scale for navigation
        scale = K * 10 # Adjust scale
        x_off = win_w/2
        y_off = win_h/2
        return [scale, 0, 0, -scale, x_off, y_off]

    def _build_observation(self):
        # get observation
        lidar_obs = np.zeros(LIDAR_NUM)
        if self.use_lidar_observation:
            lidar_obs = self.lidar.get_observation(self.vehicle.state, self.map.obstacles)
            # Normalize lidar
            lidar_obs = lidar_obs / LIDAR_RANGE
        
        # target representation
        # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)
        # theta: relative angle of parking lot
        # phi: heading of parking lot in the polar coordinate of the ego car's view
        
        # For navigation, dest is a point/box.
        # We use dest center.
        dest_center = np.mean(self.map.dest_box.coords[:-1], axis=0)
        ego_pos = self.vehicle.state.get_pos()
        
        dx = dest_center[0] - ego_pos[0]
        dy = dest_center[1] - ego_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        # Angle to target
        angle_to_target = math.atan2(dy, dx)
        relative_angle = angle_to_target - ego_pos[2]
        
        # Target heading relative to ego
        target_heading = self.map.dest.heading
        relative_heading = target_heading - ego_pos[2]
        
        # Articulation angle
        articulation_angle = ego_pos[2] - self.vehicle.state.rear_heading
        
        target_obs = np.array([
            dist/MAX_DIST_TO_DEST,
            math.cos(relative_angle),
            math.sin(relative_angle),
            math.cos(relative_heading),
            math.sin(relative_heading),
            math.cos(articulation_angle),
            math.sin(articulation_angle)
        ])
        
        # Velocity info (normalized roughly)
        # Speed range [-2.5, 2.5], Steer range [-0.6, 0.6]
        vel_obs = np.array([
            self.vehicle.state.speed / 2.5,
            self.vehicle.state.steering / 0.6
        ])

        guidance_obs = np.zeros((GUIDANCE_FEATURE_DIM,), dtype=np.float64) if ENABLE_GLOBAL_SOFT_GUIDANCE else np.zeros((0,), dtype=np.float64)
        if ENABLE_GLOBAL_SOFT_GUIDANCE and self.global_guidance is not None:
            try:
                guidance_obs = self.global_guidance.get_soft_hint(
                    state_x=float(self.vehicle.state.loc.x),
                    state_y=float(self.vehicle.state.loc.y),
                    heading=float(self.vehicle.state.heading),
                    speed=float(self.vehicle.state.speed),
                    lidar_norm=lidar_obs,
                    lidar_range=LIDAR_RANGE,
                )
            except Exception:
                guidance_obs = np.zeros((GUIDANCE_FEATURE_DIM,), dtype=np.float64)
        
        # Concatenate all
        obs = np.concatenate([lidar_obs, target_obs, vel_obs, guidance_obs])
        return obs

    def step(self, action: np.ndarray = None):
        prev_state = None
        if action is not None:
            # Scale action from [-1, 1] to [min, max]
            steer_min, steer_max = VALID_STEER
            speed_min, speed_max = VALID_SPEED
            
            scaled_action = np.zeros_like(action)
            scaled_action[0] = 0.5 * (action[0] + 1.0) * (steer_max - steer_min) + steer_min
            scaled_action[1] = 0.5 * (action[1] + 1.0) * (speed_max - speed_min) + speed_min
            
            prev_state = deepcopy(self.vehicle.state)
            self.vehicle.step(scaled_action)
            self.t += 1

        obs = self._build_observation()
        
        # calculate reward (HOPE-style shaping)
        reward, done, info = self.get_reward(action, prev_state=prev_state)
        self.reward = reward
        
        terminated = False
        truncated = False
        
        if done:
            if info['status'] == Status.OUTTIME:
                truncated = True
            else:
                terminated = True
        
        return obs, reward, terminated, truncated, info

    def _wrap_pi(self, a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _get_angle_diff(self, a: float, b: float) -> float:
        return abs(self._wrap_pi(a - b))  # 0..pi

    def _get_slot_approach_dir_diff(self, state: State) -> float:
        """Angle diff between (agent->slot-center direction) and slot heading, in [0, pi]."""
        dest_center = np.mean(self.map.dest_box.coords[:-1], axis=0)
        dx = float(dest_center[0] - state.loc.x)
        dy = float(dest_center[1] - state.loc.y)

        # If agent is almost at slot center, approach direction is ill-defined.
        if dx * dx + dy * dy < 1e-8:
            return 0.0

        approach_dir = math.atan2(dy, dx)
        return self._get_angle_diff(approach_dir, self.map.dest.heading)

    def _lerp(self, x: float, x0: float, x1: float, y0: float, y1: float) -> float:
        if abs(x1 - x0) < 1e-9:
            return float(y1)
        t = (x - x0) / (x1 - x0)
        t = max(0.0, min(1.0, t))
        return float(y0 + t * (y1 - y0))

    def _estimate_angle_gate_distance_ref(self) -> float:
        """Estimate distance reference from corridor size and central open-space size."""
        level = getattr(self, "level", MAP_LEVEL)
        corridor_span = 0.5 * (
            float(PARA_PARK_WALL_DIST_DICT.get(level, PARA_PARK_WALL_DIST_DICT['Normal']))
            + float(BAY_PARK_WALL_DIST_DICT.get(level, BAY_PARK_WALL_DIST_DICT['Normal']))
        )

        max_len = float(MAX_PARK_LOT_LEN_DICT.get(level, MAX_PARK_LOT_LEN_DICT['Normal']))
        max_wid = float(MAX_PARK_LOT_WIDTH_DICT.get(level, WIDTH + 0.6))
        central_open = 0.5 * (max(max_len - LENGTH, 0.2) + max(max_wid - WIDTH, 0.2))

        # corridor controls global maneuver radius; central_open controls slot vicinity freedom
        dist_ref = 0.18 * corridor_span + 1.5 * central_open
        return float(max(2.5, dist_ref))

    def _angle_distance_gate(self, curr_state: State) -> float:
        """Distance gate for angle rewards: peak in mid-range interval, slightly smaller near slot."""
        if not bool(ANGLE_REWARD_DIST_GATE_ENABLE):
            return 1.0

        dist = float(curr_state.loc.distance(self.map.dest.loc))
        dist_ref = self._estimate_angle_gate_distance_ref()

        near_scale = float(ANGLE_REWARD_DIST_GATE_NEAR_SCALE)
        far_scale = float(ANGLE_REWARD_DIST_GATE_FAR_SCALE)
        peak_scale = float(ANGLE_REWARD_DIST_GATE_PEAK_SCALE)

        near_end = float(ANGLE_REWARD_DIST_GATE_NEAR_END_REF_RATIO) * dist_ref
        mid_low = float(ANGLE_REWARD_DIST_GATE_MID_LOW_REF_RATIO) * dist_ref
        mid_high = float(ANGLE_REWARD_DIST_GATE_MID_HIGH_REF_RATIO) * dist_ref
        far_start = float(ANGLE_REWARD_DIST_GATE_FAR_START_REF_RATIO) * dist_ref

        # ensure monotonic boundaries
        mid_low = max(mid_low, near_end + 0.05)
        mid_high = max(mid_high, mid_low + 0.05)
        far_start = max(far_start, mid_high + 0.05)

        if dist <= near_end:
            return near_scale
        if dist <= mid_low:
            return self._lerp(dist, near_end, mid_low, near_scale, peak_scale)
        if dist <= mid_high:
            return peak_scale
        if dist <= far_start:
            return self._lerp(dist, mid_high, far_start, peak_scale, far_scale)
        return far_scale

    def _get_reward_info(self, prev_state: State, curr_state: State) -> OrderedDict:
        """HOPE-style per-step reward components (deltas)."""
        # Time penalty ramp-up:
        # start with a small penalty at episode beginning, then increase to
        # the original constant scale (typically -1.0).
        if bool(TIME_COST_RAMP_ENABLE):
            ramp_steps = max(1.0, float(TOLERANT_TIME) * float(TIME_COST_RAMP_RATIO))
            p = min(max(float(self.t) / ramp_steps, 0.0), 1.0)
            curr_scale = float(TIME_COST_INIT_SCALE) + (
                float(TIME_COST_FINAL_SCALE) - float(TIME_COST_INIT_SCALE)
            ) * p
            time_cost = -float(curr_scale)
        else:
            time_cost = -1.0
        rs_dist_reward = 0.0

        # Distance progress reward (normalized)
        dist_diff = float(curr_state.loc.distance(self.map.dest.loc))
        prev_dist_diff = float(prev_state.loc.distance(self.map.dest.loc))
        dist_norm_ratio = max(self.initial_dist, 10.0)
        dist_reward = prev_dist_diff / dist_norm_ratio - dist_diff / dist_norm_ratio

        # Heading alignment progress reward (optional; keep weight at 0 by default)
        angle_diff = self._get_angle_diff(curr_state.heading, self.map.dest.heading)
        prev_angle_diff = self._get_angle_diff(prev_state.heading, self.map.dest.heading)
        angle_norm_ratio = math.pi
        angle_reward = prev_angle_diff / angle_norm_ratio - angle_diff / angle_norm_ratio

        # Approach-direction alignment progress reward:
        # angle between (agent->slot-center direction) and slot heading.
        # Smaller angle is better.
        approach_angle_diff = self._get_slot_approach_dir_diff(curr_state)
        prev_approach_angle_diff = self._get_slot_approach_dir_diff(prev_state)
        approach_angle_reward = (
            prev_approach_angle_diff / angle_norm_ratio
            - approach_angle_diff / angle_norm_ratio
        )

        angle_gate = self._angle_distance_gate(curr_state)
        angle_reward *= angle_gate
        approach_angle_reward *= angle_gate

        # Box union reward (incremental IoU-like overlap, monotonic)
        front_box_ego = Polygon(curr_state.create_box()[0])
        front_box_dest = Polygon(self.map.dest.create_box()[0])
        inter = float(front_box_ego.intersection(front_box_dest).area)
        dest_area = float(front_box_dest.area) + 1e-9
        # HOPE uses: inter / (2*dest - inter)
        box_union = inter / max(1e-9, (2.0 * dest_area - inter))
        if box_union < self.accum_arrive_reward:
            box_union_reward = 0.0
        else:
            prev_acc = self.accum_arrive_reward
            self.accum_arrive_reward = box_union
            box_union_reward = box_union - prev_acc

        return OrderedDict(
            {
                'time_cost': float(time_cost),
                'rs_dist_reward': float(rs_dist_reward),
                'dist_reward': float(dist_reward),
                'angle_reward': float(angle_reward),
                'approach_angle_reward': float(approach_angle_reward),
                'box_union_reward': float(box_union_reward),
            }
        )

    def _reward_shaping(self, status: Status, reward_info: OrderedDict) -> float:
        """Aggregate scalar reward from reward_info (CONTINUE) or fixed terminal rewards."""
        if status == Status.CONTINUE:
            r = 0.0
            for k, w in REWARD_WEIGHT.items():
                r += float(w) * float(reward_info.get(k, 0.0))
        elif status == Status.OUTBOUND:
            r = -50.0
        elif status == Status.OUTTIME:
            r = -1.0
        elif status == Status.ARRIVED:
            r = 50.0
        elif status == Status.COLLIDED:
            r = -50.0
        else:
            r = 0.0
        return float(r) * float(REWARD_RATIO)

    def get_reward(self, action, prev_state: State = None):
        """Return (reward, done, info) in Gymnasium style.

        NOTE: when action is None (reset probing), reward is 0.
        """
        info = {}

        if action is None:
            info['status'] = Status.CONTINUE
            info['reward_info'] = OrderedDict({k: 0.0 for k in REWARD_WEIGHT.keys()})
            info['path_to_dest'] = None
            return 0.0, False, info

        # Determine status (collision / outbound / arrived / outtime)
        status = Status.CONTINUE

        # Collision with obstacles
        is_collision = False
        vehicle_boxes = self.vehicle.boxes  # [front, rear]
        for box in vehicle_boxes:
            for obst in self.map.obstacles:
                if box.intersects(obst.shape):
                    is_collision = True
                    break
            if is_collision:
                break

        # Out of map
        if not is_collision:
            x, y = self.vehicle.state.loc.x, self.vehicle.state.loc.y
            if x < self.map.xmin or x > self.map.xmax or y < self.map.ymin or y > self.map.ymax:
                status = Status.OUTBOUND

        if is_collision:
            status = Status.COLLIDED

        # Success check (same threshold as before)
        if status == Status.CONTINUE:
            heading_diff = self._get_angle_diff(self.vehicle.state.heading, self.map.dest.heading)
            front_box_ego = Polygon(self.vehicle.boxes[0])
            front_box_dest = Polygon(self.map.dest.create_box()[0])
            intersection_area = float(front_box_ego.intersection(front_box_dest).area)
            overlap_ratio = intersection_area / (float(front_box_dest.area) + 1e-9)
            if heading_diff < float(np.deg2rad(15)) and overlap_ratio > 0.7:
                status = Status.ARRIVED

        if status == Status.CONTINUE and self.t >= TOLERANT_TIME:
            status = Status.OUTTIME

        done = status != Status.CONTINUE

        # Reward info (only meaningful during CONTINUE)
        if prev_state is None:
            prev_state = self.vehicle.state
        reward_info = self.get_reward_info(status, prev_state)

        reward = self._reward_shaping(status, reward_info)

        info['status'] = status
        info['reward_info'] = reward_info
        info['path_to_dest'] = None

        return reward, done, info

    def get_reward_info(self, status: Status, prev_state: State) -> OrderedDict:
        if status != Status.CONTINUE:
            return OrderedDict({k: 0.0 for k in REWARD_WEIGHT.keys()})
        return self._get_reward_info(prev_state, self.vehicle.state)

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((WIN_W, WIN_H))
            else:
                self.screen = pygame.Surface((WIN_W, WIN_H))
        
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill(BG_COLOR)
        
        # Draw obstacles
        for obst in self.map.obstacles:
            self._draw_polygon(obst.shape, OBSTACLE_COLOR)
            
        # Draw dest
        self._draw_polygon(self.map.dest_box, DEST_COLOR)
        
        # Draw vehicle
        # Front
        self._draw_polygon(self.vehicle.boxes[0], self.vehicle.color)
        # Rear
        self._draw_polygon(self.vehicle.boxes[1], self.vehicle.color)
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.fps)
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _draw_polygon(self, polygon, color):
        def _to_screen(coords):
            transformed = []
            for x, y in coords:
                sx = x * self.matrix[0] + self.matrix[4]
                sy = y * self.matrix[3] + self.matrix[5]
                transformed.append((sx, sy))
            return transformed

        if isinstance(polygon, LinearRing):
            pygame.draw.polygon(self.screen, color, _to_screen(list(polygon.coords)))
            return

        if isinstance(polygon, Polygon):
            # Draw exterior (solid)
            pygame.draw.polygon(self.screen, color, _to_screen(list(polygon.exterior.coords)))
            # Carve holes using background color so drivable space is visible
            for interior in list(polygon.interiors):
                pygame.draw.polygon(self.screen, BG_COLOR, _to_screen(list(interior.coords)))
            return

        if isinstance(polygon, MultiPolygon):
            for g in polygon.geoms:
                self._draw_polygon(g, color)
            return

        # Unsupported geometry type
        return

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.is_open = False
