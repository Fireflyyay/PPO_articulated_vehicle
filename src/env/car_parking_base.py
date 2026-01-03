
'''
This is a collision-free env which only includes one parking case with random start position.
'''


import sys
sys.path.append("../")
from typing import Optional, Union
import math
from typing import OrderedDict
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from heapdict import heapdict
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

        if self.level in ['Normal', 'Complex', 'Extrem']:
            self.map = ParkingMapNormal(self.level)
        # elif self.level == 'dlp':
        #     self.map = ParkingMapDLP()
        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH, articulated=True, trailer_length= TRAILER_LENGTH, hitch_offset=HITCH_OFFSET)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.reward = 0.0
        self.prev_reward = 0.0
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
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM))*LIDAR_RANGE
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM,), dtype=np.float64
            )
        low_bound = np.array([0,-1,-1,-1,-1,-1,-1])
        high_bound = np.array([MAX_DIST_TO_DEST,1,1,1,1,1,1])
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float64
        )
        
        self.observation_shape = {
            'lidar': (LIDAR_NUM,),
            'target': (self.tgt_repr_size,),
        }
    
    def set_level(self, level:str=None):
        if level is None:
            self.map = ParkingMapNormal()
            return
        self.level = level
        if self.level in ['Normal', 'Complex', 'Extrem',]:
            self.map = ParkingMapNormal(self.level)
        # elif self.level == 'dlp':
        #     self.map = ParkingMapDLP()

    def reset(self, seed: int = None, options: dict = None) -> tuple:
        super().reset(seed=seed)
        
        case_id = options.get('case_id') if options else None
        data_dir = options.get('data_dir') if options else None
        level = options.get('level') if options else None

        self.reward = 0.0
        self.prev_reward = 0.0
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
        
        obs, _, _, _, info = self.step(None)
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

    def step(self, action: np.ndarray = None):
        if action is not None:
            self.vehicle.step(action)
            self.t += 1
        
        # get observation
        obs = {}
        if self.use_lidar_observation:
            obs['lidar'] = self.lidar.get_observation(self.vehicle.state, self.map.obstacles)
        
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
        
        obs['target'] = np.array([
            dist/MAX_DIST_TO_DEST,
            math.cos(relative_angle),
            math.sin(relative_angle),
            math.cos(relative_heading),
            math.sin(relative_heading),
            math.cos(articulation_angle),
            math.sin(articulation_angle)
        ])
        
        # calculate reward
        reward, done, info = self.get_reward(action)
        self.reward = reward
        
        terminated = False
        truncated = False
        
        if done:
            if info['status'] == Status.OUTTIME:
                truncated = True
            else:
                terminated = True
        
        return obs, reward, terminated, truncated, info

    def get_reward(self, action):
        reward = 0.0
        done = False
        info = {}
        
        # 1. Time cost
        reward -= REWARD_WEIGHT['time_cost'] * 0.01
        
        # 2. Distance to target
        dist = self.vehicle.state.loc.distance(Point(self.map.dest.loc))
        if self.prev_reward == 0.0:
            self.prev_reward = dist
        
        dist_reward_val = (self.prev_reward - dist) * REWARD_WEIGHT['dist_reward']
        reward += dist_reward_val
        self.prev_reward = dist
        
        # 2.5 Angle reward
        heading_diff = self.vehicle.state.heading - self.map.dest.heading
        # Normalize to [-pi, pi]
        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
        angle_reward = math.cos(heading_diff) * REWARD_WEIGHT['angle_reward'] * 0.01
        reward += angle_reward
        
        # 3. Collision
        # Check collision with obstacles
        is_collision = False
        vehicle_boxes = self.vehicle.boxes # [front, rear]
        for box in vehicle_boxes:
            for obst in self.map.obstacles:
                if box.intersects(obst.shape):
                    is_collision = True
                    break
            if is_collision:
                break
        
        # Check out of map
        if not is_collision:
             if self.vehicle.state.loc.x < self.map.xmin or self.vehicle.state.loc.x > self.map.xmax or \
                self.vehicle.state.loc.y < self.map.ymin or self.vehicle.state.loc.y > self.map.ymax:
                 is_collision = True
                 info['status'] = Status.OUTBOUND

        if is_collision:
            reward -= REWARD_WEIGHT['out_of_map_penalty'] # Using out_of_map_penalty for collision too
            done = True
            info['status'] = Status.COLLIDED
        
        # 4. Success
        # Check if arrived
        # Arrived if angle diff < 10 degrees and front box overlap > 80%
        
        heading_diff_abs = abs(heading_diff)
        
        # Calculate overlap of front parts
        front_box_ego = Polygon(self.vehicle.boxes[0])
        front_box_dest = Polygon(self.map.dest.create_box()[0])
        
        intersection_area = front_box_ego.intersection(front_box_dest).area
        overlap_ratio = intersection_area / front_box_dest.area
        
        if heading_diff_abs < np.radians(10) and overlap_ratio > 0.8:
            reward += REWARD_WEIGHT['box_union_reward']
            done = True
            info['status'] = Status.ARRIVED
        
        # 5. Turn penalty (smoothness)
        if action is not None:
            # action[0] is steering/articulation rate
            reward -= abs(action[0]) * REWARD_WEIGHT['turn_penalty']

        if self.t >= TOLERANT_TIME:
            done = True
            info['status'] = Status.OUTTIME
            
        if 'status' not in info:
            info['status'] = Status.CONTINUE
            
        info['reward_info'] = {
            'time_cost': -REWARD_WEIGHT['time_cost'] * 0.01,
            'dist_reward': dist_reward_val,
            'angle_reward': angle_reward,
            'collision': -REWARD_WEIGHT['out_of_map_penalty'] if is_collision else 0,
            'success': REWARD_WEIGHT['box_union_reward'] if info['status'] == Status.ARRIVED else 0
        }
        info['path_to_dest'] = None # No RS path

        return reward, done, info

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
        if isinstance(polygon, LinearRing):
            coords = list(polygon.coords)
        elif isinstance(polygon, Polygon):
            coords = list(polygon.exterior.coords)
        else:
            return
            
        transformed_coords = []
        for x, y in coords:
            # Transform to screen coordinates
            # matrix: [scale, 0, 0, -scale, x_off, y_off]
            sx = x * self.matrix[0] + self.matrix[4]
            sy = y * self.matrix[3] + self.matrix[5]
            transformed_coords.append((sx, sy))
            
        pygame.draw.polygon(self.screen, color, transformed_coords)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.is_open = False
