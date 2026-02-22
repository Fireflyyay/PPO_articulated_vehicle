
import os
os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42

#########################
# vehicle
WHEEL_BASE = 3.0  # wheelbase (HITCH_OFFSET + TRAILER_LENGTH)
FRONT_HANG = 1.0  # front hang length
REAR_HANG = 1.0  # rear hang length
WIDTH = 2.0  # width
TRAILER_LENGTH = 1.5
HITCH_OFFSET = 1.5
LENGTH = FRONT_HANG + WHEEL_BASE + REAR_HANG  # total length

from shapely.geometry import LinearRing
VehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])

# Defined based on vehicle dimensions for articulated vehicle
FrontVehicleBox = LinearRing([
    (-HITCH_OFFSET, -WIDTH/2), 
    (FRONT_HANG, -WIDTH/2), 
    (FRONT_HANG,  WIDTH/2),
    (-HITCH_OFFSET,  WIDTH/2)])

RearVehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH/2), 
    (TRAILER_LENGTH, -WIDTH/2), 
    (TRAILER_LENGTH,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])

COLOR_POOL = [
    (30, 144, 255, 255), # dodger blue
    (255, 127, 80, 255), # coral
    (255, 215, 0, 255) # gold
]

VALID_SPEED = [-2.5, 2.5]
VALID_STEER = [-np.radians(36), np.radians(36)]
VALID_ACCEL = [-1.0, 1.0]
VALID_ANGULAR_SPEED = [-0.5, 0.5]

NUM_STEP = 4 # Reduced from 20 to 4 for finer control (0.2s per step)
STEP_LENGTH = 5e-2

########################
# senerio
# Use 'Navigation' for tactics2d training instead of parking scenarios
MAP_LEVEL = 'Normal' # ['Normal', 'Complex', 'Extrem']
MIN_PARK_LOT_LEN_DICT = {'Extrem':LENGTH+0.6,
                            'Complex':LENGTH+0.9,
                            'Normal':LENGTH*1.25,}
MAX_PARK_LOT_LEN_DICT = {'Extrem':LENGTH+0.9,
                            'Complex':LENGTH*1.25,
                            'Normal':LENGTH*1.25+0.5}
MIN_PARK_LOT_WIDTH_DICT = {
    'Complex':WIDTH+0.4,
    'Normal':WIDTH+0.85,
}
MAX_PARK_LOT_WIDTH_DICT = {
    'Complex':WIDTH+0.85,
    'Normal':WIDTH+1.2,
}
PARA_PARK_WALL_DIST_DICT = {
    'Extrem':15.0,
    'Complex':20.0,
    'Normal':25.0,
}
BAY_PARK_WALL_DIST_DICT = {
    'Extrem':15.0,
    'Complex':20.0,
    'Normal':25.0,
}
N_OBSTACLE_DICT = {
    'Extrem':15,
    'Complex':10,
    'Normal':8,
}

# Normal level
MIN_DIST_TO_OBST = 0.1
MAX_DRIVE_DISTANCE = 60.0
DROUP_OUT_OBST = 0.0

#########################
# env
ENV_COLLIDE = False
BG_COLOR = (255, 255, 255, 255)
START_COLOR = (100, 149, 237, 255)
DEST_COLOR = (69, 139, 0, 255)
OBSTACLE_COLOR = (150, 150, 150, 255)
TRAJ_COLOR_HIGH = (10, 10, 200, 255)
TRAJ_COLOR_LOW = (10, 10, 10, 255)
TRAJ_RENDER_LEN = 20
TRAJ_COLORS = list(map(tuple,np.linspace(\
    np.array(TRAJ_COLOR_LOW), np.array(TRAJ_COLOR_HIGH), TRAJ_RENDER_LEN, endpoint=True, dtype=np.uint8)))
OBS_W = 256
OBS_H = 256
VIDEO_W = 600
VIDEO_H = 400
WIN_W = 500
WIN_H = 500
LIDAR_RANGE = 30.0
LIDAR_NUM = 120

FPS = 100
TOLERANT_TIME = 1000 # Increased from 200 to 1000 to match finer control frequency (0.2s * 1000 = 200s total duration)
USE_LIDAR = True
USE_IMG = False # Disabled as requested
USE_ACTION_MASK = False # Disabled as requested
# Increased for longer navigation scenarios (was 200, now supports up to 150m)
MAX_DIST_TO_DEST = 70.0
K = 4.0 # the render scale adjusted for smaller map (480px / 120m -> 4)
RS_MAX_DIST = 50
RENDER_TRAJ = True

# action mask
PRECISION = 10
step_speed = 1
discrete_actions = []
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, step_speed])
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, -step_speed])
N_DISCRETE_ACTION = len(discrete_actions)

#########################
# model
GAMMA_BASE = 0.98
# GAMMA will be updated based on primitive H if used
# GAMMA = 0.98 

USE_MOTION_PRIMITIVES = True
PRIMITIVE_H = 1
PRIMITIVE_STEER_LEVELS = 11
PRIMITIVE_LIBRARY_PATH = "../data/primitives_articulated_H4_S11.npz"

# -----------------------------
# Terminal Takeover (Paper-style RHP planner)
# -----------------------------
# Enable the receding-horizon takeover planner that uses an offline grid index
# for fast online pruning (no per-primitive rollout in takeover stage).
TAKEOVER_USE_RHP = True

# Dynamic trigger + hysteresis
TAKEOVER_DIST_BASE = 10.0
TAKEOVER_DIST_HYSTERESIS = 2.0
TAKEOVER_DIST_SPEED_GAIN = 0.0  # meters per (m/s) of |v|
TAKEOVER_DIST_OBS_DENSITY_GAIN = 0.0  # meters per obstacle-density (0..1)

# Early takeover difficulty triggers
TAKEOVER_EARLY_HEADING_ERR = float(np.deg2rad(35))
TAKEOVER_EARLY_ARTICULATION = float(np.deg2rad(25))
TAKEOVER_EARLY_MIN_LIDAR = 2.0  # meters

# Occupancy + index settings
GRID_RESOLUTION = 0.3  # used by offline index builder; runtime reads from index
OCCUPANCY_INFLATION_RADIUS = 1.8  # meters, approx vehicle envelope + margin

# Group scoring / prefix execution
TAKEOVER_GROUP_SCORE_TOPK = 5
TAKEOVER_MAX_PREFIX_STEPS = None  # cap prefix steps; None means use index value

TAKEOVER_SCORE_WEIGHTS = {
    "dist": 1.0,
    "dir": 0.5,
    "state": 0.2,
    "smooth": 0.2,
    "speed": 0.1,
    "clearance": 0.3,
}

# No-path fallback
TAKEOVER_FALLBACK_OLD_PLANNER = False

# Training consistency switch
# True: takeover transitions are still pushed into PPO buffer (imitation-like on-policy shaping)
# False: skip storing takeover transitions to reduce bias toward hand-coded planner.
TAKEOVER_TEACHER_FORCING = True

# Profiling switch (planner/wrapper will emit timing in info['takeover_debug'])
TAKEOVER_PROFILE = True

if USE_MOTION_PRIMITIVES:
    GAMMA = GAMMA_BASE ** PRIMITIVE_H
else:
    GAMMA = GAMMA_BASE

BATCH_SIZE = 2048  # Reduced for more frequent updates
LR = 1e-4
TAU = 0.1
MAX_TRAIN_STEP = 1e6
ORTHOGONAL_INIT = True
LR_DECAY = False
UPDATE_IMG_ENCODE = False


C_CONV = [4, 8,]
SIZE_FC = [256]

ATTENTION_CONFIG = {
                'depth': 1,
                'heads': 8,
                'dim_head': 32,
                'mlp_dim': 128,
                'hidden_dim': 128,
    }
USE_ATTENTION = True

ACTOR_CONFIGS = {
    'input_dim': LIDAR_NUM + 7 + 2, # Lidar + Target + Velocity
    'hidden_size': 400,
    'output_size': 2,
    'use_tanh_output': True,
    'orthogonal_init': True,
}

CRITIC_CONFIGS = {
    'input_dim': LIDAR_NUM + 7 + 2,
    'hidden_size': 400,
    'output_size': 1,
    'use_tanh_output': False,
    'orthogonal_init': True,
}

REWARD_RATIO = 0.1
from collections import OrderedDict

# HOPE-style reward shaping:
# - env returns `reward_info` (per-step deltas)
# - scalar reward is computed as weighted sum (when CONTINUE),
#   otherwise fixed terminal rewards are used.
REWARD_WEIGHT = OrderedDict({
    'time_cost': 1,
    'rs_dist_reward': 0,
    'dist_reward': 5,
    'angle_reward': 0,
    'box_union_reward': 10,
})


CONFIGS_ACTION = {
    'use_tanh_activate': True,
    'hidden_size':256,
    'lidar_shape':LIDAR_NUM,
    'n_hidden_layers':4,
    'n_action':len(discrete_actions),
    'discrete_actions':discrete_actions
}

VISUALIZATION_NUM = 10
