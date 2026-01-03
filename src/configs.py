
import os
os.environ["SDL_VIDEODRIVER"]="dummy"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 42

#########################
# vehicle
WHEEL_BASE = 4.0  # wheelbase (HITCH_OFFSET + TRAILER_LENGTH)
FRONT_HANG = 2.5  # front hang length
REAR_HANG = 3.0  # rear hang length
WIDTH = 3.0  # width
TRAILER_LENGTH = 2.0
HITCH_OFFSET = 2.0
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

NUM_STEP = 10
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
    'Extrem':8,
    'Complex':5,
    'Normal':3,
}

# Normal level
MIN_DIST_TO_OBST = 0.1
MAX_DRIVE_DISTANCE = 150.0
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
TOLERANT_TIME = 1000
USE_LIDAR = True
USE_IMG = False # Disabled as requested
USE_ACTION_MASK = False # Disabled as requested
# Increased for longer navigation scenarios (was 200, now supports up to 150m)
MAX_DIST_TO_DEST = 300
K = 2.0 # the render scale adjusted for larger map (240px / 120m = 2.0)
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
GAMMA = 0.98
BATCH_SIZE = 2048  # Batch size is originally 8192
LR = 5e-6
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
    'n_modal':2+int(USE_IMG)+int(USE_ACTION_MASK),
    'lidar_shape':LIDAR_NUM,
    'target_shape':7,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,64,64) if USE_IMG else None,
    'output_size':2,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':True,
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

CRITIC_CONFIGS = {
    'n_modal':2+int(USE_IMG)+int(USE_ACTION_MASK),
    'lidar_shape':LIDAR_NUM,
    'target_shape':7,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,64,64) if USE_IMG else None,
    'output_size':1,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':False,
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

REWARD_RATIO = 0.1
from typing import OrderedDict
REWARD_WEIGHT = OrderedDict({'time_cost':1,\
            'rs_dist_reward':0,\
            'dist_reward':1,\
            'angle_reward':2,\
            'box_union_reward':10,\
            'out_of_map_penalty':10,\
            'turn_penalty':0.0001,})


CONFIGS_ACTION = {
    'use_tanh_activate': True,
    'hidden_size':256,
    'lidar_shape':LIDAR_NUM,
    'n_hidden_layers':4,
    'n_action':len(discrete_actions),
    'discrete_actions':discrete_actions
}

VISUALIZATION_NUM = 10
