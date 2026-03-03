
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

# -----------------------------
# Soft global guidance (reset-time A* + step-time directional hint)
# Guidance is advisory only and never a hard waypoint constraint.
# -----------------------------
ENABLE_GLOBAL_SOFT_GUIDANCE = True
GUIDANCE_FEATURE_DIM = 4  # [u_x_soft, u_y_soft, lateral_err_soft, hint_strength]
GUIDANCE_GRID_RESOLUTION = 1.0
GUIDANCE_OBS_INFLATION = 1.0
GUIDANCE_MAP_MARGIN = 0.5
GUIDANCE_LOOKAHEAD_BASE = 4.5
GUIDANCE_LOOKAHEAD_SPEED_GAIN = 1.5
GUIDANCE_LOOKAHEAD_MIN = 3.0
GUIDANCE_LOOKAHEAD_MAX = 8.0
GUIDANCE_PROGRESS_WINDOW = 40
GUIDANCE_MIN_CLEARANCE_M = 1.2
GUIDANCE_FULL_CLEARANCE_M = 4.0
GUIDANCE_NEAR_OBS_DIST_M = 2.0
GUIDANCE_MAX_DENSE_RATIO = 0.35

FPS = 100
TOLERANT_TIME = 1000 # Increased from 200 to 1000 to match finer control frequency (0.2s * 1000 = 200s total duration)
USE_LIDAR = True
USE_IMG = False # Disabled as requested
USE_ACTION_MASK = True # Disabled as requested
# Increased for longer navigation scenarios (was 200, now supports up to 150m)
MAX_DIST_TO_DEST = 70.0
K = 4.0 # the render scale adjusted for smaller map (480px / 120m -> 4)
RS_MAX_DIST = 50
RENDER_TRAJ = True

# action mask
# mode: "fast_only" | "hybrid" | "full"
# - fast_only: only use grid-index fast prune result as mask (fastest)
# - hybrid: fast prune + precise simulation on candidates (default)
# - full: precise simulation on all actions (slowest, most conservative)
ACTION_MASK_MODE = "fast_only"
# Recompute action mask every K macro-steps; reuse last mask in between.
ACTION_MASK_UPDATE_EVERY_K = 2

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
GRID_RESOLUTION = 0.6  # used by offline index builder; runtime reads from index
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

BATCH_SIZE = 2048  # Reduced for Trmore frequent updates
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
    'input_dim': LIDAR_NUM + 7 + 2 + (GUIDANCE_FEATURE_DIM if ENABLE_GLOBAL_SOFT_GUIDANCE else 0), # Lidar + Target + Velocity + Guidance
    'hidden_size': 400,
    'output_size': 2,
    'use_tanh_output': True,
    'orthogonal_init': True,
}

CRITIC_CONFIGS = {
    'input_dim': LIDAR_NUM + 7 + 2 + (GUIDANCE_FEATURE_DIM if ENABLE_GLOBAL_SOFT_GUIDANCE else 0),
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
    # Angle-1: agent heading vs slot heading
    'angle_reward': 0,
    # Angle-2: (agent->slot direction) vs slot heading
    'approach_angle_reward': 2,
    'box_union_reward': 10,
})

# Time-cost ramp (used by CarParking._get_reward_info)
# Keep disabled by default to preserve legacy constant time penalty (-1.0).
TIME_COST_RAMP_ENABLE = False
TIME_COST_RAMP_RATIO = 0.25
TIME_COST_INIT_SCALE = 0.2
TIME_COST_FINAL_SCALE = 1.0

# Distance gate for angle rewards (both angle_reward and approach_angle_reward)
# Goal: avoid frequent micro-adjustment near slot center while keeping alignment useful in mid-range.
# Distances are scaled by an auto-estimated reference from:
# - corridor span (BAY/PARA wall distances)
# - central open-space around slot (lot size surplus over vehicle size)
ANGLE_REWARD_DIST_GATE_ENABLE = True
ANGLE_REWARD_DIST_GATE_NEAR_SCALE = 0.65
ANGLE_REWARD_DIST_GATE_PEAK_SCALE = 1.00
ANGLE_REWARD_DIST_GATE_FAR_SCALE = 0.50

# Mid-distance peak interval (plateau): [MID_LOW, MID_HIGH] * dist_ref
ANGLE_REWARD_DIST_GATE_NEAR_END_REF_RATIO = 0.55
ANGLE_REWARD_DIST_GATE_MID_LOW_REF_RATIO = 0.90
ANGLE_REWARD_DIST_GATE_MID_HIGH_REF_RATIO = 1.35
ANGLE_REWARD_DIST_GATE_FAR_START_REF_RATIO = 2.30


CONFIGS_ACTION = {
    'use_tanh_activate': True,
    'hidden_size':256,
    'lidar_shape':LIDAR_NUM,
    'n_hidden_layers':4,
    'n_action':len(discrete_actions),
    'discrete_actions':discrete_actions
}

VISUALIZATION_NUM = 10

# =============================================================
# Adaptive Primitive Mining / Sustainable Incremental Learning
# =============================================================
# Default is OFF to keep existing training stable.

# ===== Adaptive Primitive Mining =====
USE_ADAPTIVE_PRIMITIVE_EXPANSION = True

# Triggering / scheduling
AP_WARMUP_EPISODES = 1000
AP_TRIGGER_SUCCESS_RATE = 0.40
AP_TRIGGER_HARD_SUCCESS_RATE = 0.15
AP_TRIGGER_WINDOW = 200
AP_COOLDOWN_EPISODES = 500

# Mining / extraction
AP_MINING_ROLLOUTS = 120
AP_MINING_DETERMINISTIC = True

# Segmenting
AP_SEGMENT_H_MIN = 6
AP_SEGMENT_H_MAX = 24
AP_SEGMENT_STRIDE = 2
AP_EVENT_WINDOW_BEFORE = 4
AP_EVENT_WINDOW_AFTER = 8

# Scoring thresholds
AP_COMPLEXITY_THRESH = 0.35
AP_UTILITY_THRESH = 0.25
AP_NOVELTY_THRESH = 0.20
AP_FINAL_TOPK = 40

# Score weights
AP_W_COMPLEXITY = 0.35
AP_W_UTILITY = 0.40
AP_W_NOVELTY = 0.25

# Add / cap
AP_MAX_ADD_PER_ROUND = 12
AP_MAX_LIBRARY_SIZE = 256

# Dedup / novelty (brute-force, no extra deps)
AP_DEDUP_ACTION_L2_TAU = 0.35  # normalized by sqrt(H*action_dim)
AP_NOVELTY_ACTION_L2_SCALE = 1.0  # scale for novelty normalization

# Proxy pruning / validation
AP_ENABLE_PROXY_PRUNING = True
AP_PROXY_PRUNE_TOPN = 20
AP_VALIDATION_EPISODES = 60
AP_ENABLE_ROLLBACK = True
AP_ROLLBACK_DROP_TOL = 0.05

# Trace buffer for mining (stores successful / near-success episodes)
AP_TRACE_BUFFER_MAX_EPISODES = 600
AP_TRACE_KEEP_SUCCESS_ONLY = True
AP_TRACE_KEEP_NEAR_SUCCESS = True
AP_NEAR_SUCCESS_DIST_THR = 3.0  # meters (from obs)

# Complexity proxy parameters
AP_V_TH = 0.1
AP_D0_OBS = 3.0
AP_D_TERM = 10.0

AP_COMPLEXITY_WEIGHTS = {
    "rev": 0.15,
    "steer_var": 0.20,
    "switch": 0.20,
    "curv": 0.15,
    "art": 0.15,
    "obs": 0.15,
}

# ===== Reward update from discovered primitives =====
USE_DISCOVERED_PRIMITIVE_SHAPING = True
DP_SHAPING_COEF = 0.02
DP_SHAPING_SIGMA = 1.0
DP_MAX_CENTROIDS = 64

# ===== Post-expansion stabilization =====
AP_POST_EXPAND_FREEZE_EPISODES = 100
AP_POST_EXPAND_LR_SCALE = 0.3
AP_NEW_ACTION_LOGIT_BIAS_INIT = 1.5
AP_NEW_ACTION_LOGIT_BIAS_DECAY_EPISODES = 200

