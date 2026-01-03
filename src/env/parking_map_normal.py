
import sys
sys.path.append("../")
from math import pi, cos, sin
import os

import numpy as np
from numpy.random import randn, random
from typing import List
from shapely.geometry import LinearRing, Point, MultiPoint, Polygon, LineString

from env.vehicle import State
from env.map_base import *
from configs import *

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

# params for generating parking case
prob_huge_obst = 0.5
n_non_critical_car = 3
prob_non_critical_car = 0.7

def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = randn()*std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = random()*(clip_high - clip_low) + clip_low
    return rand_num

def get_rand_pos(origin_x, origin_y, angle_min, angle_max, radius_min, radius_max):
    angle_mean = (angle_max+angle_min)/2
    angle_std = (angle_max-angle_min)/4
    angle_rand = random_gaussian_num(angle_mean, angle_std, angle_min, angle_max)
    radius_rand = random_gaussian_num((radius_min+radius_max)/2, (radius_max-radius_min)/4, radius_min, radius_max)
    return origin_x+cos(angle_rand)*radius_rand, origin_y+sin(angle_rand)*radius_rand

def generate_bay_parking_case(map_level):
    '''
    Generate the parameters that a bay parking case need.
    
    Returns
    ----------
        `start` (list): [x, y, yaw]
        `dest` (list): [x, y, yaw]
        `obstacles` (list): [ obstacle (`LinearRing`) , ...]
    '''
    origin = (0., 0.)
    bay_half_len = 15.
    # params related to map level
    max_BAY_PARK_LOT_WIDTH = MAX_PARK_LOT_WIDTH_DICT[map_level]
    min_BAY_PARK_LOT_WIDTH = MIN_PARK_LOT_WIDTH_DICT[map_level]
    bay_PARK_WALL_DIST = BAY_PARK_WALL_DIST_DICT[map_level]
    n_obst = N_OBSTACLE_DICT[map_level]
    max_lateral_space = max_BAY_PARK_LOT_WIDTH - WIDTH
    min_lateral_space = min_BAY_PARK_LOT_WIDTH - WIDTH

    generate_success = True
    # generate obstacle on back
    obstacle_back = LinearRing(( 
        (origin[0]+bay_half_len, origin[1]),
        (origin[0]+bay_half_len, origin[1]-3), 
        (origin[0]-bay_half_len, origin[1]-3), 
        (origin[0]-bay_half_len, origin[1])))
    
    if DEBUG:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlim(-20,20)
        ax.set_ylim(-20,20)
        plt.axis('off')

    # generate dest
    dest_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
    rb, _, _, lb  = list(State([origin[0], origin[1], dest_yaw, 0, 0]).create_box()[0].coords)[:-1]
    min_dest_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = random_gaussian_num(min_dest_y+0.4, 0.2, min_dest_y, min_dest_y+0.8)
    car_rb, car_rf, car_lf, car_lb  = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box()[0].coords)[:-1]
    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))

    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list([car_rb, car_rf, car_lf, car_lb]), color='b'))

    non_critical_vehicle = []
    # generate obstacle on left
    # the obstacle can be another vehicle or just a simple obstacle
    if random()<prob_huge_obst: # generate simple obstacle
        max_dist_to_obst = max_lateral_space/5*4
        min_dist_to_obst = max_lateral_space/5*1
        left_obst_rf = get_rand_pos(*car_lf, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        left_obst_rb = get_rand_pos(*car_lb, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_left = LinearRing(( 
            left_obst_rf,
            left_obst_rb, 
            (origin[0]-bay_half_len, origin[1]), 
            (origin[0]-bay_half_len, left_obst_rf[1])))
    else: # generate another vehicle as obstacle on left
        max_dist_to_obst = max_lateral_space/5*4
        min_dist_to_obst = max_lateral_space/5*1
        left_car_x = origin[0] - (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        left_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
        rb, _, _, lb  = list(State([left_car_x, origin[1], left_car_yaw, 0, 0]).create_box()[0].coords)[:-1]
        min_left_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        left_car_y = random_gaussian_num(min_left_car_y+0.4, 0.2, min_left_car_y, min_left_car_y+0.8)
        obstacle_left = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()[0]

        # generate other parking vehicle
        for _ in range(n_non_critical_car):
            left_car_x -= (WIDTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
            left_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            left_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
            obstacle_left_ = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()[0]
            if random()<prob_non_critical_car:
                non_critical_vehicle.append(obstacle_left_)


    # generate obstacle on right
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = max(min_lateral_space-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    max_dist_to_obst = max(max_lateral_space-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    if random()<prob_huge_obst: # generate simple obstacle
        right_obst_lf = get_rand_pos(*car_rf, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        right_obst_lb = get_rand_pos(*car_rb, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_right = LinearRing(( 
            (origin[0]+bay_half_len, right_obst_lf[1]),
            (origin[0]+bay_half_len, origin[1]), 
            right_obst_lb, 
            right_obst_lf))
    else: # generate another vehicle as obstacle on right
        right_car_x = origin[0] + (WIDTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        right_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
        rb, _, _, lb  = list(State([right_car_x, origin[1], right_car_yaw, 0, 0]).create_box()[0].coords)[:-1]
        min_right_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        right_car_y = random_gaussian_num(min_right_car_y+0.4, 0.2, min_right_car_y, min_right_car_y+0.8)
        obstacle_right = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()[0]
        
        # generate other parking vehicle
        for _ in range(n_non_critical_car):
            right_car_x += (WIDTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
            right_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            right_car_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)
            obstacle_right_ = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()[0]
            if random()<prob_non_critical_car:
                non_critical_vehicle.append(obstacle_right_)

    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if dist_dest_to_right_obst+dist_dest_to_left_obst<min_lateral_space or \
        dist_dest_to_right_obst+dist_dest_to_left_obst>max_lateral_space or \
        dist_dest_to_left_obst<MIN_DIST_TO_OBST or \
        dist_dest_to_right_obst<MIN_DIST_TO_OBST:
        generate_success = False
    # check collision
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    obstacles.extend(non_critical_vehicle)
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles])+MIN_DIST_TO_OBST
    other_obstcales = []
    if random()<0.2: # in this case only a wall will be generate
        other_obstcales = [LinearRing(( 
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST), 
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1), 
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1)))]
    else:
        other_obstacle_range = LinearRing(( 
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y), 
        (origin[0]+bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+8), 
        (origin[0]-bay_half_len, bay_PARK_WALL_DIST+max_obstacle_y+8)))
        valid_obst_x_range = (origin[0]-bay_half_len+2, origin[0]+bay_half_len-2)
        valid_obst_y_range = (bay_PARK_WALL_DIST+max_obstacle_y+2, bay_PARK_WALL_DIST+max_obstacle_y+6)
        for _ in range(n_obst):
            obs_x = random_uniform_num(*valid_obst_x_range)
            obs_y = random_uniform_num(*valid_obst_y_range)
            obs_yaw = random()*pi*2
            obs_coords = np.array(State([obs_x, obs_y, obs_yaw, 0, 0]).create_box()[0].coords[:-1])
            obs = LinearRing(obs_coords+2.0*random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            obst_invalid = False
            for other_obs in other_obstcales:
                if obs.intersects(other_obs):
                    obst_invalid = True
                    break
            if obst_invalid:
                continue
            other_obstcales.append(obs)

    # merge two kind of obstacles
    obstacles.extend(other_obstcales)

    if DEBUG:
        for obs in obstacles:
            ax.add_patch(plt.Polygon(xy=list(obs.coords), color='gray'))

    
    # generate start position
    start_box_valid = False
    valid_start_x_range = (origin[0]-bay_half_len/2, origin[0]+bay_half_len/2)
    
    if map_level == 'Complex':
        valid_start_y_range = (max_obstacle_y+1, max_obstacle_y+1 + (bay_PARK_WALL_DIST-1)*0.6)
    elif map_level == 'Extrem':
        valid_start_y_range = (max_obstacle_y+1, max_obstacle_y+1 + (bay_PARK_WALL_DIST-1)*0.3)
    else:
        valid_start_y_range = (max_obstacle_y+1, bay_PARK_WALL_DIST+max_obstacle_y-1)

    while not start_box_valid:
        start_box_valid = True
        start_x = random_uniform_num(*valid_start_x_range)
        start_y = random_uniform_num(*valid_start_y_range)
        
        if map_level == 'Complex':
            base_yaw = dest_yaw - pi/2
            if random() < 0.5:
                base_yaw += pi
            start_yaw = random_gaussian_num(base_yaw, pi/12, base_yaw-pi/6, base_yaw+pi/6)
        elif map_level == 'Extrem':
            base_yaw = dest_yaw + pi
            start_yaw = random_gaussian_num(base_yaw, pi/12, base_yaw-pi/6, base_yaw+pi/6)
        else:
            start_yaw = random_gaussian_num(0, pi/6, -pi/2, pi/2)
            start_yaw = start_yaw+pi if random()<0.5 else start_yaw
        start_box = State([start_x, start_y, start_yaw, 0, 0]).create_box()[0]
        # check collision
        for obst in obstacles:
            if obst.intersects(start_box):
                if DEBUG:
                    print('start box intersect with obstacles, will retry to generate.')
                start_box_valid = False
        # check overlap with dest box
        if dest_box.intersects(start_box):
            if DEBUG:
                print('start box intersect with dest box, will retry to generate.')
            start_box_valid = False

    # randomly drop the obstacles
    for obs in obstacles:
        if random()<DROUP_OUT_OBST:
            obstacles.remove(obs)
    
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list(State([start_x, start_y, start_yaw, 0, 0]).create_box()[0].coords), color='g'))
        print(generate_success)
        if generate_success:
            path = './log/figure/'
            num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            fig = plt.gcf()
            fig.savefig(path+f'image_{num_files}.png')
        plt.clf()
    
    if generate_success:
        return [start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles
    else:
        # print(1)
        return generate_bay_parking_case(map_level)

def generate_parallel_parking_case(map_level):
    '''
    Generate the parameters that a parallel parking case need.
    
    Returns
    ----------
        `start` (list): [x, y, yaw]
        `dest` (list): [x, y, yaw]
        `obstacles` (list): [ obstacle (`LinearRing`) , ...]
    '''
    origin = (0., 0.)
    bay_half_len = 18.
    # params related to map level
    max_PARA_PARK_LOT_LEN = MAX_PARK_LOT_LEN_DICT[map_level]
    min_PARA_PARK_LOT_LEN = MIN_PARK_LOT_LEN_DICT[map_level]
    para_PARK_WALL_DIST = PARA_PARK_WALL_DIST_DICT[map_level]
    n_obst = N_OBSTACLE_DICT[map_level]
    max_longitude_space = max_PARA_PARK_LOT_LEN - LENGTH
    min_longitude_space = min_PARA_PARK_LOT_LEN - LENGTH

    generate_success = True
    # generate obstacle on back
    obstacle_back = LinearRing(( 
        (origin[0]+bay_half_len, origin[1]),
        (origin[0]+bay_half_len, origin[1]-3), 
        (origin[0]-bay_half_len, origin[1]-3), 
        (origin[0]-bay_half_len, origin[1])))
    
    if DEBUG:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlim(-20,20)
        ax.set_ylim(-20,20)
        plt.axis('off')

    # generate dest
    dest_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
    rb, rf, _, _  = list(State([origin[0], origin[1], dest_yaw, 0, 0]).create_box()[0].coords)[:-1]
    min_dest_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = random_gaussian_num(min_dest_y+0.4, 0.2, min_dest_y, min_dest_y+0.8)
    car_rb, car_rf, car_lf, car_lb  = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box()[0].coords)[:-1]
    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))

    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list([car_rb, car_rf, car_lf, car_lb]), color='b'))

    # generate obstacle on left
    # the obstacle can be another vehicle or just a simple obstacle
    non_critical_vehicle = []
    if random()<prob_huge_obst: # generate simple obstacle
        max_dist_to_obst = max_longitude_space/5*4
        min_dist_to_obst = min_longitude_space/5*1
        left_obst_rf = get_rand_pos(*car_lb, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        left_obst_rb = get_rand_pos(*car_rb, pi*11/12, pi*13/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_left = LinearRing(( 
            left_obst_rf,
            left_obst_rb, 
            (origin[0]-bay_half_len, origin[1]), 
            (origin[0]-bay_half_len, left_obst_rf[1])))
    else: # generate another vehicle as obstacle on left
        max_dist_to_obst = max_longitude_space/5*4
        min_dist_to_obst = min_longitude_space/5*1
        left_car_x = origin[0] - (LENGTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        left_car_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
        rb, rf, _, _  = list(State([left_car_x, origin[1], left_car_yaw, 0, 0]).create_box()[0].coords)[:-1]
        min_left_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        left_car_y = random_gaussian_num(min_left_car_y+0.4, 0.2, min_left_car_y, min_left_car_y+0.8)
        obstacle_left = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()[0]

        # generate other parking vehicle
        for _ in range(n_non_critical_car-1):
            left_car_x -= (LENGTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
            left_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            left_car_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
            obstacle_left_ = State([left_car_x, left_car_y, left_car_yaw, 0, 0]).create_box()[0]
            if random()<prob_non_critical_car:
                non_critical_vehicle.append(obstacle_left_)

    # generate obstacle on right
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = max(min_longitude_space-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    max_dist_to_obst = max(max_longitude_space-dist_dest_to_left_obst, 0)+MIN_DIST_TO_OBST
    if random()<0.5: # generate simple obstacle
        right_obst_lf = get_rand_pos(*car_lf, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        right_obst_lb = get_rand_pos(*car_rf, -pi/12, pi/12, min_dist_to_obst, max_dist_to_obst)
        obstacle_right = LinearRing(( 
            (origin[0]+bay_half_len, right_obst_lf[1]),
            (origin[0]+bay_half_len, origin[1]), 
            right_obst_lb, 
            right_obst_lf))
    else: # generate another vehicle as obstacle on right
        right_car_x = origin[0] + (LENGTH + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
        right_car_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
        rb, rf, _, _  = list(State([right_car_x, origin[1], right_car_yaw, 0, 0]).create_box()[0].coords)[:-1]
        min_right_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        right_car_y = random_gaussian_num(min_right_car_y+0.4, 0.2, min_right_car_y, min_right_car_y+0.8)
        obstacle_right = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()[0]

        # generate other parking vehicle
        for _ in range(n_non_critical_car-1):
            right_car_x += (LENGTH + MIN_DIST_TO_OBST + random_uniform_num(min_dist_to_obst, max_dist_to_obst))
            right_car_y += random_gaussian_num(0, 0.05, -0.1, 0.1)
            right_car_yaw = random_gaussian_num(0, pi/36, -pi/12, pi/12)
            obstacle_right_ = State([right_car_x, right_car_y, right_car_yaw, 0, 0]).create_box()[0]
            if random()<prob_non_critical_car:
                non_critical_vehicle.append(obstacle_right_)

    # print(dist_dest_to_left_obst, dest_box.distance(obstacle_right), dest_box.distance(obstacle_right)+dist_dest_to_left_obst)
    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if dist_dest_to_right_obst+dist_dest_to_left_obst<min_longitude_space or \
        dist_dest_to_right_obst+dist_dest_to_left_obst>max_longitude_space or \
        dist_dest_to_left_obst<MIN_DIST_TO_OBST or \
        dist_dest_to_right_obst<MIN_DIST_TO_OBST:
        generate_success = False
    # print(dist_dest_to_right_obst,dist_dest_to_left_obst,dist_dest_to_right_obst+dist_dest_to_left_obst)
    # check collision
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    obstacles.extend(non_critical_vehicle)
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles])+MIN_DIST_TO_OBST
    other_obstcales = []
    if random()<0.2: # in this case only a wall will be generate
        other_obstcales = [LinearRing(( 
        (origin[0]-bay_half_len, para_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST),
        (origin[0]+bay_half_len, para_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST), 
        (origin[0]+bay_half_len, para_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1), 
        (origin[0]-bay_half_len, para_PARK_WALL_DIST+max_obstacle_y+MIN_DIST_TO_OBST+0.1)))]
    else:
        other_obstacle_range = LinearRing(( 
        (origin[0]-bay_half_len, para_PARK_WALL_DIST+max_obstacle_y),
        (origin[0]+bay_half_len, para_PARK_WALL_DIST+max_obstacle_y), 
        (origin[0]+bay_half_len, para_PARK_WALL_DIST+max_obstacle_y+8), 
        (origin[0]-bay_half_len, para_PARK_WALL_DIST+max_obstacle_y+8)))
        valid_obst_x_range = (origin[0]-bay_half_len+2, origin[0]+bay_half_len-2)
        valid_obst_y_range = (para_PARK_WALL_DIST+max_obstacle_y+2, para_PARK_WALL_DIST+max_obstacle_y+6)
        for _ in range(n_obst):
            obs_x = random_uniform_num(*valid_obst_x_range)
            obs_y = random_uniform_num(*valid_obst_y_range)
            obs_yaw = random()*pi*2
            obs_coords = np.array(State([obs_x, obs_y, obs_yaw, 0, 0]).create_box()[0].coords[:-1])
            obs = LinearRing(obs_coords+2.0*random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            obst_invalid = False
            for other_obs in other_obstcales:
                if obs.intersects(other_obs):
                    obst_invalid = True
                    break
            if obst_invalid:
                continue
            other_obstcales.append(obs)

    # merge two kind of obstacles
    obstacles.extend(other_obstcales)

    if DEBUG:
        for obs in obstacles:
            ax.add_patch(plt.Polygon(xy=list(obs.coords), color='gray'))

    
    # generate start position
    start_box_valid = False
    valid_start_x_range = (origin[0]-bay_half_len/2, origin[0]+bay_half_len/2)
    
    if map_level == 'Complex':
        valid_start_y_range = (max_obstacle_y+1, max_obstacle_y+1 + (para_PARK_WALL_DIST-1)*0.6)
    elif map_level == 'Extrem':
        valid_start_y_range = (max_obstacle_y+1, max_obstacle_y+1 + (para_PARK_WALL_DIST-1)*0.3)
    else:
        valid_start_y_range = (max_obstacle_y+1, para_PARK_WALL_DIST+max_obstacle_y-1)

    while not start_box_valid:
        start_box_valid = True
        start_x = random_uniform_num(*valid_start_x_range)
        start_y = random_uniform_num(*valid_start_y_range)
        
        if map_level == 'Complex':
            base_yaw = dest_yaw - pi/2
            if random() < 0.5:
                base_yaw += pi
            start_yaw = random_gaussian_num(base_yaw, pi/12, base_yaw-pi/6, base_yaw+pi/6)
        elif map_level == 'Extrem':
            base_yaw = dest_yaw + pi
            start_yaw = random_gaussian_num(base_yaw, pi/12, base_yaw-pi/6, base_yaw+pi/6)
        else:
            start_yaw = random_gaussian_num(0, pi/6, -pi/2, pi/2)
            start_yaw = start_yaw+pi if random()<0.5 else start_yaw
        start_box = State([start_x, start_y, start_yaw, 0, 0]).create_box()[0]
        # check collision
        for obst in obstacles:
            if obst.intersects(start_box):
                if DEBUG:
                    print('start box intersect with obstacles, will retry to generate.')
                start_box_valid = False
        # check overlap with dest box
        if dest_box.intersects(start_box):
            if DEBUG:
                print('start box intersect with dest box, will retry to generate.')
            start_box_valid = False
    
    
    # flip the dest box so that the orientation of start matches the dest
    if cos(start_yaw)<0:
        dest_box_center = np.mean(np.array(dest_box.coords[:-1]), axis=0)
        dest_x = 2*dest_box_center[0] - dest_x
        dest_y = 2*dest_box_center[1] - dest_y
        dest_yaw += pi

    # randomly drop the obstacles
    for obs in obstacles:
        if random()<DROUP_OUT_OBST:
            obstacles.remove(obs)
    
    if DEBUG:
        ax.add_patch(plt.Polygon(xy=list(State([start_x, start_y, start_yaw, 0, 0]).create_box()[0].coords), color='g'))
        print(generate_success)
        plt.show()
    
    if generate_success:
        return [start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles
    else:
        return generate_parallel_parking_case(map_level)

def generate_navigation_case(map_level):
    '''
    Generate navigation case with polygon walls and gaps in a 80x80m area.
    '''
    # Map boundaries
    min_bound = -40.0
    max_bound = 40.0
    
    obstacles = []
    wall_thickness = 2.0
    
    # Generate random polygon vertices (convex hull of random points)
    n_points = np.random.randint(5, 9)
    points = []
    for _ in range(n_points):
        # Keep some margin from the absolute boundary
        x = random_uniform_num(min_bound + 5, max_bound - 5)
        y = random_uniform_num(min_bound + 5, max_bound - 5)
        points.append((x, y))
    
    # Create convex hull
    polygon = MultiPoint(points).convex_hull
    
    # Ensure it's a Polygon (not a LineString or Point if points are collinear)
    if not isinstance(polygon, Polygon):
        # Fallback to a simple box if generation fails
        polygon = Polygon([
            (min_bound+5, min_bound+5), (max_bound-5, min_bound+5),
            (max_bound-5, max_bound-5), (min_bound+5, max_bound-5)
        ])

    # Create walls from edges with gaps
    exterior_coords = list(polygon.exterior.coords)
    for i in range(len(exterior_coords) - 1):
        p1 = np.array(exterior_coords[i])
        p2 = np.array(exterior_coords[i+1])
        
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        edge_dir = edge_vec / edge_len
        normal_dir = np.array([-edge_dir[1], edge_dir[0]])
        
        # Randomly decide to add a gap
        if random() < 0.7: # 70% chance of gap
            gap_size = random_uniform_num(4.0, 8.0)
            if edge_len > gap_size + 4.0: # Ensure edge is long enough
                # Split into two segments
                split_point = random_uniform_num(2.0, edge_len - gap_size - 2.0)
                
                # Segment 1
                seg1_start = p1
                seg1_end = p1 + edge_dir * split_point
                
                # Segment 2
                seg2_start = p1 + edge_dir * (split_point + gap_size)
                seg2_end = p2
                
                # Create wall boxes for segments
                for start, end in [(seg1_start, seg1_end), (seg2_start, seg2_end)]:
                    # Create a box around the line segment
                    # Offset by wall_thickness/2 in normal direction
                    w_p1 = start + normal_dir * (wall_thickness/2)
                    w_p2 = end + normal_dir * (wall_thickness/2)
                    w_p3 = end - normal_dir * (wall_thickness/2)
                    w_p4 = start - normal_dir * (wall_thickness/2)
                    obstacles.append(LinearRing([tuple(w_p1), tuple(w_p2), tuple(w_p3), tuple(w_p4)]))
            else:
                # Edge too short for gap, just build full wall
                w_p1 = p1 + normal_dir * (wall_thickness/2)
                w_p2 = p2 + normal_dir * (wall_thickness/2)
                w_p3 = p2 - normal_dir * (wall_thickness/2)
                w_p4 = p1 - normal_dir * (wall_thickness/2)
                obstacles.append(LinearRing([tuple(w_p1), tuple(w_p2), tuple(w_p3), tuple(w_p4)]))

    # Generate start and dest inside the polygon
    # Increased distance range for tactics2d navigation scenarios (60-84m typical)
    min_dist = random_uniform_num(50.0, 80.0)
    
    # Safety counter to prevent infinite loop
    max_attempts = 1000
    attempts = 0
    
    while attempts < max_attempts:
        attempts += 1
        # Sample points in bounding box and check if inside polygon
        start_x = random_uniform_num(min_bound, max_bound)
        start_y = random_uniform_num(min_bound, max_bound)
        start_point = Point(start_x, start_y)
        
        if not polygon.contains(start_point):
            continue
            
        dest_x = random_uniform_num(min_bound, max_bound)
        dest_y = random_uniform_num(min_bound, max_bound)
        dest_point = Point(dest_x, dest_y)
        
        if not polygon.contains(dest_point):
            continue
            
        if np.hypot(start_x - dest_x, start_y - dest_y) > min_dist:
            # Check if start/dest are too close to walls (optional but good)
            if polygon.boundary.distance(start_point) > 3.0 and polygon.boundary.distance(dest_point) > 3.0:
                break
    
    if attempts >= max_attempts:
        # Fallback if valid points not found
        # Use centroid or simple points
        start_x, start_y = polygon.centroid.x - 10, polygon.centroid.y
        dest_x, dest_y = polygon.centroid.x + 10, polygon.centroid.y
        
    start_yaw = random_uniform_num(-pi, pi)
    dest_yaw = random_uniform_num(-pi, pi)
    
    # Internal obstacles
    n_internal_obstacles = 10
    for _ in range(n_internal_obstacles):
        obs_x = random_uniform_num(min_bound, max_bound)
        obs_y = random_uniform_num(min_bound, max_bound)
        obs_point = Point(obs_x, obs_y)
        
        if not polygon.contains(obs_point):
            continue
            
        # Check distance to start/dest/walls
        if np.hypot(obs_x - start_x, obs_y - start_y) < 8.0 or \
           np.hypot(obs_x - dest_x, obs_y - dest_y) < 8.0 or \
           polygon.boundary.distance(obs_point) < 5.0:
            continue
            
        obs_yaw = random_uniform_num(-pi, pi)
        obs_w = random_uniform_num(3.0, 6.0)
        obs_h = random_uniform_num(3.0, 6.0)
        
        cx, cy = obs_x, obs_y
        angle = obs_yaw
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        pts = [
            (cx + (obs_w/2)*cos_a - (obs_h/2)*sin_a, cy + (obs_w/2)*sin_a + (obs_h/2)*cos_a),
            (cx - (obs_w/2)*cos_a - (obs_h/2)*sin_a, cy - (obs_w/2)*sin_a + (obs_h/2)*cos_a),
            (cx - (obs_w/2)*cos_a + (obs_h/2)*sin_a, cy - (obs_w/2)*sin_a - (obs_h/2)*cos_a),
            (cx + (obs_w/2)*cos_a + (obs_h/2)*sin_a, cy + (obs_w/2)*sin_a - (obs_h/2)*cos_a)
        ]
        obstacles.append(LinearRing(pts))
        
    return [start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles


class ParkingMapNormal(object):
    def __init__(self, map_level=MAP_LEVEL):

        self.case_id:int = None
        self.map_level = map_level
        self.start: State = None
        self.dest: State = None
        self.start_box:LinearRing = None
        self.dest_box:LinearRing = None
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.n_obstacle = 0
        self.obstacles:List[Area] = []

    def reset(self, case_id: int = None, path: str = None) -> State:
        # Always use navigation case for the new task
        start, dest, obstacles = generate_navigation_case(self.map_level)
        self.case_id = 2
        
        # Add random articulation angle to initial state (critical for training!)
        # Previously always 0, causing state_norm std=0 issue
        start_articulation = random_uniform_num(np.radians(-10), np.radians(10))
        start_rear_heading = start[2] - start_articulation  # rear_heading = front_heading - articulation
        self.start = State(start+[0,0,start_rear_heading])
        self.start_box = self.start.create_box()[0]
        
        # Dest should also have articulation possibility
        dest_articulation = random_uniform_num(np.radians(-10), np.radians(10))
        dest_rear_heading = dest[2] - dest_articulation
        self.dest = State(dest+[0,0,dest_rear_heading])
        self.dest_box = self.dest.create_box()[0]
        
        # Expanded map boundaries to 120x120m (-60 to 60) for longer navigation
        self.xmin = -60.0
        self.xmax = 60.0
        self.ymin = -60.0
        self.ymax = 60.0
        
        self.obstacles = list([Area(shape=obs, subtype="obstacle", \
            color=(150, 150, 150, 255)) for obs in obstacles])
        self.n_obstacle = len(self.obstacles)

        return self.start
    
    def _flip_box_orientation(self, target_state:State):
        x, y, heading = target_state.get_pos()
        center = np.mean(target_state.create_box()[0].coords[:-1], axis=0)
        new_x = 2*center[0] - x
        new_y = 2*center[1] - y
        heading = heading + np.pi
        return State([new_x, new_y, heading])
    
    def flip_dest_orientation(self,):
        print('before:', self.dest.get_pos())
        self.dest = self._flip_box_orientation(self.dest)
        self.dest_box = self.dest.create_box()[0]
        print('after:', self.dest.get_pos())
        print('flip dest orientation')

    def flip_start_orientation(self,):
        self.start = self._flip_box_orientation(self.start)
        self.start_box = self.start.create_box()[0]


if __name__ == '__main__':
    import time
    t = time.time()
    for _ in range(10):
        generate_bay_parking_case()
    print('generate time:', time.time()-t)

    t = time.time()
    for _ in range(10):
        generate_parallel_parking_case()
    print('generate time:', time.time()-t)
