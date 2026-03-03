
import sys
sys.path.append("../")
from math import pi, cos, sin
import os

import numpy as np
from numpy.random import randn, random
from typing import List
from shapely.geometry import LinearRing, Point, MultiPoint, Polygon, LineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.prepared import prep

from env.vehicle import State
from env.map_base import *
from configs import *

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

# params for generating parking case
prob_huge_obst = 0.0
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
    obstacles = [obstacle_left, obstacle_right]
    obstacles.extend(non_critical_vehicle)
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles])+MIN_DIST_TO_OBST
    other_obstcales = []
    
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
        
        start_state = State([start_x, start_y, start_yaw, 0, 0])
        start_boxes = start_state.create_box()
        # check collision
        for obst in obstacles:
            for sb in start_boxes:
                if obst.intersects(sb):
                    if DEBUG:
                        print('start box intersect with obstacles, will retry to generate.')
                    start_box_valid = False
                    break
            if not start_box_valid: break
            
        # check overlap with dest box
        if start_box_valid:
            dest_state = State([dest_x, dest_y, dest_yaw, 0, 0])
            dest_boxes = dest_state.create_box()
            for sb in start_boxes:
                for db in dest_boxes:
                    if sb.intersects(db):
                        if DEBUG:
                            print('start box intersect with dest box, will retry to generate.')
                        start_box_valid = False
                        break
                if not start_box_valid: break

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
    obstacles = [obstacle_left, obstacle_right]
    obstacles.extend(non_critical_vehicle)
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = max([np.max(np.array(obs.coords)[:,1]) for obs in obstacles])+MIN_DIST_TO_OBST
    other_obstcales = []
    
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
        
        start_state = State([start_x, start_y, start_yaw, 0, 0])
        start_boxes = start_state.create_box()
        # check collision
        for obst in obstacles:
            for sb in start_boxes:
                if obst.intersects(sb):
                    if DEBUG:
                        print('start box intersect with obstacles, will retry to generate.')
                    start_box_valid = False
                    break
            if not start_box_valid: break
            
        # check overlap with dest box
        if start_box_valid:
            dest_state = State([dest_x, dest_y, dest_yaw, 0, 0])
            dest_boxes = dest_state.create_box()
            for sb in start_boxes:
                for db in dest_boxes:
                    if sb.intersects(db):
                        if DEBUG:
                            print('start box intersect with dest box, will retry to generate.')
                        start_box_valid = False
                        break
                if not start_box_valid: break
    
    
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

def generate_navigation_case(map_level, return_regions: bool = False):
    """Generate a closed irregular plaza + 1-2 corridors navigation case.

    Spec implemented:
    - Entire scene within 80m x 80m ([-40, 40] square).
    - Drivable space is an irregular polygon plaza (4-6 sides) plus 1-2 corridors.
    - Corridors may bend; width >= 2 vehicle lengths.
    - Everything outside drivable space (but inside the 80x80) is filled as solid obstacles.
    - No random internal obstacles.
    - Start/dest headings are perpendicular to the nearest plaza/corridor edge and face the edge;
      front bumper is ~1m from that edge.
    - Difficulty mapping:
        Normal  -> both poses in plaza
        Complex -> one in plaza, one in corridor
        Extrem  -> both in corridor
    """

    WORLD_MIN = -40.0
    WORLD_MAX = 40.0
    WALL_THICKNESS = 2.0
    INNER_WALL_THICKNESS = 1.2

    world = Polygon([
        (WORLD_MIN, WORLD_MIN), (WORLD_MAX, WORLD_MIN),
        (WORLD_MAX, WORLD_MAX), (WORLD_MIN, WORLD_MAX)
    ]).buffer(0)

    # Inner free region (inside the walls). Corridors must reach THIS boundary.
    inner = world.buffer(-WALL_THICKNESS).buffer(0)
    if inner.is_empty:
        inner = world.buffer(-1.0).buffer(0)

    # Physical boundary walls: thick strips INSIDE the 80x80 world.
    # These are true obstacles (collision), not just out-of-bounds checks.
    wall_top = Polygon([
        (WORLD_MIN, WORLD_MAX - WALL_THICKNESS), (WORLD_MAX, WORLD_MAX - WALL_THICKNESS),
        (WORLD_MAX, WORLD_MAX), (WORLD_MIN, WORLD_MAX)
    ]).buffer(0)
    wall_bottom = Polygon([
        (WORLD_MIN, WORLD_MIN), (WORLD_MAX, WORLD_MIN),
        (WORLD_MAX, WORLD_MIN + WALL_THICKNESS), (WORLD_MIN, WORLD_MIN + WALL_THICKNESS)
    ]).buffer(0)
    wall_left = Polygon([
        (WORLD_MIN, WORLD_MIN), (WORLD_MIN + WALL_THICKNESS, WORLD_MIN),
        (WORLD_MIN + WALL_THICKNESS, WORLD_MAX), (WORLD_MIN, WORLD_MAX)
    ]).buffer(0)
    wall_right = Polygon([
        (WORLD_MAX - WALL_THICKNESS, WORLD_MIN), (WORLD_MAX, WORLD_MIN),
        (WORLD_MAX, WORLD_MAX), (WORLD_MAX - WALL_THICKNESS, WORLD_MAX)
    ]).buffer(0)
    walls = [wall_top, wall_bottom, wall_left, wall_right]

    # -------------------------
    # Difficulty-dependent geometry
    # -------------------------
    # Guarantee: corridor width >= 2 * car length
    min_corridor_width = float(2.0 * LENGTH)

    if map_level == 'Normal':
        # Easier: larger plaza, wider/shorter corridor(s)
        plaza_r_min, plaza_r_max = 19.0, 24.0
        corridor_width = float(max(min_corridor_width, 14.0))
        n_corridors = 2
        len1_range = (14.0, 20.0)
        len2_range = (8.0, 12.0)
        bend_prob = 0.55
        bend_deg = 45.0
    elif map_level == 'Complex':
        # Medium: medium plaza, medium corridor(s)
        plaza_r_min, plaza_r_max = 16.0, 20.0
        corridor_width = float(max(min_corridor_width, 12.0))
        n_corridors = 2 if random() < 0.5 else 3
        len1_range = (18.0, 26.0)
        len2_range = (12.0, 18.0)
        bend_prob = 0.8
        bend_deg = 65.0
    else:  # Extrem
        # Hard: keep tighter topology via more corridors/turns, while matching Complex plaza/corridor size
        plaza_r_min, plaza_r_max = 16.0, 20.0
        corridor_width = float(max(min_corridor_width, 12.0))
        n_corridors = 3
        len1_range = (18.0, 26.0)
        len2_range = (12.0, 18.0)
        bend_prob = 0.95
        bend_deg = 80.0

    half_w = corridor_width / 2.0

    # Remove random center walls; slot-like divider walls will be added near start/dest later.
    n_inner_walls = 0

    def _unit(v):
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([1.0, 0.0])
        return v / n

    def _pick_edge(coords):
        # coords is closed (n+1)
        n = max(0, len(coords) - 1)
        if n <= 0:
            return None
        i = int(np.random.randint(0, n))
        p1 = np.array(coords[i], dtype=float)
        p2 = np.array(coords[i + 1], dtype=float)
        if float(np.linalg.norm(p2 - p1)) < 1e-6:
            return None
        return p1, p2

    def _extract_points(g):
        """Extract candidate points from shapely intersection result."""
        pts = []
        if g is None or g.is_empty:
            return pts
        if isinstance(g, Point):
            return [g]
        if hasattr(g, 'geoms'):
            for gg in list(g.geoms):
                pts.extend(_extract_points(gg))
            return pts
        if isinstance(g, LineString):
            coords = list(g.coords)
            if len(coords) >= 1:
                pts.append(Point(coords[0]))
                pts.append(Point(coords[-1]))
            return pts
        return pts

    def _extend_to_inner_boundary(p0: np.ndarray, d: np.ndarray, eps: float = 0.6):
        """Cast a ray from p0 along direction d until it hits the inner boundary."""
        d = _unit(d)
        ray_far = p0 + d * 500.0
        ray = LineString([tuple(p0), tuple(ray_far)])
        inter = ray.intersection(inner.boundary)
        pts = _extract_points(inter)
        if len(pts) == 0:
            return p0 + d * 30.0

        best = None
        best_proj = None
        for pt in pts:
            v = np.array([pt.x - p0[0], pt.y - p0[1]], dtype=float)
            proj = float(v[0] * d[0] + v[1] * d[1])
            if proj <= 1e-6:
                continue
            if best_proj is None or proj > best_proj:
                best_proj = proj
                best = np.array([pt.x, pt.y], dtype=float)

        if best is None:
            return p0 + d * 30.0

        # Stay a bit inside the inner region so corridor doesn't overlap wall polygons.
        return best - d * float(eps)

    def _edge_normals(poly: Polygon, p1, p2):
        t = _unit(p2 - p1)
        n1 = _unit(np.array([-t[1], t[0]]))
        mid = 0.5 * (p1 + p2)
        # Determine inward by probing
        probe = Point(float(mid[0] + n1[0] * 0.5), float(mid[1] + n1[1] * 0.5))
        if poly.contains(probe):
            inward = n1
        else:
            inward = -n1
        outward = -inward
        return inward, outward

    def _as_polygon(g):
        """Return a Polygon to sample edges from.

        Corridor clipping can yield MultiPolygon; we sample from its largest component.
        """
        if g is None or g.is_empty:
            return None
        if isinstance(g, Polygon):
            return g
        if isinstance(g, MultiPolygon):
            geoms = list(g.geoms)
            if len(geoms) == 0:
                return None
            return max(geoms, key=lambda x: float(x.area))
        # Best-effort: try buffer(0) to coerce
        try:
            gg = g.buffer(0)
            if isinstance(gg, Polygon):
                return gg
            if isinstance(gg, MultiPolygon):
                geoms = list(gg.geoms)
                if len(geoms) == 0:
                    return None
                return max(geoms, key=lambda x: float(x.area))
        except Exception:
            pass
        return None

    def _segment_wall_poly(p1: np.ndarray, p2: np.ndarray, thickness: float) -> Polygon:
        ls = LineString([tuple(p1), tuple(p2)])
        try:
            w = ls.buffer(thickness / 2.0, cap_style=2, join_style=2).buffer(0)
        except Exception:
            w = ls.buffer(thickness / 2.0).buffer(0)
        return w

    def _grid_reachable(region, start_xy, goal_xy, resolution: float = 0.6, max_visits: int = 120000):
        """Conservative reachability check for a point robot with clearance.

        We discretize the free region into a grid and do BFS from start to goal.
        """
        if region is None or region.is_empty:
            return False
        try:
            region_prep = prep(region)
        except Exception:
            region_prep = None

        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        if region_prep is not None:
            if not region_prep.contains(Point(sx, sy)):
                return False
            if not region_prep.contains(Point(gx, gy)):
                return False
        else:
            if not region.contains(Point(sx, sy)):
                return False
            if not region.contains(Point(gx, gy)):
                return False

        xmin, ymin, xmax, ymax = region.bounds
        xmin = float(max(xmin, WORLD_MIN))
        ymin = float(max(ymin, WORLD_MIN))
        xmax = float(min(xmax, WORLD_MAX))
        ymax = float(min(ymax, WORLD_MAX))
        if xmax <= xmin or ymax <= ymin:
            return False

        nx = int(np.ceil((xmax - xmin) / resolution))
        ny = int(np.ceil((ymax - ymin) / resolution))
        if nx <= 1 or ny <= 1:
            return False

        def _to_idx(x, y):
            ix = int(np.clip(np.floor((x - xmin) / resolution), 0, nx - 1))
            iy = int(np.clip(np.floor((y - ymin) / resolution), 0, ny - 1))
            return ix, iy

        def _center(ix, iy):
            return (xmin + (ix + 0.5) * resolution, ymin + (iy + 0.5) * resolution)

        start_idx = _to_idx(sx, sy)
        goal_idx = _to_idx(gx, gy)

        from collections import deque
        q = deque([start_idx])
        visited = set([start_idx])

        # 8-connected grid
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def _free(ix, iy):
            cx, cy = _center(ix, iy)
            pt = Point(float(cx), float(cy))
            if region_prep is not None:
                return bool(region_prep.contains(pt))
            return bool(region.contains(pt))

        # Ensure start/goal cells are free (cell center inside region)
        if not _free(*start_idx):
            # If start lies on boundary, allow nearby cell.
            found = False
            for dx, dy in nbrs:
                nx_i = start_idx[0] + dx
                ny_i = start_idx[1] + dy
                if 0 <= nx_i < nx and 0 <= ny_i < ny and _free(nx_i, ny_i):
                    start_idx = (nx_i, ny_i)
                    q = deque([start_idx])
                    visited = set([start_idx])
                    found = True
                    break
            if not found:
                return False
        if not _free(*goal_idx):
            # Similar relaxation for goal
            found = False
            for dx, dy in nbrs:
                nx_i = goal_idx[0] + dx
                ny_i = goal_idx[1] + dy
                if 0 <= nx_i < nx and 0 <= ny_i < ny and _free(nx_i, ny_i):
                    goal_idx = (nx_i, ny_i)
                    found = True
                    break
            if not found:
                return False

        visits = 0
        while q:
            ix, iy = q.popleft()
            if (ix, iy) == goal_idx:
                return True
            for dx, dy in nbrs:
                jx = ix + dx
                jy = iy + dy
                if jx < 0 or jx >= nx or jy < 0 or jy >= ny:
                    continue
                idx = (jx, jy)
                if idx in visited:
                    continue
                if not _free(jx, jy):
                    continue
                visited.add(idx)
                q.append(idx)
                visits += 1
                if visits >= int(max_visits):
                    # Conservative bail-out: treat as unreachable to avoid heavy compute.
                    return False

        return False

    def _sample_pose_near_edge(poly: Polygon, avoid_poly: Polygon = None, head_dist: float = 1.0, max_tries: int = 2000):
        """Sample a pose inside poly, close to an edge.

        Heading faces the edge (outward normal), so the vehicle points toward the boundary.
        Front bumper distance to the boundary is approximately head_dist.
        """
        poly = _as_polygon(poly)
        if poly is None or poly.is_empty:
            raise RuntimeError("poly is empty")

        for _ in range(max_tries):
            coords = list(poly.exterior.coords)
            e = _pick_edge(coords)
            if e is None:
                continue
            p1, p2 = e
            # pick an interior point on the segment
            t = float(random_uniform_num(0.2, 0.8))
            boundary_pt = (1.0 - t) * p1 + t * p2

            inward, outward = _edge_normals(poly, p1, p2)
            heading = float(np.arctan2(outward[1], outward[0]))

            # Place front axle inside poly so that the front bumper is ~head_dist from boundary.
            # bumper = loc + FRONT_HANG * outward
            # distance(boundary -> bumper) ~= head_dist => loc offset from boundary ~= (head_dist + FRONT_HANG)
            loc = boundary_pt - outward * float(head_dist + FRONT_HANG + 0.2)

            pt_loc = Point(float(loc[0]), float(loc[1]))
            if not poly.contains(pt_loc):
                continue
            if avoid_poly is not None and avoid_poly.contains(pt_loc):
                continue

            # Collision check against solid obstacles will be done by caller.
            return [float(loc[0]), float(loc[1]), heading]

        raise RuntimeError("Failed to sample pose near edge")

    def _sample_pose_near_edge_facing_blocking(
        poly: Polygon,
        blocking_poly,
        plaza_poly: Polygon = None,
        min_plaza_dist: float = 0.0,
        head_dist: float = 1.0,
        max_tries: int = 2000,
    ):
        """Sample a pose near an edge whose outward side is an obstacle (blocking).

        This prevents sampling near the corridor mouth (the cut edge created by subtracting plaza),
        and aligns the heading toward a non-free-space wall.
        """
        poly = _as_polygon(poly)
        if poly is None or poly.is_empty:
            raise RuntimeError("poly is empty")

        min_plaza_dist = float(max(0.0, min_plaza_dist))

        for _ in range(int(max_tries)):
            coords = list(poly.exterior.coords)
            e = _pick_edge(coords)
            if e is None:
                continue
            p1, p2 = e

            t = float(random_uniform_num(0.2, 0.8))
            boundary_pt = (1.0 - t) * p1 + t * p2

            if plaza_poly is not None and min_plaza_dist > 1e-9:
                try:
                    if float(Point(float(boundary_pt[0]), float(boundary_pt[1])).distance(plaza_poly)) < min_plaza_dist:
                        continue
                except Exception:
                    pass

            inward, outward = _edge_normals(poly, p1, p2)
            # Probe slightly outside the region along outward normal; it must hit obstacles.
            probe = Point(float(boundary_pt[0] + outward[0] * 0.4), float(boundary_pt[1] + outward[1] * 0.4))
            try:
                if (not blocking_poly.contains(probe)) and (not blocking_poly.intersects(probe)):
                    continue
            except Exception:
                # If blocking is invalid, be conservative.
                continue

            heading = float(np.arctan2(outward[1], outward[0]))
            loc = boundary_pt - outward * float(head_dist + FRONT_HANG + 0.2)

            pt_loc = Point(float(loc[0]), float(loc[1]))
            if not poly.contains(pt_loc):
                continue

            return [float(loc[0]), float(loc[1]), heading]

        raise RuntimeError("Failed to sample pose facing blocking edge")

    # 1) Build plaza (4-6 sides) near the center (inside inner region)
    plaza = None
    for _ in range(200):
        n_side = int(np.random.randint(4, 7))
        base_angles = np.linspace(0.0, 2.0 * np.pi, n_side, endpoint=False)
        jitter = np.random.uniform(-0.35, 0.35, size=n_side)
        angles = np.sort(base_angles + jitter)
        radii = np.random.uniform(plaza_r_min, plaza_r_max, size=n_side)
        pts = [(float(radii[i] * np.cos(angles[i])), float(radii[i] * np.sin(angles[i]))) for i in range(n_side)]
        p = Polygon(pts).buffer(0)
        if p.is_empty or (not p.is_valid) or p.area < 100.0:
            continue
        if not inner.buffer(-6.0).contains(p):
            continue
        plaza = p
        break
    if plaza is None:
        # Fallback to a simple hexagon-ish plaza
        pts = [(18.0, 0.0), (9.0, 15.0), (-9.0, 15.0), (-18.0, 0.0), (-9.0, -15.0), (9.0, -15.0)]
        plaza = Polygon(pts).buffer(0)

    # 2) Build 2-3 corridors attached to plaza; each corridor extends to the inner wall boundary.
    corridors = []

    plaza_coords = list(plaza.exterior.coords)
    for _ in range(n_corridors):
        corridor = None
        for _try in range(200):
            e = _pick_edge(plaza_coords)
            if e is None:
                continue
            p1, p2 = e
            mid = 0.5 * (p1 + p2)
            inward, outward = _edge_normals(plaza, p1, p2)

            # Start slightly inside the plaza to guarantee connectivity
            start = mid + inward * float(half_w * 0.25)

            # Build a multi-segment centerline with multiple bends.
            pts = [start]
            d = outward

            # Number of bends by difficulty (allows multiple bends)
            if map_level == 'Normal':
                n_bends = 0 if random() < 0.4 else 1
            elif map_level == 'Complex':
                n_bends = 1 if random() < 0.5 else 2
            else:  # Extrem
                n_bends = 2

            # Intermediate segments
            p_curr = start
            for _b in range(int(n_bends)):
                seg_len = float(random_uniform_num(*len1_range))
                p_next = p_curr + d * seg_len
                pts.append(p_next)
                # Bend
                bend = float(random_uniform_num(-np.deg2rad(bend_deg), np.deg2rad(bend_deg)))
                c, s = float(np.cos(bend)), float(np.sin(bend))
                d = _unit(np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c]))
                p_curr = p_next

            # Final segment: extend until we hit the inner boundary
            end = _extend_to_inner_boundary(p_curr, d)
            pts.append(end)

            centerline = LineString([tuple(p) for p in pts])
            cpoly = centerline.buffer(half_w, cap_style=2, join_style=2)
            if cpoly.is_empty:
                continue

            # Keep everything inside the inner free region
            cpoly = cpoly.intersection(inner).buffer(0)
            if cpoly.is_empty or float(cpoly.area) < 50.0:
                continue

            # Must connect to plaza
            if not cpoly.intersects(plaza):
                continue

            # Avoid excessive corridor-corridor overlap
            if any(cpoly.intersection(cc).area > 0.3 * min(float(cpoly.area), float(cc.area)) for cc in corridors):
                continue

            corridor = cpoly.buffer(0)
            break
        if corridor is not None:
            corridors.append(corridor)

    # Guaranteed fallback: ensure we end up with at least one corridor.
    # If generation failed, create two straight corridors from plaza toward +x and -x,
    # each extended to the inner boundary.
    if len(corridors) == 0:
        coords = np.array(list(plaza.exterior.coords)[:-1], dtype=float)
        east_idx = int(np.argmax(coords[:, 0]))
        west_idx = int(np.argmin(coords[:, 0]))
        for idx in [east_idx, west_idx]:
            p1 = coords[idx]
            p2 = coords[(idx + 1) % coords.shape[0]]
            mid = 0.5 * (p1 + p2)
            inward, outward = _edge_normals(plaza, p1, p2)
            start = mid + inward * float(half_w * 0.25)
            end = _extend_to_inner_boundary(start, outward)
            centerline = LineString([tuple(start), tuple(end)])
            cpoly = centerline.buffer(half_w, cap_style=2, join_style=2)
            cpoly = cpoly.intersection(inner).buffer(0)
            if not cpoly.is_empty and cpoly.intersects(plaza):
                corridors.append(cpoly)

    # Last-resort fallback: build a deterministic corridor from plaza centroid to the inner boundary.
    # This keeps difficulty constraints strict (corridor poses must exist for Complex/Extrem).
    if len(corridors) == 0:
        c = np.array([float(plaza.centroid.x), float(plaza.centroid.y)], dtype=float)
        for d in [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])]:
            end = _extend_to_inner_boundary(c, d)
            centerline = LineString([tuple(c), tuple(end)])
            cpoly = centerline.buffer(half_w, cap_style=2, join_style=2)
            cpoly = cpoly.intersection(inner).buffer(0)
            if not cpoly.is_empty:
                corridors.append(cpoly)
                break

    drivable = unary_union([plaza] + corridors).buffer(0)
    drivable = drivable.intersection(inner).buffer(0)

    # Solid fill inside the inner free region = everything except plaza+corridors
    solid_fill = inner.difference(drivable).buffer(0)

    def _sample_inner_walls(plaza_poly: Polygon, existing_walls: list):
        """Sample inner wall polygons inside the plaza.

        Walls may float near the center or attach to plaza boundary.
        """
        if n_inner_walls <= 0:
            return []

        plaza_poly = _as_polygon(plaza_poly)
        if plaza_poly is None or plaza_poly.is_empty:
            return []

        # Size scaling from plaza bbox
        bx0, by0, bx1, by1 = plaza_poly.bounds
        plaza_d = float(min(bx1 - bx0, by1 - by0))
        # Keep walls not too long; count increases with difficulty.
        base_len_min = 0.22 * plaza_d
        base_len_max = 0.50 * plaza_d
        if map_level == 'Normal':
            base_len_max *= 0.85
        elif map_level == 'Extrem':
            base_len_max *= 1.05

        # Clearance used later also informs wall sampling (avoid creating zero-gap traps)
        margin = float(max(0.6, WIDTH / 2.0 + 0.2))
        plaza_inset = plaza_poly.buffer(-margin).buffer(0)
        if plaza_inset.is_empty:
            plaza_inset = plaza_poly.buffer(-0.2).buffer(0)
        plaza_coords = list(plaza_poly.exterior.coords)

        inner_walls = []
        union_existing = unary_union([w for w in (existing_walls or []) if w is not None] + inner_walls).buffer(0)

        for _k in range(int(n_inner_walls)):
            wpoly = None
            for _try in range(250):
                attach_to_boundary = bool(random() < 0.5)

                length = float(random_uniform_num(base_len_min, base_len_max))
                angle = float(random_uniform_num(-np.pi, np.pi))
                d = np.array([np.cos(angle), np.sin(angle)], dtype=float)

                if attach_to_boundary:
                    e = _pick_edge(plaza_coords)
                    if e is None:
                        continue
                    p1, p2 = e
                    mid = 0.5 * (p1 + p2)
                    inward, _outward = _edge_normals(plaza_poly, p1, p2)

                    # Bias the wall direction to point roughly inward so clipping touches boundary.
                    # Start a little outside boundary and extend inward.
                    # Combine inward normal with random lateral component.
                    lat = _unit(np.array([-inward[1], inward[0]]))
                    wdir = _unit(inward + lat * float(random_uniform_num(-1.2, 1.2)))
                    anchor = mid + (-inward) * float(0.15)  # slightly outside
                    p_start = anchor
                    p_end = anchor + wdir * float(length + 0.6)
                    w = _segment_wall_poly(p_start, p_end, INNER_WALL_THICKNESS)
                else:
                    # Place near center (but still inside) and let it float.
                    if plaza_inset is None or plaza_inset.is_empty:
                        continue
                    # Sample point by rejection within bbox
                    px0, py0, px1, py1 = plaza_inset.bounds
                    for _ in range(80):
                        cx = float(random_uniform_num(px0, px1))
                        cy = float(random_uniform_num(py0, py1))
                        if plaza_inset.contains(Point(cx, cy)):
                            center = np.array([cx, cy], dtype=float)
                            break
                    else:
                        continue
                    p_start = center - d * float(0.5 * length)
                    p_end = center + d * float(0.5 * length)
                    w = _segment_wall_poly(p_start, p_end, INNER_WALL_THICKNESS)

                if w is None or w.is_empty:
                    continue

                # Keep wall strictly within plaza (or clipped to plaza for boundary-attached).
                try:
                    w = w.intersection(plaza_poly).buffer(0)
                except Exception:
                    continue

                if w.is_empty or float(w.area) < 1.0:
                    continue

                # Avoid covering too much of the plaza.
                if float(w.area) > 0.18 * float(plaza_poly.area):
                    continue

                # Avoid heavy overlaps with existing inner walls.
                try:
                    if (union_existing is not None) and (not union_existing.is_empty):
                        if float(w.intersection(union_existing).area) > 0.15 * float(w.area):
                            continue
                except Exception:
                    pass

                # Keep a small gap to plaza boundary in the non-attached mode.
                if (not attach_to_boundary):
                    try:
                        if float(w.distance(plaza_poly.boundary)) < 0.3:
                            continue
                    except Exception:
                        pass

                wpoly = w
                break

            if wpoly is not None and (not wpoly.is_empty):
                inner_walls.append(wpoly)
                try:
                    union_existing = unary_union([union_existing, wpoly]).buffer(0)
                except Exception:
                    pass

        return inner_walls

    # Initialize blocking region; this will be updated after we sample inner walls.
    inner_walls = []
    blocking = unary_union(walls + [solid_fill]).buffer(0)

    # Helper for collision validation
    def _pose_is_collision_free(pose_xyz):
        st = State([pose_xyz[0], pose_xyz[1], pose_xyz[2], 0, 0])
        for box in st.create_box():
            try:
                if blocking.intersects(box):
                    return False
            except Exception:
                # If solid is empty or invalid, be conservative
                return False
        return True

    # Region selection by difficulty
    # Normal: both in plaza
    # Complex: one in plaza, one in a corridor
    # Extrem: both in corridors
    mouth_exclusion = None
    try:
        if len(corridors) > 0:
            mouth_parts = []
            mouth_band = float(max(1.0, 0.35 * half_w))
            for c in corridors:
                cp = _as_polygon(c)
                if cp is None or cp.is_empty:
                    continue
                m = plaza.intersection(cp.buffer(mouth_band)).buffer(0)
                if (m is not None) and (not m.is_empty):
                    mouth_parts.append(m)
            if len(mouth_parts) > 0:
                mouth_exclusion = unary_union(mouth_parts).buffer(0)
    except Exception:
        mouth_exclusion = None

    def _sample_in_plaza():
        for _ in range(2000):
            pose = _sample_pose_near_edge(plaza, avoid_poly=None, head_dist=1.0)
            # Prevent plaza slots from being generated at corridor mouths.
            if mouth_exclusion is not None and (not mouth_exclusion.is_empty):
                try:
                    if mouth_exclusion.buffer(0.8).contains(Point(float(pose[0]), float(pose[1]))):
                        continue
                except Exception:
                    pass
            if _pose_is_collision_free(pose):
                return pose
        raise RuntimeError("Failed to sample collision-free plaza pose")

    def _sample_in_corridor():
        if len(corridors) == 0:
            raise RuntimeError("No corridors available for corridor sampling")

        # Exclude the plaza (and a small buffer around it) so "corridor" poses don't land
        # in the corridor-plaza overlap region and visually appear to be in the plaza.
        # Additionally, only accept poses whose heading faces an obstacle wall (blocking),
        # not the open-space mouth toward the plaza.
        # Try a few exclusion radii to keep sampling robust.
        exclude_radii = [float(max(1.0, 0.35 * half_w)), 1.0, 0.5, 0.2, 0.0]

        # Ensure the corridor pose is not right at the mouth: require some distance to plaza.
        # Scale by corridor half-width to move deeper into corridors for wider ones.
        min_plaza_dist = float(max(1.5, 0.6 * half_w))

        for _ in range(1800):
            corr = corridors[int(np.random.randint(0, len(corridors)))]
            corr_poly = _as_polygon(corr)
            if corr_poly is None or corr_poly.is_empty:
                continue

            pose = None
            for r in exclude_radii:
                try:
                    region = corr_poly.difference(plaza.buffer(r)).buffer(0)
                except Exception:
                    region = corr_poly

                if region is None or region.is_empty:
                    continue

                try:
                    pose = _sample_pose_near_edge_facing_blocking(
                        region,
                        blocking,
                        plaza_poly=plaza,
                        min_plaza_dist=min_plaza_dist,
                        head_dist=1.0,
                        max_tries=350,
                    )
                except Exception:
                    pose = None
                    continue

                # Hard reject if the sampled point is still in/near plaza.
                try:
                    if plaza.buffer(r).contains(Point(float(pose[0]), float(pose[1]))):
                        pose = None
                        continue
                except Exception:
                    pass

                break

            if pose is None:
                continue
            if _pose_is_collision_free(pose):
                return pose

        raise RuntimeError("Failed to sample collision-free corridor pose")

    # -------------------------
    # Sample inner walls + start/dest with connectivity checks
    # -------------------------
    start = None
    dest = None

    def _wrap_pi(a: float) -> float:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _abs_angle_diff(a: float, b: float) -> float:
        return abs(_wrap_pi(a - b))

    def _difficulty_constraints(level: str):
        # Match `src/debug/check_navigation_difficulty.py`
        if level == "Normal":
            return (0.0, 30.0), (0.0, 45.0)
        if level == "Complex":
            return (30.0, 50.0), (45.0, 90.0)
        if level == "Extrem":
            return (50.0, None), (60.0, 180.0)
        # Default: no constraints
        return (0.0, None), (0.0, 180.0)

    def _pair_satisfy_constraints(s_pose, d_pose) -> bool:
        (dmin, dmax), (amin, amax) = _difficulty_constraints(str(map_level))
        dist = float(np.hypot(float(s_pose[0]) - float(d_pose[0]), float(s_pose[1]) - float(d_pose[1])))
        ad = float(np.rad2deg(_abs_angle_diff(float(s_pose[2]), float(d_pose[2]))))

        # Extra slot-spacing guard: avoid too-close parking slots that create narrow divider gaps.
        min_slot_center_dist = float(max(6.0, 2.5 * WIDTH))
        if dist < min_slot_center_dist:
            return False

        if dist < float(dmin) - 1e-6:
            return False
        if dmax is not None and dist > float(dmax) + 1e-6:
            return False
        if ad < float(amin) - 1e-6:
            return False
        if ad > float(amax) + 1e-6:
            return False
        return True

    # Approximate clearance for reachability checks (vehicle width + tiny margin)
    reach_clearance = float(WIDTH / 2.0 + 0.2)

    for _wall_attempt in range(36):
        # Resample walls (fixed count)
        inner_walls = _sample_inner_walls(plaza, [])

        try:
            blocking = unary_union(walls + [solid_fill] + list(inner_walls)).buffer(0)
        except Exception:
            blocking = unary_union(walls + [solid_fill]).buffer(0)
            inner_walls = []

        # Reachability check on (drivable - inner_walls), shrunk by clearance
        try:
            free_region = drivable
            if len(inner_walls) > 0:
                free_region = free_region.difference(unary_union(inner_walls)).buffer(0)
            nav_region = free_region.buffer(-reach_clearance).buffer(0)
        except Exception:
            nav_region = None

        if nav_region is None or nav_region.is_empty:
            start, dest = None, None
            continue

        # Sample poses with updated collision constraints.
        # Enforce difficulty constraints by resampling start/dest until satisfied.
        start, dest = None, None
        for _pose_attempt in range(500):
            try:
                if map_level == 'Normal':
                    s = _sample_in_plaza()
                    d = _sample_in_plaza()
                elif map_level == 'Complex':
                    if random() < 0.5:
                        s = _sample_in_plaza()
                        d = _sample_in_corridor()
                    else:
                        s = _sample_in_corridor()
                        d = _sample_in_plaza()
                else:  # Extrem
                    if random() < 0.5:
                        s = _sample_in_plaza()
                        d = _sample_in_corridor()
                    else:
                        s = _sample_in_corridor()
                        d = _sample_in_plaza()
            except Exception:
                continue

            if not _pair_satisfy_constraints(s, d):
                continue

            if _grid_reachable(nav_region, (s[0], s[1]), (d[0], d[1]), resolution=0.6):
                start, dest = s, d
                break

        if start is not None and dest is not None:
            break

    if start is None or dest is None:
        # Last resort: drop inner walls to guarantee a solvable scene.
        inner_walls = []
        blocking = unary_union(walls + [solid_fill]).buffer(0)

        try:
            nav_region = drivable.buffer(-reach_clearance).buffer(0)
        except Exception:
            nav_region = None

        start, dest = None, None
        for _pose_attempt in range(700):
            try:
                if map_level == 'Normal':
                    s = _sample_in_plaza()
                    d = _sample_in_plaza()
                elif map_level == 'Complex':
                    if random() < 0.5:
                        s = _sample_in_plaza()
                        d = _sample_in_corridor()
                    else:
                        s = _sample_in_corridor()
                        d = _sample_in_plaza()
                else:
                    if random() < 0.5:
                        s = _sample_in_plaza()
                        d = _sample_in_corridor()
                    else:
                        s = _sample_in_corridor()
                        d = _sample_in_plaza()
            except Exception:
                continue

            if not _pair_satisfy_constraints(s, d):
                continue

            if nav_region is None or nav_region.is_empty or _grid_reachable(nav_region, (s[0], s[1]), (d[0], d[1]), resolution=0.6):
                start, dest = s, d
                break

        if start is None or dest is None:
            # Absolute last resort: return a pair even if constraints can't be met.
            # (Should be extremely rare; mainly prevents hard failures in production.)
            if map_level == 'Normal':
                start = _sample_in_plaza()
                dest = _sample_in_plaza()
            elif map_level == 'Complex':
                if random() < 0.5:
                    start = _sample_in_plaza()
                    dest = _sample_in_corridor()
                else:
                    start = _sample_in_corridor()
                    dest = _sample_in_plaza()
            else:
                if random() < 0.5:
                    start = _sample_in_plaza()
                    dest = _sample_in_corridor()
                else:
                    start = _sample_in_corridor()
                    dest = _sample_in_plaza()

    # Build slot-like divider walls near start/dest.
    # Divider walls are perpendicular to the nearby straight wall edge,
    # and extend with equal spacing toward both sides until edge ends.
    def _build_pose_boxes(pose_xyz):
        st = State([pose_xyz[0], pose_xyz[1], pose_xyz[2], 0, 0])
        return st.create_box()

    def _find_pose_support_edge(pose_xyz):
        """Find the nearest straight boundary edge that the pose is facing."""
        px, py, yaw = float(pose_xyz[0]), float(pose_xyz[1]), float(pose_xyz[2])
        p0 = np.array([px, py], dtype=float)
        heading = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)

        ray = LineString([tuple(p0), tuple(p0 + heading * 45.0)])
        hit_points = []
        try:
            inter = ray.intersection(drivable.boundary)
            hit_points = _extract_points(inter)
        except Exception:
            hit_points = []
        if len(hit_points) == 0:
            return None

        # Nearest forward boundary hit along heading direction.
        best_hit = None
        best_proj = None
        for pt in hit_points:
            v = np.array([float(pt.x) - p0[0], float(pt.y) - p0[1]], dtype=float)
            proj = float(v[0] * heading[0] + v[1] * heading[1])
            if proj <= 1e-6:
                continue
            if best_proj is None or proj < best_proj:
                best_proj = proj
                best_hit = np.array([float(pt.x), float(pt.y)], dtype=float)
        if best_hit is None:
            return None

        boundary_polys = [plaza] + [c for c in corridors]
        best = None
        best_dist = None
        for poly_raw in boundary_polys:
            poly = _as_polygon(poly_raw)
            if poly is None or poly.is_empty:
                continue
            coords = list(poly.exterior.coords)
            for i in range(max(0, len(coords) - 1)):
                a = np.array(coords[i], dtype=float)
                b = np.array(coords[i + 1], dtype=float)
                seg = b - a
                seg_len = float(np.linalg.norm(seg))
                if seg_len < 1e-6:
                    continue
                seg_u = seg / seg_len
                s = float(np.clip(np.dot(best_hit - a, seg_u), 0.0, seg_len))
                foot = a + seg_u * s
                d = float(np.linalg.norm(foot - best_hit))
                if best_dist is None or d < best_dist:
                    inward, _outward = _edge_normals(poly, a, b)
                    best = {
                        'p1': a,
                        'p2': b,
                        's': s,
                        'len': seg_len,
                        'inward': inward,
                    }
                    best_dist = d
        return best

    def _build_divider_walls_for_pose(pose_xyz, occupied_boxes: list):
        edge = _find_pose_support_edge(pose_xyz)
        if edge is None:
            return []

        p1 = edge['p1']
        p2 = edge['p2']
        s0 = float(edge['s'])
        seg_len = float(edge['len'])
        t = _unit(p2 - p1)
        inward = _unit(edge['inward'])

        # ~two parking spots between the nearest pair of divider walls.
        slot_pitch = float(max(2.8, WIDTH + 1.0))
        edge_margin = float(max(0.8, 0.35 * slot_pitch))
        if map_level == 'Normal':
            divider_depth = float(LENGTH / 2.0)
        elif map_level == 'Complex':
            divider_depth = float(3.0 * LENGTH / 4.0)
        else:  # Extrem
            divider_depth = float(1.2 * LENGTH)
        max_per_side = 4

        s_values = []
        for sign in (+1.0, -1.0):
            s = s0 + sign * slot_pitch
            n_side = 0
            while edge_margin <= s <= (seg_len - edge_margin):
                s_values.append(float(s))
                s += sign * slot_pitch
                n_side += 1
                if n_side >= max_per_side:
                    break

        s_values = sorted(set([round(v, 3) for v in s_values]), key=lambda x: abs(float(x) - s0))
        divider_pairs = []

        for s in s_values:
            q = p1 + t * float(s)
            q2 = q + inward * divider_depth
            w = _segment_wall_poly(q, q2, INNER_WALL_THICKNESS)
            if w is None or w.is_empty:
                continue
            try:
                w = w.intersection(drivable).buffer(0)
            except Exception:
                continue
            if w.is_empty or float(w.area) < 0.3:
                continue

            # Keep start/dest collision free.
            invalid = False
            for box in occupied_boxes:
                try:
                    if w.intersects(box):
                        invalid = True
                        break
                except Exception:
                    invalid = True
                    break
            if invalid:
                continue

            if len(divider_pairs) > 0:
                try:
                    u = unary_union([it[1] for it in divider_pairs]).buffer(0)
                    if float(w.intersection(u).area) > 0.25 * float(w.area):
                        continue
                except Exception:
                    pass

            divider_pairs.append((abs(float(s) - s0), w))

        divider_pairs.sort(key=lambda x: x[0])
        return [it[1] for it in divider_pairs]

    pose_boxes = _build_pose_boxes(start) + _build_pose_boxes(dest)
    start_slot_walls = _build_divider_walls_for_pose(start, pose_boxes)
    dest_slot_walls = _build_divider_walls_for_pose(dest, pose_boxes)
    slot_walls = list(start_slot_walls) + list(dest_slot_walls)

    # Global minimum gap between divider walls to avoid over-narrow parking slots.
    min_wall_gap = float(max(1.8, WIDTH - 0.1))
    filtered_slot_walls = []
    for w in slot_walls:
        too_close = False
        for ww in filtered_slot_walls:
            try:
                if float(w.distance(ww)) < min_wall_gap:
                    too_close = True
                    break
            except Exception:
                continue
        if not too_close:
            filtered_slot_walls.append(w)
    slot_walls = filtered_slot_walls

    if len(slot_walls) > 0:
        # Try full walls first, then relaxed set (nearest pair for each pose), then none.
        candidate_sets = [
            slot_walls,
            list(start_slot_walls[:2]) + list(dest_slot_walls[:2]),
            [],
        ]
        inner_walls = []
        for cand in candidate_sets:
            try:
                trial_nav = drivable
                if len(cand) > 0:
                    trial_nav = trial_nav.difference(unary_union(cand)).buffer(0)
                trial_nav = trial_nav.buffer(-reach_clearance).buffer(0)
            except Exception:
                continue

            if trial_nav is None or trial_nav.is_empty:
                continue
            if _grid_reachable(trial_nav, (start[0], start[1]), (dest[0], dest[1]), resolution=0.6):
                inner_walls = list(cand)
                break
    else:
        inner_walls = []

    try:
        blocking = unary_union(walls + [solid_fill] + list(inner_walls)).buffer(0)
    except Exception:
        blocking = unary_union(walls + [solid_fill]).buffer(0)
        inner_walls = []

    # Return obstacles list (shapely geometries): walls + solid fill + inner walls.
    obstacles = walls + [solid_fill] + list(inner_walls)
    if return_regions:
        return start, dest, obstacles, {
            'plaza': plaza,
            'corridors': corridors,
            'drivable': drivable,
        }
    return start, dest, obstacles


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
        
        # Keep the full scene strictly inside 80m x 80m
        self.xmin = -40.0
        self.xmax = 40.0
        self.ymin = -40.0
        self.ymax = 40.0
        
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
