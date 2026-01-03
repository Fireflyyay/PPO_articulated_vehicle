from typing import Callable, List
from enum import Enum
import copy

import numpy as np
from shapely.geometry import Point, LinearRing
from shapely.affinity import affine_transform

from configs import *



class Status(Enum):
    CONTINUE = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5


class State:
    def __init__(self, raw_state: list):
        self.loc: Point = Point(raw_state[:2])
        self.heading: float = raw_state[2]
        if len(raw_state) == 3:
            self.speed: float = 0
            self.steering: float = 0
        else:
            self.speed: float = raw_state[3]
            self.steering: float = raw_state[4]
        self.rear_heading: float = raw_state[5] if len(raw_state) > 5 else self.heading

    def create_box(self) -> List[LinearRing]:
        # Front Box
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat_front = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        front_box = affine_transform(FrontVehicleBox, mat_front)

        # Rear Box
        if not hasattr(self, 'trailer_loc'):
             l1 = HITCH_OFFSET
             l2 = TRAILER_LENGTH
             hx = self.loc.x - l1 * np.cos(self.heading)
             hy = self.loc.y - l1 * np.sin(self.heading)
             tx = hx - l2 * np.cos(self.rear_heading)
             ty = hy - l2 * np.sin(self.rear_heading)
             self.trailer_loc = Point(tx, ty)

        cos_theta_r = np.cos(self.rear_heading)
        sin_theta_r = np.sin(self.rear_heading)
        mat_rear = [cos_theta_r, -sin_theta_r, sin_theta_r, cos_theta_r, self.trailer_loc.x, self.trailer_loc.y]
        rear_box = affine_transform(RearVehicleBox, mat_rear)
        
        return [front_box, rear_box]

    def get_pos(self,):
        return (self.loc.x, self.loc.y, self.heading)


class KSModel(object):
    """Update the state of a vehicle by the Kinematic Single-Track Model.

    Kinematic Single-Track Model use the vehicle's current speed, heading, location, 
    acceleration, and velocity of steering angle as input. Then it returns the estimation of 
    speed, heading, steering angle and location after a small time step.

    Use the center of vehicle's rear wheels as the origin of local coordinate system.

    Assume the vehicle is front-wheel-only drive.
    """
    def __init__(
        self, 
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list
    ):
        self.wheel_base = wheel_base
        self.step_len = step_len
        self.n_step = n_step
        self.speed_range = speed_range
        self.angle_range = angle_range
        self.mini_iter = 20


    def step(self, state: State, action: list, step_time:int=NUM_STEP) -> State:
        """Update the state of a vehicle with the Kinematic Single-Track Model.

        Args:
            state (list): [x, y, car_angle, speed, steering]
            action (list): [steer, speed].
            step (float, optional): the step length for each simulation.
            n_step (int): number of step of updating the physical state. This value is decide by
                (physics simulation step length : rendering step length).

        """
        new_state = copy.deepcopy(state)
        x, y = new_state.loc.x, new_state.loc.y
        steer, speed = action
        new_state.steering = steer
        new_state.speed = speed
        new_state.speed = np.clip(new_state.speed, *self.speed_range)
        new_state.steering = np.clip(new_state.steering, *self.angle_range)

        for _ in range(step_time):
            for _ in range(self.mini_iter):
                x += new_state.speed * np.cos(new_state.heading) * self.step_len/self.mini_iter
                y += new_state.speed * np.sin(new_state.heading) * self.step_len/self.mini_iter
                new_state.heading += \
                    new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len/self.mini_iter 

        new_state.loc = Point(x, y)
        return new_state


class ArticulatedKSModel(KSModel):
    """Kinematic model for an Articulated Steering Vehicle (e.g., Wheel Loader).

    Unlike a tractor-trailer, this vehicle steers by actively changing the articulation angle
    (hinge angle) between the front and rear bodies. The front wheels do not steer relative
    to the front body.

    Kinematics:
        gamma (articulation angle) = steering input
        omega_f = v * sin(gamma) / (Lf * cos(gamma) + Lr)
        theta_r = theta_f - gamma

    Args:
        hitch_offset (float): Distance from Front Axle to Hinge (Lf).
        trailer_length (float): Distance from Hinge to Rear Axle (Lr).
    """
    def __init__(
        self,
        wheel_base: float,
        step_len: float,
        n_step: int,
        speed_range: list,
        angle_range: list,
        trailer_length: float = 3.0,
        hitch_offset: float = 0.0,
    ):
        super().__init__(wheel_base, step_len, n_step, speed_range, angle_range)
        self.trailer_length = trailer_length # Lr
        self.hitch_offset = hitch_offset     # Lf

    def step(self, state: State, action: list, step_time:int=NUM_STEP) -> State:
        new_state = copy.deepcopy(state)
        omega, speed = action
        
        # Clip inputs
        new_state.speed = np.clip(speed, *self.speed_range)
        new_state.steering = np.clip(omega, *self.angle_range)
        
        omega = new_state.steering
        v = new_state.speed
        
        l1 = self.hitch_offset
        l2 = self.trailer_length
        phi_max = np.deg2rad(36) # Limit from paper

        dt = self.step_len / self.mini_iter

        for _ in range(step_time):
            for _ in range(self.mini_iter):
                theta1 = new_state.heading
                theta2 = new_state.rear_heading
                
                # Calculate current phi
                phi = theta1 - theta2
                # Normalize phi to [-pi, pi]
                phi = (phi + np.pi) % (2 * np.pi) - np.pi
                
                # Check limits
                effective_omega = omega
                if phi >= phi_max and omega > 0:
                    effective_omega = 0
                elif phi <= -phi_max and omega < 0:
                    effective_omega = 0
                
                # Kinematics
                # theta1_dot = (v * sin(phi) + l2 * omega) / (l1 * cos(phi) + l2)
                denom = l1 * np.cos(phi) + l2
                if abs(denom) < 1e-6:
                    denom = 1e-6

                theta1_dot = (v * np.sin(phi) + l2 * effective_omega) / denom
                theta2_dot = theta1_dot - effective_omega
                
                x_dot = v * np.cos(theta1)
                y_dot = v * np.sin(theta1)
                
                # Update
                new_state.loc = Point(
                    new_state.loc.x + x_dot * dt,
                    new_state.loc.y + y_dot * dt
                )
                new_state.heading += theta1_dot * dt
                new_state.rear_heading += theta2_dot * dt
        
        # Update trailer_loc based on geometry
        # Hinge position O
        hx = new_state.loc.x - l1 * np.cos(new_state.heading)
        hy = new_state.loc.y - l1 * np.sin(new_state.heading)
        
        # Rear Axle position O2
        tx = hx - l2 * np.cos(new_state.rear_heading)
        ty = hy - l2 * np.sin(new_state.rear_heading)
        new_state.trailer_loc = Point(tx, ty)

        return new_state


class Vehicle:
    """_summary_
    """
    def __init__(
        self,
        wheel_base: float = WHEEL_BASE,
        step_len: float = STEP_LENGTH,
        n_step: int = NUM_STEP,
        speed_range: list = VALID_SPEED, 
        angle_range: list = VALID_STEER,
        articulated: bool = False,
        trailer_length: float = 3.0,
        hitch_offset: float = 0.0,
    ) -> None:

        self.initial_state: list = None
        self.state: State = None
        self.box: LinearRing = None
        self.trajectory: List[State] = []
        if articulated:
            self.kinetic_model = ArticulatedKSModel(wheel_base, step_len, n_step, speed_range, angle_range, trailer_length, hitch_offset)
        else:
            self.kinetic_model: Callable = KSModel(wheel_base, step_len, n_step, speed_range, angle_range)

        self.color = COLOR_POOL[0]
        self.v_max = None
        self.v_min = None


    def reset(self, initial_state: State):
        """
        Args:
            init_pos (list): [x0, y0, theta0]
        """
        self.initial_state = initial_state
        self.state = self.initial_state
        # self.color = random.sample(COLOR_POOL, 1)[0]
        self.v_max = self.initial_state.speed
        self.v_min = self.initial_state.speed
        self.boxes = self.state.create_box()
        self.box = self.boxes[0] # Keep self.box as front box for compatibility if needed, or just for centroid access
        self.trajectory.clear()
        self.trajectory.append(self.state)
        self.tmp_trajectory = self.trajectory.copy()

    def step(self, action: np.ndarray, step_time: int=NUM_STEP):
        """
        Args:
            action (list): [steer, speed]
        """
        prev_info = copy.deepcopy((self.state, self.boxes, self.v_max, self.v_min))
        self.state = self.kinetic_model.step(self.state, action, step_time)
        self.boxes = self.state.create_box()
        self.box = self.boxes[0]
        self.trajectory.append(self.state)
        self.tmp_trajectory.append(self.state)
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min
        return prev_info

    def retreat(self, prev_info):
        '''
        Retreat the vehicle state from previous one.

        Args:
            prev_info (tuple): (state, box, v_max, v_min)
        '''
        self.state, self.boxes, self.v_max, self.v_min = prev_info
        self.box = self.boxes[0]
        self.trajectory.pop(-1)
