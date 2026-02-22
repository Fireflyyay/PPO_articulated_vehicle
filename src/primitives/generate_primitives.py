import os
import numpy as np
import argparse

from configs import VALID_SPEED, VALID_STEER
from env.vehicle import Vehicle, State

def generate_primitives(H, S, output_path):
    print(f"Generating primitives with H={H}, S={S}...")
    
    # Define action candidates
    # Steer (omega/gamma_dot) levels
    steer_min, steer_max = VALID_STEER
    steers = np.linspace(steer_min, steer_max, S)
    
    # Speed levels: Forward and Backward
    speed_min, speed_max = VALID_SPEED
    # Create levels: -max, -half, half, max
    # e.g. -2.5, -1.25, 1.25, 2.5
    # speeds = [speed_min, speed_min/2, speed_max/2, speed_max]
    # Let's be explicit
    speeds = [-2.5, -1.0, 1.0, 2.5]
    
    # Generate all combinations of constant actions
    # Action = (steer, speed) held constant for H steps
    
    primitives_actions = []
    primitives_deltas = []
    
    # Create a dummy vehicle for simulation
    # We need to set up initial state such that we can measure delta.
    # Initial state: x=0, y=0, theta=0, v=0, gamma=0 (theta1=theta2)
    # Note: State expects raw_state list.
    # raw_state layout: [x, y, theta, speed, steering, rear_heading]
    # steering in State is actually gamma (articulation angle) or steer angle?
    # In ArticulatedKSModel, new_state.steering seems to be used as 'omega' (input) but also stored in state.
    # But wait, ArticulatedKSModel.step:
    # omega, speed = action
    # new_state.steering = omega
    # So state.steering stores the last applied input (omega).
    
    # FOR DELTA CALCULATION:
    # We want delta relative to the vehicle frame.
    # Delta x, Delta y in vehicle frame. Delta theta, Delta gamma.
    
    for speed in speeds:
        for steer in steers:
            # Skip if speed is 0
            if abs(speed) < 0.01:
                continue
                
            # Initialize vehicle at origin
            # x=0, y=0, theta=0, v=0, omega=0, rear_theta=0 (so gamma=0)
            init_state_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
            init_state = State(init_state_list)
            
            # Use articulated=True
            vehicle = Vehicle(articulated=True)
            vehicle.reset(init_state)
            
            # Apply action for H steps
            # Action = [steer, speed]
            action = np.array([steer, speed])
            
            trajectory = []
            valid = True
            
            for _ in range(H):
                # We need to pass step_time=1 (sim 1 step) and loop H times 
                # OR pass step_time=H. 
                # Primitive is a sequence of actions. Here action is constant.
                # Let's step 1 by 1 to simulate trajectory if needed, 
                # but KSModel.step supports step_time.
                # However, we want to store the sequence of actions [H, 2].
                # If we step once with step_time=H, we get final state.
                
                # Let's use step_time=1 in a loop to double check validity at each step if we had checks.
                # But here we just assume constant action.
                
                vehicle.step(action, step_time=1)
                
                # Check validity if possible (e.g. constraints)
                # ArticulatedKSModel already clamps inputs and handles phi limits (gamma limits).
                # So the resulting state is valid kinematically.
                
            final_state = vehicle.state
            
            # Calculate delta
            # Delta should be in the frame of the start state (which is identity)
            # dx = final_x - 0
            # dy = final_y - 0
            # BUT we want it in the local frame of the start.
            # Start was at 0,0,0. So global xy is local xy.
            
            dx = final_state.loc.x
            dy = final_state.loc.y
            dtheta = final_state.heading
            
            # Gamma delta?
            # Gamma = theta_front - theta_rear
            # Initial gamma = 0 - 0 = 0.
            # Final gamma = final.heading - final.rear_heading
            # So dgamma = Final gamma.
            
            gamma = final_state.heading - final_state.rear_heading
            # Normalize gamma?
            gamma = (gamma + np.pi) % (2 * np.pi) - np.pi
            
            delta = np.array([dx, dy, dtheta, gamma])
            
            # Create action sequence [H, 2]
            action_seq = np.tile(action, (H, 1))
            
            # Filter trivial primitives? (e.g. no movement)
            if np.linalg.norm([dx, dy]) < 0.01 and abs(dtheta) < 0.01:
                continue
                
            primitives_actions.append(action_seq)
            primitives_deltas.append(delta)

    # Convert to numpy
    primitives_actions = np.array(primitives_actions) # [N, H, 2]
    primitives_deltas = np.array(primitives_deltas)   # [N, 4]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, actions=primitives_actions, deltas=primitives_deltas, 
                        meta={'H': H, 'S': S})
    print(f"Saved {len(primitives_actions)} primitives to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults from requirements
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--S", type=int, default=11)
    
    args = parser.parse_args()
    
    output_file = f"src/data/primitives_articulated_H{args.H}_S{args.S}.npz"
    # Ensure src/data exists? The path in config is "../data/..." relative to src/config.py?
    # User said: "data/primitives_articulated_H{H}_S{S}.npz" in B1.
    # And config: "USE_MOTION_PRIMITIVES = True", "PRIMITIVE_LIBRARY_PATH = '../data/primitives_articulated_H20_S11.npz'"
    # If config is in src/configs.py, then '../data' is src/../data = data/ (root data).
    # But wait, create_file created src/primitives.
    # Where should data be? 'PPO_articulated_vehicle/data/'.
    
    # Let's target PPO_articulated_vehicle/data
    root_data_path = os.path.join(os.path.dirname(__file__), "../../data")
    output_path = os.path.join(root_data_path, f"primitives_articulated_H{args.H}_S{args.S}.npz")
    
    generate_primitives(args.H, args.S, output_path)
