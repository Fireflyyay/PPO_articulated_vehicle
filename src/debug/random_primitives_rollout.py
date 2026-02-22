import sys
import os
import numpy as np
import gymnasium as gym

# Add src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.env.car_parking_base import CarParking
from src.primitives.library import load_library
from src.env.wrappers.macro_action_wrapper import MacroActionWrapper
from src.configs import *

def random_rollout(num_episodes=100):
    print("Starting Random Rollout Check...")
    
    # Load primitives
    # Try to find the primitive file
    if os.path.exists(PRIMITIVE_LIBRARY_PATH):
        lib_path = PRIMITIVE_LIBRARY_PATH
    else:
        # relative resolve
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        lib_path = os.path.join(project_root, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))
    
    if not os.path.exists(lib_path):
        print(f"Error: Primitive library not found at {lib_path}")
        return

    print(f"Loading primitives from {lib_path}")
    primitive_lib = load_library(lib_path)
    primitive_h = getattr(primitive_lib, 'horizon', PRIMITIVE_H)
    
    # Create Env
    base_env = CarParking(fps=1000, verbose=False, render_mode=None) # faster
    env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
    
    success_count = 0
    collision_count = 0
    outbound_count = 0
    timeout_count = 0
    
    total_steps = 0
    executed_steps_list = []
    rewards_list = []

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            if 'executed_steps' in info:
                executed_steps_list.append(info['executed_steps'])
            
            if done:
                status = info.get('status')
                # Status enum: CONTINUE=1, ARRIVED=2, COLLIDED=3, OUTBOUND=4, OUTTIME=5
                # But info['status'] might be the Enum object or int. 
                # Let's check status.
                
                # In env/vehicle.py: Status is Enum.
                # In env/car_parking_base.py: info['status'] = Status.COLLIDED
                
                if str(status) == 'Status.ARRIVED':
                    success_count += 1
                elif str(status) == 'Status.COLLIDED':
                    collision_count += 1
                elif str(status) == 'Status.OUTBOUND':
                    outbound_count += 1
                elif str(status) == 'Status.OUTTIME':
                    timeout_count += 1
                
        rewards_list.append(episode_reward)
        if (i+1) % 50 == 0:
            print(f"Episode {i+1}: Reward={episode_reward:.2f}, Status={status}")

    print("\n--- Statistics ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Rate: {success_count/num_episodes:.2%}")
    print(f"Collision Rate: {collision_count/num_episodes:.2%}")
    print(f"Outbound Rate: {outbound_count/num_episodes:.2%}")
    print(f"Timeout Rate: {timeout_count/num_episodes:.2%}")
    print(f"Average Reward: {np.mean(rewards_list):.2f}")
    
    # Executed steps analysis (Early stopping)
    executed_steps = np.array(executed_steps_list)
    early_stop_mask = executed_steps < primitive_h
    early_stop_rate = np.mean(early_stop_mask)
    print(f"Macro Early Stop Rate: {early_stop_rate:.2%}")
    if len(executed_steps) > 0:
        print(f"Avg Executed Steps per Macro: {np.mean(executed_steps):.2f} (Target H={primitive_h})")

if __name__ == "__main__":
    random_rollout(200)
