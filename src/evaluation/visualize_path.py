import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from shapely.geometry import LinearRing

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.car_parking_base import CarParking
from model.agent.ppo_agent import PPOAgent as PPO
from configs import *

def plot_vehicle(ax, state, alpha=0.3, is_final=False):
    front_box, rear_box = state.create_box()
    
    # Plot front box
    x, y = front_box.xy
    color_front = 'blue' if not is_final else 'darkblue'
    ax.plot(x, y, color=color_front, alpha=alpha, linewidth=1)
    ax.fill(x, y, color=color_front, alpha=alpha/2)
    
    # Plot rear box
    x, y = rear_box.xy
    color_rear = 'red' if not is_final else 'darkred'
    ax.plot(x, y, color=color_rear, alpha=alpha, linewidth=1)
    ax.fill(x, y, color=color_rear, alpha=alpha/2)

def visualize():
    # Setup environment
    env = CarParking(render_mode='rgb_array')
    
    # Setup agent
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
        "load_params": True
    }
    
    agent = PPO(configs, load_params=True)
    
    # Load checkpoint - using the latest one provided in the context
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../log/exp/ppo_20260102_235422/PPO_best.pt'))
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Try to find any PPO_best.pt if the specific one doesn't exist
        print(f"Checkpoint not found at {checkpoint_path}, searching for alternatives...")
        exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../log/exp'))
        found = False
        if os.path.exists(exp_dir):
            for root, dirs, files in os.walk(exp_dir):
                if 'PPO_best.pt' in files:
                    checkpoint_path = os.path.join(root, 'PPO_best.pt')
                    agent.load(checkpoint_path)
                    print(f"Loaded alternative checkpoint from {checkpoint_path}")
                    found = True
                    break
        if not found:
            print("No checkpoint found. Exiting.")
            return

    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../img'))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i in range(10):
        obs, _ = env.reset()
        done = False
        
        # Run episode
        while not done:
            action, _ = agent.choose_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if len(env.vehicle.trajectory) > 2000: # Safety break
                break
        
        states = env.vehicle.trajectory
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot obstacles
        for area in env.map.obstacles:
            x, y = area.shape.xy
            ax.plot(x, y, color='black', linewidth=2)
            ax.fill(x, y, color='gray', alpha=0.4)
            
        # Plot target (destination)
        dest_front, dest_rear = env.map.dest.create_box()
        xf, yf = dest_front.xy
        ax.plot(xf, yf, color='green', linestyle='--', linewidth=2, label='Target Front')
        xr, yr = dest_rear.xy
        ax.plot(xr, yr, color='darkgreen', linestyle='--', linewidth=2, label='Target Rear')
        
        # Plot path (trajectory of the front center)
        path_x = [s.loc.x for s in states]
        path_y = [s.loc.y for s in states]
        ax.plot(path_x, path_y, color='cyan', linestyle='-', alpha=0.6, linewidth=1, label='Path')
        
        # Plot vehicle at intervals
        # We want to show about 10 intermediate states
        num_intermediate = 10
        interval = max(1, len(states) // num_intermediate)
        for j in range(0, len(states), interval):
            plot_vehicle(ax, states[j], alpha=0.2)
        
        # Plot start state
        plot_vehicle(ax, states[0], alpha=0.5)
        
        # Plot final state
        plot_vehicle(ax, states[-1], alpha=1.0, is_final=True)
        
        ax.set_aspect('equal')
        ax.set_xlim(env.map.xmin, env.map.xmax)
        ax.set_ylim(env.map.ymin, env.map.ymax)
        ax.set_title(f"Articulated Vehicle Path Planning - Episode {i+1}")
        ax.grid(True, linestyle=':', alpha=0.5)
        
        save_path = os.path.join(img_dir, f"path_planning_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    visualize()
