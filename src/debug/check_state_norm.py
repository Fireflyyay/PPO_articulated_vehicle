import sys
import os
import torch
import numpy as np
import gymnasium as gym

# Add src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.env.car_parking_base import CarParking
from src.primitives.library import load_library
from src.env.wrappers.macro_action_wrapper import MacroActionWrapper
from src.model.agent.ppo_agent import PPOAgent
from src.configs import *

def check_state_norm():
    print("Initializing environment...")
    
    # Setup Env
    base_env = CarParking(fps=100, verbose=False, render_mode=None)
    
    # Wrapper
    if USE_MOTION_PRIMITIVES:
        if os.path.exists(PRIMITIVE_LIBRARY_PATH):
            lib_path = PRIMITIVE_LIBRARY_PATH
        else:
            lib_path = os.path.join(project_root, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))
        
        if not os.path.exists(lib_path):
             # Try H4
             lib_path = os.path.join(project_root, "data", "primitives_articulated_H4_S11.npz")

        if os.path.exists(lib_path):
            print(f"Loading primitives from {lib_path}")
            primitive_lib = load_library(lib_path)
            primitive_h = getattr(primitive_lib, 'horizon', PRIMITIVE_H)
            env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
        else:
            print("Primitives not found, using base env")
            env = base_env
    else:
        env = base_env

    # Setup Agent Config
    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS

    # Ensure discrete output size if using primitives
    if USE_MOTION_PRIMITIVES and hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
        actor_params['output_size'] = env.action_space.n
    
    # Override config to FORCE state_norm=True for testing
    configs = {
        "discrete": USE_MOTION_PRIMITIVES,
        "observation_shape": base_env.observation_shape,
        "action_dim": env.action_space.n if USE_MOTION_PRIMITIVES else env.action_space.shape[0],
        "state_norm": True,  # Force True
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    
    print("Creating Agent with state_norm=True...")
    agent = PPOAgent(configs, discrete=USE_MOTION_PRIMITIVES)
    
    # Check if state_normalize exists
    if hasattr(agent, 'state_normalize'):
        print("  [OK] Agent has 'state_normalize' attribute.")
        # Fix: access state_mean/state_std directly as per StateNorm implementation
        # For flat obs, it's in a dict under 'default' key OR flat.
        # Let's inspect
        sn = agent.state_normalize
        if sn.flat_obs:
            print(f"  Initial mean (flat): {sn.state_mean['default'][:5] if 'default' in sn.state_mean else sn.state_mean}")
        else:
            print(f"  Initial mean (dict): {sn.state_mean}")
    else:
        print("  [FAIL] Agent missing 'state_normalize'. Check PPOConfig.")
        return

    print("\nCollecting observations to update normalization stats...")
    obs, _ = env.reset()
    
    # Collect some data to update stats
    observations_list = []
    
    for i in range(100):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # In PPO, we push (obs, action, reward, done, log_prob, next_obs)
        # The agent.push_memory calls deepcopy then state_norm(obs)
        # Let's simulate push memory
        
        # We need dummy log_prob
        log_prob = 0.0
        
        agent.push_memory((obs, action, reward, done, log_prob, next_obs))
        
        observations_list.append(obs)
        obs = next_obs
        if done:
            obs, _ = env.reset()

    print(f"Collected {len(observations_list)} samples.")
    
    # Check updated stats
    print("\nStats after 100 steps (via push_memory updates):")
    # For flat obs, inside 'default' key likely
    sn = agent.state_normalize
    if sn.flat_obs:
        mean = sn.state_mean['default']
        std = sn.state_std['default']
    else:
        # Assuming dict
        mean = sn.state_mean
        std = sn.state_std
        
    print(f"  New Mean (first 5): {mean[:5]}")
    print(f"  New Std (first 5): {std[:5]}")
    
    # Verify values are changing
    if np.allclose(mean, 0.0) and np.allclose(std, 1.0):
         print("  [WARNING] Stats did not change! Are they being updated?")
    else:
         print("  [OK] Stats updated.")

    # Test normalization effect
    raw_obs = observations_list[0]
    norm_obs = agent.state_normalize.state_norm(raw_obs, update=False)
    
    print("\nNormalization Check (Sample 0):")
    print(f"  Raw (first 5): {raw_obs[:5]}")
    print(f"  Norm (first 5): {norm_obs[:5]}")
    
    # Manually calc
    manual_norm = (raw_obs - mean) / (std + 1e-8)
    diff = np.abs(norm_obs - manual_norm).max()
    print(f"  Difference between internal norm and manual calc: {diff}")
    
    if diff < 1e-5:
        print("  [OK] Normalization calculation is correct.")
    else:
        print("  [FAIL] Normalization calculation mismatch!")

if __name__ == "__main__":
    check_state_norm()
