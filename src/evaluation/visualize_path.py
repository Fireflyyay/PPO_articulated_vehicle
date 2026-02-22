import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from shapely.geometry import LinearRing
from typing import Optional
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.car_parking_base import CarParking
from model.agent.ppo_agent import PPOAgent as PPO
from configs import *


def _find_checkpoint(default_path: str) -> Optional[str]:
    if os.path.exists(default_path):
        return default_path
    exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ckpt'))
    if not os.path.exists(exp_dir):
        return None
    for root, dirs, files in os.walk(exp_dir):
        if 'PPO_best.pt' in files:
            return os.path.join(root, 'PPO_best.pt')
    return None


def _infer_actor_output_size(checkpoint: object) -> Optional[int]:
    """Infer actor output size from checkpoint state_dict.

    Works with checkpoints saved by PPOAgent.save(params_only=True).
    """
    if not isinstance(checkpoint, dict):
        return None
    actor_sd = checkpoint.get('actor_net')
    if not isinstance(actor_sd, dict):
        return None
    # MultiObsEmbedding last linear is net.4
    w = actor_sd.get('net.4.weight')
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    # Fallback: last weight-like tensor
    weight_tensors = [v for k, v in actor_sd.items() if k.endswith('weight') and isinstance(v, torch.Tensor) and v.ndim == 2]
    if not weight_tensors:
        return None
    return int(weight_tensors[-1].shape[0])

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

def visualize(episodes: int = 10):
    # Setup base environment
    base_env = CarParking(render_mode='rgb_array')
    
    # Locate checkpoint
    default_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ckpt/PPO_best.pt'))
    checkpoint_path = _find_checkpoint(default_ckpt)
    if checkpoint_path is None:
        print(f"No checkpoint found under {os.path.dirname(default_ckpt)}. Exiting.")
        return

    # Peek checkpoint to infer whether it is discrete (macro-actions) and the actor output size.
    try:
        ckpt_obj = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except Exception:
        ckpt_obj = torch.load(checkpoint_path, map_location='cpu')
    inferred_actor_out = _infer_actor_output_size(ckpt_obj)

    # Decide whether we should use macro-action wrapper.
    # If actor outputs more than 2 dims, it's almost certainly discrete primitives.
    use_macro_actions = (inferred_actor_out is not None and inferred_actor_out > 2)

    env = base_env
    primitive_h = PRIMITIVE_H
    if use_macro_actions:
        from primitives.library import load_library
        from env.wrappers.macro_action_wrapper import MacroActionWrapper

        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        lib_full_path = os.path.normpath(os.path.join(src_dir, PRIMITIVE_LIBRARY_PATH))
        if not os.path.exists(lib_full_path) and os.path.exists(PRIMITIVE_LIBRARY_PATH):
            lib_full_path = PRIMITIVE_LIBRARY_PATH
        primitive_lib = load_library(lib_full_path)
        primitive_h = getattr(primitive_lib, 'horizon', PRIMITIVE_H)
        env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
        print(f"Using MacroActionWrapper: action_space.n={env.action_space.n}, H={primitive_h}")

    # For plotting/trajectory access, always use the underlying base env.
    plot_env = base_env

    # Setup agent (match training-time logic)
    actor_params = dict(ACTOR_CONFIGS)
    critic_params = dict(CRITIC_CONFIGS)
    obs_shape = env.observation_shape if hasattr(env, 'observation_shape') else base_env.observation_shape
    actor_params['input_dim'] = int(obs_shape[0])
    critic_params['input_dim'] = int(obs_shape[0])

    if use_macro_actions:
        actor_params['output_size'] = env.action_space.n
    else:
        actor_params['output_size'] = env.action_space.shape[0]

    configs = {
        "discrete": use_macro_actions,
        "observation_shape": obs_shape,
        "action_dim": env.action_space.n if use_macro_actions else env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
        "load_params": True,
        "gamma": (GAMMA_BASE ** primitive_h) if use_macro_actions else GAMMA,
    }

    agent = PPO(configs, discrete=use_macro_actions, load_params=True)
    agent.load(checkpoint_path, params_only=True)
    print(f"Loaded checkpoint from {checkpoint_path}")

    img_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../img'))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        
        # Run episode
        while not done:
            action, _ = agent.choose_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if len(plot_env.vehicle.trajectory) > 2000: # Safety break
                break
        
        states = plot_env.vehicle.trajectory
        
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot obstacles
        for area in plot_env.map.obstacles:
            x, y = area.shape.xy
            ax.plot(x, y, color='black', linewidth=2)
            ax.fill(x, y, color='gray', alpha=0.4)
            
        # Plot target (destination)
        dest_front, dest_rear = plot_env.map.dest.create_box()
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
        ax.set_xlim(plot_env.map.xmin, plot_env.map.xmax)
        ax.set_ylim(plot_env.map.ymin, plot_env.map.ymax)
        ax.set_title(f"Articulated Vehicle Path Planning - Episode {i+1}")
        ax.grid(True, linestyle=':', alpha=0.5)
        
        save_path = os.path.join(img_dir, f"path_planning_{i+1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    visualize(episodes=args.episodes)
