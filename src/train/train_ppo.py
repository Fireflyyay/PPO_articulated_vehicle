
import sys
import os
# Ensure src is in path regardless of CWD
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import time
from shutil import copyfile
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent, PrimitivePlanner
from env.car_parking_base import CarParking
from env.vehicle import VALID_SPEED, Status
from configs import *

# Primitives imports
if USE_MOTION_PRIMITIVES:
    from primitives.library import load_library
    from env.wrappers.macro_action_wrapper import MacroActionWrapper

class SceneChoose:
    """Failure-driven curriculum sampler (ported from HOPE).

    Strategy:
    - Warm-up: pick scenes to balance coverage (uniform by count).
    - After enough history: with 50% probability, sample scenes biased toward
      those whose recent success rate lags behind a target.
    """

    def __init__(self) -> None:
        self.scene_types = {
            0: 'Normal',
            1: 'Complex',
            2: 'Extrem',
        }

        # target success rates (can be tuned)
        self.target_success_rate = np.array([0.95, 0.95, 0.90], dtype=np.float64)

        # success_record indexed by scene_id
        self.success_record = {sid: [] for sid in self.scene_types.keys()}
        self.scene_record = []

        # curriculum parameters
        self.history_horizon = 200
        self.recent_window = 250

    def choose_case(self):
        if len(self.scene_record) < self.history_horizon:
            scene_id = self._choose_case_uniform()
        else:
            if np.random.random() > 0.5:
                scene_id = self._choose_case_worst_perform()
            else:
                scene_id = self._choose_case_uniform()

        self.scene_record.append(int(scene_id))
        return self.scene_types[int(scene_id)]

    def update_success_record(self, success: int):
        if len(self.scene_record) == 0:
            return
        sid = int(self.scene_record[-1])
        self.success_record[sid].append(int(success))

    def _choose_case_uniform(self):
        case_count = np.zeros(len(self.scene_types), dtype=np.int64)
        for i in range(min(len(self.scene_record), self.history_horizon)):
            sid = int(self.scene_record[-(i + 1)])
            case_count[sid] += 1
        return int(np.argmin(case_count))

    def _choose_case_worst_perform(self):
        success_rate = []
        for sid in sorted(self.scene_types.keys()):
            recent = self.success_record[sid][-min(self.recent_window, len(self.success_record[sid])) :]
            if len(recent) == 0:
                success_rate.append(0.0)
            else:
                success_rate.append(float(np.mean(recent)))

        fail_rate = self.target_success_rate - np.array(success_rate, dtype=np.float64)
        fail_rate = np.clip(fail_rate, 0.01, 1.0)
        fail_rate = fail_rate / np.sum(fail_rate)
        return int(np.random.choice(np.arange(len(fail_rate)), p=fail_rate))

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None) 
    parser.add_argument('--train_episode', type=int, default=50000)
    parser.add_argument('--eval_episode', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    verbose = args.verbose

    if args.visualize:
        base_env = CarParking(fps=100, verbose=verbose,)
    else:
        base_env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    
    # Use Motion Primitives Wrapper
    env = base_env
    if USE_MOTION_PRIMITIVES:
        print(f"Using Motion Primitives from {PRIMITIVE_LIBRARY_PATH}")
        # Resolve path: assume PRIMITIVE_LIBRARY_PATH is relative to src/configs.py
        # We need absolute path or relative to CWD.
        # If running from root, and path is "../data/...", it might be wrong if logic assumes relative to src.
        # Let's try to find it.
        # If path starts with .., assume relative to src folder
        current_dir = os.path.dirname(os.path.abspath(__file__)) # src/train
        src_dir = os.path.dirname(current_dir) # src
        root_dir = os.path.dirname(src_dir) # root
        project_root = root_dir
        
        # In configs.py, path is "../data/..." relative to configs.py location (src/)
        # So it points to root/data/...
        
        # Let's resolve relative to src_dir
        lib_full_path = os.path.normpath(os.path.join(src_dir, PRIMITIVE_LIBRARY_PATH))

        if not os.path.exists(lib_full_path):
             # Try relative to CWD if failed
             if os.path.exists(PRIMITIVE_LIBRARY_PATH):
                 lib_full_path = PRIMITIVE_LIBRARY_PATH
             else:
                 # Try typical location
                 lib_full_path = os.path.join(project_root, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))

        primitive_lib = load_library(lib_full_path)
        primitive_h = getattr(primitive_lib, 'horizon', PRIMITIVE_H)
        env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)
        print(f"Wrapped env with MacroActionWrapper. Action space: {env.action_space.n} primitives. H={primitive_h}")

    scene_chooser = SceneChoose()

    # the path to log and save model
    # Use src/log/exp directory
    current_dir = os.path.dirname(os.path.abspath(__file__)) # src/train
    src_dir = os.path.dirname(current_dir) # src
    log_exp_dir = os.path.join(src_dir, 'log', 'exp')

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = os.path.join(log_exp_dir, 'ppo_%s/' % timestamp)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    # configs log
    if os.path.exists('./src/configs.py'):
        copyfile('./src/configs.py', save_path+'configs.txt')
    elif os.path.exists('./configs.py'):
        copyfile('./configs.py', save_path+'configs.txt')
    
    # More robust tensorboard command for Python 3.8 environments
    print(f"You can track the training process with:\n  python -m tensorboard --logdir {os.path.abspath(save_path)}\nThen open http://localhost:6006 in your browser.")
    
    seed = SEED
    # env.seed(seed)
    
    # Fix for gym seeding
    # env.action_space.seed(seed) 
    # Wrapper might not forward logic or attribute
    if hasattr(env.action_space, 'seed'):
        env.action_space.seed(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Update Output Size for Discrete
    # NOTE: keep per-run copies to avoid mutating global configs.
    actor_params = dict(ACTOR_CONFIGS)
    critic_params = dict(CRITIC_CONFIGS)

    if USE_MOTION_PRIMITIVES:
        actor_params['output_size'] = env.action_space.n
        # Discrete policy uses logits; do NOT tanh-clip.
        actor_params['use_tanh_output'] = False
        # Critic input dim doesn't change (observation same)
    else:
        actor_params['output_size'] = env.action_space.shape[0]
        actor_params['use_tanh_output'] = True
    
    configs = {
        "discrete": USE_MOTION_PRIMITIVES,
        "observation_shape": env.observation_shape if hasattr(env, 'observation_shape') else base_env.observation_shape,
        # env.observation_shape might not be exposed by wrapper. 
        # CarParking has observation_shape attribute. 
        # Gym Wrapper forwards getattr usually, but let's be safe.
        
        "action_dim": env.action_space.n if USE_MOTION_PRIMITIVES else env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian", # This might be ignored if discrete is True in Agent
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
        "action_std_init": 1.5, # Increased from 0.6
        "action_std_decay_rate": 0.0002, # Decreased from 0.001 to slow down decay
        "min_action_std": 0.1,
        # Ensure gamma is consistent with macro-action horizon
        "gamma": (GAMMA_BASE ** primitive_h) if USE_MOTION_PRIMITIVES else GAMMA,
    }

    rl_agent = PPO(configs, discrete=USE_MOTION_PRIMITIVES)
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    primitive_planner = PrimitivePlanner() if USE_MOTION_PRIMITIVES else None
    parking_agent = ParkingAgent(rl_agent, planner=primitive_planner)

    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    succ_record = []
    best_success_rate = [0.0, 0.0, 0.0]

    for i in range(args.train_episode):
        scene_chosen = scene_chooser.choose_case()
        obs, _ = env.reset(options={'level': scene_chosen})
        parking_agent.reset()
        
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []

        # Takeover statistics (per-episode)
        ep_takeover_steps = 0
        ep_takeover_triggered = 0
        ep_takeover_used = False
        ep_plan_ms = []
        ep_prune_ms = []
        ep_score_ms = []
        # action distributions
        n_actions = env.action_space.n if USE_MOTION_PRIMITIVES else None
        ep_action_counts = np.zeros((n_actions,), dtype=np.int64) if n_actions is not None else None
        ep_takeover_action_counts = np.zeros((n_actions,), dtype=np.int64) if n_actions is not None else None
        
        while not done:
            step_num += 1
            action_mask = None
            if USE_MOTION_PRIMITIVES and USE_ACTION_MASK and hasattr(env, 'get_action_mask'):
                action_mask = env.get_action_mask()
            action, log_prob = parking_agent.choose_action(obs, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Takeover accounting (macro-step level)
            takeover_active = bool(info.get('takeover_active', False))
            if takeover_active:
                ep_takeover_steps += 1
                ep_takeover_used = True
            if bool(info.get('takeover_triggered', False)):
                ep_takeover_triggered = 1

            dbg = info.get('takeover_debug', None)
            if isinstance(dbg, dict):
                if 'plan_ms' in dbg:
                    ep_plan_ms.append(float(dbg['plan_ms']))
                if 'fast_prune_ms' in dbg:
                    ep_prune_ms.append(float(dbg['fast_prune_ms']))
                if 'score_ms' in dbg:
                    ep_score_ms.append(float(dbg['score_ms']))

            if ep_action_counts is not None:
                try:
                    pid = int(info.get('primitive_id', action))
                    ep_action_counts[pid] += 1
                    if takeover_active:
                        ep_takeover_action_counts[pid] += 1
                except Exception:
                    pass
            
            if 'reward_info' in info:
                reward_info.append(list(info['reward_info'].values()))
            
            total_reward += reward
            reward_per_state_list.append(reward)
            
            # Store transition in memory
            # obs, action, reward, done, log_prob, next_obs
            # Optional: skip takeover transitions to reduce teacher bias.
            store_this = True
            if USE_MOTION_PRIMITIVES:
                if (not TAKEOVER_TEACHER_FORCING) and takeover_active:
                    store_this = False

            if store_this:
                if USE_MOTION_PRIMITIVES:
                    parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs, action_mask))
                else:
                    parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            
            obs = next_obs
            
            # Update agent
            if len(parking_agent.agent.memory) % parking_agent.agent.configs.batch_size == 0 and len(parking_agent.agent.memory) >= parking_agent.agent.configs.batch_size:
                if verbose and i % 10 == 0 and step_num == 1: # Print less frequently
                    print("Updating the agent.")
                actor_loss, critic_loss = parking_agent.agent.update()
                
                # Decay action std
                parking_agent.agent.decay_action_std(
                    parking_agent.agent.configs.action_std_decay_rate, 
                    parking_agent.agent.configs.min_action_std
                )
                
                writer.add_scalar("actor_loss", actor_loss, i)
                writer.add_scalar("critic_loss", critic_loss, i)
            
            if done:
                if info['status'] == Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)

            # Terminal takeover plan (motion primitives)
            if USE_MOTION_PRIMITIVES and info.get('path_to_dest', None) is not None:
                parking_agent.set_planner_path(info['path_to_dest'])

        writer.add_scalar("total_reward", total_reward, i)
        if len(reward_per_state_list) > 0:
            writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        
        # Log std
        writer.add_scalar("action_std", parking_agent.agent.action_std, i)

        # Takeover logging
        if USE_MOTION_PRIMITIVES:
            writer.add_scalar("takeover/triggered", float(ep_takeover_triggered), i)
            writer.add_scalar("takeover/step_ratio", float(ep_takeover_steps) / float(max(1, step_num)), i)
            writer.add_scalar("takeover/steps", float(ep_takeover_steps), i)

            if len(ep_plan_ms) > 0:
                writer.add_scalar("takeover/plan_ms_mean", float(np.mean(ep_plan_ms)), i)
            if len(ep_prune_ms) > 0:
                writer.add_scalar("takeover/fast_prune_ms_mean", float(np.mean(ep_prune_ms)), i)
            if len(ep_score_ms) > 0:
                writer.add_scalar("takeover/score_ms_mean", float(np.mean(ep_score_ms)), i)

            success = 1.0 if (len(succ_record) > 0 and succ_record[-1] == 1) else 0.0
            writer.add_scalar("takeover/success_when_used", success if ep_takeover_used else 0.0, i)

            # policy vs planner distribution divergence (episode-level KL)
            if ep_action_counts is not None and ep_takeover_action_counts is not None:
                if ep_takeover_action_counts.sum() > 0 and ep_action_counts.sum() > 0:
                    p = (ep_takeover_action_counts + 1e-6) / float(ep_takeover_action_counts.sum() + 1e-6 * ep_takeover_action_counts.size)
                    q = (ep_action_counts + 1e-6) / float(ep_action_counts.sum() + 1e-6 * ep_action_counts.size)
                    kl = float(np.sum(p * (np.log(p) - np.log(q))))
                    writer.add_scalar("takeover/action_KL", kl, i)
        
        for type_id, scene_name in scene_chooser.scene_types.items():
            rec = scene_chooser.success_record[int(type_id)]
            if len(rec) > 0:
                writer.add_scalar(
                    "success_rate_%s" % scene_name,
                    float(np.mean(rec[-100:])),
                    i,
                )
        
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        
        if len(reward_info) > 0:
            reward_info_arr = np.array(reward_info, dtype=np.float64)
            reward_info_sum = np.round(np.sum(reward_info_arr, axis=0), 4)
            reward_info_list.append(list(reward_info_sum))

            # Log reward components dynamically (HOPE-style keys)
            try:
                reward_keys = list(info.get('reward_info', {}).keys())
            except Exception:
                reward_keys = []

            for idx, name in enumerate(reward_keys):
                if idx >= len(reward_info_sum):
                    break
                writer.add_scalar(f"reward_component/{name}", float(reward_info_sum[idx]), i)

        if verbose and i%10==0 and i>0:
            print('success rate:',np.sum(succ_record[-100:]),'/',len(succ_record[-100:]))
            print('std:', parking_agent.agent.action_std)
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            if len(parking_agent.agent.actor_loss_list) > 0:
                print('loss:', np.mean(parking_agent.agent.actor_loss_list[-100:]),np.mean(parking_agent.agent.critic_loss_list[-100:]))
            # Print reward component summary if available
            if len(reward_info_list) > 0:
                try:
                    keys = list(info.get('reward_info', {}).keys())
                    vals = reward_info_list[-1]
                    msg = ', '.join([f"{k}={vals[j]:.4f}" for j, k in enumerate(keys) if j < len(vals)])
                    print('reward components:', msg)
                except Exception:
                    pass
            print("")

        # save best model (scene-wise, HOPE-style): only save when each scene is not worse.
        success_rates = []
        for sid in sorted(scene_chooser.scene_types.keys()):
            rec = scene_chooser.success_record[int(sid)]
            success_rates.append(float(np.mean(rec[-100:])) if len(rec) > 0 else 0.0)

        if i > 100:
            improved_all = True
            for k in range(len(best_success_rate)):
                if success_rates[k] + 1e-12 < float(best_success_rate[k]):
                    improved_all = False
                    break

            if improved_all:
                best_success_rate = list(success_rates)
                parking_agent.agent.save("%s/PPO_best.pt" % (save_path), params_only=True)
                with open(save_path + 'best.txt', 'w') as f_best_log:
                    f_best_log.write('epoch: %s, success rate: %s' % (i + 1, success_rates))
        
        if (i+1) % 2000 == 0:
            parking_agent.agent.save("%s/PPO2_%s.pt" % (save_path, i),params_only=True)

        if verbose and i%10==0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0,j-50):j+1]) for j in range(len(reward_list))]
            plt.figure()
            plt.plot(episodes,reward_list)
            plt.plot(episodes,mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.title(f'Training Reward (Episode {i})')
            plt.savefig('%s/reward.png'%save_path)
            plt.close()
            
            # Print progress
            print(f"Episode {i}/{args.train_episode} | Reward: {total_reward:.2f} | Steps: {step_num} | Success Rate: {np.mean(succ_record[-100:]):.2f}")
            sys.stdout.flush()

