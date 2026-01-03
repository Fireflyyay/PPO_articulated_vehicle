
import sys
sys.path.append("..")
sys.path.append(".")
import time
import os
from shutil import copyfile
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent
from env.car_parking_base import CarParking
from env.vehicle import VALID_SPEED, Status
from configs import *

# Simplified SceneChoose
class SceneChoose():
    def __init__(self) -> None:
        self.scene_types = {0:'Normal', 
                            1:'Complex',
                            2:'Extrem',
                            }
        self.target_success_rate = np.array([0.95, 0.95, 0.9])
        self.success_record = {}
        for scene_name in self.scene_types.values():
            self.success_record[scene_name] = []
        self.scene_record = []
        
    def choose_case(self,):
        # Randomly choose a scene type
        scene_id = np.random.randint(0, len(self.scene_types))
        self.scene_record.append(scene_id)
        return self.scene_types[scene_id]
    
    def update_success_record(self, success:int):
        if len(self.scene_record) > 0:
            scene_id = self.scene_record[-1]
            self.success_record[self.scene_types[scene_id]].append(success)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None) 
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    verbose = args.verbose

    if args.visualize:
        env = CarParking(fps=100, verbose=verbose,)
    else:
        env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')
    
    scene_chooser = SceneChoose()

    # the path to log and save model
    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/exp/ppo_%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    # configs log
    if os.path.exists('./configs.py'):
        copyfile('./configs.py', save_path+'configs.txt')
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    seed = SEED
    # env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    }

    rl_agent = PPO(configs)
    checkpoint_path = args.agent_ckpt
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    parking_agent = ParkingAgent(rl_agent)

    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    succ_record = []
    best_success_rate = [0, 0, 0]

    for i in range(args.train_episode):
        scene_chosen = scene_chooser.choose_case()
        obs, _ = env.reset(options={'level': scene_chosen})
        parking_agent.reset()
        
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        
        while not done:
            step_num += 1
            action, log_prob = parking_agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if 'reward_info' in info:
                reward_info.append(list(info['reward_info'].values()))
            
            total_reward += reward
            reward_per_state_list.append(reward)
            
            # Store transition in memory
            # obs, action, reward, done, log_prob, next_obs
            parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            
            obs = next_obs
            
            # Update agent
            if len(parking_agent.agent.memory) % parking_agent.agent.configs.batch_size == 0 and len(parking_agent.agent.memory) >= parking_agent.agent.configs.batch_size:
                if verbose and i % 10 == 0 and step_num == 1: # Print less frequently
                    print("Updating the agent.")
                actor_loss, critic_loss = parking_agent.agent.update()
                writer.add_scalar("actor_loss", actor_loss, i)
                writer.add_scalar("critic_loss", critic_loss, i)
            
            if done:
                if info['status']==Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)

        writer.add_scalar("total_reward", total_reward, i)
        if len(reward_per_state_list) > 0:
            writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        
        # Log std if available
        if hasattr(parking_agent.agent, 'log_std'):
            writer.add_scalar("action_std0", parking_agent.agent.log_std.detach().cpu().numpy().reshape(-1)[0],i)
            if parking_agent.agent.log_std.shape[0] > 1:
                writer.add_scalar("action_std1", parking_agent.agent.log_std.detach().cpu().numpy().reshape(-1)[1],i)
        
        for type_id in scene_chooser.scene_types:
            scene_name = scene_chooser.scene_types[type_id]
            if len(scene_chooser.success_record[scene_name]) > 0:
                writer.add_scalar("success_rate_%s"%scene_name,
                    np.mean(scene_chooser.success_record[scene_name][-100:]), i)
        
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        
        if len(reward_info) > 0:
            reward_info_sum = np.sum(np.array(reward_info), axis=0)
            reward_info_sum = np.round(reward_info_sum, 2)
            reward_info_list.append(list(reward_info_sum))

        if verbose and i%10==0 and i>0:
            print('success rate:',np.sum(succ_record[-100:]),'/',len(succ_record[-100:]))
            if hasattr(parking_agent.agent, 'log_std'):
                print('std:', parking_agent.agent.log_std.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            if len(parking_agent.agent.actor_loss_list) > 0:
                print('loss:', np.mean(parking_agent.agent.actor_loss_list[-100:]),np.mean(parking_agent.agent.critic_loss_list[-100:]))
            print("")

        # save best model
        success_rates = []
        for type_id in scene_chooser.scene_types:
            scene_name = scene_chooser.scene_types[type_id]
            if len(scene_chooser.success_record[scene_name]) > 0:
                success_rates.append(np.mean(scene_chooser.success_record[scene_name][-100:]))
            else:
                success_rates.append(0)
        
        # Simple logic for best model: if average success rate is better
        avg_success = np.mean(success_rates)
        if avg_success >= np.mean(best_success_rate) and i>100:
            best_success_rate = success_rates
            parking_agent.agent.save("%s/PPO_best.pt" % (save_path),params_only=True)
            with open(save_path+'best.txt', 'w') as f_best_log:
                f_best_log.write('epoch: %s, success rate: %s'%(i+1, success_rates))
        
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

