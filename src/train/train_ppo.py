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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent
from env.car_parking_base import CarParking
from env.vehicle import Status
from configs import *


class SceneChoose:
    """Failure-driven curriculum sampler."""

    def __init__(self) -> None:
        self.scene_types = {
            0: 'Normal',
            1: 'Complex',
            2: 'Extrem',
        }
        self.target_success_rate = np.array([0.95, 0.95, 0.90], dtype=np.float64)
        self.success_record = {sid: [] for sid in self.scene_types.keys()}
        self.scene_record = []

        self.history_horizon = 200
        self.recent_window = 250
        self.extrem_success_band = tuple(float(x) for x in EXTREM_SUCCESS_BAND)
        self.extrem_focus_prob = float(EXTREM_SUCCESS_BAND_FOCUS_PROB)
        self.extrem_bridge_prob = float(EXTREM_SUCCESS_BAND_BRIDGE_PROB)

    def choose_case(self):
        if len(self.scene_record) < self.history_horizon:
            scene_id = self._choose_case_uniform()
        else:
            scene_id = self._choose_case_success_band()
            if scene_id is None and np.random.random() > 0.5:
                scene_id = self._choose_case_worst_perform()
            elif scene_id is None:
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
            recent = self.success_record[sid][-min(self.recent_window, len(self.success_record[sid])):]
            if len(recent) == 0:
                success_rate.append(0.0)
            else:
                success_rate.append(float(np.mean(recent)))

        fail_rate = self.target_success_rate - np.array(success_rate, dtype=np.float64)
        fail_rate = np.clip(fail_rate, 0.01, 1.0)
        fail_rate = fail_rate / np.sum(fail_rate)
        return int(np.random.choice(np.arange(len(fail_rate)), p=fail_rate))

    def _recent_success_rate(self, sid: int) -> float:
        rec = self.success_record[int(sid)]
        if len(rec) == 0:
            return 0.0
        recent = rec[-min(self.recent_window, len(rec)):]
        return float(np.mean(recent)) if len(recent) > 0 else 0.0

    def _choose_case_success_band(self):
        extrem_sid = 2
        complex_sid = 1
        extrem_sr = self._recent_success_rate(extrem_sid)
        low, high = self.extrem_success_band

        if extrem_sr < low:
            if np.random.random() < self.extrem_bridge_prob:
                return int(complex_sid)
            return None

        if extrem_sr <= high:
            if np.random.random() < self.extrem_focus_prob:
                return int(extrem_sid)
            return None

        return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_ckpt', type=str, default=None)
    parser.add_argument('--train_episode', type=int, default=100000)
    parser.add_argument('--eval_episode', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()

    verbose = args.verbose

    if args.visualize:
        env = CarParking(fps=100, verbose=verbose)
    else:
        env = CarParking(fps=100, verbose=verbose, render_mode='rgb_array')

    scene_chooser = SceneChoose()

    log_exp_dir = os.path.join(src_dir, 'log', 'exp')
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = os.path.join(log_exp_dir, 'ppo_%s/' % timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    if os.path.exists('./src/configs.py'):
        copyfile('./src/configs.py', save_path + 'configs.txt')
    elif os.path.exists('./configs.py'):
        copyfile('./configs.py', save_path + 'configs.txt')

    print(
        f"You can track the training process with:\n"
        f"  python -m tensorboard --logdir {os.path.abspath(save_path)}\n"
        f"Then open http://localhost:6006 in your browser."
    )

    seed = SEED
    if hasattr(env.action_space, 'seed'):
        env.action_space.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    actor_params = dict(ACTOR_CONFIGS)
    critic_params = dict(CRITIC_CONFIGS)
    actor_params['output_size'] = env.action_space.shape[0]
    actor_params['use_tanh_output'] = True
    critic_params['output_size'] = 1
    critic_params['use_tanh_output'] = False

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
        "action_std_init": 1.5,
        "action_std_decay_rate": 0.0002,
        "min_action_std": 0.1,
        "gamma": GAMMA,
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
    best_success_rate = [0.0, 0.0, 0.0]

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

            if 'reward_info' in info and isinstance(info['reward_info'], dict):
                reward_info.append(list(info['reward_info'].values()))

            total_reward += reward
            reward_per_state_list.append(reward)

            parking_agent.agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs

            if len(parking_agent.agent.memory) % parking_agent.agent.configs.batch_size == 0 and len(parking_agent.agent.memory) >= parking_agent.agent.configs.batch_size:
                if verbose and i % 10 == 0 and step_num == 1:
                    print("Updating the agent.")
                actor_loss, critic_loss = parking_agent.agent.update()

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

        writer.add_scalar("total_reward", total_reward, i)
        if len(reward_per_state_list) > 0:
            writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        writer.add_scalar("action_std", parking_agent.agent.action_std, i)

        for type_id, scene_name in scene_chooser.scene_types.items():
            rec = scene_chooser.success_record[int(type_id)]
            if len(rec) > 0:
                writer.add_scalar("success_rate_%s" % scene_name, float(np.mean(rec[-100:])), i)

        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)

        if len(reward_info) > 0:
            reward_info_sum = np.sum(np.array(reward_info), axis=0)
            reward_info_sum = np.round(reward_info_sum, 4)
            reward_info_list.append(list(reward_info_sum))

            reward_keys = list(REWARD_WEIGHT.keys())
            for idx, name in enumerate(reward_keys):
                if idx >= len(reward_info_sum):
                    break
                writer.add_scalar(f"reward_component/{name}", float(reward_info_sum[idx]), i)

        if verbose and i % 10 == 0 and i > 0:
            print('success rate:', np.sum(succ_record[-100:]), '/', len(succ_record[-100:]))
            print('std:', parking_agent.agent.action_std)
            print("episode:%s  average reward:%s" % (i, np.mean(reward_list[-50:])))
            if len(parking_agent.agent.actor_loss_list) > 0:
                print('loss:', np.mean(parking_agent.agent.actor_loss_list[-100:]), np.mean(parking_agent.agent.critic_loss_list[-100:]))
            if len(reward_info_list) > 0:
                try:
                    keys = list(REWARD_WEIGHT.keys())
                    vals = reward_info_list[-1]
                    msg = ', '.join([f"{k}={vals[j]:.4f}" for j, k in enumerate(keys) if j < len(vals)])
                    print('reward components:', msg)
                except Exception:
                    pass
            print("")

        success_rates = []
        for type_id in scene_chooser.scene_types:
            rec = scene_chooser.success_record[type_id]
            if len(rec) > 0:
                success_rates.append(np.mean(rec[-100:]))
            else:
                success_rates.append(0.0)

        avg_success = np.mean(success_rates)
        if avg_success >= np.mean(best_success_rate) and i > 100:
            best_success_rate = list(success_rates)
            parking_agent.agent.save("%s/PPO_best.pt" % save_path, params_only=True)
            with open(save_path + 'best.txt', 'w') as f_best_log:
                f_best_log.write('epoch: %s, success rate: %s' % (i + 1, success_rates))

        if (i + 1) % 2000 == 0:
            parking_agent.agent.save("%s/PPO2_%s.pt" % (save_path, i), params_only=True)

        if verbose and i % 10 == 0:
            episodes = [j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0, j - 50):j + 1]) for j in range(len(reward_list))]
            plt.figure()
            plt.plot(episodes, reward_list)
            plt.plot(episodes, mean_reward)
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.title(f'Training Reward (Episode {i})')
            plt.savefig('%s/reward.png' % save_path)
            plt.close()

            print(f"Episode {i}/{args.train_episode} | Reward: {total_reward:.2f} | Steps: {step_num} | Success Rate: {np.mean(succ_record[-100:]):.2f}")
            sys.stdout.flush()

