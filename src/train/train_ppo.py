
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

import configs as cfg

from model.agent.ppo_agent import PPOAgent as PPO
from model.agent.parking_agent import ParkingAgent, PrimitivePlanner
from env.car_parking_base import CarParking
from env.vehicle import VALID_SPEED, Status
from configs import *

# Primitives imports
if USE_MOTION_PRIMITIVES:
    from primitives.library import load_library
    from env.wrappers.macro_action_wrapper import MacroActionWrapper

# Adaptive primitive expansion imports (kept optional)
if USE_MOTION_PRIMITIVES:
    try:
        from primitives.adaptive_library_manager import AdaptivePrimitiveLibraryManager
        from primitives.trajectory_miner import EpisodeTrace, TrajectoryMiner
        from primitives.primitive_pruner import PrimitivePruner
        from train.adaptive_primitive_scheduler import AdaptivePrimitiveScheduler
        from reward.shaping_from_discovered_primitives import DiscoveredPrimitiveShaping
        from model.agent.ppo_agent import expand_discrete_actor_output
    except Exception:
        AdaptivePrimitiveLibraryManager = None
        EpisodeTrace = None
        TrajectoryMiner = None
        PrimitivePruner = None
        AdaptivePrimitiveScheduler = None
        DiscoveredPrimitiveShaping = None
        expand_discrete_actor_output = None


def _scene_is_hard(scene_name: str) -> bool:
    return str(scene_name) in ("Complex", "Extrem", "Extreme")


def _safe_mean(xs):
    if xs is None or len(xs) == 0:
        return 0.0
    return float(np.mean(xs))


def _to_scalar(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        pass
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size > 0:
            return float(arr[0])
    except Exception:
        pass
    return float(default)


def _run_eval_episodes(env, parking_agent, n_episodes: int, scene_schedule: list, deterministic: bool = True):
    """Lightweight evaluation (no learning). Returns dict metrics."""
    succ = []
    succ_extreme = []
    lengths = []
    for k in range(int(n_episodes)):
        scene = scene_schedule[k % len(scene_schedule)]
        obs, _ = env.reset(options={'level': scene})
        parking_agent.reset()
        done = False
        step_num = 0
        while not done:
            step_num += 1
            action_mask = None
            if USE_MOTION_PRIMITIVES and USE_ACTION_MASK and hasattr(env, 'get_action_mask'):
                action_mask = env.get_action_mask(obs)
            action, _ = parking_agent.choose_action(obs, deterministic=deterministic, action_mask=action_mask)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        lengths.append(step_num)
        s = 1 if info.get('status', None) == Status.ARRIVED else 0
        succ.append(s)
        if str(scene) in ("Extrem", "Extreme"):
            succ_extreme.append(s)
    return {
        "success": _safe_mean(succ),
        "success_extreme": _safe_mean(succ_extreme),
        "avg_len": _safe_mean(lengths),
    }


def _collect_rollouts_for_mining(env, parking_agent, n_episodes: int, scene_schedule: list, deterministic: bool, start_episode_id: int = 0):
    """Collect EpisodeTrace list for mining (no learning)."""
    episodes = []
    for k in range(int(n_episodes)):
        scene = scene_schedule[k % len(scene_schedule)]
        obs, _ = env.reset(options={'level': scene})
        parking_agent.reset()

        done = False
        ep_obs = []
        ep_actions = []
        ep_low = []
        ep_rewards = []
        ep_dones = []
        ep_infos = []
        total_reward = 0.0
        takeover_used = False

        while not done:
            ep_obs.append(np.asarray(obs, dtype=np.float64))
            action_mask = None
            if USE_MOTION_PRIMITIVES and USE_ACTION_MASK and hasattr(env, 'get_action_mask'):
                action_mask = env.get_action_mask(obs)
            action, _ = parking_agent.choose_action(obs, deterministic=deterministic, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            takeover_used = takeover_used or bool(info.get('takeover_active', False))
            ep_actions.append(int(info.get('primitive_id', action)))
            tr = info.get('macro_exec_trace', {}) if isinstance(info, dict) else {}
            sub_u = tr.get('sub_actions_phys', None)
            if sub_u is None:
                # fallback: use library primitive actions (may over-estimate executed steps)
                try:
                    sub_u = env.primitive_lib.get_actions(int(action))
                except Exception:
                    sub_u = np.zeros((1, 2), dtype=np.float64)
            ep_low.append(np.asarray(sub_u, dtype=np.float64))

            ep_rewards.append(float(reward))
            ep_dones.append(bool(done))
            ep_infos.append(info if isinstance(info, dict) else {})
            total_reward += float(reward)
            obs = next_obs

        success = bool(ep_infos[-1].get('status', None) == Status.ARRIVED) if len(ep_infos) > 0 else False
        episodes.append(
            EpisodeTrace(
                episode_id=int(start_episode_id + k),
                scene_type=str(scene),
                success=bool(success),
                total_reward=float(total_reward),
                step_count_macro=int(len(ep_actions)),
                takeover_used=bool(takeover_used),
                observations=ep_obs,
                actions_primitive=ep_actions,
                actions_low_level=ep_low,
                rewards=ep_rewards,
                dones=ep_dones,
                infos=ep_infos,
                states_optional=None,
            )
        )
    return episodes

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
    parser.add_argument('--train_episode', type=int, default=100000)
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
        "action_std_decay_rate": 0.0003, # Decreased from 0.001 to slow down decay
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

    # Adaptive primitive expansion components
    adaptive_enabled = bool(USE_MOTION_PRIMITIVES and USE_ADAPTIVE_PRIMITIVE_EXPANSION)
    ap_scheduler = None
    ap_lib_mgr = None
    ap_miner = None
    ap_pruner = None
    ap_shaping = None
    ap_round_id = 0
    ap_last_good_ckpt = None
    ap_last_good_version = None
    post_expand_lr_restore = None

    # mining buffer
    ap_trace_buffer = []
    ap_trace_next_id = 0

    if adaptive_enabled:
        if AdaptivePrimitiveLibraryManager is None:
            print("[adaptive] imports failed; disabling adaptive primitive expansion")
            adaptive_enabled = False
        else:
            ap_scheduler = AdaptivePrimitiveScheduler(cfg)
            ap_lib_mgr = AdaptivePrimitiveLibraryManager(verbose=True)
            ap_lib_mgr.load(base_path=lib_full_path, save_dir=save_path)
            ap_miner = TrajectoryMiner(verbose=True)
            ap_pruner = PrimitivePruner(verbose=True)
            ap_shaping = DiscoveredPrimitiveShaping(cfg) if DiscoveredPrimitiveShaping is not None else None
            ap_last_good_version = ap_lib_mgr.active_version_id
            ap_last_good_ckpt = os.path.join(save_path, "adaptive_primitives", "last_good_agent.pt")
            parking_agent.agent.save(ap_last_good_ckpt, params_only=True)

            print(f"[adaptive] enabled. base_version={ap_last_good_version}, lib_size={ap_lib_mgr.library_size}")

    def run_adaptive_primitive_round(ep_idx: int):
        global env, primitive_lib, primitive_h
        global ap_round_id, ap_last_good_ckpt, ap_last_good_version, post_expand_lr_restore

        if not adaptive_enabled:
            return

        # round id + bookkeeping
        ap_round_id = ap_scheduler.on_round_started(ep_idx)
        writer.add_scalar("adaptive/round_id", float(ap_round_id), ep_idx)
        writer.add_scalar("adaptive/triggered", 1.0, ep_idx)

        old_version = ap_lib_mgr.active_version_id
        old_lib_size = int(env.action_space.n)
        old_lr = float(parking_agent.agent.actor_optimizer.param_groups[0].get('lr', parking_agent.agent.configs.lr_actor))
        post_expand_lr_restore = old_lr

        # Save rollback checkpoint
        ckpt_before = os.path.join(save_path, "adaptive_primitives", f"before_round_{ap_round_id}.pt")
        os.makedirs(os.path.dirname(ckpt_before), exist_ok=True)
        parking_agent.agent.save(ckpt_before, params_only=True)

        # Validation before
        val_schedule = ["Complex", "Extrem"]
        val_before = _run_eval_episodes(env, parking_agent, int(AP_VALIDATION_EPISODES), val_schedule, deterministic=True)
        writer.add_scalar("adaptive/validation_success_before", float(val_before["success"]), ep_idx)
        writer.add_scalar("adaptive/validation_extreme_success_before", float(val_before["success_extreme"]), ep_idx)

        # Collect mining rollouts (bias towards hard scenes)
        mining_schedule = ["Complex", "Extrem", "Complex", "Normal"]
        rollouts = _collect_rollouts_for_mining(
            env,
            parking_agent,
            n_episodes=int(AP_MINING_ROLLOUTS),
            scene_schedule=mining_schedule,
            deterministic=bool(AP_MINING_DETERMINISTIC),
            start_episode_id=int(1000000 + ap_round_id * 10000),
        )

        # Mine
        cands = ap_miner.mine_from_episodes(rollouts, ap_lib_mgr.get_active_library(), cfg)
        writer.add_scalar("adaptive/candidates_raw_count", float(len(cands)), ep_idx)

        # Dedup
        c_dedup = ap_pruner.deduplicate(cands, ap_lib_mgr.get_active_library(), cfg)
        writer.add_scalar("adaptive/candidates_after_dedup", float(len(c_dedup)), ep_idx)

        # Proxy pruning (Phase1 no-op unless user wires env_sampler)
        c_proxy = ap_pruner.prune_by_proxy_value(c_dedup, env_sampler=None, planner_eval_fn=None, config=cfg)
        writer.add_scalar("adaptive/candidates_after_prune", float(len(c_proxy)), ep_idx)

        # Feasibility checks (best-effort)
        c_feas = ap_pruner.validate_feasibility(c_proxy, env, cfg)
        writer.add_scalar("adaptive/candidates_after_feasibility", float(len(c_feas)), ep_idx)

        # Select top-K to add
        remaining = int(AP_MAX_LIBRARY_SIZE) - int(old_lib_size)
        k_add = int(min(AP_MAX_ADD_PER_ROUND, max(0, remaining)))
        add_list = c_feas[:k_add]

        if k_add <= 0 or len(add_list) == 0:
            writer.add_scalar("adaptive/added_count", 0.0, ep_idx)
            writer.add_scalar("adaptive/library_size", float(old_lib_size), ep_idx)
            return

        # Add to manager and persist a new version
        added = ap_lib_mgr.add_candidates(add_list, round_id=int(ap_round_id), config=cfg)
        info = ap_lib_mgr.save_version(save_dir=save_path)
        new_lib = ap_lib_mgr.get_active_library()

        writer.add_scalar("adaptive/added_count", float(added), ep_idx)
        writer.add_scalar("adaptive/library_size", float(new_lib.size), ep_idx)

        writer.add_scalar(
            "adaptive/avg_complexity_added",
            float(np.mean([c.complexity_score for c in add_list])) if len(add_list) > 0 else 0.0,
            ep_idx,
        )
        writer.add_scalar(
            "adaptive/avg_utility_added",
            float(np.mean([c.utility_score for c in add_list])) if len(add_list) > 0 else 0.0,
            ep_idx,
        )
        writer.add_scalar(
            "adaptive/avg_novelty_added",
            float(np.mean([c.novelty_score for c in add_list])) if len(add_list) > 0 else 0.0,
            ep_idx,
        )

        # Expand actor output dim + clear on-policy buffer
        if expand_discrete_actor_output is not None:
            expand_discrete_actor_output(parking_agent.agent, int(new_lib.size), init_mode="random_small")
        parking_agent.agent.memory.clear()

        # Post-expansion stabilization: lower LR + freeze backbone + logit bias for new actions
        try:
            lr_scale = float(AP_POST_EXPAND_LR_SCALE)
            parking_agent.agent.actor_optimizer.param_groups[0]['lr'] = old_lr * lr_scale
        except Exception:
            pass
        try:
            parking_agent.agent.freeze_actor_backbone(True)
        except Exception:
            pass
        try:
            bias = np.zeros((int(new_lib.size),), dtype=np.float32)
            bias[int(old_lib_size) :] = float(AP_NEW_ACTION_LOGIT_BIAS_INIT)
            parking_agent.agent.set_action_logit_bias(bias)
        except Exception:
            pass

        # Update wrapper/library reference (rebuild wrapper is safest)
        primitive_lib = new_lib
        primitive_h = getattr(primitive_lib, 'horizon', primitive_h)
        env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)

        # Update shaping centroids from discovered segments (use end macro obs)
        if ap_shaping is not None and bool(USE_DISCOVERED_PRIMITIVE_SHAPING):
            try:
                by_ep = {int(ep.episode_id): ep for ep in rollouts}
                feats = []
                for c in add_list:
                    eid = int(c.source_metadata.get('episode_id'))
                    ep = by_ep.get(eid, None)
                    if ep is None:
                        continue
                    mt1 = int(c.source_metadata.get('macro_t1', min(len(ep.observations) - 1, 0)))
                    mt1 = max(0, min(mt1, len(ep.observations) - 1))
                    feats.append(ap_shaping.extract_feature_from_obs(ep.observations[mt1]))
                ap_shaping.add_centroids(feats)
            except Exception:
                pass

        # Validation after
        val_after = _run_eval_episodes(env, parking_agent, int(AP_VALIDATION_EPISODES), val_schedule, deterministic=True)
        writer.add_scalar("adaptive/validation_success_after", float(val_after["success"]), ep_idx)
        writer.add_scalar("adaptive/validation_extreme_success_after", float(val_after["success_extreme"]), ep_idx)

        # Rollback if regressed
        rollback = False
        if bool(AP_ENABLE_ROLLBACK):
            drop = float(val_before["success"] - val_after["success"])
            drop_ext = float(val_before["success_extreme"] - val_after["success_extreme"])
            if drop > float(AP_ROLLBACK_DROP_TOL) or drop_ext > float(AP_ROLLBACK_DROP_TOL):
                rollback = True

        if rollback:
            writer.add_scalar("adaptive/rollback", 1.0, ep_idx)
            try:
                ap_lib_mgr.rollback_to(old_version, save_dir=save_path)
            except Exception:
                pass
            # restore wrapper
            primitive_lib = ap_lib_mgr.get_active_library()
            primitive_h = getattr(primitive_lib, 'horizon', primitive_h)
            env = MacroActionWrapper(base_env, primitive_lib, H=primitive_h)

            # restore agent params and action dim
            try:
                expand_discrete_actor_output(parking_agent.agent, int(primitive_lib.size), init_mode="random_small")
            except Exception:
                pass
            try:
                parking_agent.agent.load(ckpt_before, params_only=True)
            except Exception:
                pass
            parking_agent.agent.memory.clear()
            try:
                parking_agent.agent.freeze_actor_backbone(False)
                parking_agent.agent.clear_action_logit_bias()
                parking_agent.agent.actor_optimizer.param_groups[0]['lr'] = old_lr
            except Exception:
                pass
        else:
            writer.add_scalar("adaptive/rollback", 0.0, ep_idx)
            ap_last_good_version = ap_lib_mgr.active_version_id
            try:
                parking_agent.agent.save(ap_last_good_ckpt, params_only=True)
            except Exception:
                pass

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

        # ---- EpisodeTrace buffers (macro-step aligned) ----
        ep_obs_trace = []
        ep_actions_trace = []
        ep_low_actions_trace = []
        ep_rewards_trace = []
        ep_dones_trace = []
        ep_infos_trace = []

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
            ep_obs_trace.append(np.asarray(obs, dtype=np.float64))
            action_mask = None
            if USE_MOTION_PRIMITIVES and USE_ACTION_MASK and hasattr(env, 'get_action_mask'):
                action_mask = env.get_action_mask(obs)
            action, log_prob = parking_agent.choose_action(obs, action_mask=action_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # weak shaping reward from discovered primitives (optional)
            shaping_r = 0.0
            if adaptive_enabled and ap_shaping is not None and bool(USE_DISCOVERED_PRIMITIVE_SHAPING):
                try:
                    shaping_r = float(ap_shaping.reward(obs, next_obs))
                except Exception:
                    shaping_r = 0.0
            reward = float(reward) + float(shaping_r)
            if shaping_r != 0.0:
                writer.add_scalar("adaptive/shaping_reward", float(shaping_r), i)

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
                ri = info.get('reward_info', {})
                if isinstance(ri, dict):
                    row = [_to_scalar(ri.get(k, 0.0), 0.0) for k in REWARD_WEIGHT.keys()]
                    reward_info.append(row)
            
            total_reward += reward
            reward_per_state_list.append(reward)

            # ---- EpisodeTrace step record ----
            try:
                ep_actions_trace.append(int(info.get('primitive_id', action)))
            except Exception:
                ep_actions_trace.append(int(action) if USE_MOTION_PRIMITIVES else -1)
            tr = info.get('macro_exec_trace', {}) if isinstance(info, dict) else {}
            sub_u = tr.get('sub_actions_phys', None)
            if sub_u is None:
                try:
                    sub_u = env.primitive_lib.get_actions(int(action))
                except Exception:
                    sub_u = np.zeros((1, 2), dtype=np.float64)
            ep_low_actions_trace.append(np.asarray(sub_u, dtype=np.float64))
            ep_rewards_trace.append(float(reward))
            ep_dones_trace.append(bool(done))
            ep_infos_trace.append(info if isinstance(info, dict) else {})
            
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
            reward_keys = list(REWARD_WEIGHT.keys())

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

        # ---- Build and store EpisodeTrace for mining ----
        if adaptive_enabled and EpisodeTrace is not None:
            try:
                success = 1 if (len(ep_infos_trace) > 0 and ep_infos_trace[-1].get('status', None) == Status.ARRIVED) else 0
                takeover_used = bool(ep_takeover_used)
                ep_trace = EpisodeTrace(
                    episode_id=int(ap_trace_next_id),
                    scene_type=str(scene_chosen),
                    success=bool(success),
                    total_reward=float(total_reward),
                    step_count_macro=int(len(ep_actions_trace)),
                    takeover_used=bool(takeover_used),
                    observations=ep_obs_trace,
                    actions_primitive=ep_actions_trace,
                    actions_low_level=ep_low_actions_trace,
                    rewards=ep_rewards_trace,
                    dones=ep_dones_trace,
                    infos=ep_infos_trace,
                    states_optional=None,
                )
                ap_trace_next_id += 1

                keep = True
                if bool(AP_TRACE_KEEP_SUCCESS_ONLY) and not bool(ep_trace.success):
                    keep = False
                    if bool(AP_TRACE_KEEP_NEAR_SUCCESS):
                        try:
                            # near-success: last obs dist
                            lidar_n = int(LIDAR_NUM)
                            dist_last = float(ep_trace.observations[-1][lidar_n]) * float(MAX_DIST_TO_DEST)
                            if dist_last <= float(AP_NEAR_SUCCESS_DIST_THR):
                                keep = True
                        except Exception:
                            pass
                if keep:
                    ap_trace_buffer.append(ep_trace)
                    # cap buffer
                    if len(ap_trace_buffer) > int(AP_TRACE_BUFFER_MAX_EPISODES):
                        ap_trace_buffer = ap_trace_buffer[-int(AP_TRACE_BUFFER_MAX_EPISODES) :]
            except Exception:
                pass

        # ---- Trigger adaptive round between episodes ----
        if adaptive_enabled:
            try:
                # recent success rates
                w = int(AP_TRIGGER_WINDOW)
                recent = succ_record[-w:] if len(succ_record) > 0 else []
                sr_recent = float(np.mean(recent)) if len(recent) > 0 else 0.0

                # hard success: use scene_chooser.scene_record aligned with succ_record
                hard = []
                for j in range(1, min(len(scene_chooser.scene_record), len(succ_record), w) + 1):
                    sid = int(scene_chooser.scene_record[-j])
                    scene_name = scene_chooser.scene_types.get(sid, 'Normal')
                    if _scene_is_hard(scene_name):
                        hard.append(int(succ_record[-j]))
                sr_hard = float(np.mean(hard)) if len(hard) > 0 else 0.0

                stats = {
                    "success_rate_recent": sr_recent,
                    "hard_success_rate_recent": sr_hard,
                    "plateau": True,
                }
                writer.add_scalar("adaptive/success_rate_recent", float(sr_recent), i)
                writer.add_scalar("adaptive/hard_success_rate_recent", float(sr_hard), i)

                if ap_scheduler.should_trigger(stats, episode_idx=int(i)):
                    run_adaptive_primitive_round(ep_idx=int(i))
            except Exception as e:
                writer.add_scalar("adaptive/triggered", 0.0, i)
                if verbose:
                    print(f"[adaptive] trigger check failed: {e}")

        # ---- Post-expansion bias decay / unfreeze ----
        if adaptive_enabled and ap_scheduler is not None:
            try:
                remaining = ap_scheduler.tick_post_expand_freeze()
                writer.add_scalar("adaptive/post_expand_freeze_remaining", float(remaining), i)

                # decay new-action bias
                if parking_agent.agent.action_logit_bias is not None:
                    decay_ep = int(AP_NEW_ACTION_LOGIT_BIAS_DECAY_EPISODES)
                    if decay_ep > 0:
                        factor = max(0.0, 1.0 - 1.0 / float(decay_ep))
                        b = parking_agent.agent.action_logit_bias.detach().cpu().numpy()
                        b = b * float(factor)
                        if float(np.max(b)) < 1e-3:
                            parking_agent.agent.clear_action_logit_bias()
                        else:
                            parking_agent.agent.set_action_logit_bias(b)

                # restore lr/unfreeze when freeze window ends
                if remaining <= 0:
                    try:
                        parking_agent.agent.freeze_actor_backbone(False)
                    except Exception:
                        pass
                    if post_expand_lr_restore is not None:
                        try:
                            parking_agent.agent.actor_optimizer.param_groups[0]['lr'] = float(post_expand_lr_restore)
                        except Exception:
                            pass
                        post_expand_lr_restore = None
            except Exception:
                pass

