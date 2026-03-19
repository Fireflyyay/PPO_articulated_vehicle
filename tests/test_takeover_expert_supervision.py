import numpy as np
import gymnasium as gym
from gymnasium import spaces

import configs as cfg
from env.wrappers.macro_action_wrapper import MacroActionWrapper
from model.agent.ppo_agent import PPOAgent


class DummyPrimitiveLib:
    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float64)
        self.deltas = np.zeros((self.actions.shape[0], 4), dtype=np.float64)

    @property
    def size(self):
        return int(self.actions.shape[0])

    @property
    def horizon(self):
        return int(self.actions.shape[1])

    def get_actions(self, primitive_id):
        return self.actions[int(primitive_id)]


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        return np.array([0.0]), {}

    def step(self, action):
        return np.array([0.0]), 0.0, False, False, {}


def _make_discrete_ppo(action_dim: int):
    obs_dim = 10
    actor_layers = {
        "input_dim": obs_dim,
        "output_size": int(action_dim),
        "use_tanh_output": False,
        "orthogonal_init": False,
    }
    critic_layers = {
        "input_dim": obs_dim,
        "output_size": 1,
        "use_tanh_output": False,
        "orthogonal_init": False,
    }
    cfg = {
        "observation_shape": (obs_dim,),
        "action_dim": int(action_dim),
        "batch_size": 8,
        "mini_batch": 4,
        "mini_epoch": 1,
        "actor_layers": actor_layers,
        "critic_layers": critic_layers,
        "gamma": 0.99,
        "lr_actor": 1e-3,
        "lr_critic": 1e-3,
        "use_imitation_loss": True,
        "imitation_buffer_size": 16,
        "imitation_batch_size": 4,
        "imitation_min_buffer": 1,
        "imitation_loss_weight": 0.05,
    }
    return PPOAgent(cfg, discrete=True)


def _make_takeover_obs(dist_m: float, rel_heading: float = 0.0, articulation: float = 0.0, min_lidar_m: float = None):
    obs = np.zeros((cfg.LIDAR_NUM + 7 + 2,), dtype=np.float64)
    lidar_m = float(cfg.LIDAR_RANGE if min_lidar_m is None else min_lidar_m)
    obs[: cfg.LIDAR_NUM] = np.clip(lidar_m / float(cfg.LIDAR_RANGE), 0.0, 1.0)

    obs[cfg.LIDAR_NUM + 0] = float(dist_m) / float(cfg.MAX_DIST_TO_DEST)
    obs[cfg.LIDAR_NUM + 1] = 1.0
    obs[cfg.LIDAR_NUM + 2] = 0.0
    obs[cfg.LIDAR_NUM + 3] = np.cos(rel_heading)
    obs[cfg.LIDAR_NUM + 4] = np.sin(rel_heading)
    obs[cfg.LIDAR_NUM + 5] = np.cos(articulation)
    obs[cfg.LIDAR_NUM + 6] = np.sin(articulation)
    return obs


def test_parse_goal_repr_wraps_angles_without_name_error():
    env = DummyEnv()
    lib = DummyPrimitiveLib(np.zeros((1, 1, 2), dtype=np.float64))
    wrapper = MacroActionWrapper(env, lib, H=1, normalize_before_step=False)

    obs = np.zeros((120 + 7 + 2,), dtype=np.float64)
    obs[120] = 0.1
    obs[121] = np.cos(4.0)
    obs[122] = np.sin(4.0)
    obs[123] = np.cos(-4.5)
    obs[124] = np.sin(-4.5)
    obs[125] = np.cos(3.8)
    obs[126] = np.sin(3.8)

    goal = wrapper._parse_goal_repr_from_obs(obs)

    assert np.isfinite(goal["goal_heading"])
    assert np.isfinite(goal["articulation"])
    assert -np.pi <= goal["goal_heading"] <= np.pi
    assert -np.pi <= goal["articulation"] <= np.pi


def test_takeover_waits_until_near_goal(monkeypatch):
    env = DummyEnv()
    lib = DummyPrimitiveLib(np.zeros((1, 1, 2), dtype=np.float64))
    wrapper = MacroActionWrapper(env, lib, H=1, normalize_before_step=False)

    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", True)
    monkeypatch.setattr(cfg, "TAKEOVER_NEAR_GOAL_ONLY", True)
    monkeypatch.setattr(cfg, "TAKEOVER_NEAR_GOAL_DIST", 4.0)
    monkeypatch.setattr(cfg, "TAKEOVER_DIST_BASE", 10.0)
    monkeypatch.setattr(cfg, "TAKEOVER_DIST_HYSTERESIS", 2.0)

    far_but_hard = _make_takeover_obs(
        dist_m=8.0,
        rel_heading=np.deg2rad(50.0),
        articulation=np.deg2rad(30.0),
        min_lidar_m=1.0,
    )
    assert wrapper._should_takeover(far_but_hard) is False

    near_goal = _make_takeover_obs(dist_m=3.5)
    assert wrapper._should_takeover(near_goal) is True

    wrapper._takeover_active = True
    assert wrapper._should_takeover(_make_takeover_obs(dist_m=5.5)) is True
    assert wrapper._should_takeover(_make_takeover_obs(dist_m=6.5)) is False


def test_ppo_agent_collects_imitation_samples_and_computes_loss():
    agent = _make_discrete_ppo(action_dim=5)

    obs = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
    mask = np.ones((5,), dtype=np.int8)

    agent.push_imitation_memory(obs, action=2, action_mask=mask)
    agent.push_imitation_memory(obs * 0.5, action=1, action_mask=mask)

    loss = agent._compute_imitation_loss()

    assert agent.imitation_buffer_size() == 2
    assert loss is not None
    assert float(loss.detach().cpu().item()) >= 0.0