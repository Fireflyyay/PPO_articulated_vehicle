import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.vehicle import Status
from env.wrappers.macro_action_wrapper import MacroActionWrapper


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
    def __init__(self, terminate_after=None):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self._t = 0
        self.terminate_after = terminate_after

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.array([0.0]), {}

    def step(self, action):
        self._t += 1
        terminated = False
        if self.terminate_after is not None and self._t >= int(self.terminate_after):
            terminated = True
        obs = np.array([float(self._t)])
        reward = 1.0
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def test_prefix_steps_executes_correct_low_level_steps():
    env = DummyEnv()
    actions = np.zeros((1, 10, 2), dtype=np.float64)
    lib = DummyPrimitiveLib(actions)

    w = MacroActionWrapper(env, lib, H=10, normalize_before_step=False)

    # Pretend planner requested only 3 steps for the next primitive
    w._prefix_steps_queue = [3]

    obs, total_reward, terminated, truncated, info = w.step(0)

    assert info["executed_steps"] == 3
    assert info["prefix_steps_used"] == 3
    assert total_reward == 3.0
    assert (not terminated) and (not truncated)


def test_prefix_steps_respects_env_done():
    env = DummyEnv(terminate_after=2)
    actions = np.zeros((1, 10, 2), dtype=np.float64)
    lib = DummyPrimitiveLib(actions)

    w = MacroActionWrapper(env, lib, H=10, normalize_before_step=False)
    w._prefix_steps_queue = [5]

    obs, total_reward, terminated, truncated, info = w.step(0)

    assert info["executed_steps"] == 2
    assert info["prefix_steps_used"] == 5  # requested prefix
    assert total_reward == 2.0
    assert terminated


def test_soft_ray_auto_prefix_executes_safe_prefix_without_planner_queue():
    env = DummyEnv()
    actions = np.zeros((1, 10, 2), dtype=np.float64)
    lib = DummyPrimitiveLib(actions)

    w = MacroActionWrapper(env, lib, H=10, normalize_before_step=False)
    w._action_mask_mode = "soft_ray"
    w._soft_prefix_family_steps = {0: 3}
    w._last_action_mask_debug = {
        "min_lidar_m": 10.0,
        "obs_density": 0.0,
        "effective_action_count": 10,
    }

    obs, total_reward, terminated, truncated, info = w.step(0)

    assert info["executed_steps"] == 3
    assert info["prefix_steps_used"] == 3
    assert info["prefix_steps_source"] == "soft_ray_auto"
    assert info["soft_safe_prefix_steps"] == 3
    assert total_reward == 3.0
    assert (not terminated) and (not truncated)


def test_soft_ray_dynamic_prefix_shortens_execution_in_narrow_scene():
    env = DummyEnv()
    actions = np.zeros((1, 10, 2), dtype=np.float64)
    lib = DummyPrimitiveLib(actions)

    w = MacroActionWrapper(env, lib, H=10, normalize_before_step=False)
    w._action_mask_mode = "soft_ray"
    w._soft_prefix_family_steps = {0: 10}
    w._last_action_mask_debug = {
        "min_lidar_m": 1.0,
        "obs_density": 0.45,
        "effective_action_count": 2,
    }

    obs, total_reward, terminated, truncated, info = w.step(0)

    assert info["executed_steps"] == 1
    assert info["prefix_steps_used"] == 1
    assert info["prefix_steps_source"] == "soft_ray_auto"
    assert info["soft_safe_prefix_steps"] == 10
    assert total_reward == 1.0
    assert (not terminated) and (not truncated)


def test_soft_ray_zero_safe_prefix_truncates_episode_instead_of_spinning():
    env = DummyEnv()
    actions = np.zeros((1, 10, 2), dtype=np.float64)
    lib = DummyPrimitiveLib(actions)

    w = MacroActionWrapper(env, lib, H=10, normalize_before_step=False)
    w._action_mask_mode = "soft_ray"
    w._soft_prefix_family_steps = {0: 0}
    w._last_action_mask_debug = {
        "min_lidar_m": 0.5,
        "obs_density": 0.8,
        "effective_action_count": 1,
    }

    obs, total_reward, terminated, truncated, info = w.step(0)

    assert info["executed_steps"] == 0
    assert info["prefix_steps_used"] == 0
    assert info["prefix_steps_source"] == "soft_ray_auto_blocked"
    assert info["soft_ray_blocked"] is True
    assert info["status"] == Status.OUTTIME
    assert total_reward == 0.0
    assert (not terminated) and truncated
