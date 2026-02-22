import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
