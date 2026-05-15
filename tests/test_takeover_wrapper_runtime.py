import numpy as np
import gymnasium as gym
from gymnasium import spaces
from types import SimpleNamespace

import configs as cfg
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
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        return np.array([0.0]), {}

    def step(self, action):
        return np.array([0.0]), 0.0, False, False, {}


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


def test_takeover_enable_is_reloaded_after_wrapper_init(monkeypatch):
    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", False)
    env = DummyEnv()
    lib = DummyPrimitiveLib(np.zeros((1, 1, 2), dtype=np.float64))
    wrapper = MacroActionWrapper(env, lib, H=1, normalize_before_step=False)

    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", True)
    monkeypatch.setattr(cfg, "TAKEOVER_NEAR_GOAL_ONLY", False)

    obs = _make_takeover_obs(dist_m=1.0)

    assert wrapper._takeover_enabled is False
    assert wrapper._should_takeover(obs) is True
    assert wrapper._takeover_enabled is True


def test_takeover_respects_runtime_config_when_patched_before_init(monkeypatch):
    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", True)
    monkeypatch.setattr(cfg, "TAKEOVER_NEAR_GOAL_ONLY", True)
    monkeypatch.setattr(cfg, "TAKEOVER_NEAR_GOAL_DIST", 4.0)
    monkeypatch.setattr(cfg, "TAKEOVER_DIST_BASE", 10.0)
    monkeypatch.setattr(cfg, "TAKEOVER_DIST_HYSTERESIS", 2.0)

    env = DummyEnv()
    lib = DummyPrimitiveLib(np.zeros((1, 1, 2), dtype=np.float64))
    wrapper = MacroActionWrapper(env, lib, H=1, normalize_before_step=False)

    assert wrapper._takeover_enabled is True
    assert wrapper._should_takeover(_make_takeover_obs(dist_m=3.5)) is True
    assert wrapper._should_takeover(_make_takeover_obs(dist_m=8.0, rel_heading=np.deg2rad(50.0), articulation=np.deg2rad(30.0), min_lidar_m=1.0)) is False


def test_soft_ray_without_ray_safety_falls_back_to_hybrid_mask():
    wrapper = object.__new__(MacroActionWrapper)
    wrapper._takeover_active = False
    wrapper._action_mask_mode = "soft_ray"
    wrapper._ray_safety_index = None
    wrapper._action_mask_update_every_k = 1
    wrapper._action_mask_cached = None
    wrapper._action_mask_calls_since_update = 0
    wrapper._last_action_mask_debug = {}
    wrapper.action_space = SimpleNamespace(n=3)

    called = {}

    def fake_hard(obs_vec=None, mode_override=None):
        called["mode_override"] = mode_override
        return np.array([1, 0, 1], dtype=np.int8)

    wrapper._compute_hard_action_mask = fake_hard

    def fail_soft(obs_vec=None):
        raise AssertionError("soft ray path should not run without ray safety")

    wrapper._compute_soft_ray_action_mask = fail_soft

    mask = MacroActionWrapper.get_action_mask(wrapper, np.array([0.0], dtype=np.float64))

    assert np.array_equal(mask, np.array([1, 0, 1], dtype=np.int8))
    assert called["mode_override"] == "hybrid"
    assert wrapper._last_action_mask_debug["fallback"] == "hybrid_no_ray_safety"
    assert wrapper._last_action_mask_debug["effective_mode"] == "hybrid"


def test_soft_ray_mask_uses_k_step_cache():
    wrapper = object.__new__(MacroActionWrapper)
    wrapper._takeover_active = False
    wrapper._action_mask_mode = "soft_ray"
    wrapper._ray_safety_index = object()
    wrapper._action_mask_update_every_k = 3
    wrapper._action_mask_cached = None
    wrapper._action_mask_calls_since_update = 0
    wrapper._last_action_mask_debug = {}
    wrapper.action_space = SimpleNamespace(n=3)

    calls = {"soft": 0}

    def fake_soft(obs_vec=None):
        calls["soft"] += 1
        return np.array([0.5, 0.2, 1.0], dtype=np.float32)

    wrapper._compute_soft_ray_action_mask = fake_soft

    obs = np.array([0.0], dtype=np.float64)
    mask1 = MacroActionWrapper.get_action_mask(wrapper, obs)
    mask2 = MacroActionWrapper.get_action_mask(wrapper, obs)
    mask3 = MacroActionWrapper.get_action_mask(wrapper, obs)
    mask4 = MacroActionWrapper.get_action_mask(wrapper, obs)

    assert np.array_equal(mask1, np.array([0.5, 0.2, 1.0], dtype=np.float32))
    assert np.array_equal(mask2, mask1)
    assert np.array_equal(mask3, mask1)
    assert np.array_equal(mask4, mask1)
    assert calls["soft"] == 2
