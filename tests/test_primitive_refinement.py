from types import SimpleNamespace
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import configs as cfg
from env.vehicle import State, Vehicle
from env.wrappers.macro_action_wrapper import MacroActionWrapper
from primitives.primitive_refinement import PlanRefinementResult, PrimitiveTrajectoryRefiner


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


class DummyRefinementEnv(gym.Env):
    def __init__(self, start_state, goal_state):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([cfg.VALID_STEER[0], cfg.VALID_SPEED[0]], dtype=np.float32),
            high=np.array([cfg.VALID_STEER[1], cfg.VALID_SPEED[1]], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.vehicle = Vehicle(
            n_step=cfg.NUM_STEP,
            step_len=cfg.STEP_LENGTH,
            articulated=True,
            trailer_length=cfg.TRAILER_LENGTH,
            hitch_offset=cfg.HITCH_OFFSET,
        )
        self._start_state = start_state
        self._goal_state = goal_state
        self.map = SimpleNamespace(
            xmin=-20.0,
            xmax=20.0,
            ymin=-20.0,
            ymax=20.0,
            obstacles=[],
            dest=goal_state,
            dest_box=goal_state.create_box()[0],
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.vehicle.reset(self._start_state)
        return np.zeros((1,), dtype=np.float64), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        steer_min, steer_max = cfg.VALID_STEER
        speed_min, speed_max = cfg.VALID_SPEED
        scaled_action = np.zeros_like(action, dtype=np.float64)
        scaled_action[0] = 0.5 * (action[0] + 1.0) * (steer_max - steer_min) + steer_min
        scaled_action[1] = 0.5 * (action[1] + 1.0) * (speed_max - speed_min) + speed_min
        self.vehicle.step(scaled_action)
        return np.zeros((1,), dtype=np.float64), 0.0, False, False, {}


def test_refiner_reduces_terminal_cost_and_keeps_drive_mode(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", False)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_OBJECTIVE", "terminal_window_v1")

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([2.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)

    actions = np.array(
        [
            [cfg.VALID_STEER[1], 1.5],
            [cfg.VALID_STEER[1], 1.5],
            [cfg.VALID_STEER[1], 1.5],
            [cfg.VALID_STEER[1], 1.5],
        ],
        dtype=np.float64,
    )

    result = refiner.refine(env, actions)

    assert result.feasible is True
    assert result.cost_after < result.cost_before
    assert np.all(result.actions[:, 1] > 0.0)
    assert np.max(np.abs(result.actions[:, 0])) <= np.max(np.abs(actions[:, 0])) + 1e-9
    debug = result.to_debug_dict()
    assert "final_heading_error_deg_before" in debug
    assert "final_mean_overlap_after" in debug
    assert "terminal_window_cost_after" in debug
    assert "final_barrier_cost_after" in debug


def test_plan_refiner_returns_phase_actions(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", False)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_OBJECTIVE", "terminal_window_v1")

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)
    lib = DummyPrimitiveLib(
        [
            [[cfg.VALID_STEER[1], 1.0], [cfg.VALID_STEER[1], 1.0]],
            [[cfg.VALID_STEER[1], 1.0], [cfg.VALID_STEER[1], 1.0]],
        ]
    )

    result = refiner.refine_plan(env, lib, [0, 1])

    assert result.plan_length == 2
    assert result.total_steps == 4
    assert len(result.phase_actions) == 2
    assert result.cost_after <= result.cost_before


def test_wrapper_executes_prepared_plan_cache_without_takeover(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", False)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_OBJECTIVE", "terminal_window_v1")

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([0.35, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    lib = DummyPrimitiveLib(
        [
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.0, -1.0], [0.0, -1.0]],
        ]
    )
    wrapper = MacroActionWrapper(env, lib, H=4, normalize_before_step=True)

    custom_actions = [
        np.array([[0.1, 0.8], [0.1, 0.8]], dtype=np.float64),
        np.array([[-0.2, -0.6], [-0.2, -0.6]], dtype=np.float64),
    ]
    custom_result = PlanRefinementResult(
        primitive_ids=[0, 1],
        phase_actions=custom_actions,
        plan_length=2,
        total_steps=4,
        applied=True,
        feasible=True,
        cost_before=2.0,
        cost_after=1.0,
        iterations=1,
        cost_evaluations=4,
        terminal_scale=1.0,
    )

    monkeypatch.setattr(wrapper._primitive_refiner, "refine_plan", lambda env_, lib_, ids: custom_result)

    wrapper.reset()
    debug = wrapper.prepare_plan_execution([0, 1], prefix_steps=1)
    _, _, _, _, info0 = wrapper.step(0)
    _, _, _, _, info1 = wrapper.step(1)

    assert debug["plan_length"] == 2
    assert debug["cost_after"] == 1.0
    assert info0["executed_steps"] == 1
    assert info0["prefix_steps_used"] == 1
    assert np.allclose(info0["macro_exec_trace"]["sub_actions_phys"][0], custom_actions[0][0])
    assert np.allclose(info1["macro_exec_trace"]["sub_actions_phys"], custom_actions[1])


def test_terminal_window_cost_prefers_precise_terminal_pose_over_shorter_traj(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "TAKEOVER_ENABLE", False)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_OBJECTIVE", "terminal_window_v1")

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)
    terminal_scale = refiner._terminal_scale(start, env.map)

    bad_final = State([2.0, 0.0, math.radians(18.0), 0.0, 0.0, math.radians(12.0)])
    mid_precise = State([1.85, 0.0, math.radians(1.0), 0.0, 0.0, math.radians(0.5)])
    good_final = State([2.0, 0.0, math.radians(2.0), 0.0, 0.0, math.radians(1.0)])

    bad_rollout = {
        "states": [start, bad_final],
        "clearances": [2.0, 2.0],
        "goal_metrics": [refiner._state_goal_metrics(env, start), refiner._state_goal_metrics(env, bad_final)],
        "feasible": True,
        "collision": False,
    }
    good_rollout = {
        "states": [start, mid_precise, good_final],
        "clearances": [2.0, 2.0, 2.0],
        "goal_metrics": [
            refiner._state_goal_metrics(env, start),
            refiner._state_goal_metrics(env, mid_precise),
            refiner._state_goal_metrics(env, good_final),
        ],
        "feasible": True,
        "collision": False,
    }

    bad_actions = np.array([[0.0, 0.25]], dtype=np.float64)
    good_actions = np.array([[0.0, 0.25], [0.0, 0.25]], dtype=np.float64)

    bad_cost = refiner._trajectory_cost(env, bad_rollout, bad_actions, mode_sign=1.0, terminal_scale=terminal_scale)
    good_cost = refiner._trajectory_cost(env, good_rollout, good_actions, mode_sign=1.0, terminal_scale=terminal_scale)

    assert refiner._state_goal_metrics(env, good_final)["mean_overlap"] > refiner._state_goal_metrics(env, bad_final)["mean_overlap"]
    assert good_cost < bad_cost


def test_state_goal_metrics_uses_goal_relative_articulation(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, math.radians(10.0)])
    goal = State([1.0, 0.0, math.radians(20.0), 0.0, 0.0, math.radians(5.0)])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)

    final_same_articulation = State([1.0, 0.0, math.radians(35.0), 0.0, 0.0, math.radians(20.0)])
    metrics = refiner._state_goal_metrics(env, final_same_articulation)

    assert abs(metrics["articulation_error"]) < 1e-9


def test_terminal_polish_only_updates_tail_steps(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_MAX_PASSES", 1)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_STEER_DELTA", 0.0)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_SPEED_DELTA", 0.0)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_ALLOW_PREFIX_SHRINK", False)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_ENABLE", True)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_TAIL_STEPS", 2)

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)

    actions = np.array(
        [
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
            [0.2, 0.8],
        ],
        dtype=np.float64,
    )

    def fake_polish(base_env, start_state, reference_actions, reference_rollout, reference_cost, step_mode_signs, mode_sign, terminal_scale, tail_start_idx):
        polished = np.asarray(reference_actions, dtype=np.float64).copy()
        polished[-2:, 0] = 0.0
        return polished, reference_rollout, reference_cost - 1.0, True, 8, 2, 1

    monkeypatch.setattr(refiner, "_terminal_polish_actions", fake_polish)

    result = refiner.refine(env, actions)

    assert np.allclose(result.actions[:2], actions[:2])
    assert np.allclose(result.actions[-2:, 0], 0.0)
    assert result.terminal_polish_triggered is True
    assert result.terminal_polish_applied is True
    assert result.terminal_polish_tail_steps == 2


def test_terminal_polish_respects_near_goal_trigger(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_MAX_PASSES", 1)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_STEER_DELTA", 0.0)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_SPEED_DELTA", 0.0)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_ENABLE", True)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_TERMINAL_POLISH_TRIGGER_DIST", 1.0)

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)
    actions = np.array([[0.0, 0.6], [0.0, 0.6]], dtype=np.float64)
    called = {"count": 0}

    def fake_polish(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("terminal polish should not run when start is outside trigger distance")

    monkeypatch.setattr(refiner, "_terminal_polish_actions", fake_polish)

    result = refiner.refine(env, actions)

    assert called["count"] == 0
    assert result.terminal_polish_triggered is False


def test_plan_refinement_limits_terminal_polish_to_last_phase(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_MAX_PASSES", 1)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_STEER_DELTA", 0.0)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_SPEED_DELTA", 0.0)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_TAIL_STEPS", 4)

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)
    class RaggedPrimitiveLib:
        def __init__(self, actions):
            self._actions = [np.asarray(item, dtype=np.float64) for item in actions]

        def get_actions(self, primitive_id):
            return self._actions[int(primitive_id)]

    lib = RaggedPrimitiveLib(
        [
            [[0.0, 0.6], [0.0, 0.6], [0.0, 0.6]],
            [[0.0, 0.6], [0.0, 0.6]],
        ]
    )
    captured = {"tail_start_idx": None}

    def fake_polish(base_env, start_state, reference_actions, reference_rollout, reference_cost, step_mode_signs, mode_sign, terminal_scale, tail_start_idx):
        captured["tail_start_idx"] = int(tail_start_idx)
        return reference_actions, reference_rollout, reference_cost, False, 0, 0, 0

    monkeypatch.setattr(refiner, "_terminal_polish_actions", fake_polish)

    result = refiner.refine_plan(env, lib, [0, 1])

    assert result.plan_length == 2
    assert captured["tail_start_idx"] == 3


def test_step_mode_signs_follow_original_drive_direction():
    refiner = PrimitiveTrajectoryRefiner(cfg)
    actions = np.array(
        [
            [0.0, 0.5],
            [0.0, 0.0],
            [0.0, -0.4],
            [0.0, -0.3],
        ],
        dtype=np.float64,
    )

    signs = refiner._infer_step_mode_signs(actions, fallback_sign=1.0)

    assert np.allclose(signs, np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64))


def test_terminal_cost_prioritizes_heading_and_front_overlap(monkeypatch):
    monkeypatch.setattr(cfg, "USE_PRIMITIVE_REFINEMENT", True)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_OBJECTIVE", "terminal_window_v1")

    start = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = State([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyRefinementEnv(start, goal)
    refiner = PrimitiveTrajectoryRefiner(cfg)
    terminal_scale = refiner._terminal_scale(start, env.map)

    bad_rollout = {
        "states": [start, goal],
        "clearances": [2.0, 2.0],
        "goal_metrics": [
            refiner._state_goal_metrics(env, start),
            {
                "position_error": 0.04,
                "heading_error": math.radians(7.0),
                "articulation_error": 0.0,
                "front_overlap": 0.66,
                "rear_overlap": 0.98,
                "mean_overlap": 0.82,
            },
        ],
        "feasible": True,
        "collision": False,
    }
    good_rollout = {
        "states": [start, goal],
        "clearances": [2.0, 2.0],
        "goal_metrics": [
            refiner._state_goal_metrics(env, start),
            {
                "position_error": 0.06,
                "heading_error": math.radians(1.0),
                "articulation_error": 0.0,
                "front_overlap": 0.90,
                "rear_overlap": 0.74,
                "mean_overlap": 0.82,
            },
        ],
        "feasible": True,
        "collision": False,
    }

    actions = np.array([[0.0, 0.25]], dtype=np.float64)

    bad_cost = refiner._trajectory_cost(env, bad_rollout, actions, mode_sign=1.0, terminal_scale=terminal_scale)
    good_cost = refiner._trajectory_cost(env, good_rollout, actions, mode_sign=1.0, terminal_scale=terminal_scale)

    assert good_cost < bad_cost


def test_front_body_first_polish_uses_lexicographic_priority(monkeypatch):
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_HEADING_EPS", math.radians(0.25))
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_OVERLAP_EPS", 0.01)
    monkeypatch.setattr(cfg, "PRIMITIVE_REFINEMENT_FRONT_BODY_FIRST_TERMINAL_POLISH_POSITION_EPS", 0.02)

    refiner = PrimitiveTrajectoryRefiner(cfg)
    incumbent = {
        "valid": 1.0,
        "front_heading_error": math.radians(2.0),
        "front_overlap_deficit": 0.10,
        "front_position_error": 0.05,
        "rear_soft_cost": 0.2,
        "scalar_fallback": 15.0,
    }
    better_front = {
        "valid": 1.0,
        "front_heading_error": math.radians(1.0),
        "front_overlap_deficit": 0.14,
        "front_position_error": 0.08,
        "rear_soft_cost": 3.0,
        "scalar_fallback": 30.0,
    }
    worse_front = {
        "valid": 1.0,
        "front_heading_error": math.radians(2.8),
        "front_overlap_deficit": 0.02,
        "front_position_error": 0.01,
        "rear_soft_cost": 0.0,
        "scalar_fallback": 4.0,
    }

    assert refiner._is_terminal_polish_candidate_better(better_front, incumbent) is True
    assert refiner._is_terminal_polish_candidate_better(worse_front, incumbent) is False