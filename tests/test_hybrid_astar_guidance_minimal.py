from types import SimpleNamespace

import numpy as np

from env.scene_generators.block_mixing_plant_generator import _serialize_warmup_reference_states
from env.vehicle import State
from guidance.hybrid_astar_guidance import HybridAStarGuidance


class DummyPrimitiveLib:
    def __init__(self):
        self.action_dim = 3
        self.family_to_motion_family = np.asarray([0, 0, 1], dtype=np.int64)
        self.gamma_bin_values = np.asarray([-0.3, 0.0, 0.3], dtype=np.float64)
        self.variant_flat_to_family = np.asarray([0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        self.speed_signs = np.asarray([1, 1, -1, 1, 1, -1, 1], dtype=np.int64)

    def get_actions(self, flat_index: int):
        return np.asarray([[0.0, 1.0]], dtype=np.float64)

    def get_rollout_states(self, flat_index: int):
        return np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0 + float(flat_index), 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    def resolve_family_variant(self, family_id: int, gamma: float, primitive_mode: str, goal_repr, selection_context=None):
        return SimpleNamespace(flat_index=int(family_id))


class DummyMap:
    def __init__(self):
        self.dest = State([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.xmin = -10.0
        self.xmax = 10.0
        self.ymin = -10.0
        self.ymax = 10.0
        self.obstacles = []
        self.scene_regions = {
            "warmup_reference_states": [
                {"x": 0.0, "y": 0.0, "theta_front": 0.0, "theta_rear": 0.0, "gamma": 0.0, "direction": 1, "time_index": 0, "source_family_id": -1, "source_primitive_id": -1, "cumulative_progress": 0.0},
                {"x": 1.0, "y": 0.0, "theta_front": 0.0, "theta_rear": 0.0, "gamma": 0.0, "direction": 1, "time_index": 1, "source_family_id": 0, "source_primitive_id": 4, "cumulative_progress": 1.0},
                {"x": 2.0, "y": 0.0, "theta_front": 0.0, "theta_rear": 0.0, "gamma": 0.0, "direction": 1, "time_index": 2, "source_family_id": 0, "source_primitive_id": 5, "cumulative_progress": 2.0},
                {"x": 3.0, "y": 0.0, "theta_front": 0.0, "theta_rear": 0.0, "gamma": 0.0, "direction": 1, "time_index": 3, "source_family_id": 1, "source_primitive_id": 6, "cumulative_progress": 3.0},
            ],
            "attempt_seed": 123,
            "generation_attempt_index": 0,
            "seed": 42,
        }
        self.guidance_path_points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)


class DummyWrapper:
    def __init__(self):
        self.action_space = SimpleNamespace(n=3)
        self.primitive_lib = DummyPrimitiveLib()
        self.env = SimpleNamespace(map=DummyMap())
        self._current_primitive_mode = "normal"

    def _current_vehicle_state(self):
        return State([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _canonical_state_to_world(self, start_state, row):
        dx = float(row[0])
        return State([
            float(start_state.loc.x) + dx,
            float(start_state.loc.y),
            float(start_state.heading),
            0.0,
            0.0,
            float(start_state.rear_heading),
        ])


def _make_cfg():
    return SimpleNamespace(
        ENABLE_HYBRID_ASTAR_GUIDANCE=True,
        HYBRID_ASTAR_GUIDANCE_CURRICULUM_ENABLE={"Warmup": True},
        HYBRID_ASTAR_GUIDANCE_LAMBDA_BY_CURRICULUM={"Warmup": 1.0},
        HYBRID_ASTAR_GUIDANCE_DROPOUT_BY_CURRICULUM={"Warmup": 0.0},
        HYBRID_ASTAR_GUIDANCE_REPLAN_EVERY_BY_CURRICULUM={"Warmup": 1},
        HYBRID_ASTAR_GUIDANCE_LOCAL_SUBGOAL_ENABLE=True,
        HYBRID_ASTAR_GUIDANCE_LOCAL_SUBGOAL_HORIZON_BY_CURRICULUM={"Warmup": 2},
        HYBRID_ASTAR_GUIDANCE_REFERENCE_ENABLE=True,
        HYBRID_ASTAR_GUIDANCE_MAX_EXPAND_NODES=4,
        HYBRID_ASTAR_GUIDANCE_MAX_PLAN_TIME_MS=1.0,
        HYBRID_ASTAR_GUIDANCE_MAX_SEARCH_DEPTH=2,
        HYBRID_ASTAR_GUIDANCE_HEURISTIC_WEIGHT=1.0,
        HYBRID_ASTAR_GUIDANCE_GREEDY_FALLBACK_ENABLE=True,
        HYBRID_ASTAR_GUIDANCE_POSITION_RESOLUTION_M=0.5,
        HYBRID_ASTAR_GUIDANCE_HEADING_RESOLUTION_DEG=10.0,
        HYBRID_ASTAR_GUIDANCE_GAMMA_RESOLUTION_DEG=5.0,
        HYBRID_ASTAR_GUIDANCE_COLLISION_MAX_TRANSLATION_M=0.5,
        HYBRID_ASTAR_GUIDANCE_COLLISION_MAX_HEADING_DEG=10.0,
        HYBRID_ASTAR_GUIDANCE_COLLISION_MAX_GAMMA_DEG=5.0,
        HYBRID_ASTAR_GUIDANCE_COLLISION_MARGIN_M=0.0,
        HYBRID_ASTAR_GUIDANCE_ARTICULATION_LIMIT_MARGIN_DEG=0.0,
        HYBRID_ASTAR_GUIDANCE_REAR_NEAR_COLLISION_MARGIN_M=0.25,
        HYBRID_ASTAR_GUIDANCE_REVERSE_PENALTY=0.1,
        HYBRID_ASTAR_GUIDANCE_DIRECTION_SWITCH_PENALTY=0.1,
        HYBRID_ASTAR_GUIDANCE_MASK_ZERO_THRESHOLD=0.0,
        HYBRID_ASTAR_GUIDANCE_LOCAL_GOAL_POS_TOL_M=1.0,
        HYBRID_ASTAR_GUIDANCE_LOCAL_GOAL_HEADING_TOL_DEG=25.0,
        HYBRID_ASTAR_GUIDANCE_FINAL_FRONT_OVERLAP_THR=0.7,
        HYBRID_ASTAR_GUIDANCE_FINAL_HEADING_TOL_DEG=15.0,
        HYBRID_ASTAR_GUIDANCE_TEACHER_LOGIT=2.0,
        HYBRID_ASTAR_GUIDANCE_NEIGHBOR_LOGIT=0.5,
        HYBRID_ASTAR_GUIDANCE_CACHE_SIZE=8,
        HYBRID_ASTAR_GUIDANCE_FAIL_DISABLE_STREAK=2,
        SEED=42,
    )


def test_serialize_warmup_reference_states_exports_articulated_fields():
    primitive_lib = DummyPrimitiveLib()
    states = [
        State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        State([1.0, 0.0, 0.1, 0.0, 0.0, 0.05]),
        State([2.0, 0.2, 0.2, 0.0, 0.0, 0.08]),
    ]

    records = _serialize_warmup_reference_states(states, [4, 5], primitive_lib)

    assert len(records) == 3
    assert set(records[0].keys()) >= {"x", "y", "theta_front", "theta_rear", "gamma", "direction", "time_index", "source_family_id", "source_primitive_id", "cumulative_progress"}
    assert records[1]["source_primitive_id"] == 4
    assert records[2]["cumulative_progress"] > records[1]["cumulative_progress"]


def test_hybrid_guidance_fail_opens_when_wrapper_context_missing():
    guidance = HybridAStarGuidance(_make_cfg())

    result = guidance.compute_guidance(None, obs=None, action_mask=None, scene_level="Warmup", episode_step=0)

    assert result.planner_success is False
    assert result.guidance_valid is False
    assert result.guidance_weight == 0.0
    assert result.guidance_logits.size == 0
    assert result.fail_reason == "missing_wrapper_context"


def test_hybrid_guidance_prefers_reference_subgoal_when_available():
    guidance = HybridAStarGuidance(_make_cfg())
    wrapper = DummyWrapper()
    state = wrapper._current_vehicle_state()

    subgoal = guidance._select_subgoal(wrapper, state, scene_level="Warmup")

    assert subgoal["reference_state_available"] is True
    assert subgoal["subgoal_index"] == 2
    assert subgoal["average_progress_along_reference"] >= 0.0


def test_hybrid_guidance_uses_greedy_fallback_after_search_budget_exhausts(monkeypatch):
    cfg = _make_cfg()
    cfg.HYBRID_ASTAR_GUIDANCE_MAX_EXPAND_NODES = 1
    guidance = HybridAStarGuidance(cfg)
    wrapper = DummyWrapper()

    monkeypatch.setattr(guidance, "_goal_reached", lambda *_args, **_kwargs: False)

    def fake_validate(_wrapper, start_state, flat_index):
        x = float(start_state.loc.x) + float(flat_index) + 1.0
        state1 = State([x, 0.0, 0.0, 0.0, 0.0, 0.0])
        return True, [start_state, state1], "", 0.3, 0

    monkeypatch.setattr(guidance, "_validate_rollout", fake_validate)

    result = guidance.compute_guidance(wrapper, obs=None, action_mask=np.ones((3,), dtype=np.float32), scene_level="Warmup", episode_step=0)

    assert result.planner_success is True
    assert result.guidance_valid is True
    assert result.teacher_family_id == 1
    assert result.family_path == [1]