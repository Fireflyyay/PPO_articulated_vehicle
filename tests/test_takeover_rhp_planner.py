import numpy as np

from primitives.primitive_index import PrimitiveGridIndex
from terminal_takeover_rhp import RecedingHorizonTakeoverPlanner


def _make_index():
    primitive_to_cells = [
        np.array([[3, 0]], dtype=np.int64),
        np.array([[5, 0]], dtype=np.int64),
        np.array([[5, 1]], dtype=np.int64),
        np.array([[5, 0]], dtype=np.int64),
    ]
    cell_to_primitives = {
        (3, 0): np.array([0], dtype=np.int64),
        (5, 0): np.array([1, 3], dtype=np.int64),
        (5, 1): np.array([2], dtype=np.int64),
    }
    primitive_to_group_id = np.array([0, 0, 1, 1], dtype=np.int64)
    group_to_primitive_ids = [
        np.array([0, 1], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
    ]
    group_prefix_steps = np.array([4, 2], dtype=np.int64)

    return PrimitiveGridIndex(
        grid_resolution=1.0,
        x_min=-1.0,
        y_min=-1.0,
        x_max=8.0,
        y_max=8.0,
        primitive_to_cells=primitive_to_cells,
        cell_to_primitives=cell_to_primitives,
        primitive_to_group_id=primitive_to_group_id,
        group_to_primitive_ids=group_to_primitive_ids,
        group_prefix_steps=group_prefix_steps,
    )


def _make_planner(max_prefix_steps=None, score_weights=None):
    primitive_actions = np.array(
        [
            [[0.0, -1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    primitive_deltas = np.array(
        [
            [-2.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
            [1.2, 0.8, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return RecedingHorizonTakeoverPlanner(
        primitive_actions=primitive_actions,
        primitive_deltas=primitive_deltas,
        grid_index=_make_index(),
        lidar_num=8,
        lidar_range=8.0,
        score_weights=score_weights or {
            "dist": 1.0,
            "dir": 0.1,
            "state": 0.0,
            "smooth": 0.0,
            "speed": 0.0,
            "clearance": 0.5,
        },
        occupancy_inflation_radius=0.0,
        group_score_topk=2,
        max_prefix_steps=max_prefix_steps,
        min_clearance_hit_penalty=1.0,
    )


def test_plan_returns_no_candidate_when_all_paths_blocked():
    planner = _make_planner()
    lidar = np.ones((8,), dtype=np.float64)
    planner._build_occupied_cells_from_lidar = lambda _lidar: (
        {(3, 0), (5, 0), (5, 1)},
        {"occupied_cells": 3, "lidar_hit_beams": 3, "occupancy_ms": 0.0},
    )

    result = planner.plan(
        state=None,
        obs=None,
        lidar=lidar,
        goal_repr={"goal_x": 2.0, "goal_y": 0.0, "goal_heading": 0.0, "articulation": 0.0},
    )

    assert result.primitive_ids == []
    assert result.prefix_steps is None
    assert result.debug["reason"] == "no_candidate"


def test_plan_respects_direction_mode_and_caps_prefix_steps():
    planner = _make_planner(max_prefix_steps=1)
    lidar = np.ones((8,), dtype=np.float64)

    result = planner.plan(
        state=None,
        obs=None,
        lidar=lidar,
        goal_repr={"goal_x": -1.0, "goal_y": 0.0, "goal_heading": 0.0, "articulation": 0.0},
        mode="reverse",
    )

    assert result.primitive_ids == [0]
    assert result.prefix_steps == 1
    assert result.debug["chosen_group"] == 0
    assert result.debug["chosen_pid"] == 0


def test_plan_prefers_clearer_group_when_progress_is_similar():
    planner = _make_planner()
    lidar = np.ones((8,), dtype=np.float64)
    planner._build_occupied_cells_from_lidar = lambda _lidar: (
        set(),
        {"occupied_cells": 0, "lidar_hit_beams": 0, "occupancy_ms": 0.0},
    )
    planner.index.fast_prune_primitives = lambda occupied_cells: np.array([True, True, True, True], dtype=np.bool_)
    planner.index.count_near_hits = lambda occupied_cells: np.array([0, 4, 0, 0], dtype=np.int32)

    result = planner.plan(
        state=None,
        obs=None,
        lidar=lidar,
        goal_repr={"goal_x": 2.0, "goal_y": 0.0, "goal_heading": 0.0, "articulation": 0.0},
        mode="forward",
    )

    assert result.primitive_ids == [3]
    assert result.prefix_steps == 2
    assert result.debug["candidates"] == 4
    assert result.debug["candidates_dir"] == 3
    assert result.debug["chosen_group"] == 1
