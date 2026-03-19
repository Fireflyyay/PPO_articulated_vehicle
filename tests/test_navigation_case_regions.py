import pytest
import numpy as np
from shapely.geometry import Point

from env.parking_map_normal import generate_navigation_case
from configs import (
    NAVIGATION_MIN_ENDPOINT_CLEARANCE_BY_LEVEL,
    NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL,
    NAVIGATION_PATH_RATIO_LIMIT_BY_LEVEL,
    NAVIGATION_TIGHT_TURN_HEADING_DEG_BY_LEVEL,
    NAVIGATION_TIGHT_TURN_MIN_ENDPOINT_CLEARANCE_BY_LEVEL,
)


def _in_plaza(pose, plaza) -> bool:
    pt = Point(float(pose[0]), float(pose[1]))
    # Pose is sampled with `poly.contains`, but add a tiny buffer for numeric robustness.
    return bool(plaza.buffer(1e-9).contains(pt))


def _support_edge_id(edge_meta):
    assert edge_meta is not None
    assert "edge_id" in edge_meta
    return int(edge_meta["edge_id"])


@pytest.mark.parametrize(
    "level, expected_in_plaza_count",
    [
        ("Normal", 2),   # easy: both in plaza
        ("Complex", 1),  # normal: exactly one in plaza
        ("Extrem", 0),   # hard: both in corridor
    ],
)
def test_navigation_case_start_goal_region_by_difficulty(level, expected_in_plaza_count):
    # Sample multiple times to guard against rare edge cases.
    for _ in range(20):
        start, dest, _obstacles, regions = generate_navigation_case(level, return_regions=True)
        plaza = regions["plaza"]
        corridors = regions["corridors"]

        assert plaza is not None and (not plaza.is_empty)
        # For Complex/Extrem, corridors must exist.
        if level in ["Complex", "Extrem"]:
            assert len(corridors) > 0

        in_plaza_count = int(_in_plaza(start, plaza)) + int(_in_plaza(dest, plaza))
        assert in_plaza_count == expected_in_plaza_count

        # If a pose is in corridor (i.e., not in plaza), it should not be right at the corridor mouth.
        # We require a small positive distance to plaza for robustness.
        for pose in [start, dest]:
            if not _in_plaza(pose, plaza):
                assert float(Point(float(pose[0]), float(pose[1])).distance(plaza)) > 1.0


@pytest.mark.parametrize("level", ["Normal", "Complex", "Extrem"])
def test_navigation_case_scene_metrics_respect_filter_gate(level):
    for _ in range(8):
        start, dest, _obstacles, regions = generate_navigation_case(level, return_regions=True)
        metrics = regions["scene_metrics"]

        assert metrics is not None
        assert float(metrics["path_ratio"]) <= float(NAVIGATION_PATH_RATIO_LIMIT_BY_LEVEL[level]) + 1e-6
        assert float(metrics["path_min_clearance"]) >= float(NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL[level]) - 1e-6
        assert float(metrics["min_endpoint_clearance"]) >= float(NAVIGATION_MIN_ENDPOINT_CLEARANCE_BY_LEVEL[level]) - 1e-6

        if float(metrics["heading_diff_deg"]) >= float(NAVIGATION_TIGHT_TURN_HEADING_DEG_BY_LEVEL[level]) - 1e-6:
            assert float(metrics["min_endpoint_clearance"]) >= float(
                NAVIGATION_TIGHT_TURN_MIN_ENDPOINT_CLEARANCE_BY_LEVEL[level]
            ) - 1e-6


@pytest.mark.parametrize("level", ["Normal", "Complex", "Extrem"])
def test_navigation_case_divider_walls_never_degenerate(level):
    for _ in range(10):
        _start, _dest, obstacles, regions = generate_navigation_case(level, return_regions=True)

        assert int(regions["start_divider_wall_count"]) > 0
        assert int(regions["dest_divider_wall_count"]) > 0
        assert int(regions["divider_wall_count"]) >= (
            int(regions["start_divider_wall_count"]) + int(regions["dest_divider_wall_count"])
        )
        assert len(obstacles) > 5


def test_navigation_case_normal_support_edges_not_always_same_wall():
    np.random.seed(0)

    total = 60
    same_edge = 0
    different_edge = 0

    for _ in range(total):
        start, dest, _obstacles, regions = generate_navigation_case("Normal", return_regions=True)
        plaza = regions["plaza"]
        start_edge = regions["start_support_edge"]
        dest_edge = regions["dest_support_edge"]

        assert _in_plaza(start, plaza)
        assert _in_plaza(dest, plaza)
        assert start_edge is not None
        assert dest_edge is not None
        assert start_edge["poly_label"] == "plaza"
        assert dest_edge["poly_label"] == "plaza"

        if _support_edge_id(start_edge) == _support_edge_id(dest_edge):
            same_edge += 1
        else:
            different_edge += 1

    assert different_edge >= 12
    assert same_edge < total
