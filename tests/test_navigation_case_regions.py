import pytest
from shapely.geometry import Point

from env.parking_map_normal import generate_navigation_case


def _in_plaza(pose, plaza) -> bool:
    pt = Point(float(pose[0]), float(pose[1]))
    # Pose is sampled with `poly.contains`, but add a tiny buffer for numeric robustness.
    return bool(plaza.buffer(1e-9).contains(pt))


@pytest.mark.parametrize(
    "level, expected_in_plaza_count",
    [
        ("Normal", 2),   # easy: both in plaza
        ("Complex", 1),  # normal: exactly one in plaza
        ("Extrem", 1),   # hard: exactly one in plaza and one in corridor
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
