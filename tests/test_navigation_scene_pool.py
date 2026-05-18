import time

import pytest

from configs import (
    BLOCK_MIXING_PLANT_CONFIG,
    NAVIGATION_MIN_ENDPOINT_CLEARANCE_BY_LEVEL,
    NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL,
    NAVIGATION_PATH_RATIO_LIMIT_BY_LEVEL,
)
from env.parking_map_normal import ParkingMapNormal


@pytest.mark.parametrize("level", ["Warmup", "Normal", "Complex", "Extrem"])
def test_navigation_scene_pool_preserves_scene_filter_constraints(level):
    parking_map = ParkingMapNormal(level, enable_scene_pool=True, scene_pool_size=3)
    cfg = BLOCK_MIXING_PLANT_CONFIG[level]

    for _ in range(4):
        parking_map.reset()
        metrics = parking_map.scene_regions["scene_metrics"]
        final_metrics = parking_map.scene_metrics

        assert metrics is not None
        assert final_metrics is not None
        assert float(metrics["path_ratio"]) <= float(cfg["scene_metric_max_path_ratio"]) + 1e-6
        assert float(metrics["path_min_clearance"]) >= float(cfg["scene_metric_min_path_clearance"]) - 1e-6
        assert float(metrics["min_endpoint_clearance"]) >= float(cfg["scene_metric_min_endpoint_clearance"]) - 1e-6
        assert float(final_metrics["path_ratio"]) > 0.0
        assert float(final_metrics["path_min_clearance"]) > 0.0
        assert float(final_metrics["min_endpoint_clearance"]) > 0.0
        assert parking_map.guidance_path_points is not None
        assert len(parking_map.guidance_path_points) >= 2
        assert int(parking_map.scene_regions["divider_wall_count"]) > 0

    stats = parking_map.get_scene_pool_stats()
    assert stats["enabled"] is True
    assert stats["pool_hits"] >= 3
    assert stats["pool_misses"] >= 1
    assert stats["prefill_generated"] == 3
    assert stats["generated_scenes"] >= stats["consumed_scenes"]


def test_navigation_scene_pool_hit_avoids_sync_generation(monkeypatch):
    call_count = {"count": 0}

    def fake_generate_navigation_case(map_level, return_regions=False, _retry_idx=0):
        call_count["count"] += 1
        time.sleep(0.02)
        start = [0.0, 0.0, 0.0]
        dest = [8.0, 0.0, 0.0]
        obstacles = []
        scene_meta = {
            "divider_scene_metrics": {
                "path_ratio": 1.0,
                "path_min_clearance": 10.0,
                "min_endpoint_clearance": 10.0,
            },
            "guidance_path_points": [[0.0, 0.0], [8.0, 0.0]],
            "divider_wall_count": 2,
        }
        if return_regions:
            return start, dest, obstacles, scene_meta
        return start, dest, obstacles

    monkeypatch.setattr(
        "env.parking_map_normal.generate_navigation_case",
        fake_generate_navigation_case,
    )

    direct_map = ParkingMapNormal("Normal", enable_scene_pool=False)
    t0 = time.perf_counter()
    direct_map.reset()
    direct_reset_s = time.perf_counter() - t0

    pooled_map = ParkingMapNormal("Normal", enable_scene_pool=True, scene_pool_size=3)
    t1 = time.perf_counter()
    pooled_map.reset()
    first_pooled_reset_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    pooled_map.reset()
    warm_pooled_reset_s = time.perf_counter() - t2

    stats = pooled_map.get_scene_pool_stats()

    assert call_count["count"] == 4
    assert first_pooled_reset_s <= direct_reset_s
    assert warm_pooled_reset_s < direct_reset_s * 0.25
    assert stats["pool_misses"] == 0
    assert stats["pool_hits"] == 2
    assert stats["generated_scenes"] == 3
    assert stats["prefill_generated"] == 3
    assert stats["top_up_generated"] == 0
    assert stats["consumed_scenes"] == 2
    assert stats["current_size"] == 1