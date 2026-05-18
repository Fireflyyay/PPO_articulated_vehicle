from __future__ import annotations

import math

import numpy as np
import pytest

from configs import BLOCK_MIXING_PLANT_CONFIG
from env.parking_map_normal import generate_navigation_case


def _world_to_cell(scene_meta: dict, pose_xyz) -> tuple[int, int]:
    origin_x, origin_y = scene_meta["grid_origin"]
    block_size = float(scene_meta["block_size"])
    gx = int(math.floor((float(pose_xyz[0]) - float(origin_x)) / block_size))
    gy = int(math.floor((float(pose_xyz[1]) - float(origin_y)) / block_size))
    return gx, gy


@pytest.mark.parametrize("level", ["Warmup", "Normal", "Complex", "Extrem"])
def test_navigation_case_block_scene_metadata_matches_config(level):
    start, dest, obstacles, scene_meta = generate_navigation_case(level, return_regions=True)

    cfg = BLOCK_MIXING_PLANT_CONFIG[level]
    occupancy = scene_meta["occupancy_grid"]
    free_grid = scene_meta["free_grid"]

    assert scene_meta["scene_type"] == "block_mixing_plant"
    assert tuple(occupancy.shape) == (int(cfg["grid_height"]), int(cfg["grid_width"]))
    assert tuple(free_grid.shape) == tuple(occupancy.shape)
    assert np.array_equal(free_grid, occupancy == 0)
    assert len(obstacles) > 0
    assert int(scene_meta["parking_bay_count"]) >= int(cfg["parking_bay_count_range"][0])
    assert float(scene_meta["free_ratio"]) >= float(cfg["min_free_ratio"]) - 1e-6
    assert float(scene_meta["free_ratio"]) <= float(cfg["max_free_ratio"]) + 1e-6
    assert scene_meta["guidance_occupancy_payload"] is not None
    assert scene_meta["guidance_path_points"] is not None
    assert len(scene_meta["guidance_path_points"]) >= 2

    sx, sy = _world_to_cell(scene_meta, start)
    dx, dy = _world_to_cell(scene_meta, dest)
    assert occupancy[sy, sx] == 0
    assert occupancy[dy, dx] == 0


@pytest.mark.parametrize("level", ["Warmup", "Normal", "Complex", "Extrem"])
def test_navigation_case_scene_metrics_respect_block_filter_gate(level):
    _start, _dest, _obstacles, scene_meta = generate_navigation_case(level, return_regions=True)
    metrics = scene_meta["scene_metrics"]
    cfg = BLOCK_MIXING_PLANT_CONFIG[level]

    assert metrics is not None
    assert float(metrics["path_ratio"]) <= float(cfg["scene_metric_max_path_ratio"]) + 1e-6
    assert float(metrics["path_min_clearance"]) >= float(cfg["scene_metric_min_path_clearance"]) - 1e-6
    assert float(metrics["min_endpoint_clearance"]) >= float(cfg["scene_metric_min_endpoint_clearance"]) - 1e-6


@pytest.mark.parametrize("level", ["Warmup", "Normal", "Complex", "Extrem"])
def test_navigation_case_parking_bays_have_access_to_free_corridor(level):
    _start, _dest, _obstacles, scene_meta = generate_navigation_case(level, return_regions=True)
    occupancy = scene_meta["occupancy_grid"]
    parking_bays = scene_meta["parking_bays"]

    assert len(parking_bays) == int(scene_meta["parking_bay_count"])
    for bay in parking_bays:
        assert bay.access_cells is not None
        assert len(bay.access_cells) > 0
        for gx, gy in bay.grid_cells[: min(4, len(bay.grid_cells))]:
            assert occupancy[gy, gx] == 0
        for ax, ay in bay.access_cells[: min(3, len(bay.access_cells))]:
            assert occupancy[ay, ax] == 0


@pytest.mark.parametrize("level", ["Warmup", "Normal", "Complex", "Extrem"])
def test_navigation_case_parking_bays_keep_far_side_wall_intact(level):
    _start, _dest, _obstacles, scene_meta = generate_navigation_case(level, return_regions=True)
    occupancy = scene_meta["occupancy_grid"]
    parking_bays = scene_meta["parking_bays"]

    for bay in parking_bays:
        bay_cells = np.asarray(bay.grid_cells, dtype=np.int64)
        gx0 = int(np.min(bay_cells[:, 0]))
        gx1 = int(np.max(bay_cells[:, 0]))
        gy0 = int(np.min(bay_cells[:, 1]))
        gy1 = int(np.max(bay_cells[:, 1]))

        access_cells = np.asarray(bay.access_cells, dtype=np.int64)
        access_x = int(access_cells[len(access_cells) // 2][0])
        access_y = int(access_cells[len(access_cells) // 2][1])
        bay_center_cell_x = 0.5 * float(gx0 + gx1)
        bay_center_cell_y = 0.5 * float(gy0 + gy1)

        if np.all(access_cells[:, 1] == access_cells[0, 1]):
            far_y = gy0 - 1 if bay_center_cell_y < float(access_y) else gy1 + 1
            assert 0 <= far_y < occupancy.shape[0]
            assert np.all(occupancy[far_y, gx0:gx1 + 1] == 1)
        else:
            far_x = gx0 - 1 if bay_center_cell_x < float(access_x) else gx1 + 1
            assert 0 <= far_x < occupancy.shape[1]
            assert np.all(occupancy[gy0:gy1 + 1, far_x] == 1)
