from __future__ import annotations

import numpy as np

from env.parking_map_normal import generate_navigation_case


def test_warmup_scene_uses_motion_primitive_carving_and_single_parking_bay():
    start, dest, _obstacles, scene_meta = generate_navigation_case("Warmup", return_regions=True)

    assert scene_meta["scene_type"] == "block_mixing_plant"
    assert scene_meta["difficulty"] == "Warmup"
    assert scene_meta.get("scene_variant") == "warmup_motion_primitive"
    assert scene_meta.get("corridor_generation_mode") == "motion_primitive_warmup"
    assert int(scene_meta.get("parking_bay_count", 0)) == 1
    assert int(scene_meta.get("navigation_candidate_pair_count", 0)) == 1

    primitive_indices = scene_meta.get("warmup_motion_primitive_indices")
    assert isinstance(primitive_indices, list)
    assert len(primitive_indices) > 0
    assert int(scene_meta.get("warmup_motion_primitive_count", 0)) == len(primitive_indices)

    metrics = scene_meta["scene_metrics"]
    assert metrics is not None
    assert float(metrics["path_ratio"]) > 0.0
    assert float(metrics["path_min_clearance"]) >= 1.0
    assert float(metrics["min_endpoint_clearance"]) >= 0.8

    path_points = scene_meta.get("guidance_path_points")
    assert path_points is not None
    assert len(path_points) >= 2

    occupancy = scene_meta["occupancy_grid"]
    origin_x, origin_y = scene_meta["grid_origin"]
    block_size = float(scene_meta["block_size"])

    def _to_cell(pose):
        gx = int(np.floor((float(pose[0]) - float(origin_x)) / block_size))
        gy = int(np.floor((float(pose[1]) - float(origin_y)) / block_size))
        return gx, gy

    sx, sy = _to_cell(start)
    dx, dy = _to_cell(dest)
    assert occupancy[sy, sx] == 0
    assert occupancy[dy, dx] == 0
