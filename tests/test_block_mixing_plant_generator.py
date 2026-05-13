from __future__ import annotations

import numpy as np

from configs import BLOCK_MIXING_PLANT_CONFIG, FRONT_HANG
from env.scene_generators.block_mixing_plant_generator import (
    _blocking_poly,
    _candidate_pose_from_bay,
    generate_block_mixing_plant_scene,
    sample_navigation_case_from_scene,
    validate_scene,
)


def _cell_center_from_scene_metadata(scene, cell):
    origin_x, origin_y = scene.metadata["grid_origin"]
    block_size = float(scene.metadata["block_size"])
    return (
        float(origin_x) + (float(cell[0]) + 0.5) * block_size,
        float(origin_y) + (float(cell[1]) + 0.5) * block_size,
    )


def _bay_signature(scene):
    return [
        (
            tuple(round(float(v), 4) for v in bay.center),
            round(float(bay.heading), 6),
            round(float(bay.length), 4),
            round(float(bay.depth), 4),
            tuple(bay.grid_cells),
            tuple(bay.access_cells or []),
        )
        for bay in scene.parking_bays
    ]


def test_block_mixing_scene_is_reproducible_for_fixed_seed():
    for level in ["Normal", "Complex", "Extrem"]:
        scene_a = generate_block_mixing_plant_scene(level, seed=7)
        scene_b = generate_block_mixing_plant_scene(level, seed=7)

        assert np.array_equal(scene_a.occupancy_grid, scene_b.occupancy_grid)
        assert np.array_equal(scene_a.free_grid, scene_b.free_grid)
        assert _bay_signature(scene_a) == _bay_signature(scene_b)
        assert scene_a.metadata["free_ratio"] == scene_b.metadata["free_ratio"]


def test_block_mixing_scene_validates_against_its_config():
    for level in ["Normal", "Complex", "Extrem"]:
        scene = generate_block_mixing_plant_scene(level, seed=3)
        assert validate_scene(scene, BLOCK_MIXING_PLANT_CONFIG[level]) is True


def test_block_mixing_scene_matches_requested_corridor_and_bay_dimensions():
    for level in ["Normal", "Complex", "Extrem"]:
        scene = generate_block_mixing_plant_scene(level, seed=7)
        config = BLOCK_MIXING_PLANT_CONFIG[level]
        corridor_low, corridor_high = config["corridor_width_range"]
        length_low, length_high = config["parking_bay_length_range"]
        depth_low, depth_high = config["parking_bay_depth_range"]

        assert corridor_low <= scene.metadata["corridor_width_min"] <= corridor_high
        assert corridor_low <= scene.metadata["corridor_width_max"] <= corridor_high
        assert len(scene.parking_bays) >= 2

        for bay in scene.parking_bays:
            assert float(length_low) <= float(bay.length) <= float(length_high)
            assert float(depth_low) <= float(bay.depth) <= float(depth_high)


def test_block_mixing_scene_bay_heading_points_into_wall():
    for level in ["Normal", "Complex", "Extrem"]:
        scene = generate_block_mixing_plant_scene(level, seed=7)

        for bay in scene.parking_bays:
            assert bay.access_cells is not None
            anchor = bay.access_cells[len(bay.access_cells) // 2]
            access_x, access_y = _cell_center_from_scene_metadata(scene, anchor)
            vector_x = float(bay.center[0]) - float(access_x)
            vector_y = float(bay.center[1]) - float(access_y)
            bay_heading = np.array([np.cos(float(bay.heading)), np.sin(float(bay.heading))], dtype=np.float64)
            into_bay = np.array([vector_x, vector_y], dtype=np.float64)

            assert np.linalg.norm(into_bay) > 0.0
            into_bay /= np.linalg.norm(into_bay)
            assert float(np.dot(bay_heading, into_bay)) > 0.999


def test_block_mixing_scene_keeps_head_about_one_meter_from_wall():
    for level in ["Normal", "Complex", "Extrem"]:
        scene = generate_block_mixing_plant_scene(level, seed=7)
        blocking_poly = _blocking_poly(list(scene.obstacle_polygons))
        wall_clearances = []

        for bay_index, bay in enumerate(scene.parking_bays):
            candidate = _candidate_pose_from_bay(scene, bay, bay_index, blocking_poly, reverse=False)
            if candidate is None:
                continue
            axis = np.array([np.cos(float(bay.heading)), np.sin(float(bay.heading))], dtype=np.float64)
            bay_coords = np.asarray(bay.polygon.exterior.coords[:-1], dtype=np.float64)
            wall_projection = float(np.max(bay_coords @ axis))
            front_projection = float(candidate.pose[0] * axis[0] + candidate.pose[1] * axis[1] + FRONT_HANG)
            wall_clearances.append(wall_projection - front_projection)

        assert len(wall_clearances) >= 2
        for clearance in wall_clearances:
            assert abs(float(clearance) - 1.0) <= 0.5


def test_block_mixing_scene_difficulty_trend_averages():
    seeds = [0, 1, 2]
    scenes = {
        level: [generate_block_mixing_plant_scene(level, seed=seed) for seed in seeds]
        for level in ["Normal", "Complex", "Extrem"]
    }

    mean_free_ratio = {
        level: float(np.mean([scene.metadata["free_ratio"] for scene in level_scenes]))
        for level, level_scenes in scenes.items()
    }
    mean_grid_width = {
        level: float(np.mean([scene.metadata["grid_width"] for scene in level_scenes]))
        for level, level_scenes in scenes.items()
    }

    assert mean_grid_width["Normal"] < mean_grid_width["Complex"] < mean_grid_width["Extrem"]
    assert mean_free_ratio["Normal"] > mean_free_ratio["Complex"] > mean_free_ratio["Extrem"]


def test_block_mixing_scene_can_sample_navigation_pair():
    for level in ["Normal", "Complex", "Extrem"]:
        scene = generate_block_mixing_plant_scene(level, seed=11)
        start, dest, nav_meta = sample_navigation_case_from_scene(scene, level, seed=11)

        assert len(start) == 3
        assert len(dest) == 3
        assert nav_meta["scene_metrics"] is not None
        assert nav_meta["guidance_path_points"] is not None
        assert len(nav_meta["guidance_path_points"]) >= 2