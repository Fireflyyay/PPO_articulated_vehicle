from types import SimpleNamespace

from configs import (
    GUIDANCE_FULL_CLEARANCE_M,
    GUIDANCE_GRID_RESOLUTION,
    GUIDANCE_LOOKAHEAD_BASE,
    GUIDANCE_LOOKAHEAD_MAX,
    GUIDANCE_LOOKAHEAD_MIN,
    GUIDANCE_LOOKAHEAD_SPEED_GAIN,
    GUIDANCE_MAP_MARGIN,
    GUIDANCE_MAX_DENSE_RATIO,
    GUIDANCE_MIN_CLEARANCE_M,
    GUIDANCE_NEAR_OBS_DIST_M,
    GUIDANCE_OBS_INFLATION,
    GUIDANCE_PROGRESS_WINDOW,
)
from env.global_guidance import SoftGlobalGuidance
from env.parking_map_normal import ParkingMapNormal, generate_navigation_case
from env.vehicle import State


def _make_guidance_planner():
    return SoftGlobalGuidance(
        grid_resolution=GUIDANCE_GRID_RESOLUTION,
        obstacle_inflation=GUIDANCE_OBS_INFLATION,
        map_margin=GUIDANCE_MAP_MARGIN,
        lookahead_base=GUIDANCE_LOOKAHEAD_BASE,
        lookahead_speed_gain=GUIDANCE_LOOKAHEAD_SPEED_GAIN,
        lookahead_min=GUIDANCE_LOOKAHEAD_MIN,
        lookahead_max=GUIDANCE_LOOKAHEAD_MAX,
        progress_search_window=GUIDANCE_PROGRESS_WINDOW,
        min_clearance_m=GUIDANCE_MIN_CLEARANCE_M,
        full_clearance_m=GUIDANCE_FULL_CLEARANCE_M,
        near_obs_dist_m=GUIDANCE_NEAR_OBS_DIST_M,
        max_dense_ratio=GUIDANCE_MAX_DENSE_RATIO,
    )


def test_navigation_scene_meta_carries_guidance_occupancy_payload():
    _, _, _, scene_meta = generate_navigation_case("Normal", return_regions=True)

    payload = scene_meta.get("guidance_occupancy_payload")
    assert payload is not None
    assert tuple(payload["occ"].shape) == (82, 82)
    assert tuple(payload["static_occ"].shape) == (82, 82)
    assert tuple(payload["dynamic_occ"].shape) == (82, 82)
    assert tuple(payload["bounds"]) == (-40.5, 40.5, -40.5, 40.5)
    assert len(str(payload["obstacle_signature"])) > 0
    assert len(str(payload["static_obstacle_signature"])) > 0
    assert len(str(payload["dynamic_obstacle_signature"])) > 0
    assert scene_meta.get("guidance_static_obstacles") is not None
    assert scene_meta.get("guidance_dynamic_obstacles") is not None

    parking_map = ParkingMapNormal("Normal", enable_scene_pool=False)
    parking_map.reset()

    assert parking_map.guidance_occupancy_payload is not None
    assert tuple(parking_map.guidance_occupancy_payload["occ"].shape) == (82, 82)
    assert parking_map.guidance_static_obstacles is not None
    assert parking_map.guidance_dynamic_obstacles is not None


def test_soft_global_guidance_reports_payload_and_instance_cache_hits():
    start, dest, obstacles, scene_meta = generate_navigation_case("Normal", return_regions=True)
    payload = scene_meta["guidance_occupancy_payload"]

    dest_state = State([dest[0], dest[1], dest[2], 0, 0])
    goal_xy = dest_state.create_box()[0].centroid.coords[0]

    scene_map = SimpleNamespace(
        xmin=-40.0,
        xmax=40.0,
        ymin=-40.0,
        ymax=40.0,
        obstacles=obstacles,
        guidance_occupancy_payload=payload,
    )

    planner = _make_guidance_planner()

    ok_with_payload = planner.plan_path(scene_map, start_xy=(start[0], start[1]), goal_xy=goal_xy)
    assert ok_with_payload is True
    assert planner.get_last_occupancy_cache_status() == "payload_hit"
    payload_details = planner.get_last_occupancy_cache_details()
    assert payload_details["combined_builder"] == "payload"

    delattr(scene_map, "guidance_occupancy_payload")
    ok_with_instance_cache = planner.plan_path(scene_map, start_xy=(start[0], start[1]), goal_xy=goal_xy)
    assert ok_with_instance_cache is True
    assert planner.get_last_occupancy_cache_status() == "instance_hit"
    instance_details = planner.get_last_occupancy_cache_details()
    assert instance_details["combined_builder"] == "instance"


def test_soft_global_guidance_reports_layered_hits_without_payload():
    start, dest, obstacles, scene_meta = generate_navigation_case("Normal", return_regions=True)

    dest_state = State([dest[0], dest[1], dest[2], 0, 0])
    goal_xy = dest_state.create_box()[0].centroid.coords[0]

    scene_map = SimpleNamespace(
        xmin=-40.0,
        xmax=40.0,
        ymin=-40.0,
        ymax=40.0,
        obstacles=obstacles,
        guidance_static_obstacles=scene_meta["guidance_static_obstacles"],
        guidance_dynamic_obstacles=scene_meta["guidance_dynamic_obstacles"],
        guidance_static_obstacle_signature=scene_meta["guidance_static_obstacle_signature"],
        guidance_dynamic_obstacle_signature=scene_meta["guidance_dynamic_obstacle_signature"],
    )

    planner = _make_guidance_planner()

    ok_first = planner.plan_path(scene_map, start_xy=(start[0], start[1]), goal_xy=goal_xy)
    assert ok_first is True
    assert planner.get_last_occupancy_cache_status() == "miss"
    first_details = planner.get_last_occupancy_cache_details()
    assert first_details["combined_builder"] == "raster"
    assert first_details["static_builder"] == "raster"
    if len(scene_meta["guidance_dynamic_obstacles"]) == 0:
        assert first_details["dynamic_builder"] == "empty"
    else:
        assert first_details["dynamic_builder"] == "raster"

    planner.clear_occupancy_cache(keep_layer_caches=True)
    if hasattr(scene_map, "guidance_occupancy_payload"):
        delattr(scene_map, "guidance_occupancy_payload")
    ok_second = planner.plan_path(scene_map, start_xy=(start[0], start[1]), goal_xy=goal_xy)
    assert ok_second is True
    assert planner.get_last_occupancy_cache_status() == "layered_hit"

    cache_details = planner.get_last_occupancy_cache_details()
    assert cache_details["static"] == "hit"
    assert cache_details["combined_builder"] == "cache"
    assert cache_details["static_builder"] == "cache"
    if len(scene_meta["guidance_dynamic_obstacles"]) == 0:
        assert cache_details["dynamic"] == "empty"
        assert cache_details["dynamic_builder"] == "empty"
    else:
        assert cache_details["dynamic"] == "hit"
        assert cache_details["dynamic_builder"] == "cache"