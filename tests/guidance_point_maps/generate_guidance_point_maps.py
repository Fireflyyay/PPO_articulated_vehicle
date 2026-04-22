import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LinearRing, MultiPolygon, Polygon


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from configs import (  # noqa: E402
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
from env.global_guidance import SoftGlobalGuidance  # noqa: E402
from env.parking_map_normal import generate_navigation_case  # noqa: E402
from env.vehicle import State  # noqa: E402


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated")


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


def _shape_coords(area_or_geom):
    geom = getattr(area_or_geom, "shape", area_or_geom)
    if isinstance(geom, LinearRing):
        return np.asarray(geom.coords, dtype=np.float64)
    if isinstance(geom, Polygon):
        return np.asarray(geom.exterior.coords, dtype=np.float64)
    if isinstance(geom, MultiPolygon):
        poly = max(list(geom.geoms), key=lambda item: float(item.area), default=None)
        if poly is None:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(poly.exterior.coords, dtype=np.float64)
    raise TypeError(f"Unsupported geometry type: {type(geom)}")


def _build_scene_map(obstacles, scene_meta):
    return SimpleNamespace(
        xmin=-40.0,
        xmax=-40.0 + 80.0,
        ymin=-40.0,
        ymax=-40.0 + 80.0,
        obstacles=obstacles,
        guidance_static_obstacles=scene_meta.get("guidance_static_obstacles", obstacles),
        guidance_dynamic_obstacles=scene_meta.get("guidance_dynamic_obstacles", []),
        guidance_occupancy_payload=scene_meta.get("guidance_occupancy_payload"),
        guidance_static_obstacle_signature=scene_meta.get("guidance_static_obstacle_signature"),
        guidance_dynamic_obstacle_signature=scene_meta.get("guidance_dynamic_obstacle_signature"),
    )


def _render_guidance_points(level: str, index: int, output_dir: str):
    start, dest, obstacles, scene_meta = generate_navigation_case(level, return_regions=True)

    planner = _make_guidance_planner()
    scene_map = _build_scene_map(obstacles, scene_meta)

    dest_state = State([dest[0], dest[1], dest[2], 0, 0])
    goal_xy = dest_state.create_box()[0].centroid.coords[0]
    ok = planner.plan_path(scene_map, start_xy=(start[0], start[1]), goal_xy=goal_xy)
    if not ok or planner.path_points_world is None:
        raise RuntimeError(f"Failed to build guidance points for level={level}")

    points = np.asarray(planner.path_points_world, dtype=np.float64)
    start_state = State([start[0], start[1], start[2], 0, 0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f7f7f7")

    for obstacle in obstacles:
        coords = _shape_coords(obstacle)
        ax.add_patch(
            MplPolygon(coords, closed=True, facecolor="#c7c7c7", edgecolor="#111111", linewidth=1.5)
        )

    start_coords = np.asarray(start_state.create_box()[0].coords, dtype=np.float64)
    dest_coords = np.asarray(dest_state.create_box()[0].coords, dtype=np.float64)
    ax.add_patch(MplPolygon(start_coords, closed=True, facecolor="#79a7ff", edgecolor="#1144aa", linewidth=1.5))
    ax.add_patch(MplPolygon(dest_coords, closed=True, fill=False, edgecolor="#0a8f2f", linewidth=2.0, linestyle="--"))

    ax.scatter(points[:, 0], points[:, 1], s=20, c="#ff6b35", alpha=0.9, label="A* guidance points")
    ax.scatter([start[0]], [start[1]], s=40, c="#1144aa", marker="o", label="start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=50, c="#0a8f2f", marker="*", label="goal center")

    ax.set_xlim(scene_map.xmin, scene_map.xmax)
    ax.set_ylim(scene_map.ymin, scene_map.ymax)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_title(f"Grid A* guidance points only | level={level} | scene={index}")
    ax.legend(loc="upper right")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"guidance_points_{level.lower()}_{index:02d}.png")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render grid A* guidance points on generated maps.")
    parser.add_argument("--level", default="Normal", choices=["Normal", "Complex", "Extrem"], help="Scene difficulty level.")
    parser.add_argument("--count", type=int, default=1, help="Number of maps to generate.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for saved PNG files.")
    args = parser.parse_args()

    saved_paths = []
    for idx in range(max(1, int(args.count))):
        saved_paths.append(_render_guidance_points(args.level, idx, args.output_dir))

    print("Saved guidance point maps:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()