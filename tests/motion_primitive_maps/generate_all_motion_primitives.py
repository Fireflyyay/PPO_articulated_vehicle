import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Polygon as MplPolygon


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from configs import HITCH_OFFSET, PRIMITIVE_LIBRARY_PATH, TRAILER_LENGTH  # noqa: E402
from env.vehicle import State, Vehicle  # noqa: E402
from primitives.library import load_library  # noqa: E402


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated")


def _resolve_library_path() -> str:
    candidate = os.path.normpath(os.path.join(SRC, PRIMITIVE_LIBRARY_PATH))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(PRIMITIVE_LIBRARY_PATH):
        return PRIMITIVE_LIBRARY_PATH
    fallback = os.path.join(ROOT, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Primitive library not found: {PRIMITIVE_LIBRARY_PATH}")


def _simulate_primitive(actions: np.ndarray):
    vehicle = Vehicle(
        articulated=True,
        trailer_length=TRAILER_LENGTH,
        hitch_offset=HITCH_OFFSET,
    )
    vehicle.reset(State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    for action in np.asarray(actions, dtype=np.float64):
        vehicle.step(action)
    return vehicle.trajectory


def _front_and_rear_polygons(state: State):
    front_box, rear_box = state.create_box()
    return np.asarray(front_box.coords, dtype=np.float64), np.asarray(rear_box.coords, dtype=np.float64)


def _plot_body(ax, state: State, facecolor: str, edgecolor: str, alpha: float, linewidth: float):
    front_coords, rear_coords = _front_and_rear_polygons(state)
    ax.add_patch(
        MplPolygon(front_coords, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    )
    ax.add_patch(
        MplPolygon(rear_coords, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth)
    )


def _collect_primitive_rollouts(primitive_lib):
    rollouts = []
    all_xy = []
    for primitive_id in range(int(primitive_lib.size)):
        actions = primitive_lib.get_actions(primitive_id)
        trajectory = _simulate_primitive(actions)
        xy = np.asarray([[float(s.loc.x), float(s.loc.y)] for s in trajectory], dtype=np.float64)
        rollouts.append(
            {
                "primitive_id": primitive_id,
                "actions": np.asarray(actions, dtype=np.float64),
                "trajectory": trajectory,
                "xy": xy,
                "final_state": trajectory[-1],
            }
        )
        all_xy.append(xy)
    all_xy = np.vstack(all_xy) if all_xy else np.zeros((0, 2), dtype=np.float64)
    return rollouts, all_xy


def _compute_axis_limits(all_xy: np.ndarray, rollouts) -> tuple:
    xs = [] if all_xy.size == 0 else [all_xy[:, 0].min(), all_xy[:, 0].max()]
    ys = [] if all_xy.size == 0 else [all_xy[:, 1].min(), all_xy[:, 1].max()]
    for rollout in rollouts:
        front_coords, rear_coords = _front_and_rear_polygons(rollout["final_state"])
        xs.extend([front_coords[:, 0].min(), front_coords[:, 0].max(), rear_coords[:, 0].min(), rear_coords[:, 0].max()])
        ys.extend([front_coords[:, 1].min(), front_coords[:, 1].max(), rear_coords[:, 1].min(), rear_coords[:, 1].max()])
    if not xs or not ys:
        return (-5.0, 5.0, -5.0, 5.0)
    xmin = float(min(xs))
    xmax = float(max(xs))
    ymin = float(min(ys))
    ymax = float(max(ys))
    span = max(xmax - xmin, ymax - ymin, 1.0)
    pad = max(1.5, 0.12 * span)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def render_all_motion_primitives(output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    lib_path = _resolve_library_path()
    primitive_lib = load_library(lib_path)
    rollouts, all_xy = _collect_primitive_rollouts(primitive_lib)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#fbfbfb")

    cmap = colormaps.get_cmap("turbo").resampled(max(primitive_lib.size, 2))
    start_state = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _plot_body(ax, start_state, facecolor="#1f77b4", edgecolor="#0d3b66", alpha=0.35, linewidth=2.0)

    for rollout in rollouts:
        primitive_id = rollout["primitive_id"]
        color = cmap(primitive_id)
        xy = rollout["xy"]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.6, alpha=0.95)
        _plot_body(ax, rollout["final_state"], facecolor=color, edgecolor="#222222", alpha=0.28, linewidth=1.0)
        ax.text(
            float(xy[-1, 0]),
            float(xy[-1, 1]),
            str(primitive_id),
            fontsize=8,
            color="#111111",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.6},
        )

    xmin, xmax, ymin, ymax = _compute_axis_limits(all_xy, rollouts)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)
    ax.set_title(
        f"All Motion Primitives | N={primitive_lib.size} | H={primitive_lib.horizon}\n"
        f"full articulated body shown at start and primitive endpoints",
        fontsize=16,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_motion_primitives_overview.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    output_path = render_all_motion_primitives()
    print("Saved motion primitive overview:")
    print(output_path)


if __name__ == "__main__":
    main()