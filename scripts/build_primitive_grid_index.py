#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from primitives.primitive_index import (
    build_approx_index_from_deltas,
    build_mask_index_from_library,
    primitive_grid_index_to_payload,
)


def build_index(
    actions: np.ndarray,
    deltas: np.ndarray,
    grid_resolution: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sample_stride: int,
    num_step: int,
    group_prefix_steps: int,
    index_mode: str = "swept_cells",
):
    mode = str(index_mode).strip().lower()
    if mode == "approx_centerline":
        index = build_approx_index_from_deltas(
            actions=actions,
            deltas=deltas,
            grid_resolution=grid_resolution,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            group_prefix_steps=group_prefix_steps,
        )
    else:
        index = build_mask_index_from_library(
            actions=actions,
            grid_resolution=grid_resolution,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            sample_stride=sample_stride,
            num_step=num_step,
            group_prefix_steps=group_prefix_steps,
        )
    return primitive_grid_index_to_payload(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, required=True, help="Path to primitive library .npz")
    parser.add_argument("--out", type=str, default=None, help="Output index file (.grid_index.npz). Default: <library>.grid_index.npz")

    parser.add_argument("--grid_resolution", type=float, default=0.3)
    parser.add_argument("--x_min", type=float, default=-6.0)
    parser.add_argument("--x_max", type=float, default=12.0)
    parser.add_argument("--y_min", type=float, default=-9.0)
    parser.add_argument("--y_max", type=float, default=9.0)

    parser.add_argument("--sample_stride", type=int, default=1, help="Trajectory sampling stride along primitive steps")
    parser.add_argument("--num_step", type=int, default=None, help="Physics substeps per env step (NUM_STEP). Default: use src.configs.NUM_STEP")
    parser.add_argument("--group_prefix_steps", type=int, default=None, help="Shared prefix steps for all groups")
    parser.add_argument("--group_prefix_ratio", type=float, default=0.3, help="If group_prefix_steps not set, use int(H*ratio)")
    parser.add_argument(
        "--index_mode",
        type=str,
        default="swept_cells",
        choices=("swept_cells", "approx_centerline"),
        help="Offline index type: conservative swept-cells mask or approximate centerline fallback",
    )

    args = parser.parse_args()

    lib_path = os.path.abspath(args.library)
    if args.out is None:
        base, _ = os.path.splitext(lib_path)
        suffix = ".mask_index.npz" if args.index_mode == "swept_cells" else ".grid_index.npz"
        out_path = base + suffix
    else:
        out_path = os.path.abspath(args.out)

    data = np.load(lib_path, allow_pickle=True)
    actions = data["actions"]
    deltas = data["deltas"]

    h = int(actions.shape[1])

    if args.num_step is None:
        try:
            from configs import NUM_STEP

            num_step = int(NUM_STEP)
        except Exception:
            num_step = 4
    else:
        num_step = int(args.num_step)

    if args.group_prefix_steps is None:
        group_prefix_steps = max(1, min(h, int(round(h * float(args.group_prefix_ratio)))))
    else:
        group_prefix_steps = int(args.group_prefix_steps)

    payload = build_index(
        actions=actions,
        deltas=deltas,
        grid_resolution=float(args.grid_resolution),
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        y_min=float(args.y_min),
        y_max=float(args.y_max),
        sample_stride=int(args.sample_stride),
        num_step=num_step,
        group_prefix_steps=group_prefix_steps,
        index_mode=str(args.index_mode),
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote grid index: {out_path}")


if __name__ == "__main__":
    main()
