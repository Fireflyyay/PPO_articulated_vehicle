#!/usr/bin/env python
import argparse
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from primitives.primitive_ray_safety import (  # noqa: E402
    build_ray_safety_index_from_library,
    default_ray_safety_path,
    save_ray_safety_index,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, required=True, help="Path to primitive library .npz")
    parser.add_argument("--out", type=str, default=None, help="Output .ray_safety.npz path")
    parser.add_argument("--lidar-num", type=int, default=None)
    parser.add_argument("--lidar-range", type=float, default=None)
    parser.add_argument("--safety-margin", type=float, default=None)
    parser.add_argument("--reverse-margin-scale", type=float, default=None)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--num-step", type=int, default=None)
    args = parser.parse_args()

    try:
        import configs as cfg
    except Exception:
        cfg = None

    lidar_num = int(args.lidar_num if args.lidar_num is not None else getattr(cfg, "LIDAR_NUM", 120))
    lidar_range = float(args.lidar_range if args.lidar_range is not None else getattr(cfg, "LIDAR_RANGE", 30.0))
    safety_margin = float(
        args.safety_margin if args.safety_margin is not None else getattr(cfg, "SOFT_MASK_SAFETY_MARGIN", 0.25)
    )
    reverse_margin_scale = float(
        args.reverse_margin_scale
        if args.reverse_margin_scale is not None
        else getattr(cfg, "SOFT_MASK_REVERSE_MARGIN_SCALE", 1.2)
    )
    num_step = int(args.num_step if args.num_step is not None else getattr(cfg, "NUM_STEP", 4))

    library_path = os.path.abspath(args.library)
    out_path = os.path.abspath(args.out) if args.out else default_ray_safety_path(library_path)
    data = np.load(library_path, allow_pickle=True)
    index = build_ray_safety_index_from_library(
        actions=data["actions"],
        deltas=data["deltas"],
        lidar_num=lidar_num,
        lidar_range=lidar_range,
        safety_margin=safety_margin,
        reverse_margin_scale=reverse_margin_scale,
        sample_stride=int(args.sample_stride),
        num_step=num_step,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_ray_safety_index(out_path, index)
    print(
        f"saved {out_path} dist_star={tuple(index.dist_star.shape)} "
        f"margin={safety_margin:.3f} reverse_scale={reverse_margin_scale:.3f}"
    )


if __name__ == "__main__":
    main()
