#!/usr/bin/env python
import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Ensure src/ is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from env.vehicle import Vehicle, State
from primitives.primitive_index import Cell


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
):
    n, h, _ = actions.shape

    def world_to_cell(x: float, y: float):
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return None
        ix = int(np.floor((x - x_min) / grid_resolution))
        iy = int(np.floor((y - y_min) / grid_resolution))
        return (ix, iy)

    primitive_to_cells: List[np.ndarray] = []
    cell_to_prims: Dict[Cell, List[int]] = defaultdict(list)

    # Approximate control-group: speed bin (paper uses control groups; here we cluster by speed value)
    speeds = actions[:, 0, 1]
    uniq = sorted(list({float(np.round(v, 3)) for v in speeds}))
    speed_to_gid = {v: i for i, v in enumerate(uniq)}
    primitive_to_group_id = np.array([speed_to_gid[float(np.round(v, 3))] for v in speeds], dtype=np.int64)

    group_to_primitive_ids: List[List[int]] = [[] for _ in range(len(uniq))]
    for pid, gid in enumerate(primitive_to_group_id.tolist()):
        group_to_primitive_ids[gid].append(pid)

    # simulate trajectories in canonical ego frame
    for pid in range(n):
        init_state = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vehicle = Vehicle(articulated=True)
        vehicle.reset(init_state)

        visited = set()
        for t in range(0, h, max(1, int(sample_stride))):
            a = actions[pid, t]
            vehicle.step(a, step_time=int(num_step))
            x = float(vehicle.state.loc.x)
            y = float(vehicle.state.loc.y)
            c = world_to_cell(x, y)
            if c is not None:
                visited.add(c)

        if len(visited) == 0:
            arr = np.zeros((0, 2), dtype=np.int64)
        else:
            arr = np.array(sorted(list(visited)), dtype=np.int64)

        primitive_to_cells.append(arr)
        for cell in visited:
            cell_to_prims[cell].append(pid)

    cell_to_primitives = {k: np.array(v, dtype=np.int64) for k, v in cell_to_prims.items()}
    group_to_primitive_ids_arr = [np.array(v, dtype=np.int64) for v in group_to_primitive_ids]
    group_prefix_steps_arr = np.full((len(group_to_primitive_ids_arr),), int(group_prefix_steps), dtype=np.int64)

    payload = dict(
        grid_resolution=float(grid_resolution),
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=float(y_min),
        y_max=float(y_max),
        primitive_to_cells=np.array(primitive_to_cells, dtype=object),
        cell_to_primitives=np.array(cell_to_primitives, dtype=object),
        primitive_to_group_id=primitive_to_group_id,
        group_to_primitive_ids=np.array(group_to_primitive_ids_arr, dtype=object),
        group_prefix_steps=group_prefix_steps_arr,
    )
    return payload


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

    args = parser.parse_args()

    lib_path = os.path.abspath(args.library)
    if args.out is None:
        base, _ = os.path.splitext(lib_path)
        out_path = base + ".grid_index.npz"
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
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote grid index: {out_path}")


if __name__ == "__main__":
    main()
