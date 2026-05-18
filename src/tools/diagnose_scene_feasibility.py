#!/usr/bin/env python
"""Diagnose scene feasibility: check config-level clearance vs vehicle width.

This script performs a static config analysis, not runtime scene generation.
For runtime analysis, run training with --exp-tag diag and use diagnose_mask_ablation.py
on the resulting episodes.jsonl.

Usage:
    python src/tools/diagnose_scene_feasibility.py [--vehicle-width 3.43]
"""
from __future__ import annotations

import argparse
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from configs import BLOCK_MIXING_PLANT_CONFIG, NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL


def main():
    parser = argparse.ArgumentParser(description="Static scene feasibility analysis")
    parser.add_argument("--vehicle-width", type=float, default=3.43, help="Vehicle width in meters")
    args = parser.parse_args()

    vehicle_width = float(args.vehicle_width)
    min_req_half = vehicle_width / 2.0

    print(f"Vehicle width: {vehicle_width}m")
    print(f"Minimum required path clearance (half-width): {min_req_half:.2f}m")
    print(f"Effective corridor width needed: {vehicle_width:.2f}m")
    print()
    print(f"{'Level':<12s} {'scene_min_clearance':>20s} {'nav_min_clearance':>20s} {'Status':<20s}")
    print("-" * 72)

    for level in ('Warmup', 'Normal', 'Complex', 'Extrem'):
        cfg = BLOCK_MIXING_PLANT_CONFIG.get(level, {})
        scene_val = float(cfg.get("scene_metric_min_path_clearance", -1.0))
        nav_val = float(NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL.get(level, -1.0))
        corridor = cfg.get("corridor_width_range", "N/A")

        scene_status = "OK" if scene_val >= min_req_half else "TOO NARROW"
        nav_status = "OK" if nav_val >= min_req_half else "TOO NARROW"

        print(f"{level:<12s} {scene_val:>20.2f}m {nav_val:>20.2f}m ", end="")
        print(f"scene={scene_status:<10s} nav={nav_status:<10s}  corridor={corridor}")

    print()
    print("Key:")
    print("  scene_min_clearance: from BLOCK_MIXING_PLANT_CONFIG[level].scene_metric_min_path_clearance")
    print("  nav_min_clearance:   from NAVIGATION_MIN_PATH_CLEARANCE_BY_LEVEL[level]")
    print("  Both measure min distance from A* path cell center to free-space boundary.")
    print("  Effective corridor = 2 * clearance value.")
    print(f"  Vehicle {vehicle_width}m wide needs clearance >= {min_req_half:.2f}m")

    worst_offenders = []
    for level in ('Warmup', 'Normal', 'Complex', 'Extrem'):
        cfg = BLOCK_MIXING_PLANT_CONFIG.get(level, {})
        scene_val = float(cfg.get("scene_metric_min_path_clearance", 999.0))
        if scene_val < min_req_half:
            worst_offenders.append((level, scene_val))

    if worst_offenders:
        print()
        print("⚠️  CONFIG ISSUES FOUND:")
        for level, val in worst_offenders:
            corridor = 2.0 * val
            print(f"  [{level}] scene_metric_min_path_clearance={val:.2f}m -> corridor={corridor:.2f}m < vehicle={vehicle_width:.2f}m")
        print()
        print("RECOMMENDATION: Set scene_metric_min_path_clearance >= {:.2f} for all levels".format(min_req_half))
    else:
        print()
        print("✓ All levels have sufficient path clearance for this vehicle width.")


if __name__ == "__main__":
    main()
