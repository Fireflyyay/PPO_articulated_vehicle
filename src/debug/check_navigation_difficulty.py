"""Validate navigation scene difficulty constraints.

Run:
  python src/debug/check_navigation_difficulty.py --n 200

It samples scenes for each level (Normal/Complex/Extrem) using
`generate_navigation_case(level)` and asserts:
- distance constraints (meters)
- heading-difference constraints (degrees)

This is a lightweight sanity check, not a full env rollout.
"""

import os
import sys
import argparse

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from env.parking_map_normal import generate_navigation_case


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _abs_angle_diff(a: float, b: float) -> float:
    return abs(_wrap_pi(a - b))


def _constraints(level: str):
    if level == "Normal":
        return (0.0, 30.0), (0.0, 45.0)
    if level == "Complex":
        return (30.0, 50.0), (45.0, 90.0)
    if level == "Extrem":
        return (50.0, None), (60.0, 180.0)
    raise ValueError(level)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    rng_levels = ["Normal", "Complex", "Extrem"]

    for level in rng_levels:
        (dmin, dmax), (amin, amax) = _constraints(level)
        ds = []
        ads = []

        for _ in range(args.n):
            start, dest, _ = generate_navigation_case(level)
            sx, sy, syaw = start
            dx, dy, dyaw = dest

            dist = float(np.hypot(sx - dx, sy - dy))
            ad = float(np.rad2deg(_abs_angle_diff(syaw, dyaw)))

            ds.append(dist)
            ads.append(ad)

            assert dist >= dmin - 1e-6, f"{level}: dist {dist:.3f} < {dmin}"
            if dmax is not None:
                assert dist <= dmax + 1e-6, f"{level}: dist {dist:.3f} > {dmax}"

            assert ad >= amin - 1e-6, f"{level}: heading diff {ad:.3f} < {amin}"
            assert ad <= amax + 1e-6, f"{level}: heading diff {ad:.3f} > {amax}"

        print(
            f"{level}: dist mean={np.mean(ds):.2f} min={np.min(ds):.2f} max={np.max(ds):.2f} | "
            f"heading_diff(deg) mean={np.mean(ads):.1f} min={np.min(ads):.1f} max={np.max(ads):.1f}"
        )

    print("OK: all sampled scenes satisfy difficulty constraints")


if __name__ == "__main__":
    main()
