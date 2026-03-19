"""Benchmark reset latency with and without the navigation scene pool.

Run:
  python src/debug/benchmark_navigation_scene_pool.py --level Extrem --resets 8 --pool-size 4
"""

import argparse
import os
import sys
import time

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from env.parking_map_normal import ParkingMapNormal


def _measure_reset_latencies(parking_map, resets: int):
    latencies = []
    for _ in range(int(resets)):
        t0 = time.perf_counter()
        parking_map.reset()
        latencies.append(time.perf_counter() - t0)
    return np.asarray(latencies, dtype=np.float64)


def _summary(name: str, latencies: np.ndarray, stats=None):
    ms = latencies * 1000.0
    print(
        f"{name}: mean={np.mean(ms):.2f}ms p50={np.percentile(ms, 50):.2f}ms "
        f"p90={np.percentile(ms, 90):.2f}ms max={np.max(ms):.2f}ms"
    )
    if stats is not None:
        print(f"  stats={stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=str, default="Extrem", choices=["Normal", "Complex", "Extrem"])
    ap.add_argument("--resets", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=4)
    args = ap.parse_args()

    t0 = time.perf_counter()
    direct_map = ParkingMapNormal(args.level, enable_scene_pool=False)
    direct_init_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pooled_map = ParkingMapNormal(args.level, enable_scene_pool=True, scene_pool_size=args.pool_size)
    pooled_init_s = time.perf_counter() - t1

    direct_latencies = _measure_reset_latencies(direct_map, args.resets)
    pooled_latencies = _measure_reset_latencies(pooled_map, args.resets)

    print(f"init: direct={direct_init_s * 1000.0:.2f}ms pooled={pooled_init_s * 1000.0:.2f}ms")
    _summary("direct", direct_latencies, direct_map.get_scene_pool_stats())
    _summary("pooled", pooled_latencies, pooled_map.get_scene_pool_stats())
    if len(pooled_latencies) > 1:
        _summary("pooled_steady", pooled_latencies[1:], pooled_map.get_scene_pool_stats())

    if len(pooled_latencies) > 0:
        direct_mean = float(np.mean(direct_latencies))
        pooled_mean = float(np.mean(pooled_latencies))
        pooled_steady_mean = float(np.mean(pooled_latencies[1:])) if len(pooled_latencies) > 1 else pooled_mean
        print(
            "speedup: "
            f"pooled_vs_direct={direct_mean / max(pooled_mean, 1e-12):.2f}x "
            f"pooled_steady_vs_direct={direct_mean / max(pooled_steady_mean, 1e-12):.2f}x"
        )


if __name__ == "__main__":
    main()