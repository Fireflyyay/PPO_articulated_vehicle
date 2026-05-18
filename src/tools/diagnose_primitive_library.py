#!/usr/bin/env python
"""Diagnose primitive library: endpoint deltas per family × gamma_bin.

Outputs CSV with: family_id, family_name, family_type, gamma_bin_id, gamma_value,
variant_id, mode, dx, dy, dtheta, dphi, effective_horizon, duration, speed_sign.

Usage:
    python src/tools/diagnose_primitive_library.py \
        --library data/primitives_family_semantic_main_G31_M3_V3.npz \
        --output outputs/diag/primitive_analysis.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def main():
    parser = argparse.ArgumentParser(description="Diagnose primitive library deltas")
    parser.add_argument("--library", type=str, required=True, help="Path to .npz family library")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=0, help="Print top N rows to stdout (0=disable)")
    args = parser.parse_args()

    lib_path = os.path.abspath(args.library)
    if not os.path.exists(lib_path):
        print(f"Library not found: {lib_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(lib_path, allow_pickle=True)
    actions = data["actions"]
    deltas = data["deltas"]
    variant_horizons = data["variant_horizons"].reshape(-1)
    durations = data["durations"].reshape(-1)
    speed_signs = data["speed_signs"].reshape(-1)
    is_compound = data["is_compound"].reshape(-1)
    gamma_bin_values = data["gamma_bin_values"].reshape(-1)

    flat_to_gamma = data["variant_flat_to_gamma"].reshape(-1)
    flat_to_family = data["variant_flat_to_family"].reshape(-1)
    flat_to_variant = data["variant_flat_to_variant"].reshape(-1)
    flat_to_mode = [str(v) for v in data["variant_flat_to_mode"].reshape(-1)]
    flat_to_type = [str(v) for v in data["variant_flat_to_family_type"].reshape(-1)]

    family_names = [str(v) for v in data["family_names"].reshape(-1)]
    family_types = [str(v) for v in data["family_types"].reshape(-1)]

    n = int(actions.shape[0])
    header = [
        "flat_index",
        "family_id",
        "family_name",
        "family_type",
        "gamma_bin_id",
        "gamma_value_deg",
        "variant_id",
        "mode",
        "dx",
        "dy",
        "dtheta_deg",
        "dphi_deg",
        "effective_horizon",
        "duration_s",
        "speed_sign",
        "is_compound",
    ]

    rows = []
    for i in range(n):
        fid = int(flat_to_family[i])
        gid = int(flat_to_gamma[i])
        gamma_val = float(gamma_bin_values[gid]) if gid < len(gamma_bin_values) else 0.0
        delta = deltas[i]
        row = [
            i,
            fid,
            family_names[fid] if fid < len(family_names) else f"family-{fid}",
            family_types[fid] if fid < len(family_types) else "unknown",
            gid,
            round(np.rad2deg(gamma_val), 2),
            int(flat_to_variant[i]),
            flat_to_mode[i] if i < len(flat_to_mode) else "unknown",
            round(float(delta[0]), 4),
            round(float(delta[1]), 4),
            round(np.rad2deg(float(delta[2])), 2),
            round(np.rad2deg(float(delta[3])), 2),
            int(variant_horizons[i]),
            round(float(durations[i]), 4),
            int(speed_signs[i]),
            int(is_compound[i]),
        ]
        rows.append(row)

    # print summary stats
    print(f"Library: {lib_path}")
    print(f"Total flat variants: {n}")
    print(f"Families: {len(family_names)}")
    print(f"Gamma bins: {len(gamma_bin_values)}")
    print(f"Horizon: {actions.shape[1]}")
    print(f"Variant horizons: min={variant_horizons.min()}, max={variant_horizons.max()}, mean={variant_horizons.mean():.1f}")
    print(f"Durations: min={durations.min():.3f}s, max={durations.max():.3f}s, mean={durations.mean():.3f}s")
    print()

    # per-family summary
    print("Per-family deltas (averaged over gamma bins & variants):")
    print(f"{'fid':>3s}  {'name':<40s}  {'type':<12s}  {'dx':>7s}  {'dy':>7s}  {'dtheta°':>8s}  {'dphi°':>8s}  {'H_eff':>5s}")
    for fid in range(len(family_names)):
        mask = flat_to_family == fid
        if not np.any(mask):
            continue
        d = deltas[mask]
        h = variant_horizons[mask]
        print(
            f"{fid:3d}  {family_names[fid]:<40s}  {family_types[fid]:<12s}  "
            f"{np.mean(d[:, 0]):7.3f}  {np.mean(d[:, 1]):7.3f}  "
            f"{np.rad2deg(np.mean(d[:, 2])):8.1f}  {np.rad2deg(np.mean(d[:, 3])):8.1f}  "
            f"{np.mean(h):5.1f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")

    if args.top_n > 0:
        print(f"\nFirst {args.top_n} rows:")
        print(",".join(header))
        for row in rows[: args.top_n]:
            print(",".join(str(v) for v in row))


if __name__ == "__main__":
    main()
