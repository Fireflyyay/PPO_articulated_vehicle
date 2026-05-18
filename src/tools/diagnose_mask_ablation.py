#!/usr/bin/env python
"""Diagnose action mask: compare original vs relaxed vs no-mask choices.

Reads a recorded episodes.jsonl file and replays mask computation
offline to compare what actions would have been selected under
different mask configurations.

Usage:
    python src/tools/diagnose_mask_ablation.py \
        --input outputs/exp/baseline_20260518_120000/episodes.jsonl \
        --library data/primitives_family_semantic_main_G31_M3_V3.npz \
        --output outputs/diag/mask_ablation/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def load_episodes(jsonl_path: str) -> list:
    episodes = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Diagnose action mask ablation offline")
    parser.add_argument("--input", type=str, required=True, help="Path to episodes.jsonl")
    parser.add_argument("--library", type=str, required=True, help="Path to primitive library .npz")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    episodes = load_episodes(args.input)
    if len(episodes) == 0:
        print("No episodes found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(episodes)} episodes from {args.input}")
    print()

    # Summarize what we can from the recorded episode data without re-running simulation
    status_counts = {}
    total_steps = 0
    total_blocked = 0
    family_hist_global = {}
    mode_ratios = {'normal': [], 'narrow_escape': [], 'terminal': []}
    valid_ratios = []
    blocked_rates = []
    min_goal_dists = []
    max_overlaps = []
    stuck_snapshots = []

    for ep in episodes:
        status = str(ep.get('status', 'UNKNOWN'))
        status_counts[status] = status_counts.get(status, 0) + 1

        total_steps += int(ep.get('step_num', 0))
        blocked_rate = float(ep.get('avg_soft_ray_blocked_rate', 0.0))
        blocked_rates.append(blocked_rate)
        if blocked_rate > 0.5:
            total_blocked += 1

        valid_ratios.append(float(ep.get('avg_valid_action_ratio', 0.0)))

        mr = ep.get('mode_ratios', {})
        for mode in mode_ratios:
            mode_ratios[mode].append(float(mr.get(mode, 0.0)))

        fh = ep.get('family_select_hist', {})
        for fid, cnt in fh.items():
            family_hist_global[fid] = family_hist_global.get(fid, 0) + int(cnt)

        d = ep.get('diag_min_goal_dist', float('nan'))
        if d is not None and not np.isnan(d) and d < float('inf'):
            min_goal_dists.append(float(d))

        o = ep.get('diag_max_front_overlap', float('nan'))
        if o is not None and not np.isnan(o):
            max_overlaps.append(float(o))

        ss = ep.get('stuck_snapshot')
        if ss is not None:
            stuck_snapshots.append(ss)

    print("=== Episode Summary ===")
    print(f"Total episodes: {len(episodes)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} ({count/len(episodes)*100:.1f}%)")
    print()
    print(f"Average steps/episode: {total_steps/max(1,len(episodes)):.1f}")
    print(f"Episodes with >50% blocked rate: {total_blocked}/{len(episodes)} ({total_blocked/max(1,len(episodes))*100:.1f}%)")
    print(f"Average valid action ratio: {np.mean(valid_ratios):.3f}" if valid_ratios else "N/A")
    print(f"Average blocked rate: {np.mean(blocked_rates):.3f}" if blocked_rates else "N/A")

    print()
    print("=== Mode Ratios (avg over episodes) ===")
    for mode, vals in mode_ratios.items():
        if vals:
            print(f"  {mode}: {np.mean(vals):.3f}")
        else:
            print(f"  {mode}: N/A")

    print()
    print("=== Goal Progress ===")
    if min_goal_dists:
        print(f"  Min goal dist: best={min(min_goal_dists):.2f}m, mean={np.mean(min_goal_dists):.2f}m, median={np.median(min_goal_dists):.2f}m")
    if max_overlaps:
        print(f"  Max front overlap: best={max(max_overlaps):.3f}, mean={np.mean(max_overlaps):.3f}")

    print()
    print("=== Family Selection (global histogram) ===")
    if family_hist_global:
        sorted_fams = sorted(family_hist_global.items(), key=lambda x: -x[1])
        top_n = min(15, len(sorted_fams))
        print(f"  Top {top_n} families:")
        for fid, cnt in sorted_fams[:top_n]:
            print(f"    family_{fid}: {cnt} selections")
        unique_fams = len(family_hist_global)
        print(f"  Unique families selected: {unique_fams}")
        # try to load family names
        try:
            data = np.load(args.library, allow_pickle=True)
            family_names = [str(v) for v in data["family_names"].reshape(-1)]
            family_types = [str(v) for v in data["family_types"].reshape(-1)]
            print(f"  Family names (top 15):")
            for fid, cnt in sorted_fams[:15]:
                name = family_names[int(fid)] if int(fid) < len(family_names) else f"family-{fid}"
                ftype = family_types[int(fid)] if int(fid) < len(family_types) else "?"
                print(f"    [{fid}] {name} ({ftype}): {cnt}")
        except Exception:
            pass

    print()
    print("=== Stuck Snapshots (if any) ===")
    if stuck_snapshots:
        print(f"  Count: {len(stuck_snapshots)}")
        # aggregate stuck snapshot stats
        keys = ['diag_stuck_goal_dist', 'diag_stuck_phi_abs', 'diag_stuck_valid_actions',
                'diag_stuck_front_clearance', 'diag_stuck_rear_clearance',
                'diag_stuck_front_overlap', 'diag_stuck_heading_error_deg',
                'diag_stuck_consecutive_blocked']
        for key in keys:
            vals = [ss.get(key, float('nan')) for ss in stuck_snapshots if ss.get(key) is not None]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                print(f"  {key}: mean={np.mean(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")
    else:
        print("  No stuck episodes recorded.")

    print()
    print("=== Mask Ablation Estimation (offline, coarse) ===")
    print("NOTE: This is a statistical analysis of recorded data only.")
    print("To properly ablate the mask, run training with modified SOFT_MASK_EPS/GAMMA.")
    print()
    print(f"Current avg blocked rate: {np.mean(blocked_rates):.3f}" if blocked_rates else "N/A")
    print("Key question: If mask were relaxed (higher EPS, lower GAMMA), would more families become selectable?")
    print("→ Check family_select_hist above. If only 2-3 families dominate, mask may be too restrictive.")
    if family_hist_global:
        unique_fams = len(family_hist_global)
        total_fams = 48
        print(f"→ Currently {unique_fams}/{total_fams} families ever selected.")
        if unique_fams < 8:
            print("⚠️  Very few families selected - mask likely overly restrictive.")
        elif unique_fams < 20:
            print("⚠️  Moderate family diversity - mask may be limiting exploration.")
        else:
            print("✓  Good family diversity.")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        summary_path = os.path.join(args.output, "mask_ablation_summary.json")
        summary = {
            'num_episodes': len(episodes),
            'status_counts': status_counts,
            'avg_steps': total_steps / max(1, len(episodes)),
            'avg_valid_action_ratio': float(np.mean(valid_ratios)) if valid_ratios else 0,
            'avg_blocked_rate': float(np.mean(blocked_rates)) if blocked_rates else 0,
            'mode_ratio_means': {m: float(np.mean(v)) for m, v in mode_ratios.items() if v},
            'min_goal_dist_best': float(min(min_goal_dists)) if min_goal_dists else -1,
            'max_overlap_best': float(max(max_overlaps)) if max_overlaps else -1,
            'unique_families_selected': len(family_hist_global),
            'stuck_snapshot_count': len(stuck_snapshots),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
