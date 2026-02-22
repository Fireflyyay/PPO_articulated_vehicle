"""Sanity check: HOPE-style reward shaping consistency.

Run:
  python src/debug/verify_reward_shaping.py --n 50

It resets the env, takes a few safe actions, and when status is CONTINUE,
checks that:
  reward == sum(REWARD_WEIGHT[k] * reward_info[k]) * REWARD_RATIO

Note: if an episode terminates early (collision/outbound), the check skips that step.
"""

import os
import sys
import argparse

import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from env.car_parking_base import CarParking
from env.vehicle import Status
from configs import REWARD_WEIGHT, REWARD_RATIO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--level", type=str, default="Normal", choices=["Normal", "Complex", "Extrem"])
    args = ap.parse_args()

    env = CarParking(render_mode="rgb_array", verbose=False)

    checked = 0
    skipped = 0

    for _ in range(args.n):
        obs, info = env.reset(options={"level": args.level})

        # take a few conservative actions: small steer, low speed
        for _t in range(5):
            action = np.array([0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            status = info.get("status", Status.CONTINUE)

            if status != Status.CONTINUE:
                skipped += 1
                break

            reward_info = info.get("reward_info", {})
            shaped = 0.0
            for k, w in REWARD_WEIGHT.items():
                shaped += float(w) * float(reward_info.get(k, 0.0))
            shaped *= float(REWARD_RATIO)

            if not np.isfinite(shaped) or not np.isfinite(float(reward)):
                raise AssertionError(f"non-finite reward detected: reward={reward}, shaped={shaped}")

            if abs(float(reward) - float(shaped)) > 1e-6:
                raise AssertionError(
                    f"reward mismatch: reward={reward}, shaped={shaped}, reward_info={reward_info}"
                )

            checked += 1

            if terminated or truncated:
                break

    print(f"OK: checked {checked} CONTINUE steps (skipped episodes={skipped}).")


if __name__ == "__main__":
    main()
