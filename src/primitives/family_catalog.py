from __future__ import annotations

from typing import List

import numpy as np

from configs import VALID_SPEED
from primitives.primitive_def import PrimitiveFamilySpec


def _normalized_steer_levels(steer_levels: int) -> np.ndarray:
    count = int(max(3, steer_levels))
    return np.linspace(-1.0, 1.0, count, dtype=np.float64)


def _speed_magnitudes_for_horizon(horizon: int) -> List[float]:
    max_speed = float(max(abs(float(VALID_SPEED[0])), abs(float(VALID_SPEED[1]))))
    if int(max(1, horizon)) >= 10:
        return [max_speed]
    return [1.0, max_speed]


def _simple_control_family_specs(steer_levels: int, horizon: int) -> List[PrimitiveFamilySpec]:
    max_speed = float(max(abs(float(VALID_SPEED[0])), abs(float(VALID_SPEED[1]))))
    speed_magnitudes = _speed_magnitudes_for_horizon(horizon)
    gamma_levels = _normalized_steer_levels(steer_levels)

    specs: List[PrimitiveFamilySpec] = []
    family_id = 0
    for speed_sign, direction_name in ((-1, "reverse"), (1, "forward")):
        for speed_value in speed_magnitudes:
            speed_tag = "full" if abs(float(speed_value) - max_speed) <= 1e-9 else "slow"
            speed_scale = float(abs(float(speed_value)) / max(max_speed, 1e-6))
            for gamma_rate_scale in gamma_levels.tolist():
                turn_name = f"steer{float(gamma_rate_scale):+.2f}".replace("+", "p").replace("-", "n")
                name = f"{direction_name}-{speed_tag}-{turn_name}"
                specs.append(
                    PrimitiveFamilySpec(
                        family_id=family_id,
                        name=name,
                        family_type="normal",
                        speed_sign=int(speed_sign),
                        speed_scale=float(speed_scale),
                        gamma_rate_scale=float(gamma_rate_scale),
                        mode="normal",
                        motion_family_id=family_id,
                        motion_family_name=name,
                        speed_level_id=0,
                        speed_level_name="full",
                        speed_level_scale=1.0,
                    )
                )
                family_id += 1
    return specs


def build_family_catalog(preset: str = "main", steer_levels: int = 11, horizon: int = 4) -> List[PrimitiveFamilySpec]:
    del preset
    return _simple_control_family_specs(steer_levels=int(steer_levels), horizon=int(horizon))