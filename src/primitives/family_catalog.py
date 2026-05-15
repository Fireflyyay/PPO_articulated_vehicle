from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from configs import NUM_STEP, STEP_LENGTH
from primitives.primitive_def import PrimitiveFamilySpec


def _speed_level_specs() -> List[dict]:
    macro_step_seconds = float(max(1, int(NUM_STEP))) * float(STEP_LENGTH)
    internal_step_seconds = float(STEP_LENGTH)
    mid_scale = internal_step_seconds / max(macro_step_seconds, 1e-6)
    return [
        {"speed_level_id": 0, "speed_level_name": "stop", "speed_level_scale": 0.0},
        {"speed_level_id": 1, "speed_level_name": "mid", "speed_level_scale": float(mid_scale)},
        {"speed_level_id": 2, "speed_level_name": "full", "speed_level_scale": 1.0},
    ]


def _normal_family_specs(gamma_levels: Sequence[float]) -> List[PrimitiveFamilySpec]:
    specs: List[PrimitiveFamilySpec] = []
    motion_family_id = 0
    for speed_sign, direction_name in ((1, "forward"), (-1, "reverse")):
        for gamma_rate_scale in gamma_levels:
            turn_name = f"gamma{gamma_rate_scale:+.2f}".replace("+", "p").replace("-", "n")
            motion_name = f"{direction_name}-{turn_name}"
            specs.append(
                PrimitiveFamilySpec(
                    family_id=motion_family_id,
                    name=motion_name,
                    family_type="normal",
                    speed_sign=int(speed_sign),
                    speed_scale=1.0,
                    gamma_rate_scale=float(gamma_rate_scale),
                    mode="normal",
                    motion_family_id=motion_family_id,
                    motion_family_name=motion_name,
                )
            )
            motion_family_id += 1
    return specs


def _expand_speed_levels(motion_specs: Sequence[PrimitiveFamilySpec]) -> List[PrimitiveFamilySpec]:
    specs: List[PrimitiveFamilySpec] = []
    family_id = 0
    for motion_spec in motion_specs:
        motion_name = str(motion_spec.motion_family_name or motion_spec.name)
        motion_family_id = int(motion_spec.motion_family_id if motion_spec.motion_family_id >= 0 else motion_spec.family_id)
        for speed_level in _speed_level_specs():
            speed_level_name = str(speed_level["speed_level_name"])
            specs.append(
                PrimitiveFamilySpec(
                    family_id=family_id,
                    name=f"{speed_level_name}-{motion_name}",
                    family_type=str(motion_spec.family_type),
                    speed_sign=int(motion_spec.speed_sign),
                    speed_scale=float(motion_spec.speed_scale),
                    gamma_rate_scale=float(motion_spec.gamma_rate_scale),
                    mode=str(motion_spec.mode),
                    compound_split=motion_spec.compound_split,
                    compound_exit_gamma_scale=float(motion_spec.compound_exit_gamma_scale),
                    motion_family_id=motion_family_id,
                    motion_family_name=motion_name,
                    speed_level_id=int(speed_level["speed_level_id"]),
                    speed_level_name=speed_level_name,
                    speed_level_scale=float(speed_level["speed_level_scale"]),
                )
            )
            family_id += 1
    return specs


def build_family_catalog(preset: str = "main") -> List[PrimitiveFamilySpec]:
    preset_name = str(preset).strip().lower()
    if preset_name == "small":
        gamma_levels = (-1.0, -0.6, -0.25, 0.0, 0.25, 0.6)
        compound_specs = [
            ("reverse-then-forward-left-align", 0.45),
            ("reverse-then-forward-right-align", -0.45),
            ("forward-then-reverse-straighten", 0.0),
        ]
        terminal_specs = [
            ("terminal-align-left", 0.35),
            ("terminal-align-right", -0.35),
        ]
    elif preset_name == "large":
        gamma_levels = (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
        compound_specs = [
            ("reverse-then-forward-left-align", 0.45),
            ("reverse-then-forward-right-align", -0.45),
            ("reverse-then-forward-straighten", 0.0),
            ("forward-then-reverse-left-correct", 0.35),
            ("forward-then-reverse-right-correct", -0.35),
            ("forward-then-reverse-straighten", 0.0),
            ("reverse-then-forward-left-wide", 0.65),
            ("reverse-then-forward-right-wide", -0.65),
        ]
        terminal_specs = [
            ("terminal-align-left", 0.35),
            ("terminal-align-right", -0.35),
            ("terminal-align-straight", 0.0),
        ]
    else:
        gamma_levels = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
        compound_specs = [
            ("reverse-then-forward-left-align", 0.45),
            ("reverse-then-forward-right-align", -0.45),
            ("forward-then-reverse-straighten", 0.0),
        ]
        terminal_specs = [
            ("terminal-align-left", 0.35),
            ("terminal-align-right", -0.35),
        ]

    motion_specs = _normal_family_specs(gamma_levels)
    next_motion_family_id = len(motion_specs)

    for speed_sign, direction_name in ((1, "forward"), (-1, "reverse")):
        for gamma_rate_scale, side_name in ((0.30, "left"), (-0.30, "right")):
            motion_name = f"{direction_name}-straighten-{side_name}"
            motion_specs.append(
                PrimitiveFamilySpec(
                    family_id=next_motion_family_id,
                    name=motion_name,
                    family_type="straighten",
                    speed_sign=int(speed_sign),
                    speed_scale=0.50,
                    gamma_rate_scale=float(gamma_rate_scale),
                    mode="straighten",
                    motion_family_id=next_motion_family_id,
                    motion_family_name=motion_name,
                )
            )
            next_motion_family_id += 1

    for name, gamma_rate_scale in terminal_specs:
        speed_sign = -1 if "reverse" in name else 1
        motion_specs.append(
            PrimitiveFamilySpec(
                family_id=next_motion_family_id,
                name=name,
                family_type="terminal",
                speed_sign=int(speed_sign),
                speed_scale=0.35,
                gamma_rate_scale=float(gamma_rate_scale),
                mode="terminal",
                motion_family_id=next_motion_family_id,
                motion_family_name=name,
            )
        )
        next_motion_family_id += 1

    for name, gamma_rate_scale in compound_specs:
        speed_sign = -1 if name.startswith("reverse") else 1
        motion_specs.append(
            PrimitiveFamilySpec(
                family_id=next_motion_family_id,
                name=name,
                family_type="compound",
                speed_sign=int(speed_sign),
                speed_scale=0.50,
                gamma_rate_scale=float(gamma_rate_scale),
                mode="compound",
                compound_split=0.50,
                compound_exit_gamma_scale=-float(gamma_rate_scale),
                motion_family_id=next_motion_family_id,
                motion_family_name=name,
            )
        )
        next_motion_family_id += 1

    return _expand_speed_levels(motion_specs)