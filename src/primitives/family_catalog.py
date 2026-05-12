from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from primitives.primitive_def import PrimitiveFamilySpec


def _normal_family_specs(gamma_levels: Sequence[float]) -> List[PrimitiveFamilySpec]:
    specs: List[PrimitiveFamilySpec] = []
    family_id = 0
    for speed_name, speed_scale in (("slow", 0.55), ("mid", 1.00)):
        for speed_sign, direction_name in ((1, "forward"), (-1, "reverse")):
            for gamma_rate_scale in gamma_levels:
                turn_name = f"gamma{gamma_rate_scale:+.2f}".replace("+", "p").replace("-", "n")
                specs.append(
                    PrimitiveFamilySpec(
                        family_id=family_id,
                        name=f"{direction_name}-{speed_name}-{turn_name}",
                        family_type="normal",
                        speed_sign=int(speed_sign),
                        speed_scale=float(speed_scale),
                        gamma_rate_scale=float(gamma_rate_scale),
                        mode="normal",
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

    specs = _normal_family_specs(gamma_levels)
    next_family_id = len(specs)

    for speed_sign, direction_name in ((1, "forward"), (-1, "reverse")):
        for gamma_rate_scale, side_name in ((0.30, "left"), (-0.30, "right")):
            specs.append(
                PrimitiveFamilySpec(
                    family_id=next_family_id,
                    name=f"{direction_name}-straighten-{side_name}",
                    family_type="straighten",
                    speed_sign=int(speed_sign),
                    speed_scale=0.50,
                    gamma_rate_scale=float(gamma_rate_scale),
                    mode="straighten",
                )
            )
            next_family_id += 1

    for name, gamma_rate_scale in terminal_specs:
        speed_sign = -1 if "reverse" in name else 1
        specs.append(
            PrimitiveFamilySpec(
                family_id=next_family_id,
                name=name,
                family_type="terminal",
                speed_sign=int(speed_sign),
                speed_scale=0.35,
                gamma_rate_scale=float(gamma_rate_scale),
                mode="terminal",
            )
        )
        next_family_id += 1

    for name, gamma_rate_scale in compound_specs:
        speed_sign = -1 if name.startswith("reverse") else 1
        specs.append(
            PrimitiveFamilySpec(
                family_id=next_family_id,
                name=name,
                family_type="compound",
                speed_sign=int(speed_sign),
                speed_scale=0.50,
                gamma_rate_scale=float(gamma_rate_scale),
                mode="compound",
                compound_split=0.50,
                compound_exit_gamma_scale=-float(gamma_rate_scale),
            )
        )
        next_family_id += 1

    return specs