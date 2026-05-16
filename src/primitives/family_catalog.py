from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from configs import NUM_STEP, STEP_LENGTH
from primitives.primitive_def import PrimitiveFamilySpec


def _speed_level_specs() -> List[dict]:
    return [
        {"speed_level_id": 0, "speed_level_name": "semantic", "speed_level_scale": 1.0},
    ]


@dataclass(frozen=True)
class PrimitiveModeProfile:
    mode_id: int
    mode_name: str
    base_speed_scale: float
    base_horizon_scale: float
    base_turn_scale: float
    group_speed_scales: Dict[str, float]
    group_horizon_scales: Dict[str, float]
    group_turn_scales: Dict[str, float]
    recover_gain_scale: float = 1.0


def _mode_profiles() -> List[PrimitiveModeProfile]:
    return [
        PrimitiveModeProfile(
            mode_id=0,
            mode_name="normal",
            base_speed_scale=1.0,
            base_horizon_scale=1.0,
            base_turn_scale=1.0,
            group_speed_scales={"basic": 1.0, "articulation": 0.72, "escape": 0.78, "terminal": 0.55},
            group_horizon_scales={"basic": 1.0, "articulation": 0.82, "escape": 0.74, "terminal": 0.52},
            group_turn_scales={"basic": 1.0, "articulation": 0.92, "escape": 1.02, "terminal": 0.62},
            recover_gain_scale=1.0,
        ),
        PrimitiveModeProfile(
            mode_id=1,
            mode_name="narrow_escape",
            base_speed_scale=0.78,
            base_horizon_scale=0.70,
            base_turn_scale=0.96,
            group_speed_scales={"basic": 0.78, "articulation": 0.70, "escape": 0.82, "terminal": 0.50},
            group_horizon_scales={"basic": 0.70, "articulation": 0.66, "escape": 0.78, "terminal": 0.46},
            group_turn_scales={"basic": 0.92, "articulation": 1.08, "escape": 1.10, "terminal": 0.72},
            recover_gain_scale=1.18,
        ),
        PrimitiveModeProfile(
            mode_id=2,
            mode_name="terminal",
            base_speed_scale=0.52,
            base_horizon_scale=0.46,
            base_turn_scale=0.64,
            group_speed_scales={"basic": 0.58, "articulation": 0.52, "escape": 0.56, "terminal": 0.42},
            group_horizon_scales={"basic": 0.48, "articulation": 0.44, "escape": 0.50, "terminal": 0.38},
            group_turn_scales={"basic": 0.70, "articulation": 0.78, "escape": 0.74, "terminal": 0.54},
            recover_gain_scale=0.92,
        ),
    ]


def _slot(
    family_id: int,
    name: str,
    semantic_group: str,
    family_type: str,
    speed_sign: int,
    gamma_rate_scale: float,
    *,
    speed_scale: float = 1.0,
    pattern: str = "single",
    horizon_scale: float = 1.0,
    secondary_speed_sign: int = 0,
    secondary_gamma_rate_scale: float = 0.0,
    compound_split: float | None = None,
    recover_gain: float = 0.0,
    recover_bias_scale: float = 0.0,
    articulation_target: float = 0.0,
) -> Dict:
    return {
        "family_id": int(family_id),
        "name": str(name),
        "semantic_group": str(semantic_group),
        "family_type": str(family_type),
        "speed_sign": int(speed_sign),
        "gamma_rate_scale": float(gamma_rate_scale),
        "speed_scale": float(speed_scale),
        "pattern": str(pattern),
        "horizon_scale": float(horizon_scale),
        "secondary_speed_sign": int(secondary_speed_sign),
        "secondary_gamma_rate_scale": float(secondary_gamma_rate_scale),
        "compound_split": compound_split,
        "recover_gain": float(recover_gain),
        "recover_bias_scale": float(recover_bias_scale),
        "articulation_target": float(articulation_target),
    }


def _base_semantic_slots() -> List[Dict]:
    slots: List[Dict] = [
        _slot(0, "forward-straight", "basic", "normal", 1, 0.0, speed_scale=1.0, horizon_scale=1.0),
        _slot(1, "forward-left-small", "basic", "normal", 1, 0.22, speed_scale=0.98, horizon_scale=0.95),
        _slot(2, "forward-right-small", "basic", "normal", 1, -0.22, speed_scale=0.98, horizon_scale=0.95),
        _slot(3, "forward-left-medium", "basic", "normal", 1, 0.45, speed_scale=0.94, horizon_scale=0.90),
        _slot(4, "forward-right-medium", "basic", "normal", 1, -0.45, speed_scale=0.94, horizon_scale=0.90),
        _slot(5, "forward-left-large", "basic", "normal", 1, 0.78, speed_scale=0.84, horizon_scale=0.82),
        _slot(6, "forward-right-large", "basic", "normal", 1, -0.78, speed_scale=0.84, horizon_scale=0.82),
        _slot(7, "reverse-straight", "basic", "normal", -1, 0.0, speed_scale=0.92, horizon_scale=0.92),
        _slot(8, "reverse-left-small", "basic", "normal", -1, 0.22, speed_scale=0.88, horizon_scale=0.88),
        _slot(9, "reverse-right-small", "basic", "normal", -1, -0.22, speed_scale=0.88, horizon_scale=0.88),
        _slot(10, "reverse-left-medium", "basic", "normal", -1, 0.45, speed_scale=0.80, horizon_scale=0.82),
        _slot(11, "reverse-right-medium", "basic", "normal", -1, -0.45, speed_scale=0.80, horizon_scale=0.82),
        _slot(12, "reverse-left-large", "basic", "normal", -1, 0.74, speed_scale=0.72, horizon_scale=0.76),
        _slot(13, "reverse-right-large", "basic", "normal", -1, -0.74, speed_scale=0.72, horizon_scale=0.76),
        _slot(14, "articulation-recover-forward", "articulation", "straighten", 1, 0.0, speed_scale=0.70, pattern="recover", horizon_scale=0.78, recover_gain=1.35),
        _slot(15, "articulation-recover-reverse", "articulation", "straighten", -1, 0.0, speed_scale=0.66, pattern="recover", horizon_scale=0.74, recover_gain=1.30),
        _slot(16, "articulation-hold-forward", "articulation", "straighten", 1, 0.0, speed_scale=0.62, pattern="hold", horizon_scale=0.72, recover_gain=0.45),
        _slot(17, "articulation-hold-reverse", "articulation", "straighten", -1, 0.0, speed_scale=0.58, pattern="hold", horizon_scale=0.70, recover_gain=0.45),
        _slot(18, "reduce-positive-phi-forward", "articulation", "straighten", 1, -0.28, speed_scale=0.68, pattern="recover", horizon_scale=0.74, recover_gain=1.20, recover_bias_scale=-0.28),
        _slot(19, "reduce-negative-phi-forward", "articulation", "straighten", 1, 0.28, speed_scale=0.68, pattern="recover", horizon_scale=0.74, recover_gain=1.20, recover_bias_scale=0.28),
        _slot(20, "reduce-positive-phi-reverse", "articulation", "straighten", -1, -0.28, speed_scale=0.64, pattern="recover", horizon_scale=0.72, recover_gain=1.15, recover_bias_scale=-0.28),
        _slot(21, "reduce-negative-phi-reverse", "articulation", "straighten", -1, 0.28, speed_scale=0.64, pattern="recover", horizon_scale=0.72, recover_gain=1.15, recover_bias_scale=0.28),
        _slot(22, "jackknife-buffer-left-forward", "articulation", "straighten", 1, 0.20, speed_scale=0.58, pattern="recover", horizon_scale=0.70, recover_gain=0.95, articulation_target=-0.15),
        _slot(23, "jackknife-buffer-right-forward", "articulation", "straighten", 1, -0.20, speed_scale=0.58, pattern="recover", horizon_scale=0.70, recover_gain=0.95, articulation_target=0.15),
        _slot(24, "jackknife-buffer-left-reverse", "articulation", "straighten", -1, 0.20, speed_scale=0.56, pattern="recover", horizon_scale=0.68, recover_gain=0.90, articulation_target=-0.15),
        _slot(25, "jackknife-buffer-right-reverse", "articulation", "straighten", -1, -0.20, speed_scale=0.56, pattern="recover", horizon_scale=0.68, recover_gain=0.90, articulation_target=0.15),
        _slot(26, "escape-forward-left-then-reverse-right", "escape", "compound", 1, 0.86, speed_scale=0.72, pattern="compound", horizon_scale=0.84, secondary_speed_sign=-1, secondary_gamma_rate_scale=-0.82, compound_split=0.50),
        _slot(27, "escape-forward-right-then-reverse-left", "escape", "compound", 1, -0.86, speed_scale=0.72, pattern="compound", horizon_scale=0.84, secondary_speed_sign=-1, secondary_gamma_rate_scale=0.82, compound_split=0.50),
        _slot(28, "escape-reverse-left-then-forward-right", "escape", "compound", -1, 0.84, speed_scale=0.68, pattern="compound", horizon_scale=0.84, secondary_speed_sign=1, secondary_gamma_rate_scale=-0.80, compound_split=0.48),
        _slot(29, "escape-reverse-right-then-forward-left", "escape", "compound", -1, -0.84, speed_scale=0.68, pattern="compound", horizon_scale=0.84, secondary_speed_sign=1, secondary_gamma_rate_scale=0.80, compound_split=0.48),
        _slot(30, "escape-forward-left-tight", "escape", "compound", 1, 0.96, speed_scale=0.62, pattern="single", horizon_scale=0.66),
        _slot(31, "escape-forward-right-tight", "escape", "compound", 1, -0.96, speed_scale=0.62, pattern="single", horizon_scale=0.66),
        _slot(32, "escape-reverse-left-tight", "escape", "compound", -1, 0.96, speed_scale=0.58, pattern="single", horizon_scale=0.64),
        _slot(33, "escape-reverse-right-tight", "escape", "compound", -1, -0.96, speed_scale=0.58, pattern="single", horizon_scale=0.64),
        _slot(34, "escape-forward-left-jab", "escape", "compound", 1, 0.72, speed_scale=0.56, pattern="single", horizon_scale=0.48),
        _slot(35, "escape-forward-right-jab", "escape", "compound", 1, -0.72, speed_scale=0.56, pattern="single", horizon_scale=0.48),
        _slot(36, "escape-reverse-left-jab", "escape", "compound", -1, 0.72, speed_scale=0.54, pattern="single", horizon_scale=0.46),
        _slot(37, "escape-reverse-right-jab", "escape", "compound", -1, -0.72, speed_scale=0.54, pattern="single", horizon_scale=0.46),
        _slot(38, "escape-forward-wiggle", "escape", "compound", 1, 0.52, speed_scale=0.54, pattern="wiggle", horizon_scale=0.68, secondary_speed_sign=1, secondary_gamma_rate_scale=-0.52, compound_split=0.50),
        _slot(39, "escape-reverse-wiggle", "escape", "compound", -1, 0.52, speed_scale=0.52, pattern="wiggle", horizon_scale=0.66, secondary_speed_sign=-1, secondary_gamma_rate_scale=-0.52, compound_split=0.50),
        _slot(40, "terminal-forward-straight-short", "terminal", "terminal", 1, 0.0, speed_scale=0.44, pattern="terminal", horizon_scale=0.42),
        _slot(41, "terminal-reverse-straight-short", "terminal", "terminal", -1, 0.0, speed_scale=0.38, pattern="terminal", horizon_scale=0.40),
        _slot(42, "terminal-forward-left-micro", "terminal", "terminal", 1, 0.18, speed_scale=0.40, pattern="terminal", horizon_scale=0.40),
        _slot(43, "terminal-forward-right-micro", "terminal", "terminal", 1, -0.18, speed_scale=0.40, pattern="terminal", horizon_scale=0.40),
        _slot(44, "terminal-reverse-left-micro", "terminal", "terminal", -1, 0.18, speed_scale=0.36, pattern="terminal", horizon_scale=0.38),
        _slot(45, "terminal-reverse-right-micro", "terminal", "terminal", -1, -0.18, speed_scale=0.36, pattern="terminal", horizon_scale=0.38),
        _slot(46, "terminal-heading-align-left", "terminal", "terminal", 1, 0.30, speed_scale=0.34, pattern="terminal", horizon_scale=0.44),
        _slot(47, "terminal-heading-align-right", "terminal", "terminal", 1, -0.30, speed_scale=0.34, pattern="terminal", horizon_scale=0.44),
    ]
    return slots


def _slot_filter_for_preset(preset_name: str, slots: Sequence[Dict]) -> List[Dict]:
    if preset_name == "small":
        allowed = list(range(0, 14)) + list(range(14, 22)) + [26, 27, 28, 29, 38, 39, 40, 41]
        return [slot for slot in slots if int(slot["family_id"]) in set(allowed)]
    if preset_name == "large":
        return list(slots)
    return list(slots)


def _expand_speed_levels(motion_specs: Sequence[PrimitiveFamilySpec]) -> List[PrimitiveFamilySpec]:
    specs: List[PrimitiveFamilySpec] = []
    for motion_spec in motion_specs:
        motion_name = str(motion_spec.motion_family_name or motion_spec.name)
        motion_family_id = int(motion_spec.motion_family_id if motion_spec.motion_family_id >= 0 else motion_spec.family_id)
        for speed_level in _speed_level_specs():
            speed_level_name = str(speed_level["speed_level_name"])
            specs.append(
                PrimitiveFamilySpec(
                    family_id=int(motion_spec.family_id),
                    name=motion_name,
                    family_type=str(motion_spec.family_type),
                    speed_sign=int(motion_spec.speed_sign),
                    speed_scale=float(motion_spec.speed_scale),
                    gamma_rate_scale=float(motion_spec.gamma_rate_scale),
                    mode=str(motion_spec.mode),
                    mode_id=int(motion_spec.mode_id),
                    semantic_slot_id=int(motion_spec.semantic_slot_id),
                    semantic_slot_name=str(motion_spec.semantic_slot_name),
                    semantic_group=str(motion_spec.semantic_group),
                    pattern=str(motion_spec.pattern),
                    horizon_scale=float(motion_spec.horizon_scale),
                    secondary_speed_sign=int(motion_spec.secondary_speed_sign),
                    secondary_gamma_rate_scale=float(motion_spec.secondary_gamma_rate_scale),
                    recover_gain=float(motion_spec.recover_gain),
                    recover_bias_scale=float(motion_spec.recover_bias_scale),
                    articulation_target=float(motion_spec.articulation_target),
                    compound_split=motion_spec.compound_split,
                    compound_exit_gamma_scale=float(motion_spec.compound_exit_gamma_scale),
                    motion_family_id=motion_family_id,
                    motion_family_name=motion_name,
                    speed_level_id=int(speed_level["speed_level_id"]),
                    speed_level_name=speed_level_name,
                    speed_level_scale=float(speed_level["speed_level_scale"]),
                )
            )
    return specs


def build_family_catalog(preset: str = "main") -> List[PrimitiveFamilySpec]:
    preset_name = str(preset).strip().lower()
    base_slots = _slot_filter_for_preset(preset_name, _base_semantic_slots())
    family_id_map = {int(slot["family_id"]): idx for idx, slot in enumerate(base_slots)}
    mode_profiles = _mode_profiles()
    semantic_specs: List[PrimitiveFamilySpec] = []

    for mode_profile in mode_profiles:
        for slot in base_slots:
            semantic_slot_id = int(slot["family_id"])
            family_id = int(family_id_map[semantic_slot_id])
            group = str(slot["semantic_group"])
            speed_scale = (
                float(slot["speed_scale"])
                * float(mode_profile.base_speed_scale)
                * float(mode_profile.group_speed_scales.get(group, 1.0))
            )
            gamma_rate_scale = (
                float(slot["gamma_rate_scale"])
                * float(mode_profile.base_turn_scale)
                * float(mode_profile.group_turn_scales.get(group, 1.0))
            )
            secondary_gamma_rate_scale = (
                float(slot["secondary_gamma_rate_scale"])
                * float(mode_profile.base_turn_scale)
                * float(mode_profile.group_turn_scales.get(group, 1.0))
            )
            horizon_scale = (
                float(slot["horizon_scale"])
                * float(mode_profile.base_horizon_scale)
                * float(mode_profile.group_horizon_scales.get(group, 1.0))
            )
            semantic_specs.append(
                PrimitiveFamilySpec(
                    family_id=family_id,
                    name=str(slot["name"]),
                    family_type=str(slot["family_type"]),
                    speed_sign=int(slot["speed_sign"]),
                    speed_scale=float(speed_scale),
                    gamma_rate_scale=float(gamma_rate_scale),
                    mode=str(mode_profile.mode_name),
                    mode_id=int(mode_profile.mode_id),
                    semantic_slot_id=semantic_slot_id,
                    semantic_slot_name=str(slot["name"]),
                    semantic_group=group,
                    pattern=str(slot["pattern"]),
                    horizon_scale=float(horizon_scale),
                    secondary_speed_sign=int(slot["secondary_speed_sign"]),
                    secondary_gamma_rate_scale=float(secondary_gamma_rate_scale),
                    recover_gain=float(slot["recover_gain"]) * float(mode_profile.recover_gain_scale),
                    recover_bias_scale=float(slot["recover_bias_scale"]),
                    articulation_target=float(slot["articulation_target"]),
                    compound_split=slot["compound_split"],
                    compound_exit_gamma_scale=float(-secondary_gamma_rate_scale),
                    motion_family_id=family_id,
                    motion_family_name=str(slot["name"]),
                )
            )

    return _expand_speed_levels(semantic_specs)