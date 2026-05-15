from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from configs import HITCH_OFFSET, NUM_STEP, STEP_LENGTH, TRAILER_LENGTH, VALID_SPEED, VALID_STEER
from env.vehicle import State, Vehicle
from primitives.family_catalog import build_family_catalog
from primitives.primitive_def import PrimitiveFamilySpec


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _state_to_array(state) -> np.ndarray:
    return np.asarray(
        [
            float(state.loc.x),
            float(state.loc.y),
            float(state.heading),
            float(state.rear_heading),
            float(getattr(state, "speed", 0.0)),
            float(getattr(state, "steering", 0.0)),
        ],
        dtype=np.float64,
    )


def build_gamma_bins(num_bins: int, gamma_max: float) -> np.ndarray:
    return np.linspace(-float(gamma_max), float(gamma_max), int(max(3, num_bins)), dtype=np.float64)


def _variant_control_scale(variant_id: int, variant_count: int) -> float:
    scales = np.linspace(0.90, 1.10, int(max(1, variant_count)), dtype=np.float64)
    return float(scales[int(variant_id)])


def _variant_duration(variant_id: int, variant_count: int, step_seconds: float, horizon: int) -> float:
    del variant_id, variant_count
    return float(float(step_seconds) * float(horizon))


def _initial_state_for_gamma(gamma0: float) -> State:
    return State([0.0, 0.0, 0.0, 0.0, 0.0, -float(gamma0)])


def _build_action_sequence(
    spec: PrimitiveFamilySpec,
    gamma0: float,
    variant_id: int,
    variant_count: int,
    horizon: int,
) -> Tuple[np.ndarray, int]:
    max_speed = float(max(abs(float(VALID_SPEED[0])), abs(float(VALID_SPEED[1]))))
    max_rate = float(max(abs(float(VALID_STEER[0])), abs(float(VALID_STEER[1]))))
    control_scale = _variant_control_scale(variant_id=variant_id, variant_count=variant_count)
    actions = np.zeros((int(horizon), 2), dtype=np.float64)

    if spec.family_type == "straighten":
        k_gamma = 1.1 * max_rate
        bias = 0.20 * float(spec.gamma_rate_scale) * max_rate
        for step_idx in range(int(horizon)):
            decay = 1.0 - 0.25 * float(step_idx) / float(max(1, horizon - 1))
            omega = np.clip(((-k_gamma * float(gamma0) + bias) * decay) * control_scale, VALID_STEER[0], VALID_STEER[1])
            speed = float(spec.speed_sign) * float(spec.speed_scale) * float(spec.speed_level_scale) * max_speed * 0.45
            actions[step_idx] = np.asarray([omega, speed], dtype=np.float64)
        return actions, -1

    if spec.family_type == "terminal":
        for step_idx in range(int(horizon)):
            taper = 1.0 - 0.35 * float(step_idx) / float(max(1, horizon - 1))
            omega = np.clip(float(spec.gamma_rate_scale) * max_rate * 0.45 * taper * control_scale, VALID_STEER[0], VALID_STEER[1])
            speed = float(spec.speed_sign) * float(spec.speed_scale) * float(spec.speed_level_scale) * max_speed * 0.35
            actions[step_idx] = np.asarray([omega, speed], dtype=np.float64)
        return actions, -1

    if spec.family_type == "compound":
        split_ratio = float(spec.compound_split if spec.compound_split is not None else 0.5)
        switch_index = int(max(1, min(horizon - 1, round(split_ratio * horizon))))
        for step_idx in range(int(horizon)):
            if step_idx < switch_index:
                omega = np.clip(float(spec.gamma_rate_scale) * max_rate * 0.80 * control_scale, VALID_STEER[0], VALID_STEER[1])
                speed = float(spec.speed_sign) * float(spec.speed_scale) * float(spec.speed_level_scale) * max_speed * 0.55
            else:
                blend = 1.0 - float(step_idx - switch_index) / float(max(1, horizon - switch_index))
                omega = np.clip(float(spec.compound_exit_gamma_scale) * max_rate * 0.75 * max(0.35, blend) * control_scale, VALID_STEER[0], VALID_STEER[1])
                speed = -float(spec.speed_sign) * float(spec.speed_scale) * float(spec.speed_level_scale) * max_speed * 0.45
            actions[step_idx] = np.asarray([omega, speed], dtype=np.float64)
        return actions, int(switch_index)

    for step_idx in range(int(horizon)):
        taper = 1.0 - 0.10 * float(step_idx) / float(max(1, horizon - 1))
        omega = np.clip(float(spec.gamma_rate_scale) * max_rate * taper * control_scale, VALID_STEER[0], VALID_STEER[1])
        speed = float(spec.speed_sign) * float(spec.speed_scale) * float(spec.speed_level_scale) * max_speed
        actions[step_idx] = np.asarray([omega, speed], dtype=np.float64)
    return actions, -1


def rollout_variant(
    spec: PrimitiveFamilySpec,
    gamma0: float,
    variant_id: int,
    variant_count: int,
    horizon: int,
) -> Dict:
    vehicle = Vehicle(
        articulated=True,
        trailer_length=TRAILER_LENGTH,
        hitch_offset=HITCH_OFFSET,
    )
    vehicle.reset(_initial_state_for_gamma(gamma0))
    actions, switch_index = _build_action_sequence(spec, gamma0, variant_id, variant_count, horizon)

    rollout_states: List[np.ndarray] = [_state_to_array(vehicle.state)]
    for action in actions:
        vehicle.step(np.asarray(action, dtype=np.float64), step_time=1)
        rollout_states.append(_state_to_array(vehicle.state))

    final_state = vehicle.state
    delta = np.asarray(
        [
            float(final_state.loc.x),
            float(final_state.loc.y),
            float(_wrap_pi(final_state.heading)),
            float(_wrap_pi(final_state.heading - final_state.rear_heading)),
        ],
        dtype=np.float64,
    )
    return {
        "actions": np.asarray(actions, dtype=np.float64),
        "rollout_states": np.asarray(rollout_states, dtype=np.float64),
        "delta": delta,
        "duration": _variant_duration(variant_id, variant_count, step_seconds=STEP_LENGTH * NUM_STEP, horizon=horizon),
        "speed_sign": int(np.sign(np.mean(actions[:, 1]))) if np.any(np.abs(actions[:, 1]) > 1e-8) else 0,
        "is_compound": int(spec.family_type == "compound"),
        "switch_index": int(switch_index),
        "mode": str(spec.mode),
        "family_type": str(spec.family_type),
    }


def generate_primitives(
    H: int,
    S: int,
    output_path: str,
    gamma_bins: int = 31,
    variant_count: int = 3,
    family_preset: str = "main",
):
    del S  # retained for CLI/backward compatibility in experiment scripts

    horizon = int(H)
    variant_count = int(max(1, variant_count))
    gamma_max = float(max(abs(float(VALID_STEER[0])), abs(float(VALID_STEER[1]))))
    gamma_bin_values = build_gamma_bins(num_bins=int(gamma_bins), gamma_max=gamma_max)
    family_specs = build_family_catalog(preset=family_preset)
    family_count = int(len(family_specs))

    flat_actions: List[np.ndarray] = []
    flat_rollout_states: List[np.ndarray] = []
    flat_deltas: List[np.ndarray] = []
    flat_variant_horizons: List[int] = []
    flat_switch_indices: List[int] = []
    flat_durations: List[float] = []
    flat_speed_signs: List[int] = []
    flat_is_compound: List[int] = []
    flat_to_gamma: List[int] = []
    flat_to_family: List[int] = []
    flat_to_motion_family: List[int] = []
    flat_to_speed_level: List[int] = []
    flat_to_variant: List[int] = []
    flat_to_family_type: List[str] = []
    flat_to_mode: List[str] = []

    index_table = np.full((len(gamma_bin_values), family_count, variant_count), -1, dtype=np.int64)
    variant_counts = np.zeros((len(gamma_bin_values), family_count), dtype=np.int64)
    default_variant_table = np.full((len(gamma_bin_values), family_count), -1, dtype=np.int64)

    for gamma_bin_id, gamma0 in enumerate(gamma_bin_values.tolist()):
        for spec in family_specs:
            for variant_id in range(variant_count):
                payload = rollout_variant(spec, float(gamma0), int(variant_id), variant_count, horizon)
                flat_index = len(flat_actions)
                index_table[int(gamma_bin_id), int(spec.family_id), int(variant_id)] = int(flat_index)
                variant_counts[int(gamma_bin_id), int(spec.family_id)] += 1
                if int(variant_id) == int(variant_count // 2):
                    default_variant_table[int(gamma_bin_id), int(spec.family_id)] = int(flat_index)

                flat_actions.append(np.asarray(payload["actions"], dtype=np.float64))
                flat_rollout_states.append(np.asarray(payload["rollout_states"], dtype=np.float64))
                flat_deltas.append(np.asarray(payload["delta"], dtype=np.float64))
                flat_variant_horizons.append(int(horizon))
                flat_switch_indices.append(int(payload["switch_index"]))
                flat_durations.append(float(payload["duration"]))
                flat_speed_signs.append(int(payload["speed_sign"]))
                flat_is_compound.append(int(payload["is_compound"]))
                flat_to_gamma.append(int(gamma_bin_id))
                flat_to_family.append(int(spec.family_id))
                flat_to_motion_family.append(int(spec.motion_family_id if spec.motion_family_id >= 0 else spec.family_id))
                flat_to_speed_level.append(int(spec.speed_level_id))
                flat_to_variant.append(int(variant_id))
                flat_to_family_type.append(str(payload["family_type"]))
                flat_to_mode.append(str(payload["mode"]))

    motion_family_names = []
    for motion_family_id in sorted({int(spec.motion_family_id if spec.motion_family_id >= 0 else spec.family_id) for spec in family_specs}):
        match = next(spec for spec in family_specs if int(spec.motion_family_id if spec.motion_family_id >= 0 else spec.family_id) == int(motion_family_id))
        motion_family_names.append(str(match.motion_family_name or match.name))
    speed_level_specs = {}
    for spec in family_specs:
        speed_level_specs[int(spec.speed_level_id)] = (str(spec.speed_level_name), float(spec.speed_level_scale))
    speed_level_ids_sorted = sorted(speed_level_specs.keys())
    speed_level_names = [speed_level_specs[idx][0] for idx in speed_level_ids_sorted]
    speed_level_scales = [speed_level_specs[idx][1] for idx in speed_level_ids_sorted]
    family_to_motion_family = np.asarray(
        [int(spec.motion_family_id if spec.motion_family_id >= 0 else spec.family_id) for spec in family_specs],
        dtype=np.int64,
    )
    family_to_speed_level = np.asarray([int(spec.speed_level_id) for spec in family_specs], dtype=np.int64)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray("family_library_v1", dtype=object),
        actions=np.asarray(flat_actions, dtype=np.float64),
        deltas=np.asarray(flat_deltas, dtype=np.float64),
        rollout_states=np.asarray(flat_rollout_states, dtype=np.float64),
        variant_horizons=np.asarray(flat_variant_horizons, dtype=np.int64),
        switch_indices=np.asarray(flat_switch_indices, dtype=np.int64),
        durations=np.asarray(flat_durations, dtype=np.float64),
        speed_signs=np.asarray(flat_speed_signs, dtype=np.int64),
        is_compound=np.asarray(flat_is_compound, dtype=np.int8),
        variant_flat_to_gamma=np.asarray(flat_to_gamma, dtype=np.int64),
        variant_flat_to_family=np.asarray(flat_to_family, dtype=np.int64),
        variant_flat_to_motion_family=np.asarray(flat_to_motion_family, dtype=np.int64),
        variant_flat_to_speed_level=np.asarray(flat_to_speed_level, dtype=np.int64),
        variant_flat_to_variant=np.asarray(flat_to_variant, dtype=np.int64),
        variant_flat_to_family_type=np.asarray(flat_to_family_type, dtype=object),
        variant_flat_to_mode=np.asarray(flat_to_mode, dtype=object),
        gamma_bin_values=np.asarray(gamma_bin_values, dtype=np.float64),
        family_names=np.asarray([spec.name for spec in family_specs], dtype=object),
        family_types=np.asarray([spec.family_type for spec in family_specs], dtype=object),
        family_to_motion_family=np.asarray(family_to_motion_family, dtype=np.int64),
        family_to_speed_level=np.asarray(family_to_speed_level, dtype=np.int64),
        motion_family_names=np.asarray(motion_family_names, dtype=object),
        motion_family_count=np.asarray(len(motion_family_names), dtype=np.int64),
        speed_level_names=np.asarray(speed_level_names, dtype=object),
        speed_level_scales=np.asarray(speed_level_scales, dtype=np.float64),
        speed_level_count=np.asarray(len(speed_level_names), dtype=np.int64),
        family_count=np.asarray(int(family_count), dtype=np.int64),
        variant_count_per_family=np.asarray(int(variant_count), dtype=np.int64),
        index_table=np.asarray(index_table, dtype=np.int64),
        variant_counts=np.asarray(variant_counts, dtype=np.int64),
        default_variant_table=np.asarray(default_variant_table, dtype=np.int64),
        step_seconds=np.asarray(float(STEP_LENGTH * NUM_STEP), dtype=np.float64),
        meta=np.asarray(
            {
                "H": int(horizon),
                "gamma_bins": int(gamma_bins),
                "variant_count": int(variant_count),
                "family_preset": str(family_preset),
                "trailer_length": float(TRAILER_LENGTH),
                "hitch_offset": float(HITCH_OFFSET),
                "step_seconds": float(STEP_LENGTH * NUM_STEP),
                "motion_family_count": int(len(motion_family_names)),
                "speed_level_names": list(speed_level_names),
                "speed_level_scales": [float(v) for v in speed_level_scales],
                "family_specs": [spec.__dict__.copy() for spec in family_specs],
            },
            dtype=object,
        ),
    )
    print(
        f"Saved family primitive library to {output_path} | families={family_count} | gamma_bins={gamma_bins} | variants={len(flat_actions)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=3)
    parser.add_argument("--S", type=int, default=11)
    parser.add_argument("--gamma-bins", type=int, default=31)
    parser.add_argument("--variant-count", type=int, default=3)
    parser.add_argument("--family-preset", type=str, default="main", choices=("small", "main", "large"))
    args = parser.parse_args()

    root_data_path = os.path.join(os.path.dirname(__file__), "../../data")
    output_path = os.path.join(root_data_path, f"primitives_family_{args.family_preset}_G{args.gamma_bins}_V{args.variant_count}.npz")
    generate_primitives(
        H=args.H,
        S=args.S,
        output_path=output_path,
        gamma_bins=args.gamma_bins,
        variant_count=args.variant_count,
        family_preset=args.family_preset,
    )