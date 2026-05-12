from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from primitives.primitive_def import PrimitiveFamilyLibraryMeta, PrimitiveVariantRef


def _wrap_pi_array(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _load_meta_dict(data) -> Dict:
    if "meta" not in data:
        return {}
    try:
        return data["meta"].item() or {}
    except Exception:
        return {}


def _object_array_to_str_list(values) -> List[str]:
    arr = np.asarray(values, dtype=object).reshape(-1)
    return [str(v) for v in arr.tolist()]


class PrimitiveLibrary:
    def __init__(self, npz_path: str, load_sidecars: bool = True):
        self.npz_path = str(npz_path)
        data = np.load(npz_path, allow_pickle=True)

        schema_version = str(data["schema_version"].item()) if "schema_version" in data else ""
        if not schema_version.startswith("family_library_v"):
            raise ValueError(
                f"PrimitiveLibrary expects a family library schema, got schema_version={schema_version!r} from {npz_path}"
            )

        self.schema_version = schema_version
        self.meta = _load_meta_dict(data)

        self.actions = np.asarray(data["actions"], dtype=np.float64)
        self.deltas = np.asarray(data["deltas"], dtype=np.float64)
        self.rollout_states = np.asarray(data["rollout_states"], dtype=np.float64)
        self.variant_horizons = np.asarray(data["variant_horizons"], dtype=np.int64).reshape(-1)
        self.switch_indices = np.asarray(data["switch_indices"], dtype=np.int64).reshape(-1)
        self.durations = np.asarray(data["durations"], dtype=np.float64).reshape(-1)
        self.speed_signs = np.asarray(data["speed_signs"], dtype=np.int64).reshape(-1)
        self.is_compound = np.asarray(data["is_compound"], dtype=np.int8).reshape(-1)

        self.variant_flat_to_gamma = np.asarray(data["variant_flat_to_gamma"], dtype=np.int64).reshape(-1)
        self.variant_flat_to_family = np.asarray(data["variant_flat_to_family"], dtype=np.int64).reshape(-1)
        self.variant_flat_to_variant = np.asarray(data["variant_flat_to_variant"], dtype=np.int64).reshape(-1)
        self.variant_flat_to_family_type = _object_array_to_str_list(data["variant_flat_to_family_type"])
        self.variant_flat_to_mode = _object_array_to_str_list(data["variant_flat_to_mode"])

        self.gamma_bin_values = np.asarray(data["gamma_bin_values"], dtype=np.float64).reshape(-1)
        self.family_names = _object_array_to_str_list(data["family_names"])
        self.family_types = _object_array_to_str_list(data["family_types"])
        self.index_table = np.asarray(data["index_table"], dtype=np.int64)
        self.variant_counts = np.asarray(data["variant_counts"], dtype=np.int64)
        self.default_variant_table = np.asarray(data["default_variant_table"], dtype=np.int64)

        self.family_count = int(data["family_count"]) if "family_count" in data else int(len(self.family_names))
        self.variant_count_per_family = int(data["variant_count_per_family"]) if "variant_count_per_family" in data else int(self.index_table.shape[-1])
        self.max_variant_horizon = int(self.actions.shape[1])
        self.step_seconds = float(data["step_seconds"]) if "step_seconds" in data else float(self.meta.get("step_seconds", 0.2))

        self.library_meta = PrimitiveFamilyLibraryMeta(
            family_names=list(self.family_names),
            family_types=list(self.family_types),
            gamma_bin_values=np.asarray(self.gamma_bin_values, dtype=np.float64).copy(),
            variant_count_per_family=int(self.variant_count_per_family),
            horizon=int(self.max_variant_horizon),
            step_seconds=float(self.step_seconds),
            meta=dict(self.meta),
        )

        self.grid_index = None
        self.mask_index = None
        self.ray_safety_index = None
        if load_sidecars:
            self._load_sidecars()

    def _load_sidecars(self) -> None:
        try:
            from primitives.primitive_index import try_load_index_for_library

            explicit = None
            if isinstance(self.meta, dict):
                explicit = self.meta.get("mask_index_path") or self.meta.get("index_path") or self.meta.get("grid_index_path")
            self.grid_index = try_load_index_for_library(self.npz_path, explicit_index_path=explicit)
            self.mask_index = self.grid_index
        except Exception:
            self.grid_index = None
            self.mask_index = None

        try:
            from primitives.primitive_ray_safety import try_load_ray_safety_for_library

            explicit = None
            if isinstance(self.meta, dict):
                explicit = self.meta.get("ray_safety_path")
            self.ray_safety_index = try_load_ray_safety_for_library(
                self.npz_path,
                actions=self.actions,
                deltas=self.deltas,
                explicit_path=explicit,
            )
        except Exception:
            self.ray_safety_index = None

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_variants(self) -> int:
        return int(self.actions.shape[0])

    @property
    def gamma_bin_count(self) -> int:
        return int(self.gamma_bin_values.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.max_variant_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.family_count)

    def family_name(self, family_id: int) -> str:
        return str(self.family_names[int(family_id)])

    def family_type(self, family_id: int) -> str:
        return str(self.family_types[int(family_id)])

    def get_actions(self, flat_index: int) -> np.ndarray:
        idx = int(flat_index)
        horizon = int(self.variant_horizons[idx])
        return np.asarray(self.actions[idx, :horizon], dtype=np.float64)

    def get_padded_actions(self, flat_index: int) -> np.ndarray:
        return np.asarray(self.actions[int(flat_index)], dtype=np.float64)

    def get_delta(self, flat_index: int) -> np.ndarray:
        return np.asarray(self.deltas[int(flat_index)], dtype=np.float64)

    def get_rollout_states(self, flat_index: int) -> np.ndarray:
        idx = int(flat_index)
        horizon = int(self.variant_horizons[idx])
        return np.asarray(self.rollout_states[idx, : horizon + 1], dtype=np.float64)

    def get_switch_index(self, flat_index: int) -> int:
        return int(self.switch_indices[int(flat_index)])

    def is_compound_variant(self, flat_index: int) -> bool:
        return bool(int(self.is_compound[int(flat_index)]) > 0)

    def gamma_to_bin(self, gamma: float) -> int:
        gamma_val = float(gamma)
        diffs = np.abs(self.gamma_bin_values - gamma_val)
        return int(np.argmin(diffs))

    def family_variant_indices(self, gamma_bin_id: int, family_id: int) -> np.ndarray:
        gids = np.asarray(self.index_table[int(gamma_bin_id), int(family_id)], dtype=np.int64).reshape(-1)
        return gids[gids >= 0]

    def get_default_variant_index(self, gamma_bin_id: int, family_id: int) -> int:
        idx = int(self.default_variant_table[int(gamma_bin_id), int(family_id)])
        if idx >= 0:
            return idx
        candidates = self.family_variant_indices(gamma_bin_id, family_id)
        if candidates.size == 0:
            raise IndexError(f"No variants available for gamma_bin={gamma_bin_id}, family={family_id}")
        return int(candidates[0])

    def _variant_local_score(self, flat_index: int, goal_repr: Optional[Dict]) -> float:
        if not isinstance(goal_repr, dict):
            return 0.0

        delta = self.get_delta(flat_index)
        goal_xy = np.array(
            [
                float(goal_repr.get("goal_x", 0.0)),
                float(goal_repr.get("goal_y", 0.0)),
            ],
            dtype=np.float64,
        )
        delta_xy = np.asarray(delta[:2], dtype=np.float64)

        pos_err = float(np.linalg.norm(delta_xy - goal_xy))
        heading_err = abs(float(_wrap_pi_array(np.asarray([delta[2] - float(goal_repr.get("goal_heading", 0.0))], dtype=np.float64))[0]))
        art_err = abs(float(_wrap_pi_array(np.asarray([delta[3] - float(goal_repr.get("articulation", 0.0))], dtype=np.float64))[0]))
        progress = 0.0
        goal_norm = float(np.linalg.norm(goal_xy))
        if goal_norm > 1e-6:
            progress = float(np.dot(delta_xy, goal_xy / goal_norm))

        family_type = str(self.variant_flat_to_family_type[int(flat_index)])
        if family_type == "terminal":
            score = -2.0 * pos_err - 2.5 * heading_err - 0.8 * art_err + 0.35 * progress
        elif family_type == "straighten":
            score = -1.2 * pos_err - 0.8 * heading_err - 2.0 * art_err + 0.25 * progress
        elif family_type == "compound":
            score = -1.5 * pos_err - 1.3 * heading_err - 0.9 * art_err + 0.45 * progress
        else:
            score = -1.0 * pos_err - 0.9 * heading_err - 0.5 * art_err + 0.40 * progress
        score -= 0.05 * float(self.variant_flat_to_variant[int(flat_index)])
        return float(score)

    def resolve_family_variant(
        self,
        family_id: int,
        gamma: float,
        goal_repr: Optional[Dict] = None,
    ) -> PrimitiveVariantRef:
        fid = int(family_id)
        if fid < 0 or fid >= int(self.family_count):
            raise IndexError(f"family_id out of range: {fid}")

        gamma_bin_id = self.gamma_to_bin(gamma)
        candidates = self.family_variant_indices(gamma_bin_id, fid)
        if candidates.size == 0:
            flat_index = self.get_default_variant_index(gamma_bin_id, fid)
        elif goal_repr is None:
            flat_index = self.get_default_variant_index(gamma_bin_id, fid)
        else:
            best_index = None
            best_score = None
            for idx in candidates.tolist():
                score = self._variant_local_score(int(idx), goal_repr=goal_repr)
                if best_index is None or score > best_score:
                    best_index = int(idx)
                    best_score = float(score)
            flat_index = int(best_index)

        return PrimitiveVariantRef(
            flat_index=int(flat_index),
            gamma_bin_id=int(gamma_bin_id),
            family_id=int(fid),
            variant_id=int(self.variant_flat_to_variant[int(flat_index)]),
        )

    def resolved_variant_debug(self, ref: PrimitiveVariantRef) -> Dict:
        flat_index = int(ref.flat_index)
        return {
            "flat_index": int(flat_index),
            "gamma_bin_id": int(ref.gamma_bin_id),
            "gamma_bin_value": float(self.gamma_bin_values[int(ref.gamma_bin_id)]),
            "family_id": int(ref.family_id),
            "family_name": str(self.family_names[int(ref.family_id)]),
            "family_type": str(self.variant_flat_to_family_type[flat_index]),
            "variant_id": int(ref.variant_id),
            "mode": str(self.variant_flat_to_mode[flat_index]),
            "speed_sign": int(self.speed_signs[flat_index]),
            "duration": float(self.durations[flat_index]),
            "effective_horizon": int(self.variant_horizons[flat_index]),
            "is_compound": bool(self.is_compound_variant(flat_index)),
            "switch_index": int(self.get_switch_index(flat_index)),
        }


def load_library(npz_path: str, load_sidecars: bool = True) -> PrimitiveLibrary:
    return PrimitiveLibrary(npz_path=npz_path, load_sidecars=load_sidecars)


def load_library_arrays(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lib = PrimitiveLibrary(npz_path=npz_path, load_sidecars=False)
    group_ids = np.asarray(lib.variant_flat_to_family, dtype=np.int64)
    return np.asarray(lib.actions, dtype=np.float64), np.asarray(lib.deltas, dtype=np.float64), group_ids