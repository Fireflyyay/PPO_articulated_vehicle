from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from primitives.library import PrimitiveLibrary
from primitives.trajectory_miner import CandidatePrimitive


@dataclass
class LibraryVersionInfo:
    version_id: str
    parent_version_id: Optional[str]
    created_time: float
    library_path: str
    meta_path: str


@dataclass
class _VariantRecord:
    gamma_bin_id: int
    family_id: int
    variant_id: int
    actions: np.ndarray
    delta: np.ndarray
    rollout_states: np.ndarray
    variant_horizon: int
    switch_index: int
    duration: float
    speed_sign: int
    is_compound: bool
    family_type: str
    mode: str
    meta: Dict[str, Any] = field(default_factory=dict)


def _wrap_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class AdaptivePrimitiveLibraryManager:
    """Versioned family-library manager with fixed PPO action dimension.

    Adaptive updates never change PPO-visible family_count. They only insert or
    replace variant slots inside a (gamma_bin, family_id) bucket.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = bool(verbose)
        self._active_version: Optional[LibraryVersionInfo] = None
        self._library: Optional[PrimitiveLibrary] = None
        self._variant_records: List[_VariantRecord] = []
        self._version_dir: Optional[str] = None

        self._base_npz_path: Optional[str] = None
        self._pending_tmp_version: Optional[LibraryVersionInfo] = None

        self._family_names: List[str] = []
        self._family_types: List[str] = []
        self._gamma_bin_values: np.ndarray = np.zeros((0,), dtype=np.float64)
        self._variant_capacity: int = 0
        self._step_seconds: float = 0.2
        self._meta_template: Dict[str, Any] = {}
        self._family_specs: List[Dict[str, Any]] = []
        self._default_variant_slots: Dict[Tuple[int, int], int] = {}

    def load(self, base_path: str, save_dir: str) -> None:
        """Load a family-library base version into a managed directory."""
        base_path = os.path.abspath(base_path)
        self._base_npz_path = base_path

        save_dir = os.path.abspath(save_dir)
        version_dir = os.path.join(save_dir, "adaptive_primitives", "versions")
        os.makedirs(version_dir, exist_ok=True)
        self._version_dir = version_dir

        active_path = os.path.join(os.path.dirname(version_dir), "active_version.json")
        if os.path.exists(active_path):
            with open(active_path, "r", encoding="utf-8") as f:
                act = json.load(f)
            self.rollback_to(str(act.get("version_id")), save_dir=save_dir)
            return

        lib = PrimitiveLibrary(base_path)
        self._load_records_from_library(lib, meta_list=None)

        # Keep the shipped base library as the in-memory active version.
        # Re-serializing the whole library and rebuilding sidecars here makes
        # training startup dominated by offline preprocessing before episode 0.
        self._active_version = LibraryVersionInfo(
            version_id="base",
            parent_version_id=None,
            created_time=time.time(),
            library_path=base_path,
            meta_path="",
        )
        self._library = lib
        self._pending_tmp_version = None

    def get_active_library(self) -> PrimitiveLibrary:
        assert self._library is not None, "Library manager not loaded"
        return self._library

    @property
    def active_version_id(self) -> str:
        return self._active_version.version_id if self._active_version is not None else "unknown"

    @property
    def library_size(self) -> int:
        return int(len(self._variant_records))

    @staticmethod
    def _mirror_actions(actions: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(actions, dtype=np.float64).copy()
        mirrored[:, 0] = -mirrored[:, 0]
        return mirrored

    @staticmethod
    def _mirror_delta(delta4: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(delta4, dtype=np.float64).copy()
        if mirrored.shape[0] >= 4:
            mirrored[1] = -mirrored[1]
            mirrored[2] = -mirrored[2]
            mirrored[3] = -mirrored[3]
        return mirrored

    def _load_records_from_library(self, lib: PrimitiveLibrary, meta_list: Optional[List[Dict[str, Any]]]) -> None:
        self._family_names = list(lib.family_names)
        self._family_types = list(lib.family_types)
        self._gamma_bin_values = np.asarray(lib.gamma_bin_values, dtype=np.float64).copy()
        self._variant_capacity = int(lib.variant_count_per_family)
        self._step_seconds = float(getattr(lib, "step_seconds", 0.2))
        self._meta_template = dict(lib.meta) if isinstance(lib.meta, dict) else {}
        self._family_specs = self._extract_family_specs(lib)
        self._default_variant_slots = {}

        for gamma_bin_id in range(lib.gamma_bin_count):
            for family_id in range(lib.family_count):
                flat_index = int(lib.default_variant_table[gamma_bin_id, family_id])
                if flat_index >= 0:
                    self._default_variant_slots[(int(gamma_bin_id), int(family_id))] = int(
                        lib.variant_flat_to_variant[flat_index]
                    )

        records: List[_VariantRecord] = []
        for flat_index in range(int(lib.size)):
            meta = self._default_meta_for_variant(lib, flat_index)
            if meta_list is not None and flat_index < len(meta_list):
                loaded_meta = dict(meta_list[flat_index] or {})
                loaded_meta.setdefault("family_id", int(lib.variant_flat_to_family[flat_index]))
                loaded_meta.setdefault("gamma_bin_id", int(lib.variant_flat_to_gamma[flat_index]))
                loaded_meta.setdefault("variant_id", int(lib.variant_flat_to_variant[flat_index]))
                meta = loaded_meta

            records.append(
                _VariantRecord(
                    gamma_bin_id=int(lib.variant_flat_to_gamma[flat_index]),
                    family_id=int(lib.variant_flat_to_family[flat_index]),
                    variant_id=int(lib.variant_flat_to_variant[flat_index]),
                    actions=np.asarray(lib.get_actions(flat_index), dtype=np.float64),
                    delta=np.asarray(lib.get_delta(flat_index), dtype=np.float64),
                    rollout_states=np.asarray(lib.get_rollout_states(flat_index), dtype=np.float64),
                    variant_horizon=int(lib.variant_horizons[flat_index]),
                    switch_index=int(lib.switch_indices[flat_index]),
                    duration=float(lib.durations[flat_index]),
                    speed_sign=int(lib.speed_signs[flat_index]),
                    is_compound=bool(lib.is_compound[flat_index]),
                    family_type=str(lib.variant_flat_to_family_type[flat_index]),
                    mode=str(lib.variant_flat_to_mode[flat_index]),
                    meta=meta,
                )
            )

        self._variant_records = records

    def _extract_family_specs(self, lib: PrimitiveLibrary) -> List[Dict[str, Any]]:
        raw_specs = []
        if isinstance(lib.meta, dict):
            raw_specs = list(lib.meta.get("family_specs", []) or [])
        if len(raw_specs) == int(lib.family_count):
            return [dict(spec) for spec in raw_specs]

        try:
            from configs import VALID_SPEED, VALID_STEER

            max_speed = float(max(abs(float(VALID_SPEED[0])), abs(float(VALID_SPEED[1]))))
            max_steer = float(max(abs(float(VALID_STEER[0])), abs(float(VALID_STEER[1]))))
        except Exception:
            max_speed = 1.0
            max_steer = 1.0

        specs: List[Dict[str, Any]] = []
        for family_id in range(int(lib.family_count)):
            family_matches = np.where(np.asarray(lib.variant_flat_to_family, dtype=np.int64) == int(family_id))[0]
            flat_index = int(family_matches[0])
            actions = np.asarray(lib.get_actions(flat_index), dtype=np.float64)
            mean_speed = float(np.mean(np.abs(actions[:, 1]))) if actions.size else 0.0
            mean_steer = float(np.mean(actions[:, 0])) if actions.size else 0.0
            specs.append(
                {
                    "family_id": int(family_id),
                    "name": str(lib.family_names[family_id]),
                    "family_type": str(lib.family_types[family_id]),
                    "speed_sign": int(lib.speed_signs[flat_index]),
                    "speed_scale": float(mean_speed / max(max_speed, 1e-6)),
                    "gamma_rate_scale": float(mean_steer / max(max_steer, 1e-6)),
                    "mode": str(lib.variant_flat_to_mode[flat_index]),
                    "compound_split": None,
                    "compound_exit_gamma_scale": float(-mean_steer / max(max_steer, 1e-6)),
                }
            )
        return specs

    def _default_meta_for_variant(self, lib: PrimitiveLibrary, flat_index: int) -> Dict[str, Any]:
        return {
            "flat_index": int(flat_index),
            "family_id": int(lib.variant_flat_to_family[flat_index]),
            "gamma_bin_id": int(lib.variant_flat_to_gamma[flat_index]),
            "variant_id": int(lib.variant_flat_to_variant[flat_index]),
            "added_round": -1,
            "source_episode_id": None,
            "source_scene_type": None,
            "source_segment_range": None,
            "reverse_ratio": None,
            "steer_change_rate": None,
            "curvature_proxy": None,
            "complexity_score": None,
            "novelty_score": None,
            "utility_score": None,
            "delta_pose": np.asarray(lib.get_delta(flat_index), dtype=np.float64)[:3].astype(float).tolist(),
            "enabled": True,
            "usage_count": 0,
            "success_usage_count": 0,
            "family_name": str(lib.family_names[int(lib.variant_flat_to_family[flat_index])]),
            "family_type": str(lib.variant_flat_to_family_type[flat_index]),
            "mode": str(lib.variant_flat_to_mode[flat_index]),
        }

    def _current_horizon(self) -> int:
        if self._library is not None:
            return int(getattr(self._library, "horizon", 1))
        if len(self._variant_records) > 0:
            return int(max(rec.variant_horizon for rec in self._variant_records))
        return 1

    def _sort_records(self) -> List[_VariantRecord]:
        return sorted(
            self._variant_records,
            key=lambda rec: (int(rec.gamma_bin_id), int(rec.family_id), int(rec.variant_id)),
        )

    def _bucket_records(self, gamma_bin_id: int, family_id: int) -> List[_VariantRecord]:
        return sorted(
            [
                rec
                for rec in self._variant_records
                if int(rec.gamma_bin_id) == int(gamma_bin_id) and int(rec.family_id) == int(family_id)
            ],
            key=lambda rec: int(rec.variant_id),
        )

    def _candidate_variants(
        self,
        cand: CandidatePrimitive,
        H: int,
        config=None,
    ) -> List[Tuple[str, np.ndarray]]:
        actions = np.asarray(cand.actions_resampled, dtype=np.float64)
        if actions.shape != (int(H), 2):
            return []

        variants: List[Tuple[str, np.ndarray]] = [("original", actions)]
        if not bool(getattr(config, "AP_AUTO_ADD_SYMMETRIC_PRIMITIVES", True)):
            return variants

        atol = float(getattr(config, "AP_SYMMETRIC_ACTION_ATOL", 1e-6))
        mirrored_actions = self._mirror_actions(actions)
        if np.allclose(mirrored_actions, actions, atol=atol, rtol=0.0):
            return variants
        variants.append(("mirrored", mirrored_actions))
        return variants

    def _candidate_articulation(self, cand: CandidatePrimitive) -> float:
        try:
            if cand.start_feature is not None and len(cand.start_feature) >= 6:
                return float(_wrap_pi(float(cand.start_feature[2]) - float(cand.start_feature[5])))
        except Exception:
            pass
        try:
            start_state = cand.source_metadata.get("start_state", {})
            return float(
                _wrap_pi(
                    float(start_state.get("heading", 0.0))
                    - float(start_state.get("rear_heading", start_state.get("heading", 0.0)))
                )
            )
        except Exception:
            return 0.0

    def _classify_candidate(self, actions: np.ndarray, cand: CandidatePrimitive) -> Tuple[int, int, Dict[str, Any]]:
        try:
            from configs import AP_V_TH, VALID_SPEED, VALID_STEER

            v_th = float(AP_V_TH)
            max_speed = float(max(abs(float(VALID_SPEED[0])), abs(float(VALID_SPEED[1]))))
            max_steer = float(max(abs(float(VALID_STEER[0])), abs(float(VALID_STEER[1]))))
        except Exception:
            v_th = 0.1
            max_speed = 1.0
            max_steer = 1.0

        actions = np.asarray(actions, dtype=np.float64)
        speeds = actions[:, 1]
        steers = actions[:, 0]

        nz_speed = np.where(np.abs(speeds) > v_th)[0]
        primary_speed_sign = int(np.sign(speeds[int(nz_speed[0])])) if nz_speed.size > 0 else 1
        primary_speed_sign = 1 if primary_speed_sign == 0 else primary_speed_sign

        steer_ref = steers[: max(1, actions.shape[0] // 2)] if actions.shape[0] > 1 else steers
        primary_steer = float(np.mean(steer_ref)) if steer_ref.size > 0 else 0.0
        mean_abs_speed = float(np.mean(np.abs(speeds))) if speeds.size > 0 else 0.0
        speed_sign_changes = int(np.sum(np.sign(speeds[1:]) != np.sign(speeds[:-1]))) if speeds.size > 1 else 0
        steer_sign_changes = int(np.sum(np.sign(steers[1:]) != np.sign(steers[:-1]))) if steers.size > 1 else 0

        if speed_sign_changes > 0:
            target_family_type = "compound"
        elif mean_abs_speed < 0.40 * max_speed:
            target_family_type = "terminal"
        elif steer_sign_changes > 0 and abs(primary_steer) < 0.35 * max_steer:
            target_family_type = "straighten"
        else:
            target_family_type = "normal"

        gamma0 = self._candidate_articulation(cand)
        gamma_bin_id = int(np.argmin(np.abs(self._gamma_bin_values - float(gamma0))))

        best_family_id = None
        best_score = None
        for spec in self._family_specs:
            family_id = int(spec.get("family_id", -1))
            if family_id < 0:
                continue

            score = 0.0
            if str(spec.get("family_type", "normal")) != target_family_type:
                score -= 5.0
            if int(spec.get("speed_sign", 1)) != int(primary_speed_sign):
                score -= 3.5

            target_speed = abs(float(spec.get("speed_scale", 1.0))) * max_speed
            target_steer = float(spec.get("gamma_rate_scale", 0.0)) * max_steer
            score -= abs(mean_abs_speed - target_speed)
            score -= 1.5 * abs(primary_steer - target_steer)

            if best_family_id is None or score > best_score:
                best_family_id = family_id
                best_score = float(score)

        if best_family_id is None:
            raise RuntimeError("failed to classify candidate family")

        return int(gamma_bin_id), int(best_family_id), {
            "gamma0": float(gamma0),
            "target_family_type": str(target_family_type),
            "primary_speed_sign": int(primary_speed_sign),
        }

    def _simulate_variant(self, actions: np.ndarray, gamma0: float) -> Tuple[np.ndarray, np.ndarray, int, int]:
        from configs import HITCH_OFFSET, TRAILER_LENGTH
        from env.vehicle import Vehicle
        from primitives.generate_primitives import _initial_state_for_gamma

        vehicle = Vehicle(
            articulated=True,
            trailer_length=TRAILER_LENGTH,
            hitch_offset=HITCH_OFFSET,
        )
        vehicle.reset(_initial_state_for_gamma(float(gamma0)))

        rollout_states = [
            np.asarray(
                [
                    float(vehicle.state.loc.x),
                    float(vehicle.state.loc.y),
                    float(vehicle.state.heading),
                    float(vehicle.state.rear_heading),
                    float(getattr(vehicle.state, "speed", 0.0)),
                    float(getattr(vehicle.state, "steering", 0.0)),
                ],
                dtype=np.float64,
            )
        ]

        speeds = np.asarray(actions[:, 1], dtype=np.float64)
        switch_index = -1
        for idx in range(1, speeds.shape[0]):
            if np.sign(speeds[idx - 1]) != np.sign(speeds[idx]) and abs(speeds[idx - 1]) > 1e-6 and abs(speeds[idx]) > 1e-6:
                switch_index = int(idx)
                break

        for action in np.asarray(actions, dtype=np.float64):
            vehicle.step(np.asarray(action, dtype=np.float64), step_time=1)
            rollout_states.append(
                np.asarray(
                    [
                        float(vehicle.state.loc.x),
                        float(vehicle.state.loc.y),
                        float(vehicle.state.heading),
                        float(vehicle.state.rear_heading),
                        float(getattr(vehicle.state, "speed", 0.0)),
                        float(getattr(vehicle.state, "steering", 0.0)),
                    ],
                    dtype=np.float64,
                )
            )

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
        speed_sign = int(np.sign(np.mean(speeds))) if np.any(np.abs(speeds) > 1e-8) else 0
        return np.asarray(rollout_states, dtype=np.float64), delta, int(speed_sign), int(switch_index)

    def _record_exists(
        self,
        gamma_bin_id: int,
        family_id: int,
        actions: np.ndarray,
        delta: np.ndarray,
        atol: float,
    ) -> bool:
        for rec in self._bucket_records(gamma_bin_id, family_id):
            if rec.actions.shape != actions.shape:
                continue
            if np.allclose(rec.actions, actions, atol=atol, rtol=0.0) and np.allclose(rec.delta, delta, atol=atol, rtol=0.0):
                return True
        return False

    def _replacement_target(self, bucket: Sequence[_VariantRecord], config) -> Optional[_VariantRecord]:
        policy = str(
            getattr(config, "PRIMITIVE_ADAPTIVE_SLOT_POLICY", getattr(config, "AP_SLOT_POLICY", "replace_low_utility"))
        )
        if policy != "replace_low_utility":
            return None

        discovered = [rec for rec in bucket if int(rec.meta.get("added_round", -1)) >= 0]
        if len(discovered) == 0:
            return None

        def utility(rec: _VariantRecord) -> float:
            value = rec.meta.get("utility_score", None)
            if value is None:
                return -np.inf
            try:
                return float(value)
            except Exception:
                return -np.inf

        return min(discovered, key=utility)

    def _config_max_variants(self, config) -> int:
        if config is not None:
            try:
                return max(int(self._variant_capacity), int(getattr(config, "PRIMITIVE_ADAPTIVE_MAX_VARIANTS_PER_FAMILY")))
            except Exception:
                pass
        try:
            import configs as cfg

            return max(int(self._variant_capacity), int(getattr(cfg, "PRIMITIVE_ADAPTIVE_MAX_VARIANTS_PER_FAMILY")))
        except Exception:
            return int(max(1, self._variant_capacity))

    def add_candidates(self, candidates: Sequence[CandidatePrimitive], round_id: int, config=None) -> int:
        """Insert or replace candidate variants inside family slots."""
        if self._library is None:
            raise RuntimeError("Library not loaded")

        if len(candidates) == 0:
            return 0

        H = int(self._current_horizon())
        atol = float(getattr(config, "AP_SYMMETRIC_ACTION_ATOL", 1e-6))
        max_variants = int(max(1, self._config_max_variants(config)))
        self._variant_capacity = max(int(self._variant_capacity), max_variants)

        changed = 0
        for cand in candidates:
            for symmetry_source, actions in self._candidate_variants(cand, H=H, config=config):
                gamma_bin_id, family_id, classify_debug = self._classify_candidate(actions, cand)
                gamma0 = float(self._gamma_bin_values[int(gamma_bin_id)])
                rollout_states, delta4, speed_sign, switch_index = self._simulate_variant(actions, gamma0=gamma0)

                if self._record_exists(gamma_bin_id, family_id, actions, delta4, atol=atol):
                    continue

                bucket = self._bucket_records(gamma_bin_id, family_id)
                family_type = str(self._family_types[int(family_id)])
                mode = str(self._family_specs[int(family_id)].get("mode", family_type))
                duration = float(self._step_seconds) * float(H)
                is_compound = bool(family_type == "compound")

                meta = {
                    "family_id": int(family_id),
                    "family_name": str(self._family_names[int(family_id)]),
                    "family_type": str(family_type),
                    "gamma_bin_id": int(gamma_bin_id),
                    "gamma_bin_value": float(gamma0),
                    "variant_id": -1,
                    "added_round": int(round_id),
                    "source_episode_id": cand.source_metadata.get("episode_id", None),
                    "source_scene_type": cand.source_metadata.get("scene_type", None),
                    "source_segment_range": cand.source_metadata.get(
                        "segment_range", cand.source_metadata.get("source_segment_range", None)
                    ),
                    "reverse_ratio": cand.tags.get("reverse_ratio", None) if isinstance(cand.tags, dict) else None,
                    "steer_change_rate": cand.tags.get("steer_sign_changes", None) if isinstance(cand.tags, dict) else None,
                    "curvature_proxy": cand.tags.get("beta_abs_mean", None) if isinstance(cand.tags, dict) else None,
                    "complexity_score": float(cand.complexity_score),
                    "novelty_score": float(cand.novelty_score),
                    "utility_score": float(cand.utility_score),
                    "delta_pose": delta4[:3].astype(float).tolist(),
                    "enabled": True,
                    "usage_count": 0,
                    "success_usage_count": 0,
                    "mode": str(mode),
                    "symmetry_source": str(symmetry_source),
                    "classification": classify_debug,
                }

                if len(bucket) < max_variants:
                    variant_id = int(max([-1] + [int(rec.variant_id) for rec in bucket]) + 1)
                    meta["variant_id"] = int(variant_id)
                    self._variant_records.append(
                        _VariantRecord(
                            gamma_bin_id=int(gamma_bin_id),
                            family_id=int(family_id),
                            variant_id=int(variant_id),
                            actions=np.asarray(actions, dtype=np.float64),
                            delta=np.asarray(delta4, dtype=np.float64),
                            rollout_states=np.asarray(rollout_states, dtype=np.float64),
                            variant_horizon=int(H),
                            switch_index=int(switch_index),
                            duration=float(duration),
                            speed_sign=int(speed_sign),
                            is_compound=bool(is_compound),
                            family_type=str(family_type),
                            mode=str(mode),
                            meta=meta,
                        )
                    )
                    changed += 1
                    continue

                target = self._replacement_target(bucket, config)
                if target is None:
                    continue

                target_utility = target.meta.get("utility_score", None)
                target_utility = float(target_utility) if target_utility is not None else -np.inf
                if float(cand.utility_score) <= float(target_utility):
                    continue

                meta["variant_id"] = int(target.variant_id)
                target.actions = np.asarray(actions, dtype=np.float64)
                target.delta = np.asarray(delta4, dtype=np.float64)
                target.rollout_states = np.asarray(rollout_states, dtype=np.float64)
                target.variant_horizon = int(H)
                target.switch_index = int(switch_index)
                target.duration = float(duration)
                target.speed_sign = int(speed_sign)
                target.is_compound = bool(is_compound)
                target.family_type = str(family_type)
                target.mode = str(mode)
                target.meta = meta
                changed += 1

        if changed > 0:
            info = self._save_version_internal(
                version_id=f"tmp_{int(time.time())}",
                parent_version_id=self.active_version_id,
                temporary=True,
            )
            self._pending_tmp_version = info
            self._library = PrimitiveLibrary(info.library_path, load_sidecars=False)

        return int(changed)

    def save_version(self, save_dir: str, version_id: Optional[str] = None) -> LibraryVersionInfo:
        if self._library is None:
            raise RuntimeError("Library not loaded")

        if version_id is None:
            version_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        info = self._save_version_internal(
            version_id=str(version_id),
            parent_version_id=self._active_version.version_id if self._active_version else None,
            temporary=False,
        )
        self._active_version = info
        self._library = PrimitiveLibrary(info.library_path)
        self._pending_tmp_version = None
        self._write_active_version(save_dir, info.version_id)
        return info

    def rollback_to(self, version_id: str, save_dir: str) -> None:
        if self._version_dir is None:
            save_dir = os.path.abspath(save_dir)
            self._version_dir = os.path.join(save_dir, "adaptive_primitives", "versions")

        npz_path = os.path.join(self._version_dir, f"primitives_v{version_id}.npz")
        meta_path = os.path.join(self._version_dir, f"primitives_v{version_id}_meta.json")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)

        lib = PrimitiveLibrary(npz_path)
        meta_list: List[Dict[str, Any]] = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_list = json.load(f)

        self._load_records_from_library(lib, meta_list=meta_list)
        self._library = lib
        self._active_version = LibraryVersionInfo(
            version_id=str(version_id),
            parent_version_id=None,
            created_time=time.time(),
            library_path=npz_path,
            meta_path=meta_path,
        )
        self._pending_tmp_version = None
        self._write_active_version(save_dir, str(version_id))

    def export_stats(self) -> Dict[str, Any]:
        return {
            "active_version": self.active_version_id,
            "library_size": int(self.library_size),
            "enabled_count": int(sum(1 for rec in self._variant_records if bool(rec.meta.get("enabled", True)))),
        }

    def get_meta(self) -> List[Dict[str, Any]]:
        return [dict(rec.meta) for rec in self._sort_records()]

    def _serialize_records(self) -> Dict[str, Any]:
        sorted_records = self._sort_records()
        gamma_count = int(len(self._gamma_bin_values))
        family_count = int(len(self._family_names))
        variant_capacity = int(max(1, self._variant_capacity))

        index_table = np.full((gamma_count, family_count, variant_capacity), -1, dtype=np.int64)
        variant_counts = np.zeros((gamma_count, family_count), dtype=np.int64)
        default_variant_table = np.full((gamma_count, family_count), -1, dtype=np.int64)

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
        flat_to_variant: List[int] = []
        flat_to_family_type: List[str] = []
        flat_to_mode: List[str] = []
        meta_list: List[Dict[str, Any]] = []

        slot_to_flat: Dict[Tuple[int, int, int], int] = {}
        horizon = int(self._current_horizon())
        state_dim = 6

        for flat_index, rec in enumerate(sorted_records):
            padded_actions = np.asarray(rec.actions, dtype=np.float64)
            if padded_actions.shape[0] < horizon:
                pad = np.repeat(padded_actions[-1:, :], horizon - padded_actions.shape[0], axis=0)
                padded_actions = np.concatenate([padded_actions, pad], axis=0)

            padded_rollout = np.asarray(rec.rollout_states, dtype=np.float64)
            if padded_rollout.shape[0] < (horizon + 1):
                if padded_rollout.shape[0] == 0:
                    padded_rollout = np.zeros((horizon + 1, state_dim), dtype=np.float64)
                else:
                    pad = np.repeat(padded_rollout[-1:, :], (horizon + 1) - padded_rollout.shape[0], axis=0)
                    padded_rollout = np.concatenate([padded_rollout, pad], axis=0)

            flat_actions.append(padded_actions)
            flat_rollout_states.append(padded_rollout)
            flat_deltas.append(np.asarray(rec.delta, dtype=np.float64))
            flat_variant_horizons.append(int(rec.variant_horizon))
            flat_switch_indices.append(int(rec.switch_index))
            flat_durations.append(float(rec.duration))
            flat_speed_signs.append(int(rec.speed_sign))
            flat_is_compound.append(int(rec.is_compound))
            flat_to_gamma.append(int(rec.gamma_bin_id))
            flat_to_family.append(int(rec.family_id))
            flat_to_variant.append(int(rec.variant_id))
            flat_to_family_type.append(str(rec.family_type))
            flat_to_mode.append(str(rec.mode))

            meta_item = dict(rec.meta)
            meta_item["flat_index"] = int(flat_index)
            meta_item["family_id"] = int(rec.family_id)
            meta_item["gamma_bin_id"] = int(rec.gamma_bin_id)
            meta_item["variant_id"] = int(rec.variant_id)
            meta_list.append(meta_item)

            slot_to_flat[(int(rec.gamma_bin_id), int(rec.family_id), int(rec.variant_id))] = int(flat_index)
            index_table[int(rec.gamma_bin_id), int(rec.family_id), int(rec.variant_id)] = int(flat_index)
            variant_counts[int(rec.gamma_bin_id), int(rec.family_id)] += 1

        for gamma_bin_id in range(gamma_count):
            for family_id in range(family_count):
                preferred_slot = self._default_variant_slots.get((int(gamma_bin_id), int(family_id)), None)
                if preferred_slot is not None and (int(gamma_bin_id), int(family_id), int(preferred_slot)) in slot_to_flat:
                    default_variant_table[int(gamma_bin_id), int(family_id)] = slot_to_flat[
                        (int(gamma_bin_id), int(family_id), int(preferred_slot))
                    ]
                    continue

                bucket = self._bucket_records(gamma_bin_id, family_id)
                if len(bucket) > 0:
                    first_slot = int(bucket[0].variant_id)
                    default_variant_table[int(gamma_bin_id), int(family_id)] = slot_to_flat[
                        (int(gamma_bin_id), int(family_id), first_slot)
                    ]

        meta_payload = dict(self._meta_template)
        meta_payload.update(
            {
                "H": int(horizon),
                "variant_count": int(variant_capacity),
                "gamma_bins": int(gamma_count),
                "family_specs": [dict(spec) for spec in self._family_specs],
            }
        )

        return {
            "schema_version": np.asarray("family_library_v1", dtype=object),
            "actions": np.asarray(flat_actions, dtype=np.float64),
            "deltas": np.asarray(flat_deltas, dtype=np.float64),
            "rollout_states": np.asarray(flat_rollout_states, dtype=np.float64),
            "variant_horizons": np.asarray(flat_variant_horizons, dtype=np.int64),
            "switch_indices": np.asarray(flat_switch_indices, dtype=np.int64),
            "durations": np.asarray(flat_durations, dtype=np.float64),
            "speed_signs": np.asarray(flat_speed_signs, dtype=np.int64),
            "is_compound": np.asarray(flat_is_compound, dtype=np.int8),
            "variant_flat_to_gamma": np.asarray(flat_to_gamma, dtype=np.int64),
            "variant_flat_to_family": np.asarray(flat_to_family, dtype=np.int64),
            "variant_flat_to_variant": np.asarray(flat_to_variant, dtype=np.int64),
            "variant_flat_to_family_type": np.asarray(flat_to_family_type, dtype=object),
            "variant_flat_to_mode": np.asarray(flat_to_mode, dtype=object),
            "gamma_bin_values": np.asarray(self._gamma_bin_values, dtype=np.float64),
            "family_names": np.asarray(self._family_names, dtype=object),
            "family_types": np.asarray(self._family_types, dtype=object),
            "family_count": np.asarray(family_count, dtype=np.int64),
            "variant_count_per_family": np.asarray(variant_capacity, dtype=np.int64),
            "index_table": np.asarray(index_table, dtype=np.int64),
            "variant_counts": np.asarray(variant_counts, dtype=np.int64),
            "default_variant_table": np.asarray(default_variant_table, dtype=np.int64),
            "step_seconds": np.asarray(float(self._step_seconds), dtype=np.float64),
            "meta": np.asarray(meta_payload, dtype=object),
            "meta_list": meta_list,
        }

    def _save_version_internal(
        self,
        version_id: str,
        parent_version_id: Optional[str],
        temporary: bool = False,
    ) -> LibraryVersionInfo:
        assert self._version_dir is not None, "version_dir not initialized"

        created_time = time.time()
        npz_name = f"primitives_v{version_id}.npz"
        meta_name = f"primitives_v{version_id}_meta.json"

        npz_path = os.path.join(self._version_dir, npz_name)
        meta_path = os.path.join(self._version_dir, meta_name)
        base, _ = os.path.splitext(npz_path)
        mask_index_path = base + ".mask_index.npz"
        ray_safety_path = base + ".ray_safety.npz"

        payload = self._serialize_records()
        meta_list = payload.pop("meta_list")
        meta_obj = payload["meta"].item() if isinstance(payload["meta"], np.ndarray) else payload["meta"]
        meta_obj = dict(meta_obj)
        meta_obj.update(
            {
                "created_time": float(created_time),
                "parent_version": parent_version_id,
                "mask_index_path": mask_index_path,
                "ray_safety_path": ray_safety_path,
            }
        )
        payload["meta"] = np.asarray(meta_obj, dtype=object)

        np.savez_compressed(npz_path, **payload)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False, indent=2)

        if not temporary:
            self._build_library_sidecars(
                npz_path=npz_path,
                actions=np.asarray(payload["actions"], dtype=np.float64),
                deltas=np.asarray(payload["deltas"], dtype=np.float64),
                H=int(np.asarray(payload["actions"]).shape[1]) if np.asarray(payload["actions"]).ndim >= 2 else 1,
            )

        return LibraryVersionInfo(
            version_id=str(version_id),
            parent_version_id=parent_version_id,
            created_time=created_time,
            library_path=npz_path,
            meta_path=meta_path,
        )

    def _build_library_sidecars(
        self,
        npz_path: str,
        actions: np.ndarray,
        deltas: np.ndarray,
        H: int,
    ) -> None:
        try:
            import configs as cfg
        except Exception:
            cfg = None

        base, _ = os.path.splitext(npz_path)
        mask_index_path = base + ".mask_index.npz"
        ray_safety_path = base + ".ray_safety.npz"

        try:
            from primitives.primitive_index import build_mask_index_from_library, primitive_grid_index_to_payload

            grid_resolution = float(getattr(cfg, "GRID_RESOLUTION", 0.3))
            num_step = int(getattr(cfg, "NUM_STEP", 4))
            group_prefix_steps = max(1, min(int(H), int(round(float(H) * 0.3))))
            index = build_mask_index_from_library(
                actions=np.asarray(actions, dtype=np.float64),
                grid_resolution=grid_resolution,
                x_min=-6.0,
                x_max=12.0,
                y_min=-9.0,
                y_max=9.0,
                sample_stride=1,
                num_step=num_step,
                group_prefix_steps=group_prefix_steps,
            )
            np.savez_compressed(mask_index_path, **primitive_grid_index_to_payload(index))
        except Exception as exc:
            if self.verbose:
                print(f"[adaptive] warning: failed to build mask index for {npz_path}: {exc}")

        try:
            from primitives.primitive_ray_safety import build_ray_safety_index_from_library, save_ray_safety_index

            ray_index = build_ray_safety_index_from_library(
                actions=np.asarray(actions, dtype=np.float64),
                deltas=np.asarray(deltas, dtype=np.float64),
                lidar_num=int(getattr(cfg, "LIDAR_NUM", 120)),
                lidar_range=float(getattr(cfg, "LIDAR_RANGE", 30.0)),
                safety_margin=float(getattr(cfg, "SOFT_MASK_SAFETY_MARGIN", 0.25)),
                reverse_margin_scale=float(getattr(cfg, "SOFT_MASK_REVERSE_MARGIN_SCALE", 1.2)),
                sample_stride=2,
                num_step=int(getattr(cfg, "NUM_STEP", 4)),
            )
            save_ray_safety_index(ray_safety_path, ray_index)
        except Exception as exc:
            if self.verbose:
                print(f"[adaptive] warning: failed to build ray safety index for {npz_path}: {exc}")

    def _write_active_version(self, save_dir: str, version_id: str) -> None:
        save_dir = os.path.abspath(save_dir)
        root = os.path.join(save_dir, "adaptive_primitives")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, "active_version.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version_id": str(version_id)}, f, ensure_ascii=False, indent=2)