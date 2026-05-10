from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
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


class AdaptivePrimitiveLibraryManager:
    """Versioned primitive library manager with incremental add + rollback.

    Persistence format per version:
    - primitives_v{version_id}.npz : actions [N,H,2], deltas [N,4], meta (dict)
    - primitives_v{version_id}_meta.json : list[primitive_meta]

    The `meta.json` is intentionally separate to keep numpy arrays compact and readable.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = bool(verbose)
        self._active_version: Optional[LibraryVersionInfo] = None
        self._library: Optional[PrimitiveLibrary] = None
        self._primitive_meta: List[Dict[str, Any]] = []
        self._version_dir: Optional[str] = None

        self._base_npz_path: Optional[str] = None
        self._pending_tmp_version: Optional[LibraryVersionInfo] = None

    def load(self, base_path: str, save_dir: str) -> None:
        """Load base library as version 'base' into a managed directory."""
        base_path = os.path.abspath(base_path)
        self._base_npz_path = base_path

        save_dir = os.path.abspath(save_dir)
        version_dir = os.path.join(save_dir, "adaptive_primitives", "versions")
        os.makedirs(version_dir, exist_ok=True)
        self._version_dir = version_dir

        # Create initial managed version if none exists
        active_path = os.path.join(os.path.dirname(version_dir), "active_version.json")
        if os.path.exists(active_path):
            with open(active_path, "r", encoding="utf-8") as f:
                act = json.load(f)
            self.rollback_to(str(act.get("version_id")), save_dir=save_dir)
            return

        lib = PrimitiveLibrary(base_path)
        H = int(getattr(lib, "horizon", lib.actions.shape[1]))

        # Build default meta
        meta_list: List[Dict[str, Any]] = []
        for pid in range(int(lib.size)):
            meta_list.append(
                {
                    "primitive_id": int(pid),
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
                    "delta_pose": None,
                    "enabled": True,
                    "usage_count": 0,
                    "success_usage_count": 0,
                }
            )

        version_id = "base"
        info = self._save_version_internal(
            version_id=version_id,
            parent_version_id=None,
            actions=np.asarray(lib.actions, dtype=np.float64),
            deltas=np.asarray(lib.deltas, dtype=np.float64),
            meta_list=meta_list,
            H=H,
        )
        self._active_version = info
        self._library = PrimitiveLibrary(info.library_path)
        self._primitive_meta = meta_list
        self._write_active_version(save_dir, info.version_id)

    def get_active_library(self) -> PrimitiveLibrary:
        assert self._library is not None, "Library manager not loaded"
        return self._library

    @property
    def active_version_id(self) -> str:
        return self._active_version.version_id if self._active_version is not None else "unknown"

    @property
    def library_size(self) -> int:
        return int(getattr(self._library, "size", 0)) if self._library is not None else 0

    @staticmethod
    def _mirror_actions(actions: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(actions, dtype=np.float64).copy()
        mirrored[:, 0] = -mirrored[:, 0]
        return mirrored

    @staticmethod
    def _mirror_delta(delta4: np.ndarray) -> np.ndarray:
        mirrored = np.asarray(delta4, dtype=np.float64).copy()
        if mirrored.shape[0] >= 3:
            mirrored[1] = -mirrored[1]
            mirrored[2] = -mirrored[2]
        return mirrored

    @staticmethod
    def _variant_exists(
        actions: np.ndarray,
        deltas: np.ndarray,
        candidate_actions: np.ndarray,
        candidate_delta: np.ndarray,
        atol: float,
    ) -> bool:
        if actions.ndim != 3 or deltas.ndim != 2:
            return False
        if actions.shape[1:] != candidate_actions.shape or deltas.shape[1:] != candidate_delta.shape:
            return False

        action_match = np.all(np.isclose(actions, candidate_actions[None, :, :], atol=atol, rtol=0.0), axis=(1, 2))
        if not np.any(action_match):
            return False

        delta_match = np.all(np.isclose(deltas, candidate_delta[None, :], atol=atol, rtol=0.0), axis=1)
        return bool(np.any(action_match & delta_match))

    def _candidate_variants(
        self,
        cand: CandidatePrimitive,
        H: int,
        config=None,
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        u = np.asarray(cand.actions_resampled, dtype=np.float64)
        if u.shape != (H, 2):
            return []

        dx, dy, dyaw = 0.0, 0.0, 0.0
        if cand.delta_feature is not None and len(cand.delta_feature) >= 3:
            dx, dy, dyaw = map(float, cand.delta_feature[:3])
        delta4 = np.asarray([dx, dy, dyaw, 0.0], dtype=np.float64)

        variants: List[Tuple[str, np.ndarray, np.ndarray]] = [("original", u, delta4)]

        if not bool(getattr(config, "AP_AUTO_ADD_SYMMETRIC_PRIMITIVES", True)):
            return variants

        atol = float(getattr(config, "AP_SYMMETRIC_ACTION_ATOL", 1e-6))
        mirrored_u = self._mirror_actions(u)
        mirrored_delta4 = self._mirror_delta(delta4)
        if np.allclose(mirrored_u, u, atol=atol, rtol=0.0) and np.allclose(mirrored_delta4, delta4, atol=atol, rtol=0.0):
            return variants

        variants.append(("mirrored", mirrored_u, mirrored_delta4))
        return variants

    def add_candidates(self, candidates: Sequence[CandidatePrimitive], round_id: int, config=None) -> int:
        """Append candidates to library in-memory (not persisted until save_version()).

        When enabled by config, each newly discovered primitive also contributes
        a left-right mirrored counterpart so downstream action-mask sidecars cover
        both steering directions.
        """
        if self._library is None:
            raise RuntimeError("Library not loaded")

        if len(candidates) == 0:
            return 0

        actions = np.asarray(self._library.actions, dtype=np.float64)
        deltas = np.asarray(self._library.deltas, dtype=np.float64)
        H = int(getattr(self._library, "horizon", actions.shape[1]))
        atol = float(getattr(config, "AP_SYMMETRIC_ACTION_ATOL", 1e-6))

        added = 0
        for cand in candidates:
            variant_parent_pid = None
            for variant_name, u, delta4 in self._candidate_variants(cand, H=H, config=config):
                if self._variant_exists(actions, deltas, u, delta4, atol=atol):
                    continue

                actions = np.concatenate([actions, u[None, :, :]], axis=0)
                deltas = np.concatenate([deltas, delta4[None, :]], axis=0)

                pid = int(actions.shape[0] - 1)
                meta = {
                    "primitive_id": pid,
                    "added_round": int(round_id),
                    "source_episode_id": cand.source_metadata.get("episode_id", None),
                    "source_scene_type": cand.source_metadata.get("scene_type", None),
                    "source_segment_range": cand.source_metadata.get("segment_range", cand.source_metadata.get("source_segment_range", None)),
                    "reverse_ratio": cand.tags.get("reverse_ratio", None) if isinstance(cand.tags, dict) else None,
                    "steer_change_rate": None,
                    "curvature_proxy": None,
                    "complexity_score": float(cand.complexity_score),
                    "novelty_score": float(cand.novelty_score),
                    "utility_score": float(cand.utility_score),
                    "delta_pose": [float(delta4[0]), float(delta4[1]), float(delta4[2])],
                    "enabled": True,
                    "usage_count": 0,
                    "success_usage_count": 0,
                    "unit": "physical",
                    "symmetry_source": variant_name,
                    "symmetry_parent_primitive_id": variant_parent_pid,
                }
                self._primitive_meta.append(meta)
                added += 1

                if variant_name == "original":
                    variant_parent_pid = pid

        # Update library in-place by saving to a temp file for re-loading.
        # IMPORTANT: do not advance active_version here; only save_version() does.
        if added > 0:
            tmp_id = f"tmp_{int(time.time())}"
            info = self._save_version_internal(
                version_id=tmp_id,
                parent_version_id=self.active_version_id,
                actions=actions,
                deltas=deltas,
                meta_list=self._primitive_meta,
                H=H,
                temporary=True,
            )
            self._pending_tmp_version = info
            self._library = PrimitiveLibrary(info.library_path)

        return int(added)

    def save_version(self, save_dir: str, version_id: Optional[str] = None) -> LibraryVersionInfo:
        """Persist current in-memory library as a new named version."""
        if self._library is None:
            raise RuntimeError("Library not loaded")

        if version_id is None:
            version_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())

        actions = np.asarray(self._library.actions, dtype=np.float64)
        deltas = np.asarray(self._library.deltas, dtype=np.float64)
        H = int(getattr(self._library, "horizon", actions.shape[1]))

        info = self._save_version_internal(
            version_id=str(version_id),
            parent_version_id=self._active_version.version_id if self._active_version else None,
            actions=actions,
            deltas=deltas,
            meta_list=self._primitive_meta,
            H=H,
            temporary=False,
        )
        self._active_version = info
        self._library = PrimitiveLibrary(info.library_path)
        self._pending_tmp_version = None
        self._write_active_version(save_dir, info.version_id)
        return info

    def rollback_to(self, version_id: str, save_dir: str) -> None:
        """Rollback active library to an existing version."""
        if self._version_dir is None:
            save_dir = os.path.abspath(save_dir)
            self._version_dir = os.path.join(save_dir, "adaptive_primitives", "versions")

        npz_path = os.path.join(self._version_dir, f"primitives_v{version_id}.npz")
        meta_path = os.path.join(self._version_dir, f"primitives_v{version_id}_meta.json")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)

        self._library = PrimitiveLibrary(npz_path)
        self._primitive_meta = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self._primitive_meta = json.load(f)

        info = LibraryVersionInfo(
            version_id=str(version_id),
            parent_version_id=None,
            created_time=time.time(),
            library_path=npz_path,
            meta_path=meta_path,
        )
        self._active_version = info
        self._pending_tmp_version = None
        self._write_active_version(save_dir, info.version_id)

    def export_stats(self) -> Dict[str, Any]:
        return {
            "active_version": self.active_version_id,
            "library_size": int(self.library_size),
            "enabled_count": int(sum(1 for m in self._primitive_meta if bool(m.get("enabled", True)))),
        }

    def get_meta(self) -> List[Dict[str, Any]]:
        return list(self._primitive_meta)

    # -------------------------
    # Internal persistence
    # -------------------------
    def _save_version_internal(
        self,
        version_id: str,
        parent_version_id: Optional[str],
        actions: np.ndarray,
        deltas: np.ndarray,
        meta_list: List[Dict[str, Any]],
        H: int,
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

        meta = {
            "H": int(H),
            "created_time": float(created_time),
            "parent_version": parent_version_id,
            "unit": "physical",
            "mask_index_path": mask_index_path,
            "ray_safety_path": ray_safety_path,
        }

        np.savez(npz_path, actions=actions, deltas=deltas, meta=meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False, indent=2)

        if not temporary:
            self._build_library_sidecars(
                npz_path=npz_path,
                actions=actions,
                deltas=deltas,
                H=H,
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
