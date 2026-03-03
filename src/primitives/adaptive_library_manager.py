from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

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

    def add_candidates(self, candidates: Sequence[CandidatePrimitive], round_id: int, config=None) -> int:
        """Append candidates to library in-memory (not persisted until save_version())."""
        if self._library is None:
            raise RuntimeError("Library not loaded")

        if len(candidates) == 0:
            return 0

        actions = np.asarray(self._library.actions, dtype=np.float64)
        deltas = np.asarray(self._library.deltas, dtype=np.float64)
        H = int(getattr(self._library, "horizon", actions.shape[1]))

        added = 0
        for cand in candidates:
            u = np.asarray(cand.actions_resampled, dtype=np.float64)
            if u.shape != (H, 2):
                continue

            # delta_pose / deltas
            dx, dy, dyaw = 0.0, 0.0, 0.0
            if cand.delta_feature is not None and len(cand.delta_feature) >= 3:
                dx, dy, dyaw = map(float, cand.delta_feature[:3])
            delta4 = np.asarray([dx, dy, dyaw, 0.0], dtype=np.float64)

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
                "delta_pose": [dx, dy, dyaw],
                "enabled": True,
                "usage_count": 0,
                "success_usage_count": 0,
                "unit": "physical",
            }
            self._primitive_meta.append(meta)
            added += 1

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

        meta = {
            "H": int(H),
            "created_time": float(created_time),
            "parent_version": parent_version_id,
            "unit": "physical",
        }

        np.savez(npz_path, actions=actions, deltas=deltas, meta=meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False, indent=2)

        return LibraryVersionInfo(
            version_id=str(version_id),
            parent_version_id=parent_version_id,
            created_time=created_time,
            library_path=npz_path,
            meta_path=meta_path,
        )

    def _write_active_version(self, save_dir: str, version_id: str) -> None:
        save_dir = os.path.abspath(save_dir)
        root = os.path.join(save_dir, "adaptive_primitives")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, "active_version.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version_id": str(version_id)}, f, ensure_ascii=False, indent=2)
