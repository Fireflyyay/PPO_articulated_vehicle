import os
from types import SimpleNamespace

import numpy as np

from primitives.adaptive_library_manager import AdaptivePrimitiveLibraryManager
from primitives.trajectory_miner import CandidatePrimitive


def _write_base_library(path):
    actions = np.array(
        [
            [[0.0, 1.0]],
            [[0.0, -1.0]],
        ],
        dtype=np.float64,
    )
    deltas = np.zeros((actions.shape[0], 4), dtype=np.float64)
    np.savez_compressed(path, actions=actions, deltas=deltas)


def test_adaptive_library_persistent_versions_generate_sidecars(tmp_path):
    base_path = tmp_path / "base_primitives.npz"
    _write_base_library(base_path)

    mgr = AdaptivePrimitiveLibraryManager(verbose=False)
    mgr.load(base_path=str(base_path), save_dir=str(tmp_path))

    base_lib = mgr.get_active_library()
    base_npz = os.path.splitext(base_lib.npz_path)[0]

    assert os.path.exists(base_npz + ".mask_index.npz")
    assert os.path.exists(base_npz + ".ray_safety.npz")
    assert base_lib.grid_index is not None
    assert base_lib.ray_safety_index is not None

    candidate = CandidatePrimitive(
        actions_raw=np.array([[0.1, 1.0]], dtype=np.float64),
        actions_resampled=np.array([[0.1, 1.0]], dtype=np.float64),
        delta_feature=np.zeros((3,), dtype=np.float64),
        source_metadata={},
        tags={},
    )

    added = mgr.add_candidates([candidate], round_id=1)
    info = mgr.save_version(save_dir=str(tmp_path), version_id="unit")
    new_lib = mgr.get_active_library()
    new_npz = os.path.splitext(info.library_path)[0]

    assert added == 2
    assert os.path.exists(new_npz + ".mask_index.npz")
    assert os.path.exists(new_npz + ".ray_safety.npz")
    assert new_lib.size == 4
    assert new_lib.grid_index is not None
    assert new_lib.ray_safety_index is not None
    assert np.allclose(new_lib.actions[2], np.array([[0.1, 1.0]], dtype=np.float64))
    assert np.allclose(new_lib.actions[3], np.array([[-0.1, 1.0]], dtype=np.float64))


def test_adaptive_library_can_disable_symmetric_augmentation(tmp_path):
    base_path = tmp_path / "base_primitives.npz"
    _write_base_library(base_path)

    mgr = AdaptivePrimitiveLibraryManager(verbose=False)
    mgr.load(base_path=str(base_path), save_dir=str(tmp_path))

    candidate = CandidatePrimitive(
        actions_raw=np.array([[0.1, 1.0]], dtype=np.float64),
        actions_resampled=np.array([[0.1, 1.0]], dtype=np.float64),
        delta_feature=np.zeros((3,), dtype=np.float64),
        source_metadata={},
        tags={},
    )

    cfg = SimpleNamespace(AP_AUTO_ADD_SYMMETRIC_PRIMITIVES=False)
    added = mgr.add_candidates([candidate], round_id=1, config=cfg)

    assert added == 1
    assert mgr.get_active_library().size == 3