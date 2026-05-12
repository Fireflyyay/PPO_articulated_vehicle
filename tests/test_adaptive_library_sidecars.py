import os
from types import SimpleNamespace

import numpy as np

from primitives.adaptive_library_manager import AdaptivePrimitiveLibraryManager
from primitives.trajectory_miner import CandidatePrimitive


def _write_base_library(path):
    actions = np.array(
        [
            [[0.0, 1.0]],
            [[0.05, 1.0]],
        ],
        dtype=np.float64,
    )
    deltas = np.array(
        [
            [0.2, 0.0, 0.0, 0.0],
            [0.2, 0.02, 0.01, 0.01],
        ],
        dtype=np.float64,
    )
    rollout_states = np.zeros((2, 2, 6), dtype=np.float64)
    rollout_states[0, 1] = np.array([0.2, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    rollout_states[1, 1] = np.array([0.2, 0.02, 0.01, 0.0, 1.0, 0.05], dtype=np.float64)
    np.savez_compressed(
        path,
        schema_version=np.asarray("family_library_v1", dtype=object),
        actions=actions,
        deltas=deltas,
        rollout_states=rollout_states,
        variant_horizons=np.asarray([1, 1], dtype=np.int64),
        switch_indices=np.asarray([-1, -1], dtype=np.int64),
        durations=np.asarray([0.2, 0.2], dtype=np.float64),
        speed_signs=np.asarray([1, 1], dtype=np.int64),
        is_compound=np.asarray([0, 0], dtype=np.int8),
        variant_flat_to_gamma=np.asarray([0, 0], dtype=np.int64),
        variant_flat_to_family=np.asarray([0, 0], dtype=np.int64),
        variant_flat_to_variant=np.asarray([0, 1], dtype=np.int64),
        variant_flat_to_family_type=np.asarray(["normal", "normal"], dtype=object),
        variant_flat_to_mode=np.asarray(["normal", "normal"], dtype=object),
        gamma_bin_values=np.asarray([0.0], dtype=np.float64),
        family_names=np.asarray(["unit-forward"], dtype=object),
        family_types=np.asarray(["normal"], dtype=object),
        family_count=np.asarray(1, dtype=np.int64),
        variant_count_per_family=np.asarray(2, dtype=np.int64),
        index_table=np.asarray([[[0, 1]]], dtype=np.int64),
        variant_counts=np.asarray([[2]], dtype=np.int64),
        default_variant_table=np.asarray([[0]], dtype=np.int64),
        step_seconds=np.asarray(0.2, dtype=np.float64),
        meta=np.asarray(
            {
                "H": 1,
                "gamma_bins": 1,
                "variant_count": 2,
                "family_preset": "unit",
                "step_seconds": 0.2,
                "family_specs": [
                    {
                        "family_id": 0,
                        "name": "unit-forward",
                        "family_type": "normal",
                        "speed_sign": 1,
                        "speed_scale": 1.0,
                        "gamma_rate_scale": 0.0,
                        "mode": "normal",
                        "compound_split": None,
                        "compound_exit_gamma_scale": 0.0,
                    }
                ],
            },
            dtype=object,
        ),
    )


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