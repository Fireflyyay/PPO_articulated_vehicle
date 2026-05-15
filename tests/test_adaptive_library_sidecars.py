import os
from types import SimpleNamespace

import numpy as np

from primitives.adaptive_library_manager import AdaptivePrimitiveLibraryManager
from primitives.primitive_index import build_mask_index_from_library, primitive_grid_index_to_payload
from primitives.primitive_ray_safety import build_ray_safety_index_from_library, save_ray_safety_index
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


def _write_base_sidecars(path):
    data = np.load(path, allow_pickle=True)
    actions = np.asarray(data["actions"], dtype=np.float64)
    deltas = np.asarray(data["deltas"], dtype=np.float64)
    horizon = int(actions.shape[1])

    mask_index = build_mask_index_from_library(
        actions=actions,
        grid_resolution=0.6,
        x_min=-6.0,
        x_max=12.0,
        y_min=-9.0,
        y_max=9.0,
        sample_stride=1,
        num_step=4,
        group_prefix_steps=max(1, min(horizon, int(round(float(horizon) * 0.3)))),
    )
    np.savez_compressed(
        os.path.splitext(path)[0] + ".mask_index.npz",
        **primitive_grid_index_to_payload(mask_index),
    )

    ray_index = build_ray_safety_index_from_library(
        actions=actions,
        deltas=deltas,
        lidar_num=120,
        lidar_range=30.0,
        safety_margin=0.25,
        reverse_margin_scale=1.2,
        sample_stride=2,
        num_step=4,
    )
    save_ray_safety_index(os.path.splitext(path)[0] + ".ray_safety.npz", ray_index)


def test_adaptive_library_load_reuses_base_sidecars_without_persisting_base_version(tmp_path):
    base_path = tmp_path / "base_primitives.npz"
    _write_base_library(base_path)
    _write_base_sidecars(base_path)

    mgr = AdaptivePrimitiveLibraryManager(verbose=False)
    mgr.load(base_path=str(base_path), save_dir=str(tmp_path))

    base_lib = mgr.get_active_library()
    versions_dir = tmp_path / "adaptive_primitives" / "versions"

    assert base_lib.npz_path == str(base_path)
    assert base_lib.grid_index is not None
    assert base_lib.ray_safety_index is not None
    assert not any(versions_dir.glob("primitives_vbase*"))


def test_adaptive_library_persistent_versions_generate_sidecars(tmp_path):
    base_path = tmp_path / "base_primitives.npz"
    _write_base_library(base_path)
    _write_base_sidecars(base_path)

    mgr = AdaptivePrimitiveLibraryManager(verbose=False)
    mgr.load(base_path=str(base_path), save_dir=str(tmp_path))

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