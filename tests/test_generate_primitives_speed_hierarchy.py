import numpy as np

from configs import VALID_SPEED, VALID_STEER
from primitives.generate_primitives import generate_primitives


def test_generate_primitives_exposes_simple_family_grid(tmp_path):
    output_path = tmp_path / "primitives_simple_grid.npz"

    generate_primitives(H=3, S=11, output_path=str(output_path), gamma_bins=5, variant_count=3, family_preset="main")

    data = np.load(output_path, allow_pickle=True)
    family_count = int(data["family_count"])
    motion_family_count = int(data["motion_family_count"])
    speed_level_count = int(data["speed_level_count"])
    family_to_motion_family = np.asarray(data["family_to_motion_family"], dtype=np.int64)
    family_to_speed_level = np.asarray(data["family_to_speed_level"], dtype=np.int64)
    speed_level_names = {str(v) for v in np.asarray(data["speed_level_names"], dtype=object).tolist()}
    family_types = {str(v) for v in np.asarray(data["family_types"], dtype=object).tolist()}

    assert speed_level_count == 1
    assert speed_level_names == {"full"}
    assert family_count == motion_family_count == 44
    assert family_to_motion_family.shape == (family_count,)
    assert family_to_speed_level.shape == (family_count,)
    np.testing.assert_array_equal(family_to_motion_family, np.arange(family_count, dtype=np.int64))
    np.testing.assert_array_equal(family_to_speed_level, np.zeros((family_count,), dtype=np.int64))
    assert family_types == {"normal"}


def test_generated_simple_families_match_old_control_grid(tmp_path):
    output_path = tmp_path / "primitives_simple_controls.npz"

    generate_primitives(H=3, S=11, output_path=str(output_path), gamma_bins=5, variant_count=3, family_preset="main")

    data = np.load(output_path, allow_pickle=True)
    actions = np.asarray(data["actions"], dtype=np.float64)
    index_table = np.asarray(data["index_table"], dtype=np.int64)
    family_count = int(data["family_count"])

    gamma_bin_values = np.asarray(data["gamma_bin_values"], dtype=np.float64)
    gamma_bin_id = int(np.argmin(np.abs(gamma_bin_values)))
    variant_id = int(index_table.shape[-1] // 2)

    selected_actions = []
    for family_id in range(family_count):
        primitive_id = int(index_table[gamma_bin_id, family_id, variant_id])
        assert primitive_id >= 0
        primitive_actions = np.asarray(actions[primitive_id], dtype=np.float64)
        np.testing.assert_allclose(primitive_actions, np.repeat(primitive_actions[:1], primitive_actions.shape[0], axis=0), atol=1e-9, rtol=0.0)
        selected_actions.append(primitive_actions[0])

    selected_actions = np.asarray(selected_actions, dtype=np.float64)
    unique_controls = np.unique(np.round(selected_actions, decimals=6), axis=0)
    expected_steers = np.linspace(float(VALID_STEER[0]), float(VALID_STEER[1]), 11, dtype=np.float64)
    expected_speeds = np.asarray([-float(max(abs(float(VALID_SPEED[0])), abs(float(VALID_SPEED[1])))), -1.0, 1.0, float(max(abs(float(VALID_SPEED[0])), abs(float(VALID_SPEED[1]))))], dtype=np.float64)

    assert unique_controls.shape == (44, 2)
    np.testing.assert_allclose(np.unique(np.round(unique_controls[:, 0], decimals=6)), np.round(expected_steers, decimals=6), atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(np.unique(np.round(unique_controls[:, 1], decimals=6)), np.round(expected_speeds, decimals=6), atol=1e-9, rtol=0.0)