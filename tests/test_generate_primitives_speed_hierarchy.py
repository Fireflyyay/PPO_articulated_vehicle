import numpy as np

from configs import NUM_STEP
from primitives.generate_primitives import generate_primitives


def test_generate_primitives_exposes_explicit_speed_hierarchy(tmp_path):
    output_path = tmp_path / "primitives_speed_hierarchy.npz"

    generate_primitives(H=3, S=11, output_path=str(output_path), gamma_bins=5, variant_count=3, family_preset="main")

    data = np.load(output_path, allow_pickle=True)
    family_count = int(data["family_count"])
    motion_family_count = int(data["motion_family_count"])
    speed_level_count = int(data["speed_level_count"])
    family_to_motion_family = np.asarray(data["family_to_motion_family"], dtype=np.int64)
    family_to_speed_level = np.asarray(data["family_to_speed_level"], dtype=np.int64)
    speed_level_names = {str(v) for v in np.asarray(data["speed_level_names"], dtype=object).tolist()}

    assert speed_level_count == 3
    assert speed_level_names == {"stop", "mid", "full"}
    assert family_count == motion_family_count * speed_level_count
    assert family_to_motion_family.shape == (family_count,)
    assert family_to_speed_level.shape == (family_count,)


def test_stop_and_mid_speed_families_preserve_family_mechanism(tmp_path):
    output_path = tmp_path / "primitives_speed_behaviour.npz"

    generate_primitives(H=3, S=11, output_path=str(output_path), gamma_bins=5, variant_count=3, family_preset="main")

    data = np.load(output_path, allow_pickle=True)
    actions = np.asarray(data["actions"], dtype=np.float64)
    rollout_states = np.asarray(data["rollout_states"], dtype=np.float64)
    gamma_bin_values = np.asarray(data["gamma_bin_values"], dtype=np.float64)
    index_table = np.asarray(data["index_table"], dtype=np.int64)
    family_to_motion_family = np.asarray(data["family_to_motion_family"], dtype=np.int64)
    family_to_speed_level = np.asarray(data["family_to_speed_level"], dtype=np.int64)
    family_types = np.asarray(data["family_types"], dtype=object)

    gamma_bin_id = int(np.argmin(np.abs(gamma_bin_values)))
    variant_id = int(index_table.shape[-1] // 2)

    selected = None
    for motion_family_id in range(int(data["motion_family_count"])):
        family_ids = {}
        for family_id in np.where(family_to_motion_family == motion_family_id)[0].tolist():
            family_ids[int(family_to_speed_level[family_id])] = int(family_id)
        if set(family_ids.keys()) != {0, 1, 2}:
            continue

        stop_pid = int(index_table[gamma_bin_id, family_ids[0], variant_id])
        mid_pid = int(index_table[gamma_bin_id, family_ids[1], variant_id])
        full_pid = int(index_table[gamma_bin_id, family_ids[2], variant_id])
        if min(stop_pid, mid_pid, full_pid) < 0:
            continue
        if str(family_types[family_ids[2]]) != "normal":
            continue
        if np.max(np.abs(actions[stop_pid, :, 0])) <= 1e-6:
            continue
        selected = (stop_pid, mid_pid, full_pid)
        break

    assert selected is not None
    stop_pid, mid_pid, full_pid = selected

    stop_actions = actions[stop_pid]
    mid_actions = actions[mid_pid]
    full_actions = actions[full_pid]

    assert np.allclose(stop_actions[:, 1], 0.0)
    assert np.max(np.abs(stop_actions[:, 0])) > 1e-6
    np.testing.assert_allclose(np.mean(np.abs(mid_actions[:, 1])) * float(NUM_STEP), np.mean(np.abs(full_actions[:, 1])), atol=1e-9, rtol=0.0)

    stop_final = rollout_states[stop_pid, -1]
    assert abs(float(stop_final[0])) < 1e-9
    assert abs(float(stop_final[1])) < 1e-9
    assert abs(float(stop_final[2]) - float(stop_final[3])) > 1e-6 or abs(float(stop_final[2])) > 1e-6