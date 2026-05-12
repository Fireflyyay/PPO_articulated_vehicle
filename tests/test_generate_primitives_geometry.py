import numpy as np

from configs import HITCH_OFFSET, TRAILER_LENGTH
from env.vehicle import State, Vehicle
from primitives.generate_primitives import generate_primitives


def _rollout_delta(actions: np.ndarray, gamma0: float) -> np.ndarray:
    vehicle = Vehicle(
        articulated=True,
        trailer_length=TRAILER_LENGTH,
        hitch_offset=HITCH_OFFSET,
    )
    vehicle.reset(State([0.0, 0.0, 0.0, 0.0, 0.0, -float(gamma0)]))

    for action in np.asarray(actions, dtype=np.float64):
        vehicle.step(action, step_time=1)

    final_state = vehicle.state
    gamma = final_state.heading - final_state.rear_heading
    gamma = (gamma + np.pi) % (2.0 * np.pi) - np.pi
    return np.array(
        [final_state.loc.x, final_state.loc.y, final_state.heading, gamma],
        dtype=np.float64,
    )


def test_generate_primitives_uses_training_geometry(tmp_path):
    output_path = tmp_path / "primitives_test.npz"

    generate_primitives(H=4, S=11, output_path=str(output_path))

    data = np.load(output_path, allow_pickle=True)
    actions = np.asarray(data["actions"], dtype=np.float64)
    deltas = np.asarray(data["deltas"], dtype=np.float64)
    variant_flat_to_gamma = np.asarray(data["variant_flat_to_gamma"], dtype=np.int64)
    gamma_bin_values = np.asarray(data["gamma_bin_values"], dtype=np.float64)
    rollout_states = np.asarray(data["rollout_states"], dtype=np.float64)
    meta = data["meta"].item()

    assert meta["trailer_length"] == float(TRAILER_LENGTH)
    assert meta["hitch_offset"] == float(HITCH_OFFSET)

    primitive_id = int(np.argmax(np.abs(deltas[:, 2])))
    gamma0 = float(gamma_bin_values[int(variant_flat_to_gamma[primitive_id])])
    expected_delta = _rollout_delta(actions[primitive_id], gamma0=gamma0)

    np.testing.assert_allclose(deltas[primitive_id], expected_delta, atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(rollout_states[primitive_id, -1, :4], np.array([
        expected_delta[0],
        expected_delta[1],
        expected_delta[2],
        expected_delta[2] - expected_delta[3],
    ], dtype=np.float64), atol=1e-9, rtol=0.0)