import numpy as np

from configs import HITCH_OFFSET, TRAILER_LENGTH
from env.vehicle import State, Vehicle
from primitives.generate_primitives import generate_primitives


def _rollout_delta(actions: np.ndarray) -> np.ndarray:
    vehicle = Vehicle(
        articulated=True,
        trailer_length=TRAILER_LENGTH,
        hitch_offset=HITCH_OFFSET,
    )
    vehicle.reset(State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

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
    meta = data["meta"].item()

    assert meta["trailer_length"] == float(TRAILER_LENGTH)
    assert meta["hitch_offset"] == float(HITCH_OFFSET)

    primitive_id = int(np.argmax(np.abs(deltas[:, 2])))
    expected_delta = _rollout_delta(actions[primitive_id])

    np.testing.assert_allclose(deltas[primitive_id], expected_delta, atol=1e-9, rtol=0.0)