import numpy as np
from types import SimpleNamespace
from shapely.geometry import Polygon

from env.car_parking_base import CarParking
from env.vehicle import State, Status


def _make_env(state: State, dest: State, obstacles):
    env = object.__new__(CarParking)
    env.vehicle = SimpleNamespace(state=state, boxes=state.create_box())
    env.map = SimpleNamespace(
        dest=dest,
        dest_box=dest.create_box()[0],
        obstacles=list(obstacles),
        xmin=-100.0,
        xmax=100.0,
        ymin=-100.0,
        ymax=100.0,
    )
    env.t = 0.0
    env.initial_dist = float(max(state.loc.distance(dest.loc), 1e-6))
    env.accum_arrive_reward = 0.0
    return env


def test_get_reward_reports_front_collision_part():
    state = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dest = State([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obstacle = SimpleNamespace(shape=Polygon(state.create_box()[0]).centroid.buffer(0.2))
    env = _make_env(state, dest, [obstacle])

    reward, done, info = CarParking.get_reward(env, np.zeros((2,), dtype=np.float64), prev_state=state)

    assert done is True
    assert info["status"] == Status.COLLIDED
    assert info["collision_part"] == "front"
    assert "front_overlap" in info
    assert "rear_overlap" in info
    assert "heading_error_deg" in info


def test_get_reward_reports_terminal_alignment_metrics():
    state = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dest = State([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = _make_env(state, dest, [])

    reward, done, info = CarParking.get_reward(env, np.zeros((2,), dtype=np.float64), prev_state=state)

    assert done is True
    assert info["status"] == Status.ARRIVED
    assert info["collision_part"] is None
    assert float(info["front_overlap"]) > 0.99
    assert float(info["rear_overlap"]) > 0.99
    assert float(info["mean_overlap"]) > 0.99
    assert abs(float(info["heading_error_deg"])) < 1e-6
