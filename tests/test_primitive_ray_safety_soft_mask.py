import numpy as np

from primitives.primitive_ray_safety import (
    PrimitiveRaySafetyIndex,
    library_signature,
    load_ray_safety_index,
    save_ray_safety_index,
)


def test_ray_safety_soft_mask_uses_prefix_safe_length():
    dist_star = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            [[1.0, 1.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]],
        ],
        dtype=np.float32,
    )
    index = PrimitiveRaySafetyIndex(
        dist_star=dist_star,
        lidar_range=10.0,
        lidar_num=2,
        horizon=4,
    )

    mask, debug = index.compute_soft_mask(np.array([0.31, 0.31]), gamma=1.0, eps=0.01, lidar_range=10.0)

    assert mask.shape == (2,)
    assert mask[0] == np.float32(0.75)
    assert mask[1] == np.float32(0.25)
    assert debug["positive_step_count"] == 2


def test_ray_safety_round_trip_and_compatibility(tmp_path):
    actions = np.zeros((2, 4, 2), dtype=np.float64)
    deltas = np.zeros((2, 4), dtype=np.float64)
    sig = library_signature(actions, deltas)
    index = PrimitiveRaySafetyIndex(
        dist_star=np.ones((2, 4, 8), dtype=np.float32),
        lidar_range=30.0,
        lidar_num=8,
        horizon=4,
        library_signature=sig,
    )

    path = tmp_path / "demo.ray_safety.npz"
    save_ray_safety_index(str(path), index)
    loaded = load_ray_safety_index(str(path))

    assert loaded.compatible_with(actions, deltas)
    assert loaded.dist_star.shape == (2, 4, 8)
