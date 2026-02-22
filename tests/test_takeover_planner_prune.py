import numpy as np

from primitives.primitive_index import PrimitiveGridIndex


def _make_index():
    # 3 primitives, 2 groups
    primitive_to_cells = [
        np.array([[0, 0], [1, 0]], dtype=np.int64),
        np.array([[0, 0]], dtype=np.int64),
        np.array([[1, 1]], dtype=np.int64),
    ]
    cell_to_primitives = {
        (0, 0): np.array([0, 1], dtype=np.int64),
        (1, 0): np.array([0], dtype=np.int64),
        (1, 1): np.array([2], dtype=np.int64),
    }
    primitive_to_group_id = np.array([0, 0, 1], dtype=np.int64)
    group_to_primitive_ids = [
        np.array([0, 1], dtype=np.int64),
        np.array([2], dtype=np.int64),
    ]
    group_prefix_steps = np.array([2, 2], dtype=np.int64)

    return PrimitiveGridIndex(
        grid_resolution=0.5,
        x_min=-1.0,
        y_min=-1.0,
        x_max=2.0,
        y_max=2.0,
        primitive_to_cells=primitive_to_cells,
        cell_to_primitives=cell_to_primitives,
        primitive_to_group_id=primitive_to_group_id,
        group_to_primitive_ids=group_to_primitive_ids,
        group_prefix_steps=group_prefix_steps,
    )


def test_fast_prune_blocks_by_occupied_cells():
    index = _make_index()

    occupied = {(0, 0)}
    candidate_mask = index.fast_prune_primitives(occupied)

    assert candidate_mask.shape == (3,)
    # primitives 0 and 1 touch (0,0) => blocked
    assert candidate_mask.tolist() == [False, False, True]


def test_near_hit_counts_follow_inverted_index():
    index = _make_index()

    occupied = {(0, 0), (1, 1)}
    counts = index.count_near_hits(occupied)

    assert counts.tolist() == [1, 1, 1]
