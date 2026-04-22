import numpy as np

from primitives.library import PrimitiveLibrary
from primitives.primitive_index import (
    build_primitive_swept_cells,
    load_primitive_grid_index,
    primitive_grid_index_to_payload,
)


def test_build_swept_cells_distinguishes_forward_and_reverse_paths():
    actions = np.array(
        [
            [[0.0, 2.0]],
            [[0.0, -2.0]],
        ],
        dtype=np.float64,
    )

    index = build_primitive_swept_cells(
        actions=actions,
        grid_resolution=1.0,
        x_min=-6.0,
        x_max=6.0,
        y_min=-3.0,
        y_max=3.0,
        sample_stride=1,
        num_step=20,
        group_prefix_steps=1,
    )

    ahead_cell = index.world_to_cell(2.2, 0.0)
    cells_forward = {tuple(cell) for cell in index.primitive_to_cells[0].tolist()}
    cells_reverse = {tuple(cell) for cell in index.primitive_to_cells[1].tolist()}

    assert index.index_kind == "swept_cells"
    assert ahead_cell in cells_forward
    assert ahead_cell not in cells_reverse
    assert index.fast_prune_primitives({ahead_cell}).astype(np.int64).tolist() == [0, 1]


def test_mask_index_round_trip_and_library_auto_load(tmp_path):
    actions = np.array(
        [
            [[0.0, 2.0]],
            [[0.0, -2.0]],
        ],
        dtype=np.float64,
    )
    deltas = np.zeros((actions.shape[0], 4), dtype=np.float64)

    library_path = tmp_path / "demo_library.npz"
    np.savez_compressed(library_path, actions=actions, deltas=deltas)

    index = build_primitive_swept_cells(
        actions=actions,
        grid_resolution=1.0,
        x_min=-6.0,
        x_max=6.0,
        y_min=-3.0,
        y_max=3.0,
        sample_stride=1,
        num_step=20,
        group_prefix_steps=1,
    )
    mask_index_path = tmp_path / "demo_library.mask_index.npz"
    np.savez_compressed(mask_index_path, **primitive_grid_index_to_payload(index))

    loaded = load_primitive_grid_index(str(mask_index_path))
    library = PrimitiveLibrary(str(library_path))

    assert loaded.index_kind == "swept_cells"
    assert loaded.primitive_envelope_bounds.shape == (2, 4)
    assert library.grid_index is not None
    assert library.mask_index is not None
    assert library.grid_index.index_kind == "swept_cells"