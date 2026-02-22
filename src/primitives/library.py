import numpy as np


class PrimitiveLibrary:
    def __init__(self, npz_path):
        self.npz_path = npz_path
        data = np.load(npz_path, allow_pickle=True)
        self.actions = data['actions'] # [N, H, 2]
        self.deltas = data['deltas']   # [N, 4]
        # handling meta if present
        self.meta = {}
        if 'meta' in data:
            try:
                self.meta = data['meta'].item() or {}
            except Exception:
                self.meta = {}

        # Optional offline grid index (for fast terminal takeover pruning).
        # Best-effort load; training should still run without it.
        self.grid_index = None
        try:
            from primitives.primitive_index import try_load_index_for_library

            explicit = None
            if isinstance(self.meta, dict):
                explicit = self.meta.get('index_path', None)
            self.grid_index = try_load_index_for_library(self.npz_path, explicit_index_path=explicit)
        except Exception:
            self.grid_index = None
        
    @property
    def size(self):
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        """Primitive horizon length H."""
        if isinstance(self.meta, dict) and 'H' in self.meta:
            try:
                return int(self.meta['H'])
            except Exception:
                pass
        return int(self.actions.shape[1])

    def get_actions(self, primitive_id):
        return self.actions[primitive_id]
        
    def get_delta(self, primitive_id):
        return self.deltas[primitive_id]

def load_library(npz_path):
    return PrimitiveLibrary(npz_path)
