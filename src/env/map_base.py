
import numpy as np
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, MultiPolygon

class Area(object):
    def __init__(
        self,
        shape: BaseGeometry = None, 
        subtype: str = None,
        color: float = None,
    ):
        self.shape = shape
        self.subtype = subtype
        self.color = color

    def get_shape(self):
        # Prefer .coords when available (e.g., LinearRing/LineString)
        if hasattr(self.shape, "coords"):
            return np.array(self.shape.coords)

        # Polygon-like: fall back to exterior ring for visualization
        if isinstance(self.shape, Polygon):
            return np.array(self.shape.exterior.coords)
        if isinstance(self.shape, MultiPolygon):
            # Return the largest component's exterior (best-effort)
            poly = max(list(self.shape.geoms), key=lambda g: float(g.area), default=None)
            if poly is None:
                return np.zeros((0, 2), dtype=float)
            return np.array(poly.exterior.coords)

        raise TypeError(f"Unsupported shape type for get_shape: {type(self.shape)}")
