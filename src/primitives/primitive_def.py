import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class Primitive:
    """
    Represents a motion primitive action sequence.
    """
    id: int
    actions: np.ndarray  # shape [H, 2] (steer, speed)
    delta: np.ndarray    # shape [4] (x, y, theta, gamma)
    meta: Dict = field(default_factory=dict)
