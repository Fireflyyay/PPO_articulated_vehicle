from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class PrimitiveFamilySpec:
    family_id: int
    name: str
    family_type: str
    speed_sign: int
    speed_scale: float
    gamma_rate_scale: float
    mode: str = "normal"
    mode_id: int = 0
    semantic_slot_id: int = -1
    semantic_slot_name: str = ""
    semantic_group: str = "normal"
    pattern: str = "single"
    horizon_scale: float = 1.0
    secondary_speed_sign: int = 0
    secondary_gamma_rate_scale: float = 0.0
    recover_gain: float = 0.0
    recover_bias_scale: float = 0.0
    articulation_target: float = 0.0
    compound_split: Optional[float] = None
    compound_exit_gamma_scale: float = 0.0
    motion_family_id: int = -1
    motion_family_name: str = ""
    speed_level_id: int = 0
    speed_level_name: str = "full"
    speed_level_scale: float = 1.0


@dataclass(frozen=True)
class PrimitiveVariantRef:
    flat_index: int
    gamma_bin_id: int
    family_id: int
    variant_id: int


@dataclass
class PrimitiveVariant:
    ref: PrimitiveVariantRef
    actions: np.ndarray
    delta: np.ndarray
    rollout_states: np.ndarray
    effective_horizon: int
    mode: str
    family_type: str
    duration: float
    speed_sign: int
    is_compound: bool = False
    switch_index: int = -1
    meta: Dict = field(default_factory=dict)


@dataclass
class PrimitiveFamilyLibraryMeta:
    family_names: List[str]
    family_types: List[str]
    gamma_bin_values: np.ndarray
    variant_count_per_family: int
    horizon: int
    step_seconds: float
    mode_names: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


@dataclass
class Primitive:
    """Backward-compatible flat primitive view over a resolved variant."""

    id: int
    actions: np.ndarray
    delta: np.ndarray
    meta: Dict = field(default_factory=dict)
