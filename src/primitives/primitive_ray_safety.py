from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def default_ray_safety_path(npz_path: str) -> str:
    base, _ = os.path.splitext(str(npz_path))
    return base + ".ray_safety.npz"


def _wrap_pi_array(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def library_signature(actions: np.ndarray, deltas: np.ndarray) -> str:
    hasher = hashlib.sha1()
    actions_arr = np.ascontiguousarray(np.asarray(actions))
    deltas_arr = np.ascontiguousarray(np.asarray(deltas))
    hasher.update(str(actions_arr.shape).encode("utf-8"))
    hasher.update(str(deltas_arr.shape).encode("utf-8"))
    hasher.update(actions_arr.view(np.uint8))
    hasher.update(deltas_arr.view(np.uint8))
    return hasher.hexdigest()


@dataclass
class PrimitiveRaySafetyIndex:
    """HOPE-like primitive safety table indexed by lidar rays.

    dist_star[primitive, primitive_step, ray] stores the required free distance
    beyond the lidar vehicle boundary for the swept front+rear vehicle footprint.
    Online use is deliberately just array comparison and prefix-length reduction.
    """

    dist_star: np.ndarray
    lidar_range: float
    lidar_num: int
    horizon: int
    library_signature: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def num_primitives(self) -> int:
        return int(self.dist_star.shape[0])

    def compatible_with(self, actions: np.ndarray, deltas: np.ndarray) -> bool:
        if self.num_primitives != int(np.asarray(actions).shape[0]):
            return False
        if int(self.horizon) != int(np.asarray(actions).shape[1]):
            return False
        expected = str(self.library_signature or "")
        if not expected:
            return True
        return expected == library_signature(actions, deltas)

    def compute_soft_mask(
        self,
        lidar_norm: np.ndarray,
        gamma: float,
        eps: float,
        lidar_range: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        lidar_range = float(self.lidar_range if lidar_range is None else lidar_range)
        lidar = np.asarray(lidar_norm, dtype=np.float64).reshape(-1)[: int(self.lidar_num)]
        if lidar.size < int(self.lidar_num):
            padded = np.ones((int(self.lidar_num),), dtype=np.float64)
            padded[: lidar.size] = lidar
            lidar = padded

        dist_obs = np.clip(lidar, 0.0, 1.0) * lidar_range
        dist_star = np.asarray(self.dist_star, dtype=np.float32)
        ray_count = min(int(dist_star.shape[2]), int(dist_obs.shape[0]))
        safe_by_ray = dist_star[:, :, :ray_count] <= dist_obs[:ray_count][None, None, :]
        safe_step = np.all(safe_by_ray, axis=2)
        prefix_safe = np.cumprod(safe_step.astype(np.int8), axis=1)
        step_len = np.sum(prefix_safe, axis=1).astype(np.float32)
        horizon = max(float(self.horizon), 1.0)
        soft = np.power(np.clip(step_len / horizon, 0.0, 1.0), float(gamma))
        soft = np.clip(soft, float(eps), 1.0).astype(np.float32)
        debug = {
            "positive_step_count": int(np.count_nonzero(step_len > 0.0)),
            "safe_step_len_mean": float(np.mean(step_len)) if step_len.size else 0.0,
            "safe_step_len_min": float(np.min(step_len)) if step_len.size else 0.0,
            "safe_step_len_max": float(np.max(step_len)) if step_len.size else 0.0,
            "safe_step_lens": np.asarray(step_len, dtype=np.int64),
            "soft_mask_min": float(np.min(soft)) if soft.size else 0.0,
            "soft_mask_max": float(np.max(soft)) if soft.size else 0.0,
            "soft_mask_mean": float(np.mean(soft)) if soft.size else 0.0,
        }
        return soft, debug


def save_ray_safety_index(path: str, index: PrimitiveRaySafetyIndex) -> None:
    payload = {
        "dist_star": np.asarray(index.dist_star, dtype=np.float32),
        "lidar_range": np.asarray(float(index.lidar_range), dtype=np.float64),
        "lidar_num": np.asarray(int(index.lidar_num), dtype=np.int64),
        "horizon": np.asarray(int(index.horizon), dtype=np.int64),
        "library_signature": np.asarray(str(index.library_signature), dtype=object),
        "metadata": np.asarray(dict(index.metadata or {}), dtype=object),
    }
    np.savez_compressed(path, **payload)


def load_ray_safety_index(path: str) -> PrimitiveRaySafetyIndex:
    data = np.load(path, allow_pickle=True)
    metadata = {}
    if "metadata" in data:
        try:
            metadata = data["metadata"].item() or {}
        except Exception:
            metadata = {}
    library_sig = ""
    if "library_signature" in data:
        try:
            library_sig = str(data["library_signature"].item())
        except Exception:
            library_sig = ""
    return PrimitiveRaySafetyIndex(
        dist_star=np.asarray(data["dist_star"], dtype=np.float32),
        lidar_range=float(data["lidar_range"]),
        lidar_num=int(data["lidar_num"]),
        horizon=int(data["horizon"]),
        library_signature=library_sig,
        metadata=metadata,
    )


def try_load_ray_safety_for_library(npz_path: str, actions: np.ndarray, deltas: np.ndarray, explicit_path: Optional[str] = None):
    candidates = []
    if explicit_path:
        candidates.append(str(explicit_path))
    candidates.append(default_ray_safety_path(npz_path))

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            index = load_ray_safety_index(path)
            if index.compatible_with(actions, deltas):
                return index
        except Exception:
            continue
    return None


def _ray_distances_to_polygon_extent(poly, ray_angles: np.ndarray, ray_length: float) -> np.ndarray:
    from shapely.geometry import LineString, Point

    origin = Point(0.0, 0.0)
    out = np.zeros((ray_angles.shape[0],), dtype=np.float32)
    if poly is None or poly.is_empty:
        return out

    for ray_idx, angle in enumerate(ray_angles):
        ray = LineString(
            [
                (0.0, 0.0),
                (float(ray_length) * math.cos(float(angle)), float(ray_length) * math.sin(float(angle))),
            ]
        )
        inter = poly.intersection(ray)
        if inter.is_empty:
            continue

        distances = []
        geom_type = getattr(inter, "geom_type", "")
        if geom_type == "Point":
            distances.append(float(origin.distance(inter)))
        elif geom_type == "LineString":
            coords = list(inter.coords)
            if coords:
                distances.extend(math.hypot(float(x), float(y)) for x, y in coords)
        elif hasattr(inter, "geoms"):
            for geom in inter.geoms:
                if getattr(geom, "geom_type", "") == "Point":
                    distances.append(float(origin.distance(geom)))
                elif getattr(geom, "geom_type", "") == "LineString":
                    coords = list(geom.coords)
                    if coords:
                        distances.extend(math.hypot(float(x), float(y)) for x, y in coords)
        if distances:
            out[ray_idx] = float(max(distances))
    return out


def _vehicle_boundary(lidar_num: int, lidar_range: float) -> np.ndarray:
    from env.vehicle import VehicleBox
    from shapely.geometry import LineString, Point

    origin = Point(0.0, 0.0)
    angles = np.linspace(0.0, 2.0 * math.pi, int(lidar_num), endpoint=False)
    boundary = np.zeros((int(lidar_num),), dtype=np.float32)
    for idx, angle in enumerate(angles):
        ray = LineString(
            [
                (0.0, 0.0),
                (float(lidar_range) * math.cos(float(angle)), float(lidar_range) * math.sin(float(angle))),
            ]
        )
        inter = ray.intersection(VehicleBox)
        if inter.is_empty:
            continue
        if getattr(inter, "geom_type", "") == "Point":
            boundary[idx] = float(origin.distance(inter))
        elif hasattr(inter, "geoms"):
            dists = [float(origin.distance(g)) for g in inter.geoms if getattr(g, "geom_type", "") == "Point"]
            if dists:
                boundary[idx] = float(min(dists))
    return boundary


def build_ray_safety_index_from_library(
    actions: np.ndarray,
    deltas: np.ndarray,
    lidar_num: int,
    lidar_range: float,
    safety_margin: float,
    reverse_margin_scale: float,
    sample_stride: int,
    num_step: int,
) -> PrimitiveRaySafetyIndex:
    from shapely.ops import unary_union

    from primitives.primitive_index import create_articulated_boxes_canonical, simulate_primitive_states_canonical

    actions = np.asarray(actions, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    n_primitives = int(actions.shape[0])
    horizon = int(actions.shape[1])
    lidar_num = int(lidar_num)
    lidar_range = float(lidar_range)
    sample_stride = max(1, int(sample_stride))
    num_step = max(1, int(num_step))

    ray_angles = np.linspace(0.0, 2.0 * math.pi, lidar_num, endpoint=False)
    boundary = _vehicle_boundary(lidar_num=lidar_num, lidar_range=lidar_range)
    dist_star = np.zeros((n_primitives, horizon, lidar_num), dtype=np.float32)
    sim_steps_per_primitive_step = int(num_step) * 20
    ray_length = lidar_range + float(np.max(boundary)) + safety_margin + 5.0

    for primitive_id in range(n_primitives):
        states = simulate_primitive_states_canonical(actions[primitive_id], num_step=num_step, mini_iter=20)
        speed0 = float(actions[primitive_id, 0, 1]) if actions.shape[1] > 0 else 0.0
        margin = float(safety_margin) * (float(reverse_margin_scale) if speed0 < 0.0 else 1.0)
        swept_polys = []
        for step_idx in range(horizon):
            end_idx = min(len(states) - 1, int((step_idx + 1) * sim_steps_per_primitive_step))
            sampled = states[: end_idx + 1 : sample_stride]
            if not sampled or sampled[-1] is not states[end_idx]:
                sampled = list(sampled) + [states[end_idx]]
            for state in sampled:
                front, rear = create_articulated_boxes_canonical(state)
                swept_polys.append(front)
                swept_polys.append(rear)
            swept = unary_union(swept_polys)
            if margin > 0.0:
                swept = swept.buffer(margin)
            extent_from_origin = _ray_distances_to_polygon_extent(swept, ray_angles, ray_length)
            dist_star[primitive_id, step_idx, :] = np.maximum(0.0, extent_from_origin - boundary)

    metadata = {
        "created_time": float(time.time()),
        "lidar_num": int(lidar_num),
        "lidar_range": float(lidar_range),
        "safety_margin": float(safety_margin),
        "reverse_margin_scale": float(reverse_margin_scale),
        "sample_stride": int(sample_stride),
        "num_step": int(num_step),
        "index_kind": "ray_swept_front_rear",
    }
    return PrimitiveRaySafetyIndex(
        dist_star=dist_star,
        lidar_range=lidar_range,
        lidar_num=lidar_num,
        horizon=horizon,
        library_signature=library_signature(actions, deltas),
        metadata=metadata,
    )
