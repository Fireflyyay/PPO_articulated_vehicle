from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class ShapingStats:
    centroid_count: int


class DiscoveredPrimitiveShaping:
    """Weak shaping reward based on discovered primitive state prototypes.

    We keep a set of centroid features C = {c_i}. The shaping reward is:

        r = k * (max_i sim(f(s_{t+1}), c_i) - max_i sim(f(s_t), c_i))

    with sim as RBF similarity.

    This module is designed to be safe:
    - k is small (DP_SHAPING_COEF)
    - if no centroids exist -> returns 0
    """

    def __init__(self, config):
        self.cfg = config
        self.centroids: np.ndarray = np.zeros((0, 6), dtype=np.float64)

    def clear(self) -> None:
        self.centroids = np.zeros((0, 6), dtype=np.float64)

    def add_centroids(self, feats: Sequence[np.ndarray]) -> ShapingStats:
        feats_arr = []
        for f in feats:
            if f is None:
                continue
            a = np.asarray(f, dtype=np.float64).reshape(-1)
            if a.size < 6:
                continue
            feats_arr.append(a[:6])

        if len(feats_arr) == 0:
            return ShapingStats(centroid_count=int(self.centroids.shape[0]))

        new_c = np.stack(feats_arr, axis=0)
        if self.centroids.shape[0] == 0:
            self.centroids = new_c
        else:
            self.centroids = np.concatenate([self.centroids, new_c], axis=0)

        # cap
        cap = int(getattr(self.cfg, "DP_MAX_CENTROIDS", 64))
        if self.centroids.shape[0] > cap:
            self.centroids = self.centroids[-cap:]

        return ShapingStats(centroid_count=int(self.centroids.shape[0]))

    def reward(self, obs: np.ndarray, next_obs: np.ndarray) -> float:
        if not bool(getattr(self.cfg, "USE_DISCOVERED_PRIMITIVE_SHAPING", True)):
            return 0.0
        if self.centroids.shape[0] == 0:
            return 0.0

        k = float(getattr(self.cfg, "DP_SHAPING_COEF", 0.02))
        sigma = float(getattr(self.cfg, "DP_SHAPING_SIGMA", 1.0))

        f0 = self.extract_feature_from_obs(obs)
        f1 = self.extract_feature_from_obs(next_obs)

        s0 = self._max_sim(f0, sigma)
        s1 = self._max_sim(f1, sigma)
        return float(k * (s1 - s0))

    def extract_feature_from_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        """Feature f(x) for shaping: (dx, dy, heading_err, articulation, speed, min_lidar).

        All are best-effort from the observation layout used by CarParking.
        """
        obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
        lidar_n = int(getattr(self.cfg, "LIDAR_NUM", 120))
        lidar_r = float(getattr(self.cfg, "LIDAR_RANGE", 30.0))
        max_dist = float(getattr(self.cfg, "MAX_DIST_TO_DEST", 70.0))

        lidar = obs_vec[:lidar_n]
        min_lidar = float(np.min(lidar)) * lidar_r if lidar.size > 0 else lidar_r

        target = obs_vec[lidar_n : lidar_n + 7]
        dist = float(target[0]) * max_dist
        rel_angle = float(np.arctan2(target[2], target[1]))
        rel_heading = float(np.arctan2(target[4], target[3]))
        articulation = float(np.arctan2(target[6], target[5]))

        dx = dist * float(np.cos(rel_angle))
        dy = dist * float(np.sin(rel_angle))
        heading_err = float(_wrap_pi(rel_heading))
        art = float(_wrap_pi(articulation))

        # speed (normalized) stored after target, first dim
        try:
            speed_norm = float(obs_vec[lidar_n + 7])
        except Exception:
            speed_norm = 0.0

        return np.asarray([dx, dy, heading_err, art, speed_norm, min_lidar], dtype=np.float64)

    def _max_sim(self, feat: np.ndarray, sigma: float) -> float:
        feat = np.asarray(feat, dtype=np.float64).reshape(1, -1)
        c = self.centroids
        # RBF sim
        d2 = np.sum((c - feat) ** 2, axis=1)
        sim = np.exp(-d2 / (sigma * sigma + 1e-9))
        return float(np.max(sim))
