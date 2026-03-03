from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class EpisodeTrace:
    """Rollout trace stored at macro-step level, with optional low-level execution trace.

    Notes:
    - observations/actions_primitive/rewards/dones/infos are macro-step aligned.
    - actions_low_level is a list of low-level action sequences per macro step.
      Each element has shape [Hi, 2] in *physical units* (steer, speed).
    - states_optional may contain low-level state snapshots if wrapper provides them.
    """

    episode_id: int
    scene_type: str
    success: bool
    total_reward: float
    step_count_macro: int
    takeover_used: bool

    observations: List[np.ndarray]
    actions_primitive: List[int]
    actions_low_level: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    infos: List[dict]
    states_optional: Optional[List[Any]] = None


@dataclass
class CandidatePrimitive:
    actions_raw: np.ndarray  # [T, 2] physical units
    actions_resampled: np.ndarray  # [H, 2] physical units

    state_seq: Optional[np.ndarray] = None  # [T, state_dim]
    start_feature: Optional[np.ndarray] = None
    end_feature: Optional[np.ndarray] = None
    delta_feature: Optional[np.ndarray] = None  # [dx, dy, d_yaw] in start local frame

    complexity_score: float = 0.0
    utility_score: float = 0.0
    novelty_score: float = 0.0
    final_score: float = 0.0

    tags: Dict[str, Any] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryMiner:
    """Mine candidate macro-actions (motion primitives) from episode traces.

    This implementation is intentionally light-weight (numpy-only) and designed
    to be stable as a training add-on.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = bool(verbose)

    # -------------------------
    # Public API
    # -------------------------
    def mine_from_episodes(
        self,
        episodes: Sequence[EpisodeTrace],
        current_library,
        config,
    ) -> List[CandidatePrimitive]:
        """Return scored & filtered candidate primitives.

        Args:
            episodes: list of EpisodeTrace
            current_library: existing PrimitiveLibrary-like object
            config: configs module or config object with AP_* attributes

        Returns:
            candidates: sorted by final_score desc
        """
        H = int(getattr(current_library, "horizon", None) or getattr(config, "PRIMITIVE_H", 1))

        # 1) filter: success first; allow near-success as optional via config
        filtered = []
        for ep in episodes:
            if getattr(ep, "success", False):
                filtered.append(ep)
                continue
            # near-success: based on last macro obs dist-to-goal (best-effort)
            try:
                if bool(getattr(config, "AP_TRACE_KEEP_NEAR_SUCCESS", True)):
                    dist_last = float(self._parse_goal_from_obs(ep.observations[-1], config)["dist"])
                    if dist_last <= float(getattr(config, "AP_NEAR_SUCCESS_DIST_THR", 3.0)):
                        filtered.append(ep)
            except Exception:
                pass

        if len(filtered) == 0:
            return []

        # 2) segment: event-driven + sliding windows (union)
        segments: List[Tuple[EpisodeTrace, int, int, Dict[str, Any]]] = []
        for ep in filtered:
            u, states, low2macro = self._flatten_low_level(ep)
            if u is None or u.shape[0] < max(2, int(getattr(config, "AP_SEGMENT_H_MIN", 6))):
                continue

            seg_a = self.segment_trajectory_event_driven(u, states, config)
            seg_b = self.segment_trajectory_sliding(u, states, config)
            for (t0, t1, meta) in (seg_a + seg_b):
                # attach ep context
                meta = dict(meta or {})
                meta["episode_id"] = int(ep.episode_id)
                meta["scene_type"] = str(ep.scene_type)
                meta["success"] = bool(ep.success)
                # map low-level range back to macro steps (best-effort)
                try:
                    meta["macro_t0"] = int(low2macro[t0])
                    meta["macro_t1"] = int(low2macro[max(t0, t1 - 1)])
                except Exception:
                    pass
                segments.append((ep, int(t0), int(t1), meta))

        if len(segments) == 0:
            return []

        # 3) score and resample to fixed H
        candidates: List[CandidatePrimitive] = []
        for ep, t0, t1, meta in segments:
            u, states, low2macro = self._flatten_low_level(ep)
            if u is None:
                continue
            t0 = int(max(0, min(t0, u.shape[0] - 1)))
            t1 = int(max(t0 + 2, min(t1, u.shape[0])))
            seg_u = u[t0:t1]
            seg_states = states[t0:t1] if states is not None and states.shape[0] >= t1 else None

            actions_H = self.resample_segment_to_H(seg_u, H)
            cand = CandidatePrimitive(actions_raw=seg_u, actions_resampled=actions_H)

            # state-derived features
            if seg_states is not None:
                cand.state_seq = seg_states
                start_feat, end_feat, delta_feat = self._state_features(seg_states)
                cand.start_feature = start_feat
                cand.end_feature = end_feat
                cand.delta_feature = delta_feat

            cand.complexity_score = float(self.score_segment_complexity(seg_u, seg_states, config))
            cand.utility_score = float(self.score_segment_utility(ep, t0, t1, low2macro, config))
            cand.novelty_score = float(self.compute_novelty(actions_H, cand.delta_feature, current_library, config))

            a = float(getattr(config, "AP_W_COMPLEXITY", 0.35))
            b = float(getattr(config, "AP_W_UTILITY", 0.40))
            c = float(getattr(config, "AP_W_NOVELTY", 0.25))
            cand.final_score = a * cand.complexity_score + b * cand.utility_score + c * cand.novelty_score

            cand.tags = self._tag_segment(seg_u, seg_states, config)
            cand.source_metadata = meta
            # attach a start_state snapshot for feasibility checks
            try:
                if seg_states is not None and seg_states.shape[0] > 0:
                    cand.source_metadata["start_state"] = {
                        "x": float(seg_states[0, 0]),
                        "y": float(seg_states[0, 1]),
                        "heading": float(seg_states[0, 2]),
                        "rear_heading": float(seg_states[0, 3]),
                        "speed": float(seg_states[0, 4]),
                        "steering": float(seg_states[0, 5]),
                    }
                    cand.source_metadata["segment_range"] = [int(t0), int(t1)]
            except Exception:
                pass

            # 4) thresholds
            if cand.complexity_score < float(getattr(config, "AP_COMPLEXITY_THRESH", 0.35)):
                continue
            if cand.utility_score < float(getattr(config, "AP_UTILITY_THRESH", 0.25)):
                continue
            if cand.novelty_score < float(getattr(config, "AP_NOVELTY_THRESH", 0.20)):
                continue

            candidates.append(cand)

        if len(candidates) == 0:
            return []

        # 5) sort + top-k
        candidates.sort(key=lambda x: float(x.final_score), reverse=True)
        topk = int(getattr(config, "AP_FINAL_TOPK", 40))
        return candidates[: max(1, topk)]

    # -------------------------
    # Segmenters
    # -------------------------
    def segment_trajectory_event_driven(
        self,
        actions: np.ndarray,
        states: Optional[np.ndarray],
        config,
    ) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Event-driven windows around sign changes / high articulation usage."""
        T = int(actions.shape[0])
        if T < 3:
            return []

        v = actions[:, 1]
        steer = actions[:, 0]
        v_th = float(getattr(config, "AP_V_TH", 0.1))
        L1 = int(getattr(config, "AP_EVENT_WINDOW_BEFORE", 4))
        L2 = int(getattr(config, "AP_EVENT_WINDOW_AFTER", 8))

        events = set()
        # speed sign switches
        sgn = np.sign(v)
        for t in range(1, T):
            if abs(v[t]) > v_th and abs(v[t - 1]) > v_th and sgn[t] != sgn[t - 1]:
                events.add(t)
        # steer sign switches
        sgns = np.sign(steer)
        for t in range(1, T):
            if abs(steer[t]) > 1e-4 and abs(steer[t - 1]) > 1e-4 and sgns[t] != sgns[t - 1]:
                events.add(t)

        # articulation near extremes (if rear_heading present)
        if states is not None and states.shape[1] >= 4:
            beta = np.array([_wrap_pi(float(states[i, 2]) - float(states[i, 3])) for i in range(T)], dtype=np.float64)
            beta_abs = np.abs(beta)
            beta_thr = float(np.deg2rad(28))
            for t in range(T):
                if beta_abs[t] >= beta_thr:
                    events.add(t)

        segs = []
        for t in sorted(events):
            t0 = max(0, t - L1)
            t1 = min(T, t + L2)
            if t1 - t0 >= 2:
                segs.append((t0, t1, {"segmenter": "event", "event_t": int(t)}))
        return segs

    def segment_trajectory_sliding(
        self,
        actions: np.ndarray,
        states: Optional[np.ndarray],
        config,
    ) -> List[Tuple[int, int, Dict[str, Any]]]:
        """Sliding windows for recall; later filtered by scores."""
        T = int(actions.shape[0])
        hmin = int(getattr(config, "AP_SEGMENT_H_MIN", 6))
        hmax = int(getattr(config, "AP_SEGMENT_H_MAX", 24))
        stride = int(getattr(config, "AP_SEGMENT_STRIDE", 2))

        hmin = max(2, min(hmin, T))
        hmax = max(hmin, min(hmax, T))
        segs = []
        for w in range(hmin, hmax + 1):
            for t0 in range(0, T - w + 1, stride):
                t1 = t0 + w
                segs.append((t0, t1, {"segmenter": "sliding", "win": int(w)}))
        return segs

    # -------------------------
    # Scoring
    # -------------------------
    def score_segment_complexity(self, actions: np.ndarray, states: Optional[np.ndarray], config) -> float:
        """Complexity proxy in [0,1] (best-effort)."""
        actions = np.asarray(actions, dtype=np.float64)
        T = int(actions.shape[0])
        if T < 2:
            return 0.0

        v = actions[:, 1]
        steer = actions[:, 0]
        v_th = float(getattr(config, "AP_V_TH", 0.1))

        # A) reverse ratio
        r_rev = float(np.mean(v < -v_th))

        # B) steer variation (normalized)
        r_steer_var = float(np.mean(np.abs(np.diff(steer))))
        try:
            steer_max = float(getattr(config, "VALID_STEER", [0.0, 1.0])[1])
        except Exception:
            steer_max = float(np.deg2rad(36))
        r_steer_var = min(1.0, r_steer_var / max(1e-6, 0.25 * steer_max))

        # C) speed direction switches
        sgn = np.sign(v)
        valid = (np.abs(v) > v_th)
        switches = 0
        for t in range(1, T):
            if valid[t] and valid[t - 1] and sgn[t] != sgn[t - 1]:
                switches += 1
        r_switch = float(switches) / float(max(1, T - 1))

        # D) curvature proxy (needs states)
        r_curv = 0.0
        if states is not None and states.shape[0] >= 3 and states.shape[1] >= 3:
            xs = states[:, 0]
            ys = states[:, 1]
            th = states[:, 2]
            kappas = []
            for t in range(T - 1):
                ds = float(np.hypot(xs[t + 1] - xs[t], ys[t + 1] - ys[t]))
                dth = float(_wrap_pi(float(th[t + 1]) - float(th[t])))
                kappas.append(abs(dth) / (ds + 1e-6))
            if len(kappas) > 0:
                r_curv = float(np.percentile(np.asarray(kappas, dtype=np.float64), 95))
                r_curv = min(1.0, r_curv / 1.5)

        # E) articulation usage (if rear_heading exists)
        r_art = 0.0
        if states is not None and states.shape[1] >= 4:
            beta = np.array([_wrap_pi(float(states[i, 2]) - float(states[i, 3])) for i in range(T)], dtype=np.float64)
            beta_abs = np.abs(beta)
            beta_max = float(np.deg2rad(36))
            r_art = float(np.mean(np.clip(beta_abs / (beta_max + 1e-6), 0.0, 1.0)))

        # F) near-obstacle proxy unavailable in trace by default
        r_obs = 0.0

        w = getattr(config, "AP_COMPLEXITY_WEIGHTS", None) or {}
        w_rev = float(w.get("rev", 0.15))
        w_sv = float(w.get("steer_var", 0.20))
        w_sw = float(w.get("switch", 0.20))
        w_curv = float(w.get("curv", 0.15))
        w_art = float(w.get("art", 0.15))
        w_obs = float(w.get("obs", 0.15))

        comp = (
            w_rev * r_rev
            + w_sv * r_steer_var
            + w_sw * r_switch
            + w_curv * r_curv
            + w_art * r_art
            + w_obs * r_obs
        )
        return float(np.clip(comp, 0.0, 1.0))

    def score_segment_utility(self, ep: EpisodeTrace, t0: int, t1: int, low2macro: np.ndarray, config) -> float:
        """Utility proxy in [0,1] based on success, scene difficulty, terminal proximity and progress."""
        u_success = 1.0 if bool(ep.success) else 0.5  # near-success allowed but discounted

        scene = str(getattr(ep, "scene_type", "Normal"))
        scene_w = {"Normal": 1.0, "Complex": 1.5, "Extrem": 2.0, "Extreme": 2.0}.get(scene, 1.0)

        # Estimate goal distance using macro observations
        macro0 = int(low2macro[int(t0)]) if low2macro is not None else 0
        macro1 = int(low2macro[int(max(t0, t1 - 1))]) if low2macro is not None else (len(ep.observations) - 1)
        macro0 = max(0, min(macro0, len(ep.observations) - 1))
        macro1 = max(0, min(macro1, len(ep.observations) - 1))
        obs0 = ep.observations[macro0]
        obs1 = ep.observations[macro1]

        g0 = self._parse_goal_from_obs(obs0, config)
        g1 = self._parse_goal_from_obs(obs1, config)
        d0 = float(g0["dist"])
        d1 = float(g1["dist"])
        dbar = 0.5 * (d0 + d1)

        d_term = float(getattr(config, "AP_D_TERM", 10.0))
        u_terminal = float(np.exp(-dbar / max(1e-6, d_term)))

        # Progress gain (distance reduction)
        progress = float(np.clip((d0 - d1) / max(1e-6, d_term), -1.0, 1.0))
        progress_gain = max(0.0, progress)

        # Heading recovery: reduce rel_heading magnitude
        hd0 = abs(float(g0.get("goal_heading", 0.0)))
        hd1 = abs(float(g1.get("goal_heading", 0.0)))
        recover = float(np.clip((hd0 - hd1) / max(1e-6, np.deg2rad(30)), 0.0, 1.0))

        a1, a2, a3 = 0.45, 0.35, 0.20
        util = u_success * scene_w * (a1 * u_terminal + a2 * recover + a3 * progress_gain)

        # Normalize to [0,1] with gentle saturation
        util = float(np.tanh(util))
        return float(np.clip(util, 0.0, 1.0))

    def compute_novelty(
        self,
        actions_H: np.ndarray,
        delta_feature: Optional[np.ndarray],
        current_library,
        config,
    ) -> float:
        """Novelty proxy in [0,1] based on min L2 distance to existing library actions."""
        try:
            lib_actions = np.asarray(getattr(current_library, "actions"), dtype=np.float64)
        except Exception:
            return 1.0

        a = np.asarray(actions_H, dtype=np.float64)
        if lib_actions.ndim != 3:
            return 1.0

        # Align horizon
        H = int(a.shape[0])
        if int(lib_actions.shape[1]) != H:
            # Cannot compare reliably; treat as novel
            return 1.0

        vec = a.reshape(-1)
        lib_vec = lib_actions.reshape(lib_actions.shape[0], -1)
        dists = np.linalg.norm(lib_vec - vec[None, :], axis=1) / np.sqrt(float(vec.size) + 1e-9)
        min_d = float(np.min(dists)) if dists.size > 0 else 1.0

        scale = float(getattr(config, "AP_NOVELTY_ACTION_L2_SCALE", 1.0))
        nov = min_d / max(1e-6, scale)
        return float(np.clip(nov, 0.0, 1.0))

    # -------------------------
    # Resampling
    # -------------------------
    def resample_segment_to_H(self, actions: np.ndarray, H: int) -> np.ndarray:
        """Linear resample actions from length T to length H.

        Args:
            actions: [T, 2]
            H: target horizon

        Returns:
            [H, 2]
        """
        actions = np.asarray(actions, dtype=np.float64)
        assert actions.ndim == 2 and actions.shape[1] == 2
        T = int(actions.shape[0])
        H = int(H)
        if T == H:
            return actions.copy()
        if T < 2:
            return np.repeat(actions[:1], H, axis=0)

        xs = np.linspace(0.0, 1.0, num=T)
        xq = np.linspace(0.0, 1.0, num=H)
        out = np.zeros((H, 2), dtype=np.float64)
        for d in range(2):
            out[:, d] = np.interp(xq, xs, actions[:, d])
        return out

    # -------------------------
    # Helpers
    # -------------------------
    def _flatten_low_level(self, ep: EpisodeTrace) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Concatenate low-level actions and (optional) low-level states.

        Returns:
            actions_low: [T,2]
            states_low:  [T,6] (x,y,heading,rear_heading,speed,steering) if available else None
            low2macro:   [T] mapping to macro-step index
        """
        if ep.actions_low_level is None or len(ep.actions_low_level) == 0:
            return None, None, None

        actions_list = []
        low2macro = []
        for m, u in enumerate(ep.actions_low_level):
            if u is None:
                continue
            u = np.asarray(u, dtype=np.float64)
            if u.ndim != 2 or u.shape[1] != 2:
                continue
            actions_list.append(u)
            low2macro.extend([m] * int(u.shape[0]))

        if len(actions_list) == 0:
            return None, None, None

        actions_low = np.concatenate(actions_list, axis=0)
        low2macro_arr = np.asarray(low2macro, dtype=np.int64)

        # states: try to reconstruct from infos' macro_exec_trace if present
        states_low = None
        try:
            state_rows = []
            for m, info in enumerate(ep.infos):
                tr = info.get("macro_exec_trace", None) if isinstance(info, dict) else None
                if not isinstance(tr, dict):
                    continue
                sub_states = tr.get("sub_states", None)
                sub_actions = tr.get("sub_actions_phys", None)
                if sub_states is None or sub_actions is None:
                    continue
                # we want state aligned with actions; sub_states may include initial state
                if isinstance(sub_actions, np.ndarray):
                    hi = int(sub_actions.shape[0])
                else:
                    hi = int(len(sub_actions))
                if isinstance(sub_states, list) and len(sub_states) == hi + 1:
                    sub_states = sub_states[1:]
                if isinstance(sub_states, list) and len(sub_states) == hi:
                    for s in sub_states:
                        state_rows.append([
                            float(s.get("x", 0.0)),
                            float(s.get("y", 0.0)),
                            float(s.get("heading", 0.0)),
                            float(s.get("rear_heading", 0.0)),
                            float(s.get("speed", 0.0)),
                            float(s.get("steering", 0.0)),
                        ])
            if len(state_rows) == actions_low.shape[0]:
                states_low = np.asarray(state_rows, dtype=np.float64)
        except Exception:
            states_low = None

        return actions_low, states_low, low2macro_arr

    def _parse_goal_from_obs(self, obs_vec: np.ndarray, config) -> Dict[str, float]:
        obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
        try:
            lidar_n = int(getattr(config, "LIDAR_NUM", 120))
            max_dist = float(getattr(config, "MAX_DIST_TO_DEST", 70.0))
        except Exception:
            lidar_n, max_dist = 120, 70.0

        target = obs_vec[lidar_n : lidar_n + 7]
        dist = float(target[0]) * max_dist
        rel_angle = float(np.arctan2(target[2], target[1]))
        rel_heading = float(np.arctan2(target[4], target[3]))
        articulation = float(np.arctan2(target[6], target[5]))
        return {
            "dist": dist,
            "goal_x": dist * float(np.cos(rel_angle)),
            "goal_y": dist * float(np.sin(rel_angle)),
            "goal_heading": _wrap_pi(rel_heading),
            "articulation": _wrap_pi(articulation),
            "rel_angle": rel_angle,
        }

    def _state_features(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (start_feature, end_feature, delta_feature) from low-level states.

        states layout: [x,y,heading,rear_heading,speed,steering]
        delta_feature: dx,dy,d_yaw in start-local frame.
        """
        s0 = states[0]
        s1 = states[-1]
        x0, y0, h0 = float(s0[0]), float(s0[1]), float(s0[2])
        x1, y1, h1 = float(s1[0]), float(s1[1]), float(s1[2])

        dxw, dyw = (x1 - x0), (y1 - y0)
        c, s = float(np.cos(h0)), float(np.sin(h0))
        dx = c * dxw + s * dyw
        dy = -s * dxw + c * dyw
        dyaw = _wrap_pi(h1 - h0)

        start_feat = np.asarray([x0, y0, h0, float(s0[3]), float(s0[4]), float(s0[5])], dtype=np.float64)
        end_feat = np.asarray([x1, y1, h1, float(s1[3]), float(s1[4]), float(s1[5])], dtype=np.float64)
        delta_feat = np.asarray([dx, dy, dyaw], dtype=np.float64)
        return start_feat, end_feat, delta_feat

    def _tag_segment(self, actions: np.ndarray, states: Optional[np.ndarray], config) -> Dict[str, Any]:
        actions = np.asarray(actions, dtype=np.float64)
        v = actions[:, 1]
        steer = actions[:, 0]
        v_th = float(getattr(config, "AP_V_TH", 0.1))

        reverse_ratio = float(np.mean(v < -v_th))
        steer_sign_changes = int(np.sum(np.sign(steer[1:]) != np.sign(steer[:-1])))
        speed_sign_changes = int(np.sum(np.sign(v[1:]) != np.sign(v[:-1])))

        tags = {
            "reverse": bool(reverse_ratio > 0.5),
            "reverse_ratio": reverse_ratio,
            "steer_sign_changes": steer_sign_changes,
            "speed_sign_changes": speed_sign_changes,
        }

        if states is not None and states.shape[1] >= 4:
            beta = np.array([_wrap_pi(float(states[i, 2]) - float(states[i, 3])) for i in range(states.shape[0])], dtype=np.float64)
            tags["beta_abs_mean"] = float(np.mean(np.abs(beta)))

        return tags
