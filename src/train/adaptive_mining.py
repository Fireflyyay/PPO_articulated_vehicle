from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from primitives.trajectory_miner import EpisodeTrace


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def parse_goal_from_obs(obs_vec: np.ndarray, config) -> Dict[str, float]:
    obs_vec = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
    lidar_n = int(getattr(config, "LIDAR_NUM", 120))
    max_dist = float(getattr(config, "MAX_DIST_TO_DEST", 70.0))
    target = obs_vec[lidar_n : lidar_n + 7]
    dist = float(target[0]) * max_dist
    rel_angle = float(np.arctan2(target[2], target[1]))
    rel_heading = float(np.arctan2(target[4], target[3]))
    return {
        "dist": dist,
        "goal_x": dist * float(np.cos(rel_angle)),
        "goal_y": dist * float(np.sin(rel_angle)),
        "goal_heading": wrap_pi(rel_heading),
    }


def _scene_weight(scene_type: str) -> float:
    scene = str(scene_type)
    if scene in ("Extrem", "Extreme"):
        return 1.5
    if scene == "Complex":
        return 1.2
    return 0.8


def trace_terminal_goal_distance(trace: EpisodeTrace, config) -> float:
    if not getattr(trace, "observations", None):
        return float("inf")
    try:
        return float(parse_goal_from_obs(trace.observations[-1], config)["dist"])
    except Exception:
        return float("inf")


def score_trace_for_replay(trace: EpisodeTrace, config) -> float:
    dist_last = trace_terminal_goal_distance(trace, config)
    dist_start = dist_last
    try:
        dist_start = float(parse_goal_from_obs(trace.observations[0], config)["dist"])
    except Exception:
        pass

    near_thr = float(getattr(config, "AP_NEAR_SUCCESS_DIST_THR", 3.0))
    if bool(getattr(trace, "success", False)):
        learnable_band = 0.55
    elif dist_last <= near_thr:
        learnable_band = 1.0
    elif dist_last <= 2.0 * near_thr:
        learnable_band = 0.65
    elif dist_last <= 3.0 * near_thr:
        learnable_band = 0.25
    else:
        learnable_band = 0.0

    progress = 0.0
    if np.isfinite(dist_last) and np.isfinite(dist_start) and dist_start > 1e-6:
        progress = float(np.clip((dist_start - dist_last) / dist_start, 0.0, 1.0))

    score = _scene_weight(getattr(trace, "scene_type", "Normal")) * (learnable_band + 0.35 * progress)
    if bool(getattr(trace, "takeover_used", False)):
        score *= 0.9
    return float(max(0.0, score))


def select_replay_episodes(trace_buffer: Sequence[EpisodeTrace], replay_count: int, config) -> List[EpisodeTrace]:
    if replay_count <= 0:
        return []

    traces = list(trace_buffer)
    if len(traces) == 0:
        return []

    min_episode_id = min(int(getattr(trace, "episode_id", 0)) for trace in traces)
    max_episode_id = max(int(getattr(trace, "episode_id", 0)) for trace in traces)
    episode_span = max(1, max_episode_id - min_episode_id)

    ranked = []
    for trace in traces:
        base_score = score_trace_for_replay(trace, config)
        if base_score <= 0.0:
            continue
        recency = float(int(getattr(trace, "episode_id", 0)) - min_episode_id) / float(episode_span)
        ranked.append((base_score + 0.05 * recency, trace))

    ranked.sort(key=lambda item: (item[0], int(getattr(item[1], "episode_id", 0))), reverse=True)
    return [trace for _, trace in ranked[: max(0, int(replay_count))]]


def build_mining_schedule(stats: Dict[str, float], config) -> List[str]:
    hard_recent = float(stats.get("hard_success_rate_recent", 0.0))
    hard_threshold = float(getattr(config, "AP_TRIGGER_HARD_SUCCESS_RATE", 0.15))

    if hard_recent <= hard_threshold + 0.10:
        return ["Complex", "Extrem", "Complex", "Extrem", "Normal"]
    if hard_recent <= hard_threshold + 0.30:
        return ["Extrem", "Complex", "Extrem", "Complex", "Normal"]
    return ["Extrem", "Complex", "Normal"]


@dataclass(frozen=True)
class AdaptiveProxyContext:
    episode_id: int
    scene_type: str
    start_state: Dict[str, float]
    goal_world: tuple
    initial_goal_dist: float
    scene_weight: float


def build_proxy_eval_contexts(episodes: Sequence[EpisodeTrace], config, max_contexts: int = None) -> List[AdaptiveProxyContext]:
    if max_contexts is None:
        max_contexts = int(getattr(config, "AP_PROXY_EVAL_CONTEXTS", 8))

    contexts: List[AdaptiveProxyContext] = []
    for trace in episodes:
        if len(contexts) >= int(max_contexts):
            break
        if not getattr(trace, "observations", None) or not getattr(trace, "infos", None):
            continue
        info0 = trace.infos[0] if len(trace.infos) > 0 else {}
        if not isinstance(info0, dict):
            continue
        macro_exec_trace = info0.get("macro_exec_trace", {})
        if not isinstance(macro_exec_trace, dict):
            continue
        sub_states = macro_exec_trace.get("sub_states", None)
        if not isinstance(sub_states, list) or len(sub_states) == 0 or not isinstance(sub_states[0], dict):
            continue

        start_state = {
            "x": float(sub_states[0].get("x", 0.0)),
            "y": float(sub_states[0].get("y", 0.0)),
            "heading": float(sub_states[0].get("heading", 0.0)),
            "rear_heading": float(sub_states[0].get("rear_heading", sub_states[0].get("heading", 0.0))),
            "speed": float(sub_states[0].get("speed", 0.0)),
            "steering": float(sub_states[0].get("steering", 0.0)),
        }

        goal_local = parse_goal_from_obs(trace.observations[0], config)
        heading = float(start_state["heading"])
        c = float(np.cos(heading))
        s = float(np.sin(heading))
        gx = float(start_state["x"]) + c * float(goal_local["goal_x"]) - s * float(goal_local["goal_y"])
        gy = float(start_state["y"]) + s * float(goal_local["goal_x"]) + c * float(goal_local["goal_y"])
        gh = wrap_pi(float(heading) + float(goal_local["goal_heading"]))

        contexts.append(
            AdaptiveProxyContext(
                episode_id=int(getattr(trace, "episode_id", 0)),
                scene_type=str(getattr(trace, "scene_type", "Normal")),
                start_state=start_state,
                goal_world=(gx, gy, gh),
                initial_goal_dist=float(goal_local["dist"]),
                scene_weight=_scene_weight(getattr(trace, "scene_type", "Normal")),
            )
        )

    return contexts