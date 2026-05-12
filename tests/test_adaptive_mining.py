from types import SimpleNamespace

import numpy as np

from primitives.trajectory_miner import EpisodeTrace
from train.adaptive_mining import build_proxy_eval_contexts, select_replay_episodes


def _make_obs(dist, lidar_num=4, max_dist=10.0):
    obs = np.ones((lidar_num + 7,), dtype=np.float64)
    target = np.zeros((7,), dtype=np.float64)
    target[0] = float(dist) / float(max_dist)
    target[1] = 1.0
    target[3] = 1.0
    target[5] = 1.0
    obs[lidar_num : lidar_num + 7] = target
    return obs


def _make_trace(episode_id, scene_type, success, start_dist, end_dist):
    info = {
        "macro_exec_trace": {
            "sub_states": [
                {
                    "x": 1.0,
                    "y": 2.0,
                    "heading": 0.0,
                    "rear_heading": 0.0,
                    "speed": 0.0,
                    "steering": 0.0,
                }
            ]
        }
    }
    return EpisodeTrace(
        episode_id=int(episode_id),
        scene_type=str(scene_type),
        success=bool(success),
        total_reward=0.0,
        step_count_macro=1,
        takeover_used=False,
        observations=[_make_obs(start_dist), _make_obs(end_dist)],
        actions_primitive=[0],
        actions_low_level=[np.zeros((1, 2), dtype=np.float64)],
        rewards=[0.0],
        dones=[True],
        infos=[info],
        states_optional=None,
    )


def test_select_replay_episodes_prefers_hard_near_success_failures():
    cfg = SimpleNamespace(LIDAR_NUM=4, MAX_DIST_TO_DEST=10.0, AP_NEAR_SUCCESS_DIST_THR=3.0)
    easy_success = _make_trace(1, "Normal", True, start_dist=3.0, end_dist=0.2)
    hard_near_fail = _make_trace(2, "Extrem", False, start_dist=6.0, end_dist=2.0)
    far_fail = _make_trace(3, "Complex", False, start_dist=8.0, end_dist=7.5)

    selected = select_replay_episodes([easy_success, hard_near_fail, far_fail], replay_count=1, config=cfg)

    assert len(selected) == 1
    assert selected[0].episode_id == 2


def test_build_proxy_eval_contexts_extracts_start_state_and_goal():
    cfg = SimpleNamespace(LIDAR_NUM=4, MAX_DIST_TO_DEST=10.0, AP_NEAR_SUCCESS_DIST_THR=3.0, AP_PROXY_EVAL_CONTEXTS=4)
    trace = _make_trace(7, "Complex", False, start_dist=5.0, end_dist=2.5)

    contexts = build_proxy_eval_contexts([trace], cfg)

    assert len(contexts) == 1
    context = contexts[0]
    assert context.episode_id == 7
    assert context.scene_type == "Complex"
    assert context.start_state["x"] == 1.0
    assert context.start_state["y"] == 2.0
    assert context.initial_goal_dist == 5.0