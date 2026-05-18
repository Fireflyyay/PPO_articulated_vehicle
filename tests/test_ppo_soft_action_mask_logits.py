from types import SimpleNamespace

import numpy as np
import torch

from model.agent.ppo_agent import PPOAgent


def _dummy_agent():
    agent = object.__new__(PPOAgent)
    agent.configs = SimpleNamespace(soft_mask_logit_lambda=2.0, soft_mask_small_value=1e-8)
    agent.action_logit_bias = None
    return agent


def _make_discrete_ppo(action_dim: int = 3):
    obs_dim = 4
    actor_layers = {
        "input_dim": obs_dim,
        "output_size": int(action_dim),
        "use_tanh_output": False,
        "orthogonal_init": False,
    }
    critic_layers = {
        "input_dim": obs_dim,
        "output_size": 1,
        "use_tanh_output": False,
        "orthogonal_init": False,
    }
    cfg = {
        "observation_shape": (obs_dim,),
        "action_dim": int(action_dim),
        "batch_size": 8,
        "mini_batch": 4,
        "mini_epoch": 1,
        "actor_layers": actor_layers,
        "critic_layers": critic_layers,
        "gamma": 0.99,
        "lr_actor": 1e-3,
        "lr_critic": 1e-3,
        "state_norm": False,
        "guidance_logit_default_weight": 0.0,
    }
    return PPOAgent(cfg, discrete=True)


def test_float_action_mask_reweights_logits_without_hard_zero():
    agent = _dummy_agent()
    logits = torch.zeros((1, 3), dtype=torch.float32)
    mask = np.array([1.0, 0.5, 0.1], dtype=np.float32)

    out = PPOAgent._mask_logits(agent, logits, mask)

    assert torch.isfinite(out).all()
    assert out[0, 0] > out[0, 1] > out[0, 2]
    assert out[0, 2] > -100.0


def test_integer_action_mask_keeps_legacy_hard_mask():
    agent = _dummy_agent()
    logits = torch.zeros((1, 3), dtype=torch.float32)
    mask = np.array([1, 0, 1], dtype=np.int8)

    out = PPOAgent._mask_logits(agent, logits, mask)

    assert out[0, 1] < -1e9
    assert out[0, 0] == 0.0
    assert out[0, 2] == 0.0


def test_guidance_logits_bias_is_broadcast_and_added_after_mask_bias():
    agent = _dummy_agent()
    logits = torch.zeros((1, 3), dtype=torch.float32)
    mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    guidance = np.array([0.0, 2.0, -1.0], dtype=np.float32)

    parts = PPOAgent._compose_discrete_logits(
        agent,
        logits,
        action_mask=mask,
        guidance_logits=guidance,
        guidance_weight=0.5,
    )

    out = parts["final_logits"]
    assert torch.isfinite(out).all()


    def test_push_memory_persists_rollout_time_guidance_fields():
        agent = _make_discrete_ppo(action_dim=3)
        obs = np.zeros((4,), dtype=np.float64)
        next_obs = np.ones((4,), dtype=np.float64)
        action_mask = np.ones((3,), dtype=np.float32)
        guidance_logits = np.array([0.0, 1.5, 0.0], dtype=np.float32)

        agent.push_memory((obs, 1, 0.5, False, np.array([0.0], dtype=np.float32), next_obs, action_mask, guidance_logits, 0.75, True))

        batches = agent.memory.get_items(np.array([0]))
        assert "guidance_logits" in batches
        assert "guidance_weight" in batches
        assert "guidance_valid" in batches
        assert np.allclose(np.asarray(batches["guidance_logits"][0], dtype=np.float32), guidance_logits)
        assert float(batches["guidance_weight"][0]) == 0.75
        assert int(batches["guidance_valid"][0]) == 1
    assert out[0, 1] > out[0, 0] > out[0, 2]
    assert float(parts["guidance_bias"][0, 1]) == 1.0


def test_decay_action_std_is_silent_by_default(capsys):
    agent = _dummy_agent()
    agent.action_std = 1.5

    PPOAgent.decay_action_std(agent, 0.001, 0.1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert agent.action_std == 1.499
