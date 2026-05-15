from types import SimpleNamespace

import numpy as np
import torch

from model.agent.ppo_agent import PPOAgent


def _dummy_agent():
    agent = object.__new__(PPOAgent)
    agent.configs = SimpleNamespace(soft_mask_logit_lambda=2.0, soft_mask_small_value=1e-8)
    agent.action_logit_bias = None
    return agent


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


def test_decay_action_std_is_silent_by_default(capsys):
    agent = _dummy_agent()
    agent.action_std = 1.5

    PPOAgent.decay_action_std(agent, 0.001, 0.1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert agent.action_std == 1.499
