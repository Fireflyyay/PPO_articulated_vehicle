"""Sanity checks for discrete PPO policy logits.

Run:
  python src/debug/verify_discrete_logits.py

What it checks:
1) In discrete mode (motion primitives), PPOAgent forces actor to output *logits* (no tanh).
2) Demonstrates how tanh-clipping logits limits softmax confidence.
"""

import os
import sys
import math

import numpy as np
import torch


# Ensure src/ is importable when running from repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from model.agent.ppo_agent import PPOAgent
from model.network import MultiObsEmbedding


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def test_agent_discrete_disables_tanh() -> None:
    configs = {
        "discrete": True,
        "observation_shape": (10,),
        "action_dim": 7,
        "batch_size": 8,
        "actor_layers": {
            "input_dim": 10,
            "hidden_size": 64,
            "output_size": 7,
            # Intentionally set True here; PPOAgent(discrete=True) must override it to False.
            "use_tanh_output": True,
            "orthogonal_init": False,
        },
        "critic_layers": {
            "input_dim": 10,
            "hidden_size": 64,
            "output_size": 1,
            "use_tanh_output": False,
            "orthogonal_init": False,
        },
    }

    agent = PPOAgent(configs, discrete=True)
    assert agent.actor_net.output_layer is None, (
        "Discrete PPO actor should output raw logits (no tanh). "
        "If this fails, logits may be clipped to [-1,1] and policy confidence will be limited."
    )


def test_tanh_clipping_limits_confidence() -> None:
    # Build two networks differing only by tanh output.
    cfg_tanh = {
        "input_dim": 4,
        "hidden_size": 64,
        "output_size": 5,
        "use_tanh_output": True,
        "orthogonal_init": False,
    }
    cfg_raw = dict(cfg_tanh)
    cfg_raw["use_tanh_output"] = False

    net_tanh = MultiObsEmbedding(cfg_tanh)
    net_raw = MultiObsEmbedding(cfg_raw)

    # Make the last linear layer produce known large logits via bias.
    # With tanh, these will be squashed to [-1, 1].
    with torch.no_grad():
        last_tanh = net_tanh.net[-1]
        last_raw = net_raw.net[-1]
        assert isinstance(last_tanh, torch.nn.Linear)
        assert isinstance(last_raw, torch.nn.Linear)

        bias = torch.linspace(-5.0, 5.0, steps=cfg_tanh["output_size"])
        last_tanh.weight.zero_()
        last_raw.weight.zero_()
        last_tanh.bias.copy_(bias)
        last_raw.bias.copy_(bias)

    x = torch.zeros(1, cfg_tanh["input_dim"])
    logits_tanh = net_tanh(x).squeeze(0).detach().cpu().numpy()
    logits_raw = net_raw(x).squeeze(0).detach().cpu().numpy()

    assert np.max(np.abs(logits_tanh)) <= 1.0 + 1e-6, "tanh output should be within [-1,1]"
    assert np.max(np.abs(logits_raw)) > 1.0, "raw logits should not be bounded to [-1,1]"

    p_tanh = _softmax(logits_tanh)
    p_raw = _softmax(logits_raw)

    ratio_tanh = float(np.max(p_tanh) / np.min(p_tanh))
    ratio_raw = float(np.max(p_raw) / np.min(p_raw))

    # Theoretical upper bound for tanh-clipped logits: exp(max_delta) with delta<=2.
    # (Numerically it may be slightly smaller.)
    assert ratio_tanh <= math.exp(2.0) + 1e-3, f"unexpectedly high ratio for tanh logits: {ratio_tanh}"
    assert ratio_raw > math.exp(2.0) + 1.0, (
        "raw logits should allow much higher confidence than tanh-clipped logits"
    )


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    test_agent_discrete_disables_tanh()
    test_tanh_clipping_limits_confidence()

    print("OK: discrete PPO uses raw logits (no tanh).")
    print("OK: demonstration shows tanh-clipping limits softmax confidence.")


if __name__ == "__main__":
    main()
