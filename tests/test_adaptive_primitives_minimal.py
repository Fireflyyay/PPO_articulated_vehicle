import numpy as np
import torch

from primitives.trajectory_miner import TrajectoryMiner
from model.agent.ppo_agent import PPOAgent, expand_discrete_actor_output


def test_resample_segment_to_H_shape_and_endpoints():
    miner = TrajectoryMiner()
    # T=5 -> H=10 linear ramp
    u = np.stack([
        np.linspace(-1.0, 1.0, 5),
        np.linspace(2.0, -2.0, 5),
    ], axis=1)

    out = miner.resample_segment_to_H(u, H=10)
    assert out.shape == (10, 2)
    # endpoints should match
    assert np.allclose(out[0], u[0])
    assert np.allclose(out[-1], u[-1])


def _make_discrete_ppo(action_dim: int):
    obs_dim = 10
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
    }
    return PPOAgent(cfg, discrete=True)


def test_expand_discrete_actor_output_copies_old_rows():
    agent = _make_discrete_ppo(action_dim=4)

    # Set last layer weights to a known pattern
    last = agent.actor_net.net[-1]
    assert last.out_features == 4
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        for i in range(4):
            last.weight[i, i % last.in_features] = float(i + 1)
            last.bias[i] = float(-(i + 1))

    old_w = last.weight.detach().cpu().clone()
    old_b = last.bias.detach().cpu().clone()

    expand_discrete_actor_output(agent, new_action_dim=6, init_mode="zero")

    new_last = agent.actor_net.net[-1]
    assert new_last.out_features == 6

    new_w = new_last.weight.detach().cpu()
    new_b = new_last.bias.detach().cpu()

    assert torch.allclose(new_w[:4], old_w)
    assert torch.allclose(new_b[:4], old_b)
