import numpy as np
import torch
from types import SimpleNamespace

from primitives.trajectory_miner import TrajectoryMiner
from model.agent.ppo_agent import PPOAgent
from primitives.generate_primitives import generate_primitives
from primitives.library import PrimitiveLibrary


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


def test_family_action_dim_stays_fixed_when_gamma_bins_and_variants_change(tmp_path):
    path_a = tmp_path / "family_small_g5_v1.npz"
    path_b = tmp_path / "family_small_g9_v3.npz"

    generate_primitives(H=1, S=11, output_path=str(path_a), gamma_bins=5, variant_count=1, family_preset="small")
    generate_primitives(H=1, S=11, output_path=str(path_b), gamma_bins=9, variant_count=3, family_preset="small")

    lib_a = PrimitiveLibrary(str(path_a), load_sidecars=False)
    lib_b = PrimitiveLibrary(str(path_b), load_sidecars=False)

    assert lib_a.family_count == lib_b.family_count
    assert lib_a.action_dim == lib_b.action_dim == lib_a.family_count
    assert lib_a.gamma_bin_count != lib_b.gamma_bin_count
    assert lib_a.size != lib_b.size

    agent = _make_discrete_ppo(action_dim=lib_a.action_dim)
    last = agent.actor_net.net[-1]
    assert last.out_features == lib_a.action_dim


def test_compute_novelty_uses_delta_difference_when_actions_match():
    miner = TrajectoryMiner()
    library = SimpleNamespace(
        actions=np.array([[[0.0, 1.0]]], dtype=np.float64),
        deltas=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        horizon=1,
    )
    cfg = SimpleNamespace(
        AP_NOVELTY_ACTION_L2_SCALE=1.0,
        AP_NOVELTY_DELTA_WEIGHT=0.5,
        AP_NOVELTY_DELTA_L2_SCALE=1.0,
    )

    novelty = miner.compute_novelty(
        np.array([[0.0, 1.0]], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        library,
        cfg,
    )

    assert novelty > 0.0
