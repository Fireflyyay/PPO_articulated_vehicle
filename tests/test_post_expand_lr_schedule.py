import pytest

from train.lr_schedule import build_scaled_learning_rates, compute_post_expand_restore_scale


def test_compute_post_expand_restore_scale_is_linear_and_clamped():
    scale0, progress0 = compute_post_expand_restore_scale(
        current_episode=120,
        restore_start_episode=120,
        restore_episodes=300,
        start_scale=0.3,
    )
    assert scale0 == 0.3
    assert progress0 == 0.0

    scale_mid, progress_mid = compute_post_expand_restore_scale(
        current_episode=270,
        restore_start_episode=120,
        restore_episodes=300,
        start_scale=0.3,
    )
    assert progress_mid == 0.5
    assert scale_mid == pytest.approx(0.65)

    scale_end, progress_end = compute_post_expand_restore_scale(
        current_episode=520,
        restore_start_episode=120,
        restore_episodes=300,
        start_scale=0.3,
    )
    assert scale_end == pytest.approx(1.0)
    assert progress_end == 1.0


def test_build_scaled_learning_rates_scales_actor_and_critic_together():
    actor_lr, critic_lr = build_scaled_learning_rates(1e-4, 5e-4, 0.3)
    assert actor_lr == 3e-5
    assert critic_lr == 1.5e-4