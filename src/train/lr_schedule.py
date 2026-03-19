from __future__ import annotations


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_post_expand_restore_scale(
    current_episode: int,
    restore_start_episode: int,
    restore_episodes: int,
    start_scale: float,
    end_scale: float = 1.0,
):
    """Return linear restore scale and normalized progress for post-expand LR recovery."""
    if int(restore_episodes) <= 0:
        return float(end_scale), 1.0

    elapsed = max(0, int(current_episode) - int(restore_start_episode))
    progress = clamp01(float(elapsed) / float(int(restore_episodes)))
    scale = float(start_scale) + (float(end_scale) - float(start_scale)) * progress
    return float(scale), float(progress)


def build_scaled_learning_rates(base_actor_lr: float, base_critic_lr: float, scale: float):
    scaled = float(scale)
    return float(base_actor_lr) * scaled, float(base_critic_lr) * scaled