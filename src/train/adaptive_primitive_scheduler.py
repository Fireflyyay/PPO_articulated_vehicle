from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AdaptiveSchedulerState:
    last_round_episode: int = -10**9
    round_id: int = 0
    post_expand_freeze_remaining: int = 0


class AdaptivePrimitiveScheduler:
    """Trigger controller for adaptive primitive expansion.

    The scheduler is intentionally simple: it watches recent success rates and
    enforces a cooldown between rounds.
    """

    def __init__(self, config):
        self.cfg = config
        self.state = AdaptiveSchedulerState()

    def get_stage(self, episode_idx: int) -> str:
        if episode_idx < int(getattr(self.cfg, "AP_WARMUP_EPISODES", 1000)):
            return "warmup"
        return "train"

    def should_trigger(self, stats: Dict[str, float], episode_idx: int) -> bool:
        """Return True if an expansion round should run now."""
        if not bool(getattr(self.cfg, "USE_ADAPTIVE_PRIMITIVE_EXPANSION", False)):
            return False

        if episode_idx < int(getattr(self.cfg, "AP_WARMUP_EPISODES", 1000)):
            return False

        cooldown = int(getattr(self.cfg, "AP_COOLDOWN_EPISODES", 500))
        if (episode_idx - int(self.state.last_round_episode)) < cooldown:
            return False

        sr = float(stats.get("success_rate_recent", 0.0))
        sr_hard = float(stats.get("hard_success_rate_recent", 0.0))

        if sr < float(getattr(self.cfg, "AP_TRIGGER_SUCCESS_RATE", 0.40)):
            return False
        if sr_hard < float(getattr(self.cfg, "AP_TRIGGER_HARD_SUCCESS_RATE", 0.15)):
            return False

        # plateau signal (optional)
        plateau = bool(stats.get("plateau", True))
        if not plateau:
            return False

        return True

    def on_round_started(self, episode_idx: int) -> int:
        self.state.round_id += 1
        self.state.last_round_episode = int(episode_idx)
        self.state.post_expand_freeze_remaining = int(getattr(self.cfg, "AP_POST_EXPAND_FREEZE_EPISODES", 100))
        return int(self.state.round_id)

    def on_round_finished(self, metrics: Optional[Dict[str, float]] = None) -> None:
        # reserved for future logic
        return

    def tick_post_expand_freeze(self) -> int:
        if self.state.post_expand_freeze_remaining > 0:
            self.state.post_expand_freeze_remaining -= 1
        return int(self.state.post_expand_freeze_remaining)
