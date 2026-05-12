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

    @staticmethod
    def _maybe_float(stats: Dict[str, float], key: str) -> Optional[float]:
        if key not in stats:
            return None
        try:
            return float(stats[key])
        except Exception:
            return None

    def should_trigger(self, stats: Dict[str, float], episode_idx: int) -> bool:
        """Return True if an expansion round should run now."""
        if not bool(getattr(self.cfg, "USE_ADAPTIVE_PRIMITIVE_EXPANSION", False)):
            return False

        if episode_idx < int(getattr(self.cfg, "AP_WARMUP_EPISODES", 1000)):
            return False

        cooldown = int(getattr(self.cfg, "AP_COOLDOWN_EPISODES", 500))
        if (episode_idx - int(self.state.last_round_episode)) < cooldown:
            return False

        try:
            if int(stats.get("capacity_remaining", 1)) <= 0:
                return False
        except Exception:
            pass

        sr = float(stats.get("success_rate_recent", 0.0))
        sr_hard = float(stats.get("hard_success_rate_recent", 0.0))

        if sr < float(getattr(self.cfg, "AP_TRIGGER_SUCCESS_RATE", 0.40)):
            return False
        if sr_hard < float(getattr(self.cfg, "AP_TRIGGER_HARD_SUCCESS_RATE", 0.15)):
            return False

        if bool(getattr(self.cfg, "AP_TRIGGER_REQUIRE_TREND", True)):
            sr_delta = self._maybe_float(stats, "success_rate_recent_delta")
            sr_hard_delta = self._maybe_float(stats, "hard_success_rate_recent_delta")
            if sr_delta is not None and sr_delta < float(getattr(self.cfg, "AP_TRIGGER_MIN_SUCCESS_DELTA", 0.0)):
                return False
            if sr_hard_delta is not None and sr_hard_delta < float(getattr(self.cfg, "AP_TRIGGER_MIN_HARD_DELTA", 0.0)):
                return False

        if bool(getattr(self.cfg, "AP_TRIGGER_REQUIRE_POSITIVE_UPLIFT", True)):
            added_since_baseline = self._maybe_float(stats, "added_since_baseline")
            uplift = self._maybe_float(stats, "post_expand_hard_success_uplift_per_added_primitive_recent")
            if added_since_baseline is not None and added_since_baseline > 0.0 and uplift is not None:
                if uplift < float(getattr(self.cfg, "AP_TRIGGER_MIN_HARD_UPLIFT_PER_ADDED", 0.0)):
                    return False

            validation_uplift = self._maybe_float(stats, "last_validation_extreme_success_gain_per_added_primitive")
            if validation_uplift is not None:
                if validation_uplift < float(getattr(self.cfg, "AP_TRIGGER_MIN_VALIDATION_EXTREME_GAIN_PER_ADDED", 0.0)):
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
