from types import SimpleNamespace

from train.adaptive_primitive_scheduler import AdaptivePrimitiveScheduler


def _make_cfg(**overrides):
    cfg = {
        "USE_ADAPTIVE_PRIMITIVE_EXPANSION": True,
        "AP_WARMUP_EPISODES": 100,
        "AP_COOLDOWN_EPISODES": 50,
        "AP_TRIGGER_SUCCESS_RATE": 0.4,
        "AP_TRIGGER_HARD_SUCCESS_RATE": 0.15,
        "AP_TRIGGER_REQUIRE_TREND": True,
        "AP_TRIGGER_MIN_SUCCESS_DELTA": 0.0,
        "AP_TRIGGER_MIN_HARD_DELTA": 0.0,
        "AP_TRIGGER_REQUIRE_POSITIVE_UPLIFT": True,
        "AP_TRIGGER_MIN_HARD_UPLIFT_PER_ADDED": 0.0,
        "AP_TRIGGER_MIN_VALIDATION_EXTREME_GAIN_PER_ADDED": 0.0,
        "AP_POST_EXPAND_FREEZE_EPISODES": 10,
    }
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_scheduler_triggers_when_thresholds_and_trend_are_met():
    scheduler = AdaptivePrimitiveScheduler(_make_cfg())

    stats = {
        "success_rate_recent": 0.45,
        "hard_success_rate_recent": 0.20,
        "success_rate_recent_delta": 0.01,
        "hard_success_rate_recent_delta": 0.02,
        "capacity_remaining": 4,
    }

    assert scheduler.should_trigger(stats, episode_idx=150) is True


def test_scheduler_blocks_when_hard_trend_is_negative():
    scheduler = AdaptivePrimitiveScheduler(_make_cfg())

    stats = {
        "success_rate_recent": 0.45,
        "hard_success_rate_recent": 0.20,
        "success_rate_recent_delta": 0.03,
        "hard_success_rate_recent_delta": -0.01,
        "capacity_remaining": 4,
    }

    assert scheduler.should_trigger(stats, episode_idx=150) is False


def test_scheduler_blocks_when_previous_uplift_is_negative():
    scheduler = AdaptivePrimitiveScheduler(_make_cfg())

    stats = {
        "success_rate_recent": 0.45,
        "hard_success_rate_recent": 0.20,
        "success_rate_recent_delta": 0.03,
        "hard_success_rate_recent_delta": 0.01,
        "capacity_remaining": 4,
        "added_since_baseline": 6,
        "post_expand_hard_success_uplift_per_added_primitive_recent": -0.01,
    }

    assert scheduler.should_trigger(stats, episode_idx=150) is False


def test_scheduler_blocks_when_validation_gain_per_added_is_negative():
    scheduler = AdaptivePrimitiveScheduler(_make_cfg())

    stats = {
        "success_rate_recent": 0.45,
        "hard_success_rate_recent": 0.20,
        "success_rate_recent_delta": 0.03,
        "hard_success_rate_recent_delta": 0.01,
        "capacity_remaining": 4,
        "last_validation_extreme_success_gain_per_added_primitive": -0.001,
    }

    assert scheduler.should_trigger(stats, episode_idx=150) is False


def test_scheduler_blocks_when_library_is_full():
    scheduler = AdaptivePrimitiveScheduler(_make_cfg())

    stats = {
        "success_rate_recent": 0.50,
        "hard_success_rate_recent": 0.25,
        "success_rate_recent_delta": 0.02,
        "hard_success_rate_recent_delta": 0.02,
        "capacity_remaining": 0,
    }

    assert scheduler.should_trigger(stats, episode_idx=150) is False