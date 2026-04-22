from debug.compare_training_runs import RunData, SELECTED_TAGS, TAG_TITLES, summarize_adaptive_rounds, summarize_adaptive_uplift


def _make_run(tags):
    available = sorted(tags.keys())
    return RunData(
        label="run_x",
        run_dir="/tmp/run_x",
        event_file="/tmp/events.out.tfevents.fake",
        best_epoch=None,
        best_success=None,
        tags=tags,
        available_tags=available,
    )


def test_summarize_adaptive_rounds_includes_uplift_fields():
    run = _make_run(
        {
            "adaptive/triggered": {"steps": [100, 200], "values": [1.0, 1.0]},
            "adaptive/validation_success_before": {"steps": [100, 200], "values": [0.2, 0.3]},
            "adaptive/validation_success_after": {"steps": [100, 200], "values": [0.4, 0.35]},
            "adaptive/validation_extreme_success_before": {"steps": [100, 200], "values": [0.1, 0.2]},
            "adaptive/validation_extreme_success_after": {"steps": [100, 200], "values": [0.25, 0.22]},
            "adaptive/validation_success_gain": {"steps": [100, 200], "values": [0.2, 0.05]},
            "adaptive/validation_extreme_success_gain": {"steps": [100, 200], "values": [0.15, 0.02]},
            "adaptive/validation_success_gain_per_added_primitive": {"steps": [100, 200], "values": [0.04, 0.01]},
            "adaptive/validation_extreme_success_gain_per_added_primitive": {"steps": [100, 200], "values": [0.03, 0.004]},
            "adaptive/library_size_absolute": {"steps": [100, 200], "values": [48.0, 56.0]},
        }
    )

    rounds = summarize_adaptive_rounds(run)

    assert len(rounds) == 2
    assert rounds[0]["val_success_gain"] == 0.2
    assert rounds[0]["val_extreme_gain"] == 0.15
    assert rounds[0]["val_success_gain_per_added"] == 0.04
    assert rounds[0]["val_extreme_gain_per_added"] == 0.03
    assert rounds[0]["library_size"] == 48.0


def test_summarize_adaptive_uplift_collects_new_tags():
    run = _make_run(
        {
            "adaptive/validation_success_gain": {"steps": [100, 200], "values": [0.1, 0.2]},
            "adaptive/validation_success_gain_per_added_primitive": {"steps": [100, 200], "values": [0.02, 0.04]},
            "adaptive/post_expand_success_uplift_recent": {"steps": [110, 210], "values": [0.03, 0.05]},
        }
    )

    summary = summarize_adaptive_uplift(run)

    assert "adaptive/validation_success_gain" in summary
    assert summary["adaptive/validation_success_gain"]["last_value"] == 0.2
    assert "adaptive/validation_success_gain_per_added_primitive" in summary
    assert "adaptive/post_expand_success_uplift_recent" in summary


def test_refinement_tags_are_exposed_in_compare_script():
    assert "refinement/applied_ratio" in SELECTED_TAGS
    assert "refinement/cost_delta_mean" in SELECTED_TAGS
    assert "refinement/plan_length_mean" in SELECTED_TAGS
    assert TAG_TITLES["refinement/prefix_shrink_ratio"] == "Primitive refinement prefix shrink ratio"
