from evaluation import visualize_path as vp


def test_resolve_visualization_mode_for_macro_refined():
    assert vp._resolve_visualization_mode(True, True) == ("refined", "Refined Primitive Execution")


def test_resolve_visualization_mode_for_macro_raw():
    assert vp._resolve_visualization_mode(True, False) == ("raw", "Raw Primitive Execution")


def test_resolve_visualization_mode_for_continuous_policy():
    assert vp._resolve_visualization_mode(False, True) == ("continuous", "Continuous Policy")


def test_build_output_filename_includes_mode_slug():
    assert vp._build_output_filename(3, "refined") == "path_planning_refined_3.png"


def test_build_run_summary_lines_marks_refinement_not_applicable():
    mode_info = {
        "slug": "continuous",
        "policy_label": "continuous",
        "refinement_enabled": True,
        "refinement_applicable": False,
    }

    lines = vp._build_run_summary_lines(
        mode_info=mode_info,
        checkpoint_path="/tmp/PPO_best.pt",
        primitive_library_path=None,
        level="Complex",
        takeover_trigger_count=2,
    )

    assert "mode=continuous" in lines
    assert "policy=continuous" in lines
    assert "refinement=n/a" in lines
    assert "level=Complex" in lines
    assert "takeover_triggers=2" in lines
    assert "ckpt=PPO_best.pt" in lines