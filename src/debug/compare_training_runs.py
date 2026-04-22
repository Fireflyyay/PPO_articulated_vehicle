import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tensorboard.backend.event_processing import event_accumulator
except ModuleNotFoundError:
    event_accumulator = None


SELECTED_TAGS = [
    "success_rate_Normal",
    "success_rate_Complex",
    "success_rate_Extrem",
    "total_reward",
    "avg_reward",
    "actor_loss",
    "critic_loss",
    "takeover/triggered",
    "takeover/step_ratio",
    "takeover/steps",
    "takeover/plan_ms_mean",
    "takeover/fast_prune_ms_mean",
    "takeover/score_ms_mean",
    "takeover/success_when_used",
    "takeover/action_KL",
    "refinement/enabled",
    "refinement/triggered",
    "refinement/plan_count",
    "refinement/plan_rate",
    "refinement/applied_plan_ratio",
    "refinement/feasible_plan_ratio",
    "refinement/plan_length_mean",
    "refinement/attempted_steps",
    "refinement/applied_steps",
    "refinement/attempted_ratio",
    "refinement/applied_ratio",
    "refinement/feasible_ratio",
    "refinement/cost_delta_mean",
    "refinement/cost_delta_sum",
    "refinement/prefix_shrink_ratio",
    "refinement/prefix_shrink_steps_mean",
    "refinement/runtime_ms_mean",
    "refinement/terminal_scale_mean",
    "adaptive/triggered",
    "adaptive/library_size",
    "adaptive/library_size_absolute",
    "adaptive/library_size_from_base",
    "adaptive/library_capacity_utilization",
    "adaptive/post_expand_freeze_remaining",
    "adaptive/validation_success_before",
    "adaptive/validation_success_after",
    "adaptive/validation_extreme_success_before",
    "adaptive/validation_extreme_success_after",
    "adaptive/validation_success_gain",
    "adaptive/validation_extreme_success_gain",
    "adaptive/validation_success_gain_per_added_primitive",
    "adaptive/validation_extreme_success_gain_per_added_primitive",
    "adaptive/success_rate_recent",
    "adaptive/hard_success_rate_recent",
    "adaptive/post_expand_success_uplift_recent",
    "adaptive/post_expand_hard_success_uplift_recent",
    "adaptive/post_expand_success_uplift_per_added_primitive_recent",
    "adaptive/post_expand_hard_success_uplift_per_added_primitive_recent",
    "adaptive/post_expand_episode_delta",
    "lr/actor",
    "lr/critic",
    "lr/post_expand_scale",
    "lr/post_expand_restore_progress",
]

SUCCESS_TAGS = [
    "success_rate_Normal",
    "success_rate_Complex",
    "success_rate_Extrem",
]

PLOT_GROUPS = {
    "success_rates": [
        "success_rate_Normal",
        "success_rate_Complex",
        "success_rate_Extrem",
    ],
    "rewards": ["avg_reward", "total_reward"],
    "losses": ["actor_loss", "critic_loss"],
    "takeover": [
        "takeover/triggered",
        "takeover/step_ratio",
        "takeover/steps",
        "takeover/success_when_used",
        "takeover/action_KL",
    ],
    "refinement": [
        "refinement/triggered",
        "refinement/plan_count",
        "refinement/plan_rate",
        "refinement/applied_plan_ratio",
        "refinement/feasible_plan_ratio",
        "refinement/plan_length_mean",
        "refinement/attempted_ratio",
        "refinement/applied_ratio",
        "refinement/feasible_ratio",
        "refinement/cost_delta_mean",
        "refinement/prefix_shrink_ratio",
        "refinement/runtime_ms_mean",
    ],
    "adaptive": [
        "adaptive/library_size",
        "adaptive/library_size_absolute",
        "adaptive/library_size_from_base",
        "adaptive/library_capacity_utilization",
        "adaptive/post_expand_freeze_remaining",
        "adaptive/success_rate_recent",
        "adaptive/hard_success_rate_recent",
    ],
    "adaptive_uplift": [
        "adaptive/validation_success_gain",
        "adaptive/validation_extreme_success_gain",
        "adaptive/validation_success_gain_per_added_primitive",
        "adaptive/validation_extreme_success_gain_per_added_primitive",
        "adaptive/post_expand_success_uplift_recent",
        "adaptive/post_expand_hard_success_uplift_recent",
        "adaptive/post_expand_success_uplift_per_added_primitive_recent",
        "adaptive/post_expand_hard_success_uplift_per_added_primitive_recent",
        "adaptive/post_expand_episode_delta",
    ],
    "learning_rate": [
        "lr/actor",
        "lr/critic",
        "lr/post_expand_scale",
        "lr/post_expand_restore_progress",
    ],
}

TAG_TITLES = {
    "success_rate_Normal": "Normal success rate",
    "success_rate_Complex": "Complex success rate",
    "success_rate_Extrem": "Extrem success rate",
    "avg_reward": "Average reward",
    "total_reward": "Episode reward",
    "actor_loss": "Actor loss",
    "critic_loss": "Critic loss",
    "takeover/triggered": "Takeover triggered",
    "takeover/step_ratio": "Takeover step ratio",
    "takeover/steps": "Takeover steps per episode",
    "takeover/plan_ms_mean": "Takeover planning latency",
    "takeover/fast_prune_ms_mean": "Takeover fast prune latency",
    "takeover/score_ms_mean": "Takeover scoring latency",
    "takeover/success_when_used": "Success when takeover used",
    "takeover/action_KL": "Policy vs takeover action KL",
    "refinement/enabled": "Primitive refinement enabled",
    "refinement/triggered": "Primitive plan refinement triggered",
    "refinement/plan_count": "Primitive plan refinement count",
    "refinement/plan_rate": "Primitive plan refinement rate",
    "refinement/applied_plan_ratio": "Primitive plan refinement applied ratio",
    "refinement/feasible_plan_ratio": "Primitive plan refinement feasible ratio",
    "refinement/plan_length_mean": "Primitive plan refinement mean plan length",
    "refinement/attempted_steps": "Primitive refinement attempted steps",
    "refinement/applied_steps": "Primitive refinement applied steps",
    "refinement/attempted_ratio": "Primitive refinement attempted ratio",
    "refinement/applied_ratio": "Primitive refinement applied ratio",
    "refinement/feasible_ratio": "Primitive refinement feasible ratio",
    "refinement/cost_delta_mean": "Primitive refinement mean cost drop",
    "refinement/cost_delta_sum": "Primitive refinement total cost drop",
    "refinement/prefix_shrink_ratio": "Primitive refinement prefix shrink ratio",
    "refinement/prefix_shrink_steps_mean": "Primitive refinement mean prefix shrink steps",
    "refinement/runtime_ms_mean": "Primitive refinement latency",
    "refinement/terminal_scale_mean": "Primitive refinement terminal scale",
    "adaptive/library_size": "Adaptive library size",
    "adaptive/library_size_absolute": "Adaptive library size (absolute)",
    "adaptive/library_size_from_base": "Adaptive library growth from base",
    "adaptive/library_capacity_utilization": "Adaptive library capacity utilization",
    "adaptive/post_expand_freeze_remaining": "Freeze episodes remaining",
    "adaptive/validation_success_before": "Validation success before expansion",
    "adaptive/validation_success_after": "Validation success after expansion",
    "adaptive/validation_extreme_success_before": "Validation extreme success before expansion",
    "adaptive/validation_extreme_success_after": "Validation extreme success after expansion",
    "adaptive/validation_success_gain": "Validation success gain",
    "adaptive/validation_extreme_success_gain": "Validation extreme success gain",
    "adaptive/validation_success_gain_per_added_primitive": "Validation success gain per added primitive",
    "adaptive/validation_extreme_success_gain_per_added_primitive": "Validation extreme gain per added primitive",
    "adaptive/success_rate_recent": "Recent success rate",
    "adaptive/hard_success_rate_recent": "Recent hard-scene success rate",
    "adaptive/post_expand_success_uplift_recent": "Post-expand recent success uplift",
    "adaptive/post_expand_hard_success_uplift_recent": "Post-expand hard-scene success uplift",
    "adaptive/post_expand_success_uplift_per_added_primitive_recent": "Post-expand success uplift per added primitive",
    "adaptive/post_expand_hard_success_uplift_per_added_primitive_recent": "Post-expand hard uplift per added primitive",
    "adaptive/post_expand_episode_delta": "Episodes since latest expansion",
    "lr/actor": "Actor learning rate",
    "lr/critic": "Critic learning rate",
    "lr/post_expand_scale": "Post-expand LR scale",
    "lr/post_expand_restore_progress": "Post-expand LR restore progress",
}


@dataclass
class RunData:
    label: str
    run_dir: str
    event_file: str
    best_epoch: Optional[int]
    best_success: Optional[List[float]]
    tags: Dict
    available_tags: List[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two PPO TensorBoard runs.")
    parser.add_argument("--run-a", required=True, help="Path to the first run directory.")
    parser.add_argument("--run-b", required=True, help="Path to the second run directory.")
    parser.add_argument("--label-a", default="run_a", help="Display label for the first run.")
    parser.add_argument("--label-b", default="run_b", help="Display label for the second run.")
    parser.add_argument("--output-dir", required=True, help="Directory to store plots and report.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=101,
        help="Moving-average window for plots. Use 1 to disable smoothing.",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_event_file(run_dir: str) -> str:
    candidates = []
    for name in os.listdir(run_dir):
        if name.startswith("events.out.tfevents"):
            candidates.append(os.path.join(run_dir, name))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    candidates.sort()
    return candidates[0]


def parse_best_txt(run_dir: str):
    path = os.path.join(run_dir, "best.txt")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    epoch = None
    success = None
    try:
        if text.startswith("epoch:"):
            left, right = text.split(", success rate:")
            epoch = int(left.split(":", 1)[1].strip())
            success = json.loads(right.strip().replace("'", '"'))
    except Exception:
        pass
    return epoch, success


def load_scalars(event_file: str) -> Tuple[Dict, List[str]]:
    if event_accumulator is None:
        raise ModuleNotFoundError(
            "tensorboard is required to read event files. Install it with `pip install tensorboard`."
        )
    acc = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    scalar_tags = sorted(acc.Tags().get("scalars", []))
    data = {}
    for tag in scalar_tags:
        events = acc.Scalars(tag)
        data[tag] = {
            "steps": [int(ev.step) for ev in events],
            "values": [float(ev.value) for ev in events],
        }
    return data, scalar_tags


def load_run(label: str, run_dir: str) -> RunData:
    event_file = find_event_file(run_dir)
    best_epoch, best_success = parse_best_txt(run_dir)
    tags, available_tags = load_scalars(event_file)
    return RunData(
        label=label,
        run_dir=run_dir,
        event_file=event_file,
        best_epoch=best_epoch,
        best_success=best_success,
        tags=tags,
        available_tags=available_tags,
    )


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    use_window = min(window, len(values))
    if use_window <= 1:
        return list(values)
    half = use_window // 2
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        seg = values[start:end]
        smoothed.append(sum(seg) / float(len(seg)))
    return smoothed


def last_window_mean(values: List[float], tail: int = 100) -> Optional[float]:
    if not values:
        return None
    seg = values[-min(tail, len(values)) :]
    return sum(seg) / float(len(seg))


def best_point(steps: List[int], values: List[float], prefer: str = "max"):
    if not values:
        return None, None
    best_idx = 0
    if prefer == "min":
        best_idx = min(range(len(values)), key=lambda idx: values[idx])
    else:
        best_idx = max(range(len(values)), key=lambda idx: values[idx])
    return steps[best_idx], values[best_idx]


def extract_trigger_steps(run: RunData) -> List[int]:
    series = run.tags.get("adaptive/triggered")
    if not series:
        return []
    result = []
    for step, value in zip(series["steps"], series["values"]):
        if float(value) >= 0.5:
            result.append(int(step))
    return result


def config_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


ASSIGNMENT_RE = re.compile(r"^([A-Z][A-Z0-9_]+)\s*=\s*(.+)$")


def parse_config_assignments(path: str) -> Dict[str, str]:
    result = {}
    for raw_line in config_lines(path):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ASSIGNMENT_RE.match(stripped)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        result[key] = value
    return result


def diff_configs(run_a: RunData, run_b: RunData) -> List[Dict]:
    file_a = os.path.join(run_a.run_dir, "configs.txt")
    file_b = os.path.join(run_b.run_dir, "configs.txt")
    cfg_a = parse_config_assignments(file_a)
    cfg_b = parse_config_assignments(file_b)
    all_keys = sorted(set(cfg_a.keys()) | set(cfg_b.keys()))
    diffs = []
    for key in all_keys:
        left = cfg_a.get(key, "")
        right = cfg_b.get(key, "")
        if left == right:
            continue
        diffs.append({"key": key, "a": left, "b": right})
    return diffs


def select_config_diffs(config_diffs: List[Dict], prefixes: Tuple[str, ...], explicit: Tuple[str, ...] = ()) -> List[Dict]:
    selected = []
    for item in config_diffs:
        key = item["key"]
        if key in explicit or key.startswith(prefixes):
            selected.append(item)
    return selected


def summarize_series(run: RunData, tag: str) -> Optional[Dict]:
    series = run.tags.get(tag)
    if not series:
        return None
    steps = series["steps"]
    values = series["values"]
    prefer = "min" if tag.endswith("loss") else "max"
    best_step, best_value = best_point(steps, values, prefer=prefer)
    return {
        "count": len(values),
        "first_step": int(steps[0]) if steps else None,
        "last_step": int(steps[-1]) if steps else None,
        "last_value": float(values[-1]) if values else None,
        "tail_mean": last_window_mean(values, tail=100),
        "best_step": best_step,
        "best_value": best_value,
    }


def summarize_success(run: RunData) -> Dict:
    summary = {}
    for tag in SUCCESS_TAGS:
        stats = summarize_series(run, tag)
        if stats is not None:
            summary[tag] = stats
    return summary


def summarize_learning_rate(run: RunData) -> Dict:
    result = {
        "has_lr_logging": "lr/actor" in run.tags,
        "trigger_steps": extract_trigger_steps(run),
    }
    for tag in ["lr/actor", "lr/critic", "lr/post_expand_scale", "lr/post_expand_restore_progress"]:
        stats = summarize_series(run, tag)
        if stats is not None:
            result[tag] = stats
    return result


def extract_positive_steps(run: RunData, tag: str, threshold: float = 1e-9) -> List[int]:
    series = run.tags.get(tag)
    if not series:
        return []
    result = []
    for step, value in zip(series["steps"], series["values"]):
        if float(value) > threshold:
            result.append(int(step))
    return result


def summarize_takeover(run: RunData) -> Dict:
    tags_present = any(tag.startswith("takeover/") for tag in run.available_tags)
    triggered_steps = extract_positive_steps(run, "takeover/triggered", threshold=0.5)
    used_steps = extract_positive_steps(run, "takeover/steps")
    step_ratio = run.tags.get("takeover/step_ratio", {"values": []}).get("values", [])
    takeover_steps = run.tags.get("takeover/steps", {"values": []}).get("values", [])
    success_when_used = run.tags.get("takeover/success_when_used", {"values": []}).get("values", [])
    plan_ms = run.tags.get("takeover/plan_ms_mean", {"values": []}).get("values", [])
    prune_ms = run.tags.get("takeover/fast_prune_ms_mean", {"values": []}).get("values", [])
    score_ms = run.tags.get("takeover/score_ms_mean", {"values": []}).get("values", [])

    used_success_values = [float(v) for v in success_when_used if float(v) > 0.0]

    return {
        "has_takeover_logging": bool(tags_present),
        "triggered_episode_count": len(triggered_steps),
        "used_episode_count": len(used_steps),
        "triggered_steps": triggered_steps,
        "used_steps": used_steps,
        "ever_triggered": len(triggered_steps) > 0,
        "ever_used": len(used_steps) > 0,
        "tail_step_ratio_mean": last_window_mean(step_ratio, tail=100),
        "tail_steps_mean": last_window_mean(takeover_steps, tail=100),
        "max_steps": max(takeover_steps) if takeover_steps else None,
        "mean_success_when_used": (sum(used_success_values) / float(len(used_success_values))) if used_success_values else None,
        "mean_plan_ms": (sum(plan_ms) / float(len(plan_ms))) if plan_ms else None,
        "mean_fast_prune_ms": (sum(prune_ms) / float(len(prune_ms))) if prune_ms else None,
        "mean_score_ms": (sum(score_ms) / float(len(score_ms))) if score_ms else None,
    }


def summarize_adaptive_rounds(run: RunData) -> List[Dict]:
    triggers = extract_trigger_steps(run)
    val_before = run.tags.get("adaptive/validation_success_before")
    val_after = run.tags.get("adaptive/validation_success_after")
    ext_before = run.tags.get("adaptive/validation_extreme_success_before")
    ext_after = run.tags.get("adaptive/validation_extreme_success_after")
    lib_size = run.tags.get("adaptive/library_size")
    lib_size_abs = run.tags.get("adaptive/library_size_absolute")
    success_gain = run.tags.get("adaptive/validation_success_gain")
    extreme_gain = run.tags.get("adaptive/validation_extreme_success_gain")
    success_gain_per_added = run.tags.get("adaptive/validation_success_gain_per_added_primitive")
    extreme_gain_per_added = run.tags.get("adaptive/validation_extreme_success_gain_per_added_primitive")
    rounds = []
    for idx, step in enumerate(triggers):
        item = {"index": idx + 1, "step": int(step)}
        if val_before and idx < len(val_before["values"]):
            item["val_success_before"] = float(val_before["values"][idx])
        if val_after and idx < len(val_after["values"]):
            item["val_success_after"] = float(val_after["values"][idx])
        if ext_before and idx < len(ext_before["values"]):
            item["val_extreme_before"] = float(ext_before["values"][idx])
        if ext_after and idx < len(ext_after["values"]):
            item["val_extreme_after"] = float(ext_after["values"][idx])
        if success_gain and idx < len(success_gain["values"]):
            item["val_success_gain"] = float(success_gain["values"][idx])
        if extreme_gain and idx < len(extreme_gain["values"]):
            item["val_extreme_gain"] = float(extreme_gain["values"][idx])
        if success_gain_per_added and idx < len(success_gain_per_added["values"]):
            item["val_success_gain_per_added"] = float(success_gain_per_added["values"][idx])
        if extreme_gain_per_added and idx < len(extreme_gain_per_added["values"]):
            item["val_extreme_gain_per_added"] = float(extreme_gain_per_added["values"][idx])
        lib_series = lib_size_abs or lib_size
        if lib_series:
            size_step = None
            size_value = None
            for ls, lv in zip(lib_series["steps"], lib_series["values"]):
                if int(ls) == int(step):
                    size_step = int(ls)
                    size_value = float(lv)
                    break
            if size_step is not None:
                item["library_size"] = size_value
        rounds.append(item)
    return rounds


def summarize_adaptive_uplift(run: RunData) -> Dict:
    result = {}
    for tag in [
        "adaptive/validation_success_gain",
        "adaptive/validation_extreme_success_gain",
        "adaptive/validation_success_gain_per_added_primitive",
        "adaptive/validation_extreme_success_gain_per_added_primitive",
        "adaptive/post_expand_success_uplift_recent",
        "adaptive/post_expand_hard_success_uplift_recent",
        "adaptive/post_expand_success_uplift_per_added_primitive_recent",
        "adaptive/post_expand_hard_success_uplift_per_added_primitive_recent",
    ]:
        stats = summarize_series(run, tag)
        if stats is not None:
            result[tag] = stats
    return result


def save_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def plot_group(output_dir: str, name: str, tags: List[str], run_a: RunData, run_b: RunData, smooth_window: int):
    rows = len(tags)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3.6 * rows), squeeze=False)
    any_data = False
    triggers_a = extract_trigger_steps(run_a)
    triggers_b = extract_trigger_steps(run_b)

    for idx, tag in enumerate(tags):
        ax = axes[idx][0]
        ax.set_title(TAG_TITLES.get(tag, tag))
        ax.set_xlabel("episode")
        ax.set_ylabel(tag.split("/")[-1])

        for run, color in ((run_a, "tab:blue"), (run_b, "tab:orange")):
            series = run.tags.get(tag)
            if not series:
                continue
            any_data = True
            steps = series["steps"]
            values = series["values"]
            smooth = moving_average(values, smooth_window)
            ax.plot(steps, values, color=color, alpha=0.20, linewidth=0.9)
            ax.plot(steps, smooth, color=color, linewidth=1.6, label=run.label)

        if run_a.label and triggers_a and tag.startswith(("success_", "avg_", "total_", "adaptive/", "lr/")):
            for step in triggers_a:
                ax.axvline(step, color="tab:blue", alpha=0.08, linewidth=0.8)
        if run_b.label and triggers_b and tag.startswith(("success_", "avg_", "total_", "adaptive/", "lr/")):
            for step in triggers_b:
                ax.axvline(step, color="tab:orange", alpha=0.08, linewidth=0.8)

        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")

    if not any_data:
        plt.close(fig)
        return None

    fig.tight_layout()
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def compare_best(run_a: RunData, run_b: RunData) -> List[Dict]:
    names = ["Normal", "Complex", "Extrem"]
    rows = []
    a = run_a.best_success or []
    b = run_b.best_success or []
    for idx, name in enumerate(names):
        if idx >= len(a) or idx >= len(b):
            continue
        rows.append(
            {
                "scene": name,
                "run_a": float(a[idx]),
                "run_b": float(b[idx]),
                "delta_b_minus_a": float(b[idx] - a[idx]),
            }
        )
    return rows


def format_float(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.4f}"
    return str(value)


def build_interpretation(
    run_a: RunData,
    run_b: RunData,
    best_rows: List[Dict],
    lr_a: Dict,
    lr_b: Dict,
    rounds_a: List[Dict],
    rounds_b: List[Dict],
    takeover_a: Dict,
    takeover_b: Dict,
    config_diffs: List[Dict],
    uplift_a: Dict,
    uplift_b: Dict,
) -> List[str]:
    notes = []
    complex_row = next((row for row in best_rows if row["scene"] == "Complex"), None)
    extrem_row = next((row for row in best_rows if row["scene"] == "Extrem"), None)
    normal_row = next((row for row in best_rows if row["scene"] == "Normal"), None)

    cfg_map = {item["key"]: item for item in config_diffs}

    takeover_enable = cfg_map.get("TAKEOVER_ENABLE")
    if takeover_enable is not None:
        notes.append(
            f"配置快照显示 takeover 主开关发生了变化：{run_a.label} 为 {takeover_enable['a'] or 'missing'}，{run_b.label} 为 {takeover_enable['b'] or 'missing'}。"
        )
    elif takeover_a.get("has_takeover_logging") or takeover_b.get("has_takeover_logging"):
        notes.append(
            f"{run_a.label} 与 {run_b.label} 的训练日志都包含 takeover 指标，但配置快照中没有明确记录两边都含有 TAKEOVER_ENABLE；需要结合运行时标量判断接管是否真实启动。"
        )

    if takeover_a.get("has_takeover_logging") and not takeover_a.get("ever_triggered"):
        notes.append(
            f"{run_a.label} 的 TensorBoard 已记录 takeover 指标，但 takeover/triggered 与 takeover/steps 没有出现正值，说明接管逻辑至少在日志层面从未真正进入执行状态。"
        )

    if takeover_b.get("has_takeover_logging") and not takeover_b.get("ever_triggered"):
        notes.append(
            f"{run_b.label} 的 takeover 指标同样没有正值；结合配置快照里 TAKEOVER_ENABLE=False，这更像是按配置被显式关闭，而不是运行时失效。"
        )

    action_mask_diff = cfg_map.get("ACTION_MASK_MODE")
    if action_mask_diff is not None:
        notes.append(
            f"ACTION_MASK_MODE 从 {action_mask_diff['a']} 变成 {action_mask_diff['b']}；后者的 'hyrbid' 是拼写错误，运行时会被 wrapper 回退到 'hybrid' 默认逻辑，这会让 3 月 12 日比 3 月 2 日更保守地过滤动作。"
        )

    if complex_row and extrem_row:
        if complex_row["delta_b_minus_a"] < 0 and extrem_row["delta_b_minus_a"] < 0:
            notes.append(
                f"{run_b.label} 在中高难度场景的最佳成功率低于 {run_a.label}，Complex 差值 {complex_row['delta_b_minus_a']:.2f}，Extrem 差值 {extrem_row['delta_b_minus_a']:.2f}。"
            )
        elif complex_row["delta_b_minus_a"] < 0 and extrem_row["delta_b_minus_a"] > 0:
            notes.append(
                f"{run_b.label} 出现了难度分化：Complex 最佳成功率比 {run_a.label} 低 {abs(complex_row['delta_b_minus_a']):.2f}，但 Extrem 高 {extrem_row['delta_b_minus_a']:.2f}；说明改动并不是单纯让整体策略更强，而是改变了不同场景的偏置。"
            )

    if normal_row and normal_row["delta_b_minus_a"] > 0:
        notes.append(
            f"{run_b.label} 的 Normal 最佳成功率比 {run_a.label} 提高了 {normal_row['delta_b_minus_a']:.2f}，因此后续调参应优先避免为了提升 Complex/Extrem 而损失已得到的简单场景稳定性。"
        )

    if not lr_a.get("has_lr_logging") and lr_b.get("has_lr_logging"):
        notes.append(
            f"{run_a.label} 的 TensorBoard 未记录学习率标量，而 {run_b.label} 已记录 lr/actor、lr/critic 与 post-expand 恢复进度，因此学习率变化只能对 {run_b.label} 直接观察，对 {run_a.label} 只能结合代码与 adaptive 事件间接推断。"
        )

    if len(rounds_a) > 0 and len(rounds_b) > 0:
        notes.append(
            f"两次实验都触发了多轮 adaptive primitive expansion：{run_a.label} 记录到 {len(rounds_a)} 轮，{run_b.label} 记录到 {len(rounds_b)} 轮，因此性能差异不能简单归因于‘有没有启用扩库’，而要看扩库后的稳定化过程。"
        )

    if len(rounds_b) > 0:
        regressions = 0
        for row in rounds_b:
            before = row.get("val_success_before")
            after = row.get("val_success_after")
            if before is not None and after is not None and after < before:
                regressions += 1
        if regressions > 0:
            notes.append(
                f"{run_b.label} 在 {regressions}/{len(rounds_b)} 次 adaptive round 后，validation_success_after 低于 validation_success_before，说明扩展后并非每次都立即带来收益。"
            )

    gain_a = uplift_a.get("adaptive/validation_success_gain", {}).get("tail_mean")
    gain_b = uplift_b.get("adaptive/validation_success_gain", {}).get("tail_mean")
    if gain_a is not None or gain_b is not None:
        notes.append(
            f"轮次级扩容收益可直接从 validation_success_gain 观察：{run_a.label} 的尾段均值为 {format_float(gain_a)}，{run_b.label} 为 {format_float(gain_b)}。"
        )

    per_added_a = uplift_a.get("adaptive/validation_success_gain_per_added_primitive", {}).get("tail_mean")
    per_added_b = uplift_b.get("adaptive/validation_success_gain_per_added_primitive", {}).get("tail_mean")
    if per_added_a is not None or per_added_b is not None:
        notes.append(
            f"按新增基元归一化后，validation_success_gain_per_added_primitive 更适合比较扩库效率：{run_a.label} 为 {format_float(per_added_a)}，{run_b.label} 为 {format_float(per_added_b)}。"
        )

    scale_stats = lr_b.get("lr/post_expand_scale")
    if scale_stats is not None:
        notes.append(
            f"{run_b.label} 的 post-expand LR scale 最低降到 {format_float(min(run_b.tags.get('lr/post_expand_scale', {}).get('values', [1.0])))}，结合冻结窗口与恢复逻辑，训练在扩展后会经历明显的保守学习阶段。"
        )

    tail_complex_a = summarize_series(run_a, "success_rate_Complex")
    tail_complex_b = summarize_series(run_b, "success_rate_Complex")
    if tail_complex_a and tail_complex_b:
        ma = tail_complex_a.get("tail_mean")
        mb = tail_complex_b.get("tail_mean")
        if ma is not None and mb is not None and mb < ma:
            notes.append(
                f"从末段 100 个记录点均值看，{run_b.label} 的 Complex 成功率尾段均值 {mb:.3f} 低于 {run_a.label} 的 {ma:.3f}，下降不是只出现在单个峰值点。"
            )

    tail_ext_a = summarize_series(run_a, "success_rate_Extrem")
    tail_ext_b = summarize_series(run_b, "success_rate_Extrem")
    if tail_ext_a and tail_ext_b:
        ma = tail_ext_a.get("tail_mean")
        mb = tail_ext_b.get("tail_mean")
        if ma is not None and mb is not None and mb > ma:
            notes.append(
                f"从末段 100 个记录点均值看，{run_b.label} 的 Extrem 成功率尾段均值 {mb:.3f} 高于 {run_a.label} 的 {ma:.3f}，说明其极难场景收益并非只体现在 best.txt 的单点峰值。"
            )

    return notes


def write_report(path: str, run_a: RunData, run_b: RunData, summary: dict):
    best_rows = summary["best_comparison"]
    config_diffs = summary["config_diffs"]
    critical_diffs = summary["critical_config_diffs"]
    takeover_summary = summary["takeover"]
    adaptive_uplift = summary["adaptive_uplift"]
    lines = []
    lines.append("# PPO training run comparison")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Run A: {run_a.label} -> {run_a.run_dir}")
    lines.append(f"- Run B: {run_b.label} -> {run_b.run_dir}")
    lines.append("- takeover, adaptive primitive expansion, learning-rate recovery and scene-wise success are all included")
    lines.append("")
    lines.append("## Best result comparison")
    lines.append("")
    lines.append("| Scene | Run A | Run B | B - A |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in best_rows:
        lines.append(
            f"| {row['scene']} | {row['run_a']:.2f} | {row['run_b']:.2f} | {row['delta_b_minus_a']:+.2f} |"
        )
    lines.append("")
    lines.append("## TensorBoard coverage")
    lines.append("")
    lines.append(f"- {run_a.label}: {len(run_a.available_tags)} scalar tags")
    lines.append(f"- {run_b.label}: {len(run_b.available_tags)} scalar tags")
    lines.append(f"- {run_a.label} has LR tags: {'yes' if 'lr/actor' in run_a.tags else 'no'}")
    lines.append(f"- {run_b.label} has LR tags: {'yes' if 'lr/actor' in run_b.tags else 'no'}")
    lines.append(f"- {run_a.label} has takeover tags: {'yes' if takeover_summary[run_a.label].get('has_takeover_logging') else 'no'}")
    lines.append(f"- {run_b.label} has takeover tags: {'yes' if takeover_summary[run_b.label].get('has_takeover_logging') else 'no'}")
    lines.append(f"- {run_a.label} has adaptive uplift tags: {'yes' if bool(adaptive_uplift[run_a.label]) else 'no'}")
    lines.append(f"- {run_b.label} has adaptive uplift tags: {'yes' if bool(adaptive_uplift[run_b.label]) else 'no'}")
    lines.append("")
    lines.append("## Critical config differences")
    lines.append("")
    if not critical_diffs:
        lines.append("No critical config difference was found between the saved configs.txt snapshots.")
    else:
        lines.append("| Key | Run A | Run B |")
        lines.append("| --- | --- | --- |")
        for item in critical_diffs:
            lines.append(f"| {item['key']} | {item['a']} | {item['b']} |")
    lines.append("")
    lines.append("## Takeover diagnostics")
    lines.append("")
    lines.append("| Run | Logged | Triggered episodes | Used episodes | Tail step-ratio mean | Tail steps mean | Max steps | Mean success when used |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label in (run_a.label, run_b.label):
        item = takeover_summary[label]
        lines.append(
            f"| {label} | {'yes' if item.get('has_takeover_logging') else 'no'} | {item.get('triggered_episode_count', 0)} | {item.get('used_episode_count', 0)} | {format_float(item.get('tail_step_ratio_mean'))} | {format_float(item.get('tail_steps_mean'))} | {format_float(item.get('max_steps'))} | {format_float(item.get('mean_success_when_used'))} |"
        )
    lines.append("")
    lines.append("## Full config differences")
    lines.append("")
    if not config_diffs:
        lines.append("No line-level assignment difference was found between the saved configs.txt snapshots.")
    else:
        lines.append("| Key | Run A | Run B |")
        lines.append("| --- | --- | --- |")
        for item in config_diffs:
            lines.append(f"| {item['key']} | {item['a']} | {item['b']} |")
    lines.append("")
    lines.append("## Adaptive rounds")
    lines.append("")
    for label, rounds in ((run_a.label, summary["adaptive_rounds"][run_a.label]), (run_b.label, summary["adaptive_rounds"][run_b.label])):
        lines.append(f"### {label}")
        if not rounds:
            lines.append("")
            lines.append("No adaptive round trigger was logged.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| Round | Episode | Val before | Val after | Gain | Gain/add | Extreme before | Extreme after | Extreme gain | Extreme gain/add | Library size |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rounds:
            lines.append(
                "| {index} | {step} | {before} | {after} | {gain} | {gain_per_added} | {ext_before} | {ext_after} | {ext_gain} | {ext_gain_per_added} | {size} |".format(
                    index=row.get("index", "N/A"),
                    step=row.get("step", "N/A"),
                    before=format_float(row.get("val_success_before")),
                    after=format_float(row.get("val_success_after")),
                    gain=format_float(row.get("val_success_gain")),
                    gain_per_added=format_float(row.get("val_success_gain_per_added")),
                    ext_before=format_float(row.get("val_extreme_before")),
                    ext_after=format_float(row.get("val_extreme_after")),
                    ext_gain=format_float(row.get("val_extreme_gain")),
                    ext_gain_per_added=format_float(row.get("val_extreme_gain_per_added")),
                    size=format_float(row.get("library_size")),
                )
            )
        lines.append("")
    lines.append("## Adaptive uplift summary")
    lines.append("")
    lines.append("| Metric | Run | Last | Tail mean | Best | Best step |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for label in (run_a.label, run_b.label):
        for tag, stats in adaptive_uplift[label].items():
            lines.append(
                f"| {tag} | {label} | {format_float(stats.get('last_value'))} | {format_float(stats.get('tail_mean'))} | {format_float(stats.get('best_value'))} | {format_float(stats.get('best_step'))} |"
            )
    lines.append("")
    lines.append("## Key metric summaries")
    lines.append("")
    for tag in SELECTED_TAGS:
        sa = summary["series_summary"][run_a.label].get(tag)
        sb = summary["series_summary"][run_b.label].get(tag)
        if sa is None and sb is None:
            continue
        lines.append(f"### {tag}")
        lines.append("")
        lines.append("| Run | Last | Tail mean | Best | Best step |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        if sa is not None:
            lines.append(
                f"| {run_a.label} | {format_float(sa.get('last_value'))} | {format_float(sa.get('tail_mean'))} | {format_float(sa.get('best_value'))} | {format_float(sa.get('best_step'))} |"
            )
        if sb is not None:
            lines.append(
                f"| {run_b.label} | {format_float(sb.get('last_value'))} | {format_float(sb.get('tail_mean'))} | {format_float(sb.get('best_value'))} | {format_float(sb.get('best_step'))} |"
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for item in summary["interpretation"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    for name, figure_path in summary["figures"].items():
        if figure_path is None:
            continue
        lines.append(f"- {name}: {figure_path}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    run_a = load_run(args.label_a, args.run_a)
    run_b = load_run(args.label_b, args.run_b)

    figures = {}
    for name, tags in PLOT_GROUPS.items():
        figures[name] = plot_group(args.output_dir, name, tags, run_a, run_b, args.smooth_window)

    series_summary = {
        run_a.label: {},
        run_b.label: {},
    }
    for tag in SELECTED_TAGS:
        summary_a = summarize_series(run_a, tag)
        summary_b = summarize_series(run_b, tag)
        if summary_a is not None:
            series_summary[run_a.label][tag] = summary_a
        if summary_b is not None:
            series_summary[run_b.label][tag] = summary_b

    best_rows = compare_best(run_a, run_b)
    lr_a = summarize_learning_rate(run_a)
    lr_b = summarize_learning_rate(run_b)
    rounds_a = summarize_adaptive_rounds(run_a)
    rounds_b = summarize_adaptive_rounds(run_b)
    takeover_a = summarize_takeover(run_a)
    takeover_b = summarize_takeover(run_b)
    uplift_a = summarize_adaptive_uplift(run_a)
    uplift_b = summarize_adaptive_uplift(run_b)
    config_diffs = diff_configs(run_a, run_b)
    critical_config_diffs = select_config_diffs(
        config_diffs,
        prefixes=("TAKEOVER_", "AP_", "USE_ADAPTIVE_", "ACTION_MASK_", "PRIMITIVE_", "LR", "GAMMA"),
        explicit=("USE_MOTION_PRIMITIVES", "USE_ACTION_MASK", "ENABLE_GLOBAL_SOFT_GUIDANCE"),
    )

    summary = {
        "runs": {
            run_a.label: {
                "run_dir": run_a.run_dir,
                "event_file": run_a.event_file,
                "best_epoch": run_a.best_epoch,
                "best_success": run_a.best_success,
                "available_tags": run_a.available_tags,
            },
            run_b.label: {
                "run_dir": run_b.run_dir,
                "event_file": run_b.event_file,
                "best_epoch": run_b.best_epoch,
                "best_success": run_b.best_success,
                "available_tags": run_b.available_tags,
            },
        },
        "best_comparison": best_rows,
        "config_diffs": config_diffs,
        "critical_config_diffs": critical_config_diffs,
        "series_summary": series_summary,
        "learning_rate": {
            run_a.label: lr_a,
            run_b.label: lr_b,
        },
        "takeover": {
            run_a.label: takeover_a,
            run_b.label: takeover_b,
        },
        "adaptive_rounds": {
            run_a.label: rounds_a,
            run_b.label: rounds_b,
        },
        "adaptive_uplift": {
            run_a.label: uplift_a,
            run_b.label: uplift_b,
        },
        "figures": figures,
    }
    summary["interpretation"] = build_interpretation(
        run_a,
        run_b,
        best_rows,
        lr_a,
        lr_b,
        rounds_a,
        rounds_b,
        takeover_a,
        takeover_b,
        config_diffs,
        uplift_a,
        uplift_b,
    )

    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    write_report(os.path.join(args.output_dir, "report.md"), run_a, run_b, summary)

    print(json.dumps({
        "report": os.path.join(args.output_dir, "report.md"),
        "summary": os.path.join(args.output_dir, "summary.json"),
        "figures": figures,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()