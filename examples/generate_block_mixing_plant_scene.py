import argparse
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from env.scene_generators.block_mixing_plant_generator import (
    generate_block_mixing_plant_scene,
    render_scene,
    sample_navigation_case_from_scene,
)


def _default_output_path(repo_root: str, difficulty: str, seed: int) -> str:
    file_name = f"block_mixing_plant_{difficulty}_seed{seed}.png"
    return os.path.join(repo_root, "outputs", "scenes", file_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate block-based mixing plant navigation scenes.")
    parser.add_argument(
        "--difficulty",
        choices=["Normal", "Complex", "Extrem", "All"],
        default="All",
        help="Which difficulty to render. Use All to generate all three levels.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic scene generation.")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output path. Only valid for a single difficulty.")
    parser.add_argument("--show-parking-bays", action="store_true", help="Overlay parking bay outlines for debugging.")
    args = parser.parse_args()

    difficulties = [args.difficulty] if args.difficulty != "All" else ["Normal", "Complex", "Extrem"]
    if args.output is not None and len(difficulties) != 1:
        raise ValueError("--output can only be used when a single difficulty is selected")

    for difficulty in difficulties:
        output_path = args.output if args.output is not None else _default_output_path(REPO_ROOT, difficulty, int(args.seed))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        scene = generate_block_mixing_plant_scene(difficulty=difficulty, seed=int(args.seed))
        start, dest, nav_meta = sample_navigation_case_from_scene(scene, difficulty, seed=int(args.seed))
        render_scene(
            scene,
            show_parking_bays=True,
            start_pose=start,
            dest_pose=dest,
            start_bay_index=int(nav_meta.get("start_bay_index", -1)),
            dest_bay_index=int(nav_meta.get("dest_bay_index", -1)),
            save_path=output_path,
        )

        print(
            f"saved {difficulty} scene to {output_path} | "
            f"bays={scene.metadata.get('parking_bay_count')} | "
            f"free_ratio={scene.metadata.get('free_ratio'):.3f} | "
            f"start_bay={nav_meta.get('start_bay_index')} | "
            f"goal_bay={nav_meta.get('dest_bay_index')}"
        )


if __name__ == "__main__":
    main()