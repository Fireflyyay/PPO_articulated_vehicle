import os
import sys
import argparse
import numpy as np

# Add project root + src to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.configs import VALID_SPEED, VALID_STEER, PRIMITIVE_LIBRARY_PATH, PRIMITIVE_H
from src.env.car_parking_base import CarParking
from src.primitives.library import load_library
from src.env.wrappers.macro_action_wrapper import MacroActionWrapper


def _resolve_primitive_library_path() -> str:
    src_dir = os.path.join(project_root, "src")
    candidate = os.path.normpath(os.path.join(src_dir, PRIMITIVE_LIBRARY_PATH))
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(project_root, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))
    if os.path.exists(candidate):
        return candidate
    # Fallback to H4 default if present
    fallback = os.path.join(project_root, "data", "primitives_articulated_H4_S11.npz")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Primitive library not found. Tried: {candidate} and {fallback}")


def carparking_internal_scale(action_in: np.ndarray) -> np.ndarray:
    """Replicate CarParking.step() scaling: interpret action_in as [-1, 1]."""
    steer_min, steer_max = VALID_STEER
    speed_min, speed_max = VALID_SPEED
    action_in = np.asarray(action_in, dtype=np.float64)
    out = np.zeros_like(action_in, dtype=np.float64)
    out[0] = 0.5 * (action_in[0] + 1.0) * (steer_max - steer_min) + steer_min
    out[1] = 0.5 * (action_in[1] + 1.0) * (speed_max - speed_min) + speed_min
    # CarParking.vehicle model will clip; we clip here for comparability
    out[0] = float(np.clip(out[0], steer_min, steer_max))
    out[1] = float(np.clip(out[1], speed_min, speed_max))
    return out


def physical_to_normalized(action_phys: np.ndarray) -> np.ndarray:
    """Inverse mapping of CarParking.step() scaling."""
    steer_min, steer_max = VALID_STEER
    speed_min, speed_max = VALID_SPEED
    action_phys = np.asarray(action_phys, dtype=np.float64)
    out = np.zeros_like(action_phys, dtype=np.float64)
    out[0] = 2.0 * (action_phys[0] - steer_min) / (steer_max - steer_min) - 1.0
    out[1] = 2.0 * (action_phys[1] - speed_min) / (speed_max - speed_min) - 1.0
    out = np.clip(out, -1.0, 1.0)
    return out


def _fmt(a: np.ndarray) -> str:
    return "[" + ", ".join([f"{float(x): .4f}" for x in np.asarray(a).ravel()]) + "]"


def main():
    parser = argparse.ArgumentParser(description="Detect macro-action scaling mismatch (physical actions re-scaled as if normalized).")
    parser.add_argument("--primitive_id", type=int, default=None, help="Optional primitive id to test.")
    parser.add_argument("--num", type=int, default=5, help="Number of primitives to sample when primitive_id is not set.")
    args = parser.parse_args()

    lib_path = _resolve_primitive_library_path()
    lib = load_library(lib_path)
    H = int(getattr(lib, "horizon", PRIMITIVE_H))

    rng = np.random.default_rng(0)
    if args.primitive_id is not None:
        pids = [int(args.primitive_id)]
    else:
        n = int(lib.size)
        k = int(min(args.num, n))
        pids = [int(x) for x in rng.choice(n, size=k, replace=False)]

    print(f"Primitive library: {lib_path}")
    print(f"Library size: {lib.size}, horizon H: {H}")
    print(f"Testing primitive IDs: {pids}")

    # 1) Pure formula check (no env)
    print("\n[Check A] Compare one low-level action: physical vs CarParking-internal scaled result")
    for pid in pids:
        a_phys = np.asarray(lib.get_actions(pid)[0], dtype=np.float64)
        a_effective = carparking_internal_scale(a_phys)
        a_norm = physical_to_normalized(a_phys)
        delta = np.abs(a_effective - a_phys)
        print(f"pid={pid:4d} phys={_fmt(a_phys)}  effective_if_scaled={_fmt(a_effective)}  |diff|={_fmt(delta)}  norm_inv={_fmt(a_norm)}")

    # 2) Empirical env check: CarParking.step() changes vehicle.state.speed/steering
    print("\n[Check B] Step CarParking once with (phys) vs (norm_inv) and inspect resulting speed/steer")
    base_env = CarParking(render_mode=None, verbose=False)
    obs, info = base_env.reset(seed=0)

    pid0 = pids[0]
    a_phys0 = np.asarray(lib.get_actions(pid0)[0], dtype=np.float64)
    a_norm0 = physical_to_normalized(a_phys0)

    # Step with physical (this is the problematic path)
    base_env.reset(seed=0)
    base_env.step(a_phys0)
    got_phys = np.array([base_env.vehicle.state.steering, base_env.vehicle.state.speed], dtype=np.float64)

    # Step with normalized inverse (this is what wrapper SHOULD send)
    base_env.reset(seed=0)
    base_env.step(a_norm0)
    got_norm = np.array([base_env.vehicle.state.steering, base_env.vehicle.state.speed], dtype=np.float64)

    print(f"Using pid={pid0}")
    print(f"Input physical action       : {_fmt(a_phys0)}")
    print(f"CarParking applies (scaled) : {_fmt(got_phys)}")
    print(f"Input normalized (inv map)  : {_fmt(a_norm0)}")
    print(f"CarParking applies (scaled) : {_fmt(got_norm)}")

    # 3) Wrapper instrumentation: verify what MacroActionWrapper actually passes into env.step()
    print("\n[Check C] Instrument MacroActionWrapper -> base_env.step() input range")
    base_env = CarParking(render_mode=None, verbose=False)
    base_step = base_env.step
    captured = []

    def step_hook(action):
        if action is not None:
            captured.append(np.asarray(action, dtype=np.float64).copy())
        return base_step(action)

    base_env.step = step_hook

    env = MacroActionWrapper(base_env, lib, H=H)
    env.reset(seed=0)
    env.step(pid0)

    if not captured:
        print("No low-level actions captured (unexpected).")
    else:
        arr = np.stack(captured, axis=0)
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)
        out_of_range = np.any(arr < -1.0001) or np.any(arr > 1.0001)
        print(f"Captured {len(captured)} low-level actions passed into base_env.step().")
        print(f"min={_fmt(mn)}  max={_fmt(mx)}  out_of_[-1,1]_range={out_of_range}")
        print("(If you see out_of_range=True, wrapper is sending physical actions into a normalized-action env.)")


if __name__ == "__main__":
    main()
