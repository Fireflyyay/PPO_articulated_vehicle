import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.configs import PRIMITIVE_LIBRARY_PATH

def analyze_primitives():
    # Load primitives
    if os.path.exists(PRIMITIVE_LIBRARY_PATH):
        lib_path = PRIMITIVE_LIBRARY_PATH
    else:
        # relative resolve
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        lib_path = os.path.join(project_root, "data", os.path.basename(PRIMITIVE_LIBRARY_PATH))
    
    if not os.path.exists(lib_path):
        print(f"Error: Primitive library not found at {lib_path}")
        return

    print(f"Loading primitives from {lib_path}")
    data = np.load(lib_path, allow_pickle=True)
    actions = data['actions'] # [N, H, 2]
    deltas = data['deltas']   # [N, 4] dx, dy, dtheta, dgamma
    
    N, H, D = actions.shape
    print(f"Total Primitives (N): {N}")
    print(f"Horizon (H): {H}")
    
    # 1. Speed Distribution (Forward vs Backward)
    # actions[:, :, 1] is speed
    mean_speeds = np.mean(actions[:, :, 1], axis=1)
    n_forward = np.sum(mean_speeds > 0.01)
    n_backward = np.sum(mean_speeds < -0.01)
    n_stop = np.sum(np.abs(mean_speeds) <= 0.01)
    
    print("\n--- 1. Speed Distribution ---")
    print(f"Forward Primitives: {n_forward} ({n_forward/N:.2%})")
    print(f"Backward Primitives: {n_backward} ({n_backward/N:.2%})")
    print(f"Static/Stop Primitives: {n_stop} ({n_stop/N:.2%})")
    
    # 2. Steer Distribution
    # actions[:, :, 0] is steer
    mean_steers = np.mean(actions[:, :, 0], axis=1)
    print("\n--- 2. Steer Distribution ---")
    print(f"Steer Mean: {np.mean(mean_steers):.4f}")
    print(f"Steer Min: {np.min(mean_steers):.4f}")
    print(f"Steer Max: {np.max(mean_steers):.4f}")
    print(f"Steer Unique Values: {np.unique(np.round(mean_steers, 4))}")

    # 3. Displacement Distribution
    displacements = np.linalg.norm(deltas[:, :2], axis=1)
    print("\n--- 3. Displacement Distribution (End-to-End) ---")
    print(f"Mean Displacement: {np.mean(displacements):.4f}")
    print(f"Min Displacement: {np.min(displacements):.4f}")
    print(f"Max Displacement: {np.max(displacements):.4f}")
    print("Displacement Percentiles [10, 50, 90]:", np.percentile(displacements, [10, 50, 90]))
    
    # Check for tiny displacements
    tiny_threshold = 0.1
    n_tiny = np.sum(displacements < tiny_threshold)
    print(f"Prims with displacement < {tiny_threshold}: {n_tiny} ({n_tiny/N:.2%})")

    # 4. Heading/Gamma Change
    dtheta = deltas[:, 2]
    dgamma = deltas[:, 3]
    print("\n--- 4. Attitude Change ---")
    print("Delta Theta Percentiles [10, 50, 90]:", np.percentile(dtheta, [10, 50, 90]))
    print("Delta Gamma Percentiles [10, 50, 90]:", np.percentile(dgamma, [10, 50, 90]))

if __name__ == "__main__":
    analyze_primitives()
