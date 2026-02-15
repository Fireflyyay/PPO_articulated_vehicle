# PPO Articulated Vehicle

This project implements an articulated vehicle (tractor-trailer) parking agent using Proximal Policy Optimization (PPO). The agent is trained to navigate and park an articulated vehicle in a custom environment.

## Features

- **Articulated Vehicle Simulation:** accurately models the kinematics of a tractor-trailer system.
- **PPO Reinforcement Learning:** Uses Proximal Policy Optimization for stable and efficient training.
- **Custom Environment:** specific scenarios for parking tasks (Normal, Complex, Extreme).
- **Visualization:** Tools to visualize the vehicle's path and parking performance.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gymnasium
- Shapely
- Pygame
- Matplotlib
- Tensorboard
- Heapdict
- Einops
- TQDM

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Fireflyyay/PPO_articulated_vehicle.git
   cd PPO_articulated_vehicle
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## File Structure

- `src/configs.py`: Global configuration parameters (vehicle dimensions, training hyperparameters).
- `src/env/`: Environment implementation.
  - `car_parking_base.py`: Main environment logic.
  - `vehicle.py`: Vehicle kinematics and state.
  - `map_base.py`, `parking_map_normal.py`: Map definitions.
- `src/model/`: Neural network and agent implementation.
  - `agent/ppo_agent.py`: PPO Algorithm implementation.
  - `network.py`: Neural network architecture.
- `src/train/`: Training scripts.
  - `train_ppo.py`: Main training loop.
- `src/evaluation/`: Evaluation and visualization tools.
  - `visualize_path.py`: Script to visualize the agent's path.

## Usage

### Training

To start training the PPO agent, run the following command from the `src` directory:

```bash
cd src
python train/train_ppo.py
```

You can optionally resume training from a checkpoint:

```bash
python train/train_ppo.py --agent_ckpt /path/to/checkpoint.pt
```

Training logs are saved in the `src/log/exp` directory and can be viewed using Tensorboard.

### Visualization

To visualize the agent's performance and path:

```bash
cd src
python evaluation/visualize_path.py
```

## Configuration

You can modify the simulation and training parameters in `src/configs.py`. Key configurations include:

- **Vehicle Dimensions:** `WHEEL_BASE`, `FRONT_HANG`, `TRAILER_LENGTH`, etc.
- **Training Params:** `ACTOR_CONFIGS`, `CRITIC_CONFIGS`, `MAX_EPISODES`, etc.

## License

[MIT License](LICENSE)