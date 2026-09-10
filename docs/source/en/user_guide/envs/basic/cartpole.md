# CartPole

CartPole is a classic control task in reinforcement learning. The goal is to keep the pole balanced by controlling the cart's left-right movement.
![cartpole](/_static/images/poster/cartpole.jpg)

## Task Description

-   **State Space**: Cart position, cart velocity, pole angle, pole angular velocity
-   **Action Space**: Apply force left or right
-   **Reward Function**: +1 reward for each step the pole stays upright
-   **Termination Conditions**: Pole angle exceeds ±15 degrees or episode length exceeds 10 seconds

## Quick Start

### 1. Environment Preview

```bash
python scripts/view.py env=cartpole
```

### 2. Start Training

```bash
python scripts/train.py task=cartpole/skrl.ppo
```

### 3. View Training Progress

```bash
tensorboard --logdir runs/cartpole
```

### 4. Test Training Results

```bash
python scripts/play.py env=cartpole
```

> **Tip**: The system finds the latest metadata-backed run under `runs/cartpole/` and loads its `best_policy` artifact. Use `policy=...` to select another checkpoint from a metadata-backed run.

## Expected Results

-   Pole angle stays within ±5 degrees most of the time
-   Cart displacement range is reasonable

## Performance

![Episode return](/_static/images/performance/cartpole.svg)

The current curve uses the single random seed `42`; no confidence interval is shown. The bottom axis reports total environment steps. The top axis reports elapsed wall time.

| Training Task       | TensorBoard metric             | Seeds |
| ------------------- | ------------------------------ | ----- |
| `cartpole/skrl.ppo` | `Reward / Total reward (mean)` | `42`  |

## Troubleshooting

If training performance is poor, you can try:

1. Adjust learning rate (try 1e-4 to 1e-3)
2. Increase number of environments (more parallel training)
3. Adjust reward function weights
4. Check if physical parameters are reasonable
