# Motrix RL

`motrix-rl` provides MotrixLab's framework-neutral training, playback, checkpoint, run metadata, and policy-export
interfaces. Concrete providers currently support SKRL PPO with JAX or PyTorch, RSLRL PPO with PyTorch, and the built-in
FastSAC implementation.

Environment-specific training presets live under `configs/task/`, while shared provider defaults live under
`configs/algo_base/`. From the workspace root, select a preset with Hydra's `task=<env-id>/<method>` syntax (run from the
activated workspace environment, `source .venv/bin/activate`):

```bash
python scripts/train.py task=cartpole/skrl.ppo
python scripts/train.py task=cartpole/rslrl.ppo
python scripts/train.py task=g1-walk-rough/motrix.fastsac
```

Install the extra required by the selected provider before training:

```bash
sh install.sh --skrl-jax   # JAX (Linux only)
sh install.sh --skrl-torch # PyTorch (default)
sh install.sh --rslrl      # RSLRL (PyTorch only)
```
