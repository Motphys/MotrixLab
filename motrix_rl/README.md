# Motrix RL

`motrix-rl` is MotrixLab's RL control plane. It provides framework-neutral training/playback orchestration, checkpoint and
run metadata services, backend selection, and the stable `motrix_rl.frameworks` registration/query entry point. Concrete
algorithm implementations are optional plugins, not implementation-owned modules in this package:

- `motrix-rl-builtin` — built-in `motrix.fastsac` (FastSAC)
- `motrix-rl-skrl` — SKRL PPO with JAX or PyTorch
- `motrix-rl-rslrl` — RSLRL PPO with PyTorch

Plugins register through the `motrix_rl.frameworks` Python entry-point group when installed. The loader is invoked by
registry queries and explicitly before Hydra training-task composition; importing the control plane alone does not
hard-code plugin imports. Provider/trainer contracts live in `motrix_rl.contracts` inside this package.

The control plane has its own general runtime dependencies (including Torch); splitting provider packages does not make
`motrix-rl` itself a dependency-free package. `motrix-rl` never depends on provider plugins, including through optional
extras. The workspace root requires `motrix-rl-builtin` and `motrix-rl-skrl`, ensuring FastSAC and SKRL Torch are available
for the built-in demos. The root `skrl-jax` extra adds the JAX backend; `rslrl` adds the optional RSLRL plugin.
Runtime installation does not force other optional workspace packages.

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
