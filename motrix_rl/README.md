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

## GPU installation profiles

`motrix-rl[cuda]` and `motrix-rl[rocm]` define the tested NVIDIA and AMD Torch dependency profiles. Without either extra,
Torch uses the default package index; this does not imply a CPU-only build. The workspace forwards its `cuda`/`rocm` extras
to this package, so `sh install.sh --gpu cuda` and `sh install.sh --gpu rocm` remain unchanged.

For an external **uv project consuming Git sources**, select one profile and pin the source revision:

```toml
[project]
dependencies = ["motrix-rl[cuda]"] # or motrix-rl[rocm]

[tool.uv.sources]
motrix-rl = { git = "https://github.com/Motphys/MotrixLab", rev = "<commit>", subdirectory = "motrix_rl" }
```

uv reads this source package's extra-gated PyTorch index routing and resolves its internal workspace dependencies
(`motrix-env-core` and `motrix-deploy`) from the same Git revision. No consumer-side Torch version pins, PyTorch index
mappings, or sibling-package source mappings are needed.

If the consuming project exposes **both** profiles as optional extras, declare both the forwarding and package-qualified
conflict pairs at the consuming root. Workspace resolution collects member-local conflicts, but external Git consumers
need the explicit package-qualified pair for the index-specific source split:

```toml
[project.optional-dependencies]
cuda = ["motrix-rl[cuda]"]
rocm = ["motrix-rl[rocm]"]

[tool.uv]
conflicts = [
    [{ extra = "cuda" }, { extra = "rocm" }],
    [{ package = "motrix-rl", extra = "cuda" }, { package = "motrix-rl", extra = "rocm" }],
]
```

Select one with `uv sync --extra cuda` or `uv sync --extra rocm`. CUDA routing applies on Linux/Windows; ROCm routing is
Linux x86_64 only. On other platforms these profile-specific requirements are inactive and the base Torch dependency
uses the default index. Universal lockfiles that combine GPU profiles and macOS may also need explicit platform splits,
as in the workspace root's `tool.uv.environments`.

Use `uv sync` for profile installation. In uv 0.9.8 and 0.9.14, `uv export` does not correctly preserve GPU selection
through forwarding extras; do not use its exported requirements to install these GPU profiles.

**Wheel/PyPI and pip installations:** extras retain their version requirements, but wheel metadata does not carry
`tool.uv.sources` or index definitions, and pip does not interpret them even for Git sources. Configure the appropriate
[PyTorch wheel index](https://pytorch.org/get-started/locally/) in the consuming installer/project explicitly. In particular,
a `rocm` extra's Torch version alone does not guarantee a ROCm build. GPU selection is explicit, not hardware auto-detection.

## Workspace training

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
