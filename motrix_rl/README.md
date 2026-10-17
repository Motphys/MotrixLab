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

## Usage

Installing `motrix-rl` provides `mxtrain`, `mxplay`, and `mxexport`, backed by `motrix_rl.commands.train`,
`motrix_rl.commands.play`, and the `motrix_rl.commands.export` dispatcher. `mxexport` requires an explicit
`onnx` or `deploy` subcommand; there is no default subcommand. Their primary Hydra configurations ship in
`motrix_rl.configs`; the commands do not require a repository checkout or a particular working directory.
For training and playback, install an environment package and the selected provider too.
For the built-in environments, activate the installed workspace environment (`source .venv/bin/activate`) and run from
the workspace root:

```bash
mxtrain task=cartpole/skrl.ppo
mxtrain task=cartpole/rslrl.ppo
mxtrain task=g1-walk-rough/motrix.fastsac
mxplay env=cartpole
mxplay env=cartpole policy=/path/to/run
mxexport onnx run_dir=/path/to/run output=artifacts/policy.onnx
mxexport deploy run_dir=/path/to/run output=artifacts/policy.deploy
mxexport deploy env=go2-walk-flat output=artifacts/go2-walk-flat.deploy
```

`mxexport onnx` exports a saved run's best policy without selecting a `task` or creating an environment. Install the provider used by
that run; see the [ONNX export tutorial](../docs/source/en/user_guide/tutorial/advanced/export_onnx.md) for supported
configurations. `mxexport deploy` creates a self-contained deployment artifact; specify exactly one of `run_dir` or
`env` (the latter selects the latest metadata-backed run). Its numerical checks use `validation.*`, while ONNX export
uses `parity.*`. The `export_onnx.yaml` and `export_deploy.yaml` configurations ship in `motrix_rl.configs` and follow
the same project-first discovery described below, falling back to packaged files when the project does not provide them.

Deployment profile compilers come from installed environment plugins, without hardcoded environment imports in the CLI.
See the [deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md#add-a-deployment-profile-compiler)
for the plugin contract. `motrix-rl` depends on `motrix-deploy` for artifact creation; `motrix-deploy` does not depend on RL.

The equivalent module commands are `python -m motrix_rl.commands.train`, `python -m motrix_rl.commands.play`, and
`python -m motrix_rl.commands.export onnx` / `python -m motrix_rl.commands.export deploy`. The subcommands are implemented
in `motrix_rl.commands.export_onnx` and `motrix_rl.commands.export_deploy`; the repository's export scripts are thin
wrappers around those commands. In an already installed uv environment, use
`uv run --no-sync mxtrain ...` (or `mxplay` / `mxexport`) to avoid re-resolving GPU profiles.
Run artifacts are written under `runs/` relative to the working directory by default.

Primary train/play configurations and shared provider defaults ship in `motrix_rl.configs`, with sources under
`motrix_rl/src/motrix_rl/configs/` (`algo_base/` for defaults). Concrete task presets belong to the consuming project:
the workspace keeps its built-in demos under root `configs/task/`, discovered through the current directory's
`./configs`, just like an external project. The packaged train configuration uses a `task: null` placeholder so bare
`--help` works without choosing a default environment. Actual training requires an explicit `task=...` selection;
mandatory `TrainConfig` fields reject execution without a task. Playback uses saved run configuration and needs no task
config directory. Environment packages do not own training configuration, and there is no built-in Hydra search-path plugin.

Hydra configuration discovery and runtime environment discovery are separate. Core registry queries automatically load
installed environment providers from `motrix_env_core.envs`; train/play need no registration-module configuration or
side-effect imports. Environment packages remain independent of training/Hydra Task configs. See
[core discovery](../motrix_env_core/README.md#environment-discovery) for callback and dependency-error semantics.

In the workspace, install the extra required by the selected provider before training:

```bash
sh install.sh --skrl-jax   # JAX (Linux only)
sh install.sh --skrl-torch # PyTorch (default)
sh install.sh --rslrl      # RSLRL (PyTorch only)
```

## External project integration

The CLI discovers `./configs` in the current working directory when it exists. It does not search parent directories,
and a missing default directory is ignored. An explicit `--config-dir <path>` overrides this default directory.
The selected project directory takes precedence over packaged defaults, including same-name `algo_base` configurations;
files not provided by the project fall back to the package. Hydra selects the first matching file, rather than merging
same-name files from multiple directories:

```bash
mxtrain task=my-task/skrl.ppo
mxtrain --config-dir /path/to/project/configs task=my-task/skrl.ppo
```

Install the external environment package; train and play discover its registered environments automatically.
Playback uses the same discovery: `mxplay env=my-task`. See
[Environment registration](../docs/source/en/user_guide/tutorial/building_envs/direct_env.md#environment-registration)
for packaging an environment.

An installed consuming project can expose packaged training presets through a Hydra `SearchPathPlugin` without coupling configuration
discovery to environment registration. For example, put the following in `src/hydra_plugins/my_project_searchpath.py`,
with task YAML files in `src/my_project/configs/task/`:

```python
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class MyProjectSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if not any(
            item.provider == "main" and item.path == "pkg://motrix_rl.configs"
            for item in search_path.get_path()
        ):
            return
        search_path.append(provider="my_project", path="pkg://my_project.configs")
```

Make `my_project.configs` an importable package (include `__init__.py`) and include its YAML resources in the wheel.
Package both `my_project` and `hydra_plugins/my_project_searchpath.py` in the installed distribution. Keep `hydra_plugins`
a shared namespace package: **do not add `hydra_plugins/__init__.py`**. Hydra discovers installed plugin modules in that
namespace; no custom entry point or working-directory import convention is needed. Once installed, run
`mxtrain task=my-task/skrl.ppo` from any directory. Environment callbacks only register environments; they do not
install Hydra search paths. RL provider plugins still register separately through `motrix_rl.frameworks`.
