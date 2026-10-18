# MotrixLab

MotrixLab is a reinforcement learning framework for robotics. This application integration package
provides the unified `mxlab` command, environment and robot visualization, and a single installation entry point
for the backend-neutral environment framework, policy deployment runtime, RL control plane, built-in FastSAC
provider, and MotrixSim simulation backend.

## Installation

Use Python 3.10–3.13:

```bash
python -m pip install motrixlab
```

For a beta release, add `--pre`. All five component packages are pinned to the
same release version. The MotrixSim backend installs `motrixsim` from public PyPI.
PyTorch and its dependencies are installed through the component packages;
GPU-specific wheel selection and driver setup remain environment-specific.

## Usage and scope

`mxlab` is the command-line entry point for training, playback, environment and robot visualization, export, and deployment:

```bash
mxlab --help
mxlab --version
mxlab train --help
mxlab play --help
mxlab view --help
mxlab export-onnx --help
mxlab export-deploy --help
mxlab deploy --help
mxlab deploy inspect artifact=/path/to/policy.deploy
```

With the selected environment or robot package installed, inspect an environment with random actions or a robot in its default pose:

```bash
mxlab view env=<env-id>
mxlab view robot=<robot-id>
```

Subcommands preserve their component's parameter syntax, including Hydra `key=value`
overrides and configuration-directory options. `export-deploy` creates an artifact;
`deploy` runs it. `mxlab --version` is the sole version operation. Install this package
for the CLI; the standalone component packages retain their Python APIs and internal
callable command adapters, not independent CLI entry points.

The CLI imports only the selected command; top-level help and version do not load
Torch or a simulator. Environment and robot visualization belongs to this application package;
its `view.yaml` configuration is packaged in `motrixlab.configs`. Training, playback, export, and deployment
implementations remain in their component packages. Public Python APIs remain in `motrix_env_core`,
`motrix_deploy`, `motrix_rl`, `motrix_rl_builtin`, and `motrix_env_motrixsim`; the integration package does not re-export them.
The built-in RL provider and MotrixSim backend are discovered via package entry points.

This installation does **not** include the repository's task environments, robot
assets, SKRL/RSLRL providers, or deployment backend plugins. Running a task requires
installing or supplying its environment and assets separately. For the full
repository task collection and GPU-aware source installation, follow the
[repository quick start](https://github.com/Motphys/MotrixLab#-quick-start).

See the [user documentation](https://motrixlab.readthedocs.io) for framework and
training usage.
