# MotrixLab

MotrixLab is a reinforcement learning framework for robotics. This metapackage
provides a single installation entry point for the backend-neutral environment
framework, policy deployment runtime, RL control plane, built-in FastSAC provider,
and MotrixSim simulation backend.

## Installation

Use Python 3.10 or 3.11:

```bash
python -m pip install motrixlab
```

For a beta release, add `--pre`. All five component packages are pinned to the
same release version. The MotrixSim backend installs `motrixsim` from public PyPI.
PyTorch and its dependencies are installed through the component packages;
GPU-specific wheel selection and driver setup remain environment-specific.

## Usage and scope

The component packages provide `mxtrain`, `mxplay`, `mxexport`, and
`motrix-deploy`. For example, inspect the training configuration with:

```bash
mxtrain --help
```

The built-in RL provider and MotrixSim environment backend are discovered via
package entry points. Import APIs remain under `motrix_env_core`, `motrix_deploy`,
`motrix_rl`, `motrix_rl_builtin`, and `motrix_env_motrixsim`: this metadata-only
package does not introduce an `import motrixlab` API or duplicate their code.

This installation does **not** include the repository's task environments, robot
assets, SKRL/RSLRL providers, or deployment backend plugins. Running a task requires
installing or supplying its environment and assets separately. For the full
repository task collection and GPU-aware source installation, follow the
[repository quick start](https://github.com/Motphys/MotrixLab#-quick-start).

See the [user documentation](https://motrixlab.readthedocs.io) for framework and
training usage.
