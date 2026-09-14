# Basic Framework

MotrixLab separates environment implementation, simulation backends, training methods, configuration, and
command-line orchestration into independent layers. This page unfolds top-down: first the typical
simulation-RL loop, then which part of that loop each MotrixLab package covers, and finally the
capabilities of each part.

## The typical simulation RL loop

Simulation reinforcement learning is driven by the interaction loop between a **policy** and an
**environment**, with a **trainer** updating the policy on top of that loop:

```{image} /_static/images/tutorial/rl-loop-light.svg
:alt: The simulation RL loop: the policy outputs actions to the environment, the environment returns observations, rewards, and termination flags, and the trainer collects transitions and updates the policy parameters
:class: only-light
```

```{image} /_static/images/tutorial/rl-loop-dark.svg
:alt: The simulation RL loop: the policy outputs actions to the environment, the environment returns observations, rewards, and termination flags, and the trainer collects transitions and updates the policy parameters
:class: only-dark
```

- The policy takes the environment observation oₜ as input and outputs an action aₜ.
- The environment runs N simulated instances in parallel, applies the action, advances the physics, and
  produces the reward rₜ, terminated (failure) / truncated (time limit), and the next observation oₜ₊₁.
- The next observation oₜ₊₁ returns to the policy as the input for the next action; the reward and the
  termination flags **never enter the policy network** — together with (oₜ, aₜ) they form the transitions
  handed to the trainer, which updates the policy parameters with the selected RL algorithm, repeating
  until convergence.

One timing detail: the observation the environment produces in response to aₜ becomes the policy's
input at step t+1; both interaction edges in the figure are labeled with the current cycle's oₜ, aₜ, rₜ.
With this loop in mind, every MotrixLab package has a place in the figure.

## Which part of the loop each package covers

MotrixLab is a UV workspace made of nine packages, grouped by the loop above:

| Package                 | Loop stage                     | Responsibility                                                       |
| ----------------------- | ------------------------------ | -------------------------------------------------------------------- |
| `motrix_env_core`       | Policy–environment interaction | Backend-agnostic environment framework: `EnvCfg`, registry, frontends, lifecycle |
| `motrix_envs`           | Policy–environment interaction | Built-in environments, robot models, and task assets                 |
| `motrix_rl`             | Policy training and updates    | RL framework integrations (providers, trainers) and training tools   |
| `configs/`, `scripts/`  | Configuration and orchestration | Hydra algorithm base configs and Task recipes; train / play / view / export entry points |
| `motrix_deploy*`        | Policy deployment              | Framework-agnostic artifacts and runtime contracts, MuJoCo replay and Unitree hardware backends |

A simulation backend (such as `motrix_env_motrixsim`) is isolated behind the `SimBackend` interface —
when using an environment you normally do not need to care which one it is. To select or integrate a
backend, see [SimBackend: Decoupling from the Simulator](building_envs/sim_backend.md).

## One full training run

```bash
python scripts/train.py task=cartpole/skrl.ppo num_envs=1024
```

1. Hydra composes `configs/train.yaml`, `configs/algo_base/skrl.ppo.yaml`, and
   `configs/task/cartpole/skrl.ppo.yaml`; `num_envs=1024` only overrides this run.
2. The runner resolves the SKRL PPO provider and automatically picks an available JAX/Torch backend.
3. The trainer creates the registered `cartpole` environment through the registry and starts optimizing.
4. Run metadata, the final Task snapshot, logs, and the checkpoint manifest land in `runs/cartpole/`.

The same environment can have several Task recipes without touching the environment implementation:

```bash
python scripts/train.py task=cartpole/skrl.ppo
python scripts/train.py task=cartpole/rslrl.ppo
```

## What the layering buys you

1. **Environment reuse**: one registered environment can be trained by multiple RL frameworks.
2. **Typed configuration**: provider schemas reject misspelled or mistyped YAML/CLI values before training.
3. **Reproducible experiments**: every run stores its final Task config and provider identity.
4. **Multiple backends**: the backend is a config-level choice; environment implementations never see the
   concrete simulator.
5. **Easy extension**: a new environment needs a registration plus a Task YAML; a new RL integration needs
   a provider and trainer; a new simulator needs a registered SimBackend.
