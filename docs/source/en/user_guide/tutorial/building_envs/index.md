# Building Environments Overview

This section explains how to write your own reinforcement learning environment in
MotrixLab. Before the details, three building blocks of the big picture.
## Anatomy of an environment

An environment consists of two parts, each registered under the environment name:

- **Config class**: a `@configclass` data class declaring the scene, simulation
  parameters, and task parameters, registered via `@registry.envcfg("name")`;
- **Environment class**: implements the task logic, registered via
  `@registry.env("name")`.

Once registered, `scripts/view.py`, the trainers, and the replay flow all create the
environment by name through the same registry.


## Lifecycle overview

Both workflows share the same vectorized environment lifecycle; `step(actions)` follows
a fixed sequence:

```{image} /_static/images/tutorial/env-lifecycle-light.svg
:alt: ArrayEnv single control step pipeline: apply_action, physics_step, compute_transition, truncation check, auto-reset, compute_observation
:class: only-light
```

```{image} /_static/images/tutorial/env-lifecycle-dark.svg
:alt: ArrayEnv single control step pipeline: apply_action, physics_step, compute_transition, truncation check, auto-reset, compute_observation
:class: only-dark
```

Every stage is orchestrated by the `ArrayEnv` base class; environment implementations only fill in
hooks and must not re-implement the lifecycle. Semantics:

- `terminated` marks episode-ending conditions such as task failure; `truncated` marks
  the time limit, and `info["time_outs"]` flags rows that timed out without failing;
- environments that are done are reset automatically at the end of each step, and
  observations are recomputed after the reset.

Who implements each stage in the two workflows: DirectEnv hook by hook in
[Writing DirectEnv Environments](direct_env.md#lifecycle), and ManagerEnv driven by the config
groups in [Writing ManagerEnv Environments](manager_env.md#lifecycle).


## Choosing between the two workflows

MotrixLab provides two environment workflows; the difference is where the task logic
lives:

| Aspect      | DirectEnv (direct workflow)            | ManagerEnv (manager workflow)                        |
| ----------- | -------------------------------------- | ---------------------------------------------------- |
| Config base | `DirectEnvCfg`                         | `ManagerBasedEnvCfg`                                 |
| Task logic  | Hand-written inside environment hooks  | Declared per term in config; compiled into a kernel  |
| Rewards     | Hand-written array math in `compute_transition` | `rewards` / `terminations` config groups    |
| Observations| Hand-written assembly in `compute_observation`  | `observations` config groups (`policy`/`value`) |
| Reset logic | Hand-written `reset(env_ids)` override | `sim_reset` config group + command reset hooks       |
| Typical use | Simple or highly customized tasks     | Compositional tasks such as locomotion               |

For a simple task whose logic fits on one screen, the direct workflow is the least
ceremony; once reward, observation, and termination terms multiply and need reuse across
environments, the manager workflow's declarative composition is easier to maintain.


## Rewards and terminations

The two workflows only differ in how rewards are **written**, not in how to design them:

- DirectEnv hand-writes the reward array math in `compute_transition`; ManagerEnv
  declares terms in the `rewards` config group and sums them weighted by `weight`.
  Terminations work the same way, as the `terminated` mask in `compute_transition` or
  the `terminations` config group.
- Design guidance: let each reward term own one clear objective; prefer smooth
  functions (for example exponential decay) over hard thresholds; keep weights in
  config so terms can be tuned one by one; check for reward loopholes that allow high
  reward without the intended behavior.

For concrete examples, see the built-in environments and the rewards-and-terminations
section of [Writing ManagerEnv Environments](manager_env.md).


## Reading path

1. [Writing DirectEnv Environments](direct_env.md): understand the environment skeleton
   from a minimal example;
2. [Writing ManagerEnv Environments](manager_env.md): switch to the manager workflow
   when you need declarative composition;
3. [SceneCfg: Setting Up the Physics Scene](scene.md): model files and simulation
   parameters in detail;
4. [SimBackend: Decoupling from the Simulator](sim_backend.md): the environment–simulator
   boundary, backend selection and integration.

```{toctree}
:hidden:

direct_env
manager_env
scene
sim_backend
```
