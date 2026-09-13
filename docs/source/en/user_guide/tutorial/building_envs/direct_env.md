# Writing DirectEnv Environments

`DirectEnv` is MotrixLab's direct-workflow frontend: the environment implementation holds
the backend-neutral simulator interface `SimBackend` directly (accessed as `self.sim`)
and implements action application, rewards, termination, and observations inside fixed
lifecycle hooks. It suits tasks with highly custom logic that do not need declarative
per-term composition of rewards, observations, and terminations.

This page starts with a complete minimal example to build the overall picture, then
unfolds the concepts behind it; for when to choose DirectEnv over the manager workflow,
see the [Building Environments Overview](index.md).
## A minimal example

The built-in `cartpole` environment (full source under
`motrix_envs/src/motrix_envs/basic/cartpole/`) shows the pattern. A DirectEnv environment
consists of a **config class** and an **environment class**, each registered separately:

```python
import os

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnv, DirectEnvCfg
from motrix_env_core.sim import DofPositionQuery, DofVelocityQuery
from motrix_env_core.sim.write import CtrlTargetsWrite, JointPositionWrite, JointVelocityWrite

model_file = os.path.dirname(__file__) + "/cartpole.xml"

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
}


@registry.envcfg("cartpole")
@configclass
class CartPoleEnvCfg(DirectEnvCfg):
    """Move a cart to keep an inverted pendulum upright.

    zh_CN: 移动小车以保持倒立摆直立。
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    max_episode_seconds: float = 10
    reset_noise_scale: float = 0.01


@registry.env("cartpole")
class CartPoleEnv(DirectEnv):
    def __init__(self, cfg: CartPoleEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.compile_writes({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.compile_writes(
            {
                "position": JointPositionWrite(("slider", "hinge")),
                "velocity": JointVelocityWrite(("slider", "hinge")),
            },
            reset=True,
        )
        self._action_space = gym.spaces.Box(-3.0, 3.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, env_ids: np.ndarray):
        rows = len(env_ids)
        scale = self._cfg.reset_noise_scale
        self._reset_program.buffer("position")[env_ids] = np.random.uniform(
            -scale, scale, (rows, 2)
        ).astype(np.float32)
        self._reset_program.buffer("velocity")[env_ids] = np.random.uniform(
            -scale, scale, (rows, 2)
        ).astype(np.float32)
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
        return {}

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState):
        self._ctrl_writes.buffer("ctrl")[:] = actions.astype(np.float32, copy=False)
        self._ctrl_writes.execute()
        return state

    def compute_transition(self, state: ArrayEnvState):
        self.sim_data.execute()
        dof_pos = self.sim_data["dof_pos"]
        cart_pos, angle = dof_pos[:, 0], dof_pos[:, 1]
        state.reward = np.ones((self.num_envs,), dtype=np.float32)
        state.terminated = (
            np.isnan(angle)
            | (np.abs(angle) > 0.2)
            | (cart_pos < -0.8)
            | (cart_pos > 0.8)
        )
        return state

    def compute_observation(self, state: ArrayEnvState):
        obs = np.concatenate([self.sim_data["dof_pos"], self.sim_data["dof_vel"]], axis=-1)
        return state.replace(obs=obs)
```

Once registered, preview it with `python scripts/view.py env=cartpole` and start
training by creating a Task as described in
[Task Configuration](../training/task_config.md).

Three core elements of DirectEnv are visible in the example:

1. The **config** inherits `DirectEnvCfg` and declares the scene and task parameters;
2. The **environment class** compiles queries and read/write programs against `self.sim`
   in its constructor — see [SimBackend: Decoupling from the Simulator](sim_backend.md);
3. The **task logic** lives in the four lifecycle hooks `reset` / `apply_action` /
   `compute_transition` / `compute_observation`.

The next section builds the big picture with the lifecycle figure, before the details of
the config and the SimBackend programs.

## Lifecycle

`DirectEnv` inherits `ArrayEnv`: the step / auto-reset / truncation lifecycle is owned by
the base class and must not be re-implemented in subclasses. In the `step(actions)`
sequence below, the four orange-highlighted stages are exactly what a direct workflow
fills in:

```{image} /_static/images/tutorial/direct-env-lifecycle-light.svg
:alt: DirectEnv single control step pipeline: apply_action (hook), physics_step, compute_transition (hook), truncation check, auto-reset (reset(env_ids)), compute_observation (hook)
:class: only-light
```

```{image} /_static/images/tutorial/direct-env-lifecycle-dark.svg
:alt: DirectEnv single control step pipeline: apply_action (hook), physics_step, compute_transition (hook), truncation check, auto-reset (reset(env_ids)), compute_observation (hook)
:class: only-dark
```

Hooks a subclass implements:

| Hook                                        | Responsibility                                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `reset(env_ids)`                            | Write reset state (randomized initial poses, ...) for the selected rows and return an info dict; observations are produced afterwards by `compute_observation` |
| `apply_action(actions, state)`              | Write the action into the simulator (usually ctrl targets)                             |
| `compute_transition(state)`                 | Execute the read program and derive `state.reward`, `state.terminated`, ...; this is the only full data refresh of a step and must **not** write `state.obs` |
| `compute_observation(state)`                | Assemble `state.obs` purely from already refreshed simulator data, without further reads |
| `observation_space` / `action_space`        | Defined as properties, usually pre-built in `__init__`                                 |

Semantics:

- `terminated` marks episode-ending conditions such as task failure; `truncated` marks
  the time limit at `max_episode_steps`. `ArrayEnv` combines both into `done` and
  triggers auto-reset; `info["time_outs"]` flags rows that timed out without failing.
- The environment dimension must use vectorized NumPy operations; plain loops are only
  allowed over a fixed number of joints, feet, or terms.
- Constants (initial poses, space definitions, query names) are precomputed in
  `__init__` or config construction, not recreated inside the step loop.


## Configuration

`DirectEnvCfg` inherits `EnvCfg`; direct-workflow configs are written as
`class MyEnvCfg(..., DirectEnvCfg)`. `EnvCfg` provides these common fields:

| Field                 | Meaning                                                              |
| --------------------- | -------------------------------------------------------------------- |
| `scene`               | `SceneCfg`, the scene and model source; required, `validate()` fails otherwise |
| `sim`                 | `SimCfg`, simulation parameters (`dt`, `solver_iterations`, ...)     |
| `ctrl_dt`             | Control step in seconds, default `0.01`                              |
| `max_episode_seconds` | Maximum episode length in seconds, converted to `max_episode_steps = max_episode_seconds / ctrl_dt`; `None` means unlimited |
| `render_spacing`      | Spacing between environments in grid rendering                       |

Each control step advances the simulation by `sim_substeps = round(ctrl_dt / sim.dt)`
physics substeps, so `sim.dt` must be less than or equal to `ctrl_dt`. Model files and
simulation parameters are covered in [Setting Up the Physics Scene](scene.md).



## Registration rules

- **The config is registered before the environment class**: `@registry.env("name")`
  requires a config with the same name registered via `@registry.envcfg("name")`.
- The config registration target may be a config class or a zero-argument factory with
  a return type annotation (useful for returning customized config instances).
- The registry infers the frontend type from the class hierarchy (`DirectEnv` belongs
  to the `"np"` data backend); never specify it manually.
- The first docstring line and the `zh_CN:` line of the config serve as the
  environment's English and Chinese descriptions shown in the environment overview.
