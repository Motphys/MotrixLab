# Writing ManagerEnv Environments

`ManagerEnv` is the manager-workflow frontend: actions, commands, resets, observations,
rewards, and terminations are not hand-written in the environment class. Instead they are
declared term by term in the config groups of `ManagerBasedEnvCfg`, and the manager
runtime executes them as batched torch kernels. It suits tasks with many terms
that reuse composable mdp terms, such as locomotion and whole-body tracking.

For choosing between the workflows, see the
[Building Environments Overview](index.md). This page first shows a minimal skeleton to
establish the whole picture, then walks through each term group: what it does and how to
define one.

## A minimal skeleton

The manager workflow does not subclass `ManagerEnv`: write a `ManagerBasedEnvCfg`
config, then register the generic environment class under the environment name:

```python
from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.manager import (
    ManagerBasedEnvCfg,
    ManagerEnv,
)

# —— the config groups are declared below and omitted here for brevity ——

@registry.envcfg("my-task")
@configclass
class MyTaskEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = SceneCfg(file="my_model.xml")
    actions: ActionsCfg = ActionsCfg()          # at least one action term is required
    observations: ObservationsCfg = ObservationsCfg()  # the policy group is required
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    sim_reset: ManagerResetCfg = ResetCfg()


registry.env("my-task")(ManagerEnv)
```

Each config group is a `@configclass` data class whose field names are the term names
and whose values are term config objects. The sections below cover each group: what the
term does and how to define it.

## Lifecycle

`ManagerEnv` also inherits `ArrayEnv`; the step sequence and the auto-reset / truncation
semantics match DirectEnv. The manager frontend's public currency is uniformly torch:
`step(actions)` and `reset(env_ids)` (including `sim_reset_ids`) take integer
`torch.Tensor`s, e.g. `env.reset(torch.tensor([0, 2], dtype=torch.long))`; passing a
numpy array raises a `TypeError`. The figure below integrates the term callbacks into the
pipeline: green parts run inside manager kernels — plain functions the manager runtime
resolves by signature and dispatches as batched tensor calls — purple parts are the
manager's host-side Python callbacks, and blue parts are the framework — the
orchestrating stages and its own data plumbing:

```{image} /_static/images/tutorial/manager-env-lifecycle-light.svg
:alt: ManagerEnv single control step pipeline with a done-row branch: apply_action → physics_step → compute_transition → truncation check → done? splits done rows into auto-reset microflow and alive rows toward compute_observation
:class: only-light
```

```{image} /_static/images/tutorial/manager-env-lifecycle-dark.svg
:alt: ManagerEnv single control step pipeline with a done-row branch: apply_action → physics_step → compute_transition → truncation check → done? splits done rows into auto-reset microflow and alive rows toward compute_observation
:class: only-dark
```

Two differences from DirectEnv: term callbacks are not hand-written in the environment
class — they are declared in the config groups and executed as batched torch kernels by
the manager runtime; kernel compilation and warm-up happen at the first `init_state()`
(startup), never inside the step loop.

## Action terms (actions)

**What they do**: each control step, an action term transforms the policy's action batch
into actuator controls and writes them to the simulator. Each term's action space
(one-dimensional `Box`, float32) is concatenated in declaration order into the
environment's `action_space`, and the policy action is split the same way when
distributed to the terms.

**How to define one**: an action term has two parts —

1. An `ActionCfg` subclass declaring static parameters and the actuator route; its
   `__call__(env, actuators)` returns the runtime term. With `actuator_names` an empty
   tuple `()` controls all actuators, explicit names control only the listed ones, and
   one actuator may belong to a single term. `None` means the term writes no actuator
   targets.
2. An `ActionState` subclass — a plain frozen dataclass — holds the term's persistent
   state (`torch.Tensor` fields, one row per environment) and static model data
   (scalars or batch-free tensors); the manager appends the shared raw-action history
   (`action_queue`, `action_ptr`). An `ActionTerm` subclass takes
   `(action_space, state)` in `__init__` and implements `_process(actions)` — transform
   the action batch into this term's routed actuator controls — and `_reset(env_ids)` —
   clear persistent state for selected rows. The base class owns the raw-action
   history: its `process()` / `reset()` wrap yours.

```python
@dataclass(frozen=True, slots=True)
class MyJointActionState(ActionState):
    default_angles: torch.Tensor     # static model data: per-actuator default pose (A,)
    action_scales: torch.Tensor      # static model data: per-actuator scale (A,)


class MyJointAction(ActionTerm):
    state: MyJointActionState

    def _process(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.state.action_scales + self.state.default_angles

    def _reset(self, env_ids: torch.Tensor) -> None:
        ...  # resample per-episode randomization for the reset rows


@configclass(kw_only=True)
class MyJointActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()
    scale: float = 0.5

    def __call__(self, env: ManagerEnv, actuators) -> ActionTerm:
        default_angles = ...  # (A,) resolved from the robot's default key pose
        state = MyJointActionState(
            action_queue=torch.zeros((env.num_envs, 2, len(actuators)), dtype=torch.float32),
            action_ptr=torch.zeros(1, dtype=torch.int64),
            default_angles=default_angles,
            action_scales=torch.full((len(actuators),), self.scale),
        )
        space = joint_position_action_space(actuators, default_angles, self.scale)
        return MyJointAction(space, state)
```

## Observation terms (observations)

**What they do**: assemble simulator state and commands into the inputs of the policy
and value networks. Only the `policy` and `value` groups exist, and `policy` is
required: `policy` feeds the actor (noise allowed), `value` feeds the critic (usually
noise-free and may include privileged information). Term outputs are concatenated in
declaration order into the observation vector.

**How to define one**: an observation term is an `ObservationTermCfg` subclass whose
`__call__(env)` returns `ObsTerm(size, dispatch, *args)`. `size` is the term's output
width; the dispatch has the shape `def xxx_obs(ctx, out, *args) -> None` and writes the
term's observation into `out` — `ctx.sim["key"]` reads data declared in `queries`, and
noise is passed in as a scalar argument and added inside the dispatch.

```python
def projected_gravity_obs(ctx: ManagerContext, out: torch.Tensor,
                          noise_amplitude: float) -> None:
    ...  # compute and write out (width 3)
    out.add_(ctx.rand.next_uniform(3) * noise_amplitude)


@configclass(kw_only=True)
class ProjectedGravityObsCfg(ObservationTermCfg):
    noise_amplitude: float = 0.05

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        return ObsTerm(3, projected_gravity_obs, self.noise_amplitude)
```

Simulation data an observation depends on is declared through the environment's
`queries` group or the term's `required_sim_queries()` — see the Queries section below.

## Reward terms (rewards)

**What they do**: each control step, a reward term scores the current state with a
scalar; the total reward is the sum of the terms weighted by `weight` — a positive
`weight` rewards, a negative one penalizes.

**How to define one**: a reward term is a `RewardTermCfg` subclass (the base carries
`weight: float`) whose `__call__(env)` returns `RewardTerm(dispatch, *args)`. The
dispatch has the shape `def xxx_reward(ctx, *args) -> torch.Tensor` — one score per
environment — and reads declared data through `ctx.sim["key"]`.

```python
def base_height_reward(ctx: ManagerContext, target_z: float,
                       sigma: float) -> torch.Tensor:
    error = ctx.sim["robot_base_pos"][:, 2] - target_z
    return torch.exp(-(error * error) / (sigma * sigma))


@configclass(kw_only=True)
class BaseHeightRewardCfg(RewardTermCfg):
    target_z: float
    sigma: float

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        return RewardTerm(base_height_reward, self.target_z, self.sigma)
```

## Termination terms (terminations)

**What they do**: each control step, a termination term outputs a boolean per
environment; any true term ends the episode as `terminated` (as opposed to the
time-limit `truncated`).

**How to define one**: a termination term is a `TerminationTermCfg` subclass whose
`__call__(env)` returns `TerminationTerm(dispatch, *args, metric_names=(...))`. The
dispatch has the shape `def xxx_termination(ctx, *args) -> torch.Tensor` — a boolean
mask, one per environment; the optional `metric_names` register per-environment
quantities the dispatch writes into `ctx.metrics` as readable metrics.

```python
def bad_orientation_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    tilt_sq = ...  # squared tilt of the base, per environment
    ctx.metrics["base_tilt"][:, 0] = tilt_sq
    return tilt_sq > threshold * threshold


@configclass(kw_only=True)
class BadOrientationTerminationCfg(TerminationTermCfg):
    threshold: float

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        return TerminationTerm(bad_orientation_termination,
                               self.threshold,
                               metric_names=("base_tilt",))
```

## Reset terms (sim_reset)

**What they do**: on episode resets (and on mid-transition recomputation requested by a
command term, see the commands section) they rewrite the simulator state of the selected
environment rows — initial poses, velocities, joint angles, ... — executed by the reset
kernel in declaration order.

**How to define one**: a reset term is a `ResetTermCfg` subclass whose `__call__(env)`
returns `ResetTerm(dispatch, *args, writes={...})`. `writes` declares which simulator
state the term writes (`BodyPositionWrite`, `JointVelocityWrite`, ...; the dict keys are
the write channels inside the dispatch); the dispatch computes the values to write.

```python
def _reset_body_pos(ctx: ManagerContext, sim_writes: Map[torch.Tensor],
                    spawn: tuple[float, float, float],
                    noise: tuple[float, float, float]) -> None:
    position = sim_writes["position"]  # compact (K, ...) rows for this reset pass
    ...  # write the randomized initial position


@configclass(kw_only=True)
class BodyPosResetCfg(ResetTermCfg):
    spawn: tuple[float, float, float]
    noise: tuple[float, float, float] = (0.02, 0.02, 0.005)

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        return ResetTerm(
            _reset_body_pos,
            self.spawn,
            self.noise,
            writes={"position": BodyPositionWrite((base_link,))},
        )
```

## Command terms (commands)

**What they do**: maintain command state and advance it every cycle — velocity command
sampling, curriculum, reference-motion progression, ... A command term also publishes
per-environment metrics: declare tensor fields with `metric(name=...)` (or
`metric(name=..., scalar=True)` for batch-level gauges) and read them back through
`env.state.metrics` / `env.metrics`. Most tasks can reuse built-in commands and
never need a custom one.

**How to define one**: a command term is a `CommandCfg` subclass whose `__call__(env)`
returns a `CommandTerm` subclass — a plain frozen dataclass — implementing three
instance-method callbacks:

| Callback          | Runs in           | Responsibility                                              |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| `update(ctx)`     | evaluate kernel   | update this cycle's derived command data                     |
| `advance(ctx)`    | transition kernel | advance persistent command state; setting `ctx.sim_reset_requested` requests the row's simulator state be recomputed at the end of the step |
| `reset(ctx)`      | reset kernel      | reset persistent command state; `ctx.reset_ids` carries this pass's compact reset rows, `ctx.reset_episode` marks the episode resets, and `ctx.terminated` mirrors the latest termination union for episode-boundary statistics |

## Queries (queries)

**What they do**: declare the simulation data and structural information terms need;
the framework compiles the declarations into batched reads. `queries.data` holds
simulation data queries (`JointPositionQuery`, `LinkPositionQuery`, ...) and
`queries.model` holds model queries (`ActuatorKpQuery`,
`BodyJointPositionLimitsQuery`, ...).

**How to declare them**: usually filled in the config's `__post_init__` from the robot
configuration:

```python
def __post_init__(self) -> None:
    joint_names = ...  # resolved from scene.objs.robot
    base_link = ...    # base link name
    self.queries.data["robot_dof_pos"] = JointPositionQuery(joints=joint_names)
    self.queries.model["robot_joint_position_limits"] = BodyJointPositionLimitsQuery(body=base_link)
```

Observation terms can also declare their needs via `required_sim_queries(env_cfg)`;
all declarations merge by key, and declarations sharing a key must be exactly equal or
the config fails. How queries are compiled and executed is covered in
[SimBackend: Decoupling from the Simulator](sim_backend.md).

## Registration and variants

- **The config is registered before the environment class**; the registry validates
  that the class inherits an `ArrayEnv` frontend base (`ManagerEnv` or `DirectEnv`),
  so never specify a frontend type manually.
- The first docstring line and the `zh_CN:` line serve as the environment's English and
  Chinese descriptions.
- Multiple presets of one environment are derived through config factories: a variant
  overrides only what differs from the shared config (for example lowering one reward
  weight).

The built-in `microduck-ball-balance` environment (source under
`motrix_envs/src/motrix_envs/locomotion/ball_balance/microduck.py`) shows the factory
form:

```python
@registry.envcfg("microduck-ball-balance")
def make_microduck_ball_balance_cfg() -> MicroduckBallBalanceEnvCfg:
    """Balance on top of a basketball with Microduck.

    zh_CN: 让 Microduck 双脚站在篮球上并保持平衡。
    """
    return MicroduckBallBalanceEnvCfg()


registry.env("microduck-ball-balance")(ManagerEnv)
```

Once registered, preview it with `mxlab view env=<name>` and start training
by creating a Task as described in [Task Configuration](../training/task_config.md).
