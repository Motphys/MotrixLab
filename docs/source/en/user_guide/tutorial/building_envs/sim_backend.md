# SimBackend: Decoupling from the Simulator

All simulation interaction in `DirectEnv` and `ManagerEnv` goes through the backend-neutral
`SimBackend` interface (held as `self.sim`). No concrete simulator type crosses the core boundary —
the same environment code runs on any backend implementing that interface.

## Compiling three kinds of programs

Environments compile three kinds of programs against `self.sim` at construction; each step then only
writes small buffers and calls `execute()`:

- `compile_model(queries)`: model queries — structural information that does not change during
  simulation (actuator lists, joint limits, ...);
- `compile_reads(queries)`: simulation data reads (joint positions, velocities, link poses, ...);
  each `execute(env_ids)` refreshes the batch, and results are indexed by the keys used at declaration;
- `compile_writes({name: Write})`: write programs (carried by the `write_compiler`). Control targets use `CtrlTargetsWrite`;
  reset writes (initial poses, ...) pass `reset=True`. Fill `buffer(name)` first, then call
  `execute(env_ids)`.

Division of labor between the workflows: a `DirectEnv` compiles these programs in its own constructor
(see the minimal example in [Writing DirectEnv Environments](direct_env.md)); a `ManagerEnv` generates
them from the config's `queries` group and its terms, so users normally never touch them directly
(see [Writing ManagerEnv Environments](manager_env.md)).

## Choosing and registering a backend

- The backend is a construction-time string: creating an environment with `backend=None` selects the
  registered default backend (currently `motrixsim`); pass a backend name explicitly, or use
  `registry.make(..., sim="motrixsim")`.
- Backends are lazily discovered through the `motrix_env.sim_backends` entry-point group: a third-party
  simulator plugs in by registering a `SimBackend` factory in that group, at zero import cost.
- `motrix_env_mujoco` only compiles MuJoCo scenes and is not a training backend.
