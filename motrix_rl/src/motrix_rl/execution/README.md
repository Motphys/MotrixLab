# Training execution

## Summary

`motrix_rl.execution` is an opt-in, algorithm-neutral execution layer beneath the
existing provider/trainer API. The current framework-ready scope covers phase
scheduling, bounded handoffs, process supervision, tensor transport, and versioned
inference snapshots for provider adapters. It does not implement APPO or any other
algorithm. It schedules collect/learn handoffs and supervises local workers; it does
not define rollout length, replay, losses, optimizers, UTD, or acceptable policy lag.
Providers still receive `TrainerContext` from the runner and return through
`TrainerHandle`; there is no second run registry or CLI.

## Contracts and ownership

Import metadata from `motrix_rl.execution.contracts`, synchronous scheduling from
`motrix_rl.execution.synchronous`, the bounded asynchronous pipeline from
`motrix_rl.execution.asynchronous`, and process supervision from
`motrix_rl.execution.processes`.

- `Parameters(value, version)` is an algorithm-owned snapshot. Versions increase
  on publication; nested payloads must be immutable or detached copies, not views
  of a concurrently updated model. Version zero need not mean an untrained model.
- `Sample(value, policy_version, sequence)` transfers data ownership to the learner.
  The collector must not reuse/mutate the transferred buffer. The sequence starts
  at zero per execution; the policy version identifies the snapshot actually used.
- `learn` decides whether to accept stale data, write replay, or perform zero,
  one, or many optimizer steps. It returns a newer `Parameters` or `None` when no
  snapshot should be published. Rejection/warmup still completes a handoff.
- `Progress` counts completed handoffs (`iterations`/`samples`) and publications
  (`updates`), not transitions or optimizer steps. The adapter maps these counters
  to environment steps, resumed global steps, logging axes and stop targets.
- `Cancellation` is cooperative. Callbacks should check it during long operations.
  Cancellation raises `ExecutionCancelled`, which `TrainerHandle` reports as
  `killed`, like Ctrl+C. Other exceptions propagate and mark the run failed.

## Synchronous execution

`SyncExecution.run(collect, learn, initial_parameters=..., max_iterations=...,
on_step=...)` runs collect → learn → publish → observer without overlap. The next
collection always sees the last published version. Callbacks may share mutable
state because they run serially. The caller owns and closes resources with
`try/finally` or context managers, including on construction/resume failures.

FastSAC sync adapts environment/replay collection and its configured optimizer
updates to these phases. Warmup, episode metrics, periodic saves and global-step
mapping remain in its adapter. This is not a framework-imposed replay protocol.

## Asynchronous execution

`AsyncExecution` accepts collector/learner objects with `collect`/`learn` and
`close` methods. Collection runs on one background thread; learning and `on_step`
run on the calling thread. A bounded FIFO queue applies backpressure. Parameter
publication is latest-only: a collector takes one complete snapshot at the start
of each collection, while queued/in-flight samples keep their original version.
There is no automatic stale-data filtering or assumed suitability for off-policy
training. An on-policy algorithm must explicitly implement its validity rules or
choose sync execution.

`max_iterations` bounds both collection and learning. Normal completion processes
all requested handoffs; cancellation/failure discards pending data. Queue waits
poll cancellation and worker completion rather than waiting forever. Worker
exceptions reach the caller. Worker-origin `SystemExit` and `KeyboardInterrupt`
are wrapped in `RuntimeError` with the original cause, so an unexpected worker
exit cannot masquerade as successful completion or caller cancellation. Ordinary
worker exceptions preserve their identity; caller-thread interrupts still cancel.
The collector
closes on its own thread; the learner closes on the caller thread. Shutdown joins
are bounded. Python cannot kill a thread stuck in a native callback: callbacks
must cooperate; a shutdown timeout is an error, not successful cleanup. Use
supervised processes for isolation or callbacks that may block indefinitely.

The threaded pipeline is suitable for CPU/I/O collection or libraries that release
the GIL; it is not a GPU topology manager and does not create distributed workers.

### Existing multi-process execution

`AsyncTrainerBase` owns the parent lifecycle for process-backed trainers. It
registers the algorithm session and resources, starts module-level worker entry
points, polls readiness, releases the shared startup barrier, observes progress,
and performs bounded stop/join/terminate/kill cleanup through `ProcessSupervisor`.
Workers are reaped before queues and transports are closed; cleanup failures do
not mask an original startup or worker exception.

FastSAC's adapter retains topology, NUMA placement, readiness keys, CPU/CUDA
shared rings, weights, DDP, backpressure, replay, UTD and panel observations.
It supplies completion and clean-exit predicates but does not own another parent
monitor loop or process manager. Its specialized transport remains independent
of the reference threaded `AsyncExecution` queue, and the same base can
supervise other algorithm-owned process transports without knowing their data.

A worker may exit successfully only at an adapter-defined completion boundary;
an early zero exit during startup is a failure. During training the adapter defines
collector completion independently of learner completion. Shared stop requests
must be checked by algorithm-specific transport waits; forced process termination
is the final fallback, not a guarantee of resumable algorithm state.

## New algorithm adapter

1. Keep configuration/provider discovery in the existing plugin. Create resources
   using its `TrainerContext` and choose execution mode explicitly.
2. For sync, supply callable collection and learning phases. For async, separate
   actor and learner state, supply immutable parameter snapshots, implement the
   two objects' `close` methods, and decide acceptable lag in `learn`.
3. Put the algorithm data format inside `Sample.value`. Use `Sample.policy_version`
   for validity, not elapsed wall time or the learner's current version.
4. Pass a cancellation token and a bounded handoff count. For a resumed run, derive
   the remaining count from the algorithm checkpoint; counters are not restored
   implicitly. Own environment/model construction cleanup in the trainer.
5. Use `on_step` for algorithm logging and checkpoint policy in the callable
   sync/thread reference executors. For specialized multi-process transports,
   subclass `AsyncTrainerBase` and implement `prepare_async_session(resources)`.
   Return `AsyncTrainingSession` with `WorkerSpec` entries, startup reports,
   progress/error polling, stop, completion and clean-exit predicates. Do not
   override `train()` or add another parent monitor loop.

The process session's `poll(observed, released)` is a nonblocking observer,
called before release as well as during training; it must consume errors even
when no training metrics exist yet. Warmup/first-update display gates must not
hold the start barrier. `close()` runs after workers are reaped and must drain
final payloads and close presentation. Register queues/transports immediately
in the supplied `ExitStack`; they close after the session. Targets and their
arguments must be spawn-pickleable; the parent-only session callbacks need not
be pickleable. Placement decisions and seed offsets remain adapter-owned.
`execution.startup` provides a readiness gate and finite, cancellable opaque
resource delivery; `execution.worker_runtime` provides explicit Torch seed,
thread budget, signal and stdio helpers. CPU facts and affinity belong to
`motrix_rl.numa`. None of these helpers defines replay or algorithm scheduling.

For example, an integer-data algorithm can use the same sync executor without
any tensors, environments or replay:

```python
from motrix_rl.execution.contracts import Parameters
from motrix_rl.execution.synchronous import SyncExecution

progress = SyncExecution().run(
    collect=lambda parameters, cancel: parameters.value + 1,
    learn=lambda sample, cancel: Parameters(sample.value, sample.policy_version + 1),
    initial_parameters=Parameters(0),
    max_iterations=10,
)
assert progress.policy_version == 10
```

The execution tests also implement an independent minimal async algorithm and
exercise FIFO/version handoff, backpressure, cancellation and failed workers.

## Run, logging and checkpoint boundaries

`on_step` runs after the learner and publication, never concurrently with learner
mutation. Serialize learner model/optimizer/counters here using the existing
`checkpoints` artifact services and `TrainingLogger` supplied/created by the
trainer. Exceptions in the observer abort execution just like update failures.
The executor does not create a logger, new event schema or manifest. Log-event
plugins can be integrated at this boundary without algorithm imports in execution.

An async learner checkpoint is **not** a snapshot of the queue, collector RNG,
environment or replay unless the algorithm explicitly persists them. Collection
may continue while a checkpoint is saved, but must use detached parameter state;
exact replay of in-flight trajectories is not promised. FastSAC async checkpoints
remain learner-owned (primary rank for replicated learning) at its existing
update boundary, and resume refills replay. The executor never labels a worker's
forced shutdown or incomplete checkpoint as successful training.

## Tensor transport

Production async adapters can use `motrix_rl.transport` independently of the
reference thread/queue executor. `SharedTensorRing` accepts named tensor layouts;
`CudaTensorRing` supplies fixed-shape CUDA slots with event-proven ownership.
The snapshot channel publishes model parameters and named inference auxiliary
state consistently. FastSAC uses these primitives through its transition and
normalizer adapters, retaining its SPSC topology, sparse final observations,
replay, UTD, DDP and device-selection semantics. No MPMC queue is introduced.

The ownership and algorithm boundaries are described in the
[RL architecture design](../../../../wiki/design/rl-multi-algorithm-architecture.md).

Shared cursors and snapshot channels reject hosts other than x86-64/AMD64: ARM
acquire/release ordering is not implemented. CPU fields require exact shape/dtype
and CPU tensors; implicit broadcasting/conversion is not a transport protocol.
A rejected payload is not published. Consumption confirmations must be nonnegative
integer counts no larger than published, unconsumed data; CUDA confirmations also
exclude already-issued reads. Confirmations do not prove that the caller actually
finished reading: the caller must obey the documented host/stream lifetime rules.

Snapshot auxiliary keys, shapes and dtypes are fixed by the channel schema. Supply
an empty schema for a model without auxiliary state, not an empty update to a
nonempty channel. Parameters and auxiliary tensors are validated before publication;
receivers validate a private snapshot before modifying inference state. The adapter
owns model construction and buffer state: module buffers are not included unless
explicitly declared as auxiliary tensors. The channel is single-writer/single-reader;
create a channel per collector rather than sharing a receiver cursor.

## Learning and collection clocks

The operator-facing `iter` is a completed **learning iteration**, not a single
optimizer step. PPO counts each full `alg.update()` (all epochs/mini-batches).
FastSAC counts each successful positive `agent.update(n)` call; strict async
catch-up may execute several such calls. Warmup, idle polls and `update(0)` do
not advance this clock. `update_idx` remains FastSAC gradient rounds (one critic
step per round, actor gated by policy_frequency), not the total number of all
optimizer steps. Parameter-publication versions remain a separate clock.

FastSAC collection progress and its stop target retain their existing meaning:
`global_step` and `trainer.num_learning_iterations` are full-num_envs collection
batches. The console displays this fraction as collection steps, separately from
learner `iter`; percentage and ETA use the collection target, never learner count
divided by a collection target. Checkpoint names/events and resume global_step
remain collection-based. New checkpoints store learning_iteration separately;
legacy checkpoints start that counter at zero because historical update-call
counts cannot be reconstructed exactly. Cross-algorithm comparisons use the
explicit cumulative env_steps (transitions) and wall time, not equal iter counts.

For both algorithms, logging.interval is measured in completed learner iterations.
Scalar points use that learner count as their step and include cumulative env_steps;
FastSAC also records progress/collection_step and progress/update_idx explicitly.
Sync FastSAC may emit an initial warmup point at learner step zero; warmup-only
runs can also publish a final point at zero with a different env_steps value.
Normal completion flushes the remaining window without duplicating an exact final
threshold. Async snapshots can coalesce thresholds crossed during one catch-up
operation; they do not fabricate intermediate observations. Checkpoint/event step
fields continue to follow their owning algorithm's checkpoint progress contract.
The async weight-publication diagnostic is named
async/weight_publishes_per_collection_step to make its denominator explicit.
Console rendering still occurs at sync observation boundaries (async has its
existing wall-clock throttle); this change does not add a background UI timer.

## Scope and validation

RSLRL PPO also uses SyncExecution: one handoff collects a complete rollout,
computes returns and executes a native PPO update. Its storage stays inside the
adapter, without copying through Sample; the next rollout sees the updated model.
The shared console displays rollout/optimization/system statistics without a
replay occupancy row. RSLRL's backend writers retain native tags; publication
across writers and shared console/live follows logging.interval and
includes the last iteration. Checkpoints preserve native runner fields, with
`iter` recording completed updates; periodic saves occur after updates and final
saves also register the policy artifact. MotrixLab RSLRL resume remains unsupported.
Distributed setup and initial broadcast still use RSLRL; only rank zero renders
and records checkpoints. Multi-GPU execution requires dedicated hardware validation.

SKRL retains its library-owned execution loop; opting into these primitives
is not required for third-party trainers. Generic executors contain no provider
imports or algorithm-name branches. Tests use deterministic minimal algorithms;
process tests use CPU subprocesses separately from default unit tests. Passing
these tests validates execution contracts, not GPU training quality, distributed
numerics or FastSAC throughput.

From an installed workspace development environment (activate `.venv` first):

```bash
# Fast, deterministic public contracts and FastSAC adapter regression
python -m pytest motrix_rl/tests motrix_rl_builtin/tests -m "not slow and not integration and not numba"
# Real CPU spawn and available single-GPU transport paths
python -m pytest motrix_rl/tests/test_execution_processes.py motrix_rl/tests/test_execution_transport_pipeline.py motrix_rl/tests/test_transport_spawn.py motrix_rl/tests/test_transport_ring.py motrix_rl/tests/test_transport_snapshot.py motrix_rl_builtin/tests/test_fastsac_ipc_ring.py -m integration
```

GPU tests require an available CUDA runtime; skips are not evidence of GPU
compatibility. Multi-GPU NCCL training and cross-NUMA throughput remain separate
hardware acceptance work, tracked in the [FastSAC multi-learner plan](../../../../wiki/plan/fastsac-async-multi-learner.md).
