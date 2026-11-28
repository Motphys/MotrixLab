# motrix_rl_appo

Torch asynchronous PPO implementation for MotrixLab, using V-trace value
correction, an IMPACT-style clipped importance-weighted policy objective, and a
target actor. The collector and learner run in separate processes with a bounded
shared-memory rollout queue. This package is an implementation dependency, not a
public framework provider or entry-point package; the public provider name and
configuration registration are owned by `motrix_rl_builtin` as `motrix.appo`.
It does not depend on Ray or RSL-RL.

Install from the workspace with `sh install.sh` (add `--rslrl` for the comparison
baseline). The default builtin dependency set includes APPO; no separate extra is
needed. Use the public algorithm name through the unified CLI:

```sh
mxlab train task=cartpole/motrix.appo
mxlab train task=go2-walk-flat/motrix.appo
```

Transition transport defaults to `algo.transition_transport=auto`: collector inference and
learner on the same CUDA device use a CUDA-IPC ring; other combinations use host
shared memory. Set `algo.transition_transport=host` or `cuda_ipc` for explicit comparisons.
The chunk storage follows the transition path: IPC keeps the rollout on the
inference GPU and publishes it by D2D copies; host transport uses CPU chunks.
CPU simulator observations/rewards still need upload and actions still return to
CPU; this is not an entirely transfer-free simulator.

`algo.weight_transport=auto` independently selects CUDA-IPC parameter slots for
same-GPU learner/collector inference, otherwise host slots. Force `host` or
`cuda_ipc` for comparisons; unavailable explicit IPC raises an error. Actor and
critic parameters use the selected transport; normalizer statistics remain in
host shared memory, with int64 counts and the same bundle version checks.

The collector handles timeout final-observation bootstraps locally; the learner
uses explicit chunk endpoints. Checkpoints support warm resume, not exact
restoration of the collector's environment trajectory. Timeout bootstrap prefers
pre-reset final observations; if absent, it uses the pre-action value approximation
and logs a warning once. Partially missing final-observation rows remain an error.
Playback uses the policy mean. ONNX export and multi-collector training
are not provided; asynchronous training rendering is explicitly unsupported.

The Go2 flat/rough and ANYmal-C flat task recipes default to CUDA Graph with
fused Adam, matching their validated training configuration:

```sh
mxlab train task=go2-walk-flat/motrix.appo
# Override for CPU execution or an eager comparison:
mxlab train task=go2-walk-flat/motrix.appo algo.device=cpu algo.collector_device=cpu algo.algorithm.execution=eager algo.algorithm.fused_optimizer=false
# Alternative compiled loss:
mxlab train task=go2-walk-flat/motrix.appo algo.algorithm.execution=compile
```

The generic provider and CartPole defaults remain `eager` for CPU support.
CUDA Graph requires a CUDA learner, fused Adam, and
fixed, equally sized minibatches. Compile/capture startup cost is separate from
steady-state performance. Floating-point fusion can change adaptive KL decisions
near a threshold; accelerated runs are not promised to be bitwise identical.

Algorithm and data ownership details: [APPO design](../wiki/design/appo.md).
