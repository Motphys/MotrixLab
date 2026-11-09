# Benchmarks

All benchmarks live in the `mxlab` CLI.

## Environment `env.step()` benchmark

```bash
# Whole-machine throughput: 10 seconds, one worker per usable NUMA node
mxlab bench --mode env --env g1-wbt-dance

# Batch sweep with JSONL output
mxlab bench --mode env --env g1-wbt-dance --duration 10 --num-envs 2048 8192 --json

# Compare two environments
mxlab bench --mode env --env cartpole --compare-env cartpole --duration 5

# Single process, preserving the caller's CPU/memory placement
mxlab bench --mode env --duration 5 --numa single

# Single-process step latency and CPU utilization
mxlab bench --mode env --steps 1000 --num-envs 2048
```

`--num-envs` is the total batch, split across usable NUMA nodes. Workers respect
inherited CPU/cpuset limits and bind CPU and memory before loading the environment;
unavailable topology or memory binding is reported as a fallback in the output.

After warmup, workers share one measurement window. Throughput is the total number
of completed environment transitions divided by the window duration; calls that
finish after the deadline are excluded. This measures `env.step()`, not policy
inference or training data transfer. JSONL includes per-worker counts and placement.

`--steps` and `--duration` are mutually exclusive. `--breakdown` requires step-count
mode and cannot be combined with `--json`.

## MotrixSim physics-only benchmark

```bash
# In-process measurement of one robot
mxlab bench --mode physics --robot g1-29dof --num-envs 2048

# One pinned worker per NUMA node via numactl
mxlab bench --mode physics --numa shard
```

Times `model.step_n(data, nstep)` on a robot-on-flat-ground scene with state reset
before every timed call; no task env, no observation/reward/termination. `--numa`
selects `single`, `node` (pinned to node 0), or `shard` (one numactl-pinned worker
per NUMA node; requires numactl on PATH).

See `mxlab bench --help` for all options.

The remaining script in this directory, `bench_live_metrics.py`, is a standalone
simulator-side tool that is not part of the CLI.
