# Environment benchmarks

Activate the project environment before running:

```bash
source .venv/bin/activate

# Whole-machine throughput: 10 seconds, one worker per usable NUMA node
python bench/bench_env.py --env g1-wbt-dance

# Batch sweep with JSONL output
python bench/bench_env.py --env g1-wbt-dance --duration 10 --num-envs 2048 8192 --json

# Compare two environments
python bench/bench_env.py --env cartpole --compare-env cartpole --duration 5

# Single process, preserving the caller's CPU/memory placement
python bench/bench_env.py --duration 5 --numa single

# Single-process step latency and CPU utilization
python bench/bench_env.py --steps 1000 --num-envs 2048
```

`--num-envs` is the total batch, split across usable NUMA nodes. Workers respect
inherited CPU/cpuset limits and bind CPU and memory before loading the environment;
unavailable topology or memory binding is reported as a fallback in the output.

After warmup, workers share one measurement window. Throughput is the total number
of completed environment transitions divided by the window duration; calls that
finish after the deadline are excluded. This measures `env.step()`, not policy
inference or training data transfer. JSONL includes per-worker counts and placement.

`--steps` and `--duration` are mutually exclusive. `--breakdown` requires step-count
mode and cannot be combined with `--json`. See `--help` for all options.
