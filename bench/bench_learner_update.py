# Profile the FastSAC learner update: where does wall time go?
# Builds a real FastSacAgent, fills the replay buffer with random transitions,
# then times agent.update(n) and profiles it with torch.profiler to split
# wall time into CUDA kernel time vs host-side gaps.
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import numpy as np
import torch
from omegaconf import OmegaConf

import motrix_envs  # noqa: F401  (unused; keeps import parity with training)
from motrix_env_core import registry
from motrix_rl.fastsac.async_impl.worker import build_agent
from motrix_rl.fastsac.config import FastSacAgentCfg

num_envs = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
n_updates = int(sys.argv[2]) if len(sys.argv) > 2 else 4
run_cfg = OmegaConf.load(sys.argv[3]) if len(sys.argv) > 3 else None
agent_cfg = FastSacAgentCfg(**OmegaConf.to_container(run_cfg.algo.agent)) if run_cfg else FastSacAgentCfg()

env = registry.make("g1-wbt-dance", num_envs=num_envs)
obs_space = env.observation_space
obs_dim = obs_space.policy.shape[0]
critic_dim = obs_space.value_or_policy.shape[0]
act_dim = env.action_space.shape[0]
del env

cfg = SimpleNamespace(agent=agent_cfg)

device = torch.device("cuda")
agent = build_agent(
    cfg, (obs_dim, critic_dim, act_dim), num_envs, device,
    torch.ones(act_dim), torch.zeros(act_dim),
)
rng = np.random.default_rng(0)
cap = agent.rb.buffer_size
for _ in range(cap):
    agent.rb.extend(
        torch.randn(num_envs, obs_dim, device=device),
        torch.randn(num_envs, critic_dim, device=device),
        torch.rand(num_envs, act_dim, device=device) * 2 - 1,
        torch.randn(num_envs, device=device),
        torch.zeros(num_envs, dtype=torch.long, device=device),
        torch.zeros(num_envs, dtype=torch.long, device=device),
    )

# warmup (compile/cudagraph) then timed windows
for _ in range(3):
    agent.update(n_updates)
torch.cuda.synchronize()

import time

for trial in range(3):
    t0 = time.perf_counter()
    agent.update(n_updates)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    print(f"update({n_updates}) wall: {wall_ms:8.2f} ms  ({wall_ms/n_updates:.2f} ms/update)")

from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    agent.update(n_updates)
    torch.cuda.synchronize()

events = prof.key_averages()
total_cuda_ms = sum(getattr(e, "device_time_total", 0) for e in events) / 1e3
total_cpu_ms = sum(getattr(e, "self_cpu_time_total", 0) for e in events) / 1e3
print(f"\nprofiled update: cuda kernel time ~{total_cuda_ms:.2f} ms, host self time ~{total_cpu_ms:.2f} ms")
print(f"\ntop ops by CUDA time:")
events.sort(key=lambda e: getattr(e, "device_time_total", 0), reverse=True)
for e in events[:12]:
    d = getattr(e, "device_time_total", 0) / 1e3
    c = e.count
    if d > 0.01:
        print(f"  {e.key[:64]:<64} {d:8.2f} ms  x{c}")
print(f"\ntop ops by host (self) time:")
events.sort(key=lambda e: e.self_cpu_time_total, reverse=True)
for e in events[:12]:
    if e.self_cpu_time_total / 1e3 > 0.05:
        print(f"  {e.key[:64]:<64} {e.self_cpu_time_total/1e3:8.2f} ms  x{e.count}")
