# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the FastSAC collector actor-only inference boundary.

The measured interval is exactly ``Collector._infer``: CPU policy observation
staging, optional H2D, read-only normalization, stochastic actor inference,
and optional D2H plus synchronization. Environment stepping, transition-ring
push, weight synchronization and learner work are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from motrix_rl.fastsac.async_impl.collector import Collector
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing, WeightSnapshot
from motrix_rl.fastsac.buffer import EmpiricalNormalization
from motrix_rl.fastsac.networks import Actor


class _BenchmarkEnv:
    def __init__(self, num_envs: int, obs_dim: int, critic_obs_dim: int):
        self.num_envs = num_envs
        self._obs = torch.randn(num_envs, obs_dim)
        self._critic_obs = torch.zeros(num_envs, critic_obs_dim)
        self.last_info = {}

    def reset(self):
        return self._obs, self._critic_obs


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[index]


def _source_policy(args):
    action_scale = torch.ones(args.act_dim)
    action_bias = torch.zeros(args.act_dim)
    actor = Actor(
        n_obs=args.obs_dim,
        n_act=args.act_dim,
        hidden_dim=args.hidden_dim,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        action_scale=action_scale,
        action_bias=action_bias,
        device="cpu",
    )
    with torch.no_grad():
        actor.fc_mu.weight.normal_(0.0, 0.02)
        actor.fc_mu.bias.normal_(0.0, 0.02)
        actor.fc_logstd.weight.normal_(0.0, 0.02)
        actor.fc_logstd.bias.normal_(0.0, 0.02)
    normalizer = EmpiricalNormalization(args.obs_dim, device="cpu")
    normalizer._mean.normal_(0.0, 0.1)
    normalizer._std.uniform_(0.8, 1.2)
    normalizer._var.copy_(normalizer._std.square())
    normalizer.count.fill_(1_000_000)
    return actor, normalizer, action_scale, action_bias


def _collector(args, source_actor, source_normalizer, action_scale, action_bias):
    cfg = SimpleNamespace(
        policy_variant="default",
        variant={},
        agent=SimpleNamespace(
            actor_hidden_dim=args.hidden_dim,
            log_std_max=0.0,
            log_std_min=-5.0,
            use_tanh=True,
            use_layer_norm=True,
            obs_normalization=True,
            learning_starts=0,
        ),
        trainer=SimpleNamespace(
            async_options=SimpleNamespace(
                collector_inference_device=args.device,
                collector_compile=args.compile,
                collector_amp=args.amp,
                collector_amp_dtype=args.amp_dtype,
                weight_poll_interval=1,
            )
        ),
    )
    env = _BenchmarkEnv(args.num_envs, args.obs_dim, args.critic_obs_dim)
    # Stub ring (never pushed: the bench only exercises inference/sync); capacity
    # 2 satisfies the ring's successor-slot contract.
    ring = SharedTransitionRing(2, args.num_envs, args.obs_dim, args.critic_obs_dim, args.act_dim)
    weights = WeightSnapshot(sum(p.numel() for p in source_actor.parameters()), args.obs_dim)
    weights.publish(source_actor, source_normalizer)
    collector = Collector(
        env,
        cfg,
        args.obs_dim,
        args.critic_obs_dim,
        args.act_dim,
        action_scale,
        action_bias,
        ring,
        weights,
        Control(),
    )
    collector.reset()
    collector.sync_weights()
    collector.warmup_inference()
    return collector, weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--obs-dim", type=int, default=154)
    parser.add_argument("--critic-obs-dim", type=int, default=154)
    parser.add_argument("--act-dim", type=int, default=29)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--sync-samples", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_num_threads(args.cpu_threads)
    source_actor, source_normalizer, action_scale, action_bias = _source_policy(args)
    collector, weights = _collector(args, source_actor, source_normalizer, action_scale, action_bias)
    obs = torch.randn(args.num_envs, args.obs_dim)

    for _ in range(args.warmup):
        collector._infer(obs)

    samples_ms = []
    first = None
    last = None
    for index in range(args.samples):
        start = time.perf_counter()
        actions = collector._infer(obs)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        if index == 0:
            first = actions.clone()
        if index == args.samples - 1:
            last = actions.clone()

    with torch.no_grad():
        normalized = source_normalizer(obs, update=False)
        cpu_deterministic = source_actor.explore(normalized, deterministic=True)
        with collector._autocast():
            device_deterministic = collector._policy.deterministic(obs.to(collector.device)).cpu()
    abs_error = (device_deterministic - cpu_deterministic).abs()
    sync_samples_ms = []
    for _ in range(args.sync_samples):
        weights.publish(source_actor, source_normalizer)
        start = time.perf_counter()
        collector.sync_weights()
        sync_samples_ms.append((time.perf_counter() - start) * 1000.0)
    staging_ptrs = [
        buffer.data_ptr()
        for buffer in (collector._obs_host, collector._obs_device, collector._actions_host)
        if buffer is not None
    ]
    result = {
        "torch_version": torch.__version__,
        "device": str(collector.device),
        "device_name": torch.cuda.get_device_name(collector.device) if collector.device.type == "cuda" else None,
        "compile": args.compile,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype if args.amp else None,
        "num_envs": args.num_envs,
        "obs_dim": args.obs_dim,
        "critic_obs_dim": args.critic_obs_dim,
        "act_dim": args.act_dim,
        "hidden_dim": args.hidden_dim,
        "cpu_threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "samples": args.samples,
        "latency_ms": {
            "median": statistics.median(samples_ms),
            "p90": _percentile(samples_ms, 0.90),
            "mean": statistics.fmean(samples_ms),
            "min": min(samples_ms),
            "max": max(samples_ms),
        },
        "weight_sync_ms": {
            "median": statistics.median(sync_samples_ms),
            "p90": _percentile(sync_samples_ms, 0.90),
            "mean": statistics.fmean(sync_samples_ms),
            "min": min(sync_samples_ms),
            "max": max(sync_samples_ms),
        },
        "deterministic_vs_cpu": {
            "max_abs_error": float(abs_error.max()),
            "mean_abs_error": float(abs_error.mean()),
        },
        "stochastic_samples_differ": not torch.equal(first, last),
        "actions_finite": bool(torch.isfinite(last).all()),
        "actions_in_bounds": bool(torch.all(last >= -1.0) and torch.all(last <= 1.0)),
        "staging_ptrs": staging_ptrs,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
