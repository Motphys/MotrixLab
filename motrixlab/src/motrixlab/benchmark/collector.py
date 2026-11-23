# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC collector-step benchmark.

Measures one collector ``step_once`` round trip: policy inference, environment
step, host transition ring push, and bookkeeping.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

from motrix_bench import BenchmarkConfig, BenchmarkResult, PreparedCase, run_case

from .argtypes import nonnegative_int, positive_int
from .registry import BenchmarkSpec


@dataclass(frozen=True, slots=True)
class CollectorConfig(BenchmarkConfig):
    env_name: str = ""
    num_envs: tuple[int, ...] = ()
    mode: str = "train"
    seed: int = 20260808
    ring_capacity: int = 8
    sim_backend: str | None = None

    def __post_init__(self) -> None:
        BenchmarkConfig.__post_init__(self)
        if not self.env_name:
            raise ValueError("env_name is required")
        if not self.num_envs:
            raise ValueError("num_envs requires at least one value")
        if any(num_envs < 1 for num_envs in self.num_envs):
            raise ValueError("num_envs values must be positive")
        if self.ring_capacity < 2:
            raise ValueError("ring_capacity must be at least 2")


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", required=True, help="Registered environment name.")
    parser.add_argument("--env-mode", default="train", choices=("train", "play"))
    parser.add_argument("--num-envs", type=positive_int, nargs="+", default=[2048])
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--warmup", type=nonnegative_int, default=10)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--ring-capacity", type=positive_int, default=8, help="Host transition ring capacity.")
    parser.add_argument("--sim-backend")


def build_config(args: argparse.Namespace) -> CollectorConfig:
    return CollectorConfig(
        steps=args.steps,
        warmup=args.warmup,
        env_name=args.env,
        num_envs=tuple(args.num_envs),
        mode=args.env_mode,
        seed=args.seed,
        ring_capacity=args.ring_capacity,
        sim_backend=args.sim_backend,
    )


def _inference_config():
    """Build the inference-only config surface required by ``Collector``."""
    return SimpleNamespace(
        agent=SimpleNamespace(
            actor_hidden_dim=256,
            log_std_max=2.0,
            log_std_min=-5.0,
            use_tanh=True,
            use_layer_norm=True,
            obs_normalization=True,
            learning_starts=0,
        ),
        trainer=SimpleNamespace(
            async_options=SimpleNamespace(
                collector_amp=False,
                collector_amp_dtype="fp16",
                collector_compile=False,
                weight_poll_interval=2**31 - 1,
            )
        ),
    )


def prepare(config: CollectorConfig) -> PreparedCase:
    import torch
    from motrix_rl_builtin.fastsac.async_impl.collector import Collector
    from motrix_rl_builtin.fastsac.async_impl.transport import (
        Control,
        HostTransitionRing,
        HostWeightReceiver,
        HostWeightSender,
        WeightChannelShared,
        fastsac_inference_bundle_spec,
        fastsac_transition_schema,
    )
    from motrix_rl_builtin.fastsac.buffer import EmpiricalNormalization
    from motrix_rl_builtin.fastsac.networks import Actor
    from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap

    from motrix_env_core import registry

    env = registry.make(
        config.env_name,
        mode=config.mode,
        num_envs=config.num_envs[0],
        sim=config.sim_backend,
        seed=config.seed,
        headless=True,
    )
    wrapped = FastSacEnvWrap(env, torch.device("cpu"))
    spec = wrapped.observation_spec
    obs_dim = spec.policy.space.shape[-1]
    frame_dim = spec.policy.frame_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    low = wrapped.action_low
    high = wrapped.action_high
    action_scale = (high - low) / 2.0
    action_bias = (high + low) / 2.0
    cfg = _inference_config()

    source_actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=cfg.agent.actor_hidden_dim,
        log_std_max=cfg.agent.log_std_max,
        log_std_min=cfg.agent.log_std_min,
        use_tanh=cfg.agent.use_tanh,
        use_layer_norm=cfg.agent.use_layer_norm,
        action_scale=action_scale,
        action_bias=action_bias,
        device="cpu",
    )
    normalizer = EmpiricalNormalization(obs_dim, device="cpu")
    parameter_count = sum(p.numel() for p in source_actor.parameters())
    shared = WeightChannelShared(fastsac_inference_bundle_spec(obs_dim, parameter_count))
    sender = HostWeightSender(shared)
    receiver = HostWeightReceiver(shared, sender.params)
    sender.publish({"actor": source_actor}, {"obs": normalizer})
    ring = HostTransitionRing(
        fastsac_transition_schema(frame_dim, spec.value_or_policy.frame_space.shape[-1], act_dim),
        config.ring_capacity,
        config.num_envs[0],
    )
    collector = Collector(
        wrapped,
        cfg,
        obs_dim,
        act_dim,
        action_scale,
        action_bias,
        ring,
        receiver,
        Control(),
        torch.device("cpu"),
    )

    def _drain_ring() -> None:
        count, _ = ring.read_span()
        if count:
            ring.commit_reads(count)

    def _step_once() -> bool:
        if not collector.step_once():
            raise RuntimeError("collector ring unexpectedly became full")
        return True

    collector.reset()
    return PreparedCase(
        step=_step_once,
        close=wrapped.close,
        before_step=_drain_ring,
        throughput=float(config.num_envs[0]),
        unit="env steps",
    )


def run(config: CollectorConfig) -> list[BenchmarkResult]:
    results = []
    for num_envs in config.num_envs:
        single = replace(config, num_envs=(num_envs,))
        with prepare(single) as case:
            results.append(
                run_case(
                    case,
                    BenchmarkConfig(single.steps, single.warmup),
                    benchmark="fastsac.collector",
                    label=f"{single.env_name} (num_envs={num_envs})",
                    config=asdict(single),
                    metadata={"ring_capacity": single.ring_capacity, "device": "cpu"},
                )
            )
    return results


def register() -> BenchmarkSpec:
    return BenchmarkSpec(
        "fastsac.collector",
        "Benchmark the FastSAC collector step (policy inference + env.step + host ring push).",
        configure,
        build_config,
        prepare,
        run,
    )
