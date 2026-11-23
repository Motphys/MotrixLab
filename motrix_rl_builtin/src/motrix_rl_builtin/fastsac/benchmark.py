# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC collector-step benchmark definition for the task-neutral contract."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from types import SimpleNamespace

from motrix_bench.execution import worker_execution

from motrix_bench import BenchmarkConfig, BenchmarkResult, BenchmarkSpec, PreparedCase, run_case

_BENCH_NAME = "fastsac.collector.step"


@dataclass(frozen=True, slots=True)
class FastSacCollectorConfig(BenchmarkConfig):
    """Typed benchmark config; steps/warmup come from ``BenchmarkConfig``."""

    # ``BenchmarkConfig.warmup`` carries a default, so benchmark-specific
    # required fields must be keyword-only to keep a valid field order.
    env_name: str = field(kw_only=True)
    num_envs: tuple[int, ...] = field(kw_only=True)
    ring_capacity: int = 8
    seed: int = 20260808
    sim_backend: str | None = None

    def __post_init__(self) -> None:
        # Direct call: zero-arg super() resolves to the pre-slots class.
        BenchmarkConfig.__post_init__(self)
        if self.ring_capacity < 2:
            raise ValueError("ring_capacity must be at least 2")
        if not self.num_envs:
            raise ValueError("num_envs requires at least one value")
        if any(num_envs < 1 for num_envs in self.num_envs):
            raise ValueError("num_envs values must be positive")


def _config() -> SimpleNamespace:
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


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", required=True, help="Registered environment name.")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[2048], help="Environment batch sizes.")
    parser.add_argument("--steps", type=int, default=1000, help="Timed collector steps.")
    parser.add_argument("--warmup", type=int, default=100, help="Untimed collector steps.")
    parser.add_argument("--ring-capacity", type=int, default=8, help="Host transition ring capacity.")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--sim-backend", help="Optional simulator backend override.")


def build_config(args: argparse.Namespace) -> FastSacCollectorConfig:
    return FastSacCollectorConfig(
        steps=args.steps,
        warmup=args.warmup,
        env_name=args.env,
        num_envs=tuple(args.num_envs),
        ring_capacity=args.ring_capacity,
        seed=args.seed,
        sim_backend=args.sim_backend,
    )


def prepare(config: FastSacCollectorConfig) -> PreparedCase:
    """Assemble one collector case for ``config.num_envs[0]``."""
    import torch

    from motrix_env_core import registry
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

    num_envs = config.num_envs[0]
    stack = ExitStack()
    env = stack.enter_context(
        worker_execution(
            lambda: registry.make(
                config.env_name,
                mode="train",
                num_envs=num_envs,
                sim=config.sim_backend,
                seed=config.seed,
                headless=True,
            ),
            cleanup=lambda env: getattr(env, "close", lambda: None)(),
        )
    )
    wrapped = stack.enter_context(worker_execution(lambda: FastSacEnvWrap(env, torch.device("cpu"))))
    spec = wrapped.observation_spec
    obs_dim = spec.policy.space.shape[-1]
    frame_dim = spec.policy.frame_space.shape[-1]
    value_frame_dim = spec.value_or_policy.frame_space.shape[-1]
    act_dim = env.action_space.shape[-1]
    action_scale = (wrapped.action_high - wrapped.action_low) / 2.0
    action_bias = (wrapped.action_high + wrapped.action_low) / 2.0
    cfg = _config()

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
    parameter_count = sum(parameter.numel() for parameter in source_actor.parameters())
    shared = WeightChannelShared(fastsac_inference_bundle_spec(obs_dim, parameter_count))
    sender = HostWeightSender(shared)
    receiver = HostWeightReceiver(shared, sender.params)
    sender.publish({"actor": source_actor}, {"obs": normalizer})
    ring = HostTransitionRing(
        fastsac_transition_schema(frame_dim, value_frame_dim, act_dim), config.ring_capacity, num_envs
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
    collector.reset()

    def step_once() -> None:
        if not collector.step_once():
            raise RuntimeError("collector ring unexpectedly became full")
        count, _ = ring.read_span()
        if count:
            ring.commit_reads(count)

    return PreparedCase(step=step_once, close=stack.close, items_per_step=float(num_envs), unit="transitions")


def run(config: FastSacCollectorConfig) -> Sequence[BenchmarkResult]:
    results = []
    for num_envs in config.num_envs:
        batch = replace(config, num_envs=(num_envs,))
        case = prepare(batch)
        results.append(
            run_case(
                case,
                BenchmarkConfig(config.steps, config.warmup),
                benchmark=_BENCH_NAME,
                label=f"{config.env_name} (num_envs={num_envs})",
                config=asdict(batch),
                metadata={
                    "device": "cpu",
                    "ring_capacity": config.ring_capacity,
                    "scope": "policy inference + env.step + host ring push + bookkeeping",
                },
            )
        )
    return results


def register() -> BenchmarkSpec:
    """Return the benchmark definition without importing runtime dependencies."""
    return BenchmarkSpec(_BENCH_NAME, "Benchmark one FastSAC collector step.", configure, build_config, prepare, run)


__all__ = [
    "FastSacCollectorConfig",
    "BenchmarkSpec",
    "build_config",
    "configure",
    "prepare",
    "register",
    "run",
]
