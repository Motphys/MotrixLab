# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixLab framework registry and control-plane integration.

Public provider/trainer contracts are defined in :mod:`motrix_rl.contracts`.
This module handles registration, Hydra schema installation, and policy-export dispatch.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from hydra.core.config_store import ConfigStore

from motrix_rl import runs
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import (
    AgentProvider,
    AgentRegistration,
    RlFramework,
    TrainerBase,
    TrainerContext,
    TrainerHandle,
)

__all__ = [
    "AgentProvider",
    "AgentRegistration",
    "RlFramework",
    "TrainerBase",
    "TrainerContext",
    "TrainerHandle",
    "create_trainer",
    "create_trainer_context",
    "get_agent_provider",
    "get_config_type",
    "get_framework",
    "install_algo_schema",
    "iter_agent_providers",
    "iter_frameworks",
    "register_framework",
    "supported_agents",
    "supported_train_backends",
]

ALGO_GROUP = "algo_base"
_frameworks: dict[str, RlFramework] = {}


def install_algo_schema(rllib: str, algo: str, config_type: type[Any]) -> None:
    """Install one framework-owned algorithm config type in Hydra."""
    ConfigStore.instance().store(group=ALGO_GROUP, name=f"_{rllib}_{algo}_schema", node=config_type)


def register_framework(framework: RlFramework) -> RlFramework:
    """Register a framework and install its provider config schemas."""
    if not isinstance(framework.name, str) or not framework.name:
        raise ValueError(f"{type(framework).__name__} must declare a non-empty name")
    existing = _frameworks.get(framework.name)
    if existing is framework:
        return framework
    if existing is not None:
        raise ValueError(f"RL framework '{framework.name}' is already registered.")
    for agent_name in framework.supported_agents():
        install_algo_schema(framework.name, agent_name, framework.get_config_type(agent_name))
    _frameworks[framework.name] = framework
    return framework


def get_framework(name: str) -> RlFramework:
    from motrix_rl.plugins import load_plugins

    load_plugins()
    try:
        return _frameworks[name]
    except KeyError as exc:
        raise ValueError(f"RL framework '{name}' is not registered.") from exc


def iter_frameworks() -> tuple[RlFramework, ...]:
    from motrix_rl.plugins import load_plugins

    load_plugins()
    return tuple(_frameworks[name] for name in sorted(_frameworks))


def agent_exists(framework_name: str, agent_name: str, train_backend: str) -> bool:
    return get_agent_provider(framework_name, agent_name, train_backend) is not None


def supported_train_backends(framework_name: str, agent_name: str) -> tuple[str, ...]:
    return get_framework(framework_name).supported_train_backends(agent_name)


def get_config_type(framework_name: str, agent_name: str) -> type[Any]:
    return get_framework(framework_name).get_config_type(agent_name)


def get_agent_provider(framework_name: str, agent_name: str, train_backend: str) -> AgentProvider[Any] | None:
    try:
        return get_framework(framework_name).get_agent_provider(agent_name, train_backend)
    except ValueError:
        return None


def supported_agents(framework_name: str) -> tuple[str, ...]:
    return get_framework(framework_name).supported_agents()


def iter_agent_providers(
    framework_name: str | None = None,
    train_backend: str | None = None,
    agent_name: str | None = None,
) -> tuple[AgentProvider[Any], ...]:
    framework_names = (
        (framework_name,) if framework_name is not None else tuple(framework.name for framework in iter_frameworks())
    )
    items: list[tuple[str, str, str, AgentProvider[Any]]] = []
    for name in framework_names:
        framework = get_framework(name)
        for candidate_agent in framework.supported_agents():
            if agent_name is not None and candidate_agent != agent_name:
                continue
            for backend in framework.supported_train_backends(candidate_agent):
                if train_backend is not None and backend != train_backend:
                    continue
                provider = framework.get_agent_provider(candidate_agent, backend)
                if provider is not None:
                    items.append((name, backend, candidate_agent, provider))
    return tuple(provider for _, _, _, provider in sorted(items, key=lambda item: item[:3]))


def exists(name: str) -> bool:
    from motrix_rl.plugins import load_plugins

    load_plugins()
    return name in _frameworks


def create_trainer(context: TrainerContext[Any]) -> TrainerHandle:
    metadata = context.run.metadata
    provider = get_agent_provider(metadata.rllib, metadata.algo, metadata.train_backend)
    if provider is None:
        raise ValueError(
            f"No agent provider found for RL framework '{metadata.rllib}', train backend "
            f"'{metadata.train_backend}', agent '{metadata.algo}'."
        )
    typed_cfg = provider.validate_config(context.rl_cfg)
    typed_context = replace(context, rl_cfg=typed_cfg)
    typed_context = _validated_trainer_context(provider, typed_context)
    from motrix_rl.result import TrainResult

    return TrainerHandle(
        run=context.run,
        trainer=provider.create_trainer(typed_context),
        result_factory=TrainResult,
    )


def create_trainer_context(
    run: runs.RunContext,
    *,
    num_envs: int,
    play_num_envs: int,
    seed: int | None,
    rl_cfg: Any,
    logging: LoggingConfig,
    checkpoint: CheckpointConfig,
    render: Any = None,
    resume_from: str | None = None,
) -> TrainerContext[Any]:
    _validate_runtime_config(logging, checkpoint)
    metadata = run.metadata
    return TrainerContext(
        run=run,
        env_name=metadata.env_name,
        run_dir=run.run_dir,
        checkpoint_dir=run.checkpoint_dir,
        sim=run.sim,
        checkpoint_format=metadata.checkpoint_format,
        num_envs=num_envs,
        play_num_envs=play_num_envs,
        seed=seed,
        rl_cfg=rl_cfg,
        logging=logging,
        checkpoint=checkpoint,
        render=render,
        resume_from=resume_from,
    )


def _validate_runtime_config(logging: LoggingConfig, checkpoint: CheckpointConfig) -> None:
    if not logging.backend:
        raise ValueError("logging.backend must be non-empty")
    if logging.interval <= 0:
        raise ValueError("logging.interval must be positive")
    if checkpoint.interval < 0:
        raise ValueError("checkpoint.interval must be non-negative")


def _validated_trainer_context(provider: AgentProvider[Any], context: TrainerContext[Any]) -> TrainerContext[Any]:
    metadata = context.run.metadata
    if provider.train_backend != metadata.train_backend or provider.agent_name != metadata.algo:
        raise ValueError(
            f"AgentProvider ({provider.train_backend}, {provider.agent_name}) does not match run metadata "
            f"({metadata.train_backend}, {metadata.algo})."
        )
    checkpoint_format = context.checkpoint_format or provider.checkpoint_format or ""
    if checkpoint_format != context.checkpoint_format:
        return replace(context, checkpoint_format=checkpoint_format)
    return context
