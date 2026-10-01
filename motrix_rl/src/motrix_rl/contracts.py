# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public provider and trainer contracts for MotrixLab RL integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

CfgT = TypeVar("CfgT")


class TrainerBase(ABC):
    """Runnable trainer contract implemented by RL integrations."""

    @abstractmethod
    def train(self) -> None:
        """Run training for the configured run."""

    @abstractmethod
    def play(self, policy: str) -> None:
        """Run policy playback from a checkpoint path."""


@dataclass(frozen=True)
class TrainerContext(Generic[CfgT]):
    """Runtime context passed to one provider's trainer."""

    run: Any
    env_name: str
    run_dir: Path
    checkpoint_dir: Path
    sim: str | None
    checkpoint_format: str
    num_envs: int
    play_num_envs: int
    seed: int | None
    rl_cfg: CfgT
    logging: Any
    checkpoint: Any
    render: Any = None
    resume_from: str | None = None


@dataclass(frozen=True)
class TrainerHandle:
    """Framework-managed handle around a trainer implementation."""

    run: Any
    trainer: TrainerBase
    result_factory: Any = None

    def train(self):
        """Run training and return the control plane's result view."""
        self.trainer.train()
        if self.result_factory is None:
            return self.run
        return self.result_factory(self.run)

    def play(self, policy: str) -> None:
        """Run policy playback through the wrapped trainer."""
        self.trainer.play(policy)


class AgentProvider(Generic[CfgT], ABC):
    """Provider for one algorithm and one train backend."""

    config_type: type[CfgT]

    @property
    @abstractmethod
    def train_backend(self) -> str:
        """Train backend handled by this provider."""

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Algorithm or agent name handled by this provider."""

    @property
    @abstractmethod
    def checkpoint_format(self) -> str | None:
        """Preferred checkpoint format produced by this provider."""

    def validate_config(self, config: object) -> CfgT:
        """Narrow a dynamically composed config to this provider's type."""
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"Provider for '{self.agent_name}' expects {self.config_type.__name__}, got {type(config).__name__}"
            )
        return config

    @abstractmethod
    def create_trainer(self, context: TrainerContext[CfgT]) -> TrainerBase:
        """Create a trainer for the supplied run context."""

    def create_policy_exporter(self) -> Any:
        """Create an optional policy exporter owned by the control plane."""
        return None


@dataclass(frozen=True)
class AgentRegistration:
    """One algorithm's shared config type and backend-specific providers."""

    config_type: type[Any]
    providers: dict[str, AgentProvider[Any]]


class RlFramework(ABC):
    """Registration entry point for one RL framework integration."""

    def __init__(self, providers: Iterable[AgentProvider[Any]]) -> None:
        self._agents = self._index_agents(providers)

    @property
    @abstractmethod
    def name(self) -> str:
        """Framework namespace used as the run metadata namespace."""

    @staticmethod
    def _index_agents(providers: Iterable[AgentProvider[Any]]) -> dict[str, AgentRegistration]:
        providers_by_agent: dict[str, list[AgentProvider[Any]]] = {}
        for provider in providers:
            config_type = getattr(provider, "config_type", None)
            if not isinstance(config_type, type):
                raise TypeError(f"{type(provider).__name__} must declare a config_type class")
            if not is_dataclass(config_type):
                raise TypeError(f"{type(provider).__name__}.config_type must be a dataclass")
            if not provider.agent_name:
                raise ValueError(f"{type(provider).__name__} must declare a non-empty agent_name")
            if not provider.train_backend:
                raise ValueError(f"{type(provider).__name__} must declare a non-empty train backend")
            providers_by_agent.setdefault(provider.agent_name, []).append(provider)

        index: dict[str, AgentRegistration] = {}
        for agent_name, agent_providers in providers_by_agent.items():
            config_types = {provider.config_type for provider in agent_providers}
            if len(config_types) != 1:
                details = ", ".join(
                    f"{provider.train_backend}={provider.config_type.__name__}"
                    for provider in sorted(agent_providers, key=lambda item: item.train_backend)
                )
                raise ValueError(f"Providers for agent '{agent_name}' declare inconsistent config types: {details}")

            providers_by_backend: dict[str, AgentProvider[Any]] = {}
            for provider in agent_providers:
                if provider.train_backend in providers_by_backend:
                    raise ValueError(
                        f"Duplicate agent provider for agent '{agent_name}' and train backend "
                        f"'{provider.train_backend}'."
                    )
                providers_by_backend[provider.train_backend] = provider
            index[agent_name] = AgentRegistration(
                config_type=next(iter(config_types)),
                providers=providers_by_backend,
            )
        return index

    def supported_agents(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents))

    def get_config_type(self, agent_name: str) -> type[Any]:
        try:
            return self._agents[agent_name].config_type
        except KeyError as exc:
            raise ValueError(f"Agent '{agent_name}' is not registered for RL framework '{self.name}'.") from exc

    def supported_train_backends(self, agent_name: str) -> tuple[str, ...]:
        registration = self._agents.get(agent_name)
        return tuple(sorted(registration.providers)) if registration is not None else ()

    def get_agent_provider(self, agent_name: str, train_backend: str) -> AgentProvider[Any] | None:
        registration = self._agents.get(agent_name)
        return registration.providers.get(train_backend) if registration is not None else None

    def supported_policy_exports(self) -> tuple[tuple[str, str], ...]:
        """Return provider/backend combinations with policy export support."""
        combinations = []
        for agent_name in self.supported_agents():
            for train_backend in self.supported_train_backends(agent_name):
                provider = self.get_agent_provider(agent_name, train_backend)
                if provider is not None and provider.create_policy_exporter() is not None:
                    combinations.append((train_backend, agent_name))
        return tuple(sorted(combinations))

    def export_policy(self, request: Any) -> Any:
        """Export a policy through the selected provider."""
        metadata = request.run.metadata
        if metadata.rllib != self.name:
            raise ValueError(f"Export request is for RL framework '{metadata.rllib}', but '{self.name}' was selected.")
        provider = self.get_agent_provider(metadata.algo, metadata.train_backend)
        if provider is None:
            supported = ", ".join(
                f"{self.name}/{backend}/{agent}" for backend, agent in self.supported_policy_exports()
            )
            raise ValueError(
                f"No agent provider found for ONNX export {self.name}/{metadata.train_backend}/{metadata.algo}; "
                f"supported combinations: {supported or 'none'}."
            )
        provider.validate_config(request.task_config.algo)
        exporter = provider.create_policy_exporter()
        if exporter is None:
            supported = ", ".join(
                f"{self.name}/{backend}/{agent}" for backend, agent in self.supported_policy_exports()
            )
            raise ValueError(
                f"ONNX policy export is not supported for {self.name}/{metadata.train_backend}/{metadata.algo}; "
                f"supported combinations: {supported or 'none'}."
            )
        return exporter.export(request)
