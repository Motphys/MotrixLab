# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from motrix_rl import backend_runtime, frameworks
from motrix_rl.frameworks import RlFramework
from motrix_rl.method import RlMethod
from motrix_rl.utils import DeviceSupports


def test_frameworks_list_supported_train_backends(monkeypatch):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)

    _register_framework("runtime", _RuntimeProvider("torch", "ppo"), _RuntimeProvider("jax", "ppo"))

    assert frameworks.supported_train_backends("runtime", "ppo") == ("jax", "torch")


def test_framework_lists_supported_train_backends_by_agent_name(monkeypatch):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    framework = _RuntimeFramework(
        "external",
        (
            _RuntimeProvider("torch", "ppo"),
            _RuntimeProvider("jax", "ppo"),
            _RuntimeProvider("torch", "sac"),
        ),
    )
    frameworks.register_framework(framework)

    assert framework.supported_agents() == ("ppo", "sac")
    assert frameworks.supported_agents("external") == ("ppo", "sac")
    assert framework.supported_train_backends("ppo") == ("jax", "torch")
    assert frameworks.supported_train_backends("external", "sac") == ("torch",)
    assert frameworks.supported_train_backends("external", "missing") == ()


def test_resolve_train_backend_prefers_jax_when_available(monkeypatch):
    _register_method(monkeypatch, rllib="runtime", algo="ppo", backends=("jax", "torch"))

    backend = backend_runtime.resolve_train_backend(
        "demo-env",
        RlMethod(rllib="runtime", algo="ppo"),
        None,
        DeviceSupports(jax=True, torch=True),
    )

    assert backend == "jax"


def test_resolve_train_backend_honors_requested_backend(monkeypatch):
    _register_method(monkeypatch, rllib="runtime", algo="ppo", backends=("jax", "torch"))

    backend = backend_runtime.resolve_train_backend(
        "demo-env",
        RlMethod(rllib="runtime", algo="ppo"),
        "torch",
        DeviceSupports(jax=True, torch=True),
    )

    assert backend == "torch"


def test_resolve_train_backend_allows_external_backend(monkeypatch):
    _register_method(monkeypatch, rllib="external", algo="demo", backends=("custom",))

    backend = backend_runtime.resolve_train_backend(
        "external-env",
        RlMethod(rllib="external", algo="demo"),
        None,
        DeviceSupports(),
    )

    assert backend == "custom"


def test_resolve_train_backend_rejects_missing_trainer(monkeypatch):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)

    with pytest.raises(ValueError, match="No trainer found"):
        backend_runtime.resolve_train_backend(
            "demo-env",
            RlMethod(rllib="runtime", algo="ppo"),
            "torch",
            DeviceSupports(torch=True),
        )


def test_resolve_train_backend_rejects_unavailable_builtin_backend(monkeypatch):
    _register_method(
        monkeypatch,
        rllib="runtime",
        algo="ppo",
        backends=("torch",),
    )

    with pytest.raises(ValueError, match="not available"):
        backend_runtime.resolve_train_backend(
            "demo-env",
            RlMethod(rllib="runtime", algo="ppo"),
            "torch",
            DeviceSupports(torch=False),
        )


def _register_method(
    monkeypatch,
    rllib: str,
    algo: str,
    backends: tuple[str, ...],
) -> None:
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    _register_framework(
        rllib,
        *(_RuntimeProvider(backend, algo) for backend in backends),
    )


def _register_framework(rllib: str, *providers: "_RuntimeProvider") -> None:
    frameworks.register_framework(_RuntimeFramework(rllib, providers))


class _RuntimeFramework(RlFramework):
    def __init__(self, name: str, providers) -> None:
        self._name = name
        super().__init__(providers)

    @property
    def name(self) -> str:
        return self._name


@dataclass(frozen=True)
class _RuntimeCfg:
    pass


@dataclass(frozen=True)
class _RuntimeProvider(frameworks.AgentProvider[_RuntimeCfg]):
    config_type = _RuntimeCfg

    _train_backend: str
    _agent_name: str

    @property
    def train_backend(self) -> str:
        return self._train_backend

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(
        self,
        context: frameworks.TrainerContext[_RuntimeCfg],
    ) -> frameworks.TrainerBase:
        raise NotImplementedError
