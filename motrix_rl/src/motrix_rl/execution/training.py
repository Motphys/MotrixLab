# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Algorithm-neutral declarations used by process-backed training.

The objects in this module describe a training topology and observations of it;
they do not own processes, queues, or other operating-system resources.  The
mapping fields on observations are exposed read-only.  ``WorkerSpec.args`` and
``WorkerSpec.kwargs`` intentionally remain ordinary pickle-friendly containers,
because they are passed to a spawned worker.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from motrix_rl.console import TrainingPanelStats
from motrix_rl.execution.workers import WorkerSettings

WorkerTarget = Callable[..., Any]
WorkerState = Literal["starting", "ready", "running", "finished", "exited", "failed"]


def _readonly(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make a shallow, read-only view for parent-process observations."""
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class WorkerSpec:
    """Pickle-friendly declaration for one worker entry point.

    ``target`` is a module-level callable with the eventual invocation shape
    ``target(runtime, *args, **kwargs)``.  A spec never contains a Process or
    supervisor handle.
    """

    id: str
    role: str
    target: WorkerTarget
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    settings: WorkerSettings = field(default_factory=WorkerSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        # Do not use MappingProxyType here: kwargs are spawn arguments and must
        # remain serializable by multiprocessing's spawn pickler.
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class TrainingPlan:
    """Algorithm-owned state and the workers declared for its execution."""

    workers: tuple[WorkerSpec, ...]
    state: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "workers", tuple(self.workers))


@dataclass(frozen=True)
class WorkerStatus:
    """Read-only observation of one worker, including lifecycle diagnostics."""

    state: WorkerState
    ready: bool = False
    finished: bool = False
    exited: bool = False
    exitcode: int | None = None
    error: str | None = None
    progress: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "progress", _readonly(self.progress))


@dataclass(frozen=True)
class TrainingProgress:
    """Read-only snapshot passed to an algorithm's progress decision hook."""

    workers: Mapping[str, WorkerStatus] = field(default_factory=dict)
    counters: Mapping[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workers", _readonly(self.workers))
        object.__setattr__(self, "counters", _readonly(self.counters))


@dataclass(frozen=True)
class WorkerReports:
    """Read-only reports already consumed by the public execution layer.

    Payloads stay opaque to the execution package and retain arrival order.  The
    metadata flags let a final drain be distinguished from an ordinary poll.
    """

    reports: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    now: float | None = None
    force: bool = False
    events: tuple[TrainingEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reports",
            _readonly({worker_id: tuple(payloads) for worker_id, payloads in self.reports.items()}),
        )


@dataclass(frozen=True)
class TrainingDecision:
    """Algorithm's progress decision and the workers allowed to finish."""

    complete: bool
    completed_workers: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TrainingEvent:
    """A named event suitable for ``TrainingLogger.event``."""

    name: str
    step: int | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _readonly(self.data))


@dataclass(frozen=True)
class TrainingArtifact:
    """A public notification that an algorithm produced an artifact."""

    kind: str
    path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _readonly(self.metadata))


@dataclass(frozen=True)
class TrainingMetrics:
    """Algorithm metrics for the shared logger and optional console panel."""

    panel: TrainingPanelStats | None = None
    scalars: Mapping[str, float] = field(default_factory=dict)
    step: int | None = None
    env_steps: int | None = None
    placeholders: frozenset[str] = frozenset()
    events: tuple[TrainingEvent, ...] = ()
    artifacts: tuple[TrainingArtifact, ...] = ()
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalars", _readonly(self.scalars))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
