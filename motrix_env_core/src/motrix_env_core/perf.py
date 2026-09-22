# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, ParamSpec, TypeVar


@dataclass(frozen=True, slots=True)
class PerfNode:
    """Immutable aggregate for one named scope at one position in a profile tree."""

    name: str
    count: int
    total_ns: int
    self_ns: int
    min_ns: int
    max_ns: int
    children: tuple[PerfNode, ...]

    @property
    def mean_ns(self) -> float:
        return self.total_ns / self.count

    def child(self, name: str) -> PerfNode:
        for child in self.children:
            if child.name == name:
                return child
        raise KeyError(name)


@dataclass(slots=True)
class _MutablePerfNode:
    name: str
    count: int = 0
    total_ns: int = 0
    self_ns: int = 0
    min_ns: int = 0
    max_ns: int = 0
    children: dict[str, _MutablePerfNode] = field(default_factory=dict)

    def snapshot(self) -> PerfNode:
        return PerfNode(
            name=self.name,
            count=self.count,
            total_ns=self.total_ns,
            self_ns=self.self_ns,
            min_ns=self.min_ns,
            max_ns=self.max_ns,
            children=tuple(child.snapshot() for child in self.children.values()),
        )


@dataclass(slots=True)
class _PerfFrame:
    node: _MutablePerfNode
    started_ns: int
    child_ns: int = 0


class _DisabledScope:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None


_DISABLED_SCOPE = _DisabledScope()
_ACTIVE_PERF: ContextVar[Perf | None] = ContextVar("motrix_env_core_active_perf", default=None)
_Clock = Callable[[], int]
_P = ParamSpec("_P")
_R = TypeVar("_R")


class Perf:
    """Opt-in hierarchical wall-clock profiler owned by one environment."""

    __slots__ = ("_clock", "_depth", "_enabled", "_frames", "_roots", "_scope_cache")

    def __init__(self, *, enabled: bool = False, clock: _Clock = perf_counter_ns) -> None:
        self._enabled = enabled
        self._clock = clock
        self._roots: dict[str, _MutablePerfNode] = {}
        self._frames: list[_PerfFrame] = []
        self._depth = 0
        self._scope_cache: dict[str, _PerfScope] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._set_enabled(True)

    def disable(self) -> None:
        self._set_enabled(False)

    def _set_enabled(self, enabled: bool) -> None:
        if self._depth:
            raise RuntimeError("Cannot change Perf.enabled while a scope is active.")
        self._enabled = enabled

    def reset(self) -> None:
        """Clear all aggregates while retaining the enabled state."""
        if self._depth:
            raise RuntimeError("Cannot reset Perf while a scope is active.")
        self._roots.clear()

    def begin(self, name: str) -> None:
        """Begin a named child of the currently active scope."""
        if not self._enabled:
            return
        if not isinstance(name, str) or not name:
            raise ValueError("Perf scope name must be a non-empty string.")
        depth = self._depth
        nodes = self._roots if depth == 0 else self._frames[depth - 1].node.children
        node = nodes.get(name)
        if node is None:
            node = _MutablePerfNode(name)
            nodes[name] = node
        started_ns = self._clock()
        if depth == len(self._frames):
            self._frames.append(_PerfFrame(node=node, started_ns=started_ns))
        else:
            frame = self._frames[depth]
            frame.node = node
            frame.started_ns = started_ns
            frame.child_ns = 0
        self._depth = depth + 1

    def end(self, name: str | None = None) -> int:
        """End the innermost scope and return its inclusive elapsed nanoseconds."""
        if not self._enabled:
            return 0
        depth = self._depth
        if depth == 0:
            raise RuntimeError("Perf.end() called without an active scope.")
        frame = self._frames[depth - 1]
        if name is not None and frame.node.name != name:
            raise RuntimeError(f"Perf scope mismatch: expected {frame.node.name!r}, got {name!r}.")
        elapsed_ns = self._clock() - frame.started_ns
        if elapsed_ns < 0:
            raise RuntimeError("Perf clock moved backwards.")
        self._depth = depth - 1

        node = frame.node
        node.count += 1
        node.total_ns += elapsed_ns
        node.self_ns += elapsed_ns - frame.child_ns
        node.min_ns = elapsed_ns if node.count == 1 else min(node.min_ns, elapsed_ns)
        node.max_ns = max(node.max_ns, elapsed_ns)
        if depth > 1:
            self._frames[depth - 2].child_ns += elapsed_ns
        return elapsed_ns

    def scope(self, name: str) -> _DisabledScope | _PerfScope:
        """Return an exception-safe context manager for one scope."""
        if not self._enabled:
            return _DISABLED_SCOPE
        scope = self._scope_cache.get(name)
        if scope is None:
            scope = _PerfScope(self, name)
            self._scope_cache[name] = scope
        return scope

    @contextmanager
    def activate(self) -> Iterator[Perf]:
        """Make this profiler available to :func:`perf_scope` in the current context."""
        token = _ACTIVE_PERF.set(self)
        try:
            yield self
        finally:
            _ACTIVE_PERF.reset(token)

    def snapshot(self) -> tuple[PerfNode, ...]:
        """Return an immutable snapshot of all completed root scopes."""
        if self._depth:
            raise RuntimeError("Cannot snapshot Perf while a scope is active.")
        return tuple(root.snapshot() for root in self._roots.values())

    def stage_mean_ms(self, root: str) -> dict[str, float]:
        """Per-call mean milliseconds of every sub-scope under ``root``.

        Keys are dotted paths relative to the root's children (``stage`` or
        ``stage.sub``), so nested scope names — which may themselves contain
        underscores — stay unambiguously separable. Returns ``{}`` when the
        root has not run.
        """
        for node in self._roots.values():
            if node.name == root:
                means: dict[str, float] = {}

                def emit(subtree: _MutablePerfNode, path: tuple[str, ...]) -> None:
                    for child in subtree.children.values():
                        means[".".join((*path, child.name))] = child.total_ns / 1e6 / max(child.count, 1)
                        emit(child, (*path, child.name))

                emit(node, ())
                return means
        return {}

    def call(self, name: str, function: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Call a function inside a named scope."""
        if not self._enabled:
            return function(*args, **kwargs)
        with self.scope(name):
            return function(*args, **kwargs)


class _PerfScope:
    __slots__ = ("_name", "_perf")

    def __init__(self, perf: Perf, name: str) -> None:
        self._perf = perf
        self._name = name

    def __enter__(self) -> None:
        self._perf.begin(self._name)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._perf.end(self._name)


def perf_scope(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Profile a function when called inside an active :class:`Perf` context."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            perf = _ACTIVE_PERF.get()
            if perf is None or not perf.enabled:
                return function(*args, **kwargs)
            with perf.scope(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def active_perf_scope(name: str) -> _DisabledScope | _PerfScope:
    """Return a scope on the profiler active in the current call context.

    Backend helpers that do not own an environment can use this to contribute
    children to an enclosing :func:`perf_root` without threading a ``Perf``
    instance through their runtime APIs.
    """
    perf = _ACTIVE_PERF.get()
    if perf is None:
        return _DISABLED_SCOPE
    return perf.scope(name)


def perf_root(name: str, *, attribute: str = "perf") -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Activate the profiler stored on the first argument and record a root scope."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if not args:
                raise TypeError("perf_root requires an instance method.")
            perf = getattr(args[0], attribute)
            if not isinstance(perf, Perf):
                raise TypeError(f"{attribute!r} must contain Perf, got {type(perf).__name__}.")
            if not perf.enabled:
                return function(*args, **kwargs)
            active = _ACTIVE_PERF.get()
            if active is perf:
                with perf.scope(name):
                    return function(*args, **kwargs)
            token = _ACTIVE_PERF.set(perf)
            try:
                with perf.scope(name):
                    return function(*args, **kwargs)
            finally:
                _ACTIVE_PERF.reset(token)

        return wrapped

    return decorate


__all__ = ["Perf", "PerfNode", "active_perf_scope", "perf_root", "perf_scope"]
