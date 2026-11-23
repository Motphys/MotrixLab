# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Named benchmark registration and lazy plugin discovery.

``BenchmarkRegistry`` owns registration state and can be used to isolate
applications or tests.  Module-level functions below intentionally retain the
original facade used by the ``mxlab`` CLI.
"""

from __future__ import annotations

import argparse
import re
from importlib.metadata import entry_points
from typing import Any

from motrix_bench import BenchmarkSpec

_BENCHMARK_GROUP = "motrixlab.benchmarks"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _lazy_builtin(name: str, description: str, module: str) -> BenchmarkSpec:
    """Build metadata-only spec; benchmark module is imported on use."""

    def _load() -> Any:
        from importlib import import_module

        return import_module(module)

    def configure(parser: argparse.ArgumentParser) -> None:
        _load().configure(parser)

    def build_config(args: argparse.Namespace) -> object:
        return _load().build_config(args)

    def prepare(config: object):
        return _load().prepare(config)

    def run(config: object):
        loaded = _load().run
        return None if loaded is None else loaded(config)

    return BenchmarkSpec(name, description, configure, build_config, prepare, run)


class BenchmarkRegistry:
    """Registry of named benchmark definitions.

    Each instance has independent registrations and entry-point load state.
    Entry points are discovered lazily by lookup/list operations, while built-in
    specs remain metadata-only until one of their hooks is invoked.
    """

    def __init__(self) -> None:
        self._specs: dict[str, BenchmarkSpec] = {}
        self._loaded: set[str] = set()

    @staticmethod
    def builtin_benchmarks() -> tuple[BenchmarkSpec, ...]:
        """Return metadata-only built-in specs without importing implementations."""
        return (
            _lazy_builtin("env.step", "Benchmark the public environment step contract.", "motrixlab.benchmark.env"),
            _lazy_builtin(
                "physics.step", "Benchmark native MotrixSim physics throughput.", "motrixlab.benchmark.physics"
            ),
        )

    def register_benchmark(self, spec: BenchmarkSpec) -> None:
        """Register one benchmark, rejecting malformed or duplicate names."""
        if not _NAME_RE.fullmatch(spec.name):
            raise ValueError(f"invalid benchmark name {spec.name!r}")
        previous = self._specs.get(spec.name)
        if previous is not None and previous != spec:
            raise ValueError(f"benchmark {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    @staticmethod
    def _coerce_spec(value: object, *, entry_name: str) -> BenchmarkSpec:
        if isinstance(value, BenchmarkSpec):
            return value
        try:
            return BenchmarkSpec(
                name=str(getattr(value, "name")),
                description=str(getattr(value, "description")),
                configure=getattr(value, "configure"),
                build_config=getattr(value, "build_config"),
                prepare=getattr(value, "prepare"),
                run=getattr(value, "run", None),
            )
        except (AttributeError, TypeError) as error:
            raise TypeError(
                f"benchmark entry point {entry_name!r} must return a BenchmarkSpec-compatible object"
            ) from error

    def discover_benchmarks(self) -> dict[str, BenchmarkSpec]:
        """Discover installed benchmark plugins without importing implementations."""
        seen: set[str] = set()
        for entry in sorted(entry_points(group=_BENCHMARK_GROUP), key=lambda item: item.name):
            if entry.name in seen:
                raise ValueError(f"benchmark {entry.name!r} is already registered by duplicate entry points")
            seen.add(entry.name)
            if entry.name in self._loaded:
                continue
            loaded = entry.load()
            result = loaded() if callable(loaded) else loaded
            spec = self._coerce_spec(result, entry_name=entry.name)
            if spec.name != entry.name:
                spec = BenchmarkSpec(
                    entry.name, spec.description, spec.configure, spec.build_config, spec.prepare, spec.run
                )
            self.register_benchmark(spec)
            self._loaded.add(entry.name)
        return dict(self._specs)

    def get_benchmark(self, name: str) -> BenchmarkSpec | None:
        self.discover_benchmarks()
        return self._specs.get(name)

    def list_benchmarks(self) -> tuple[BenchmarkSpec, ...]:
        self.discover_benchmarks()
        return tuple(self._specs[name] for name in sorted(self._specs))

    @staticmethod
    def configure_benchmark_parser(spec: BenchmarkSpec) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog=f"mxlab bench {spec.name}", description=spec.description)
        spec.configure(parser)
        return parser

    def reset_for_testing(self) -> None:
        self._specs.clear()
        self._loaded.clear()


_default_registry = BenchmarkRegistry()
# Private aliases retained for existing tests and introspection of the default facade.
_specs = _default_registry._specs
_loaded = _default_registry._loaded


def builtin_benchmarks() -> tuple[BenchmarkSpec, ...]:
    return _default_registry.builtin_benchmarks()


def configure_benchmark_parser(spec: BenchmarkSpec) -> argparse.ArgumentParser:
    return _default_registry.configure_benchmark_parser(spec)


def discover_benchmarks() -> dict[str, BenchmarkSpec]:
    return _default_registry.discover_benchmarks()


def get_benchmark(name: str) -> BenchmarkSpec | None:
    return _default_registry.get_benchmark(name)


def list_benchmarks() -> tuple[BenchmarkSpec, ...]:
    return _default_registry.list_benchmarks()


def register_benchmark(spec: BenchmarkSpec) -> None:
    _default_registry.register_benchmark(spec)


def reset_benchmarks_for_testing() -> None:
    _default_registry.reset_for_testing()


__all__ = [
    "BenchmarkRegistry",
    "BenchmarkSpec",
    "builtin_benchmarks",
    "configure_benchmark_parser",
    "discover_benchmarks",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    "reset_benchmarks_for_testing",
]
