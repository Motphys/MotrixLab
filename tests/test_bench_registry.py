# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from motrixlab.benchmark import registry
from motrixlab.commands import bench

from motrix_bench import BenchmarkConfig, BenchmarkResult, PreparedCase, run_case


@pytest.fixture(autouse=True)
def clear_registry():
    registry.reset_benchmarks_for_testing()
    yield
    registry.reset_benchmarks_for_testing()


@dataclass(frozen=True, slots=True)
class DemoConfig(BenchmarkConfig):
    value: int = 0


def _spec(name="demo.v1", value=0):
    def configure(parser: argparse.ArgumentParser):
        parser.add_argument("--value", type=int, default=value)

    def build_config(args: argparse.Namespace) -> DemoConfig:
        return DemoConfig(steps=2, warmup=0, value=args.value)

    def prepare(config: DemoConfig) -> PreparedCase:
        return PreparedCase(lambda: None, items_per_step=float(config.value or 1))

    def run(config: DemoConfig) -> list[BenchmarkResult]:
        return [
            run_case(
                PreparedCase(lambda: None, items_per_step=float(config.value or 1)),
                BenchmarkConfig(steps=config.steps, warmup=config.warmup),
                benchmark=name,
                label=f"value={config.value}",
            )
        ]

    return registry.BenchmarkSpec(name, "demo benchmark", configure, build_config, prepare, run)


def test_register_supports_dotted_names_and_lists_sorted(monkeypatch):
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [])
    registry.register_benchmark(_spec("zeta"))
    registry.register_benchmark(_spec("demo.v1"))
    assert [item.name for item in registry.list_benchmarks()] == ["demo.v1", "zeta"]


def test_registry_instances_isolate_state(monkeypatch):
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [])
    first = registry.BenchmarkRegistry()
    second = registry.BenchmarkRegistry()
    first.register_benchmark(_spec("first"))
    second.register_benchmark(_spec("second"))
    assert [item.name for item in first.list_benchmarks()] == ["first"]
    assert [item.name for item in second.list_benchmarks()] == ["second"]


def test_module_functions_are_default_registry_facade(monkeypatch):
    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [])
    registry.register_benchmark(_spec("facade"))
    assert registry.get_benchmark("facade") is registry.list_benchmarks()[0]
    assert registry._default_registry.get_benchmark("facade") is not None


def test_discover_accepts_duck_spec_from_entry_point(monkeypatch):
    class DuckSpec:
        name = "ignored"
        description = "external"
        configure = staticmethod(lambda parser: None)
        build_config = staticmethod(lambda args: DemoConfig(steps=1))
        prepare = staticmethod(lambda config: PreparedCase(lambda: None))
        run = staticmethod(lambda config: [])

    class Entry:
        name = "external.v1"

        @staticmethod
        def load():
            return lambda: DuckSpec()

    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [Entry()])
    found = registry.discover_benchmarks()
    assert found["external.v1"].description == "external"
    assert found["external.v1"].name == "external.v1"


def test_register_rejects_duplicates_and_invalid_names():
    registry.register_benchmark(_spec("demo.v1"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register_benchmark(_spec("demo.v1"))
    with pytest.raises(ValueError, match="invalid benchmark name"):
        registry.register_benchmark(_spec("Demo"))


def test_builtin_specs_are_lazy_until_hook_invocation(monkeypatch):
    imported = []
    module = SimpleNamespace(
        configure=lambda parser: imported.append(("configure", parser.prog)),
        build_config=lambda args: (
            imported.append(("build_config", args.value)) or DemoConfig(steps=1, value=args.value)
        ),
        prepare=lambda config: imported.append(("prepare", config.value)) or PreparedCase(lambda: None),
        run=lambda config: imported.append(("run", config.value)) or [],
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(("import", name)) or module)
    spec = registry.builtin_benchmarks()[0]
    assert imported == []
    parser = argparse.ArgumentParser(prog="test")
    spec.configure(parser)
    assert imported == [("import", "motrixlab.benchmark.env"), ("configure", "test")]
    config = spec.build_config(argparse.Namespace(value=4, steps=2, warmup=0))
    assert config.value == 4
    spec.prepare(config)
    assert spec.run(config) == []
    kinds = [event[0] for event in imported]
    # Every hook loads the module lazily before dispatching.
    assert kinds.count("import") == 4
    assert kinds == ["import", "configure", "import", "build_config", "import", "prepare", "import", "run"]


def test_discover_rejects_malformed_entry_point_spec(monkeypatch):
    class Entry:
        name = "broken"

        @staticmethod
        def load():
            return lambda: object()

    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [Entry()])
    with pytest.raises(TypeError, match="must return a BenchmarkSpec-compatible object"):
        registry.discover_benchmarks()
    assert "broken" not in registry._loaded


def test_discover_rejects_duplicate_entry_points(monkeypatch):
    class Entry:
        def __init__(self, description):
            self.name = "same"
            self.description = description

        def load(self):
            return lambda: _spec("same")

    monkeypatch.setattr(registry, "entry_points", lambda **kwargs: [Entry("first"), Entry("second")])
    with pytest.raises(ValueError, match="already registered"):
        registry.discover_benchmarks()
    assert "same" in registry._specs
    assert "same" in registry._loaded


def test_named_dispatch_builds_config_and_renders_results(monkeypatch, capsys):
    registry.register_benchmark(_spec())
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "demo.v1", "--value", "9"])
    assert bench.main() == 0
    out = capsys.readouterr().out
    assert "demo.v1" in out
    assert "value=9" in out


def test_named_dispatch_json_outputs_structured_results(monkeypatch, capsys):
    registry.register_benchmark(_spec())
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "demo.v1", "--value", "3", "--json"])
    assert bench.main() == 0
    out = capsys.readouterr().out.strip()
    assert '"benchmark": "demo.v1"' in out
    assert '"label": "value=3"' in out


def test_named_dispatch_build_config_errors_exit_two(monkeypatch):
    def build_config(args):
        raise ValueError("value must be positive")

    spec = _spec()
    registry.register_benchmark(
        registry.BenchmarkSpec(spec.name, spec.description, spec.configure, build_config, spec.prepare, spec.run)
    )
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "demo.v1", "--value", "-1"])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2


def test_named_dispatch_default_path_measures_prepared_case(monkeypatch, capsys):
    spec = _spec()
    registry.register_benchmark(
        registry.BenchmarkSpec(spec.name, spec.description, spec.configure, spec.build_config, spec.prepare, None)
    )
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "demo.v1", "--value", "5"])
    assert bench.main() == 0
    out = capsys.readouterr().out
    assert "demo.v1" in out


def test_named_help_lists_benchmarks(monkeypatch, capsys):
    registry.register_benchmark(_spec())
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "--help"])
    assert bench.main() == 0
    assert "demo.v1" in capsys.readouterr().out


def test_named_list_lists_benchmarks(monkeypatch, capsys):
    registry.register_benchmark(_spec())
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "--list"])
    assert bench.main() == 0
    assert "demo.v1\tdemo benchmark" in capsys.readouterr().out


def test_legacy_mode_is_removed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "--mode", "env"])
    with pytest.raises(SystemExit):
        bench.main()
