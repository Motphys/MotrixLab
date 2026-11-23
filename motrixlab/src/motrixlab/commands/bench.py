# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""``mxlab bench`` named benchmark dispatcher.

The dispatcher is the only CLI-aware layer: it parses arguments into the
benchmark's typed config, executes the benchmark through the standard engine
(or the benchmark's own orchestration), and renders structured results.
Benchmarks themselves never see ``argparse.Namespace``.
"""

from __future__ import annotations

import sys

from motrix_bench import BenchmarkConfig, render_results, run_case
from motrixlab.benchmark.registry import (
    builtin_benchmarks,
    configure_benchmark_parser,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)


def _load_registry() -> None:
    from motrixlab.benchmark import registry

    for spec in builtin_benchmarks():
        if registry.get_benchmark(spec.name) is None:
            register_benchmark(spec)


def _measurement_of(config: object, parser) -> BenchmarkConfig:
    if not isinstance(config, BenchmarkConfig):
        parser.error("benchmark config must extend BenchmarkConfig for the default dispatch path")
    return BenchmarkConfig(config.steps, config.warmup)


def _run_named(argv: list[str]) -> int | None:
    _load_registry()
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: mxlab bench NAME [OPTIONS]")
        print("\nAvailable benchmarks:")
        for spec in list_benchmarks():
            print(f"  {spec.name:<24} {spec.description}")
        return 0
    if argv[0] == "--list":
        for spec in list_benchmarks():
            print(f"{spec.name}\t{spec.description}")
        return 0
    name, *bench_argv = argv
    spec = get_benchmark(name)
    if spec is None:
        available = ", ".join(item.name for item in list_benchmarks()) or "none"
        raise SystemExit(f"unknown benchmark {name!r}; available: {available}")
    parser = configure_benchmark_parser(spec)
    parser.add_argument("--json", action="store_true", help="Emit one JSON object per result instead of a table.")
    args = parser.parse_args(bench_argv)
    try:
        config = spec.build_config(args)
    except ValueError as error:
        parser.error(str(error))
    if spec.run is not None:
        results = spec.run(config)
    else:
        with spec.prepare(config) as case:
            results = [
                run_case(
                    case,
                    _measurement_of(config, parser),
                    benchmark=spec.name,
                )
            ]
    render_results(results, json_output=args.json)
    return 0


def main() -> int | None:
    return _run_named(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main() or 0)
