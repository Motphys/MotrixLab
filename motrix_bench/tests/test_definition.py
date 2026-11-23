# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass, replace

from motrix_bench import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSpec,
    PreparedCase,
    dump_results,
    load_results,
    render_results,
    run_case,
)


@dataclass(frozen=True, slots=True)
class DemoConfig(BenchmarkConfig):
    subject: str = "a"


def _definition() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="demo.step",
        description="Demo benchmark.",
        configure=lambda parser: None,
        build_config=lambda args: DemoConfig(steps=args.steps, warmup=args.warmup, subject=args.subject),
        prepare=lambda config: PreparedCase(lambda: None, items_per_step=float(len(config.subject))),
    )


def _result(**overrides) -> BenchmarkResult:
    keywords: dict = {"benchmark": "demo.step", "label": "subject a"}
    keywords.update(overrides)
    return run_case(
        PreparedCase(lambda: None, items_per_step=1.0),
        BenchmarkConfig(steps=2, warmup=0),
        **keywords,
    )


def test_definition_build_config_and_prepare_compose():
    definition = _definition()
    args = argparse.Namespace(steps=3, warmup=1, subject="abcd")
    config = definition.build_config(args)
    assert isinstance(config, DemoConfig)
    assert config.steps == 3
    case = definition.prepare(config)
    assert case.items_per_step == 4.0


def test_render_table_has_generic_columns_and_aligns_numbers(capsys):
    results = [_result(), _result(label="subject b")]
    render_results(results, json_output=False)
    out = capsys.readouterr().out
    assert "benchmark" in out and "median ms" in out and "items/s" in out
    assert "demo.step" in out and "subject a" in out


def test_render_json_emits_one_object_per_line(capsys):
    results = [_result(), _result(label="subject b")]
    render_results(results, json_output=True)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert '"benchmark": "demo.step"' in lines[0]


def test_render_results_with_no_results_prints_nothing(capsys):
    render_results([], json_output=False)
    render_results([], json_output=True)
    assert capsys.readouterr().out == ""


def test_dump_and_load_results_round_trip(tmp_path):
    results = [_result(metadata={"note": "shard"}), _result(label="subject b")]
    path = tmp_path / "worker.json"
    dump_results(path, results)
    loaded = load_results(path)
    assert loaded == results
    assert loaded[0].metadata == {"note": "shard"}


def test_load_results_rejects_unknown_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('[{"totally_unknown": 1}]')
    try:
        load_results(path)
    except ValueError as error:
        assert "totally_unknown" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_config_replace_enables_batch_sweeps():
    config = DemoConfig(steps=2, subject="x")
    swept = replace(config, subject="xyz")
    assert swept.steps == 2
    assert swept.subject == "xyz"
