# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from motrix_bench import BenchmarkConfig, PreparedCase, render_results, run_case


def test_run_case_measures_throughput_and_closes():
    calls = []
    result = run_case(
        PreparedCase(step=lambda: calls.append(1), close=lambda: calls.append("closed"), items_per_step=4),
        BenchmarkConfig(steps=3, warmup=2),
    )
    assert result.steps == 3
    assert result.throughput is not None
    assert calls == [1, 1, 1, 1, 1, "closed"]


def test_run_case_closes_when_step_fails():
    closed = []

    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_case(PreparedCase(fail, close=lambda: closed.append(True)), BenchmarkConfig(steps=1))
    assert closed == [True]


def test_before_step_runs_outside_timed_region_for_warmup_and_samples():
    events = []
    run_case(
        PreparedCase(
            step=lambda: events.append("step"),
            before_step=lambda: events.append("before"),
        ),
        BenchmarkConfig(steps=2, warmup=1),
    )
    assert events == ["before", "step", "before", "step", "before", "step"]


def test_synchronize_hook_runs_for_warmup_and_samples():
    synchronized = []
    run_case(
        PreparedCase(step=lambda: None, synchronize=lambda: synchronized.append(True)),
        BenchmarkConfig(steps=2, warmup=1),
    )
    assert len(synchronized) == 3


def test_typed_config_and_prepared_case_lifecycle_aliases():
    assert BenchmarkConfig(2).steps == 2
    closed = []
    with PreparedCase(lambda: None, close=lambda: closed.append(True)) as case:
        case.step()
    case.close_once()
    assert closed == [True]


def test_result_reports_both_throughput_semantics():
    result = run_case(
        PreparedCase(lambda: None, items_per_step=2),
        BenchmarkConfig(steps=2),
        benchmark="demo",
        label="case-a",
    )
    assert result.benchmark == "demo"
    assert result.label == "case-a"
    assert result.items_per_second is not None
    # Sustained covers the whole timed window; equivalent is the median
    # reciprocal. Both are populated but are distinct statistics.
    assert result.sustained_items_per_second is not None
    assert result.equivalent_items_per_second is not None
    assert result.schema_version == 1
    assert result.throughput == result.sustained_items_per_second


def test_result_without_items_reports_no_throughput():
    result = run_case(PreparedCase(lambda: None), BenchmarkConfig(steps=1))
    assert result.items_per_step is None
    assert result.sustained_items_per_second is None
    assert result.equivalent_items_per_second is None


def test_case_unit_flows_into_result_and_headers(capsys):
    result = run_case(
        PreparedCase(lambda: None, items_per_step=2.0, unit="transitions"),
        BenchmarkConfig(steps=2),
    )
    assert result.unit == "transitions"
    render_results([result], json_output=False)
    assert "transitions/s" in capsys.readouterr().out
