# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic rendering and serialization for benchmark results.

Renderers consume only :class:`~motrix_bench.runner.BenchmarkResult` values, so
benchmarks never hand-format tables or JSON themselves.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path

from .runner import BenchmarkResult


# The throughput column divides all produced items by the whole timed window.
# The item unit itself is declared by each benchmark and taken from the results.
def _headers(unit: str) -> tuple[str, ...]:
    return (
        "benchmark",
        "label",
        "steps",
        "p10 ms",
        "median ms",
        "p90 ms",
        f"{unit}/s",
    )


_NUMERIC_CELL = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:%|x)?")
_DASH = {"—", "-"}


def _is_numeric(cell: str) -> bool:
    return _NUMERIC_CELL.fullmatch(cell) is not None


def _result_row(result: BenchmarkResult) -> list[str]:
    return [
        result.benchmark,
        result.label,
        str(result.steps),
        f"{result.p10_ms:.3f}",
        f"{result.median_ms:.3f}",
        f"{result.p90_ms:.3f}",
        "—" if result.sustained_items_per_second is None else f"{result.sustained_items_per_second:,.0f}",
    ]


def render_table(results: Sequence[BenchmarkResult]) -> str:
    """Render results as an aligned ASCII table; numeric columns right-align."""
    if not results:
        return ""
    headers = _headers(results[0].unit)
    rows = [_result_row(result) for result in results]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    right_align = [
        all(_is_numeric(row[column]) or row[column] in _DASH for row in rows) for column in range(len(headers))
    ]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("-" * (width + 2) for width in widths) + right

    def format_row(cells: list[str]) -> str:
        parts = [
            cell.rjust(width) if numeric else cell.ljust(width)
            for cell, width, numeric in zip(cells, widths, right_align)
        ]
        return "| " + " | ".join(parts) + " |"

    return "\n".join(
        [
            border("+", "+", "+"),
            format_row(list(headers)),
            border("+", "+", "+"),
            *(format_row(row) for row in rows),
            border("+", "+", "+"),
        ]
    )


def _to_json_object(result: BenchmarkResult) -> dict[str, object]:
    payload = asdict(result)
    # Round-trip guarantees: tuples serialize as lists; that is fine on the wire
    # but document the canonical key set via the dataclass fields.
    return payload


def render_json(results: Sequence[BenchmarkResult]) -> str:
    """Render results as one JSON object per line (JSON Lines)."""
    return "\n".join(json.dumps(_to_json_object(result), sort_keys=True, allow_nan=False) for result in results)


def render_results(results: Sequence[BenchmarkResult], *, json_output: bool) -> None:
    """Print results as JSON Lines or a table."""
    if not results:
        return
    print(render_json(results) if json_output else render_table(results))


def result_fields() -> tuple[str, ...]:
    return tuple(field.name for field in fields(BenchmarkResult))


def dump_results(path: str | Path, results: Sequence[BenchmarkResult]) -> None:
    """Serialize results to a JSON file (worker/subprocess handoff)."""
    Path(path).write_text(json.dumps([_to_json_object(result) for result in results]), encoding="utf-8")


def load_results(path: str | Path) -> list[BenchmarkResult]:
    """Load results written by :func:`dump_results`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    known = set(result_fields())
    results = []
    for item in payload:
        unknown = set(item) - known
        if unknown:
            raise ValueError(f"unknown benchmark result fields: {sorted(unknown)}")
        item = dict(item)
        samples = item.get("samples_ms")
        if isinstance(samples, list):
            item["samples_ms"] = tuple(samples)
        config = item.get("config")
        if not isinstance(config, Mapping):
            item["config"] = dict(config or {})
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            item["metadata"] = dict(metadata or {})
        results.append(BenchmarkResult(**item))
    return results


__all__ = [
    "dump_results",
    "load_results",
    "render_json",
    "render_results",
    "render_table",
    "result_fields",
]
