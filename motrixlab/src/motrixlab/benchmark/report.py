# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Console table rendering for bench scripts.

Bench scripts collect result rows while measurements run and render a single
aligned table after the run finishes, so engine log output emitted during the
benchmark cannot interleave with the report.
"""

from __future__ import annotations

import re

_NUMERIC_CELL = re.compile(r"-?[\d.,]+(%|x)?")


def _is_numeric(cell: str) -> bool:
    return _NUMERIC_CELL.fullmatch(cell) is not None


def _is_dash(cell: str) -> bool:
    return cell in {"—", "-"}


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an ASCII table; numeric-looking columns are right-aligned."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    right_align = [
        all(_is_numeric(row[column]) or _is_dash(row[column]) for row in rows) for column in range(len(headers))
    ]

    def border(left: str, middle: str, right: str) -> str:
        return left + middle.join("-" * (width + 2) for width in widths) + right

    def format_row(cells: list[str]) -> str:
        parts = []
        for cell, width, numeric in zip(cells, widths, right_align):
            parts.append(cell.rjust(width) if numeric else cell.ljust(width))
        return "| " + " | ".join(parts) + " |"

    lines = [
        border("+", "+", "+"),
        format_row(headers),
        border("+", "+", "+"),
        *(format_row(row) for row in rows),
        border("+", "+", "+"),
    ]
    return "\n".join(lines)
