# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Rich panel rendering for the async FastSAC trainer's parent process.

The boot panel (worker startup table + worker-log tail) runs on the
alternate screen via ``rich.live.Live`` while workers boot at different
speeds. Everything here is pure rendering; the boot loop, readiness
tracking and error handling stay in ``train.py``. The post-boot training
panel is shared across frameworks and lives in ``motrix_rl.console``.

Frame geometry is the core contract: the panel is a vertical ``Layout``
where the worker table takes a fixed number of lines and the log region
expands to every remaining terminal line. Log lines are either cropped or
folded into full-width continuations, and the DISPLAY line count is capped
so the frame height never changes between refreshes — a Live redraw whose
frame grows, wraps or overflows the terminal cannot erase its previous
frame and tears.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_LOG_TAIL_PLACEHOLDER = "(waiting for worker logs…)"


def worker_log_tail(
    log_dir: Path,
    log_names: Sequence[str],
    max_lines: int = 4,
    line_width: int = 120,
    per_file: int = 2,
    wrap: bool = False,
) -> Text:
    """Render the boot panel's worker-log region.

    Reads the last bytes of each worker log file and returns at most
    ``max_lines`` display lines (the last ``per_file`` source lines per file,
    newest last). Each source line becomes exactly one display line cropped
    to ``line_width``, or — with ``wrap=True`` — as many full-width
    continuation lines as it needs. Either way the panel's geometry stays
    constant across refreshes (the display-line count is always capped at
    ``max_lines``), which is what keeps the Live redraw tear-free.

    Lines are grouped per worker in ``log_names`` order (collectors first,
    learners last), untagged — the surrounding panel/table names the worker.
    Missing files (worker hasn't created its log yet) are skipped silently.
    """
    display: list[str] = []
    for name in log_names:
        path = Path(log_dir) / name
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                chunk = f.read().decode(errors="replace")
        except OSError:
            continue
        tail = [ln for ln in chunk.splitlines() if ln.strip()][-per_file:]
        for ln in tail:
            if wrap:
                display.extend(ln[j : j + line_width] for j in range(0, len(ln), line_width))
            else:
                display.append(ln[:line_width])
    return Text("\n".join(display[-max_lines:]) or _LOG_TAIL_PLACEHOLDER, style="dim")


class BootPanel:
    """Worker-startup panel: worker table below a grid of per-worker log cells."""

    def __init__(self, title: str, log_dir: Path, log_names: Sequence[str], workers: Sequence[tuple[str, int]]) -> None:
        self._title = title
        self._log_dir = Path(log_dir)
        self._log_names = list(log_names)
        self._workers = list(workers)
        self.console = Console()
        # Outer frame mirrors the training panel: task name on top, cyan
        # border. Its border + padding cost 2 lines and 4 columns of the
        # terminal, shrinking the inner regions accordingly.
        self._frame_lines = 2
        self._frame_cols = 4
        # Table height = header + 3 rule rows + one row per worker,
        # plus the always-reserved gate line.
        # The log grid VERTICALLY EXPANDS to every remaining terminal line.
        self._table_lines = len(self._workers) + 4 + 1
        # Near-square grid: 1-2 workers stay in one row, more pack into
        # ceil(sqrt(n)) columns so cells keep a readable width.
        n = max(1, len(self._log_names))
        self._grid_cols = math.ceil(math.sqrt(n))
        self._grid_rows = math.ceil(n / self._grid_cols)
        # Cell inner size: the log region split by the grid shape, minus each
        # cell Panel's border and padding columns/rows.
        region_lines = max(self._grid_rows * 3, self.console.height - self._frame_lines - self._table_lines)
        self.log_tail_lines = max(1, region_lines // self._grid_rows - 2)
        inner_width = self.console.width - self._frame_cols
        self.log_line_width = max(20, inner_width // self._grid_cols - 4)
        # Without a TTY (piped/redirected stdout) Live cannot redraw in place
        # and every refresh appends a new frame; the caller renders one
        # static frame instead (auto_refresh=False, no updates).
        self.interactive = self.console.is_terminal

    def _worker_panel(self, name: str) -> Panel:
        """One worker's log cell: fixed-size Panel padded to full height."""
        worker = name.removesuffix(".log")
        text = worker_log_tail(
            self._log_dir,
            [name],
            self.log_tail_lines,
            self.log_line_width,
            per_file=self.log_tail_lines,
            wrap=True,
        )
        # Pad to a fixed line count so the panel (and thus the whole frame)
        # keeps a constant height between refreshes.
        body = text.plain.splitlines()
        body += [""] * (self.log_tail_lines - len(body))
        return Panel(
            Text("\n".join(body), style="dim"),
            title=worker,
            border_style="dim",
            expand=True,
            height=self.log_tail_lines + 2,
        )

    def render(self, ready: set[tuple[str, int]], gate: str | None = None, starting: bool = False) -> Panel:
        """Render one boot/handoff frame.

        The whole view is wrapped in the same outer frame the training panel
        uses (task name on top, cyan border). ``ready`` drives the boot phase
        (booting…/ready). After the barrier releases, the caller sets
        ``starting`` (every worker flips to a "starting" status) and passes
        ``gate``, the one-line panel-data readiness summary; the same panel
        then serves as the handoff view until the training panel's quiescence
        gate opens, so the terminal never switches views mid-boot.
        """
        inner_width = self.console.width - self._frame_cols
        table = Table(expand=True, width=inner_width)
        table.add_column("worker")
        table.add_column("status")
        for role, idx in sorted(self._workers):
            name = f"{role}[{idx}]"
            if starting:
                table.add_row(name, "[yellow]starting[/yellow]")
            elif (role, idx) in ready:
                table.add_row(name, "[green]ready[/green]")
            else:
                table.add_row(name, "[dim]booting…[/dim]")
        # Worker-log grid: rows of equal-height cells, each cell one worker.
        logs = Layout(name="logs")
        rows = []
        for r in range(self._grid_rows):
            cells = self._log_names[r * self._grid_cols : (r + 1) * self._grid_cols]
            row = Layout(name=f"log-row{r}")
            row.split_row(*(Layout(name=name.removesuffix(".log"), ratio=1) for name in cells))
            rows.append(row)
        logs.split_column(*rows)
        for name in self._log_names:
            logs[name.removesuffix(".log")].update(self._worker_panel(name))
        # The gate line is always reserved (empty during the boot phase) so
        # the frame geometry is identical across the boot→handoff switch.
        bottom = Group(table, Text(gate if gate else "", style="dim"))
        layout = Layout()
        layout.split_column(
            Layout(name="logs"),
            Layout(name="table", size=self._table_lines),
        )
        layout["logs"].update(logs)
        layout["table"].update(bottom)
        return Panel(
            layout,
            title=self._title,
            border_style="cyan",
            padding=(0, 1),
            height=self.console.height,
        )
