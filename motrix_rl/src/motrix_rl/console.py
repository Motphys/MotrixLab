# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Console display helpers shared by RL training backends."""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from motrix_rl.system_metrics import CpuLoad, GpuDeviceUsage, MemoryUsage

try:  # cbreak keyboard input needs a POSIX terminal; other platforms keep a plain Live
    import select
    import termios
    import tty

    _POSIX_TTY = True
except ImportError:
    _POSIX_TTY = False

try:  # optional pretty console; callers fall back to plain text if unavailable
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False


@dataclass(frozen=True)
class TrainingPanelStats:
    """Structured stats consumed by the shared RL training panel.

    Fields required by the panel layout are explicit dataclass attributes.
    Backend-specific values belong in the extension dictionaries.
    """

    iteration: int
    total_iterations: int
    steps_per_second: float | None
    elapsed_seconds: float
    mean_return: float
    mean_episode_length: float
    episodes: int
    buffer_size: float
    buffer_capacity: float
    collect_ms: float
    learn_ms: float
    learn_percent: float
    warming: bool = False
    # Rolling-window iteration rate; None = same "warming up" semantics as
    # steps_per_second. Falls back to the cumulative average when absent.
    iterations_per_second: float | None = None
    training_metrics: Mapping[str, Any] | None = None
    reward_terms: Mapping[str, Any] = field(default_factory=dict)
    env_metrics: Mapping[str, Any] = field(default_factory=dict)
    timing_metrics: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    # Group name -> items. Item values are scalars, or a nested mapping rendered
    # as an indented sub-tree (e.g. per-process timing with stage breakdowns).
    timing_groups: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    cpu_load: CpuLoad | None = None
    gpu_utilization_percent: float | None = None
    memory_usage: MemoryUsage | None = None
    gpu_memory_usage: MemoryUsage | None = None
    # Per-device GPU utilization/memory; when present the system card renders
    # one line per accelerator instead of the aggregate GPU/VRAM fields.
    gpu_devices: Sequence[GpuDeviceUsage] | None = None
    checkpoint_path: str | None = None


class _InputLive(Live):
    """Rich Live display with direct single-key input from the controlling TTY."""

    def start(self):
        # One try/except spans the whole setup: a termios failure mid-way must
        # still restore the saved TTY state and close an opened /dev/tty FD.
        self._stdin_state = None
        self._input_fd = None
        self._owns_input_fd = False
        try:
            if sys.stdin.isatty():
                self._input_fd = sys.stdin.fileno()
            else:
                try:
                    self._input_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
                    self._owns_input_fd = True
                except OSError:
                    self._input_fd = None
            if self._input_fd is not None:
                fd = self._input_fd
                self._stdin_state = (fd, termios.tcgetattr(fd))
                tty.setcbreak(fd)
                attrs = termios.tcgetattr(fd)
                attrs[3] &= ~termios.ECHO
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
            return super().start()
        except BaseException:
            self._restore_stdin()
            raise

    def stop(self):
        try:
            return super().stop()
        finally:
            self._restore_stdin()

    def _restore_stdin(self):
        state = self._stdin_state
        if state is not None:
            fd, attrs = state
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            self._stdin_state = None
        if self._owns_input_fd and self._input_fd is not None:
            os.close(self._input_fd)
        self._input_fd = None
        self._owns_input_fd = False


def open_training_live():
    """Start a rich live panel on TTYs, otherwise return ``(None, None)``."""
    if not _RICH or not sys.stdout.isatty():
        return None, None
    console = Console()
    live = (_InputLive if _POSIX_TTY else Live)(console=console, auto_refresh=False, vertical_overflow="visible")
    live.start()
    return console, live


def emit_training_panel(live, stats: TrainingPanelStats, *, title: str = "rl") -> None:
    """Render one training panel; 1/2/3 switch overview, timing and system views."""
    if live is not None:
        view = getattr(live, "_motrix_view", "overview")
        input_fd = getattr(live, "_input_fd", None)
        if input_fd is not None:
            ready, _, _ = select.select([input_fd], [], [], 0)
            if ready:
                try:
                    command = os.read(input_fd, 1).decode("utf-8", errors="ignore").lower()
                except OSError:
                    command = ""
                if command in ("1", "2", "3"):
                    view = {"1": "overview", "2": "timing", "3": "system"}[command]
        live._motrix_view = view
        panel = render_training_panel(stats, title=title, view=view)
        live.update(panel, refresh=True)
    else:
        print(format_training_panel(stats, title=title))


def _format_value(value: Any, precision: int = 3, signed: bool = False) -> str:
    if isinstance(value, float):
        sign = "+" if signed else ""
        magnitude = abs(value)
        use_scientific = math.isfinite(value) and value != 0.0 and (magnitude < 10**-precision or magnitude >= 1e6)
        notation = "e" if use_scientific else "f"
        return f"{value:{sign}.{precision}{notation}}"
    if isinstance(value, int):
        return f"{value:+d}" if signed else str(value)
    return str(value)


def _format_metric_items(items: Mapping[str, Any], *, precision: int = 3, signed: bool = False) -> list[str]:
    return [f"{k} {_format_value(v, precision=precision, signed=signed)}" for k, v in items.items()]


def _format_duration(seconds: float) -> str:
    """Compact elapsed/remaining time: ``3h05m``, ``12m07s`` or ``0m45s``."""
    t = max(0, int(seconds))
    h, m, sec = t // 3600, (t % 3600) // 60, t % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def _eta_seconds(stats: TrainingPanelStats) -> float | None:
    """Remaining-time estimate from the cumulative iteration rate."""
    if stats.iteration <= 0 or stats.elapsed_seconds <= 0.0:
        return None
    rate = stats.iteration / stats.elapsed_seconds
    if rate <= 0.0:
        return None
    return (stats.total_iterations - stats.iteration) / rate


def _iter_rate_text(stats: TrainingPanelStats) -> str:
    if stats.iterations_per_second is not None:
        return f"{stats.iterations_per_second:,.0f} iter/s"
    if stats.steps_per_second is None:
        return "warming up"
    return f"{stats.iteration / max(stats.elapsed_seconds, 1e-9):,.0f} iter/s"


def format_training_panel(stats: TrainingPanelStats, *, title: str = "rl") -> str:
    """Render a plain-text RL training panel from backend-provided scalar stats."""

    hms = _format_duration

    def si(n: float) -> str:
        for unit in ("", "k", "M"):
            if abs(n) < 1000:
                return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
            n /= 1000.0
        return f"{n:.1f}G"

    width = 84
    pct = 100.0 * stats.iteration / max(stats.total_iterations, 1)
    eta = _eta_seconds(stats)
    eta_text = f" - eta ~{hms(eta)}" if eta is not None else ""
    sps_text = f"{stats.steps_per_second:.0f} env-steps/s" if stats.steps_per_second is not None else "warming up"
    header = (
        f" {title} - iter {stats.iteration}/{stats.total_iterations} ({pct:.1f}%) - "
        f"{sps_text} - {hms(stats.elapsed_seconds)}{eta_text}"
    )
    lines = [
        "-" * width,
        header,
        "-" * width,
        f" rollout     return {stats.mean_return:>9.3f}   "
        f"ep_len {stats.mean_episode_length:>7.1f}   episodes {stats.episodes:>6d}",
    ]
    if stats.cpu_load is not None:
        load = stats.cpu_load
        cores = f", {load.physical_core_count}C" if load.physical_core_count is not None else ""
        lines.append(
            f" system      cpu {load.utilization_percent:>5.1f}% "
            f"({load.used_logical_cpus:.1f}/{load.logical_cpu_count}T{cores})   "
            f"iowait {load.iowait_percent:.1f}%   steal {load.steal_percent:.1f}%"
        )
    # Plain-text fallback mirrors the overview card: aggregate GPU stats only
    # (per-device detail is the interactive System view's job).
    devices = stats.gpu_devices or ()
    if devices:
        gpu_util, gpu_mem = _aggregate_gpu_devices(devices)
    else:
        gpu_util, gpu_mem = stats.gpu_utilization_percent, stats.gpu_memory_usage
    if gpu_util is not None or gpu_mem is not None:
        parts = [f"{gpu_util:.0f}%" if gpu_util is not None else "n/a", _format_memory(gpu_mem)]
        lines.append(" system      gpu " + "  ".join(parts))
    metrics = stats.training_metrics
    buf = f"{si(stats.buffer_size)}/{si(stats.buffer_capacity)}"
    if metrics is None:
        message = "warmup - filling replay buffer" if stats.warming else "training metrics unavailable"
        lines.append(f" train       {message} ({buf})")
    else:
        toks = _format_metric_items(metrics)
        colw = max(len(t) for t in toks) + 2 if toks else 1
        per_line = max(1, (width - 13) // colw)
        for i in range(0, len(toks), per_line):
            prefix = " train       " if i == 0 else " " * 13
            row = "".join(t.ljust(colw) for t in toks[i : i + per_line])
            lines.append((prefix + row).rstrip())
        lines.append(f" buffer      {buf}")

    terms = stats.reward_terms
    if terms:
        sorted_terms = dict(sorted(terms.items(), key=lambda kv: -abs(float(kv[1]))))
        toks = _format_metric_items(sorted_terms, precision=4, signed=True)
        colw = max(len(t) for t in toks) + 2
        per_line = max(1, (width - 13) // colw)
        for i in range(0, len(toks), per_line):
            prefix = " reward(dt)  " if i == 0 else " " * 13
            row = "".join(t.ljust(colw) for t in toks[i : i + per_line])
            lines.append((prefix + row).rstrip())

    env_metrics = stats.env_metrics
    if env_metrics:
        toks = _format_metric_items(dict(sorted(env_metrics.items())), signed=True)
        lines.append(" metrics     " + "   ".join(toks))

    # The standalone collect/learn row is for panels without timing groups;
    # tree panels fold those values into their group titles instead.
    if not stats.timing_groups:
        lines.append(f" timing      collect {stats.collect_ms:.1f}ms   learn {stats.learn_ms:.1f}ms")
    if stats.timing_metrics:
        lines.append(" timing(detail) " + "   ".join(_format_metric_items(stats.timing_metrics)))
    for group, items in stats.timing_groups.items():
        _append_timing_group_lines(lines, group, items, width)
    if stats.diagnostics:
        lines.append(" diagnostics " + "   ".join(_format_metric_items(stats.diagnostics)))
    lines.append("-" * width)
    return "\n".join(lines)


def _strip_markup(text: str) -> str:
    """Drop rich-style ``[tag]`` markup so plain-text output stays clean."""
    return re.sub(r"\[[^\]]*\]", "", text)


def _append_timing_group_lines(lines: list[str], group: str, items: Mapping[str, Any], width: int) -> None:
    """Append one timing group as a tree of aligned ``key value`` token rows.

    Scalar items flow as wrapped tokens on the group's label column. A nested
    mapping renders as a tree branch (``├─``/``└─`` by sibling position) whose
    children form an indented token block, each wrapped row joined with branch
    glyphs so the hierarchy stays visible. A ``total`` key inside a nested
    mapping is the node's own aggregate and renders inline after its name.
    Group titles may carry rich markup; it is stripped for plain text.
    """
    group_name = _strip_markup(group)
    group_width = max(18, len(group_name) + 1)
    row_prefix = " " * (group_width + 1)

    def token(key: str, value: Any) -> str:
        return f"{key} {_format_value(value)}"

    def render_block(
        toks: list[str], first_prefix: str, cont_prefix: str, last_prefix: str, single_prefix: str | None = None
    ) -> None:
        if not toks:
            return
        colw = max(len(t) for t in toks) + 2
        per_line = max(1, (width - len(cont_prefix)) // colw)
        chunks = [toks[i : i + per_line] for i in range(0, len(toks), per_line)]
        for j, chunk in enumerate(chunks):
            if len(chunks) == 1:
                prefix = single_prefix if single_prefix is not None else first_prefix
            elif j == 0:
                prefix = first_prefix
            elif j == len(chunks) - 1:
                prefix = last_prefix
            else:
                prefix = cont_prefix
            lines.append((prefix + "".join(t.ljust(colw) for t in chunk)).rstrip())

    pending: list[str] = []
    lead = f" {group_name:<{group_width}}"
    entries = list(items.items())
    for idx, (key, value) in enumerate(entries):
        if isinstance(value, Mapping):
            render_block(pending, lead, row_prefix, row_prefix)
            pending = []
            lead = row_prefix
            branch, hang = ("└─", "   ") if idx == len(entries) - 1 else ("├─", "│  ")
            below = row_prefix + hang
            children = dict(value)
            node_total = children.pop("total", None)
            label = f"{key} {_format_value(node_total)}" if node_total is not None else f"{key}:"
            lines.append(f"{row_prefix}{branch} {label}")
            render_block(
                [token(k, v) for k, v in children.items()],
                below + "├─ ",
                below + "├─ ",
                below + "└─ ",
                single_prefix=below + "└─ ",
            )
        else:
            pending.append(token(key, value))
    render_block(pending, lead, row_prefix, row_prefix)


def _timing_totals(stats: TrainingPanelStats) -> tuple[float, float]:
    """Return the canonical parent totals stored on TrainingPanelStats."""
    return stats.collect_ms, stats.learn_ms


def _timing_group_total(group: str, stats: TrainingPanelStats, index: int) -> float:
    name = _strip_markup(group).lower()
    if name.startswith("collector"):
        return stats.collect_ms
    if name.startswith("learner") and "idle" not in name:
        return stats.learn_ms
    return stats.collect_ms if index == 0 else stats.learn_ms


# ---------------------------------------------------------------------------
# Prototype layout renderer
#
# Keep the plain-text helpers above for callers that imported them during the
# initial console experiment.  The definitions below are the public renderer
# used by ``emit_training_panel`` and deliberately mirror the compact/timing
# prototypes in docs/prototypes.
def _prototype_metric_cell(
    key: str,
    value: Any,
    *,
    precision: int = 3,
    signed: bool = False,
    label_width: int = 22,
):
    from rich.text import Text

    name = _strip_markup(str(key))
    # Reward terms commonly have descriptive names, so keep a wider label
    # while still bounding unusual keys to preserve the grid layout.
    if len(name) > label_width:
        name = name[: label_width - 3] + "..."
    text = Text(f"{name:<{label_width}} ", style="white")
    color = "white"
    if signed and isinstance(value, (float, int)):
        color = "green" if value >= 0 else "red"
    text.append(f"{_compact_metric_value(value, precision=precision, signed=signed):>9}", style=color)
    return text


def _compact_metric_value(value: Any, *, precision: int = 3, signed: bool = False) -> str:
    """Fit a metric value into the fixed 9-char cell so metric rows never wrap.

    Values that outgrow the cell (e.g. ``-3.6200e-05`` at precision 4) fall
    back to two-digit scientific notation, keeping the grid geometry stable
    whatever magnitudes are currently on screen.
    """
    formatted = _format_value(value, precision=precision, signed=signed)
    if len(formatted) <= 9 or not isinstance(value, (int, float)) or not math.isfinite(value):
        return formatted
    return f"{value:+.2e}" if signed else f"{value:.2e}"


def _rich_compact_metric_grid(items: Mapping[str, Any], *, limit: int | None, precision: int = 3, signed: bool = False):
    grid = Table.grid(expand=True, padding=(0, 1))
    # A metric cell is roughly 25–30 terminal columns wide. Adapt the number
    # of columns to the real TTY instead of forcing every group into two.
    terminal_width = shutil.get_terminal_size((120, 24)).columns
    column_count = max(2, min(4, terminal_width // 30))
    for index in range(column_count):
        grid.add_column(ratio=1)
        if index < column_count - 1:
            grid.add_column(width=1, justify="center")
    pairs = list(items.items())
    visible = pairs if limit is None else pairs[:limit]
    # Compact grids render one Panel deep (Training/Rewards/Environment), so
    # budget for that panel's border and padding. Each metric cell needs label
    # + separator + 9-char value; deriving the label width from the exact rich
    # column arithmetic keeps every cell on one line down to narrow widths.
    column_width = (terminal_width - 8 - 3 * column_count) // column_count
    label_width = max(12, min(32, column_width - 11))
    cells = [
        _prototype_metric_cell(k, v, precision=precision, signed=signed, label_width=label_width) for k, v in visible
    ]
    omitted = 0 if limit is None else len(pairs) - len(visible)
    if omitted:
        from rich.text import Text

        cells.append(Text(f"+{omitted} more", style="cyan"))
    for index in range(0, len(cells), column_count):
        row: list[Any] = []
        for column, cell in enumerate(cells[index : index + column_count]):
            if column:
                from rich.text import Text

                row.append(Text("│", style="grey50"))
            row.append(cell)
        grid.add_row(*row)
    return grid


def _rich_timing_grid(items: Mapping[str, Any], *, total: float | None = None):
    from rich.text import Text

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(justify="right", width=11)
    grid.add_column(justify="right", width=8)
    grid.add_row(Text("STAGE", style="dim"), Text("MEAN", style="dim"), Text("SHARE", style="dim"))

    def append(values: Mapping[str, Any], level: int = 0) -> None:
        entries = list(values.items())
        for index, (key, value) in enumerate(entries):
            branch = "'- " if index == len(entries) - 1 else "|- "
            prefix = "  " * level + branch
            if isinstance(value, Mapping):
                children = dict(value)
                node_total = children.pop("total", None)
                if node_total is not None:
                    share = f"{100.0 * float(node_total) / total:.0f}%" if total else "-"
                    grid.add_row(Text(prefix + str(key), style="white"), f"{_format_value(node_total)} ms", share)
                append(children, level + 1)
            else:
                share = f"{100.0 * float(value) / total:.0f}%" if total else "-"
                grid.add_row(Text(prefix + str(key), style="white"), f"{_format_value(value)} ms", share)

    append(items)
    return grid


def _prototype_bar(fraction: float, *, width: int = 22, style: str = "cyan"):
    from rich.text import Text

    filled = max(0, min(width, round(width * fraction)))
    result = Text("━" * filled, style=style)
    result.append("━" * (width - filled), style="grey23")
    return result


def _cpu_spectrum_rows(
    per_core: Sequence[float], load_style, core_ids: Sequence[int] | None = None, per_row: int = 48
) -> list:
    """Per-core CPU utilization spectrum rows for the System view.

    Style is selected by ``MOTRIX_PANEL_CPU_SPECTRUM`` (height | shade |
    bitmap; default height). The code cannot probe the terminal font, so the
    override exists for fonts that render partial-height or shade block
    glyphs as indistinguishable equal blocks:

    * ``height`` — one ``▁▂▃▄▅▆▇█`` glyph per core (prettiest; needs a font
      with correct block-element metrics, e.g. Cascadia Mono, Sarasa Term,
      any Nerd Font);
    * ``shade`` — density glyphs ``░▒▓█``, same height by design;
    * ``bitmap`` — btop-style columns of ``█``/spaces on 4 rows, only two
      distinct characters so it survives every font.

    One space separates adjacent cores so same-height neighbours don't fuse
    into a solid bar; the default row of 48 cores + 47 gaps spans the same
    ~95 columns as the previous unspaced 96-core row.
    """
    from rich.text import Text

    style = os.environ.get("MOTRIX_PANEL_CPU_SPECTRUM", "height").lower()
    if style not in ("height", "shade", "bitmap"):
        style = "height"
    rows: list = []
    label_width = 14  # "cores 192-199  "
    for start in range(0, len(per_core), per_row):
        chunk = per_core[start : start + per_row]
        if core_ids is not None:
            ids = core_ids[start : start + per_row]
            label = f"cores {ids[0]}-{ids[-1]}".ljust(label_width)
        else:
            label = f"cores {start}-{start + len(chunk) - 1}".ljust(label_width)
        if style in ("height", "shade"):
            # Leading space: idle cores render blank so the dim ▁ ceiling caps
            # are the ONLY ▁ on screen (otherwise an idle column would look
            # like it touches the 100% reference).
            levels = " ▁▂▃▄▅▆▇█" if style == "height" else "░▒▓█"
            spectrum = Text()
            for i, value in enumerate(chunk):
                if i:
                    spectrum.append(" ")
                glyph = levels[min(int(value / 100.0 * (len(levels) - 1)), len(levels) - 1)]
                spectrum.append(glyph, style=load_style(value))
            if style == "height":
                # Ceiling line above the skyline: marks each column's 100%
                # reference. The lower-one-eighth block is used because it sits
                # at the BOTTOM of its own cell — a full-height column below
                # touches it, forming one connected bar (▔ would leave a 7/8-row
                # gap between the cap and the skyline).
                rows.append(Text.assemble((" " * label_width, "dim"), (" ".join("▁" for _ in chunk), "dim")))
            rows.append(Text.assemble((label, "dim"), (spectrum,)))
        else:
            height = 4
            for level in range(height, 0, -1):  # top row first
                threshold = 100.0 * (level - 1) / height
                row = Text()
                for i, value in enumerate(chunk):
                    if i:
                        row.append(" ")
                    row.append(
                        "█" if value > threshold else " ", style=load_style(value) if value > threshold else "dim"
                    )
                rows.append(Text.assemble((label if level == height else " " * label_width, "dim"), (row,)))
    return rows


def _prototype_system_page(stats: TrainingPanelStats, load_style, memory_style):
    """Dedicated per-device system view: host summary + one row per GPU.

    The overview/timing cards stay aggregate-only, so 8-GPU hosts get their
    per-device detail here instead of an overflowing System health card. CPU
    load is shown as a compact per-core spectrum (one block glyph per logical
    core, height and color by utilization) — readable at 192 cores without a
    per-core table.
    """
    from rich.table import Table
    from rich.text import Text

    load = stats.cpu_load
    cpu_parts: list[Any] = []
    if load is not None and load.model_name:
        cpu_parts.append(Text(load.model_name, style="dim"))
    if load is not None and load.per_core_percent:
        cpu_parts.extend(_cpu_spectrum_rows(load.per_core_percent, load_style, load.per_core_ids))
    else:
        # RAM is not repeated here — the always-visible System health card in
        # the summary row already carries it.
        cpu_parts.append(Text("no per-core samples", style="dim"))

    devices = stats.gpu_devices or ()
    columns = Table(header_style="dim", expand=True, show_edge=False, box=None)
    columns.add_column("Device", no_wrap=True)
    columns.add_column("Utilization", justify="right", no_wrap=True)
    columns.add_column("VRAM", justify="right", no_wrap=True)
    columns.add_column("VRAM load", ratio=1)
    if not devices:
        columns.add_row(
            Text("no per-device GPU samples", style="dim"),
            Text(
                f"{stats.gpu_utilization_percent:.0f}%" if stats.gpu_utilization_percent is not None else "n/a",
                style=load_style(stats.gpu_utilization_percent),
            ),
            Text(_format_memory(stats.gpu_memory_usage), style=memory_style(stats.gpu_memory_usage)),
            "",
        )
    for device in devices:
        util = device.utilization_percent
        bar = ""
        if device.memory is not None and device.memory.total_bytes > 0:
            width = 16
            filled = round(width * device.memory.used_bytes / device.memory.total_bytes)
            bar = "█" * filled + "░" * (width - filled)
        device_cell = Text(f"GPU{device.index}", style="white")
        if device.name:
            device_cell.append(f"  {device.name}", style="dim")
        columns.add_row(
            device_cell,
            Text(f"{util:.0f}%" if util is not None else "n/a", style=load_style(util)),
            Text(_format_memory(device.memory), style=memory_style(device.memory)),
            Text(bar, style=memory_style(device.memory)),
        )

    # CPU and GPU are separate blocks: the aggregate CPU summary sits in the
    # CPU card's title (top-right corner), RAM heads the body next to the
    # per-core spectrum, and the device table fills the GPU card.
    if load is not None:
        summary = (
            f"CPU {load.utilization_percent:.0f}% "
            f"({load.used_logical_cpus:.1f}/{load.logical_cpu_count}T"
            + (f", {load.physical_core_count}C" if load.physical_core_count is not None else "")
            + ")"
        )
        cpu_title = Text(summary, justify="right", style=f"bold {load_style(load.utilization_percent)}")
    else:
        cpu_title = Text("CPU n/a", justify="right", style="dim")
    cpu_card = Panel(
        Group(*cpu_parts),
        title=cpu_title,
        border_style="grey37",
        padding=(0, 1),
    )
    gpu_card = Panel(
        columns,
        title=f"GPU ({len(devices)} devices)" if devices else "GPU",
        border_style="grey37",
        padding=(0, 1),
    )
    return Group(cpu_card, gpu_card)


def _prototype_tabs(stats: TrainingPanelStats, view: str):
    from rich.text import Text

    tabs = Table.grid(expand=True, padding=(0, 2))
    tabs.add_column()
    tabs.add_column()
    tabs.add_column()
    tabs.add_column(ratio=1, justify="right")
    labels = ("Overview", "Timing", "System")
    row = [Text(label, style="bold cyan" if label.lower() == view else "dim") for label in labels]
    if _POSIX_TTY:  # key handling is POSIX-only; don't advertise it elsewhere
        row.append(Text("keyboard: 1/2/3 switch tabs", style="dim"))
    tabs.add_row(*row)
    return tabs


def _format_memory(memory: MemoryUsage | None) -> str:
    if memory is None:
        return "n/a"
    gib = 1024**3
    return f"{memory.used_bytes / gib:.1f}/{memory.total_bytes / gib:.1f} GiB"


def _aggregate_gpu_devices(devices: Sequence[GpuDeviceUsage]) -> tuple[float | None, MemoryUsage | None]:
    """Mean utilization and summed memory across the reported devices."""
    utils = [d.utilization_percent for d in devices if d.utilization_percent is not None]
    mean_util = sum(utils) / len(utils) if utils else None
    known = [d.memory for d in devices if d.memory is not None and d.memory.total_bytes > 0]
    total_mem = (
        MemoryUsage(used_bytes=sum(m.used_bytes for m in known), total_bytes=sum(m.total_bytes for m in known))
        if known
        else None
    )
    return mean_util, total_mem


def _run_progress_time_text(stats: TrainingPanelStats):
    """One ``elapsed · ~remaining`` line for the Run progress card."""
    from rich.text import Text

    elapsed = _format_duration(stats.elapsed_seconds)
    eta = _eta_seconds(stats)
    line = Text(f"{elapsed} elapsed", style="dim")
    if eta is not None:
        line.append(f" · ~{_format_duration(eta)} left", style="cyan")
    return line


def render_training_panel(stats: TrainingPanelStats, *, title: str = "rl", view: str = "overview"):
    if not _RICH:
        raise RuntimeError("rich is not available")
    from rich.text import Text

    if view not in ("overview", "timing", "system"):
        raise ValueError(f"unknown panel view {view!r}")
    progress = max(0.0, min(1.0, stats.iteration / max(stats.total_iterations, 1)))
    collect, learn = _timing_totals(stats)

    def card(label: str, body: Any, footer: Any = "") -> Panel:
        parts: list[Any] = [body]
        if footer:
            parts.append(footer)
        return Panel(Group(*parts), title=label, border_style="grey37", padding=(0, 1))

    # Overview keeps the five operator-facing cards on one row. Replay-buffer
    # occupancy remains available in the training/metrics data, but is not a
    # headline card.
    summary = Table.grid(expand=True, padding=(0, 1))
    # System health contains two side-by-side stacks (CPU/GPU and RAM/VRAM),
    # so give it 1.5x the width of the other cards instead of forcing each
    # value into a narrow half-column.
    for ratio in (2, 2, 2, 2, 3):
        summary.add_column(ratio=ratio)
    load = stats.cpu_load

    def load_style(value: float | None) -> str:
        if value is None:
            return "dim"
        return "green" if value < 70.0 else "yellow" if value < 90.0 else "red"

    def memory_style(memory: MemoryUsage | None) -> str:
        if memory is None or memory.total_bytes <= 0:
            return "dim"
        ratio = memory.used_bytes / memory.total_bytes
        return "green" if ratio < 0.70 else "yellow" if ratio < 0.90 else "red"

    cpu_text = f"CPU {load.utilization_percent:.0f}%" if load is not None else "CPU n/a"
    progress_row = Table.grid(expand=True, padding=(0, 1))
    progress_row.add_column(ratio=1)
    progress_row.add_row(_prototype_bar(progress, width=18))
    # The overview/timing cards stay aggregate-only; per-device detail lives on
    # the dedicated System view so 8-GPU hosts don't blow up the card layout.
    devices = stats.gpu_devices or ()
    if devices:
        gpu_util, gpu_mem = _aggregate_gpu_devices(devices)
    else:
        gpu_util, gpu_mem = stats.gpu_utilization_percent, stats.gpu_memory_usage
    left_group = Group(
        Text(cpu_text, style=f"bold {load_style(load.utilization_percent if load else None)}"),
        Text(f"GPU {gpu_util:.0f}%" if gpu_util is not None else "GPU n/a", style=f"{load_style(gpu_util)}"),
    )
    right_group = Group(
        Text(f"RAM {_format_memory(stats.memory_usage)}", style=memory_style(stats.memory_usage)),
        Text(f"VRAM {_format_memory(gpu_mem)}", style=memory_style(gpu_mem)),
    )
    system_health = Table.grid(expand=True, padding=(0, 1))
    system_health.add_column(ratio=1)
    system_health.add_column(ratio=1)
    system_health.add_row(left_group, right_group)
    summary.add_row(
        card(
            f"Run progress ({progress * 100:.1f}%)",
            Group(
                Text(f"{stats.iteration:,}/{stats.total_iterations:,} iters", style="white"),
                progress_row,
                _run_progress_time_text(stats),
            ),
        ),
        card(
            "Episode stats",
            Group(
                Text(f"return  {stats.mean_return:+.2f}", style="bold green"),
                Text(f"length  {stats.mean_episode_length:.1f}", style="white"),
            ),
        ),
        card(
            "Throughput",
            Group(
                Text(
                    f"{stats.steps_per_second:,.0f} env-steps/s"
                    if stats.steps_per_second is not None
                    else "warming up",
                    style="bold cyan",
                ),
                Text(_iter_rate_text(stats), style="white"),
            ),
        ),
        card(
            "Timing",
            Group(
                Text(f"Collect {collect:.1f} ms", style="bold yellow"),
                Text(f"Learn {learn:.1f} ms", style="magenta"),
            ),
        ),
        card("System health", system_health),
    )

    # UTD (update-to-data ratio) describes learner/training efficiency, so
    # keep it with algorithm metrics rather than environment observations.
    utd_items = {k: v for k, v in stats.diagnostics.items() if str(k).strip().lower() == "utd"}
    train_items = dict(stats.training_metrics or {})
    train_items.update(utd_items)
    train_parts: list[Any] = []
    if train_items:
        train_parts.append(_rich_compact_metric_grid(train_items, limit=None))
    else:
        train_parts.append(
            Text("warmup - filling replay buffer" if stats.warming else "training metrics unavailable", style="grey70")
        )
    training = Panel(Group(*train_parts), title=f"Training ({len(train_items)})", border_style="grey37", padding=(0, 1))
    left_blocks: list[Any] = [training]
    if stats.reward_terms:
        rewards = dict(sorted(stats.reward_terms.items(), key=lambda kv: -abs(float(kv[1]))))
        left_blocks.append(
            Panel(
                _rich_compact_metric_grid(rewards, limit=None, precision=4, signed=True),
                title=f"Rewards ({len(rewards)})",
                border_style="grey37",
                padding=(0, 1),
            )
        )

    env_items = dict(sorted(stats.env_metrics.items()))
    env_parts: list[Any] = []
    if env_items:
        env_parts.append(_rich_compact_metric_grid(env_items, limit=None, signed=True))
    else:
        env_parts.append(Text("no environment metrics reported", style="grey70"))
    other_diagnostics = {k: v for k, v in stats.diagnostics.items() if str(k).strip().lower() != "utd"}
    if other_diagnostics:
        env_parts.append(
            Text(" · ".join(f"{k} {_format_value(v)}" for k, v in other_diagnostics.items()), style="cyan")
        )
    environment = Panel(
        Group(*env_parts), title=f"Environment metrics ({len(env_items)})", border_style="grey37", padding=(0, 1)
    )

    if view == "system":
        lower = Group(_prototype_system_page(stats, load_style, memory_style))
    elif view == "timing":
        timing_blocks: list[Any] = []
        columns = Table.grid(expand=True, padding=(0, 1))
        columns.add_column(ratio=1)
        columns.add_column(ratio=1)
        groups = list(stats.timing_groups.items())
        for index in range(0, len(groups), 2):
            row: list[Any] = []
            for offset in (0, 1):
                if index + offset >= len(groups):
                    row.append("")
                    continue
                group, items = groups[index + offset]
                total = _timing_group_total(group, stats, index + offset)
                row.append(
                    Panel(
                        _rich_timing_grid(items, total=total),
                        title=f"{_strip_markup(group)}  {total:.1f} ms",
                        border_style="grey37",
                        padding=(0, 1),
                    )
                )
            columns.add_row(*row)
        if groups:
            timing_blocks.append(columns)
        if stats.timing_metrics:
            timing_blocks.append(
                Panel(
                    _rich_compact_metric_grid(stats.timing_metrics, limit=None),
                    title="Timing detail",
                    border_style="grey37",
                    padding=(0, 1),
                )
            )
        if stats.diagnostics:
            timing_blocks.append(
                Panel(
                    _rich_compact_metric_grid(stats.diagnostics, limit=None, signed=True),
                    title="Diagnostics",
                    border_style="grey37",
                    padding=(0, 1),
                )
            )
        lower = Group(*timing_blocks) if timing_blocks else Text("timing details unavailable", style="grey70")
    else:
        lower = Group(*left_blocks, environment)
    blocks = [summary, lower, _prototype_tabs(stats, view)]
    if stats.checkpoint_path:
        blocks.insert(-1, Text(f"✓ saved checkpoint  {stats.checkpoint_path}", style="green"))
    # Let Rich use the actual terminal width.  The card bodies are intentionally
    # single-line so wide terminals gain space without introducing vertical gaps.
    return Panel(Group(*blocks), title=_strip_markup(title), border_style="cyan", padding=(0, 1))
