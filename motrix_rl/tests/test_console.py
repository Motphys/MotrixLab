# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from rich.console import Console

import motrix_rl.console as console_module
from motrix_rl.console import (
    TrainingPanelStats,
    _compact_metric_value,
    _format_memory,
    _format_value,
    format_training_panel,
    render_training_panel,
)
from motrix_rl.system_metrics import CpuLoad, MemoryUsage


def test_format_value_uses_fixed_notation_for_regular_floats() -> None:
    assert _format_value(0.0) == "0.000"
    assert _format_value(0.001) == "0.001"
    assert _format_value(-12.3456, signed=True) == "-12.346"


def test_format_value_uses_scientific_notation_for_small_floats() -> None:
    assert _format_value(0.0003623) == "3.623e-04"
    assert _format_value(-0.00002, precision=4, signed=True) == "-2.0000e-05"


def test_format_value_uses_scientific_notation_for_large_floats() -> None:
    assert _format_value(1_000_000.0) == "1.000e+06"
    assert _format_value(-1_234_567.0, signed=True) == "-1.235e+06"


def test_format_training_panel_groups_timing_metrics() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        timing_groups={"collector": {"wait_ms": 4.0, "env_step_ms": 1.5}},
    )

    panel = format_training_panel(stats)

    assert "collector         wait_ms 4.000" in panel
    assert "env_step_ms 1.500" in panel
    assert "collector_wait_ms" not in panel
    assert "collect 2.0ms" not in panel  # standalone timing row suppressed for tree panels


def test_format_training_panel_renders_nested_timing_groups_as_tree() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        timing_groups={
            "learner(ms)": {
                "drain": 0.5,
                "update": {"total": 3.0, "critic_alpha": 2.0, "actor": 1.0},
            },
        },
    )

    panel = format_training_panel(stats)

    assert "drain 0.500" in panel
    assert sum(1 for line in panel.splitlines() if line.strip() == "└─ update 3.000") == 1
    child_lines = [line for line in panel.splitlines() if line.lstrip().startswith("└─ critic_alpha 2.000")]
    assert len(child_lines) == 1
    assert child_lines[0].startswith(" " * 22)
    assert "actor 1.000" in child_lines[0]


def test_format_training_panel_strips_markup_from_group_titles() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        timing_groups={"learner [magenta]3.0[/]ms": {"drain": 0.5}},
    )

    panel = format_training_panel(stats)

    assert "[magenta]" not in panel
    assert "learner 3.0ms" in panel
    assert "drain 0.500" in panel


def test_format_training_panel_shows_cpu_load_with_explicit_topology() -> None:
    stats = TrainingPanelStats(
        iteration=1,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=0.0,
        mean_episode_length=0.0,
        episodes=0,
        buffer_size=0,
        buffer_capacity=1,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=0.0,
        cpu_load=CpuLoad(
            utilization_percent=50.0,
            used_logical_cpus=16.0,
            logical_cpu_count=32,
            physical_core_count=16,
            iowait_percent=1.5,
            steal_percent=0.25,
        ),
    )

    panel = format_training_panel(stats)

    assert "cpu  50.0% (16.0/32T, 16C)" in panel
    assert "iowait 1.5%" in panel
    assert "steal 0.2%" in panel


def _panel_stats(**overrides: Any) -> TrainingPanelStats:
    values: dict[str, Any] = dict(
        iteration=2,
        total_iterations=10,
        steps_per_second=100.0,
        elapsed_seconds=1.0,
        mean_return=1.5,
        mean_episode_length=8.0,
        episodes=4,
        buffer_size=512,
        buffer_capacity=100_000,
        collect_ms=2.0,
        learn_ms=3.0,
        learn_percent=50.0,
    )
    values.update(overrides)
    return TrainingPanelStats(**values)


def _render_panel(stats: TrainingPanelStats, *, view: str = "overview", width: int = 200) -> str:
    console = Console(width=width)
    with console.capture() as capture:
        console.print(render_training_panel(stats, view=view))
    return capture.get()


def test_render_training_panel_overview_keeps_timing_tree_hidden() -> None:
    stats = _panel_stats(
        timing_groups={
            "collector": {"env_step": 1.0, "sync": {"total": 0.5, "weights": 0.25}},
            "learner": {"update": {"total": 3.0, "critic": 2.0}},
        },
    )

    panel = _render_panel(stats)

    assert "Run progress" in panel
    assert "Episode stats" in panel
    assert "Throughput" in panel
    # run-progress card carries elapsed and remaining-time estimate
    assert "elapsed" in panel
    assert "left" in panel
    assert "System health" in panel
    assert "Training (" in panel
    assert "Environment metrics (" in panel
    # per-stage timing belongs to the detail view only
    assert "STAGE" not in panel
    assert "env_step" not in panel


def test_render_training_panel_detail_view_shows_timing_tree_with_shares() -> None:
    stats = _panel_stats(
        timing_groups={
            "collector": {
                "env_step": {
                    "total": 18.0,
                    "apply_action": 1.0,
                    "physics": {"total": 15.0, "read": 12.0},
                    "reset": 2.0,
                },
                "sync": {"total": 0.5, "weights": 0.25},
            },
            "learner": {"update": {"total": 3.0, "critic": 2.0}},
        },
        timing_metrics={"queue_depth": 1.0},
        diagnostics={"UTD": 0.5},
    )

    overview = _render_panel(stats)
    detail = _render_panel(stats, view="timing")

    assert "STAGE" not in overview
    assert "STAGE" in detail
    assert "collector" in detail
    assert "learner" in detail
    assert "env_step" in detail
    assert "Timing detail" in detail
    assert "Diagnostics" in detail
    # env_step sub-stages render as an indented subtree under their total,
    # with second-level stages (physics -> read) nested one level deeper
    assert "apply_action" in detail
    assert "physics" in detail
    assert "read" in detail
    assert "reset" in detail
    # known group totals render a share column
    assert "%" in detail


def test_render_training_panel_shows_saved_checkpoint_path() -> None:
    panel = _render_panel(_panel_stats(checkpoint_path="/runs/cartpole/model_0000002.pt"))

    assert "saved checkpoint" in panel
    assert "/runs/cartpole/model_0000002.pt" in panel


def test_render_training_panel_reports_system_health() -> None:
    stats = _panel_stats(
        gpu_utilization_percent=85.0,
        memory_usage=MemoryUsage(used_bytes=1024**3, total_bytes=2 * 1024**3),
    )

    panel = _render_panel(stats)

    assert "CPU" in panel
    assert "GPU 85%" in panel
    assert "RAM 1.0/2.0 GiB" in panel
    assert "VRAM n/a" in panel


def test_overview_aggregates_gpu_devices_and_system_view_shows_per_gpu() -> None:
    from motrix_rl.system_metrics import GpuDeviceUsage

    stats = _panel_stats(
        gpu_devices=[
            GpuDeviceUsage(
                index=0,
                utilization_percent=85.0,
                memory=MemoryUsage(used_bytes=1024**3, total_bytes=2 * 1024**3),
                name="NVIDIA GeForce RTX 4090",
            ),
            GpuDeviceUsage(index=1, utilization_percent=40.0, memory=None),
        ]
    )

    overview = _render_panel(stats, view="overview")

    # Overview/timing cards stay aggregate-only (mean util, summed VRAM)
    assert "GPU 62%" in overview
    assert "VRAM 1.0/2.0 GiB" in overview
    assert "GPU0" not in overview
    assert "GPU1" not in overview

    from motrix_rl.system_metrics import CpuLoad

    stats = _panel_stats(
        gpu_devices=stats.gpu_devices,
        cpu_load=CpuLoad(
            utilization_percent=62.5,
            used_logical_cpus=125.0,
            logical_cpu_count=200,
            physical_core_count=100,
            iowait_percent=0.0,
            steal_percent=0.0,
            per_core_percent=tuple(100.0 if i % 50 == 0 else 0.0 for i in range(200)),
            model_name="AMD EPYC 9654 96-Core Processor",
        ),
    )

    system = _render_panel(stats, view="system")

    # The CPU card leads with the host's model name
    assert "EPYC 9654" in system

    # The dedicated system view lists every device, with its model name
    assert "GPU0" in system and "85%" in system
    assert "RTX 4090" in system
    assert "GPU1" in system and "40%" in system
    assert "n/a" in system
    assert "1.0/2.0 GiB" in system
    # per-core spectrum: one glyph per logical core, wrapped at 48 per row
    # (48 glyphs + 47 inter-core gaps span the same width as the old 96-glyph row)
    assert system.count("cores 0-47") == 1
    assert system.count("cores 48-95") == 1
    assert system.count("cores 192-199") == 1
    # default style is the partial-height glyph spectrum: idle cores show the
    # lowest glyph, busy ones the full block, and no bitmap rows appear
    assert "▁" in system
    assert "▄" not in system
    # a dim ceiling line marks each column's 100% reference: lower-eighth
    # blocks on the row above, touching full-height columns below
    assert "▔" not in system
    # overview shows none of the per-core detail
    assert "cores" not in _render_panel(stats, view="overview")


def test_render_training_panel_falls_back_to_aggregate_gpu_without_device_stats() -> None:
    stats = _panel_stats(gpu_utilization_percent=85.0)

    panel = _render_panel(stats)

    assert "GPU 85%" in panel
    assert "VRAM n/a" in panel


def test_cpu_spectrum_style_env_override(monkeypatch) -> None:
    from motrix_rl.system_metrics import CpuLoad

    stats = _panel_stats(
        cpu_load=CpuLoad(
            utilization_percent=50.0,
            used_logical_cpus=1.0,
            logical_cpu_count=2,
            physical_core_count=1,
            iowait_percent=0.0,
            steal_percent=0.0,
            per_core_percent=(100.0, 25.0),
        )
    )

    monkeypatch.setenv("MOTRIX_PANEL_CPU_SPECTRUM", "bitmap")
    bitmap = _render_panel(stats, view="system")
    assert "▄" not in bitmap
    assert bitmap.count("cores 0-1") == 1
    # bitmap mode: 4 stacked rows for one chunk
    assert sum(1 for line in bitmap.splitlines() if "█" in line and "cores" not in line) >= 3

    monkeypatch.setenv("MOTRIX_PANEL_CPU_SPECTRUM", "bogus")
    assert "▂" in _render_panel(stats, view="system")  # unknown value falls back to height


def test_format_memory_renders_gib_and_missing_values() -> None:
    assert _format_memory(None) == "n/a"
    assert _format_memory(MemoryUsage(used_bytes=1024**3, total_bytes=4 * 1024**3)) == "1.0/4.0 GiB"


def test_compact_metric_value_fits_the_nine_char_value_cell() -> None:
    assert _compact_metric_value(0.4, precision=4, signed=True) == "+0.4000"
    # precision-4 magnitudes below 1e-4 would render as "-3.6200e-05"
    assert _compact_metric_value(-3.62e-05, precision=4, signed=True) == "-3.62e-05"
    # precision-3 values already fit until signed ("+1.235e-04" is 10 chars)
    assert _compact_metric_value(1.235e-04, precision=3) == "1.235e-04"
    assert _compact_metric_value(4.56e-04, precision=3, signed=True) == "+4.56e-04"
    for value in (-3.62e-05, -0.0123, 0.4, 1.2345, 999999.5, 12_345_678.9, 123456789):
        assert len(_compact_metric_value(value, precision=4, signed=True)) <= 9


def test_render_training_panel_keeps_metric_values_on_their_label_line(monkeypatch) -> None:
    stats = _panel_stats(
        reward_terms={"torque": -3.62e-05, "action_rate": -0.0123},
        training_metrics={"alpha_loss": 1.235e-04},
    )

    for width in (120, 150, 190):
        monkeypatch.setenv("COLUMNS", str(width))
        lines = _render_panel(stats, width=width).splitlines()
        assert any("torque" in line and "e-05" in line for line in lines), f"wrapped at width {width}"
        assert any("action_rate" in line and "-0.0123" in line for line in lines), f"wrapped at width {width}"
        assert any("alpha_loss" in line and "e-04" in line for line in lines), f"wrapped at width {width}"


def test_render_training_panel_only_advertises_keyboard_on_posix_tty(monkeypatch) -> None:
    # 1/2 key handling needs a POSIX TTY; other platforms get a plain Live
    stats = _panel_stats()
    monkeypatch.setattr(console_module, "_POSIX_TTY", True)
    assert "keyboard: 1/2/3 switch tabs" in _render_panel(stats)
    monkeypatch.setattr(console_module, "_POSIX_TTY", False)
    assert "keyboard" not in _render_panel(stats)


@pytest.fixture
def _reload_console_module() -> Iterator[None]:
    """Reload the console module after the test, once monkeypatch undid import blocks.

    Listed before ``monkeypatch`` in the test signature so this teardown runs last.
    """
    yield
    importlib.reload(console_module)


def test_console_module_degrades_without_posix_tty_support(_reload_console_module, monkeypatch) -> None:
    # Windows has no termios/tty; hidden modules raise ImportError on import.
    for name in ("termios", "tty", "select"):
        monkeypatch.setitem(sys.modules, name, None)
    reloaded = importlib.reload(console_module)

    assert reloaded._POSIX_TTY is False
    # rendering and the plain-text fallback stay fully functional
    stats = _panel_stats()
    assert "Run progress" in _render_panel(stats)
    assert "iter" in reloaded.format_training_panel(stats)
