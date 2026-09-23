# Copyright Motphys Technology Co., Ltd. 2025, 2026

"""Boot panel worker-log tail tests: geometry stability is the contract.

The async trainer renders its startup panel with rich Live erase-and-redraw;
the log region must therefore keep a constant line count and line width (no
wrapping) and tolerate workers that have not created their log file yet.
"""

from pathlib import Path

from motrix_rl.fastsac.async_impl.panels import BootPanel, worker_log_tail


def test_tail_shows_placeholder_before_any_log_exists(tmp_path: Path) -> None:
    result = worker_log_tail(tmp_path, ["collector0.log", "learner.log"])

    assert result.plain == "(waiting for worker logs…)"


def test_tail_skips_missing_files_and_keeps_last_two_per_file(tmp_path: Path) -> None:
    (tmp_path / "collector0.log").write_text("old line\nline1\nline2\n")
    # learner.log deliberately absent — learner not spawned its log yet

    result = worker_log_tail(tmp_path, ["collector0.log", "learner.log"])

    assert result.plain == "line1\nline2"


def test_tail_orders_files_in_given_order_newest_last(tmp_path: Path) -> None:
    (tmp_path / "collector0.log").write_text("c-line\n")
    (tmp_path / "learner.log").write_text("l-line\n")

    result = worker_log_tail(tmp_path, ["collector0.log", "learner.log"])

    assert result.plain == "c-line\nl-line"


def test_tail_caps_total_lines_to_max_lines(tmp_path: Path) -> None:
    (tmp_path / "collector0.log").write_text("a\nb\n")
    (tmp_path / "learner.log").write_text("c\nd\n")

    result = worker_log_tail(tmp_path, ["collector0.log", "learner.log"], max_lines=2)

    # keeps the newest tail: the learner's two lines, collector0 dropped
    assert result.plain == "c\nd"


def test_tail_truncates_long_lines_to_fixed_width(tmp_path: Path) -> None:
    long_line = "x" * 500
    (tmp_path / "learner.log").write_text(long_line + "\n")

    result = worker_log_tail(tmp_path, ["learner.log"], line_width=80)

    line = result.plain.splitlines()[0]
    assert len(line) == 80


def test_tail_reads_only_the_end_of_large_files(tmp_path: Path) -> None:
    filler = "\n".join(f"filler-{i}" for i in range(5000))
    (tmp_path / "learner.log").write_text(f"{filler}\nfinal-line\n")

    result = worker_log_tail(tmp_path, ["learner.log"])

    assert "filler-0" not in result.plain
    assert result.plain.endswith("final-line")


def test_tail_blank_lines_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "learner.log").write_text("\n\n   \nreal-line\n\n")

    result = worker_log_tail(tmp_path, ["learner.log"])

    assert result.plain == "real-line"


def test_tail_per_file_controls_lines_read_from_each_file(tmp_path: Path) -> None:
    (tmp_path / "learner.log").write_text("l1\nl2\nl3\n")

    result = worker_log_tail(tmp_path, ["learner.log"], max_lines=3, per_file=3)

    assert result.plain == "l1\nl2\nl3"


def test_tail_splits_evenly_across_files_with_per_file(tmp_path: Path) -> None:
    (tmp_path / "collector0.log").write_text("c1\nc2\nc3\n")
    (tmp_path / "learner.log").write_text("l1\nl2\nl3\n")

    result = worker_log_tail(tmp_path, ["collector0.log", "learner.log"], max_lines=4, per_file=2)

    assert result.plain == "c2\nc3\nl2\nl3"


def test_tail_wrap_folds_long_lines_into_full_width_continuations(tmp_path: Path) -> None:
    long_line = "x" * 100
    (tmp_path / "learner.log").write_text(long_line + "\n")

    result = worker_log_tail(tmp_path, ["learner.log"], max_lines=4, line_width=40, wrap=True)

    lines = result.plain.splitlines()
    # 100 chars fold into ceil(100/40) = 3 full-width continuation lines
    assert [len(ln) for ln in lines] == [40, 40, 20]
    assert "".join(lines) == long_line


def test_tail_wrap_keeps_display_line_count_capped_at_max_lines(tmp_path: Path) -> None:
    (tmp_path / "learner.log").write_text("y" * 200 + "\nshort\n")

    result = worker_log_tail(tmp_path, ["learner.log"], max_lines=2, line_width=40, wrap=True, per_file=2)

    lines = result.plain.splitlines()
    # the long line folds into 5 display lines; the cap keeps only the last 2
    assert len(lines) == 2
    assert lines[-1].endswith("short")


def test_boot_panel_render_fits_terminal_height(tmp_path: Path, monkeypatch) -> None:
    from rich.console import Console

    (tmp_path / "collector0.log").write_text("c-line\n")
    monkeypatch.setattr("motrix_rl.fastsac.async_impl.panels.Console", lambda: Console(width=100, height=20))
    panel = BootPanel(
        title="env/motrix.fastsac — worker startup",
        log_dir=tmp_path,
        log_names=["collector0.log", "learner.log"],
        workers=[("collector", 0), ("learner", 0)],
    )

    probe = Console(width=100, height=20, force_terminal=True, file=open("/dev/null", "w"))
    rendered = ["".join(seg.text for seg in row) for row in probe.render_lines(panel.render({("collector", 0)}))]

    # outer frame 2 lines; table region = 2 workers + 4 + gate 1 = 7;
    # log region = 20 - 2 - 7 = 11; cell = 11 - 2 borders = 9 lines
    assert panel.log_tail_lines == 9
    assert len(rendered) == 20  # full-height frame, never taller
    assert any("ready" in ln for ln in rendered)
    assert any("booting" in ln for ln in rendered)
    assert any("c-line" in ln for ln in rendered)


def test_boot_panel_handoff_mode_renders_progress_and_gate(tmp_path: Path, monkeypatch) -> None:
    from rich.console import Console

    (tmp_path / "collector0.log").write_text("c-line\n")
    import motrix_rl.fastsac.async_impl.panels as panels

    devnull = open("/dev/null", "w")
    monkeypatch.setattr(panels, "Console", lambda: Console(width=100, height=20, force_terminal=True, file=devnull))
    panel = panels.BootPanel(
        "env/motrix.fastsac",
        tmp_path,
        ["collector0.log", "learner.log"],
        [("collector", 0), ("learner", 0)],
    )
    probe = Console(width=100, height=20, force_terminal=True, file=devnull)
    boot = ["".join(s.text for s in row) for row in probe.render_lines(panel.render({("collector", 0)}))]
    handoff = ["".join(s.text for s in row) for row in probe.render_lines(panel.render(set(), gate="g", starting=True))]

    # Boot phase: ready/booting statuses; the reserved gate line is blank.
    boot_text = "\n".join(boot)
    assert "ready" in boot_text and "booting" in boot_text and "c-line" in boot_text
    assert "worker startup" not in boot_text  # table title removed; task name is the frame title
    # Handoff phase: same frame height, every worker "starting", gate line filled.
    handoff_text = "\n".join(handoff)
    assert len(handoff) == len(boot) == 20
    assert handoff_text.count("starting") == 2
    assert "│ g" in handoff_text  # gate line filled inside the outer frame
