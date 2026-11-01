# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl.console import EpisodeWindow, PanelClock, RateTracker, ScalarWindow, TimingWindow


def test_panel_clock_uses_monotonic_start() -> None:
    clock = PanelClock(start=10.0)
    assert clock.elapsed(now=12.5) == 2.5
    clock.restart(now=20.0)
    assert clock.elapsed(now=21.0) == 1.0


def test_rate_tracker_returns_rolling_counter_rate() -> None:
    tracker = RateTracker(window_seconds=10.0)
    assert tracker.update(0.0, now=0.0) is None
    assert tracker.update(20.0, now=2.0) == 10.0
    assert tracker.update(40.0, now=12.0) == 2.0


def test_episode_window_keeps_bounded_recent_values_and_total_count() -> None:
    window = EpisodeWindow(max_episodes=2)
    window.add([1.0, 3.0], [10.0, 30.0])
    window.add([5.0], [50.0])
    stats = window.snapshot()
    assert stats.mean_return == 4.0
    assert stats.mean_length == 40.0
    assert stats.episodes == 3


def test_timing_window_reports_mean_milliseconds_and_resets() -> None:
    window = TimingWindow()
    window.add("learn", 0.1)
    window.add("learn", 0.3)
    assert window.mean_ms("learn") == 200.0
    window.reset()
    assert window.mean_ms("learn") == 0.0


def test_scalar_window_drains_means_and_ranges() -> None:
    window = ScalarWindow()
    window.add({"loss": 1.0}, track_range=True)
    window.add({"loss": 3.0}, track_range=True)
    assert window.drain() == {"loss": 2.0, "loss_min": 1.0, "loss_max": 3.0}
    assert window.drain() == {}
