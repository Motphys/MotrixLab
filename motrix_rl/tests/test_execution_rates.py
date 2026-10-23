# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl.execution.rates import RollingRate


def test_rolling_rate_is_none_until_time_span_exists() -> None:
    rate = RollingRate()

    assert rate.sample(10.0, 100.0) is None
    assert rate.sample(10.0, 200.0) is None


def test_rolling_rate_reports_stall_as_zero() -> None:
    rate = RollingRate()

    rate.sample(0.0, 12.0)
    assert rate.sample(2.0, 12.0) == 0.0


def test_rolling_rate_evicts_samples_outside_configured_window() -> None:
    rate = RollingRate(window_seconds=10.0)

    rate.sample(0.0, 0.0)
    rate.sample(5.0, 10.0)
    # The t=0 sample is outside the inclusive 10-second window.
    assert rate.sample(15.0, 30.0) == 2.0


def test_rolling_rate_leaves_scale_to_caller() -> None:
    rate = RollingRate(window_seconds=4.0)

    rate.sample(2.0, 100.0)
    assert rate.sample(4.0, 104.0) == 2.0


def test_rolling_rate_can_reset() -> None:
    rate = RollingRate()
    rate.sample(0.0, 1.0)
    rate.reset()

    assert rate.sample(20.0, 50.0) is None
