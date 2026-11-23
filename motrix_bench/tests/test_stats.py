# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from motrix_bench import percentile, percentiles


def test_percentile_uses_linear_interpolation():
    assert percentile([1, 4, 9, 16], 0.5) == pytest.approx(6.5)
    assert percentiles([16, 1, 9, 4], 0.0, 0.5, 1.0) == (1.0, 6.5, 16.0)


def test_percentile_rejects_empty_or_invalid_quantile():
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between"):
        percentile([1], 1.1)
