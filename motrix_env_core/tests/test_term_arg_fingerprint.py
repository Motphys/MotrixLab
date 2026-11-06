# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Carrier dataclass term-argument fingerprint contract.

Carrier dataclasses passed as manager term arguments resolve as constants;
their structural fingerprint must keep compiled-program caches invalidated by
any content change (tensor bytes or scalar values).
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from motrix_env_core.manager.resolver import _carrier_fingerprint


@dataclass(frozen=True, slots=True)
class _Nested:
    flag: bool


@dataclass(frozen=True, slots=True)
class _Carrier:
    grid: torch.Tensor
    label: str
    offset: float
    nested: _Nested


def test_fingerprint_changes_on_tensor_content_and_scalar() -> None:
    carrier = _Carrier(torch.arange(6, dtype=torch.float32).reshape(2, 3), "a", 1.5, _Nested(True))
    base = _carrier_fingerprint(carrier)
    edited_grid = _Carrier(carrier.grid.clone(), "a", 1.5, _Nested(True))
    edited_grid.grid[0, 0] += 1.0
    assert _carrier_fingerprint(edited_grid) != base
    assert _carrier_fingerprint(_Carrier(carrier.grid, "b", 1.5, _Nested(True))) != base
    assert _carrier_fingerprint(_Carrier(carrier.grid, "a", 2.0, _Nested(True))) != base
    assert _carrier_fingerprint(_Carrier(carrier.grid, "a", 1.5, _Nested(False))) != base


def test_fingerprint_stable_for_equal_content() -> None:
    first = _Carrier(torch.ones(3), "a", 1.5, _Nested(True))
    second = _Carrier(torch.ones(3), "a", 1.5, _Nested(True))
    assert _carrier_fingerprint(first) == _carrier_fingerprint(second)


def test_fingerprint_rejects_unsupported_field_type() -> None:
    @dataclass(frozen=True, slots=True)
    class _Bad:
        leak: np.float32

    with pytest.raises(TypeError, match="unsupported type"):
        _carrier_fingerprint(_Bad(np.float32(1.0)))
