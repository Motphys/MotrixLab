# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.numba.threading import preload_tbb


def test_preload_tbb_returns_bool_without_raising() -> None:
    # Whether the library exists depends on the tbb extra; the contract is
    # only that discovery is side-effect-safe and reports success as bool.
    assert isinstance(preload_tbb(), bool)
