# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import numba
import numpy as np
import pytest

from motrix_env_core.numba.kernel import clone_kernel_value


@pytest.mark.parametrize("layout", ["c", "f", "strided", "reversed", "broadcast", "empty", "empty_strided"])
@pytest.mark.parametrize("readonly", [False, True])
def test_kernel_clone_preserves_numba_array_type_and_isolates_storage(layout, readonly):
    source = np.arange(48, dtype=np.float32).reshape(8, 6)
    if layout == "f":
        source = np.asfortranarray(source)
    elif layout == "strided":
        source = source[::2, ::2]
    elif layout == "reversed":
        source = source[::-1, ::-2]
    elif layout == "broadcast":
        source = np.broadcast_to(source[:1], (8, 6))
    elif layout == "empty":
        source = source[:0]
    elif layout == "empty_strided":
        source = np.ndarray((0, 2), dtype=np.float32, buffer=bytearray(4), strides=(0, 2**62))
    source.flags.writeable = not readonly
    copied = clone_kernel_value(source)

    assert numba.typeof(copied) == numba.typeof(source)
    assert copied.strides == source.strides
    assert not np.shares_memory(copied, source)
    np.testing.assert_array_equal(copied, source)
    if not readonly:
        copied[...] = -1
        assert np.all(source >= 0)


def test_kernel_clone_preserves_nested_namedtuple_and_scalar_values():
    outputs = namedtuple("Outputs", ["value", "count"])
    source = (outputs(np.arange(4, dtype=np.float32), 3), np.float32(0.5))
    copied = clone_kernel_value(source)
    assert type(copied[0]) is outputs
    assert copied[0].count == 3
    assert copied[1] == np.float32(0.5)
    assert not np.shares_memory(source[0].value, copied[0].value)
