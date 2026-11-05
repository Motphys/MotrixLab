# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from motrix_env_core.array_api import Array


def test_array_wrapper_preserves_numpy_storage_and_operators() -> None:
    source = np.asarray([1.0, 2.0], dtype=np.float32)
    value = Array(source)
    result = (value + 1.0) * 2.0

    assert value.native is source
    np.testing.assert_allclose(result.native, [4.0, 6.0])


def test_array_wrapper_preserves_torch_storage_and_operators() -> None:
    source = torch.tensor([1.0, 2.0])
    value = Array(source)
    result = (value + 1.0) * 2.0

    assert value.native is source
    torch.testing.assert_close(result.native, torch.tensor([4.0, 6.0]))
