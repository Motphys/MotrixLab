# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Type-specialized namespace selection for compiled batch computations."""

import array_api_compat.numpy as numpy_namespace
import array_api_compat.torch as torch_namespace
import torch


def array_namespace(value, *others):
    # Dynamo specializes this type check; no runtime backend switch in kernels.
    return torch_namespace if isinstance(value, torch.Tensor) else numpy_namespace
