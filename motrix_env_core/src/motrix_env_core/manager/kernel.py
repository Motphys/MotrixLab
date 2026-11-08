# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import torch


class KernelOutputs(NamedTuple):
    """Preallocated tensors written by one manager task kernel."""

    policy_obs: torch.Tensor
    value_obs: torch.Tensor | None
    reward: torch.Tensor
    terminated: torch.Tensor


__all__ = ["KernelOutputs"]
