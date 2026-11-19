# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import torch


class EvaluationOutputs(NamedTuple):
    """Preallocated transition tensors written by the evaluate kernel."""

    reward: torch.Tensor
    terminated: torch.Tensor


__all__ = ["EvaluationOutputs"]
