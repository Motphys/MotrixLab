# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Default Torch device resolution shared by the RL provider plugins.

CUDA first (identical to the historical behavior), then MPS on Apple
Silicon, then CPU. Explicitly configured devices (``rl_cfg.device``,
``collector_inference_device``) never pass through here.
"""

from __future__ import annotations

import torch


def resolve_default_device() -> torch.device:
    """The trainer device when the configuration leaves it unset."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
