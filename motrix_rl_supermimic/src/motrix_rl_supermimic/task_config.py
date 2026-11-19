# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Explicit simulator-free MBRL declarations; no simulator dependency inference."""

from __future__ import annotations

from typing import Protocol

import torch

from motrix_env_core.config import configclass
from motrix_rl_supermimic.world_state import WorldStateTermCfg


@configclass
class WorldModelStateCfg:
    """Composable predicted-state terms and independent dynamics conditioning."""

    terms: dict[str, WorldStateTermCfg]
    context_dim: int = 0


class TaskMirror(Protocol):
    """Task symmetry used together for world-model window augmentation."""

    def state(self, state: torch.Tensor) -> torch.Tensor: ...

    def action(self, action: torch.Tensor) -> torch.Tensor: ...

    def context(self, context: torch.Tensor) -> torch.Tensor: ...
