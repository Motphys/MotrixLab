# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-provided, read-only numerical terms for real MBRL transitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext, ManagerContext


class TransitionTerms(ABC):
    """Pure batched dispatches; write only supplied outputs, never advance task state.

    Context is read before action application. World-state terms independently
    capture physical endpoints; MbrlEnv captures action history and the reference
    frame directly from their owning terms.
    Queries must be declared by the environment configuration.
    """

    @abstractmethod
    def context(self, ctx: ManagerContext, out: torch.Tensor) -> None:
        """Write the task context."""


@configclass
class TransitionTermsCfg(ABC):
    @abstractmethod
    def __call__(self, ctx: BuildContext) -> TransitionTerms:
        """Build environment-local transition dispatches."""
