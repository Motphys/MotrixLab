# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned reference and predicted-input binding for model imagination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_rl_supermimic.task_config import TaskMirror


@dataclass(kw_only=True)
class ImaginationTerm(ABC):
    """Simulator-free task behavior and evaluation data owned by one build.

    This is not a scheduled Manager numerical term: it binds predicted inputs
    and reference frames without running real-environment command lifecycle.
    """

    mirror: TaskMirror | None = None
    evaluation_metadata: dict[str, str | int | float] = field(default_factory=dict)
    return_normalizer: float | None = None

    @abstractmethod
    def bind_context(
        self,
        frame: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        previous: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> ManagerContext:
        """Bind original batch-shaped inputs to one flat numerical context.

        Implementations flatten all leading batch dimensions into N rows without
        detaching predicted tensors. With state supplied, action, previous and
        context must all be explicit, including zero-width context tensors. With
        only frame supplied, bind reference data without physical inputs so goals
        can be evaluated independently of predicted state. Validate missing inputs
        here rather than manufacturing physical values in numerical heads.
        """
        raise NotImplementedError

    @abstractmethod
    def valid_starts(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return which reference frames support the requested imagined horizon."""
        raise NotImplementedError

    def reference_action(self) -> nn.Module | None:
        """Construct the optional reference policy within the learner seed scope."""
        return None


@configclass
class ImaginationCfg(ABC):
    @abstractmethod
    def __call__(self, ctx: BuildContext) -> ImaginationTerm:
        raise NotImplementedError
