# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reference-command state shared by real SuperMimic tasks and replay capture."""

from dataclasses import dataclass

import torch

from motrix_env_core.manager.commands import CommandTerm


@dataclass(frozen=True)
class ReferenceCommand(CommandTerm):
    """Task-owned reference timeline with one int64 frame index per environment.

    The command's lifecycle hooks own progression and reset. MbrlEnv snapshots
    ``frame`` before action application; imagination advances the saved indices
    independently of the live command.
    """

    frame: torch.Tensor
