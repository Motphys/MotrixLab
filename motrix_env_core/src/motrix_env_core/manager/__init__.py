# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public API for manager-based environments and terms."""

from motrix_env_core.config.sim_reset import ManagerResetCfg, ResetTermCfg
from motrix_env_core.manager.actions import (
    ActionCfg,
    ActionState,
    ActionTerm,
    ManagerActionsCfg,
)
from motrix_env_core.manager.commands import CommandCfg, CommandTerm, ManagerCommandsCfg
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.dispatch import dispatch
from motrix_env_core.manager.env import ManagerBasedEnvCfg, ManagerEnv
from motrix_env_core.manager.kernel import ManagerWarmupResult
from motrix_env_core.manager.kernel_data import kernel_data
from motrix_env_core.manager.metrics import metric
from motrix_env_core.manager.observations import (
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.manager.rewards import ManagerRewardsCfg, RewardTerm, RewardTermCfg
from motrix_env_core.manager.sim_reset import ResetTerm
from motrix_env_core.manager.terminations import (
    ManagerTerminationsCfg,
    TerminationManager,
    TerminationTerm,
    TerminationTermCfg,
)
from motrix_env_core.manager.terms import BaseTerm
from motrix_env_core.sim import SimQueriesCfg

__all__ = [
    "ActionCfg",
    "ActionTerm",
    "ActionState",
    "CommandCfg",
    "CommandTerm",
    "ManagerActionsCfg",
    "ManagerBasedEnvCfg",
    "ManagerCommandsCfg",
    "BuildContext",
    "ManagerContext",
    "dispatch",
    "ManagerEnv",
    "ManagerObservationGroupCfg",
    "ManagerObservationsCfg",
    "ManagerRewardsCfg",
    "ManagerTerminationsCfg",
    "SimQueriesCfg",
    "ManagerWarmupResult",
    "ObsTerm",
    "ObservationTermCfg",
    "BaseTerm",
    "RewardTerm",
    "RewardTermCfg",
    "ManagerResetCfg",
    "ResetTerm",
    "ResetTermCfg",
    "TerminationManager",
    "TerminationTermCfg",
    "TerminationTerm",
    "kernel_data",
    "metric",
]
