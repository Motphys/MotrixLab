# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public API for manager-based environments and terms."""

from numba import njit

from motrix_env_core.config.sim_reset import ManagerResetCfg, ResetTermCfg
from motrix_env_core.numba.kernel import ManagerWarmupResult
from motrix_env_core.numba.kernel_data import SharedArray, kernel_data
from motrix_env_core.numba.manager.actions import (
    ActionCfg,
    ActionTerm,
    ManagerActionsCfg,
)
from motrix_env_core.numba.manager.commands import CommandCfg, CommandTerm, ManagerCommandsCfg
from motrix_env_core.numba.manager.context import ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.env import ManagerBasedEnvCfg, ManagerEnv
from motrix_env_core.numba.manager.metrics import metric
from motrix_env_core.numba.manager.observations import (
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.numba.manager.rewards import ManagerRewardsCfg, RewardTerm, RewardTermCfg
from motrix_env_core.numba.manager.sim_reset import ResetTerm
from motrix_env_core.numba.manager.terminations import (
    ManagerTerminationsCfg,
    TerminationManager,
    TerminationTerm,
    TerminationTermCfg,
)
from motrix_env_core.numba.manager.terms import BaseTerm
from motrix_env_core.sim import SimQueriesCfg

__all__ = [
    "ActionCfg",
    "ActionTerm",
    "CommandCfg",
    "CommandTerm",
    "ManagerActionsCfg",
    "ManagerBasedEnvCfg",
    "ManagerCommandsCfg",
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
    "SharedArray",
    "TerminationManager",
    "TerminationTermCfg",
    "TerminationTerm",
    "kernel_data",
    "metric",
    "njit",
]
