# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC: an in-tree port of holosoma's FastSAC distributional SAC."""

from motrix_rl.fastsac import framework as _framework
from motrix_rl.fastsac.config import (
    FastSacAgentCfg,
    FastSacAsyncOptionsCfg,
    FastSacCfg,
    FastSacTrainerCfg,
)

_framework.register_framework()

__all__ = ["FastSacCfg", "FastSacAgentCfg", "FastSacTrainerCfg", "FastSacAsyncOptionsCfg"]
