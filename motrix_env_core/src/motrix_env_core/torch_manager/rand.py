# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Torch SplitMix64 primitives for manager environments.

The batched tensor implementation lives on the core
:class:`~motrix_env_core.manager.rand.RandValue` so terms and the torch
executor share one stream definition; this module keeps the historical import
path for the torch executor surface.
"""

from motrix_env_core.manager.rand import draw_uniform

__all__ = ["draw_uniform"]
