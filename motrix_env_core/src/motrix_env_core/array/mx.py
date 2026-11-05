# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared MDP primitive namespace.

The functions in this module are the only array operations that shared MDP
code should call when it must run eagerly on NumPy and Torch values. Compiler
lowerings may replace these calls in compiled regions; this module itself is
kept small and transparent for host-side tests.
"""

from motrix_env_core.array.primitives import clip, exp, sum, where, zeros_like

__all__ = ["clip", "exp", "sum", "where", "zeros_like"]
