# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reward shaping helpers for direct-workflow environments.

The direct frontend is torch-native; the implementation lives in
:mod:`motrix_env_core.torch.reward` and is re-exported here as the direct
task-authoring surface.
"""

from motrix_env_core.torch.reward import tolerance

__all__ = ["tolerance"]
