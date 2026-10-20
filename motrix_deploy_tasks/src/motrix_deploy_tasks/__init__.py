# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Concrete deployment tasks discovered through installed package entry points."""

from importlib import metadata

__version__ = metadata.version("motrix-deploy-tasks")
