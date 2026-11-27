# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Assemble Motrix-owned algorithm providers under one framework namespace."""

from motrix_rl_supermimic.provider import SuperMimicProvider

from motrix_rl.contracts import RlFramework
from motrix_rl_builtin.fastsac.provider import FastSacProvider


class MotrixFramework(RlFramework):
    def __init__(self) -> None:
        super().__init__((FastSacProvider(), SuperMimicProvider()))

    @property
    def name(self) -> str:
        return "motrix"
