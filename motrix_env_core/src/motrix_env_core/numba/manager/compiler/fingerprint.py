# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Plan-key helpers for the manager kernel compiler."""

import hashlib
from typing import Any


def plan_key(parts: list[Any]) -> str:
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def type_name(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


__all__ = ["plan_key", "type_name"]
