# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Optional C++ Unitree backend bridge.

This module is intentionally small: the native extension owns DDS and the
1 kHz command writer, while this adapter preserves the Motrix RobotInterface
contract.  The extension is imported lazily so simulation and Python-SDK
diagnostic paths do not require a C++ SDK installation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol


class NativeBackend(Protocol):
    def open(self) -> None: ...

    def close(self) -> None: ...


def load_native_module() -> Any:
    """Load the separately built C++ extension only when hardware is selected."""
    try:
        return import_module("motrix_unitree_native")
    except ImportError as error:
        raise RuntimeError(
            "The native Unitree backend requires the motrix_unitree_native extension. "
            "Build it against Unitree SDK2 or select transport=python_sdk for diagnostics."
        ) from error


__all__ = ["NativeBackend", "load_native_module"]
