# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Decode the Unitree wireless remote payload used in LowState."""

import struct
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from motrix_deploy.errors import ValidationError
from motrix_env_core.input import GamePadDevice

BUTTON_NAMES = (
    "R1",
    "L1",
    "start",
    "select",
    "R2",
    "L2",
    "F1",
    "F2",
    "A",
    "B",
    "X",
    "Y",
    "up",
    "right",
    "down",
    "left",
)
BUTTON_ALIASES = {"RB": "R1", "LB": "L1", "RT": "R2", "LT": "L2", "back": "select"}
ACCEPTED_BUTTON_NAMES = BUTTON_NAMES + tuple(BUTTON_ALIASES)


def canonical_button_name(name: str) -> str:
    """Resolve SDK-wrapper aliases without changing the stable wire bit names."""
    return BUTTON_ALIASES.get(name, name)


@dataclass(frozen=True)
class UnitreeRemoteState:
    """Decoded button bitset and joystick axes."""

    keys: int = 0
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0

    def pressed(self, name: str) -> bool:
        name = canonical_button_name(name)
        try:
            index = BUTTON_NAMES.index(name)
        except ValueError as error:
            raise ValidationError("remote.button", f"one of {list(BUTTON_NAMES)}", name) from error
        return bool(self.keys & (1 << index))

    def pressed_buttons(self) -> frozenset[str]:
        """Return active canonical names plus their accepted SDK aliases."""
        active = {name for name in BUTTON_NAMES if self.pressed(name)}
        active.update(alias for alias, canonical in BUTTON_ALIASES.items() if canonical in active)
        return frozenset(active)


class UnitreeRemoteGamePadDevice(GamePadDevice):
    """Expose the latest Unitree remote sample through the common gamepad API."""

    _AXES = {"lx", "ly", "rx", "ry"}

    def __init__(self, state_provider: Callable[[], UnitreeRemoteState]) -> None:
        self._state_provider = state_provider
        self._current = UnitreeRemoteState()
        self._previous = UnitreeRemoteState()

    def poll(self) -> None:
        self._previous = self._current
        self._current = self._state_provider()

    def axis_value(self, axis: str) -> float:
        if axis not in self._AXES:
            raise ValidationError("remote.axis", f"one of {sorted(self._AXES)}", axis)
        return float(np.clip(getattr(self._current, axis), -1.0, 1.0))

    def is_button_down(self, button: str) -> bool:
        return self._current.pressed(button) and not self._previous.pressed(button)

    def is_button_up(self, button: str) -> bool:
        return not self._current.pressed(button) and self._previous.pressed(button)

    def is_button_pressing(self, button: str) -> bool:
        return self._current.pressed(button)


def decode_wireless_remote(data: object) -> UnitreeRemoteState:
    """Decode Unitree's 40-byte remote packet using explicit little-endian fields."""
    try:
        payload = bytes(data)
    except (TypeError, ValueError) as error:
        raise ValidationError("state.wireless_remote", "a byte sequence", type(data).__name__) from error
    if len(payload) < 24:
        raise ValidationError("state.wireless_remote", "at least 24 bytes", len(payload))
    keys = struct.unpack_from("<H", payload, 2)[0]
    lx = struct.unpack_from("<f", payload, 4)[0]
    rx = struct.unpack_from("<f", payload, 8)[0]
    ry = struct.unpack_from("<f", payload, 12)[0]
    ly = struct.unpack_from("<f", payload, 20)[0]
    axes = np.asarray((lx, ly, rx, ry), dtype=np.float32)
    if not np.all(np.isfinite(axes)):
        raise ValidationError("state.wireless_remote.axes", "finite joystick values", axes.tolist())
    return UnitreeRemoteState(keys=keys, lx=float(lx), ly=float(ly), rx=float(rx), ry=float(ry))


__all__ = [
    "ACCEPTED_BUTTON_NAMES",
    "BUTTON_ALIASES",
    "BUTTON_NAMES",
    "UnitreeRemoteGamePadDevice",
    "UnitreeRemoteState",
    "canonical_button_name",
    "decode_wireless_remote",
]
