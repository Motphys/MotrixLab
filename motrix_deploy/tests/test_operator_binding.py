# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from motrix_deploy.operator import (
    ButtonGestureCfg,
    OperatorCommandBinding,
    OperatorProfileCfg,
    PlanarVelocityOperatorCfg,
)
from motrix_env_core.input import GamePadDevice, KeyboardDevice


class _Keyboard(KeyboardDevice):
    def __init__(self, frames):
        self.frames = iter(frames)
        self.current = frozenset()
        self.previous = frozenset()

    def poll(self):
        self.previous = self.current
        self.current = frozenset(next(self.frames))

    def is_key_down(self, key):
        return key in self.current and key not in self.previous

    def is_key_up(self, key):
        return key not in self.current and key in self.previous

    def is_pressing(self, key):
        return key in self.current


class _GamePad(GamePadDevice):
    def __init__(self, frames):
        self.frames = iter(frames)
        self.current = ({}, frozenset())
        self.previous = self.current

    def poll(self):
        self.previous = self.current
        self.current = next(self.frames)

    def axis_value(self, axis):
        return self.current[0].get(axis, 0.0)

    def is_button_down(self, button):
        return button in self.current[1] and button not in self.previous[1]

    def is_button_up(self, button):
        return button not in self.current[1] and button in self.previous[1]

    def is_button_pressing(self, button):
        return button in self.current[1]


def test_keyboard_binding_produces_intent_and_velocity_from_one_polled_frame():
    device = _Keyboard([set(), {"1", "w"}])
    profile = OperatorProfileCfg(intents={"request_walk": ButtonGestureCfg(pressed="1")})
    binding = OperatorCommandBinding(device, profile, command_lower=(-1, -2, -3), command_upper=(1, 2, 3))
    assert not binding.read_command().intents
    command = binding.read_command()
    assert command.intents == frozenset({"request_walk"})
    np.testing.assert_array_equal(command.planar_velocity.values, [[1, 0, 0]])


def test_gamepad_binding_applies_deadzone_inversion_range_and_gesture():
    device = _GamePad(
        [
            ({}, frozenset()),
            ({"ly": 1.0, "lx": 0.5, "rx": -1.0}, frozenset({"RB", "X"})),
        ]
    )
    profile = OperatorProfileCfg(
        intents={"request_walk": ButtonGestureCfg(held=["RB"], pressed="X")},
        planar_velocity=PlanarVelocityOperatorCfg(
            linear_x_axis="ly",
            linear_y_axis="lx",
            yaw_axis="rx",
            invert_linear_y=True,
            invert_yaw=True,
            deadzone=0.0,
            range_scale=[0.25, 0.5, 1.0],
        ),
    )
    binding = OperatorCommandBinding(device, profile, command_lower=(-1, -2, -3), command_upper=(1, 2, 3))
    binding.read_command()
    command = binding.read_command()
    assert command.intents == frozenset({"request_walk"})
    np.testing.assert_array_equal(command.planar_velocity.values, [[0.25, -0.5, 3.0]])
