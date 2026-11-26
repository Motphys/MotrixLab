# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Common input-device, command, and binding contract tests."""

import numpy as np
import pytest
import torch

from motrix_env_core.input.bindings import (
    BoundedGamePadPlanarVelocityBinding,
    ConstantPlanarVelocityBinding,
    GamePadDevice,
    GamePadPlanarVelocityBinding,
    KeyboardDevice,
    KeyboardPlanarVelocityBinding,
    PlanarVelocityCommand,
    RateLimitedPlanarVelocityBinding,
    ScalableKeyboardPlanarVelocityBinding,
)


class _Keyboard(KeyboardDevice):
    def __init__(self, pressing: set[str], down: set[str] | None = None) -> None:
        self.pressing = pressing
        self.down = down if down is not None else frozenset()
        self.poll_count = 0
        self.frame_down = frozenset()
        self.frame_pressing = frozenset()

    def poll(self) -> None:
        self.poll_count += 1
        # Match focused viewer devices: an edge is visible only during the
        # frame in which it is frozen.
        self.frame_down = self.down
        self.frame_pressing = self.pressing

    def is_key_down(self, key: str) -> bool:
        return key in self.frame_down

    def is_key_up(self, key: str) -> bool:
        del key
        return False

    def is_pressing(self, key: str) -> bool:
        return key in self.frame_pressing


class _GamePad(GamePadDevice):
    def __init__(self, axes: dict[str, float], buttons: set[str] | None = None) -> None:
        self.axes = axes
        self.buttons = buttons or set()
        self.poll_count = 0

    def poll(self) -> None:
        self.poll_count += 1

    def axis_value(self, axis: str) -> float:
        return self.axes[axis]

    def is_button_down(self, button: str) -> bool:
        del button
        return False

    def is_button_up(self, button: str) -> bool:
        del button
        return False

    def is_button_pressing(self, button: str) -> bool:
        return button in self.buttons


def test_planar_velocity_command_is_batch_first_copied_and_float32() -> None:
    source = np.array([[0.5, -0.2, 0.3], [1.0, 0.0, -0.5]], dtype=np.float64)
    command = PlanarVelocityCommand(source)
    source[:] = 9.0

    assert command.batch_size == 2
    assert command.values.dtype == torch.float32
    # Owns a private copy: mutating the source leaves the command unchanged.
    np.testing.assert_array_equal(command.linear_velocity_x_mps, [0.5, 1.0])
    np.testing.assert_allclose(command.linear_velocity_y_mps, [-0.2, 0.0])
    np.testing.assert_allclose(command.yaw_rate_rad_s, [0.3, -0.5])


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(3),
        np.zeros((0, 3)),
        np.zeros((2, 2)),
        np.array([[np.nan, 0.0, 0.0]]),
        np.array([[0.0, np.inf, 0.0]]),
    ],
)
def test_planar_velocity_command_rejects_invalid_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        PlanarVelocityCommand(values)


def test_keyboard_binding_polls_once_and_replicates_held_key_mapping() -> None:
    device = _Keyboard({"w", "a", "e"})
    binding = KeyboardPlanarVelocityBinding(
        device,
        command_lower=[-0.5, -0.4, -0.6],
        command_upper=[0.8, 0.4, 0.6],
    )

    command = binding.read_command(batch_size=2)

    assert device.poll_count == 1
    np.testing.assert_allclose(command.values, [[0.8, 0.4, -0.6], [0.8, 0.4, -0.6]])


def test_keyboard_binding_uses_asymmetric_lower_command_bounds() -> None:
    binding = KeyboardPlanarVelocityBinding(
        _Keyboard({"s", "d", "q"}),
        command_lower=[-0.5, -0.4, -1.0],
        command_upper=[0.8, 0.4, 1.0],
    )

    np.testing.assert_array_equal(
        binding.read_command().values,
        np.array([[-0.5, -0.4, 1.0]], dtype=np.float32),
    )


def test_keyboard_binding_opposite_keys_cancel() -> None:
    device = _Keyboard({"w", "s", "a", "d", "q", "e"})
    binding = KeyboardPlanarVelocityBinding(
        device,
        command_lower=[-0.5, -0.4, -0.6],
        command_upper=[0.8, 0.4, 0.6],
    )

    np.testing.assert_array_equal(binding.read_command().values, np.zeros((1, 3), dtype=np.float32))


def test_scalable_keyboard_binding_scales_all_planar_axes() -> None:
    device = _Keyboard({"w", "a", "q"})
    binding = ScalableKeyboardPlanarVelocityBinding(
        device,
        command_lower=[-1.0, -0.5, -0.5],
        command_upper=[1.0, 0.5, 0.5],
        levels=(0.2, 0.4, 0.8),
    )

    np.testing.assert_array_equal(binding.speed_levels, (0.2, 0.4, 0.8))
    np.testing.assert_array_equal(binding.axis_speed_level_indexes, (2, 2, 2))
    np.testing.assert_allclose(binding.read_command(batch_size=2).values, [[0.8, 0.4, 0.4]] * 2)

    device.down = {"down"}
    binding.read_command()
    device.down = set()
    device.pressing = {"w"}
    np.testing.assert_array_equal(binding.axis_speed_level_indexes, (1, 1, 1))
    np.testing.assert_allclose(binding.read_command().values, [[0.4, 0.0, 0.0]])

    device.down = {"up"}
    binding.read_command()
    device.down = set()
    binding.read_command()
    np.testing.assert_array_equal(binding.axis_speed_level_indexes, (2, 2, 2))

    device.down = {"down", "down"}
    binding.read_command()
    device.down = set()
    binding.read_command()
    device.down = {"down"}
    binding.read_command()
    np.testing.assert_array_equal(binding.axis_speed_level_indexes, (0, 0, 0))


def test_scalable_keyboard_binding_adjusts_each_axis_independently() -> None:
    device = _Keyboard({"w", "a", "q"})
    binding = ScalableKeyboardPlanarVelocityBinding(
        device,
        command_lower=[-1.0, -0.5, -0.5],
        command_upper=[1.0, 0.5, 0.5],
        levels=(0.2, 0.4, 0.8),
    )

    device.down = {"2"}
    binding.read_command()
    device.down = {"4"}
    binding.read_command()
    device.down = {"6"}
    binding.read_command()
    device.down = set()

    np.testing.assert_array_equal(binding.axis_speed_scales, (0.4, 0.4, 0.4))
    device.pressing = {"w", "a", "q"}
    np.testing.assert_allclose(binding.read_command().values, [[0.4, 0.2, 0.2]])

    device.down = {"3"}
    binding.read_command()
    device.down = {"5"}
    binding.read_command()
    device.down = set()
    np.testing.assert_allclose(binding.read_command().values, [[0.4, 0.4, 0.4]])

    device.down = {"1"}
    binding.read_command()
    device.down = {"3"}
    binding.read_command()
    device.down = {"5"}
    binding.read_command()
    device.down = set()
    np.testing.assert_array_equal(binding.axis_speed_scales, (0.8, 0.8, 0.8))


@pytest.mark.parametrize(
    "levels",
    [
        (),
        (0.0, 0.5),
        (0.5, 1.5),
        (0.4, 0.4),
        (0.4, float("nan")),
    ],
)
def test_scalable_keyboard_binding_rejects_invalid_levels(levels: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        ScalableKeyboardPlanarVelocityBinding(
            _Keyboard(set()),
            command_lower=[-1.0, -0.5, -0.5],
            command_upper=[1.0, 0.5, 0.5],
            levels=levels,
        )


def test_gamepad_binding_applies_deadzone_inversion_scale_and_batch() -> None:
    device = _GamePad({"left_y": 0.5, "left_x": 0.05, "right_x": -0.25})
    binding = GamePadPlanarVelocityBinding(
        device,
        linear_x_axis="left_y",
        linear_y_axis="left_x",
        yaw_axis="right_x",
        linear_x_scale=2.0,
        linear_y_scale=3.0,
        yaw_scale=4.0,
        deadzone=0.1,
        invert_linear_x=True,
    )

    command = binding.read_command(batch_size=3)

    assert device.poll_count == 1
    np.testing.assert_array_equal(command.values, np.tile([-1.0, 0.0, -1.0], (3, 1)))


def test_bounded_gamepad_binding_maps_axes_into_artifact_range_and_deadman():
    device = _GamePad({"ly": 0.55, "lx": -0.55, "rx": 1.0}, {"L1"})
    binding = BoundedGamePadPlanarVelocityBinding(
        device,
        linear_x_axis="ly",
        linear_y_axis="lx",
        yaw_axis="rx",
        command_lower=[-0.5, -0.4, -1.0],
        command_upper=[0.5, 0.4, 1.0],
        deadzone=0.1,
        deadman_button="L1",
    )
    np.testing.assert_allclose(binding.read_command().values, [[0.25, -0.2, 1.0]], atol=1e-6)

    device.buttons.clear()
    np.testing.assert_array_equal(binding.read_command().values, np.zeros((1, 3), dtype=np.float32))


def test_gamepad_binding_does_not_revalidate_device_axis_output() -> None:
    device = _GamePad({"x": 1.5, "y": -1.25, "yaw": 0.0})
    binding = GamePadPlanarVelocityBinding(
        device,
        linear_x_axis="x",
        linear_y_axis="y",
        yaw_axis="yaw",
        linear_x_scale=2.0,
        linear_y_scale=2.0,
        yaw_scale=1.0,
    )

    np.testing.assert_array_equal(binding.read_command().values, [[3.0, -2.5, 0.0]])


def test_constant_binding_matches_keyboard_command_without_a_device_contract() -> None:
    keyboard = KeyboardPlanarVelocityBinding(
        _Keyboard({"w", "e"}),
        command_lower=[-0.5, -0.4, -1.0],
        command_upper=[0.5, 0.4, 1.0],
    )
    constant = ConstantPlanarVelocityBinding([0.5, 0.0, -1.0])

    np.testing.assert_array_equal(
        constant.read_command(batch_size=4).values,
        keyboard.read_command(batch_size=4).values,
    )


def test_rate_limited_binding_ramps_each_axis_toward_target() -> None:
    keyboard = _Keyboard({"w"})
    binding = RateLimitedPlanarVelocityBinding(
        KeyboardPlanarVelocityBinding(
            keyboard,
            command_lower=[-1.0, -0.5, -0.5],
            command_upper=[1.0, 0.5, 0.5],
        ),
        max_acceleration=[1.0, 0.25, 0.5],
        dt_s=0.02,
    )

    first = binding.read_command()
    keyboard.pressing.clear()
    second = binding.read_command()
    keyboard.pressing = {"w", "a"}
    third = binding.read_command()

    np.testing.assert_allclose(first.values, [[0.02, 0.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(second.values, [[0.0, 0.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(third.values, [[0.02, 0.005, 0.0]], atol=1e-7)


def test_rate_limited_binding_uses_separate_startup_and_stop_accelerations() -> None:
    binding = RateLimitedPlanarVelocityBinding(
        ConstantPlanarVelocityBinding([1.0, 0.0, 0.0]),
        max_acceleration=[1.2, 1.2, 1.2],
        startup_acceleration=[1.2, 1.2, 1.2],
        stop_acceleration=[2.0, 2.0, 2.0],
        dt_s=0.02,
    )

    startup = binding.read_command()
    binding._binding._value = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    stop = binding.read_command()

    np.testing.assert_allclose(startup.values, [[0.024, 0.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(stop.values, [[0.0, 0.0, 0.0]], atol=1e-7)


def test_rate_limited_binding_brakes_through_zero_on_reversal() -> None:
    binding = RateLimitedPlanarVelocityBinding(
        ConstantPlanarVelocityBinding([-1.0, 0.0, 0.0]),
        max_acceleration=[1.2, 1.2, 1.2],
        startup_acceleration=[1.2, 1.2, 1.2],
        stop_acceleration=[2.0, 2.0, 2.0],
        dt_s=0.02,
    )

    first = binding.read_command()
    binding._binding._value = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    reversal = binding.read_command()

    np.testing.assert_allclose(first.values, [[-0.024, 0.0, 0.0]], atol=1e-7)
    np.testing.assert_allclose(reversal.values, [[0.016, 0.0, 0.0]], atol=1e-7)


@pytest.mark.parametrize(
    ("max_acceleration", "dt_s"),
    [
        ([1.0, 1.0], 0.02),
        ([1.0, 1.0, float("nan")], 0.02),
        ([1.0, 0.0, 1.0], 0.02),
        ([1.0, 1.0, 1.0], 0.0),
        ([1.0, 1.0, 1.0], float("nan")),
    ],
)
def test_rate_limited_binding_rejects_invalid_limits(max_acceleration: object, dt_s: object) -> None:
    with pytest.raises(ValueError):
        RateLimitedPlanarVelocityBinding(
            ConstantPlanarVelocityBinding([0.0, 0.0, 0.0]),
            max_acceleration=max_acceleration,
            dt_s=dt_s,
        )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_bindings_reject_invalid_batch_size(batch_size: object) -> None:
    binding = ConstantPlanarVelocityBinding([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="batch_size"):
        binding.read_command(batch_size=batch_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [[1.0, 2.0], [1.0, 2.0, np.nan]])
def test_constant_binding_rejects_invalid_vectors(value: list[float]) -> None:
    with pytest.raises(ValueError):
        ConstantPlanarVelocityBinding(value).read_command()
