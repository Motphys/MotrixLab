# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Focused GLFW keyboard and minimal MuJoCo viewer tests."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from motrix_deploy.runtime.scheduler import FixedStepScheduler
from motrix_deploy_mujoco.viewer import MujocoGlfwViewer
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.input import KeyboardPlanarVelocityBinding


class _Window:
    should_close = False


class _FakeGlfw:
    TRUE = 1
    VISIBLE = 0x00020004
    PRESS = 1
    RELEASE = 0
    REPEAT = 2
    KEY_SPACE = 32
    KEY_0 = 48
    KEY_9 = 57
    KEY_A = 65
    KEY_Z = 90
    KEY_ESCAPE = 256
    KEY_RIGHT = 262
    KEY_LEFT = 263
    KEY_DOWN = 264
    KEY_UP = 265
    KEY_LEFT_SHIFT = 340
    KEY_RIGHT_SHIFT = 344
    MOUSE_BUTTON_LEFT = 0
    MOUSE_BUTTON_RIGHT = 1
    MOUSE_BUTTON_MIDDLE = 2

    def __init__(self) -> None:
        self.window = _Window()
        self.callbacks: dict[str, Any] = {}
        self.mouse_buttons: dict[int, int] = {}
        self.keys: dict[int, int] = {}
        self.swap_count = 0
        self.poll_count = 0
        self.destroy_count = 0
        self.terminate_count = 0
        self.events: list[Any] = []

    def init(self) -> bool:
        return True

    def window_hint(self, hint: int, value: int) -> None:
        del hint, value

    def create_window(self, width: int, height: int, title: str, monitor: Any, share: Any) -> _Window:
        del width, height, title, monitor, share
        return self.window

    def make_context_current(self, window: Any) -> None:
        del window

    def swap_interval(self, interval: int) -> None:
        assert interval == 0

    def get_cursor_pos(self, window: _Window) -> tuple[float, float]:
        del window
        return (10.0, 20.0)

    def set_key_callback(self, window: _Window, callback: Any) -> None:
        del window
        self.callbacks["key"] = callback

    def set_window_focus_callback(self, window: _Window, callback: Any) -> None:
        del window
        self.callbacks["focus"] = callback

    def set_window_close_callback(self, window: _Window, callback: Any) -> None:
        del window
        self.callbacks["close"] = callback

    def set_cursor_pos_callback(self, window: _Window, callback: Any) -> None:
        del window
        self.callbacks["cursor"] = callback

    def set_scroll_callback(self, window: _Window, callback: Any) -> None:
        del window
        self.callbacks["scroll"] = callback

    def poll_events(self) -> None:
        self.poll_count += 1
        events, self.events = self.events, []
        for event in events:
            event()

    def window_should_close(self, window: _Window) -> bool:
        return window.should_close

    def get_framebuffer_size(self, window: _Window) -> tuple[int, int]:
        del window
        return (1200, 900)

    def swap_buffers(self, window: _Window) -> None:
        del window
        self.swap_count += 1

    def destroy_window(self, window: _Window) -> None:
        del window
        self.destroy_count += 1

    def terminate(self) -> None:
        self.terminate_count += 1

    def get_mouse_button(self, window: _Window, button: int) -> int:
        del window
        return self.mouse_buttons.get(button, self.RELEASE)

    def get_key(self, window: _Window, key: int) -> int:
        del window
        return self.keys.get(key, self.RELEASE)

    def get_window_size(self, window: _Window) -> tuple[int, int]:
        del window
        return (1200, 900)

    def get_key_name(self, key: int, scancode: int) -> None:
        del key, scancode
        return None


class _Context:
    def __init__(self) -> None:
        self.free_count = 0

    def free(self) -> None:
        self.free_count += 1


class _FakeMujoco:
    mjtFontScale = SimpleNamespace(mjFONTSCALE_150=150)
    mjtCatBit = SimpleNamespace(mjCAT_ALL=7)
    mjtMouse = SimpleNamespace(
        mjMOUSE_ROTATE_V=1,
        mjMOUSE_ROTATE_H=2,
        mjMOUSE_MOVE_V=3,
        mjMOUSE_MOVE_H=4,
        mjMOUSE_ZOOM=5,
    )

    def __init__(self) -> None:
        self.context = _Context()
        self.update_count = 0
        self.render_count = 0
        self.camera_moves: list[tuple[Any, float, float]] = []

    def MjvCamera(self) -> object:
        return SimpleNamespace(lookat=np.zeros(3), distance=0.0, elevation=0.0, azimuth=0.0)

    def mjv_defaultFreeCamera(self, model: object, camera: object) -> None:
        del model, camera

    def MjvOption(self) -> object:
        return object()

    def MjvPerturb(self) -> object:
        return object()

    def MjvScene(self, model: object, *, maxgeom: int) -> object:
        del model
        assert maxgeom == 10_000
        return object()

    def MjrContext(self, model: object, font_scale: int) -> _Context:
        del model
        assert font_scale == 150
        return self.context

    def mjv_updateScene(self, *args: Any) -> None:
        del args
        self.update_count += 1

    def MjrRect(self, left: int, bottom: int, width: int, height: int) -> tuple[int, int, int, int]:
        return (left, bottom, width, height)

    def mjr_render(self, viewport: Any, scene: Any, context: Any) -> None:
        del viewport, scene, context
        self.render_count += 1

    def mjv_moveCamera(
        self,
        model: Any,
        action: Any,
        dx: float,
        dy: float,
        scene: Any,
        camera: Any,
    ) -> None:
        del model, scene, camera
        self.camera_moves.append((action, dx, dy))


def _viewer(
    camera_config: SystemCameraCfg = SystemCameraCfg(),
) -> tuple[MujocoGlfwViewer, _FakeGlfw, _FakeMujoco]:
    glfw = _FakeGlfw()
    mujoco = _FakeMujoco()
    viewer = MujocoGlfwViewer(mujoco, camera_config, glfw_module=glfw)
    viewer.open(object(), object())
    return viewer, glfw, mujoco


def test_glfw_keyboard_freezes_edges_and_clears_held_keys_on_focus_loss() -> None:
    viewer, glfw, _ = _viewer()
    device = viewer.keyboard_device
    binding = KeyboardPlanarVelocityBinding(
        device,
        command_lower=[-0.5, -0.4, -1.0],
        command_upper=[1.0, 0.4, 1.0],
    )
    key = glfw.callbacks["key"]

    key(glfw.window, glfw.KEY_A + ord("w") - ord("a"), 0, glfw.PRESS, 0)
    key(glfw.window, glfw.KEY_A + ord("w") - ord("a"), 0, glfw.REPEAT, 0)
    np.testing.assert_array_equal(binding.read_command().values, [[1.0, 0.0, 0.0]])
    assert device.is_key_down("w")
    assert device.is_pressing("W")
    assert not device.is_key_up("w")

    device.poll()
    assert not device.is_key_down("w")
    assert device.is_pressing("w")

    glfw.callbacks["focus"](glfw.window, 0)
    np.testing.assert_array_equal(binding.read_command().values, [[0.0, 0.0, 0.0]])
    assert device.is_key_up("w")
    assert not device.is_pressing("w")
    viewer.close()


def test_glfw_escape_and_window_close_interrupt_input() -> None:
    viewer, glfw, _ = _viewer()
    glfw.callbacks["key"](glfw.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)

    with pytest.raises(KeyboardInterrupt):
        viewer.keyboard_device.poll()

    viewer.close()
    viewer, glfw, _ = _viewer()
    glfw.callbacks["close"](glfw.window)
    with pytest.raises(KeyboardInterrupt):
        viewer.keyboard_device.poll()
    viewer.close()


def test_render_event_pump_retains_keyboard_edges_for_command_input() -> None:
    viewer, glfw, _ = _viewer()
    device = viewer.keyboard_device
    try:
        glfw.events.append(lambda: glfw.callbacks["key"](glfw.window, glfw.KEY_A, 0, glfw.PRESS, 0))
        viewer.sync()
        viewer.sync()
        # Rendering delivers callbacks but never freezes or consumes input frames.
        assert not device.is_key_down("a")
        device.poll()
        assert device.is_key_down("a") and device.is_pressing("a")
        viewer.sync()
        assert device.is_key_down("a")
        device.poll()
        assert not device.is_key_down("a") and device.is_pressing("a")
        glfw.events.append(lambda: glfw.callbacks["key"](glfw.window, glfw.KEY_A, 0, glfw.RELEASE, 0))
        viewer.sync()
        device.poll()
        assert device.is_key_up("a") and not device.is_pressing("a")
    finally:
        viewer.close()


@pytest.mark.parametrize("event", ["escape", "close"])
def test_render_event_pump_interrupts_without_keyboard_binding(event) -> None:
    viewer, glfw, mujoco = _viewer()
    try:
        if event == "escape":
            glfw.events.append(lambda: glfw.callbacks["key"](glfw.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0))
        else:
            glfw.events.append(lambda: glfw.callbacks["close"](glfw.window))
        with pytest.raises(KeyboardInterrupt):
            viewer.sync()
        assert glfw.poll_count == 1
        assert mujoco.render_count == 0
    finally:
        viewer.close()


def test_gui_runtime_pumps_startup_camera_and_close_with_constant_input(
    control_scene_config, control_gains, monkeypatch
) -> None:
    from control_helpers import DummyServoPolicyProcessor

    from motrix_deploy.policy import NoOpPolicy
    from motrix_deploy.runtime.control import ControlSession, PolicyController
    from motrix_deploy_mujoco import runtime
    from motrix_env_core.input import ConstantPlanarVelocityBinding

    glfw = _FakeGlfw()
    mujoco = _FakeMujoco()
    viewer = MujocoGlfwViewer(mujoco, glfw_module=glfw)
    from dataclasses import replace

    simulation = runtime.MujocoRuntime(replace(control_scene_config, render=True), viewer_factory=lambda *_: viewer)
    robot = simulation.robot
    controller = PolicyController(
        DummyServoPolicyProcessor(robot.spec, *control_gains), NoOpPolicy(robot.spec.joint_count), steps=10
    )
    waits = []

    class Scheduler(FixedStepScheduler):
        def __init__(self, period_s):
            super().__init__(period_s)
            assert period_s == simulation.control_period_s

        def reset(self):
            pass

        def wait(self, step):
            waits.append(step)

    monkeypatch.setattr(runtime, "RealtimeScheduler", Scheduler)
    poll = glfw.poll_events

    def deliver_events():
        if glfw.poll_count == 1:
            glfw.events.append(lambda: glfw.callbacks["scroll"](glfw.window, 0.0, 2.0))
        elif glfw.poll_count == 3:
            glfw.events.append(lambda: glfw.callbacks["close"](glfw.window))
        poll()

    monkeypatch.setattr(glfw, "poll_events", deliver_events)
    session = ControlSession(
        controller=controller,
        robot=robot,
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=0.02,
        state_timeout_s=0.1,
    )
    simulation.bind_session(session)
    with simulation:
        assert glfw.poll_count == 1 and glfw.swap_count == 1
        result = simulation.run()
        assert result.exit_reason == "interrupted" and result.completed_steps == 3
        assert result.policy_simulation_time_s == pytest.approx(0.06)
        assert waits == [1, 2]
        assert mujoco.camera_moves == [(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.1)]
    assert glfw.destroy_count == 1 and glfw.terminate_count == 1
    assert mujoco.context.free_count == 1 and not viewer.is_running()


def test_glfw_viewer_renders_and_moves_camera_without_stepping_physics() -> None:
    camera_config = SystemCameraCfg(lookat=(1.0, 2.0, 3.0), distance=4.0, elevation=-25.0, azimuth=120.0)
    viewer, glfw, mujoco = _viewer(camera_config)

    np.testing.assert_array_equal(viewer._camera.lookat, camera_config.lookat)
    assert viewer._camera.distance == camera_config.distance
    assert viewer._camera.elevation == camera_config.elevation
    assert viewer._camera.azimuth == camera_config.azimuth

    viewer.sync()
    assert mujoco.update_count == 1
    assert mujoco.render_count == 1
    assert glfw.swap_count == 1
    assert glfw.poll_count == 1

    glfw.mouse_buttons[glfw.MOUSE_BUTTON_LEFT] = glfw.PRESS
    glfw.callbacks["cursor"](glfw.window, 100.0, 200.0)
    glfw.callbacks["scroll"](glfw.window, 0.0, 2.0)
    assert mujoco.camera_moves[0][0] == mujoco.mjtMouse.mjMOUSE_ROTATE_V
    assert mujoco.camera_moves[1] == (mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.1)

    context = mujoco.context
    viewer.close()
    viewer.close()
    assert context.free_count == 1
    assert glfw.destroy_count == 1
    assert glfw.terminate_count == 1
