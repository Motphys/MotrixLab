# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Minimal GLFW viewer and focused keyboard device for MuJoCo deployment."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ctypes

    import glfw
    import mujoco

from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.input import KeyboardDevice


class MujocoKeyboardDevice(KeyboardDevice):
    """Freeze keyboard events delivered to one focused GLFW viewer window."""

    def __init__(self, viewer: MujocoGlfwViewer) -> None:
        self._viewer = viewer
        self._pressing: set[str] = set()
        self._pending_down: set[str] = set()
        self._pending_up: set[str] = set()
        self._frame_pressing: frozenset[str] = frozenset()
        self._frame_down: frozenset[str] = frozenset()
        self._frame_up: frozenset[str] = frozenset()
        self._interrupted = False

    def poll(self) -> None:
        """Poll GLFW and freeze one non-consuming keyboard event frame."""
        window = self._viewer._require_window()
        glfw = self._viewer._glfw
        assert glfw is not None
        glfw.poll_events()
        if self._interrupted or glfw.window_should_close(window):
            raise KeyboardInterrupt
        self._frame_down = frozenset(self._pending_down)
        self._frame_up = frozenset(self._pending_up)
        self._frame_pressing = frozenset(self._pressing)
        self._pending_down.clear()
        self._pending_up.clear()

    def is_key_down(self, key: str) -> bool:
        return self._normalize_query(key) in self._frame_down

    def is_key_up(self, key: str) -> bool:
        return self._normalize_query(key) in self._frame_up

    def is_pressing(self, key: str) -> bool:
        return self._normalize_query(key) in self._frame_pressing

    def _on_key(
        self, window: ctypes._Pointer[glfw._GLFWwindow], key: int, scancode: int, action: int, modifiers: int
    ) -> None:
        del window, modifiers
        glfw = self._viewer._glfw
        assert glfw is not None
        name = self._key_name(key, scancode)
        if name is None:
            return
        if action == glfw.PRESS:
            if name not in self._pressing:
                self._pressing.add(name)
                self._pending_down.add(name)
            if name == "esc":
                self._interrupted = True
        elif action == glfw.RELEASE and name in self._pressing:
            self._pressing.remove(name)
            self._pending_up.add(name)

    def _on_focus(self, window: ctypes._Pointer[glfw._GLFWwindow], focused: int) -> None:
        del window
        if not focused:
            self._pending_up.update(self._pressing)
            self._pressing.clear()

    def _on_close(self, window: ctypes._Pointer[glfw._GLFWwindow]) -> None:
        del window
        self._interrupted = True

    def _key_name(self, key: int, scancode: int) -> str | None:
        glfw = self._viewer._glfw
        assert glfw is not None
        if glfw.KEY_A <= key <= glfw.KEY_Z:
            return chr(ord("a") + key - glfw.KEY_A)
        if glfw.KEY_0 <= key <= glfw.KEY_9:
            return chr(ord("0") + key - glfw.KEY_0)
        special = {
            glfw.KEY_ESCAPE: "esc",
            glfw.KEY_SPACE: "space",
            glfw.KEY_UP: "up",
            glfw.KEY_DOWN: "down",
            glfw.KEY_LEFT: "left",
            glfw.KEY_RIGHT: "right",
        }
        name = special.get(key)
        if name is not None:
            return name
        printable = glfw.get_key_name(key, scancode)
        return printable.lower() if isinstance(printable, str) else None

    @staticmethod
    def _normalize_query(key: str) -> str:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        return key.lower()

    def _clear(self) -> None:
        self._pressing.clear()
        self._pending_down.clear()
        self._pending_up.clear()
        self._frame_pressing = frozenset()
        self._frame_down = frozenset()
        self._frame_up = frozenset()
        self._interrupted = False


class MujocoGlfwViewer:
    """Render MuJoCo state and own input events from its GLFW window."""

    def __init__(
        self,
        mujoco_module: ModuleType,
        camera_config: SystemCameraCfg = SystemCameraCfg(),
        *,
        glfw_module: ModuleType | None = None,
    ) -> None:
        self._mj = mujoco_module
        self._glfw = glfw_module
        self._camera_config = camera_config
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._window: ctypes._Pointer[glfw._GLFWwindow] | None = None
        self._scene: mujoco.MjvScene | None = None
        self._context: mujoco.MjrContext | None = None
        self._camera: mujoco.MjvCamera | None = None
        self._option: mujoco.MjvOption | None = None
        self._perturb: mujoco.MjvPerturb | None = None
        self._glfw_initialized = False
        self._cursor_position = (0.0, 0.0)
        self.keyboard_device = MujocoKeyboardDevice(self)

    def open(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Create the window and MuJoCo rendering resources."""
        if self._window is not None:
            raise RuntimeError("MuJoCo GLFW viewer is already open")
        if self._glfw is None:
            try:
                import glfw
            except ImportError as error:
                raise RuntimeError("MuJoCo deployment viewer requires glfw") from error
            self._glfw = glfw
        glfw = self._glfw
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW for the MuJoCo deployment viewer")
        self._glfw_initialized = True
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            window = glfw.create_window(1200, 900, "Motrix Deploy - MuJoCo", None, None)
            if window is None:
                raise RuntimeError("Failed to create the MuJoCo deployment viewer window")
            self._window = window
            self._model = model
            self._data = data
            glfw.make_context_current(window)
            glfw.swap_interval(0)
            self._camera = self._mj.MjvCamera()
            self._mj.mjv_defaultFreeCamera(model, self._camera)
            if self._camera_config.lookat is not None:
                self._camera.lookat[:] = self._camera_config.lookat
            self._camera.distance = self._camera_config.distance
            self._camera.elevation = self._camera_config.elevation
            self._camera.azimuth = self._camera_config.azimuth
            self._option = self._mj.MjvOption()
            self._perturb = self._mj.MjvPerturb()
            self._scene = self._mj.MjvScene(model, maxgeom=10_000)
            self._context = self._mj.MjrContext(model, self._mj.mjtFontScale.mjFONTSCALE_150)
            self._cursor_position = glfw.get_cursor_pos(window)
            glfw.set_key_callback(window, self.keyboard_device._on_key)
            glfw.set_window_focus_callback(window, self.keyboard_device._on_focus)
            glfw.set_window_close_callback(window, self.keyboard_device._on_close)
            glfw.set_cursor_pos_callback(window, self._on_cursor_position)
            glfw.set_scroll_callback(window, self._on_scroll)
        except Exception:
            self.close()
            raise

    def sync(self) -> None:
        """Pump window events and render without consuming keyboard input or stepping."""
        window = self._require_window()
        assert self._glfw is not None
        # Window responsiveness must not depend on a keyboard command binding.
        # Callbacks retain pending key edges until KeyboardDevice.poll freezes them.
        self._glfw.poll_events()
        if self.keyboard_device._interrupted or self._glfw.window_should_close(window):
            raise KeyboardInterrupt
        assert self._model is not None
        assert self._data is not None
        assert self._scene is not None
        assert self._context is not None
        assert self._camera is not None
        assert self._option is not None
        assert self._perturb is not None
        self._glfw.make_context_current(window)
        self._mj.mjv_updateScene(
            self._model,
            self._data,
            self._option,
            self._perturb,
            self._camera,
            self._mj.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        width, height = self._glfw.get_framebuffer_size(window)
        if width > 0 and height > 0:
            viewport = self._mj.MjrRect(0, 0, width, height)
            self._mj.mjr_render(viewport, self._scene, self._context)
            self._glfw.swap_buffers(window)

    def is_running(self) -> bool:
        if self._window is None:
            return False
        assert self._glfw is not None
        return not self._glfw.window_should_close(self._window)

    def close(self) -> None:
        """Release rendering resources and the GLFW window; safe to repeat."""
        glfw = self._glfw
        window = self._window
        context = self._context
        self._window = None
        self._context = None
        try:
            if glfw is not None and window is not None:
                glfw.make_context_current(window)
            if context is not None:
                context.free()
        finally:
            if glfw is not None and window is not None:
                glfw.make_context_current(None)
                glfw.destroy_window(window)
            if glfw is not None and self._glfw_initialized:
                glfw.terminate()
            self._glfw_initialized = False
            self._model = None
            self._data = None
            self._scene = None
            self._camera = None
            self._option = None
            self._perturb = None
            self.keyboard_device._clear()

    def _on_cursor_position(self, window: ctypes._Pointer[glfw._GLFWwindow], xpos: float, ypos: float) -> None:
        assert self._glfw is not None
        previous_x, previous_y = self._cursor_position
        self._cursor_position = (xpos, ypos)
        dx = xpos - previous_x
        dy = ypos - previous_y
        left = self._glfw.get_mouse_button(window, self._glfw.MOUSE_BUTTON_LEFT) == self._glfw.PRESS
        right = self._glfw.get_mouse_button(window, self._glfw.MOUSE_BUTTON_RIGHT) == self._glfw.PRESS
        middle = self._glfw.get_mouse_button(window, self._glfw.MOUSE_BUTTON_MIDDLE) == self._glfw.PRESS
        if not (left or right or middle):
            return
        _, height = self._glfw.get_window_size(window)
        if height <= 0:
            return
        shift = (
            self._glfw.get_key(window, self._glfw.KEY_LEFT_SHIFT) == self._glfw.PRESS
            or self._glfw.get_key(window, self._glfw.KEY_RIGHT_SHIFT) == self._glfw.PRESS
        )
        if right:
            action = self._mj.mjtMouse.mjMOUSE_MOVE_H if shift else self._mj.mjtMouse.mjMOUSE_MOVE_V
        elif left:
            action = self._mj.mjtMouse.mjMOUSE_ROTATE_H if shift else self._mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = self._mj.mjtMouse.mjMOUSE_ZOOM
        self._move_camera(action, dx / height, dy / height)

    def _on_scroll(self, window: ctypes._Pointer[glfw._GLFWwindow], xoffset: float, yoffset: float) -> None:
        del window, xoffset
        self._move_camera(self._mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset)

    def _move_camera(self, action: mujoco.mjtMouse, dx: float, dy: float) -> None:
        if self._model is None or self._scene is None or self._camera is None:
            return
        self._mj.mjv_moveCamera(self._model, action, dx, dy, self._scene, self._camera)

    def _require_window(self) -> ctypes._Pointer[glfw._GLFWwindow]:
        if self._window is None:
            raise RuntimeError("MuJoCo GLFW viewer is not open")
        return self._window


__all__ = ["MujocoGlfwViewer", "MujocoKeyboardDevice"]
