# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Interactive lifecycle using the installed MotrixSim RenderApp API."""

from motrixsim.render import RenderApp

from motrix_env_core.input import KeyboardDevice


class MotrixSimKeyboard(KeyboardDevice):
    def __init__(self, input=None):
        self._input = input
        self._previous = {}
        self._held = {}
        self._down = {}
        self._up = {}

    def bind(self, input) -> None:
        self._input = input
        self._previous.clear()
        self._held.clear()
        self._down.clear()
        self._up.clear()

    def poll(self) -> None:
        self._require_input()
        self._previous = self._held
        self._held = {}
        self._down = {}
        self._up = {}
        for key in self._previous:
            self._sample(key)

    def _require_input(self):
        if self._input is None:
            raise RuntimeError("Open the MotrixSim viewer before polling keyboard input")
        return self._input

    def _sample(self, key):
        input = self._require_input()
        if key not in self._held:
            held = input.is_key_pressed(key)
            self._held[key] = held
            self._down[key] = input.is_key_just_pressed(key)
            self._up[key] = self._previous.get(key, False) and not held

    def is_key_down(self, key: str) -> bool:
        self._sample(key)
        return self._down[key]

    def is_key_up(self, key: str) -> bool:
        self._sample(key)
        return self._up[key]

    def is_pressing(self, key: str) -> bool:
        self._sample(key)
        return self._held[key]


class MotrixSimViewer:
    def __init__(self, camera) -> None:
        self._camera = camera
        self._render = None
        self.keyboard_device = MotrixSimKeyboard()

    def open(self, model) -> None:
        self._render = RenderApp(headless=False)
        try:
            self._render.launch(model, batch=1)
            camera = self._camera
            self._render.system_camera.set_view(
                list(camera.lookat) if camera.lookat is not None else [0.0, 0.0, 0.0],
                camera.distance,
                camera.elevation,
                camera.azimuth,
            )
            self.keyboard_device.bind(self._render.input)
        except BaseException:
            self.close()
            raise

    def is_running(self) -> bool:
        return self._render is not None and not self._render.is_closed

    def sync(self, data) -> None:
        self._render.sync(data=data)

    def close(self) -> None:
        try:
            if self._render is not None:
                self._render.close()
        finally:
            self._render = None
            self.keyboard_device.bind(None)
