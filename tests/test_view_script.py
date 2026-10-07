# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_env_core import registry
from motrix_rl.config import ViewConfig

VIEW_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "view.py"
VIEW_MODULE_SPEC = importlib.util.spec_from_file_location("motrix_test_view_script", VIEW_SCRIPT)
if VIEW_MODULE_SPEC is None or VIEW_MODULE_SPEC.loader is None:
    raise ImportError(f"Unable to load {VIEW_SCRIPT}")
view = importlib.util.module_from_spec(VIEW_MODULE_SPEC)
sys.modules[VIEW_MODULE_SPEC.name] = view
VIEW_MODULE_SPEC.loader.exec_module(view)


def test_default_view_target_is_cartpole(monkeypatch):
    env = Mock(spec=view.ArrayEnv)
    make_calls = []
    run_calls = []

    def make(name, *, num_envs):
        make_calls.append((name, num_envs))
        return env

    monkeypatch.setattr(view.registry, "make", make)
    monkeypatch.setattr(view, "_run_np", run_calls.append)

    view.run(ViewConfig())

    assert make_calls == [("cartpole", 1)]
    assert run_calls == [env]


def test_robot_view_uses_registered_robot_config(monkeypatch):
    robot = object()
    make_calls = []
    run_calls = []

    def make_robot_config(name):
        make_calls.append(name)
        return robot

    monkeypatch.setattr(view.registry, "make_robot_config", make_robot_config)
    monkeypatch.setattr(view, "_run_robot", lambda robot, sim: run_calls.append((robot, sim)))

    view.run(ViewConfig(robot="go2"))

    assert make_calls == ["go2"]
    assert run_calls == [(robot, None)]


def test_robot_view_passes_mujoco_backend(monkeypatch):
    robot = object()
    run_calls = []
    monkeypatch.setattr(view.registry, "make_robot_config", lambda _: robot)
    monkeypatch.setattr(view, "_run_robot", lambda robot, sim: run_calls.append((robot, sim)))

    view.run(ViewConfig(robot="go2", sim="mujoco"))

    assert run_calls == [(robot, "mujoco")]


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (ViewConfig(env="cartpole", robot="go2"), "mutually exclusive"),
        (ViewConfig(robot="go2", num_envs=2), "num_envs"),
    ],
)
def test_robot_view_rejects_incompatible_options(cfg, message):
    with pytest.raises(ValueError, match=message):
        view.run(cfg)


def test_motrixsim_robot_view_builds_scene_and_syncs_until_closed(monkeypatch):
    robot = UnitreeGo2Robot()
    model = SimpleNamespace(forward_kinematic_calls=[])

    def forward_kinematic(data):
        model.forward_kinematic_calls.append(data)

    model.forward_kinematic = forward_kinematic
    data = SimpleNamespace(reset_calls=[])
    data.reset = data.reset_calls.append
    scenes = []
    scene_data_calls = []

    def build_scene_model(scene):
        scenes.append(scene)
        return model

    def scene_data(model_arg, *, batch):
        scene_data_calls.append((model_arg, batch))
        return data

    class FakeCamera:
        def __init__(self):
            self.views = []
            self.active = False

        def set_view(self, *args):
            self.views.append(args)

    class FakeRenderer:
        def __init__(self):
            self.system_camera = FakeCamera()
            self.is_closed = False
            self.launch_calls = []
            self.sync_calls = []
            self.exit_calls = 0

        def launch(self, *args, **kwargs):
            self.launch_calls.append((args, kwargs))

        def sync(self, *, data):
            self.sync_calls.append(data)
            self.is_closed = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exit_calls += 1

    renderer = FakeRenderer()
    settings = SimpleNamespace(enable_shadow=False)
    monkeypatch.setattr(view, "build_scene_model", build_scene_model)
    monkeypatch.setattr(view.mtx, "SceneData", scene_data)
    monkeypatch.setattr(view, "RenderApp", lambda: renderer)
    monkeypatch.setattr(view, "RenderSettings", SimpleNamespace(performance=lambda: settings))
    monkeypatch.setattr(view.time, "sleep", lambda _: None)

    view._run_robot(robot, "motrixsim")

    assert scenes[0].objs.robot is robot
    assert scene_data_calls == [(model, [1])]
    assert data.reset_calls == [model]
    assert model.forward_kinematic_calls == [data]
    assert settings.enable_shadow is True
    assert renderer.launch_calls[0][0] == (model,)
    assert renderer.sync_calls == [data]
    assert renderer.system_camera.active is True
    assert renderer.exit_calls == 1


def test_mujoco_robot_view_builds_scene_and_syncs_until_closed(monkeypatch):
    robot = UnitreeGo2Robot()
    model = object()
    data = object()
    compiled_scenes = []
    forward_calls = []
    launch_calls = []

    class FakeCompiler:
        def compile(self, scene, sim):
            compiled_scenes.append((scene, sim))
            return model

    class FakeCamera:
        def __init__(self):
            self.lookat = [0.0, 0.0, 0.0]
            self.distance = 0.0
            self.elevation = 0.0
            self.azimuth = 0.0

    class FakeLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

    class FakeViewer:
        def __init__(self):
            self.cam = FakeCamera()
            self.running = True
            self.sync_calls = 0
            self.exit_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exit_calls += 1

        def is_running(self):
            return self.running

        def lock(self):
            return FakeLock()

        def sync(self):
            self.sync_calls += 1
            self.running = False

    viewer = FakeViewer()
    fake_mujoco = SimpleNamespace(MjData=lambda model_arg: data, mj_forward=lambda *args: forward_calls.append(args))
    fake_viewer_module = SimpleNamespace(
        launch_passive=lambda *args: launch_calls.append(args) or viewer,
    )
    fake_mujoco.viewer = fake_viewer_module
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer_module)
    fake_compiler_module = SimpleNamespace(MuJoCoSceneCompiler=lambda: FakeCompiler())
    monkeypatch.setitem(sys.modules, "motrix_env_mujoco.compiler", fake_compiler_module)
    monkeypatch.setattr(view.time, "sleep", lambda _: None)

    view._run_robot(robot, "mujoco")

    scene, sim = compiled_scenes[0]
    assert scene.objs.robot is robot
    assert isinstance(sim, view.SimCfg)
    assert forward_calls == [(model, data)]
    assert launch_calls == [(model, data)]
    assert viewer.cam.lookat == [0.0, 0.0, 0.75]
    assert viewer.cam.distance == 3.0
    assert viewer.cam.elevation == -20.0
    assert viewer.cam.azimuth == 180.0
    assert viewer.sync_calls == 1
    assert viewer.exit_calls == 1


def test_robot_view_rejects_unknown_backend():
    with pytest.raises(ValueError, match="sim"):
        view._run_robot(UnitreeGo2Robot(), "np")


def test_builtin_go2_is_available_to_view():
    assert isinstance(registry.make_robot_config("go2"), UnitreeGo2Robot)
