# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.base import ABEnv
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import MjcfFileCfg, RobotCfg, SceneCfg
from motrix_env_core.direct.env import DirectEnv, DirectEnvCfg
from motrix_env_core.numba.manager.env import ManagerBasedEnvCfg, ManagerEnv
from motrix_env_core.sim import ModelQuery
from motrix_env_core.sim.backend import SimBackend
from motrix_env_core.sim.model import SimModel
from motrix_env_core.sim.read import PhysicsReadProgram
from motrix_env_core.sim.registry import register_sim_backend


class _EmptyReadProgram(PhysicsReadProgram):
    """Serves an empty declared query set."""

    @property
    def arena_bytes(self) -> int:
        return 0

    @property
    def keys(self) -> tuple[str, ...]:
        return ()

    def query(self, key: str):
        raise KeyError(key)

    def view(self, key: str) -> np.ndarray:
        raise KeyError(key)

    def execute(self, env_ids=None) -> None:
        pass


class _FakeRegistryBackend(SimBackend):
    """Minimal compile-free backend so core tests never need a real simulator."""

    name = "fake-registry"

    def __init__(self, scene, sim, num_envs: int) -> None:
        super().__init__(scene, sim, num_envs)
        self.num_envs = num_envs

    @property
    def num_dof_pos(self) -> int:
        return 0

    @property
    def num_dof_vel(self) -> int:
        return 0

    @property
    def num_actuators(self) -> int:
        return 0

    def step(self, substeps: int) -> None:
        pass

    @property
    def model_compiler(self):
        return self

    def compile(self, scene, queries: Mapping[str, ModelQuery]) -> SimModel:
        del scene, queries
        return SimModel(
            actuators=(),
            init_dof_pos=np.zeros((0,), dtype=np.float32),
        )

    def compile_reads(self, queries) -> PhysicsReadProgram:
        assert not queries
        return _EmptyReadProgram()


register_sim_backend("fake-registry", lambda: _FakeRegistryBackend)


@pytest.fixture(autouse=True)
def _restore_global_registry(monkeypatch):
    """Drop any envs/robots a test registers so they do not leak across tests.

    Several tests below register throwaway env/robot configs into the module-level
    registry. Without cleanup they leak into later tests (e.g. the robot-docs suite
    asserts the registered robot set matches its metadata) and into whole-repo runs.
    """
    # Core unit tests must not discover installed environment/robot distributions.
    monkeypatch.setattr(registry, "_envs", {})
    monkeypatch.setattr(registry, "_envs_discovered", False)
    monkeypatch.setattr(registry, "_envs_discovering", False)
    monkeypatch.setattr(registry, "_env_entry_points", None)
    monkeypatch.setattr(registry, "_completed_env_entry_points", 0)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: ())
    monkeypatch.setattr(registry, "_robots", {})
    monkeypatch.setattr(registry, "_robots_discovered", False)
    monkeypatch.setattr(registry, "_robots_discovering", False)
    monkeypatch.setattr(registry, "_robot_entry_points", None)
    monkeypatch.setattr(registry, "_completed_robot_entry_points", 0)
    envs_before = set(registry._envs)
    robots_before = set(registry._robots)
    yield
    for name in set(registry._envs) - envs_before:
        registry._envs.pop(name, None)
    for name in set(registry._robots) - robots_before:
        registry._robots.pop(name, None)


def test_importing_framework_does_not_import_builtin_environments():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import motrix_env_core; assert 'motrix_envs' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("base_class", [DirectEnv, ManagerEnv])
def test_environment_backend_is_inferred_from_base_class(base_class) -> None:
    class CustomEnvironment(base_class):
        pass

    assert registry._infer_sim_backend(CustomEnvironment) == "np"

    class UnknownEnv(ABEnv):
        pass

    with pytest.raises(ValueError, match="must inherit DirectEnv or TorchEnv"):
        registry._infer_sim_backend(UnknownEnv)


def test_environment_metadata_resolves_the_registered_backend() -> None:
    class TorchEnv(ABEnv):
        pass

    meta = registry.EnvMeta(
        env_cfg_cls=registry.EnvCfg,
        env_cfg_factory=registry.EnvCfg,
        env_cls_dict={"np": DirectEnv, "torch": TorchEnv},
    )

    assert registry._resolve_backend("test-env", meta) == ("np", DirectEnv)


def test_manager_seed_is_passed_to_manager_environment():
    captured = {}

    class FakeManagerEnv(ManagerEnv):
        def __init__(self, cfg, num_envs=1, backend=None, seed=None):
            captured["seed"] = seed

    cfg = ManagerBasedEnvCfg()
    registry._construct_env(FakeManagerEnv, cfg, 1, "fake-registry", 17)

    assert captured["seed"] == 17


def test_custom_environment_can_be_registered_and_created_without_builtins():
    env_name = "core-test-custom-env"

    @registry.envcfg(env_name)
    @configclass
    class CustomEnvCfg(DirectEnvCfg):
        """Custom environment used by the registry test.

        zh_CN: Registry 测试使用的自定义环境。
        """

        scene: SceneCfg = SceneCfg()

    @registry.env(env_name)
    class CustomEnv(DirectEnv[CustomEnvCfg]):
        @property
        def observation_space(self) -> gym.spaces.Box:
            return gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        @property
        def action_space(self) -> gym.spaces.Box:
            return gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
            return state

        def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
            return state

        def reset(self, env_ids: np.ndarray) -> None:
            pass

    env = registry.make(env_name, num_envs=2, sim="fake-registry")

    assert isinstance(env, CustomEnv)
    assert env.num_envs == 2
    assert registry.list_registered_envs()[env_name]["description"] == {
        "en": "Custom environment used by the registry test.",
        "zh_CN": "Registry 测试使用的自定义环境。",
    }


def test_custom_robot_config_can_be_registered_and_created(tmp_path):
    robot_name = "core-test-custom-robot"
    model_path = tmp_path / "robot.xml"
    model_path.write_text("<mujoco/>", encoding="utf-8")

    @registry.robotcfg(robot_name)
    @configclass(kw_only=True)
    class CustomRobotCfg(RobotCfg):
        model: MjcfFileCfg = MjcfFileCfg(file=model_path)
        base_link_name: str = "base"

    first = registry.make_robot_config(robot_name)
    second = registry.make_robot_config(robot_name)

    assert isinstance(first, CustomRobotCfg)
    assert isinstance(second, CustomRobotCfg)
    assert first is not second
    assert registry.list_registered_robots()[robot_name] == {"config_class": "CustomRobotCfg"}


def test_direct_env_resolves_sim_backend_through_registry():
    env_name = "core-test-direct-env"

    @registry.envcfg(env_name)
    @configclass
    class DirectSimCfg(DirectEnvCfg):
        """Direct env used by the registry sim test."""

        scene: SceneCfg = SceneCfg()

    @registry.env(env_name)
    class DirectSimEnv(DirectEnv[DirectSimCfg]):
        @property
        def observation_space(self) -> gym.spaces.Box:
            return gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        @property
        def action_space(self) -> gym.spaces.Box:
            return gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
            return state

        def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
            return state

        def reset(self, env_ids: np.ndarray) -> None:
            pass

    env = registry.make(env_name, num_envs=2, sim="fake-registry")

    assert isinstance(env, DirectSimEnv)
    assert env.num_envs == 2


@dataclass
class _EnvEntryPoint:
    name: str
    value: str
    callback: object
    loads: int = 0
    load_error: Exception | None = None

    def load(self):
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error
        return self.callback


@configclass
class _PluginEnvCfg(DirectEnvCfg):
    scene: SceneCfg = SceneCfg()


class _PluginEnv(DirectEnv[_PluginEnvCfg]):
    pass


def _register_plugin_env(name):
    registry.envcfg(name)(_PluginEnvCfg)
    registry.env(name)(_PluginEnv)


@pytest.mark.parametrize("query", ["contains", "config", "resolve", "make", "list"])
def test_env_plugins_are_lazy_sorted_once_at_every_query_boundary(monkeypatch, query):
    calls = []
    metadata_queries = []

    def register(name):
        calls.append(name)
        _register_plugin_env(name)

    a = _EnvEntryPoint("a", "external_a:register", lambda: register("plugin-a"))
    b = _EnvEntryPoint("b", "external_b:register", lambda: register("plugin-b"))

    def entry_points(*, group):
        metadata_queries.append(group)
        return [b, a]

    monkeypatch.setattr(registry, "entry_points", entry_points)
    monkeypatch.setattr(registry, "_construct_env", lambda cls, cfg, *args: (cls, cfg))
    _register_plugin_env("manual")
    assert metadata_queries == []
    if query == "contains":
        assert registry.contains("plugin-a")
    elif query == "config":
        assert isinstance(registry.make_env_config("plugin-a"), _PluginEnvCfg)
    elif query == "resolve":
        assert registry.resolve("plugin-a", env_cfg=_PluginEnvCfg()).env_cls is _PluginEnv
    elif query == "make":
        cls, cfg = registry.make("plugin-a")
        assert cls is _PluginEnv
        assert isinstance(cfg, _PluginEnvCfg)
    else:
        assert "plugin-a" in registry.list_registered_envs()
    first = registry.make_env_config("plugin-a")
    assert first is not registry.make_env_config("plugin-a")
    assert registry.resolve("manual").env_cls is _PluginEnv
    assert set(registry.list_registered_envs()) == {"manual", "plugin-a", "plugin-b"}
    assert registry.contains("manual")
    assert metadata_queries == [registry.ENV_ENTRY_POINT_GROUP]
    assert calls == ["plugin-a", "plugin-b"]
    assert a.loads == b.loads == 1


def test_env_plugin_registration_can_reenter_queries_and_decorators(monkeypatch):
    def register():
        assert registry.list_registered_envs() == {}
        assert not registry.contains("recursive")
        _register_plugin_env("recursive")
        assert registry.contains("recursive")
        assert isinstance(registry.make_env_config("recursive"), _PluginEnvCfg)
        assert registry.resolve("recursive").env_cls is _PluginEnv
        assert "recursive" in registry.list_registered_envs()

    ep = _EnvEntryPoint("recursive", "external:register", register)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [ep])
    assert registry.contains("recursive")
    assert ep.loads == 1


@pytest.mark.parametrize("failure", ["metadata", "load", "noncallable", "callback"])
def test_env_plugin_errors_propagate_and_retry_even_for_manual_queries(monkeypatch, failure):
    error = RuntimeError("external provider failed")
    _register_plugin_env("manual")

    def fail():
        raise error

    ep = _EnvEntryPoint("broken", "external:register", fail)
    if failure == "load":
        ep.load_error = error
    elif failure == "noncallable":
        ep.callback = object()

    def entry_points(*, group):
        if failure == "metadata":
            raise error
        return [ep]

    monkeypatch.setattr(registry, "entry_points", entry_points)
    for _ in range(2):
        if failure == "noncallable":
            with pytest.raises(TypeError, match="Environment entry point 'broken' must load a callable"):
                registry.contains("manual")
        else:
            with pytest.raises(RuntimeError) as caught:
                registry.contains("manual")
            assert caught.value is error
    if failure != "metadata":
        assert ep.loads == 2
    ep.load_error = None
    ep.callback = lambda: _register_plugin_env("recovered")
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [ep])
    assert registry.contains("recovered")
    assert registry.contains("manual")


def test_env_plugin_retry_does_not_repeat_completed_providers(monkeypatch):
    def fail_after_registering():
        _register_plugin_env("partial")
        raise RuntimeError("retry provider")

    first = _EnvEntryPoint("a", "external_a:register", lambda: _register_plugin_env("first"))
    second = _EnvEntryPoint("b", "external_b:register", fail_after_registering)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [first, second])
    with pytest.raises(RuntimeError, match="retry provider"):
        registry.list_registered_envs()
    # Side effects are retained, not rolled back; providers own partial recovery.
    with pytest.raises(ValueError, match="Environment 'partial' is already registered"):
        registry.list_registered_envs()
    second.callback = lambda: _register_plugin_env("second")
    assert set(registry.list_registered_envs()) == {"first", "partial", "second"}
    assert first.loads == 1
    assert second.loads == 3


@pytest.mark.parametrize("second_value", ["external_a:register", "external_b:register"])
def test_env_duplicate_provider_names_fail_before_loading(monkeypatch, second_value):
    first = _EnvEntryPoint("duplicate", "external_a:register", lambda: None)
    second = _EnvEntryPoint("duplicate", second_value, lambda: None)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [first, second])
    for _ in range(2):
        with pytest.raises(ValueError, match="Duplicate environment provider name 'duplicate'"):
            registry.list_registered_envs()
    assert first.loads == second.loads == 0


def test_env_plugin_registration_conflicts_are_not_silently_skipped(monkeypatch):
    _register_plugin_env("manual")
    ep = _EnvEntryPoint("external", "external:register", lambda: _register_plugin_env("manual"))
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [ep])
    with pytest.raises(ValueError, match="Environment 'manual' is already registered"):
        registry.make_env_config("manual")
    assert registry._envs["manual"].env_cfg_cls is _PluginEnvCfg
    assert registry._envs["manual"].env_cls_dict == {"np": _PluginEnv}
