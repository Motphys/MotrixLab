# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import MjcfFileCfg, RobotCfg


@dataclass
class FakeEntryPoint:
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


@pytest.fixture(autouse=True)
def isolated_robot_registry(monkeypatch):
    monkeypatch.setattr(registry, "_robots", {})
    monkeypatch.setattr(registry, "_robots_discovered", False)
    monkeypatch.setattr(registry, "_robots_discovering", False)
    monkeypatch.setattr(registry, "_robot_entry_points", None)
    monkeypatch.setattr(registry, "_completed_robot_entry_points", 0)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: ())


@pytest.fixture
def robot_type(tmp_path):
    @configclass(kw_only=True)
    class CustomRobotCfg(RobotCfg):
        model: MjcfFileCfg = MjcfFileCfg(file=tmp_path / "robot.xml")
        base_link_name: str = "base"

        def validate(self, path="robot"):
            self.validated_path = path

    return CustomRobotCfg


@pytest.mark.parametrize("query", ["make", "list"])
def test_plugins_are_lazy_sorted_and_discovered_once(monkeypatch, robot_type, query):
    calls = []
    metadata_queries = []

    def register_a():
        calls.append("a")
        registry.register_robot_config("plugin-a", robot_type)

    def register_b():
        calls.append("b")
        registry.register_robot_config("plugin-b", robot_type)

    a = FakeEntryPoint("a", "a:register", register_a)
    b = FakeEntryPoint("b", "b:register", register_b)

    def entry_points(*, group):
        metadata_queries.append(group)
        return [b, a]

    monkeypatch.setattr(registry, "entry_points", entry_points)
    registry.register_robot_config("manual", robot_type)
    assert not metadata_queries
    if query == "make":
        registry.make_robot_config("plugin-a")
    else:
        registry.list_registered_robots()
    first = registry.make_robot_config("plugin-a")
    second = registry.make_robot_config("plugin-a")
    assert first is not second
    assert first.validated_path == "robot"
    assert registry.list_registered_robots() == {
        name: {"config_class": "CustomRobotCfg"} for name in ("manual", "plugin-a", "plugin-b")
    }
    assert metadata_queries == [registry.ROBOT_ENTRY_POINT_GROUP]
    assert calls == ["a", "b"]
    assert a.loads == b.loads == 1


def test_no_plugins_supports_manual_registration_and_unknown_errors(robot_type):
    assert registry.list_registered_robots() == {}
    registry.register_robot_config("manual", robot_type)
    assert isinstance(registry.make_robot_config("manual"), robot_type)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_robot_config("manual", robot_type)
    with pytest.raises(ValueError, match="not registered"):
        registry.make_robot_config("unknown")


def test_discovered_factory_preserves_validation_and_return_type_checks(monkeypatch, robot_type):
    class InvalidRobotCfg(robot_type):
        def validate(self, path="robot"):
            raise ValueError("invalid robot")

    def register():
        registry.register_robot_config("wrong-result", lambda: object(), cfg_type=robot_type)
        registry.register_robot_config("invalid", InvalidRobotCfg)

    monkeypatch.setattr(
        registry, "entry_points", lambda *, group: [FakeEntryPoint("custom", "custom:register", register)]
    )
    with pytest.raises(TypeError, match="config factory must return CustomRobotCfg"):
        registry.make_robot_config("wrong-result")
    with pytest.raises(ValueError):
        registry.make_robot_config("invalid")


@pytest.mark.parametrize("failure", ["load", "noncallable", "callback"])
def test_plugin_errors_propagate_and_are_retried(monkeypatch, failure):
    error = RuntimeError("plugin failed")

    def fail():
        raise error

    ep = FakeEntryPoint("broken", "broken:register", fail)
    if failure == "load":
        ep.load_error = error
    elif failure == "noncallable":
        ep.callback = object()
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [ep])
    for _ in range(2):
        if failure == "noncallable":
            with pytest.raises(TypeError, match="Robot entry point 'broken' must load a callable"):
                registry.list_registered_robots()
        else:
            with pytest.raises(RuntimeError) as caught:
                registry.list_registered_robots()
            assert caught.value is error
        assert not registry._robots_discovered
        assert not registry._robots_discovering
    assert ep.loads == 2


def test_retry_does_not_repeat_completed_callbacks(monkeypatch, robot_type):
    def register_first():
        registry.register_robot_config("first", robot_type)

    def fail():
        raise RuntimeError("try again")

    first = FakeEntryPoint("a", "a:register", register_first)
    second = FakeEntryPoint("b", "b:register", fail)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [first, second])
    with pytest.raises(RuntimeError, match="try again"):
        registry.list_registered_robots()
    second.callback = lambda: registry.register_robot_config("second", robot_type)
    assert set(registry.list_registered_robots()) == {"first", "second"}
    assert first.loads == 1
    assert second.loads == 2


def test_failed_callback_registration_side_effects_are_not_rolled_back(monkeypatch, robot_type):
    def fail_after_registering():
        registry.register_robot_config("partial", robot_type)
        raise RuntimeError("partial registration")

    ep = FakeEntryPoint("partial", "partial:register", fail_after_registering)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [ep])
    with pytest.raises(RuntimeError, match="partial registration"):
        registry.list_registered_robots()
    assert "partial" in registry._robots
    # Plugins own recovery from partial registration; the registry is not transactional.
    with pytest.raises(ValueError, match="already registered"):
        registry.list_registered_robots()


def test_identical_entry_point_declarations_are_not_silently_skipped(monkeypatch, robot_type):
    def register():
        registry.register_robot_config("duplicate", robot_type)

    first = FakeEntryPoint("same", "same:register", register)
    second = FakeEntryPoint("same", "same:register", register)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [first, second])
    with pytest.raises(ValueError, match="already registered"):
        registry.list_registered_robots()
    assert first.loads == second.loads == 1


def test_registration_callback_can_reenter_queries(monkeypatch, robot_type):
    calls = []

    def register():
        calls.append("register")
        assert registry.list_registered_robots() == {}
        registry.register_robot_config("recursive", robot_type)
        assert isinstance(registry.make_robot_config("recursive"), robot_type)
        assert "recursive" in registry.list_registered_robots()

    ep = FakeEntryPoint("recursive", "recursive:register", register)
    monkeypatch.setattr(registry, "entry_points", lambda *, group: [ep])
    assert "recursive" in registry.list_registered_robots()
    assert calls == ["register"]
    assert ep.loads == 1
