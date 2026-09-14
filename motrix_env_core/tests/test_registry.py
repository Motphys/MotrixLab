# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from collections.abc import Mapping

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
def _restore_global_registry():
    """Drop any envs/robots a test registers so they do not leak across tests.

    Several tests below register throwaway env/robot configs into the module-level
    registry. Without cleanup they leak into later tests (e.g. the robot-docs suite
    asserts the registered robot set matches its metadata) and into whole-repo runs.
    """
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


def test_environment_backend_is_inferred_from_base_class() -> None:
    class TorchEnv(ABEnv):
        """Name-based stand-in: inference matches the base-class *name* in the MRO."""

    assert registry._infer_sim_backend(DirectEnv) == "np"
    assert registry._infer_sim_backend(TorchEnv) == "torch"

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

        def reset(self, env_ids: np.ndarray):
            return {}

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

        def reset(self, env_ids: np.ndarray):
            return {}

    env = registry.make(env_name, num_envs=2, sim="fake-registry")

    assert isinstance(env, DirectSimEnv)
    assert env.num_envs == 2
