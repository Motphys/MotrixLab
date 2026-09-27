# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

import motrix_env_core
import motrix_envs  # noqa: F401
from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.config.scene import SceneCfg
from motrix_envs.locomotion.wbt.dex_evt import DexEvtWbtEnvCfg
from motrix_envs.locomotion.wbt.g1.common import G1WbtEnvCfg
from motrix_envs.locomotion.wbt.g1.dance import make_g129dof_wbt_dance_cfg
from motrix_envs.locomotion.wbt.k1 import K1WbtEnvCfg


def test_importing_builtins_registers_environments_in_core_registry():
    assert registry.contains("cartpole")
    assert registry.contains("g1-wbt-dance")


@pytest.mark.parametrize("cfg_type", [G1WbtEnvCfg, K1WbtEnvCfg, DexEvtWbtEnvCfg])
def test_wbt_robot_config_accepts_motion_file(cfg_type):
    cfg = cfg_type(motion_file="dance.npz")

    assert cfg.commands.motion.motion_file == "dance.npz"


@pytest.mark.parametrize("cfg_type", [G1WbtEnvCfg, K1WbtEnvCfg, DexEvtWbtEnvCfg])
def test_wbt_robot_config_preserves_nested_yaml_motion_file(cfg_type):
    base_cfg = cfg_type(motion_file="default.npz")
    cfg = OmegaConf.merge(
        OmegaConf.structured(base_cfg),
        {"commands": {"motion": {"motion_file": "dance.yaml.npz"}}},
    )
    typed_cfg = OmegaConf.to_object(cfg)

    assert typed_cfg.commands.motion.motion_file == "dance.yaml.npz"


def test_builtin_and_framework_defaults_are_isolated():
    first_env = EnvCfg()
    second_env = EnvCfg()
    first_scene = SceneCfg()
    second_scene = SceneCfg()
    first_g1 = make_g129dof_wbt_dance_cfg()
    second_g1 = make_g129dof_wbt_dance_cfg()

    second_env_dt = second_env.sim.dt
    second_scene_distance = second_scene.system_camera.distance
    second_scene_obj_count = len(second_scene.objs)
    hip_index = second_g1.scene.objs.robot.key_pose.joint_names.index("left_hip_pitch_joint")
    second_g1_hip_angle = second_g1.scene.objs.robot.key_pose.poses["default"][hip_index]
    second_g1_camera_distance = second_g1.scene.system_camera.distance
    second_g1_obj_count = len(second_g1.scene.objs)
    second_g1_ground_roughness = second_g1.scene.assets.mat_ground.roughness
    second_g1_floor_height = second_g1.scene.objs.floor.height
    second_g1_robot_prefix = second_g1.scene.objs.robot.prefix

    first_env.sim.dt = 0.005
    first_scene.system_camera.distance = 8.0
    first_g1.scene.objs.robot.key_pose.poses["default"][hip_index] = 1.0
    first_g1.scene.system_camera.distance = 9.0
    first_g1.scene.assets.mat_ground.roughness = 0.9
    first_g1.scene.objs.floor.height = 0.5
    first_g1.scene.objs.robot.prefix = "first_"

    assert second_env.sim.dt == second_env_dt
    assert second_scene.system_camera.distance == second_scene_distance
    assert len(second_scene.objs) == second_scene_obj_count
    assert second_g1.scene.objs.robot.key_pose.poses["default"][hip_index] == second_g1_hip_angle
    assert second_g1.scene.system_camera.distance == second_g1_camera_distance
    assert len(second_g1.scene.objs) == second_g1_obj_count
    assert second_g1.scene.assets.mat_ground.roughness == second_g1_ground_roughness
    assert second_g1.scene.objs.floor.height == second_g1_floor_height
    assert second_g1.scene.objs.robot.prefix == second_g1_robot_prefix

    assert first_env.sim is not second_env.sim
    assert first_scene.system_camera is not second_scene.system_camera
    assert first_scene.objs is not second_scene.objs
    assert first_g1.scene is not second_g1.scene
    assert first_g1.scene.system_camera is not second_g1.scene.system_camera
    assert first_g1.scene.assets is not second_g1.scene.assets
    assert first_g1.scene.assets.mat_ground is not second_g1.scene.assets.mat_ground
    assert first_g1.scene.objs is not second_g1.scene.objs
    assert first_g1.scene.objs.floor is not second_g1.scene.objs.floor
    assert first_g1.scene.objs.robot is not second_g1.scene.objs.robot
    assert first_g1.scene.objs.robot.key_pose is not second_g1.scene.objs.robot.key_pose


def test_core_facade_reexports_framework_public_api():
    import motrix_envs.core as facade

    assert facade.__all__ == motrix_env_core.__all__
    for name in motrix_env_core.__all__:
        assert getattr(facade, name) is getattr(motrix_env_core, name)
