# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from importlib import import_module

import pytest
from hydra.utils import instantiate
from motrix_robots.anymal import AnymalC
from motrix_robots.booster import BoosterK1
from motrix_robots.dex_evt import DexEvt
from motrix_robots.microduck import Microduck
from motrix_robots.unitree import UnitreeG129Dof, UnitreeGo1Robot, UnitreeGo2Robot
from omegaconf import OmegaConf

import motrix_robots
from motrix_env_core.config.scene import KeyPoseCfg
from motrix_env_core.config.scene.actuator import MotorActuatorCfg
from motrix_env_core.config.scene.base import RobotCfg

MODEL_TYPES = [AnymalC, BoosterK1, DexEvt, Microduck, UnitreeG129Dof, UnitreeGo1Robot, UnitreeGo2Robot]


@pytest.mark.parametrize(
    ("module_name", "type_name"),
    [
        ("anymal", "AnymalC"),
        ("booster", "BoosterK1"),
        ("dex_evt", "DexEvt"),
        ("humanoid", "HumanoidRobotCfg"),
        ("microduck", "Microduck"),
        ("quadruped", "QuadrupedLegCfg"),
        ("quadruped", "QuadrupedLegsCfg"),
        ("quadruped", "QuadrupedRobotCfg"),
        ("unitree", "UnitreeG129Dof"),
        ("unitree", "UnitreeGo1Robot"),
        ("unitree", "UnitreeGo2Robot"),
    ],
)
def test_public_types_are_their_defining_module_types(module_name, type_name) -> None:
    defining_module = import_module(f"motrix_robots.{module_name}")
    assert getattr(motrix_robots, type_name) is getattr(defining_module, type_name)
    assert type_name in motrix_robots.__all__


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_public_model_constructors_return_fresh_configurations(model_type) -> None:
    public_type = getattr(motrix_robots, model_type.__name__)
    first = public_type()
    second = public_type()
    first.validate("robot")
    second.validate("robot")
    assert isinstance(first, model_type)
    assert first is not second
    assert first.model is not second.model
    assert first.key_pose is not second.key_pose


@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("copy_kind", ["fresh", "deepcopy", "omegaconf"])
def test_model_owned_key_poses_are_valid_and_isolated(model_type, copy_kind) -> None:
    robot = model_type()
    assert isinstance(robot, RobotCfg)
    assert isinstance(robot.key_pose, KeyPoseCfg)
    robot.validate("robot")
    assert robot.key_pose.joint_names
    assert robot.init_key_pose in robot.key_pose.poses

    cfg = OmegaConf.structured(robot) if copy_kind == "omegaconf" else None
    if copy_kind == "fresh":
        copied = model_type()
    elif copy_kind == "deepcopy":
        copied = deepcopy(robot)
    else:
        copied = OmegaConf.to_object(cfg)
    assert isinstance(copied, model_type)
    copied.validate("robot")
    assert copied.key_pose == robot.key_pose
    assert copied.key_pose is not robot.key_pose
    assert copied.key_pose.joint_names is not robot.key_pose.joint_names
    assert copied.key_pose.poses is not robot.key_pose.poses
    for name, pose in copied.key_pose.poses.items():
        assert pose is not robot.key_pose.poses[name]
        pose[0] += 1.0
    copied.key_pose.joint_names[0] = "custom_joint"
    copied.key_pose.poses["custom"] = list(copied.key_pose.poses[copied.init_key_pose])
    assert robot.key_pose == model_type().key_pose
    if cfg is not None:
        assert OmegaConf.to_object(cfg).key_pose == robot.key_pose


def test_builtin_model_inherits_scene_placement_and_actuator_contract() -> None:
    actuator = MotorActuatorCfg(joint_name="hinge", force_range=(-2.0, 2.0))
    robot = UnitreeGo2Robot(
        prefix="robot_",
        suffix="_1",
        translation=(1.0, 2.0, 3.0),
        actuators={"hinge_motor": actuator},
    )
    assert isinstance(robot, RobotCfg)
    assert robot.resolved_base_link_name == robot.resolve_name(robot.base_link_name)
    assert robot.resolve_name(actuator.joint_name) == "robot_hinge_1"
    restored = OmegaConf.to_object(OmegaConf.structured(robot))
    restored.validate("robot")
    assert tuple(restored.translation) == robot.translation
    assert isinstance(restored.actuators["hinge_motor"], MotorActuatorCfg)
    assert restored.actuators["hinge_motor"].joint_name == actuator.joint_name
    assert tuple(restored.actuators["hinge_motor"].force_range) == actuator.force_range
    restored.actuators["hinge_motor"].joint_name = "custom_hinge"
    assert robot.actuators["hinge_motor"].joint_name == "hinge"


def test_scene_pose_override_and_hydra_composition() -> None:
    override = KeyPoseCfg(joint_names=["hinge"], poses={"default": [0.2]})
    robot = UnitreeGo2Robot(key_pose=override)
    assert robot.key_pose is override
    robot.validate("robot")
    assert OmegaConf.to_object(OmegaConf.structured(robot)).key_pose == override
    composed = instantiate(
        {
            "_target_": "motrix_robots.unitree.UnitreeGo2Robot",
            "key_pose": {
                "_target_": "motrix_env_core.config.scene.KeyPoseCfg",
                "joint_names": ["hinge"],
                "poses": {"default": [0.3]},
            },
        }
    )
    assert composed.key_pose == KeyPoseCfg(joint_names=["hinge"], poses={"default": [0.3]})
    composed.validate("robot")


def test_omegaconf_custom_key_pose_override() -> None:
    original = UnitreeGo2Robot()
    cfg = OmegaConf.structured(original)
    cfg.key_pose = OmegaConf.structured(KeyPoseCfg(joint_names=["custom_hinge"], poses={"custom": [0.4]}))
    cfg.init_key_pose = "custom"
    restored = OmegaConf.to_object(cfg)
    assert isinstance(restored, UnitreeGo2Robot)
    assert restored.key_pose == KeyPoseCfg(joint_names=["custom_hinge"], poses={"custom": [0.4]})
    restored.validate("robot")
    restored.key_pose.poses["custom"][0] += 1.0
    assert original.key_pose == UnitreeGo2Robot().key_pose
    assert cfg.key_pose.poses.custom == [0.4]


def test_explicit_empty_key_pose_is_preserved() -> None:
    empty = KeyPoseCfg()
    robot = UnitreeGo2Robot(key_pose=empty)
    assert robot.key_pose is empty
    assert robot.key_pose == KeyPoseCfg()
    robot.validate("robot")
    restored = OmegaConf.to_object(OmegaConf.structured(robot))
    assert restored.key_pose == empty
    restored.validate("robot")
