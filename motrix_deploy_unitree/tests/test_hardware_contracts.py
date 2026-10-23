# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hardware-owned configuration, wiring, and dependency-boundary contracts."""

import subprocess
import sys
import textwrap

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from motrix_deploy_unitree.config.go2 import UnitreeGo2HardwareCfg
from motrix_deploy_unitree.config.hardware import (
    Actuation,
    HardwareRobotCfg,
    UnitreeMotorBinding,
)
from motrix_env_core.config.scene.base import KeyPoseCfg


def test_hardware_preserves_motor_mapping() -> None:
    robot = UnitreeGo2HardwareCfg()
    robot.validate()
    assert type(robot.key_pose) is KeyPoseCfg
    assert "default" in robot.key_pose.poses
    assert "lie_down" in robot.key_pose.poses
    # SDK motor addresses and enabled modes are hardware contracts.
    assert [robot.actuation.motors[name].index for name in robot.key_pose.joint_names] == [
        3,
        4,
        5,
        0,
        1,
        2,
        9,
        10,
        11,
        6,
        7,
        8,
    ]
    assert all(binding.enabled_mode == 0x0A for binding in robot.actuation.motors.values())
    assert set(robot.actuation.motors) == set(robot.key_pose.joint_names)


def test_hardware_control_contract_rejects_invalid_pose_and_limits() -> None:
    robot = UnitreeGo2HardwareCfg()
    spec = robot.to_robot_spec()
    assert spec.joint_names == tuple(robot.key_pose.joint_names)
    with pytest.raises(ValueError, match="unknown hold pose"):
        robot.to_robot_spec("unknown")
    robot.key_pose.poses["outside"] = [float(value) for value in spec.position_upper + 1.0]
    with pytest.raises(ValueError, match="inside position range"):
        robot.to_robot_spec("outside")
    robot.position_lower = [0.0]
    with pytest.raises(ValueError, match="shape"):
        robot.to_robot_spec()


def test_hardware_defaults_are_instance_local() -> None:
    first = UnitreeGo2HardwareCfg()
    second = UnitreeGo2HardwareCfg()
    assert first == second
    assert first.key_pose.joint_names is not second.key_pose.joint_names
    assert first.key_pose.poses is not second.key_pose.poses
    for pose_name in first.key_pose.poses:
        assert first.key_pose.poses[pose_name] is not second.key_pose.poses[pose_name]
        first.key_pose.poses[pose_name][0] += 1.0
    first.key_pose.joint_names[0] = "custom_joint"
    first.actuation.motors.clear()
    assert second == UnitreeGo2HardwareCfg()


@pytest.mark.parametrize("source", [UnitreeGo2HardwareCfg, UnitreeGo2HardwareCfg()])
def test_hardware_structured_configuration_overrides_round_trip(source) -> None:
    robot = UnitreeGo2HardwareCfg()
    cfg = OmegaConf.structured(source)
    cfg.key_pose.poses.default[0] += 1.0
    restored = OmegaConf.to_object(cfg)
    restored.validate()
    assert type(restored) is UnitreeGo2HardwareCfg
    assert type(restored.key_pose) is KeyPoseCfg
    assert restored.key_pose.poses["default"][0] == robot.key_pose.poses["default"][0] + 1.0
    assert robot == UnitreeGo2HardwareCfg()


def test_hardware_hydra_composition_and_instantiation() -> None:
    schema_name = "hardware_wiring_test_schema"
    ConfigStore.instance().store(name=schema_name, node=UnitreeGo2HardwareCfg)
    first_joint = UnitreeGo2HardwareCfg().key_pose.joint_names[0]
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name=schema_name)
    # Frozen actuation mappings are replaced as typed values, not mutated field-by-field.
    motors = UnitreeGo2HardwareCfg().actuation.motors.copy()
    motors[first_joint] = UnitreeMotorBinding(motors[first_joint].index, enabled_mode=1)
    cfg.actuation = Actuation(motors=motors)
    restored = OmegaConf.to_object(cfg)
    restored.validate()
    assert isinstance(restored.actuation, Actuation)
    assert restored.actuation.motors[first_joint] == UnitreeMotorBinding(
        UnitreeGo2HardwareCfg().actuation.motors[first_joint].index, enabled_mode=1
    )
    composed = instantiate(
        {"_target_": "motrix_deploy_unitree.config.go2.UnitreeGo2HardwareCfg", "actuation": cfg.actuation},
        _convert_="object",
    )
    composed.validate()
    assert composed.actuation == restored.actuation
    assert composed.key_pose == restored.key_pose


def test_custom_asset_free_hardware_key_pose() -> None:
    key_pose = KeyPoseCfg(joint_names=["hip"], poses={"default": [0.0]})
    robot = HardwareRobotCfg(
        key_pose=key_pose,
        actuation=Actuation(motors={"hip": UnitreeMotorBinding(0)}),
    )
    robot.validate()
    assert robot.key_pose is key_pose
    cfg = OmegaConf.structured(robot)
    cfg.key_pose.poses.default[0] = 0.1
    restored = OmegaConf.to_object(cfg)
    restored.validate()
    assert restored.key_pose.poses["default"] == [0.1]
    assert robot.key_pose.poses["default"] == [0.0]
    assert restored.actuation == robot.actuation


def test_key_pose_validates_canonical_order() -> None:
    with pytest.raises(ValueError, match="unique"):
        HardwareRobotCfg(key_pose=KeyPoseCfg(joint_names=["hip", "hip"])).validate()
    with pytest.raises(ValueError, match="contain 1"):
        KeyPoseCfg(joint_names=["hip"], poses={"default": [0.0, 1.0]}).validate()


def test_actuation_rejects_duplicate_motor_indices_and_unknown_joints() -> None:
    with pytest.raises(ValueError, match="unique"):
        Actuation(motors={"a": UnitreeMotorBinding(1), "b": UnitreeMotorBinding(1)})
    with pytest.raises(ValueError, match="unknown canonical joints"):
        HardwareRobotCfg(actuation=Actuation(motors={"unknown": UnitreeMotorBinding(0)})).validate()


@pytest.mark.integration
def test_hardware_configuration_imports_without_models_assets_or_sdk() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys
        from pathlib import Path

        blocked = ('motrix_robots', 'motrix_envs', 'motrixsim', 'mujoco',
                   'torch', 'onnx', 'onnxruntime', 'unitree_sdk2py')

        class BlockHardwareRuntimeDependencies(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
                    raise ImportError('hardware configuration cannot import ' + fullname)

        sys.meta_path.insert(0, BlockHardwareRuntimeDependencies())
        importlib.import_module('motrix_deploy_unitree.config.go2')
        from motrix_deploy_unitree.config.runtime import UnitreeTransportConfig
        from motrix_deploy_unitree.config.go2 import UnitreeGo2HardwareCfg

        def forbidden_exists(self):
            raise AssertionError('hardware must not check asset path existence')

        Path.exists = forbidden_exists
        hardware = UnitreeGo2HardwareCfg()
        hardware.validate()
        config = UnitreeTransportConfig(network_interface='eth0')
        assert tuple(hardware.key_pose.joint_names)
        assert not any(name == root or name.startswith(root + '.')
                       for name in sys.modules for root in blocked)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
