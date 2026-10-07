# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hardware-owned configuration, wiring, and dependency-boundary contracts."""

import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, ConfigKeyError, ReadonlyConfigError

from motrix_deploy_unitree.actuation import Actuation, UnitreeMotorBinding
from motrix_deploy_unitree.hardware import HardwareRobotCfg, UnitreeGo2HardwareCfg
from motrix_deploy_unitree.sensor import HardwareSensorBinding, HardwareSensorBindings
from motrix_env_core.config.scene.base import KeyPoseCfg


def test_hardware_preserves_sensor_and_motor_mapping() -> None:
    robot = UnitreeGo2HardwareCfg()
    robot.validate()
    assert type(robot.key_pose) is KeyPoseCfg
    assert "default" in robot.key_pose.poses
    assert "lie_down" in robot.key_pose.poses
    assert robot.sensors == HardwareSensorBindings(
        base_orientation_xyzw=HardwareSensorBinding("imu_state.quaternion", quaternion_order="wxyz"),
        base_angular_velocity=HardwareSensorBinding("imu_state.gyroscope"),
        base_linear_acceleration=HardwareSensorBinding("imu_state.accelerometer"),
    )
    # SDK motor addresses, enabled modes, and wire quaternion order are hardware contracts.
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


def test_hardware_defaults_are_instance_local() -> None:
    first = UnitreeGo2HardwareCfg()
    second = UnitreeGo2HardwareCfg()
    assert first == second
    assert first.sensors is not second.sensors
    assert first.sensors.base_angular_velocity is not second.sensors.base_angular_velocity
    assert first.key_pose.joint_names is not second.key_pose.joint_names
    assert first.key_pose.poses is not second.key_pose.poses
    for pose_name in first.key_pose.poses:
        assert first.key_pose.poses[pose_name] is not second.key_pose.poses[pose_name]
        first.key_pose.poses[pose_name][0] += 1.0
    first.sensors.base_angular_velocity = HardwareSensorBinding("custom.gyro")
    first.key_pose.joint_names[0] = "custom_joint"
    first.actuation.motors.clear()
    assert second == UnitreeGo2HardwareCfg()

    first_generic = HardwareRobotCfg()
    second_generic = HardwareRobotCfg()
    first_generic.sensors.base_angular_velocity = HardwareSensorBinding("custom.gyro")
    assert first_generic.sensors is not second_generic.sensors
    assert second_generic.sensors == HardwareSensorBindings()


def test_sensor_binding_leaf_is_frozen_but_role_can_be_replaced() -> None:
    bindings = HardwareSensorBindings(base_angular_velocity=HardwareSensorBinding("imu.gyro"))
    with pytest.raises(FrozenInstanceError):
        bindings.base_angular_velocity.field = "custom.gyro"
    cfg = OmegaConf.structured(bindings)
    with pytest.raises(ReadonlyConfigError):
        cfg.base_angular_velocity.field = "custom.gyro"
    cfg.base_angular_velocity = HardwareSensorBinding("custom.gyro")
    restored = OmegaConf.to_object(cfg)
    assert type(restored) is HardwareSensorBindings
    assert type(restored.base_angular_velocity) is HardwareSensorBinding
    assert restored.base_angular_velocity.field == "custom.gyro"
    assert bindings.base_angular_velocity.field == "imu.gyro"


@pytest.mark.parametrize("source", [UnitreeGo2HardwareCfg, UnitreeGo2HardwareCfg()])
def test_hardware_structured_configuration_overrides_round_trip(source) -> None:
    robot = UnitreeGo2HardwareCfg()
    cfg = OmegaConf.structured(source)
    cfg.key_pose.poses.default[0] += 1.0
    cfg.sensors.base_angular_velocity = HardwareSensorBinding("custom.gyro")
    cfg.sensors.base_linear_acceleration = None
    restored = OmegaConf.to_object(cfg)
    restored.validate()
    assert type(restored) is UnitreeGo2HardwareCfg
    assert type(restored.key_pose) is KeyPoseCfg
    assert type(restored.sensors) is HardwareSensorBindings
    assert restored.key_pose.poses["default"][0] == robot.key_pose.poses["default"][0] + 1.0
    assert restored.sensors == HardwareSensorBindings(
        base_orientation_xyzw=robot.sensors.base_orientation_xyzw,
        base_angular_velocity=HardwareSensorBinding("custom.gyro"),
    )
    assert robot == UnitreeGo2HardwareCfg()


def test_hardware_hydra_composition_and_instantiation() -> None:
    schema_name = "hardware_sensor_bindings_test_schema"
    ConfigStore.instance().store(name=schema_name, node=UnitreeGo2HardwareCfg)
    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name=schema_name,
            overrides=[
                "sensors.base_angular_velocity=null",
                "sensors.base_linear_acceleration=null",
            ],
        )
    restored = OmegaConf.to_object(cfg)
    restored.validate()
    assert type(restored.sensors) is HardwareSensorBindings
    assert restored.sensors == HardwareSensorBindings(
        base_orientation_xyzw=UnitreeGo2HardwareCfg().sensors.base_orientation_xyzw,
    )
    # Frozen leaf mappings are replaced as typed values rather than merged field-by-field.
    cfg.sensors.base_angular_velocity = HardwareSensorBinding("custom.gyro")
    restored = OmegaConf.to_object(cfg)
    assert restored.sensors.base_angular_velocity == HardwareSensorBinding("custom.gyro")
    composed = instantiate(
        {"_target_": "motrix_deploy_unitree.hardware.UnitreeGo2HardwareCfg", "sensors": cfg.sensors},
        _convert_="object",
    )
    composed.validate()
    assert composed.sensors == restored.sensors
    assert type(composed.sensors) is HardwareSensorBindings

    customized = instantiate(
        {
            "_target_": "motrix_deploy_unitree.hardware.HardwareRobotCfg",
            "sensors": {
                "_target_": "motrix_deploy_unitree.sensor.HardwareSensorBindings",
                "base_angular_velocity": {
                    "_target_": "motrix_deploy_unitree.sensor.HardwareSensorBinding",
                    "field": "custom.gyro",
                },
            },
        }
    )
    customized.validate()
    assert customized.sensors == HardwareSensorBindings(
        base_angular_velocity=HardwareSensorBinding("custom.gyro"),
    )
    assert type(customized.sensors.base_angular_velocity) is HardwareSensorBinding


def test_sensor_roles_are_checked_at_structured_configuration_boundary() -> None:
    cfg = OmegaConf.structured(HardwareRobotCfg)
    with pytest.raises(ConfigAttributeError, match="base_angluar_velocity"):
        OmegaConf.update(cfg, "sensors.base_angluar_velocity", {"field": "imu.gyro"})
    with pytest.raises(ConfigKeyError, match="base_angluar_velocity"):
        OmegaConf.merge(cfg, {"sensors": {"base_angluar_velocity": {"field": "imu.gyro"}}})


def test_custom_asset_free_hardware_key_pose() -> None:
    key_pose = KeyPoseCfg(joint_names=["hip"], poses={"default": [0.0]})
    sensors = HardwareSensorBindings(base_angular_velocity=HardwareSensorBinding("imu.gyro"))
    robot = HardwareRobotCfg(
        key_pose=key_pose,
        sensors=sensors,
        actuation=Actuation(motors={"hip": UnitreeMotorBinding(0)}),
    )
    robot.validate()
    assert robot.key_pose is key_pose
    assert robot.sensors is sensors
    assert robot.sensors.base_angular_velocity == HardwareSensorBinding("imu.gyro")
    cfg = OmegaConf.structured(robot)
    cfg.sensors.base_linear_acceleration = HardwareSensorBinding("imu.accelerometer")
    restored = OmegaConf.to_object(cfg)
    restored.validate()
    assert restored.sensors == HardwareSensorBindings(
        base_angular_velocity=HardwareSensorBinding("imu.gyro"),
        base_linear_acceleration=HardwareSensorBinding("imu.accelerometer"),
    )


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


@pytest.mark.parametrize("module", ["config", "sensor"])
def test_hardware_configuration_imports_without_models_assets_or_sdk(module: str) -> None:
    script = textwrap.dedent(
        f"""
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
        importlib.import_module('motrix_deploy_unitree.{module}')
        from motrix_deploy_unitree.config import UnitreeGo2BackendConfig
        from motrix_deploy_unitree.hardware import UnitreeGo2HardwareCfg
        from motrix_deploy_unitree.interface import _load_sdk_bindings

        def forbidden_exists(self):
            raise AssertionError('hardware must not check asset path existence')

        Path.exists = forbidden_exists
        hardware = UnitreeGo2HardwareCfg()
        hardware.validate()
        config = UnitreeGo2BackendConfig(network_interface='eth0', robot=hardware)
        assert config.robot is hardware
        assert config.lie_down_position().shape == (12,)
        assert not any(name == root or name.startswith(root + '.')
                       for name in sys.modules for root in blocked)
        # The SDK is requested only by the explicit transport loader, not imports/configuration.
        try:
            _load_sdk_bindings()
        except RuntimeError as error:
            assert 'Unitree Go2 support requires' in str(error)
            assert isinstance(error.__cause__, ImportError)
        else:
            raise AssertionError('transport loader must request the blocked SDK')
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
