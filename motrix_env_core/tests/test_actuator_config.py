# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Format-neutral actuator declarations belong to model instances."""

import pytest
from omegaconf import OmegaConf

from motrix_env_core.config.scene.actuator import MotorActuatorCfg, PositionActuatorCfg
from motrix_env_core.config.scene.base import BodyCfg, RobotCfg
from motrix_env_core.config.scene.mjcf import MjcfFileCfg


def test_body_actuator_defaults_and_instance_isolation():
    model = MjcfFileCfg(file="unused.xml")
    preserved = BodyCfg(model=model, base_link_name="base")
    removed = BodyCfg(model=model, base_link_name="base", actuators={})
    assert preserved.actuators is None
    assert removed.actuators == {}
    removed.actuators["motor"] = MotorActuatorCfg(joint_name="hinge")
    assert BodyCfg(model=model, base_link_name="base", actuators={}).actuators == {}
    robot = RobotCfg(model=model, base_link_name="base", actuators={"motor": MotorActuatorCfg(joint_name="hinge")})
    assert robot.actuators["motor"].joint_name == "hinge"


@pytest.mark.parametrize("kp,kv", [(0.0, 0.0), (-1.0, 0.0), (float("nan"), 0.0), (1.0, -1.0), (1.0, float("inf"))])
def test_position_actuator_validates_finite_gains(kp, kv):
    with pytest.raises(ValueError):
        PositionActuatorCfg(joint_name="hinge", kp=kp, kv=kv).validate()


def test_named_actuator_override_preserves_other_fields():
    body = BodyCfg(
        model=MjcfFileCfg(file="unused.xml"),
        base_link_name="base",
        actuators={"servo": PositionActuatorCfg(joint_name="hinge", kp=10.0, kv=1.0)},
    )
    cfg = OmegaConf.merge(OmegaConf.structured(body), {"actuators": {"servo": {"kp": 20.0}}})
    overridden = OmegaConf.to_object(cfg)
    assert overridden.actuators["servo"].kp == 20.0
    assert overridden.actuators["servo"].kv == 1.0
    assert overridden.actuators["servo"].joint_name == "hinge"
    assert body.actuators["servo"].kp == 10.0


def test_position_actuator_range_sources_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PositionActuatorCfg(joint_name="hinge", kp=1.0, inherit_joint_range=True, ctrl_range=(-1.0, 1.0)).validate()
