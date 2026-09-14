# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""BodyModel assembly contract: permutation, validation, and the init snapshot."""

import numpy as np
import pytest

from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg, RobotCfg
from motrix_env_core.sim.body import assemble_body_model
from motrix_env_core.sim.model import ActuatorSpec, ActuatorType, SimModel

_SPEC = ActuatorSpec(
    name="j1_motor",
    actuator_type=ActuatorType.POSITION,
    target_name="j1",
    ctrl_range=None,
    force_range=None,
)
_J2_SPEC = ActuatorSpec(
    name="j2_motor",
    actuator_type=ActuatorType.POSITION,
    target_name="j2",
    ctrl_range=None,
    force_range=None,
)
_OTHER_SPEC = ActuatorSpec(
    name="slider_motor",
    actuator_type=ActuatorType.POSITION,
    target_name="slider",
    ctrl_range=None,
    force_range=None,
)


def _robot_cfg(**kwargs) -> RobotCfg:
    defaults = dict(
        model=MjcfFileCfg(file="unused.xml"),
        base_link_name="base",
        key_pose=KeyPoseCfg(
            joint_names=["j2", "j1"],
            poses={"default": [0.4, -0.1], "stand": [0.0, 0.0]},
        ),
    )
    defaults.update(kwargs)
    return RobotCfg(**defaults)


def _facts(**kwargs) -> dict:
    defaults = dict(
        name="robot",
        base_link_name="base",
        link_names=("base", "tip"),
        joint_names=("j1", "j2"),
        joint_pos_limits=(np.asarray([-1.0, -2.0]), np.asarray([1.0, 2.0])),
        scene_actuators=(_OTHER_SPEC, _SPEC, _J2_SPEC),
        init_base_position=np.asarray([0.0, 0.0, 0.8]),
        init_base_quat=np.asarray([0.0, 0.0, 0.0, 1.0]),
        init_link_positions=np.zeros((2, 3), dtype=np.float32),
        init_link_quats=np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
    )
    defaults.update(kwargs)
    return defaults


def test_assemble_permutes_init_key_pose_into_body_joint_order():
    body = assemble_body_model(robot_cfg=_robot_cfg(), **_facts())

    # KeyPoseCfg declares (j2, j1); the body order is (j1, j2).
    np.testing.assert_allclose(body.init_joint_pos, [-0.1, 0.4])


def test_assemble_uses_designated_init_key_pose():
    body = assemble_body_model(robot_cfg=_robot_cfg(init_key_pose="stand"), **_facts())

    np.testing.assert_allclose(body.init_joint_pos, [0.0, 0.0])


def test_assemble_resolves_key_pose_joint_names():
    cfg = _robot_cfg(
        base_link_name="base",
        prefix="r_",
        key_pose=KeyPoseCfg(joint_names=["j2", "j1"], poses={"default": [0.4, -0.1]}),
    )
    body = assemble_body_model(robot_cfg=cfg, **_facts(joint_names=("r_j1", "r_j2")))

    np.testing.assert_allclose(body.init_joint_pos, [-0.1, 0.4])


def test_assemble_derives_body_actuators_from_scene_order():
    body = assemble_body_model(robot_cfg=_robot_cfg(), **_facts())

    # Specs targeting this body's joints, keeping the scene-wide order; the
    # off-body "slider" actuator is excluded.
    assert body.actuators == (_SPEC, _J2_SPEC)


def test_assemble_prop_body_bakes_zero_init_joints_and_empty_actuator_view():
    body = assemble_body_model(
        robot_cfg=None,
        **_facts(joint_names=(), joint_pos_limits=None),
    )

    assert body.init_joint_pos.shape == (0,)
    assert body.actuators == ()


def test_assemble_rejects_unknown_key_pose_joint():
    cfg = _robot_cfg(key_pose=KeyPoseCfg(joint_names=["j1", "ghost"], poses={"default": [0.0, 0.0]}))

    with pytest.raises(ValueError, match="unknown to the body.*ghost"):
        assemble_body_model(robot_cfg=cfg, **_facts())


def test_assemble_rejects_missing_init_key_pose():
    cfg = _robot_cfg(init_key_pose="crouch")

    with pytest.raises(ValueError, match="has no key pose 'crouch'"):
        assemble_body_model(robot_cfg=cfg, **_facts())


def test_assemble_rejects_key_pose_not_covering_body_joints():
    cfg = _robot_cfg(key_pose=KeyPoseCfg(joint_names=["j1"], poses={"default": [0.0]}))

    with pytest.raises(ValueError, match="must cover every joint.*j2"):
        assemble_body_model(robot_cfg=cfg, **_facts())


def test_assemble_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="init_link_positions"):
        assemble_body_model(robot_cfg=None, **_facts(init_link_positions=np.zeros((3, 3))))
    with pytest.raises(ValueError, match="init_link_quats"):
        assemble_body_model(robot_cfg=None, **_facts(init_link_quats=np.zeros((2, 3))))
    with pytest.raises(ValueError, match="init_base_position"):
        assemble_body_model(robot_cfg=None, **_facts(init_base_position=np.zeros(4)))
    with pytest.raises(ValueError, match="init_base_quat"):
        assemble_body_model(robot_cfg=None, **_facts(init_base_quat=np.zeros(3)))
    with pytest.raises(ValueError, match="joint_pos_limits"):
        assemble_body_model(
            robot_cfg=None,
            **_facts(joint_pos_limits=(np.zeros(2), np.zeros(3))),
        )


def test_robot_cfg_validate_rejects_missing_init_key_pose(tmp_path):
    model_file = tmp_path / "robot.xml"
    model_file.touch()

    with pytest.raises(ValueError, match="init_key_pose 'crouch'"):
        _robot_cfg(model=MjcfFileCfg(file=model_file), init_key_pose="crouch").validate("robot")


def test_sim_model_defaults_to_empty_bodies():
    model = SimModel(actuators=(), init_dof_pos=np.zeros(0, dtype=np.float32))

    assert model.bodies == {}
