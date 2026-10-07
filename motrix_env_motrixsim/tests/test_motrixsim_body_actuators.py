# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.actuator import ActuatorCfg, MotorActuatorCfg, PositionActuatorCfg
from motrix_env_core.config.scene.base import BodyCfg, SceneCfg, SceneObjsCfg
from motrix_env_core.config.scene.mjcf import MjcfFileCfg
from motrix_env_core.config.scene.urdf import JointCfg, SiteCfg, UrdfFileCfg

mtx = pytest.importorskip("motrixsim")

from motrix_env_motrixsim.actuators import apply_actuator_cfgs  # noqa: E402
from motrix_env_motrixsim.compiler import build_scene_world  # noqa: E402


@pytest.fixture(params=["mjcf", "urdf"])
def source(request, tmp_path):
    if request.param == "mjcf":
        path = tmp_path / "tiny.xml"
        path.write_text("""<mujoco><compiler angle="radian"/>
        <worldbody><body name="base"><joint name="hinge" range="-1 2"/>
        <geom type="sphere" size="0.1" mass="1"/></body></worldbody>
        <actuator><motor name="source" joint="hinge" gear="3" ctrlrange="-4 4"/></actuator>
        <keyframe><key name="home" qpos="0.2" qvel="0.3" ctrl="1"/></keyframe></mujoco>""")
        return MjcfFileCfg(file=path)
    path = tmp_path / "tiny.urdf"
    path.write_text("""<robot name="tiny"><link name="base">
    <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    </link><link name="tip">
    <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><sphere radius="0.1"/></geometry></collision></link>
    <joint name="hinge" type="revolute"><parent link="base"/><child link="tip"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="2" effort="10" velocity="5"/></joint></robot>""")
    return UrdfFileCfg(file=path)


def _scene(*bodies):
    @configclass
    class Objects(SceneObjsCfg):
        first: BodyCfg = bodies[0]
        second: BodyCfg | None = bodies[1] if len(bodies) > 1 else None

    return SceneCfg(objs=Objects())


def _body(source, actuators=None, prefix="", suffix=""):
    return BodyCfg(model=source, base_link_name="base", actuators=actuators, prefix=prefix, suffix=suffix)


def test_preserve_imported_actuators(source):
    world = build_scene_world(_scene(_body(source)))
    assert len(world.actuators) == 1
    actuator = world.actuators[0]
    assert actuator.target.value == "hinge"
    if isinstance(source, MjcfFileCfg):
        assert actuator.name == "source"
        np.testing.assert_allclose(actuator.gear, [3, 0, 0, 0, 0, 0])
        # Native attachment does not transfer keyframes; direct imports preserve them.
        imported = build_scene_world(SceneCfg(file=source.file))
        np.testing.assert_allclose(imported.keyframes[0].ctrl, [1])
    else:
        assert actuator.name == "hinge_motor"
    assert world.build().num_actuators == 1


@pytest.mark.parametrize("actuators", [{}, {"source": MotorActuatorCfg(joint_name="hinge")}])
def test_replace_remove_and_reset_keyframes(source, actuators):
    world = build_scene_world(_scene(_body(source, actuators)))
    assert len(world.actuators) == len(actuators)
    if isinstance(source, MjcfFileCfg):
        imported = build_scene_world(SceneCfg(file=source.file))
        apply_actuator_cfgs(imported, actuators)
        assert len(imported.keyframes) == 1
        key = imported.keyframes[0]
        np.testing.assert_allclose(key.ctrl, np.zeros(len(actuators)))
        np.testing.assert_allclose(key.dof_pos, [0.2])
        np.testing.assert_allclose(key.dof_vel, [0.3])
        assert imported.build().num_actuators == len(actuators)
    model = world.build()
    assert model.num_actuators == len(actuators)


def test_position_and_motor_parameters(source):
    cfg = PositionActuatorCfg(joint_name="hinge", kp=12, kv=3, inherit_joint_range=True, force_range=(-8, 9))
    world = build_scene_world(_scene(_body(source, {"servo": cfg})))
    actuator = world.actuators[0]
    assert actuator.actuator_type.variant == "position"
    assert actuator.actuator_type.value.kp == 12
    assert actuator.actuator_type.value.damping_value == 3
    np.testing.assert_allclose(actuator.gear, [1, 0, 0, 0, 0, 0])
    assert world.build().num_actuators == 1
    world = build_scene_world(_scene(_body(source, {"motor": MotorActuatorCfg(joint_name="hinge")})))
    assert world.actuators[0].actuator_type.variant == "motor"
    assert world.actuators[0].ctrlrange is None
    assert world.actuators[0].forcerange is None
    np.testing.assert_allclose(world.actuators[0].gear, [1, 0, 0, 0, 0, 0])


def test_two_instances_prefix_suffix_without_shared_mutation(source):
    scene = _scene(
        _body(source, {"drive": MotorActuatorCfg(joint_name="hinge")}, "left_", "_a"),
        _body(source, {"drive": PositionActuatorCfg(joint_name="hinge", kp=7)}, "right_", "_b"),
    )
    for _ in range(2):
        world = build_scene_world(scene)
        assert [a.name for a in world.actuators] == ["left_drive_a", "right_drive_b"]
        assert [a.target.value for a in world.actuators] == ["left_hinge_a", "right_hinge_b"]
        assert world.build().num_actuators == 2
    assert len(build_scene_world(_scene(_body(source))).actuators) == 1


@pytest.mark.parametrize("actuators", [None, {}, {"servo": PositionActuatorCfg(joint_name="hinge", kp=5)}])
def test_urdf_augmentation_is_independent(source, actuators):
    if not isinstance(source, UrdfFileCfg):
        pytest.skip("URDF augmentations only")
    source.joints = [JointCfg(joint_name="hinge", armature=0.04)]
    source.sites = [SiteCfg(name="marker", parent_link_name="tip")]
    world = build_scene_world(_scene(_body(source, actuators, prefix="u_")))
    assert len(world.actuators) == (1 if actuators is None else len(actuators))
    model = world.build()
    assert model.get_site("u_marker") is not None


@pytest.mark.parametrize(
    ("actuators", "error", "message"),
    [
        (
            {"first": MotorActuatorCfg(joint_name="hinge"), "second": MotorActuatorCfg(joint_name="hinge")},
            ValueError,
            "same joint",
        ),
        ({"": MotorActuatorCfg(joint_name="hinge")}, ValueError, "must not be empty"),
        ({"motor": MotorActuatorCfg(joint_name="missing")}, ValueError, "do not exist"),
        ({"motor": ActuatorCfg(joint_name="hinge")}, TypeError, "does not support"),
    ],
)
def test_invalid_replacement_does_not_mutate_world(source, actuators, error, message):
    world = build_scene_world(_scene(_body(source)))
    before = [(a.name, a.target.value) for a in world.actuators]
    with pytest.raises(error, match=message):
        apply_actuator_cfgs(world, actuators)
    assert [(a.name, a.target.value) for a in world.actuators] == before
