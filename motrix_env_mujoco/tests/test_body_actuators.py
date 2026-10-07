# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.actuator import ActuatorCfg, MotorActuatorCfg, PositionActuatorCfg
from motrix_env_core.config.scene.base import BodyCfg, SceneCfg, SceneObjsCfg
from motrix_env_core.config.scene.mjcf import MjcfFileCfg
from motrix_env_core.config.scene.urdf import JointCfg, SiteCfg, UrdfFileCfg

mj = pytest.importorskip("mujoco")

from motrix_env_mujoco.actuators import apply_actuator_cfgs  # noqa: E402
from motrix_env_mujoco.compiler import build_mujoco_model  # noqa: E402


@pytest.fixture
def mjcf_file(tmp_path):
    path = tmp_path / "tiny.xml"
    path.write_text(
        """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <joint name="hinge" range="-1 2"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="tip" pos="0 0 0.2">
        <joint name="other" range="-2 3"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="source" joint="hinge" gear="3" ctrlrange="-4 4"/>
    <general name="dynamic" joint="other" dyntype="filter" dynprm="0.1"/>
  </actuator>
  <keyframe><key name="home" qpos="0.2 0.3" qvel="0.4 0.5" ctrl="1 2" act="0.7"/></keyframe>
</mujoco>""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def urdf_file(tmp_path):
    path = tmp_path / "tiny.urdf"
    path.write_text(
        """<robot name="tiny">
  <mujoco><compiler discardvisual="false" fusestatic="false"/></mujoco>
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="tip">
    <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><sphere radius="0.1"/></geometry></collision>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="2" effort="10" velocity="5"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )
    return path


def _scene(*bodies):
    @configclass
    class Objects(SceneObjsCfg):
        first: BodyCfg = bodies[0]
        second: BodyCfg | None = bodies[1] if len(bodies) > 1 else None

    return SceneCfg(objs=Objects())


def _body(model, actuators=None, prefix="", suffix=""):
    return BodyCfg(model=model, base_link_name="base", actuators=actuators, prefix=prefix, suffix=suffix)


def test_mjcf_preserve_source_actuators_and_root_model(mjcf_file):
    source = mj.MjSpec.from_file(str(mjcf_file)).compile()
    for scene in (SceneCfg(file=mjcf_file), _scene(_body(MjcfFileCfg(file=mjcf_file)))):
        model = build_mujoco_model(scene)
        assert model.nu == 2
        np.testing.assert_array_equal(model.actuator_gear, source.actuator_gear)
        np.testing.assert_array_equal(model.actuator_dyntype, source.actuator_dyntype)
        np.testing.assert_allclose(model.key_ctrl, [[1, 2]])
        np.testing.assert_allclose(model.key_act, [[0.7]])


@pytest.mark.parametrize("actuators", [{}, {"source": MotorActuatorCfg(joint_name="hinge")}])
def test_mjcf_replacement_resets_keyframe_controls_and_activations(mjcf_file, actuators):
    model = build_mujoco_model(_scene(_body(MjcfFileCfg(file=mjcf_file), actuators)))
    assert model.nu == len(actuators)
    assert model.na == 0
    assert model.nkey == 1
    np.testing.assert_allclose(model.key_ctrl, np.zeros((1, len(actuators))))
    assert model.key_act.shape == (1, 0)
    np.testing.assert_allclose(model.key_qpos, [[0.2, 0.3]])
    np.testing.assert_allclose(model.key_qvel, [[0.4, 0.5]])


def test_position_and_motor_parameters(mjcf_file):
    actuators = {
        "servo": PositionActuatorCfg(joint_name="hinge", kp=12, kv=3, inherit_joint_range=True, force_range=(-8, 9)),
        "torque": MotorActuatorCfg(joint_name="other", ctrl_range=(-4, 5), force_range=(-6, 7)),
    }
    model = build_mujoco_model(_scene(_body(MjcfFileCfg(file=mjcf_file), actuators)))
    assert [model.actuator(i).name for i in range(model.nu)] == ["servo", "torque"]
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], [12, 1])
    np.testing.assert_allclose(model.actuator_biasprm[:, :3], [[0, -12, -3], [0, 0, 0]])
    np.testing.assert_array_equal(model.actuator_biastype, [mj.mjtBias.mjBIAS_AFFINE, mj.mjtBias.mjBIAS_NONE])
    np.testing.assert_array_equal(model.actuator_dyntype, [mj.mjtDyn.mjDYN_NONE] * 2)
    np.testing.assert_array_equal(model.actuator_gaintype, [mj.mjtGain.mjGAIN_FIXED] * 2)
    np.testing.assert_array_equal(model.actuator_trntype, [mj.mjtTrn.mjTRN_JOINT] * 2)
    np.testing.assert_allclose(model.actuator_gear, [[1, 0, 0, 0, 0, 0]] * 2)
    np.testing.assert_allclose(model.actuator_ctrlrange, [[-1, 2], [-4, 5]])
    np.testing.assert_allclose(model.actuator_forcerange, [[-8, 9], [-6, 7]])
    assert np.all(model.actuator_ctrllimited)
    assert np.all(model.actuator_forcelimited)


@pytest.mark.parametrize("file_kind", ["mjcf", "urdf"])
def test_two_instances_prefix_suffix_without_shared_mutation(mjcf_file, urdf_file, file_kind):
    source = MjcfFileCfg(file=mjcf_file) if file_kind == "mjcf" else UrdfFileCfg(file=urdf_file)
    scene = _scene(
        _body(source, {"drive": MotorActuatorCfg(joint_name="hinge")}, prefix="left_", suffix="_a"),
        _body(source, {"drive": PositionActuatorCfg(joint_name="hinge", kp=7)}, prefix="right_", suffix="_b"),
    )
    for _ in range(2):
        model = build_mujoco_model(scene)
        assert model.nu == 2
        for index, (prefix, suffix) in enumerate((("left_", "_a"), ("right_", "_b"))):
            assert model.actuator(index).name == f"{prefix}drive{suffix}"
            joint = model.joint(int(model.actuator_trnid[index, 0]))
            assert joint.name == f"{prefix}hinge{suffix}"
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], [1, 7])
    original = mj.MjSpec.from_file(str(source.file)).compile()
    assert original.nu == (2 if file_kind == "mjcf" else 0)


@pytest.mark.parametrize(
    "actuators", [None, {}, {"servo": PositionActuatorCfg(joint_name="hinge", kp=5, ctrl_range=(-0.5, 0.8))}]
)
def test_urdf_overrides_follow_joint_and_site_augmentation(urdf_file, actuators):
    source = UrdfFileCfg(
        file=urdf_file,
        joints=[JointCfg(joint_name="hinge", armature=0.04)],
        sites=[SiteCfg(name="marker", parent_link_name="tip")],
    )
    model = build_mujoco_model(_scene(_body(source, actuators, prefix="u_")))
    assert model.nu == (0 if not actuators else 1)
    assert model.site("u_marker").id >= 0
    np.testing.assert_allclose(model.dof_armature, [0.04])
    if actuators:
        np.testing.assert_allclose(model.actuator_ctrlrange, [[-0.5, 0.8]])


@pytest.mark.parametrize(
    ("actuators", "error", "message"),
    [
        (
            {"first": MotorActuatorCfg(joint_name="hinge"), "second": MotorActuatorCfg(joint_name="hinge")},
            ValueError,
            "same joint",
        ),
        (
            {"": MotorActuatorCfg(joint_name="hinge")},
            ValueError,
            "non-empty",
        ),
        ({"motor": MotorActuatorCfg(joint_name="missing")}, ValueError, "do not exist"),
        ({"unsupported": ActuatorCfg(joint_name="hinge")}, TypeError, "does not support"),
    ],
)
def test_invalid_replacement_does_not_mutate_spec(mjcf_file, actuators, error, message):
    spec = mj.MjSpec.from_file(str(mjcf_file))
    before = spec.to_xml()
    with pytest.raises(error, match=message):
        apply_actuator_cfgs(spec, actuators)
    assert spec.to_xml() == before
    assert spec.compile().nu == 2


@pytest.mark.parametrize("compiler", ["", '<compiler angle="degree"/>', '<compiler angle="radian"/>'])
def test_inherited_joint_range_uses_compiled_units_without_mutating_source(compiler):
    spec = mj.MjSpec.from_string(
        f"""<mujoco>{compiler}<worldbody><body name="base">
        <joint name="hinge" type="hinge" range="-90 90"/>
        <joint name="slider" type="slide" range="-2 3"/>
        <geom size="0.1"/></body></worldbody></mujoco>"""
    )
    original_ranges = {joint.name: list(joint.range) for joint in spec.joints}
    apply_actuator_cfgs(
        spec,
        {name: PositionActuatorCfg(joint_name=name, kp=1.0, inherit_joint_range=True) for name in original_ranges},
    )
    model = spec.compile()
    for index, name in enumerate(original_ranges):
        np.testing.assert_allclose(model.actuator_ctrlrange[index], model.joint(name).range)
        np.testing.assert_allclose(spec.joint(name).range, original_ranges[name])
    np.testing.assert_allclose(model.actuator_ctrlrange[1], [-2, 3])


def test_source_defaults_do_not_leak_into_canonical_actuators():
    spec = mj.MjSpec.from_string(
        """<mujoco><default><general gear="4" dyntype="filter" dynprm="0.1"
        actlimited="true" actrange="-3 3" ctrllimited="true" ctrlrange="-2 2"
        forcelimited="true" forcerange="-5 5" gainprm="8" biasprm="1 2 3"/></default>
        <worldbody><body name="base"><joint name="hinge"/><geom size="0.1"/></body></worldbody></mujoco>"""
    )
    apply_actuator_cfgs(spec, {"motor": MotorActuatorCfg(joint_name="hinge")})
    model = spec.compile()
    assert model.na == 0
    assert not model.actuator_ctrllimited[0]
    assert not model.actuator_forcelimited[0]
    np.testing.assert_allclose(model.actuator_gear[0], [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(model.actuator_gainprm[0], [1] + [0] * 9)
    np.testing.assert_allclose(model.actuator_biasprm[0], [0] * 10)
