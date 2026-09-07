# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import motrixsim as mtx
import pytest

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    KeyPoseCfg,
    MjcfFileCfg,
    RobotCfg,
    SceneCfg,
    SceneObjsCfg,
    SiteCfg,
    UrdfFileCfg,
)
from motrix_env_motrixsim.compiler import build_scene_model
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import QuadrupedSceneCfg
from motrix_envs.robot import (
    AnymalC,
    BoosterK1,
    DexEvt,
    HumanoidRobotCfg,
    Microduck,
    QuadrupedLegCfg,
    QuadrupedRobotCfg,
    UnitreeG129Dof,
    UnitreeGo1Robot,
    UnitreeGo2Robot,
)
from motrix_envs.robot.anymal import ANYMAL_C_ASSET_DIR
from motrix_envs.robot.booster import BOOSTER_K1_ASSET_DIR
from motrix_envs.robot.dex_evt import DEX_EVT_ASSET_DIR
from motrix_envs.robot.microduck import MICRODUCK_ASSET_DIR
from motrix_envs.robot.unitree import UNITREE_G1_ASSET_DIR, UNITREE_GO1_ASSET_DIR, UNITREE_GO2_ASSET_DIR


@configclass
class K1SceneObjsCfg(SceneObjsCfg):
    robot: BoosterK1 = BoosterK1()


@configclass
class G1SceneObjsCfg(SceneObjsCfg):
    robot: UnitreeG129Dof = UnitreeG129Dof()


@configclass
class Go1SceneObjsCfg(SceneObjsCfg):
    robot: UnitreeGo1Robot = UnitreeGo1Robot()


@configclass
class Go2SceneObjsCfg(SceneObjsCfg):
    robot: UnitreeGo2Robot = UnitreeGo2Robot()


@configclass
class DexEvtSceneObjsCfg(SceneObjsCfg):
    robot: DexEvt = DexEvt()


@configclass
class AnymalCSceneObjsCfg(SceneObjsCfg):
    robot: AnymalC = AnymalC()


@configclass
class MicroduckSceneObjsCfg(SceneObjsCfg):
    robot: Microduck = Microduck()


def test_builtin_robot_configs_are_registered():
    expected_types = {
        "anymal_c": AnymalC,
        "dex-evt": DexEvt,
        "g1-29dof": UnitreeG129Dof,
        "go1": UnitreeGo1Robot,
        "go2": UnitreeGo2Robot,
        "k1": BoosterK1,
        "microduck": Microduck,
    }

    assert set(expected_types).issubset(registry.list_registered_robots())
    for name, expected_type in expected_types.items():
        assert isinstance(registry.make_robot_config(name), expected_type)


def test_builtin_robot_configs_use_robot_only_files():
    anymal_c = AnymalC()
    k1 = BoosterK1()
    g1 = UnitreeG129Dof()
    go1 = UnitreeGo1Robot()
    go2 = UnitreeGo2Robot()
    dex_evt = DexEvt()
    microduck = Microduck()

    assert all(isinstance(robot, RobotCfg) for robot in (anymal_c, k1, g1, go1, go2, dex_evt, microduck))
    assert isinstance(anymal_c, QuadrupedRobotCfg)
    assert isinstance(anymal_c.model, MjcfFileCfg)
    assert all(isinstance(robot, HumanoidRobotCfg) for robot in (k1, g1, dex_evt, microduck))
    assert isinstance(k1.model, MjcfFileCfg)
    assert isinstance(g1.model, MjcfFileCfg)
    assert isinstance(go1, QuadrupedRobotCfg)
    assert isinstance(go1.model, MjcfFileCfg)
    assert isinstance(go2, QuadrupedRobotCfg)
    assert isinstance(go2.model, MjcfFileCfg)
    assert isinstance(dex_evt.model, UrdfFileCfg)
    assert Path(anymal_c.model.file).parent == ANYMAL_C_ASSET_DIR
    assert Path(k1.model.file).parent == BOOSTER_K1_ASSET_DIR
    assert Path(g1.model.file).parent == UNITREE_G1_ASSET_DIR
    assert Path(go1.model.file).parent == UNITREE_GO1_ASSET_DIR
    assert Path(go2.model.file).parent == UNITREE_GO2_ASSET_DIR
    assert Path(dex_evt.model.file).parent == DEX_EVT_ASSET_DIR
    assert Path(microduck.model.file).parent == MICRODUCK_ASSET_DIR
    assert Path(microduck.model.file).name == "microduck.xml"
    assert not Path(microduck.model.file).name.startswith("scene_")
    assert Path(k1.model.file).name == "k1_22dof.xml"
    assert Path(g1.model.file).name == "g1_29dof.xml"
    assert Path(go1.model.file).name == "go1_position_actuator.xml"
    assert Path(go2.model.file).name == "go2_mjx.xml"
    assert Path(dex_evt.model.file).name == "dex_evt.urdf"
    assert Path(anymal_c.model.file).name == "anymal_c.xml"
    assert not Path(anymal_c.model.file).name.startswith("scene_")
    assert not Path(k1.model.file).name.startswith("scene_")
    assert not Path(g1.model.file).name.startswith("scene_")
    assert not Path(go1.model.file).name.startswith("scene_")
    assert not Path(go2.model.file).name.startswith("scene_")
    assert not Path(dex_evt.model.file).name.startswith("scene_")


@pytest.mark.parametrize(
    ("robot", "foot_link_names"),
    [
        (UnitreeG129Dof(), ("left_ankle_roll_link", "right_ankle_roll_link")),
        (DexEvt(), ("ankle_roll_l_link", "ankle_roll_r_link")),
        (BoosterK1(), ("left_foot_link", "right_foot_link")),
        (Microduck(), ("ankle_left", "ankle_right")),
    ],
)
def test_builtin_humanoids_expose_foot_link_semantics(robot, foot_link_names):
    robot.validate("robot")

    assert robot.resolved_foot_link_names == foot_link_names

    robot.prefix = "robot_"
    assert robot.resolved_foot_link_names == tuple(f"robot_{name}" for name in foot_link_names)


def test_humanoid_robot_rejects_duplicate_foot_link_names():
    robot = UnitreeG129Dof(right_foot_link_name="left_ankle_roll_link")

    with pytest.raises(ValueError, match="foot link names must be unique"):
        robot.validate("robot")


@pytest.mark.parametrize(
    ("robot", "base_link_name"),
    [
        (UnitreeGo1Robot(), "trunk"),
        (UnitreeGo2Robot(), "base"),
    ],
)
def test_builtin_unitree_quadrupeds_expose_source_independent_semantics(robot, base_link_name):

    assert [cfg_field.name for cfg_field in fields(QuadrupedLegCfg)] == [
        "contact_geom_name",
    ]
    assert robot.base_link_name == base_link_name
    assert robot.foot_contact_geom_names == ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    assert tuple(robot.key_pose.joint_names) == (
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    )
    assert len(robot.key_pose.poses["default"]) == len(robot.key_pose.joint_names)


def test_key_pose_cfg_validates_joint_order_and_named_pose_values():
    key_pose = KeyPoseCfg(
        joint_names=["hip", "knee"],
        poses={
            "standing": [0.0, 0.5],
            "crouching": [0.25, 1.0],
        },
    )

    key_pose.validate()

    with pytest.raises(ValueError, match="must contain 2 joint positions"):
        KeyPoseCfg(joint_names=["hip", "knee"], poses={"invalid": [0.0]}).validate()
    with pytest.raises(ValueError, match="joint_names must be unique"):
        KeyPoseCfg(joint_names=["hip", "hip"], poses={"invalid": [0.0, 0.0]}).validate()
    with pytest.raises(ValueError, match="joint positions must be finite"):
        KeyPoseCfg(joint_names=["hip"], poses={"invalid": [float("inf")]}).validate()


def test_quadruped_leg_model_references_are_optional():
    leg = QuadrupedLegCfg()
    leg.validate()

    assert leg.contact_geom_name is None

    robot = UnitreeGo2Robot(prefix="robot_")
    robot.legs.front_left = leg
    robot.validate("robot")

    assert robot.foot_contact_geom_names == (None, "robot_FR_foot", "robot_RL_foot", "robot_RR_foot")

    with pytest.raises(ValueError, match="requires a contact geom name for every leg"):
        QuadrupedSceneCfg(objs=StandardSceneObjsCfg(robot=robot))


def test_quadruped_scene_derives_contact_sensors_from_robot():
    scene = QuadrupedSceneCfg(
        objs=StandardSceneObjsCfg(
            robot=UnitreeGo2Robot(prefix="robot_"),
        )
    )

    assert list(scene.sensors) == [
        "front_left_contact",
        "front_right_contact",
        "rear_left_contact",
        "rear_right_contact",
    ]
    assert scene.sensors.front_left_contact.geom2 == "robot_FL_foot"
    assert scene.sensors.rear_right_contact.geom2 == "robot_RR_foot"


def test_builtin_k1_robot_builds_model():
    robot = BoosterK1()
    model = build_scene_model(SceneCfg(objs=K1SceneObjsCfg(robot=robot)))

    assert model.body_names == ["Trunk"]
    assert model.get_body("Trunk").floatingbase is not None
    assert "Left_Hip_Pitch" in model.joint_names
    assert "Right_Ankle_Roll" in model.joint_names
    assert model.num_actuators == 22
    assert model.get_site("left_foot").parent_link.name == "left_foot_link"
    assert model.get_site("left_foot").local_pos == pytest.approx([0.026, 0.0, -0.038])
    assert model.get_site("right_foot").parent_link.name == "right_foot_link"
    assert model.get_site("right_foot").local_pos == pytest.approx([0.026, 0.0, -0.038])
    assert all(actuator.ctrl_range is not None for actuator in model.actuators)
    assert all(actuator.force_range is not None for actuator in model.actuators)
    assert [actuator.target_name for actuator in model.actuators] == robot.key_pose.joint_names
    assert len(robot.key_pose.poses["default"]) == len(model.actuators)


def test_builtin_k1_asset_contains_only_referenced_meshes():
    root = ET.parse(BOOSTER_K1_ASSET_DIR / "k1_22dof.xml").getroot()
    referenced_meshes = {mesh.get("file") for mesh in root.findall("./asset/mesh")}
    packaged_meshes = {path.name for path in (BOOSTER_K1_ASSET_DIR / "meshes").iterdir() if path.is_file()}

    assert referenced_meshes == packaged_meshes


def test_builtin_k1_mjcf_names_every_collision_geom():
    root = ET.parse(BOOSTER_K1_ASSET_DIR / "k1_22dof.xml").getroot()
    collision_names = [
        geom.get("name") for geom in root.findall("./worldbody//geom") if geom.get("class") == "collision"
    ]

    assert len(collision_names) == 28
    assert all(collision_names)
    assert len(collision_names) == len(set(collision_names))
    assert root.find("./default/geom").get("solref") == "0.01 1"


def test_builtin_g1_robot_builds_model():
    model = build_scene_model(SceneCfg(objs=G1SceneObjsCfg()))

    assert "pelvis" in model.body_names
    assert "left_hip_pitch_joint" in model.joint_names
    assert "left_foot_contact_point" not in model.link_names
    assert model.get_site("left_foot_contact_point").parent_link.name == "left_ankle_roll_link"
    assert model.get_site("left_foot_contact_point").local_pos == pytest.approx([0.0, 0.0, -0.037])
    assert all(actuator.ctrl_range is not None for actuator in model.actuators)
    assert all(actuator.force_range is not None for actuator in model.actuators)


def test_builtin_go2_robot_builds_model():
    model = build_scene_model(SceneCfg(objs=Go2SceneObjsCfg()))

    assert model.body_names == ["base"]
    assert model.get_body("base").floatingbase is not None
    assert "FL_hip_joint" in model.joint_names
    assert "RR_calf_joint" in model.joint_names
    assert model.num_actuators == 12
    assert model.get_site("FL").parent_link.name == "FL_calf"
    assert model.get_site("RR").parent_link.name == "RR_calf"
    assert all(actuator.ctrl_range is not None for actuator in model.actuators)
    assert all(actuator.force_range is not None for actuator in model.actuators)


def test_builtin_anymal_c_robot_builds_without_navigation_markers():
    model = build_scene_model(SceneCfg(objs=AnymalCSceneObjsCfg()))

    assert model.body_names == ["base"]
    assert model.get_body("base").floatingbase is not None
    assert model.num_actuators == 12
    assert model.get_body("target_marker") is None
    assert model.get_body("robot_heading_arrow") is None
    assert model.get_body("desired_heading_arrow") is None


def test_anymal_c_navigation_scene_owns_navigation_markers():
    env = registry.make("anymal_c_navigation_flat", num_envs=1, mode="train")

    model = build_scene_model(env.cfg.scene, env.cfg.sim)
    assert model.get_body("target_marker").mocap is not None
    assert model.get_body("robot_heading_arrow").mocap is not None
    assert model.get_body("desired_heading_arrow").mocap is not None


def test_builtin_go1_robot_builds_model():
    model = build_scene_model(SceneCfg(objs=Go1SceneObjsCfg()))

    assert model.body_names == ["trunk"]
    assert model.get_body("trunk").floatingbase is not None
    assert "FL_hip_joint" in model.joint_names
    assert "RR_calf_joint" in model.joint_names
    assert model.num_actuators == 12
    assert model.get_site("FL").parent_link.name == "FL_calf"
    assert model.get_site("RR").parent_link.name == "RR_calf"
    data = mtx.SceneData(model)
    assert model.get_sensor_values(["FL_pos", "FR_pos", "RL_pos", "RR_pos"], data).shape == (12,)
    assert all(isinstance(actuator, mtx.PositionActuator) for actuator in model.actuators)
    assert all(actuator.ctrl_range is not None for actuator in model.actuators)


def test_builtin_go2_asset_contains_all_referenced_meshes():
    root = ET.parse(UNITREE_GO2_ASSET_DIR / "go2_mjx.xml").getroot()
    referenced_meshes = {mesh.get("file") for mesh in root.findall("./asset/mesh")}
    packaged_meshes = {f"assets/{path.name}" for path in (UNITREE_GO2_ASSET_DIR / "assets").iterdir() if path.is_file()}

    assert referenced_meshes == packaged_meshes


def test_builtin_microduck_robot_builds_model():
    model = build_scene_model(SceneCfg(objs=MicroduckSceneObjsCfg()))

    assert model.body_names == ["trunk_base"]
    assert model.get_body("trunk_base").floatingbase is not None
    assert model.num_actuators == 14
    assert "left_hip_yaw" in model.joint_names
    assert "right_ankle" in model.joint_names
    assert model.get_site("left_foot").parent_link.name == "ankle_left"
    assert model.get_site("right_foot").parent_link.name == "ankle_right"
    assert all(actuator.ctrl_range is not None for actuator in model.actuators)
    assert all(actuator.force_range is not None for actuator in model.actuators)
    assert [actuator.target_name for actuator in model.actuators] == Microduck().key_pose.joint_names
    assert len(Microduck().key_pose.poses["default"]) == model.num_actuators


def test_builtin_microduck_asset_contains_only_referenced_meshes():
    root = ET.parse(MICRODUCK_ASSET_DIR / "microduck.xml").getroot()
    referenced_meshes = {mesh.get("file") for mesh in root.findall("./asset/mesh")}
    packaged_meshes = {path.name for path in (MICRODUCK_ASSET_DIR / "assets").iterdir() if path.is_file()}

    assert referenced_meshes == packaged_meshes


def test_builtin_dex_evt_robot_builds_model():
    model = build_scene_model(SceneCfg(objs=DexEvtSceneObjsCfg()))

    assert "pelvis" in model.body_names
    assert "hip_pitch_l_joint" in model.joint_names
    assert model.get_geom("pelvis_collision") is not None
    assert model.get_geom("waist_pitch_link_collision") is not None
    assert model.get_site("left_foot_contact_point").parent_link.name == "ankle_roll_l_link"
    assert model.get_site("left_foot_contact_point").local_pos == pytest.approx([0.045, 0.0, -0.058])
    assert all(actuator.ctrl_range is not None for actuator in model.actuators)
    assert all(actuator.force_range is not None for actuator in model.actuators)


def test_builtin_dex_evt_urdf_names_every_collision():
    root = ET.parse(DEX_EVT_ASSET_DIR / "dex_evt.urdf").getroot()
    collision_names = [
        collision.get("name") for link in root.findall("link") for collision in link.findall("collision")
    ]

    assert len(collision_names) == 28
    assert all(collision_names)
    assert len(collision_names) == len(set(collision_names))


def test_urdf_robot_rejects_site_with_unknown_parent_link():
    invalid_robot = DexEvt()
    invalid_robot.model.sites = [SiteCfg(name="probe", parent_link_name="missing_link")]

    @configclass
    class InvalidDexEvtSceneObjsCfg(SceneObjsCfg):
        robot: DexEvt = invalid_robot

    with pytest.raises(ValueError, match="Configured site parent links do not exist"):
        build_scene_model(SceneCfg(objs=InvalidDexEvtSceneObjsCfg()))
