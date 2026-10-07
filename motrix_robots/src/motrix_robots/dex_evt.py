# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    JointCfg,
    KeyPoseCfg,
    PositionActuatorCfg,
    SiteCfg,
    UrdfFileCfg,
    UrdfGeomCfg,
)
from motrix_robots.humanoid import HumanoidRobotCfg

DEX_EVT_ASSET_DIR = Path(__file__).parent / "assets" / "dex_evt"
_DEX_EVT_URDF = DEX_EVT_ASSET_DIR / "dex_evt.urdf"

_FOOT_COLLISION_GEOM_NAMES = (
    "foot_left_front_outer",
    "foot_left_front_inner",
    "foot_left_mid_outer",
    "foot_left_mid_inner",
    "foot_left_strip_outer",
    "foot_left_strip_center",
    "foot_left_strip_inner",
    "foot_right_front_outer",
    "foot_right_front_inner",
    "foot_right_mid_outer",
    "foot_right_mid_inner",
    "foot_right_strip_outer",
    "foot_right_strip_center",
    "foot_right_strip_inner",
)

_DRIVE_PARAMS = {
    "hip_pitch_l_joint": (300.0, 15.0, 235.0),
    "hip_roll_l_joint": (300.0, 15.0, 235.0),
    "hip_yaw_l_joint": (150.0, 7.5, 150.0),
    "knee_pitch_l_joint": (350.0, 15.0, 400.0),
    "ankle_pitch_l_joint": (30.0, 3.75, 55.0),
    "ankle_roll_l_joint": (16.8, 2.1, 55.0),
    "hip_pitch_r_joint": (300.0, 15.0, 235.0),
    "hip_roll_r_joint": (300.0, 15.0, 235.0),
    "hip_yaw_r_joint": (150.0, 7.5, 150.0),
    "knee_pitch_r_joint": (350.0, 15.0, 400.0),
    "ankle_pitch_r_joint": (30.0, 3.75, 55.0),
    "ankle_roll_r_joint": (16.8, 2.1, 55.0),
    "waist_yaw_joint": (400.0, 7.5, 91.0),
    "waist_roll_joint": (400.0, 15.0, 91.0),
    "waist_pitch_joint": (400.0, 15.0, 91.0),
    "shoulder_pitch_l_joint": (150.0, 7.4, 90.0),
    "shoulder_roll_l_joint": (150.0, 7.4, 90.0),
    "shoulder_yaw_l_joint": (130.0, 5.9, 50.0),
    "elbow_pitch_l_joint": (130.0, 5.9, 50.0),
    "shoulder_pitch_r_joint": (150.0, 7.4, 90.0),
    "shoulder_roll_r_joint": (150.0, 7.4, 90.0),
    "shoulder_yaw_r_joint": (130.0, 5.9, 50.0),
    "elbow_pitch_r_joint": (130.0, 5.9, 50.0),
}


def _dex_evt_joints() -> list[JointCfg]:
    return [JointCfg(joint_name=name, armature=0.03, friction_loss=0.1) for name in _DRIVE_PARAMS]


def _dex_evt_geoms() -> list[UrdfGeomCfg]:
    return [
        UrdfGeomCfg(
            geom_name=name,
            friction=(1.0, 0.02, 0.001),
            condim=3,
            priority=2,
        )
        for name in _FOOT_COLLISION_GEOM_NAMES
    ]


def _dex_evt_sites() -> list[SiteCfg]:
    return [
        SiteCfg(
            name="left_foot_contact_point",
            parent_link_name="ankle_roll_l_link",
            position=(0.045, 0.0, -0.058),
        ),
        SiteCfg(
            name="right_foot_contact_point",
            parent_link_name="ankle_roll_r_link",
            position=(0.045, 0.0, -0.058),
        ),
    ]


def _dex_evt_actuators() -> dict[str, PositionActuatorCfg]:
    return {
        name: PositionActuatorCfg(
            joint_name=name,
            kp=kp,
            kv=kv,
            inherit_joint_range=True,
            force_range=(-effort, effort),
        )
        for name, (kp, kv, effort) in _DRIVE_PARAMS.items()
    }


@configclass(kw_only=True)
class DexEvt(HumanoidRobotCfg):
    """Built-in Dex-EVT URDF robot with simulation drive parameters."""

    model: UrdfFileCfg = UrdfFileCfg(
        file=_DEX_EVT_URDF,
        geoms=_dex_evt_geoms(),
        sites=_dex_evt_sites(),
        joints=_dex_evt_joints(),
    )
    actuators: dict[str, PositionActuatorCfg] = _dex_evt_actuators()
    base_link_name: str = "pelvis"
    left_foot_link_name: str = "ankle_roll_l_link"
    right_foot_link_name: str = "ankle_roll_r_link"
    translation: tuple[float, float, float] | None = (0.0, 0.0, 0.95)
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "hip_pitch_l_joint",
            "hip_roll_l_joint",
            "hip_yaw_l_joint",
            "knee_pitch_l_joint",
            "ankle_pitch_l_joint",
            "ankle_roll_l_joint",
            "hip_pitch_r_joint",
            "hip_roll_r_joint",
            "hip_yaw_r_joint",
            "knee_pitch_r_joint",
            "ankle_pitch_r_joint",
            "ankle_roll_r_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "shoulder_pitch_l_joint",
            "shoulder_roll_l_joint",
            "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint",
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
        ],
        poses={
            "default": [
                -0.25,
                0.0,
                0.0,
                0.5,
                -0.25,
                0.0,
                -0.25,
                0.0,
                0.0,
                0.5,
                -0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3,
                0.0,
                -0.3,
                0.0,
                -0.3,
                0.0,
                -0.3,
            ]
        },
    )
