# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""ANYmal-C flat- and rough-terrain walk configuration and environment registration.

The ANYmal-C MJCF only ships base linear-velocity and gyro sensors, so the frame
sensors that :class:`QuadrupedWalkTask` expects (local linear velocity, up vector,
per-foot position) are declared here via :class:`FrameSensorCfg` instead of edited
into the robot file.
"""

from motrix_robots.anymal import AnymalC

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    FlatTerrainCfg,
    FrameObjectKind,
    FrameRefKind,
    FrameSensorCfg,
    FrameSensorType,
    HFieldTerrainCfg,
)
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import (
    QuadrupedSceneCfg,
    QuadrupedTaskSensorsCfg,
    QuadrupedWalkEnvCfg,
    QuadrupedWalkTerrainSceneAssetsCfg,
    RewardConfig,
    RewardScales,
    Sensor,
)
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


@configclass
class AnymalCWalkSensorsCfg(QuadrupedTaskSensorsCfg):
    """Quadruped contact sensors plus the ANYmal-C frame sensors its MJCF lacks."""

    local_linvel: FrameSensorCfg = FrameSensorCfg(
        object_type=FrameObjectKind.site,
        object_name="imu_site",
        sensor_type=FrameSensorType.framelinvel,
        ref_kind=FrameRefKind.local,
    )
    upvector: FrameSensorCfg = FrameSensorCfg(
        object_type=FrameObjectKind.site,
        object_name="imu_site",
        sensor_type=FrameSensorType.zaxis,
    )
    FL_pos: FrameSensorCfg = FrameSensorCfg(
        object_type=FrameObjectKind.geom,
        object_name="LF_FOOT",
        sensor_type=FrameSensorType.framepos,
        ref_kind=FrameRefKind.object,
        ref_object_type=FrameObjectKind.site,
        ref_object_name="imu_site",
    )
    FR_pos: FrameSensorCfg = FrameSensorCfg(
        object_type=FrameObjectKind.geom,
        object_name="RF_FOOT",
        sensor_type=FrameSensorType.framepos,
        ref_kind=FrameRefKind.object,
        ref_object_type=FrameObjectKind.site,
        ref_object_name="imu_site",
    )
    RL_pos: FrameSensorCfg = FrameSensorCfg(
        object_type=FrameObjectKind.geom,
        object_name="LH_FOOT",
        sensor_type=FrameSensorType.framepos,
        ref_kind=FrameRefKind.object,
        ref_object_type=FrameObjectKind.site,
        ref_object_name="imu_site",
    )
    RR_pos: FrameSensorCfg = FrameSensorCfg(
        object_type=FrameObjectKind.geom,
        object_name="RH_FOOT",
        sensor_type=FrameSensorType.framepos,
        ref_kind=FrameRefKind.object,
        ref_object_type=FrameObjectKind.site,
        ref_object_name="imu_site",
    )


@configclass
class AnymalCWalkSceneCfg(QuadrupedSceneCfg):
    """Walk scene whose sensors include the ANYmal-C frame sensors."""

    def __post_init__(self) -> None:
        super().__post_init__()
        base = self.sensors
        self.sensors = AnymalCWalkSensorsCfg(
            front_left_contact=base.front_left_contact,
            front_right_contact=base.front_right_contact,
            rear_left_contact=base.rear_left_contact,
            rear_right_contact=base.rear_right_contact,
        )


@registry.envcfg("anymalc-walk-flat")
@configclass
class AnymalCWalkDirectEnvCfg(QuadrupedWalkEnvCfg):
    """Track locomotion commands with ANYmal-C on flat ground.

    zh_CN: 控制 ANYmal-C 在平地上跟踪移动速度指令。
    """

    render_spacing: float = 0.0
    spawn_xy_range: float = 4.0
    scene: AnymalCWalkSceneCfg = AnymalCWalkSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=AnymalC(),
        ),
    )

    # Reward/action tuning mirrors the proven Go2 walk recipe; only the base
    # height and spawn z are adapted for ANYmal-C. action_scale stays at the
    # QuadrupedWalkEnvCfg default (0.25) instead of the navigation task's 0.06,
    # which was too small to express a stepping gait.
    reward_config: RewardConfig = RewardConfig(
        scales=RewardScales(similar_to_default=-0.05),
        tracking_ang_vel_sigma=0.05,
        target_foot_height=0.05,
        base_height_target=0.5,
    )
    initial_base_position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    # ANYmal-C MJCF names its gyro ``base_gyro``; reuse it instead of adding a new one.
    sensor: Sensor = Sensor(gyro="base_gyro")


@registry.envcfg("anymalc-walk-rough")
@configclass
class AnymalCWalkRoughDirectEnvCfg(AnymalCWalkDirectEnvCfg):
    """Track walking commands with ANYmal-C on a procedural rough height field.

    zh_CN: 控制 ANYmal-C 在程序化粗糙高度场上跟踪行走指令。
    """

    scene: AnymalCWalkSceneCfg = AnymalCWalkSceneCfg(
        assets=QuadrupedWalkTerrainSceneAssetsCfg(),
        objs=StandardSceneObjsCfg(
            floor=HFieldTerrainCfg(
                hfield="terrain",
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=AnymalC(),
        ),
    )


registry.env("anymalc-walk-flat")(QuadrupedWalkTask)
registry.env("anymalc-walk-rough")(QuadrupedWalkTask)
