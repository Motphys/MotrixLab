# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the generic quadruped flat-terrain walk task."""

from dataclasses import replace

import numpy as np
from motrix_robots.quadruped import QuadrupedRobotCfg
from numpy.typing import NDArray
from omegaconf import MISSING

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    ContactSensorCfg,
    ContactSensorReduce,
    FrameObjectKind,
    FrameSensorCfg,
    FrameSensorType,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    SceneSensorsCfg,
)
from motrix_env_core.direct.env import DirectEnvCfg
from motrix_envs.config.scene import StandardSceneAssetsCfg, StandardSceneCfg


def _contact_sensor(geom_name: str | None) -> ContactSensorCfg:
    if geom_name is None:
        raise ValueError("Quadruped task requires a contact geom name for every leg")
    return ContactSensorCfg(geom1="floor", geom2=geom_name, reduce=ContactSensorReduce.mindist)


@configclass
class QuadrupedTaskSensorsCfg(SceneSensorsCfg):
    """Required contact sensors for the four quadruped task legs."""

    front_left_contact: ContactSensorCfg = MISSING
    front_right_contact: ContactSensorCfg = MISSING
    rear_left_contact: ContactSensorCfg = MISSING
    rear_right_contact: ContactSensorCfg = MISSING
    front_left_world_pos: FrameSensorCfg = MISSING
    front_right_world_pos: FrameSensorCfg = MISSING
    rear_left_world_pos: FrameSensorCfg = MISSING
    rear_right_world_pos: FrameSensorCfg = MISSING


def _world_position_sensor(geom_name: str | None) -> FrameSensorCfg:
    if geom_name is None:
        raise ValueError("Quadruped task requires a contact geom name for every leg")
    return FrameSensorCfg(
        object_type=FrameObjectKind.geom,
        object_name=geom_name,
        sensor_type=FrameSensorType.framepos,
    )


@configclass
class QuadrupedSceneCfg(StandardSceneCfg):
    """A standard scene whose foot contact sensors are derived from its quadruped robot."""

    sensors: QuadrupedTaskSensorsCfg = QuadrupedTaskSensorsCfg()

    def __post_init__(self) -> None:
        robot = self.objs.robot
        if not isinstance(robot, QuadrupedRobotCfg):
            raise TypeError(f"QuadrupedSceneCfg robot must be QuadrupedRobotCfg, got {type(robot).__name__}")
        front_left, front_right, rear_left, rear_right = robot.foot_contact_geom_names
        self.sensors = QuadrupedTaskSensorsCfg(
            front_left_contact=_contact_sensor(front_left),
            front_right_contact=_contact_sensor(front_right),
            rear_left_contact=_contact_sensor(rear_left),
            rear_right_contact=_contact_sensor(rear_right),
            front_left_world_pos=_world_position_sensor(front_left),
            front_right_world_pos=_world_position_sensor(front_right),
            rear_left_world_pos=_world_position_sensor(rear_left),
            rear_right_world_pos=_world_position_sensor(rear_right),
        )


@configclass
class QuadrupedWalkTerrainSceneAssetsCfg(StandardSceneAssetsCfg):
    """Standard scene assets plus the procedural terrain used by rough-walk tasks."""

    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=NoiseTerrainGeneratorCfg(
            seed=0,
            height_scale=0.1,
            flip_y=True,
        ),
        size=(64.0, 64.0),
        shape=(320, 320),
    )


@configclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@configclass
class ControlConfig:
    # action scale: target angle = action_scale * action + default_angle
    action_scale: float = 0.25
    simulate_action_latency: bool = False


@configclass
class QuadrupedWalkRandomizationCfg:
    """Episode-level domain randomization for quadruped walking."""

    enabled: bool = False
    joint_pos_noise: float = 0.0
    joint_vel_noise: float = 0.0
    base_lin_vel_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_ang_vel_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_delay_steps: tuple[int, int] = (0, 0)
    kp_scale_range: tuple[float, float] = (1.0, 1.0)
    damping_scale_range: tuple[float, float] = (1.0, 1.0)
    sliding_friction_range: tuple[float, float] | None = None
    base_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    base_com_offset_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        scalar_noise = {
            "joint_pos_noise": self.joint_pos_noise,
            "joint_vel_noise": self.joint_vel_noise,
        }
        vector_noise = {
            "base_lin_vel_noise": self.base_lin_vel_noise,
            "base_ang_vel_noise": self.base_ang_vel_noise,
            "base_com_offset_noise": self.base_com_offset_noise,
        }
        for name, value in scalar_noise.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        for name, value in vector_noise.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must contain three finite non-negative values, got {value}")

        delay_low, delay_high = self.action_delay_steps
        if (
            isinstance(delay_low, bool)
            or isinstance(delay_high, bool)
            or not isinstance(delay_low, int)
            or not isinstance(delay_high, int)
            or delay_low < 0
            or delay_low > delay_high
            or delay_high > 1
        ):
            raise ValueError(
                "action_delay_steps must be an ordered integer range within the currently supported [0, 1], "
                f"got {self.action_delay_steps}"
            )

        for name, value in {
            "kp_scale_range": self.kp_scale_range,
            "damping_scale_range": self.damping_scale_range,
            "base_mass_scale_range": self.base_mass_scale_range,
        }.items():
            low, high = value
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low > high:
                raise ValueError(f"{name} must be a finite positive ordered range, got {value}")
        if self.sliding_friction_range is not None:
            low, high = self.sliding_friction_range
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low > high:
                raise ValueError(
                    f"sliding_friction_range must be a finite positive ordered range, got {self.sliding_friction_range}"
                )


@configclass
class VelocityCommandCfg:
    lower: NDArray[np.float32] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    upper: NDArray[np.float32] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    standing_probability: float = 0.0
    standing_threshold: float = 0.05
    resampling_seconds_range: tuple[float, float] | None = None

    def validate(self) -> None:
        if not np.isfinite(self.standing_probability) or not 0.0 <= self.standing_probability <= 1.0:
            raise ValueError(f"standing_probability must be finite and inside [0, 1], got {self.standing_probability}")
        if self.resampling_seconds_range is not None:
            low, high = self.resampling_seconds_range
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low > high:
                raise ValueError(
                    "resampling_seconds_range must be a finite positive ordered range, "
                    f"got {self.resampling_seconds_range}"
                )


@configclass
class Commands:
    velocity: VelocityCommandCfg = VelocityCommandCfg()


@configclass
class Sensor:
    # Scene-declared names are used directly; model-local names follow robot affixes.
    gyro: str = "gyro"
    local_linvel: str = "local_linvel"
    upvector: str = "upvector"
    foot_positions: tuple[str, str, str, str] = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")


@configclass
class RewardScales:
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 1.0
    lin_vel_z: float = -5.0
    ang_vel_xy: float = -0.1
    base_height: float = -100.0
    action_rate: float = -0.1
    similar_to_default: float = -0.1
    contact: float = 0.24
    swing_feet_z: float = 2.0
    swing_contact: float = -1.0
    flat_orientation: float = 0.0
    stand_still: float = 0.0
    feet_slide: float = 0.0
    action_smoothness: float = 0.0


@configclass
class FootLateralRewardCfg:
    """A body-frame lateral foot-placement reward for a named set of feet."""

    name: str = ""
    feet: tuple[str, ...] = ()
    phase: str = "swing"
    target_abs_y: float | None = None
    min_abs_y: float | None = None
    max_abs_y: float | None = None
    width_tolerance: float = 0.02
    width_scale: float = 0.0
    lateral_speed_tolerance: float = 0.15
    lateral_speed_scale: float = 0.0
    speed_only_outside_band: bool = True

    def validate(self) -> None:
        if not self.name:
            raise ValueError("foot lateral reward name must not be empty")
        if not self.feet or len(set(self.feet)) != len(self.feet):
            raise ValueError(f"foot lateral reward {self.name!r} must name unique feet")
        if self.phase not in {"swing", "stance"}:
            raise ValueError(f"foot lateral reward {self.name!r} phase must be 'swing' or 'stance'")
        if not np.isfinite(self.width_tolerance) or self.width_tolerance <= 0.0:
            raise ValueError(f"foot lateral reward {self.name!r} width_tolerance must be positive")
        if not np.isfinite(self.lateral_speed_tolerance) or self.lateral_speed_tolerance <= 0.0:
            raise ValueError(f"foot lateral reward {self.name!r} lateral_speed_tolerance must be positive")
        if not np.isfinite(self.width_scale) or not np.isfinite(self.lateral_speed_scale):
            raise ValueError(f"foot lateral reward {self.name!r} scales must be finite")

        has_target = self.target_abs_y is not None
        has_band = self.min_abs_y is not None or self.max_abs_y is not None
        if has_target == has_band:
            raise ValueError(
                f"foot lateral reward {self.name!r} must define either target_abs_y or min_abs_y and max_abs_y"
            )
        if has_target:
            if not np.isfinite(self.target_abs_y) or self.target_abs_y < 0.0:
                raise ValueError(f"foot lateral reward {self.name!r} target_abs_y must be non-negative")
            return
        if self.min_abs_y is None or self.max_abs_y is None:
            raise ValueError(f"foot lateral reward {self.name!r} band requires min_abs_y and max_abs_y")
        if (
            not np.isfinite(self.min_abs_y)
            or not np.isfinite(self.max_abs_y)
            or self.min_abs_y < 0.0
            or self.min_abs_y > self.max_abs_y
        ):
            raise ValueError(f"foot lateral reward {self.name!r} must define a non-negative ordered band")


@configclass
class RewardConfig:
    scales: RewardScales = RewardScales()
    tracking_lin_vel_sigma: float = 0.25
    tracking_ang_vel_sigma: float = 0.25
    target_foot_height: float = 0.1
    swing_feet_height_sigma: float = 0.05
    base_height_target: float = 0.3
    stand_still_threshold: float = 0.05
    foot_lateral_terms: tuple[FootLateralRewardCfg, ...] = ()

    def validate(self) -> None:
        names: set[str] = set()
        for term in self.foot_lateral_terms:
            term.validate()
            if term.name in names:
                raise ValueError(f"duplicate foot lateral reward name {term.name!r}")
            names.add(term.name)


@configclass
class QuadrupedWalkEnvCfg(DirectEnvCfg):
    """Base configuration for quadruped walk tasks."""

    max_episode_seconds: float = 20.0
    scene: QuadrupedSceneCfg = MISSING
    noise_config: NoiseConfig = NoiseConfig()
    control_config: ControlConfig = ControlConfig()
    randomization: QuadrupedWalkRandomizationCfg = QuadrupedWalkRandomizationCfg()
    commands: Commands = Commands()
    sensor: Sensor = Sensor()
    reward_config: RewardConfig = RewardConfig()
    key_pose_name: str = "default"
    ground_geom_name: str = "floor"
    initial_base_position: tuple[float, float, float] = (0.0, 0.0, 0.3)
    spawn_xy_range: float = 0.0
    trot_pairs: tuple[tuple[int, int], ...] = ((0, 3), (1, 2))
    gait_frequency: float = 2.0
    # Shared by training and the compiled deployment task, not backend recipes.
    termination_min_up_z: float = 0.5
    sim: SimCfg = SimCfg(dt=0.01, solver_iterations=1)
    ctrl_dt: float = 0.02

    def validate(self) -> None:
        super().validate()
        self.randomization.validate()
        self.commands.velocity.validate()
        self.reward_config.validate()
        if not np.isfinite(self.termination_min_up_z) or not -1.0 <= self.termination_min_up_z <= 1.0:
            raise ValueError("termination_min_up_z must be finite and inside [-1, 1]")
        if (
            self.randomization.enabled
            and any(self.randomization.action_delay_steps)
            and self.control_config.simulate_action_latency
        ):
            raise ValueError("random action delay cannot be combined with simulate_action_latency")

    def for_play(self) -> "QuadrupedWalkEnvCfg":
        """Disable domain randomization and command changes for play/evaluation."""

        return replace(
            self,
            randomization=replace(self.randomization, enabled=False),
            commands=replace(
                self.commands,
                velocity=replace(self.commands.velocity, resampling_seconds_range=None),
            ),
        )


__all__ = [
    "Commands",
    "ControlConfig",
    "FootLateralRewardCfg",
    "NoiseConfig",
    "QuadrupedWalkRandomizationCfg",
    "QuadrupedSceneCfg",
    "QuadrupedTaskSensorsCfg",
    "QuadrupedWalkEnvCfg",
    "QuadrupedWalkTerrainSceneAssetsCfg",
    "RewardConfig",
    "RewardScales",
    "Sensor",
    "VelocityCommandCfg",
]
