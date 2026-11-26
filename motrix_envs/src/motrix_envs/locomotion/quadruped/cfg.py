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
    hard_case_probability: float = 0.0
    hard_cases: tuple[tuple[float, float, float], ...] = ()
    # Optional linear curriculum for hard-case command sampling.  Each point
    # is ``(training_iteration, probability)``; probabilities are linearly
    # interpolated between points and held at the nearest endpoint outside
    # the specified range.
    hard_case_curriculum: tuple[tuple[int, float], ...] = ()
    hard_case_curriculum_steps_per_iteration: int = 24
    standing_threshold: float = 0.05
    resampling_seconds_range: tuple[float, float] | None = None

    def validate(self) -> None:
        for name in ("standing_probability", "hard_case_probability"):
            probability = getattr(self, name)
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be finite and inside [0, 1], got {probability}")
        if self.hard_case_probability > 0.0 and not self.hard_cases:
            raise ValueError("hard_case_probability requires at least one hard case")
        for hard_case in self.hard_cases:
            if len(hard_case) != 3 or not np.all(np.isfinite(hard_case)):
                raise ValueError(f"hard case must contain three finite values, got {hard_case}")
        if self.hard_case_curriculum_steps_per_iteration <= 0:
            raise ValueError("hard_case_curriculum_steps_per_iteration must be positive")
        previous_iteration = -1
        for point in self.hard_case_curriculum:
            if len(point) != 2:
                raise ValueError(f"hard-case curriculum point must be (iteration, probability), got {point}")
            iteration, probability = point
            if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
                raise ValueError(f"hard-case curriculum iteration must be a non-negative integer, got {iteration}")
            if iteration <= previous_iteration:
                raise ValueError("hard-case curriculum iterations must be strictly increasing")
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"hard-case curriculum probability must be inside [0, 1], got {probability}")
            previous_iteration = iteration
        if self.hard_case_curriculum and not self.hard_cases:
            raise ValueError("hard_case_curriculum requires at least one hard case")
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
    feet_spacing: float = 0.0
    contact: float = 0.24
    swing_feet_z: float = 2.0
    swing_contact: float = -1.0
    touchdown_velocity: float = 0.0
    # Optional posture/smoothness terms.  They remain disabled for the
    # existing quadruped baselines unless a robot config opts in explicitly.
    flat_orientation: float = 0.0
    dof_acc: float = 0.0
    hip_position: float = 0.0
    base_com_position: float = 0.0


@configclass
class RewardConfig:
    scales: RewardScales = RewardScales()
    excluded_rewards: tuple[str, ...] = ()
    tracking_lin_vel_sigma: float = 0.25
    tracking_lin_vel_forward_sigma: float | None = None
    tracking_lin_vel_lateral_sigma: float | None = None
    swing_contact_forward_scale: float | None = None
    tracking_ang_vel_sigma: float = 0.25
    target_foot_height: float = 0.1
    swing_feet_height_sigma: float = 0.05
    swing_fraction: float = 0.4
    swing_feet_z_sinusoidal: bool = False
    swing_rise_fraction: float = 0.2
    swing_plateau_fraction: float = 0.2
    foot_height_reference_offset: float = 0.0
    touchdown_height_threshold: float = 0.03
    touchdown_speed_sigma: float = 0.5
    touchdown_speed_clip: float = 3.0
    feet_spacing_tolerance: float = 0.01
    feet_spacing_sigma: float = 0.03
    hip_position_tolerance: float = 0.05
    hip_position_sigma: float = 0.05
    base_com_position_tolerance: float = 0.02
    base_com_position_sigma: float = 0.05
    base_height_target: float = 0.3


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
    # Measured Go2 gait anchors.  The runtime interpolates between these
    # points as planar command speed changes; ``gait_frequency`` remains the
    # legacy fallback/default for callers that do not opt into scheduling.
    gait_schedule_enabled: bool = True
    gait_schedule_speed_low: float = 0.2
    gait_schedule_speed_high: float = 0.8
    gait_schedule_frequency_low: float = 1.05
    gait_schedule_frequency_high: float = 2.14
    gait_schedule_swing_fraction_low: float = 0.368
    gait_schedule_swing_fraction_high: float = 0.500
    gait_schedule_smoothing_time_s: float = 0.60
    gait_schedule_yaw_weight: float = 0.80
    gait_schedule_lateral_weight: float = 1.00
    # Shared by training and the compiled deployment task, not backend recipes.
    termination_min_up_z: float = 0.5
    sim: SimCfg = SimCfg(dt=0.01, solver_iterations=1)
    ctrl_dt: float = 0.02

    def validate(self) -> None:
        super().validate()
        self.randomization.validate()
        self.commands.velocity.validate()
        if not np.isfinite(self.termination_min_up_z) or not -1.0 <= self.termination_min_up_z <= 1.0:
            raise ValueError("termination_min_up_z must be finite and inside [-1, 1]")
        if not np.isfinite(self.gait_schedule_smoothing_time_s) or self.gait_schedule_smoothing_time_s <= 0.0:
            raise ValueError("gait_schedule_smoothing_time_s must be finite and positive")
        if not np.isfinite(self.gait_schedule_yaw_weight) or self.gait_schedule_yaw_weight < 0.0:
            raise ValueError("gait_schedule_yaw_weight must be finite and non-negative")
        if not np.isfinite(self.gait_schedule_lateral_weight) or self.gait_schedule_lateral_weight < 0.0:
            raise ValueError("gait_schedule_lateral_weight must be finite and non-negative")
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
                velocity=replace(
                    self.commands.velocity,
                    resampling_seconds_range=None,
                    hard_case_probability=0.0,
                    hard_case_curriculum=(),
                ),
            ),
        )


__all__ = [
    "Commands",
    "ControlConfig",
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
