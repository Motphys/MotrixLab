# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Action terms for manager-based whole-body tracking."""

import gymnasium as gym
import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import ActionCfg, ActionTerm, ManagerEnv, SharedArray, kernel_data
from motrix_env_core.sim.model import ActuatorSpec, ActuatorType
from motrix_envs.locomotion.action_space import joint_position_action_space_from_ctrl_ranges


@configclass
class WbtControlCfg:
    """Position-action scaling for WBT actuators."""

    action_scale: float = 0.25
    action_scales_by_effort_limit_over_p_gain: bool = True


@kernel_data
class WbtJointPositionAction(ActionTerm):
    """Persistent WBT action pipeline, history, and shared model data."""

    current: np.ndarray
    previous: np.ndarray
    default_angles: SharedArray
    joint_lower: SharedArray
    joint_upper: SharedArray
    action_scales: SharedArray

    def action_space(self, env: ManagerEnv, actuators: tuple[ActuatorSpec, ...] | None) -> gym.spaces.Box:
        assert actuators is not None
        ctrl_ranges = []
        for spec in actuators:
            if spec.actuator_type is not ActuatorType.POSITION:
                raise ValueError(f"actuator {spec.name!r} must be a position actuator, got {spec.actuator_type!r}")
            if spec.ctrl_range is None:
                raise ValueError(f"position actuator {spec.name!r} must define or inherit ctrl_range")
            ctrl_ranges.append(spec.ctrl_range)
        return joint_position_action_space_from_ctrl_ranges(
            np.asarray(ctrl_ranges, dtype=np.float32),
            self.default_angles,
            self.action_scales,
        )

    def process(self, actions: np.ndarray) -> np.ndarray:
        np.copyto(self.previous, self.current)
        np.copyto(self.current, actions, casting="unsafe")
        return self.current * self.action_scales + self.default_angles

    def reset(self, env_ids: np.ndarray) -> None:
        self.current[env_ids] = 0.0
        self.previous[env_ids] = 0.0


@configclass(kw_only=True)
class WbtJointPositionActionCfg(ActionCfg):
    control: WbtControlCfg = WbtControlCfg()
    actuator_names: tuple[str, ...] = ()

    def __call__(self, env: ManagerEnv, actuators: tuple[ActuatorSpec, ...] | None) -> ActionTerm:
        assert actuators is not None
        robot = env.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"WBT scene robot must be RobotCfg, got {type(robot).__name__}")
        if "default" not in robot.key_pose.poses:
            raise ValueError("WBT robot must define key pose 'default'")
        kps = self._read_position_actuator_kps(env)
        default_angles = self._resolve_default_pose(
            env,
            tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names),
            tuple(robot.key_pose.poses["default"]),
        )
        action_scales = self._init_action_scales(env, kps)
        joint_lower, joint_upper = env.model.others["robot_joint_position_limits"]
        expected_joint_shape = env.sim_data["robot_dof_pos"].shape[1:]
        if joint_lower.shape != expected_joint_shape or joint_upper.shape != expected_joint_shape:
            raise ValueError(
                "WBT robot joint position limits must match robot_dof_pos: "
                f"lower={joint_lower.shape}, upper={joint_upper.shape}, dof_pos={expected_joint_shape}."
            )
        actuator_names = tuple(spec.name for spec in actuators)
        all_names = tuple(spec.name for spec in env.model.actuators)
        indices = np.asarray([all_names.index(name) for name in actuator_names], dtype=np.int64)
        shape = (env.num_envs, len(actuators))
        return WbtJointPositionAction(
            current=np.zeros(shape, dtype=np.float32),
            previous=np.zeros(shape, dtype=np.float32),
            default_angles=default_angles[indices],
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            action_scales=action_scales[indices],
        )

    @staticmethod
    def _resolve_default_pose(
        env: ManagerEnv,
        joint_names: tuple[str, ...],
        joint_positions: tuple[float, ...],
    ) -> np.ndarray:
        if len(joint_names) != len(joint_positions):
            raise ValueError(
                f"default pose must contain one position per joint: {len(joint_names)} names, "
                f"{len(joint_positions)} positions"
            )
        actuator_joint_names = []
        for spec in env.model.actuators:
            actuator_joint_names.append(spec.target_name)
        positions = dict(zip(joint_names, joint_positions, strict=True))
        missing = sorted(set(actuator_joint_names).difference(positions))
        extra = sorted(set(positions).difference(actuator_joint_names))
        if missing or extra:
            raise ValueError(
                "WBT robot key pose 'default' must match actuator joint targets exactly: "
                f"missing={missing}, extra={extra}"
            )
        return np.asarray([positions[name] for name in actuator_joint_names], dtype=np.float32)

    @staticmethod
    def _read_position_actuator_kps(env: ManagerEnv) -> np.ndarray:
        return np.asarray(env.model.others["actuator_kp"], dtype=np.float32)

    @staticmethod
    def _read_position_actuator_effort_limits(env: ManagerEnv) -> np.ndarray:
        actuators = env.model.actuators
        effort_limits = np.empty(len(actuators), dtype=np.float32)
        for index, spec in enumerate(actuators):
            if spec.force_range is None:
                raise ValueError(f"WBT actuator '{spec.name}' must define force_range")
            force_range = np.asarray(spec.force_range, dtype=np.float32)
            if force_range.shape != (2,) or not np.all(np.isfinite(force_range)):
                raise ValueError(f"WBT actuator '{spec.name}' force_range must contain two finite values")
            effort_limit = float(np.max(np.abs(force_range)))
            if effort_limit <= 0.0:
                raise ValueError(f"WBT actuator '{spec.name}' force_range must define a positive effort limit")
            effort_limits[index] = effort_limit
        return effort_limits

    def _init_action_scales(self, env: ManagerEnv, kps: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.control.action_scale) or self.control.action_scale <= 0.0:
            raise ValueError(f"action_scale must be positive and finite, got {self.control.action_scale}")
        if self.control.action_scales_by_effort_limit_over_p_gain:
            effort = self._read_position_actuator_effort_limits(env)
            safe_kp = np.where(kps == 0.0, 1.0, kps)
            return np.where(kps == 0.0, 0.0, self.control.action_scale * effort / safe_kp).astype(np.float32)
        return np.full(env.num_actuators, self.control.action_scale, dtype=np.float32)


__all__ = ["WbtControlCfg", "WbtJointPositionAction", "WbtJointPositionActionCfg"]
