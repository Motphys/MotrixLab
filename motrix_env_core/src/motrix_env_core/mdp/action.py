# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic joint-position action terms for manager-based environments."""

import gymnasium as gym
import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import ActionCfg, ActionState, ActionTerm, ManagerEnv, SharedArray, kernel_data
from motrix_env_core.mdp.action_space import joint_position_action_space_from_ctrl_ranges
from motrix_env_core.sim.model import ActuatorSpec, ActuatorType


@kernel_data
class JointPositionActionState(ActionState):
    """Persistent joint-position action pipeline, history, and shared model data.

    The applied position target is ``target_raw * action_scales +
    default_angles``, where ``target_raw`` is the raw or delayed policy
    action.

    Attributes:
        action_queue: Raw policy-action ring buffer ``(N, W, A)`` with
            ``W = max(delay_hi + 1, 2)``. Current and previous actions occupy
            slots ``action_ptr`` and ``(action_ptr - 1) % W`` respectively.
            It is consumed by action
            observations, the action-rate penalty, and delayed target lookup.
        default_angles: Per-actuator default pose, ``(A,)``.
        joint_lower: Joint position lower limits, ``(A,)``.
        joint_upper: Joint position upper limits, ``(A,)``.
        action_scales: Per-actuator target scaling, ``(A,)``.
        delay_steps: Per-lane applied delay in control steps, resampled
            uniformly from ``[delay_lo, delay_hi]`` at reset.
        action_ptr: Ring-buffer write cursor as a one-element int array: the
            column holding the latest raw action; advanced once per control
            step.
        delay_lo: Inclusive lower delay-range bound in control steps.
        delay_hi: Inclusive upper delay-range bound; 0 disables the delay
            path entirely.
    """

    default_angles: SharedArray
    joint_lower: SharedArray
    joint_upper: SharedArray
    action_scales: SharedArray
    delay_steps: np.ndarray
    delay_lo: int
    delay_hi: int


class JointPositionActionTerm(ActionTerm):
    """Host runtime for joint-position targets and reset-time action delays."""

    state: JointPositionActionState

    def __init__(self, space: gym.spaces.Box, state: JointPositionActionState) -> None:
        super().__init__(space, state)

    def _process(self, actions: np.ndarray) -> np.ndarray:
        """Map raw policy actions to targets using manager-owned action history."""
        state = self.state
        ptr = int(state.action_ptr[0])
        target_raw = actions
        if state.delay_hi > 0:
            width = state.action_queue.shape[1]
            delay_indices = (ptr - state.delay_steps) % width
            target_raw = np.take_along_axis(
                state.action_queue,
                delay_indices[:, None, None],
                axis=1,
            )[:, 0, :]
        return target_raw * state.action_scales + state.default_angles

    def _reset(self, env_ids: np.ndarray) -> None:
        """Resample per-episode delay after the base term clears raw history."""
        state = self.state
        if state.delay_hi > 0:
            state.delay_steps[env_ids] = np.random.randint(state.delay_lo, state.delay_hi + 1, size=env_ids.size)


@configclass(kw_only=True)
class JointPositionActionCfg(ActionCfg):
    """Joint-position targets from normalized policy actions.

    Attributes:
        action_scale: Positive target-scaling factor in rad per unit action.
            With ``action_scales_by_effort_limit_over_p_gain`` True each
            actuator scales it by ``effort_limit / kp`` so torque-weak
            actuators move more conservatively, instead of using the value
            directly.
        action_scales_by_effort_limit_over_p_gain: Select the uniform vs
            effort/kp-normalized scaling mode.
        action_delay_steps: Inclusive per-lane control-step delay range
            ``(lo, hi)``, resampled at reset; ``(0, 0)`` disables delayed
            target lookup but still retains current and previous raw actions.
    """

    action_scale: float = 0.25
    action_scales_by_effort_limit_over_p_gain: bool = False
    action_delay_steps: tuple[int, int] = (0, 0)

    def __call__(self, env: ManagerEnv, actuators: tuple[ActuatorSpec, ...] | None) -> JointPositionActionTerm:
        if actuators is None:
            actuators = env.model.actuators
        robot = env.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"scene robot must be RobotCfg, got {type(robot).__name__}")
        if "default" not in robot.key_pose.poses:
            raise ValueError("robot must define key pose 'default'")
        kps = self._read_position_actuator_kps(env)
        default_angles = self._resolve_default_pose(
            env,
            tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names),
            tuple(robot.key_pose.poses["default"]),
        )
        action_scales = self._init_action_scales(env, kps)
        joint_lower, joint_upper = env.model.others["robot_joint_position_limits"]
        expected_model_shape = (env.num_actuators,)
        if joint_lower.shape != expected_model_shape or joint_upper.shape != expected_model_shape:
            raise ValueError(
                "robot joint position limits must match the model actuator count: "
                f"lower={joint_lower.shape}, upper={joint_upper.shape}, expected={expected_model_shape}."
            )
        actuator_names = tuple(spec.name for spec in actuators)
        all_names = tuple(spec.name for spec in env.model.actuators)
        indices = np.asarray([all_names.index(name) for name in actuator_names], dtype=np.int64)
        joint_lower = joint_lower[indices]
        joint_upper = joint_upper[indices]
        lo, hi = self.action_delay_steps
        if lo < 0 or hi < lo:
            raise ValueError(f"action_delay_steps must be 0 <= lo <= hi, got {self.action_delay_steps!r}")
        # Keep current and previous actions even when delay is disabled.
        width = max(hi + 1, 2)
        action_queue = np.zeros((env.num_envs, width, len(actuators)), dtype=np.float32)
        delay_steps = np.zeros(env.num_envs, dtype=np.int64)
        action_ptr = np.zeros(1, dtype=np.int64)
        state = JointPositionActionState(
            action_queue=action_queue,
            default_angles=default_angles[indices],
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            action_scales=action_scales[indices],
            delay_steps=delay_steps,
            action_ptr=action_ptr,
            delay_lo=int(lo),
            delay_hi=int(hi),
        )
        ctrl_ranges = []
        for spec in actuators:
            if spec.actuator_type is not ActuatorType.POSITION:
                raise ValueError(f"actuator {spec.name!r} must be a position actuator, got {spec.actuator_type!r}")
            if spec.ctrl_range is None:
                raise ValueError(f"position actuator {spec.name!r} must define or inherit ctrl_range")
            ctrl_ranges.append(spec.ctrl_range)
        space = joint_position_action_space_from_ctrl_ranges(
            np.asarray(ctrl_ranges, dtype=np.float32), state.default_angles, state.action_scales
        )
        return JointPositionActionTerm(space, state)

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
                f"robot key pose 'default' must match actuator joint targets exactly: missing={missing}, extra={extra}"
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
                raise ValueError(f"actuator '{spec.name}' must define force_range")
            force_range = np.asarray(spec.force_range, dtype=np.float32)
            if force_range.shape != (2,) or not np.all(np.isfinite(force_range)):
                raise ValueError(f"actuator '{spec.name}' force_range must contain two finite values")
            effort_limit = float(np.max(np.abs(force_range)))
            if effort_limit <= 0.0:
                raise ValueError(f"actuator '{spec.name}' force_range must define a positive effort limit")
            effort_limits[index] = effort_limit
        return effort_limits

    def _init_action_scales(self, env: ManagerEnv, kps: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.action_scale) or self.action_scale <= 0.0:
            raise ValueError(f"action_scale must be positive and finite, got {self.action_scale}")
        if self.action_scales_by_effort_limit_over_p_gain:
            effort = self._read_position_actuator_effort_limits(env)
            safe_kp = np.where(kps == 0.0, 1.0, kps)
            return np.where(kps == 0.0, 0.0, self.action_scale * effort / safe_kp).astype(np.float32)
        return np.full(env.num_actuators, self.action_scale, dtype=np.float32)


__all__ = ["JointPositionActionState", "JointPositionActionTerm", "JointPositionActionCfg"]
