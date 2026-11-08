# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic joint-position action terms for manager-based environments."""

import math
from dataclasses import dataclass

import gymnasium as gym
import torch

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import ActionCfg, ActionState, ActionTerm, ManagerEnv
from motrix_env_core.mdp.action_space import joint_position_action_space_from_ctrl_ranges
from motrix_env_core.sim.model import ActuatorSpec, ActuatorType


@dataclass(frozen=True, slots=True)
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

    default_angles: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    action_scales: torch.Tensor
    delay_steps: torch.Tensor
    delay_lo: int
    delay_hi: int


class JointPositionActionTerm(ActionTerm):
    """Host runtime for joint-position targets and reset-time action delays."""

    state: JointPositionActionState

    def __init__(self, space: gym.spaces.Box, state: JointPositionActionState) -> None:
        super().__init__(space, state)
        # Seeded generator driving reset-time delay resampling; lives on the
        # term (not the kernel-data state) because it is host-only.
        self.generator = torch.Generator()

    def _process(self, actions: torch.Tensor) -> torch.Tensor:
        """Map raw policy actions to targets using manager-owned action history.

        Stays on tensors end to end: the delayed-slot lookup uses
        ``torch.take_along_dim`` and the affine target map multiplies torch
        buffers directly, mirroring
        ``motrix_env_core.torch_manager.actions.process_joint_position_actions``.
        """
        state = self.state
        ptr = int(state.action_ptr[0])
        target_raw = actions
        if state.delay_hi > 0:
            queue = state.action_queue
            width = queue.shape[1]
            delay_indices = (ptr - state.delay_steps) % width
            target_raw = torch.take_along_dim(queue, delay_indices[:, None, None], dim=1)[:, 0, :]
        return target_raw * state.action_scales + state.default_angles

    def _reset(self, env_ids: torch.Tensor) -> None:
        """Resample per-episode delay after the base term clears raw history."""
        state = self.state
        if state.delay_hi > 0:
            state.delay_steps[env_ids] = torch.randint(
                state.delay_lo,
                state.delay_hi + 1,
                (env_ids.numel(),),
                generator=self.generator,
            )


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
        indices = torch.tensor([all_names.index(name) for name in actuator_names], dtype=torch.int64)
        lo, hi = self.action_delay_steps
        if lo < 0 or hi < lo:
            raise ValueError(f"action_delay_steps must be 0 <= lo <= hi, got {self.action_delay_steps!r}")
        # Keep current and previous actions even when delay is disabled.
        width = max(hi + 1, 2)
        state = JointPositionActionState(
            action_queue=torch.zeros((env.num_envs, width, len(actuators)), dtype=torch.float32),
            default_angles=default_angles[indices],
            joint_lower=joint_lower[indices],
            joint_upper=joint_upper[indices],
            action_scales=action_scales[indices],
            delay_steps=torch.zeros(env.num_envs, dtype=torch.int64),
            action_ptr=torch.zeros(1, dtype=torch.int64),
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
            torch.tensor(ctrl_ranges, dtype=torch.float32), state.default_angles, state.action_scales
        )
        term = JointPositionActionTerm(space, state)
        # Reset-time delay resampling is reproducible per environment seed.
        term.generator.manual_seed(int(getattr(env, "_rand_seed", 0)) + 0x5EED)
        return term

    @staticmethod
    def _resolve_default_pose(
        env: ManagerEnv,
        joint_names: tuple[str, ...],
        joint_positions: tuple[float, ...],
    ) -> torch.Tensor:
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
        return torch.tensor([positions[name] for name in actuator_joint_names], dtype=torch.float32)

    @staticmethod
    def _read_position_actuator_kps(env: ManagerEnv) -> torch.Tensor:
        return torch.as_tensor(env.model.others["actuator_kp"]).detach().cpu()

    @staticmethod
    def _read_position_actuator_effort_limits(env: ManagerEnv) -> torch.Tensor:
        effort_limits = []
        for spec in env.model.actuators:
            if spec.force_range is None or len(spec.force_range) != 2:
                raise ValueError(f"actuator '{spec.name}' must define a two-valued force_range")
            low, high = float(spec.force_range[0]), float(spec.force_range[1])
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError(f"actuator '{spec.name}' force_range must contain two finite values")
            effort_limit = max(abs(low), abs(high))
            if effort_limit <= 0.0:
                raise ValueError(f"actuator '{spec.name}' force_range must define a positive effort limit")
            effort_limits.append(effort_limit)
        return torch.tensor(effort_limits, dtype=torch.float32)

    def _init_action_scales(self, env: ManagerEnv, kps: torch.Tensor) -> torch.Tensor:
        if not math.isfinite(self.action_scale) or self.action_scale <= 0.0:
            raise ValueError(f"action_scale must be positive and finite, got {self.action_scale}")
        if self.action_scales_by_effort_limit_over_p_gain:
            effort = self._read_position_actuator_effort_limits(env)
            safe_kp = torch.where(kps == 0.0, torch.ones_like(kps), kps)
            return torch.where(kps == 0.0, torch.zeros_like(kps), self.action_scale * effort / safe_kp)
        return torch.full((env.num_actuators,), self.action_scale, dtype=torch.float32)


__all__ = ["JointPositionActionState", "JointPositionActionTerm", "JointPositionActionCfg"]
