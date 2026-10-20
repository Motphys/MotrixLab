# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Persistent G1 deployment states (passive/fixstand/walk/wbt) and their controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motrix_deploy.contracts import FloatArray, JointServoCommand, RobotSpec, RobotState, float32_array
from motrix_deploy.operator import OperatorCommand
from motrix_deploy.policy import Policy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import PolicyController
from motrix_deploy.runtime.hot_swapping import BaselineStates, RobotHotSwappingController, SkillModule
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.runtime.preparation import JointServoPreparationState, SkillPreparationSpec
from motrix_deploy.runtime.state_machine import (
    AdaptiveServoTransition,
    SmoothServoTransition,
    TransitionRule,
)
from motrix_deploy_tasks.tasks.g1_walk import G1WalkPolicyProcessor, G1WalkTaskSpec
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtMotion, G1WbtPolicyProcessor, G1WbtTaskSpec
from motrix_env_core.input import PlanarVelocityCommand


def _vector(value: object, *, path: str, joint_count: int) -> FloatArray:
    return float32_array(np.asarray(value, dtype=np.float32), path=path, shape=(joint_count,))


def _state_in_joint_order(
    state: RobotState, source_names: tuple[str, ...], target_names: tuple[str, ...]
) -> RobotState:
    if source_names == target_names:
        return state
    indices = np.asarray([source_names.index(name) for name in target_names], dtype=np.int64)
    return RobotState(
        sample_time_ns=state.sample_time_ns,
        receive_time_ns=state.receive_time_ns,
        joint_position=state.joint_position[indices].astype(np.float32),
        joint_velocity=state.joint_velocity[indices].astype(np.float32),
        base_orientation_xyzw=state.base_orientation_xyzw,
        base_angular_velocity=state.base_angular_velocity,
        base_linear_acceleration=state.base_linear_acceleration,
        base_linear_velocity=state.base_linear_velocity,
    )


class G1PassiveState:
    """Persistent zero-stiffness damping without ending the control session."""

    name = "passive"

    def __init__(
        self,
        joint_names: tuple[str, ...],
        kd: FloatArray,
        position_lower: FloatArray,
        position_upper: FloatArray,
    ) -> None:
        self.joint_names = joint_names
        self._kd = _vector(kd, path="g1.passive.kd", joint_count=len(joint_names))
        self._position_lower = _vector(position_lower, path="g1.passive.position_lower", joint_count=len(joint_names))
        self._position_upper = _vector(position_upper, path="g1.passive.position_upper", joint_count=len(joint_names))
        self._zeros = np.zeros(len(joint_names), dtype=np.float32)

    def reset(self, state: RobotState) -> None:
        del state

    def request_stop(self) -> None:
        pass

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep:
        del context
        target = np.clip(state.joint_position, self._position_lower, self._position_upper).astype(np.float32)
        return ControllerStep(command=JointServoCommand(target, self._zeros, self._zeros, self._zeros, self._kd))


class G1FixStandState:
    """Persistent joint-name-defined standing posture and support gains."""

    name = "fixstand"

    def __init__(
        self,
        joint_names: tuple[str, ...],
        joint_position: FloatArray,
        kp: FloatArray,
        kd: FloatArray,
    ) -> None:
        joint_count = len(joint_names)
        self.joint_names = joint_names
        self._target = _vector(joint_position, path="g1.fixstand.joint_position", joint_count=joint_count)
        self._kp = _vector(kp, path="g1.fixstand.kp", joint_count=joint_count)
        self._kd = _vector(kd, path="g1.fixstand.kd", joint_count=joint_count)
        self._zeros = np.zeros(joint_count, dtype=np.float32)

    def reset(self, state: RobotState) -> None:
        del state

    def request_stop(self) -> None:
        pass

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep:
        del state, context
        return ControllerStep(command=JointServoCommand(self._target, self._zeros, self._zeros, self._kp, self._kd))


class G1WalkState:
    """Persistent locomotion backed by the exported Walk processor and ONNX policy."""

    name = "walk"

    def __init__(
        self,
        spec: G1WalkTaskSpec,
        robot: RobotSpec,
        policy: Policy,
        canonical_joint_names: tuple[str, ...],
    ) -> None:
        self.joint_names = robot.joint_names
        self._canonical_joint_names = canonical_joint_names
        self.processor = G1WalkPolicyProcessor(spec, robot)
        self.controller = PolicyController(self.processor, policy)
        self._zero_velocity = PlanarVelocityCommand(np.zeros((1, 3), dtype=np.float32))

    def _state(self, state: RobotState) -> RobotState:
        return _state_in_joint_order(state, self._canonical_joint_names, self.joint_names)

    def reset(self, state: RobotState) -> None:
        self.controller.steps = None
        self.controller.reset(self._state(state))

    def request_stop(self) -> None:
        self.controller.request_stop()

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep:
        operator = context.command
        velocity = (
            self._zero_velocity if operator is None or operator.planar_velocity is None else operator.planar_velocity
        )
        local = ControlContext(context.step, context.elapsed_time_s, velocity, context.dt_s)
        return self.controller.step(self._state(state), local)


class G1WbtState:
    """Finite WBT playback with a frame-zero policy warm-up during takeover."""

    name = "wbt"

    def __init__(
        self,
        spec: G1WbtTaskSpec,
        robot: RobotSpec,
        motion: G1WbtMotion,
        policy: Policy,
        canonical_joint_names: tuple[str, ...],
        *,
        takeover_hold_steps: int,
    ) -> None:
        if takeover_hold_steps < 1:
            raise ValueError("takeover_hold_steps must be at least one policy interval")
        self.joint_names = robot.joint_names
        self._canonical_joint_names = canonical_joint_names
        self.motion = motion
        self.processor = G1WbtPolicyProcessor(spec, robot, motion)
        self.controller = PolicyController(self.processor, policy)
        self._takeover_hold_steps = takeover_hold_steps
        self._ticks = 0
        self._playback_frame = 0
        self._playback_started = False

    def _state(self, state: RobotState) -> RobotState:
        return _state_in_joint_order(state, self._canonical_joint_names, self.joint_names)

    def reset(self, state: RobotState) -> None:
        self._ticks = 0
        self._playback_frame = 0
        self._playback_started = False
        local = self._state(state)
        self.processor.reset_reference()
        # Align the clip's heading with the robot's heading at takeover.
        self.processor.align_reference_to(local)
        self.controller.steps = None
        self.controller.reset(local)

    def request_stop(self) -> None:
        self.controller.request_stop()

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep:
        if not self._playback_started and self._ticks >= self._takeover_hold_steps:
            self._playback_started = True
            self.controller.steps = self.controller.completed_steps + len(self.motion)
        frame = self._playback_frame if self._playback_started else 0
        self.processor.set_reference_frame(min(frame, len(self.motion) - 1))
        local = ControlContext(context.step, context.elapsed_time_s, None, context.dt_s)
        decision = self.controller.step(self._state(state), local)
        self._ticks += 1
        if self._playback_started and decision.policy_tick:
            self._playback_frame += 1
        return decision


@dataclass(frozen=True)
class G1TransitionSettings:
    """Durations for normal policy takeover and safe posture recovery."""

    policy_duration_s: float = 0.5
    passive_duration_min_s: float = 0.5
    passive_duration_max_s: float = 3.0
    passive_max_joint_speed_rad_s: float = 0.5
    passive_gain_ramp_s: float = 0.5


class G1HotSwappingController(RobotHotSwappingController):
    """Bind G1 baseline states and WBT as one standard hot-swappable skill."""

    def __init__(
        self,
        *,
        robot: RobotSpec,
        walk: G1WalkState,
        wbt: G1WbtState,
        fixstand_joint_position: FloatArray,
        fixstand_kp: FloatArray,
        fixstand_kd: FloatArray,
        passive_kd: FloatArray,
        wbt_preparation: SkillPreparationSpec,
        initial_state: str = "fixstand",
        settings: G1TransitionSettings = G1TransitionSettings(),
    ) -> None:
        joint_names = robot.joint_names
        passive = G1PassiveState(joint_names, passive_kd, robot.position_lower, robot.position_upper)
        fixstand = G1FixStandState(joint_names, fixstand_joint_position, fixstand_kp, fixstand_kd)
        baseline_transitions = (
            TransitionRule(
                "passive",
                "fixstand",
                lambda: AdaptiveServoTransition(
                    settings.passive_duration_min_s,
                    settings.passive_duration_max_s,
                    settings.passive_max_joint_speed_rad_s,
                ),
            ),
            TransitionRule("fixstand", "passive", lambda: SmoothServoTransition(settings.passive_gain_ramp_s)),
            TransitionRule("fixstand", "walk", lambda: SmoothServoTransition(settings.policy_duration_s)),
            TransitionRule("walk", "fixstand", lambda: SmoothServoTransition(settings.policy_duration_s)),
        )
        super().__init__(
            joint_names=joint_names,
            baseline=BaselineStates(passive=passive, fixstand=fixstand, walk=walk),
            preparation=JointServoPreparationState(robot),
            skills=(
                SkillModule(
                    name="wbt",
                    state=wbt,
                    preparation=wbt_preparation,
                    request_intent="request_wbt",
                ),
            ),
            initial_state=initial_state,
            baseline_transitions=baseline_transitions,
            passive_transition=lambda: SmoothServoTransition(settings.passive_gain_ramp_s),
        )


__all__ = [
    "G1FixStandState",
    "G1HotSwappingController",
    "G1PassiveState",
    "G1TransitionSettings",
    "G1WalkState",
    "G1WbtState",
]
