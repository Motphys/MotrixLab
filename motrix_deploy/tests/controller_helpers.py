# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable policy, policy I/O and bound hardware runtime helpers for tests."""

import numpy as np

from motrix_deploy.artifact import DeploymentManifest
from motrix_deploy.contracts import (
    FloatArray,
    JointControlMode,
    JointServoCommand,
    JointTorqueCommand,
    RobotCommand,
    RobotState,
    float32_array,
)
from motrix_deploy.policy import Policy
from motrix_deploy.policy.processing import PolicyProcessor
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import ControlSession, PolicyController
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy.runtime.scheduler import FixedStepScheduler
from motrix_env_core.input import ConstantPlanarVelocityBinding, PlanarVelocityCommand


class ConstantPolicy(Policy):
    def __init__(self, input_size: int, action: FloatArray) -> None:
        self._input_size = input_size
        self._action = float32_array(action, path="test.action", shape=(action.size,))
        self.observations: list[FloatArray] = []

    def infer(self, observation: FloatArray) -> FloatArray:
        value = float32_array(observation, path="test.observation", shape=(self._input_size,))
        self.observations.append(value)
        return self._action


class _TestPolicyProcessor(PolicyProcessor[PlanarVelocityCommand]):
    def __init__(self, manifest: DeploymentManifest) -> None:
        self.robot = manifest.robot
        self._previous_action = np.zeros(self.robot.joint_count, dtype=np.float32)

    def reset(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> None:
        del state, context
        self._previous_action = np.zeros(2, dtype=np.float32)

    def build_observation(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> FloatArray:
        del context
        return np.concatenate((state.joint_position - self.robot.default_joint_position, state.joint_velocity)).astype(
            np.float32
        )

    def process_action(self, action: FloatArray) -> RobotCommand:
        executed = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._previous_action = executed
        zeros = np.zeros(2, dtype=np.float32)
        return JointServoCommand(
            joint_position=self.robot.default_joint_position + np.float32(0.25) * executed,
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=np.full(2, 35.0, dtype=np.float32),
            kd=np.full(2, 0.5, dtype=np.float32),
        )

    def check_termination(self, state: RobotState) -> str | None:
        return None

    def validate_command(self, command: PlanarVelocityCommand | None) -> None:
        if not isinstance(command, PlanarVelocityCommand) or command.batch_size != 1:
            raise ValueError("expected one planar velocity command")


class _TorquePolicyProcessor(_TestPolicyProcessor):
    def process_action(self, action: FloatArray) -> JointTorqueCommand:
        return JointTorqueCommand(np.array(action, dtype=np.float32, copy=True))


class _AlternatingPolicyProcessor(_TestPolicyProcessor):
    def __init__(self, manifest: DeploymentManifest) -> None:
        super().__init__(manifest)
        self._next_mode = JointControlMode.SERVO

    def process_action(self, action: FloatArray) -> RobotCommand:
        if self._next_mode is JointControlMode.SERVO:
            self._next_mode = JointControlMode.TORQUE
            return super().process_action(action)
        self._next_mode = JointControlMode.SERVO
        return JointTorqueCommand(np.array(action, dtype=np.float32, copy=True))


class _StatefulPolicyProcessor(_TestPolicyProcessor):
    def __init__(self, manifest: DeploymentManifest) -> None:
        super().__init__(manifest)
        self.contexts: list[ControlContext[PlanarVelocityCommand]] = []
        self.reset_contexts: list[ControlContext[PlanarVelocityCommand]] = []

    def reset(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> None:
        super().reset(state, context)
        self.reset_contexts.append(context)

    def build_observation(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> FloatArray:
        self.contexts.append(context)
        observation = super().build_observation(state, context)
        observation[:2] += self._previous_action
        observation[2:] += np.float32(context.elapsed_time_s)
        return observation


class _StatefulPolicy(ConstantPolicy):
    def __init__(self) -> None:
        super().__init__(4, np.zeros(2, dtype=np.float32))
        self.reset_count = 0
        self.counter = 99

    def reset(self) -> None:
        self.reset_count += 1
        self.counter = 0

    def infer(self, observation: FloatArray) -> FloatArray:
        super().infer(observation)
        self.counter += 1
        return np.tanh(observation[:2] + np.float32(self.counter * 0.1)).astype(np.float32)


def runtime_for(manifest, backend, *, steps=3, task=None, model=None, binding=None, scheduler=None):
    task = task or _TestPolicyProcessor(manifest)
    model = model or ConstantPolicy(4, np.array([0.4, -0.4], np.float32))
    controller = PolicyController(task, model, steps=steps)
    runtime = HardwareRuntime(backend, scheduler=scheduler or FixedStepScheduler(manifest.control.period_s))
    session = ControlSession(
        controller=controller,
        robot=backend,
        command_binding=binding or ConstantPlanarVelocityBinding((0.5, 0, 0)),
        period_s=manifest.control.period_s,
        state_timeout_s=manifest.control.state_timeout_s,
    )
    runtime.bind_session(session)
    return runtime, controller
