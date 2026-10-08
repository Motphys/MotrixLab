# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unified control loop and fake backend contract tests."""

from collections.abc import Callable

import numpy as np
import pytest
from fake_robot import FakeRobotInterface

from motrix_deploy.artifact import DeploymentManifest
from motrix_deploy.contracts import (
    FloatArray,
    JointControlMode,
    JointServoCommand,
    JointTorqueCommand,
    RobotCapabilities,
    RobotCommand,
    RobotSpec,
    RobotState,
    float32_array,
)
from motrix_deploy.errors import LieDownRequestedError, ValidationError
from motrix_deploy.policy import PolicyRuntime
from motrix_deploy.runtime import (
    ControlSession,
    FixedStepScheduler,
    HardwareRuntime,
    PolicyContext,
    RealtimeScheduler,
)
from motrix_deploy.task import DeployTask
from motrix_env_core.input import CommandBinding, ConstantPlanarVelocityBinding, PlanarVelocityCommand


class ConstantPolicy(PolicyRuntime):
    def __init__(self, input_size: int, action: FloatArray) -> None:
        self._input_size = input_size
        self._action = float32_array(action, path="test.action", shape=(action.size,))
        self.observations: list[FloatArray] = []

    def infer(self, observation: FloatArray) -> FloatArray:
        value = float32_array(observation, path="test.observation", shape=(self._input_size,))
        self.observations.append(value)
        return self._action


class _TestDeployTask(DeployTask[PlanarVelocityCommand]):
    def __init__(self, manifest: DeploymentManifest) -> None:
        self.robot = manifest.robot
        self._previous_action = np.zeros(self.robot.joint_count, dtype=np.float32)

    def reset(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> None:
        del state, context
        self._previous_action = np.zeros(2, dtype=np.float32)

    def build_observation(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> FloatArray:
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

    def validate_command(self, command: PlanarVelocityCommand) -> None:
        if not isinstance(command, PlanarVelocityCommand) or command.batch_size != 1:
            raise ValueError("expected one planar velocity command")


class _TorqueDeployTask(_TestDeployTask):
    def process_action(self, action: FloatArray) -> JointTorqueCommand:
        return JointTorqueCommand(np.array(action, dtype=np.float32, copy=True))


@pytest.mark.parametrize("requires_enable", [False, True])
def test_torque_session_success_is_deterministic_without_fake_physics(
    manifest_factory: Callable[[], DeploymentManifest],
    requires_enable: bool,
) -> None:
    manifest = manifest_factory()
    traces = []
    for _ in range(2):
        backend = EnablingBackend(manifest.robot) if requires_enable else FakeRobotInterface(manifest.robot)
        if requires_enable:
            backend._capabilities = RobotCapabilities(
                control_modes=(JointControlMode.SERVO, JointControlMode.TORQUE),
                state_fields=backend.capabilities.state_fields,
                requires_enable=True,
            )
        session = _session(manifest, backend)
        session.task = _TorqueDeployTask(manifest)
        assert session.start()
        for step in range(3):
            assert session.tick(elapsed_time_s=step * manifest.control.period_s)
        session.stop()
        result = session.result(steps=3)
        assert result.success
        if requires_enable:
            assert backend.events.index("enable") < backend.events.index("write")
        assert all(command.mode is JointControlMode.TORQUE for command in backend.commands)
        np.testing.assert_array_equal(backend._position, manifest.robot.default_joint_position)
        np.testing.assert_array_equal(backend._velocity, np.zeros(2, dtype=np.float32))
        assert backend._sample_time_ns == 3 * backend._period_ns
        traces.append(result.trace_sha256)
    assert traces[0] == traces[1]


@pytest.mark.parametrize("requires_enable", [False, True])
def test_unsupported_command_mode_prevents_write_after_enable(
    manifest_factory: Callable[[], DeploymentManifest],
    requires_enable: bool,
) -> None:
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot) if requires_enable else FakeRobotInterface(manifest.robot)
    backend._capabilities = RobotCapabilities(
        control_modes=(JointControlMode.SERVO,),
        state_fields=backend.capabilities.state_fields,
        requires_enable=requires_enable,
    )
    session = _session(manifest, backend)
    session.task = _TorqueDeployTask(manifest)
    assert session.start()
    assert not session.tick(elapsed_time_s=0.0)
    session.stop()
    result = session.result(steps=1)
    assert result.exit_reason == "invalid_action"
    assert "backend.control_modes" in result.error
    assert ("enable" in backend.events) is requires_enable
    assert "write" not in backend.events
    assert backend.commands == []
    assert backend.events[-2:] == ["stop", "close"]


class _AlternatingDeployTask(_TestDeployTask):
    def __init__(self, manifest: DeploymentManifest) -> None:
        super().__init__(manifest)
        self._next_mode = JointControlMode.SERVO

    def process_action(self, action: FloatArray) -> RobotCommand:
        if self._next_mode is JointControlMode.SERVO:
            self._next_mode = JointControlMode.TORQUE
            return super().process_action(action)
        self._next_mode = JointControlMode.SERVO
        return JointTorqueCommand(np.array(action, dtype=np.float32, copy=True))


@pytest.mark.parametrize("requires_enable", [False, True])
def test_one_session_alternates_servo_and_torque_commands(
    manifest_factory: Callable[[], DeploymentManifest],
    requires_enable: bool,
) -> None:
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot) if requires_enable else FakeRobotInterface(manifest.robot)
    backend._capabilities = RobotCapabilities(
        control_modes=(JointControlMode.SERVO, JointControlMode.TORQUE),
        state_fields=backend.capabilities.state_fields,
        requires_enable=requires_enable,
    )
    session = _session(manifest, backend)
    session.task = _AlternatingDeployTask(manifest)
    assert session.start()
    for step in range(4):
        assert session.tick(elapsed_time_s=step * session.period_s)
    session.stop()
    assert session.result(steps=4).success
    expected_modes = [JointControlMode.SERVO, JointControlMode.TORQUE] * 2
    if requires_enable:
        assert backend.events.count("enable") == 1
    assert [command.mode for command in backend.commands] == expected_modes
    assert backend.events.count("open") == 1
    assert backend.events.count("write") == 4


def test_switch_to_unsupported_mode_preserves_successful_tick(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    backend._capabilities = RobotCapabilities(
        control_modes=(JointControlMode.SERVO,),
        state_fields=backend.capabilities.state_fields,
    )
    session = _session(manifest, backend)
    session.task = _AlternatingDeployTask(manifest)
    assert session.start()
    assert session.tick(elapsed_time_s=0.0)
    trace = session.result(steps=1).trace_sha256
    assert not session.tick(elapsed_time_s=session.period_s)
    session.stop()
    result = session.result(steps=2)
    assert result.exit_reason == "invalid_action"
    assert "backend.control_modes" in result.error
    assert result.completed_steps == 1
    assert result.trace_sha256 == trace
    assert len(backend.commands) == 1
    assert backend.commands[0].mode is JointControlMode.SERVO
    assert backend.events.count("write") == 1


@pytest.mark.parametrize("torque", [[1.0], [100.0, 0.0]])
def test_fake_validates_torque_shape_and_limits(
    manifest_factory: Callable[[], DeploymentManifest],
    torque: list[float],
) -> None:
    from motrix_deploy.errors import ValidationError

    backend = FakeRobotInterface(manifest_factory().robot)
    backend.open()
    try:
        with pytest.raises(ValidationError, match="command.torque"):
            backend.write_command(JointTorqueCommand(np.array(torque, dtype=np.float32)))
        assert backend.commands == []
        assert "write" not in backend.events
    finally:
        backend.close()


def _loop(
    manifest: DeploymentManifest,
    backend: FakeRobotInterface,
    *,
    command_binding: CommandBinding[PlanarVelocityCommand] | None = None,
) -> tuple[HardwareRuntime, ControlSession]:
    runtime = HardwareRuntime(backend, scheduler=FixedStepScheduler(manifest.control.period_s))
    return runtime, _session(manifest, backend, command_binding=command_binding)


def test_fake_interface_binds_readonly_spec_at_construction(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    spec = manifest_factory().robot
    backend = FakeRobotInterface(spec)

    assert backend.spec is spec
    with pytest.raises(AttributeError):
        backend.spec = spec
    backend.open()
    np.testing.assert_array_equal(backend.read_state(0.1).joint_position, spec.default_joint_position)
    backend.close()


def _session(
    manifest: DeploymentManifest,
    backend: FakeRobotInterface,
    *,
    command_binding: CommandBinding[PlanarVelocityCommand] | None = None,
) -> ControlSession:
    return ControlSession(
        robot=backend,
        task=_TestDeployTask(manifest),
        policy=ConstantPolicy(manifest.policy.input.shape[1], np.array([0.4, -0.4], dtype=np.float32)),
        command_binding=command_binding or ConstantPlanarVelocityBinding([0.5, 0.0, 0.0]),
        period_s=manifest.control.period_s,
        state_timeout_s=manifest.control.state_timeout_s,
    )


class _SimulationOwnedFakeRobotInterface(FakeRobotInterface):
    """Cache commands while a separate simulation callback advances state."""

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        if self._fail_write_at == self._write_count:
            raise RuntimeError("injected write failure")
        assert command.joint_position.shape == self._position.shape
        self.commands.append(command)
        self.events.append("write")
        self._write_count += 1

    def advance(self) -> None:
        self._require_open()
        self.events.append("advance")
        previous = self._position.copy()
        self._position = self.commands[-1].joint_position.copy()
        self._velocity = ((self._position - previous) / np.float32(self._period_s)).astype(np.float32)
        self._sample_time_ns += self._period_ns


class _TrackingPolicy(ConstantPolicy):
    def __init__(self, input_size: int, action: FloatArray, events: list[str]) -> None:
        super().__init__(input_size, action)
        self.events = events

    def infer(self, observation: FloatArray) -> FloatArray:
        self.events.append("infer")
        return super().infer(observation)


def test_control_session_allows_external_simulation_advance_after_each_tick(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = _SimulationOwnedFakeRobotInterface(manifest.robot)
    loop = _session(manifest, backend)
    policy = _TrackingPolicy(manifest.policy.input.shape[1], np.array([0.4, -0.4], dtype=np.float32), backend.events)
    loop.policy = policy

    assert loop.start()
    try:
        for step in range(3):
            assert loop.tick(elapsed_time_s=step * manifest.control.period_s)
            backend.advance()
    finally:
        loop.stop()
    result = loop.result(steps=3)

    assert result.success
    assert result.completed_steps == 3
    assert result.simulation_time_s == pytest.approx(3 * manifest.control.period_s)
    assert backend.events == ["open", "read", *(["read", "infer", "write", "advance"] * 3), "stop", "close"]
    assert backend._sample_time_ns == 3 * backend._period_ns
    np.testing.assert_array_equal(policy.observations[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(
        policy.observations[1][:2], backend.commands[0].joint_position - manifest.robot.default_joint_position
    )
    np.testing.assert_array_equal(backend._position, backend.commands[-1].joint_position)


def test_standalone_driver_does_not_advance_simulation(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = _SimulationOwnedFakeRobotInterface(manifest.robot)
    runtime, loop = _loop(manifest, backend)

    result = runtime.run(loop, steps=3)

    assert result.success
    assert backend._sample_time_ns == 0
    assert "advance" not in backend.events
    np.testing.assert_array_equal(backend._position, manifest.robot.default_joint_position)
    for observation in loop.policy.observations:
        np.testing.assert_array_equal(observation, np.zeros(4, dtype=np.float32))


def test_external_simulation_failure_preserves_completed_control_tick(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = _SimulationOwnedFakeRobotInterface(manifest.robot)
    session = _session(manifest, backend)
    assert session.start()
    assert session.tick(elapsed_time_s=0.0)
    backend.events.append("advance")
    session.fail(RuntimeError("simulation advance failed"))
    assert not session.active
    assert not session.tick(elapsed_time_s=manifest.control.period_s)
    session.stop()
    result = session.result(steps=3)

    assert not result.success
    assert result.exit_reason == "backend_error"
    assert result.error == "simulation advance failed"
    assert result.completed_steps == 1
    assert result.simulation_time_s == manifest.control.period_s
    assert "write" in result.latency
    assert "loop" in result.latency
    assert backend.events == ["open", "read", "read", "write", "advance", "stop", "close"]
    assert len(backend.commands) == 1


@pytest.mark.parametrize("response", [1.0, 0.5])
def test_control_loop_runs_with_different_robot_responses(
    manifest_factory: Callable[[], DeploymentManifest],
    response: float,
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot, response=response)

    runtime, control = _loop(manifest, backend)
    result = runtime.run(control, steps=5)

    assert result.success
    assert result.exit_reason == "completed"
    assert result.completed_steps == 5
    assert result.simulation_time_s == pytest.approx(0.1)
    assert len(backend.commands) == 5
    assert backend.events[-2:] == ["stop", "close"]
    assert set(result.latency) == {"input", "read", "observation", "inference", "action", "write", "loop"}


def test_fixed_step_rollout_is_deterministic(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory()
    first_backend = FakeRobotInterface(manifest.robot)
    second_backend = FakeRobotInterface(manifest.robot)

    first_runtime, first_loop = _loop(manifest, first_backend)
    second_runtime, second_loop = _loop(manifest, second_backend)
    first = first_runtime.run(first_loop, steps=4)
    second = second_runtime.run(second_loop, steps=4)

    assert (first.success, first.exit_reason, first.completed_steps) == (
        second.success,
        second.exit_reason,
        second.completed_steps,
    )
    for first_command, second_command in zip(first_backend.commands, second_backend.commands, strict=True):
        np.testing.assert_array_equal(first_command.joint_position, second_command.joint_position)
    assert isinstance(first_loop.policy, ConstantPolicy)
    assert isinstance(second_loop.policy, ConstantPolicy)
    for first_observation, second_observation in zip(
        first_loop.policy.observations,
        second_loop.policy.observations,
        strict=True,
    ):
        np.testing.assert_array_equal(first_observation, second_observation)


class _FailingBinding(CommandBinding[PlanarVelocityCommand]):
    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        del batch_size
        raise RuntimeError("input disconnected")


class _TrackingBinding(CommandBinding[PlanarVelocityCommand]):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        self.batch_sizes.append(batch_size)
        return PlanarVelocityCommand(np.zeros((batch_size, 3), dtype=np.float32))


class _InterruptingBinding(CommandBinding[PlanarVelocityCommand]):
    def __init__(self, interrupt_at: int) -> None:
        self._interrupt_at = interrupt_at
        self.read_count = 0

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        if self.read_count == self._interrupt_at:
            raise KeyboardInterrupt
        self.read_count += 1
        return PlanarVelocityCommand(np.zeros((batch_size, 3), dtype=np.float32))


def test_control_loop_reads_one_singleton_command_batch_per_tick(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    binding = _TrackingBinding()
    runtime, loop = _loop(manifest, FakeRobotInterface(manifest.robot), command_binding=binding)
    loop.policy = ConstantPolicy(manifest.policy.input.shape[1], np.zeros(2, dtype=np.float32))

    result = runtime.run(loop, steps=3)

    assert result.success
    assert binding.batch_sizes == [1, 1, 1]


def test_control_loop_runs_without_a_step_bound_until_interrupted(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    binding = _InterruptingBinding(interrupt_at=3)
    runtime, loop = _loop(manifest, FakeRobotInterface(manifest.robot), command_binding=binding)
    loop.policy = ConstantPolicy(manifest.policy.input.shape[1], np.zeros(2, dtype=np.float32))

    result = runtime.run(loop)

    assert not result.success
    assert result.exit_reason == "interrupted"
    assert result.completed_steps == 3


def test_command_binding_failure_returns_input_error(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    runtime, loop = _loop(manifest, backend, command_binding=_FailingBinding())
    loop.policy = ConstantPolicy(manifest.policy.input.shape[1], np.zeros(2, dtype=np.float32))

    result = runtime.run(loop, steps=1)

    assert not result.success
    assert result.exit_reason == "input_error"
    assert result.error == "input disconnected"
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize(
    ("backend_kwargs", "reason", "completed_steps"),
    [
        ({"fail_read_at": 0}, "state_timeout", 0),
        ({"unhealthy_at": 1}, "backend_unhealthy", 0),
        ({"fail_write_at": 2}, "backend_write_error", 2),
    ],
)
def test_backend_failures_return_structured_results_and_cleanup(
    manifest_factory: Callable[[], DeploymentManifest],
    backend_kwargs: dict[str, int],
    reason: str,
    completed_steps: int,
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot, **backend_kwargs)

    runtime, control = _loop(manifest, backend)
    result = runtime.run(control, steps=5)

    assert not result.success
    assert result.exit_reason == reason
    assert result.completed_steps == completed_steps
    assert backend.events[-2:] == ["stop", "close"]


def test_backend_open_validation_failure_cleans_up_before_first_command(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    class InvalidRobot(FakeRobotInterface):
        def open(self) -> None:
            self.events.append("open")
            raise ValidationError("backend.joint_names", "matching robot joints", "missing_joint")

    manifest = manifest_factory()
    backend = InvalidRobot(manifest.robot)

    runtime, control = _loop(manifest, backend)
    result = runtime.run(control, steps=2)

    assert result.exit_reason == "validation_error"
    assert result.completed_steps == 0
    assert backend.commands == []
    assert backend.events == ["open", "stop", "close"]


class InterruptingBackend(FakeRobotInterface):
    def read_state(self, timeout_s: float):
        del timeout_s
        raise KeyboardInterrupt


def test_user_interrupt_returns_result_and_cleans_up(manifest_factory: Callable[[], DeploymentManifest]) -> None:
    manifest = manifest_factory()
    backend = InterruptingBackend(manifest.robot)

    runtime, control = _loop(manifest, backend)
    result = runtime.run(control, steps=2)

    assert result.exit_reason == "interrupted"
    assert backend.events[-2:] == ["stop", "close"]


class InvalidStateBackend(FakeRobotInterface):
    def read_state(self, timeout_s: float):
        del timeout_s
        return None


def test_invalid_state_fails_and_lifecycle_cleanup_is_idempotent(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = InvalidStateBackend(manifest.robot)

    runtime, control = _loop(manifest, backend)
    result = runtime.run(control, steps=2)
    events_after_run = list(backend.events)
    backend.stop()
    backend.close()

    assert result.exit_reason == "invalid_state"
    assert result.completed_steps == 0
    assert backend.events == events_after_run


def test_realtime_scheduler_uses_absolute_deadlines() -> None:
    now = [1_000_000_000]
    sleeps: list[float] = []

    def clock_ns() -> int:
        return now[0]

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        now[0] += round(duration * 1e9)

    scheduler = RealtimeScheduler(0.02, clock_ns=clock_ns, sleep=sleep)
    scheduler.reset()
    scheduler.wait(1)
    now[0] += 5_000_000
    scheduler.wait(2)

    assert sleeps == pytest.approx([0.02, 0.015])
    assert scheduler.elapsed_time_s(2) == pytest.approx(0.04)


class EnablingBackend(FakeRobotInterface):
    def __init__(self, spec: RobotSpec, *, sample_period_s: float = 0.02) -> None:
        super().__init__(spec, sample_period_s=sample_period_s)
        self._capabilities = RobotCapabilities(
            control_modes=("joint_servo",),
            state_fields=self._capabilities.state_fields,
            max_command_rate_hz=1.0 / sample_period_s,
            requires_enable=True,
        )
        self.enabled = False

    def enable(self) -> None:
        assert "write" not in self.events
        self.events.append("enable")
        self.enabled = True

    def write_command(self, command: RobotCommand) -> None:
        assert self.enabled
        super().write_command(command)


class LieDownBackend(FakeRobotInterface):
    def write_command(self, command: RobotCommand) -> None:
        del command
        raise LieDownRequestedError("remote B requested lie-down shutdown")


def test_lie_down_request_returns_structured_result_and_cleans_up(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = LieDownBackend(manifest.robot)

    runtime, control = _loop(manifest, backend)
    result = runtime.run(control, steps=1)

    assert not result.success
    assert result.exit_reason == "lie_down"
    assert result.completed_steps == 0
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize("terminate_at", [0, 1, 3])
def test_task_termination_prevents_inference_and_commands(
    manifest_factory: Callable[[], DeploymentManifest], terminate_at: int
) -> None:
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot) if terminate_at == 0 else FakeRobotInterface(manifest.robot)
    runtime, loop = _loop(manifest, backend)
    checks = []

    def check(state: RobotState) -> str | None:
        checks.append(state)
        return "task fell" if len(checks) - 1 == terminate_at else None

    loop.task.check_termination = check
    result = runtime.run(loop, steps=5)

    assert result.exit_reason == "task_terminated"
    assert result.error == "task fell"
    assert not result.success
    assert result.completed_steps == max(0, terminate_at - 1)
    assert len(loop.policy.observations) == result.completed_steps
    assert len(backend.commands) == result.completed_steps
    if terminate_at == 0:
        assert "enable" not in backend.events
    assert backend.events[-2:] == ["stop", "close"]


def test_task_termination_checks_state_after_enable_before_inference(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot)
    runtime, loop = _loop(manifest, backend)
    loop.task.check_termination = lambda state: "fell after enable" if backend.enabled else None
    result = runtime.run(loop, steps=1)
    assert result.exit_reason == "task_terminated"
    assert result.completed_steps == 0
    assert backend.commands == []
    assert loop.policy.observations == []
    assert backend.events == ["open", "read", "enable", "read", "stop", "close"]


def test_task_termination_hook_failure_has_task_error_and_cleanup(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    runtime, loop = _loop(manifest, backend)

    def fail(state: RobotState) -> str | None:
        raise RuntimeError("termination failed")

    loop.task.check_termination = fail
    result = runtime.run(loop, steps=1)
    assert result.exit_reason == "task_termination_error"
    assert result.error == "termination failed"
    assert backend.commands == []
    assert backend.events[-2:] == ["stop", "close"]


def test_required_enable_completes_before_first_policy_write(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot)

    runtime, control = _loop(manifest, backend)
    result = runtime.run(control, steps=1)

    assert result.success
    assert backend.events[:6] == ["open", "read", "enable", "read", "read", "write"]


def test_enable_does_not_process_a_synthetic_policy_action(manifest_factory) -> None:
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot)
    session = _session(manifest, backend)
    actions = []
    process_action = session.task.process_action

    def record_action(action):
        actions.append(np.array(action, copy=True))
        return process_action(action)

    session.task.process_action = record_action
    assert session.start()
    assert actions == []
    assert session.tick(elapsed_time_s=0.0)
    assert len(actions) == 1
    session.stop()
    assert session.result(steps=1).success


def test_direct_session_lifecycle_is_single_use_and_cleanup_is_explicit(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    session = _session(manifest, backend)
    assert not session.active
    assert not session.tick(elapsed_time_s=0.0)
    assert session.start()
    assert session.start()
    assert backend.events == ["open", "read"]
    assert session.tick(elapsed_time_s=7.25)
    assert session.completed_steps == 1
    assert session.active
    session.stop()
    events = list(backend.events)
    session.stop()
    assert not session.start()
    assert not session.tick(elapsed_time_s=8.0)
    assert not session.active
    assert backend.events == events
    result = session.result(steps=1, overrun_count=2)
    assert result.success
    assert result.overrun_count == 2
    assert result.simulation_time_s == manifest.control.period_s
    assert not session.result(steps=None).success
    assert not session.result(steps=2).success
    assert session.result(steps=1).wall_time_s == result.wall_time_s


def test_failed_start_records_error_without_implicit_cleanup(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot, fail_read_at=0)
    session = _session(manifest, backend)
    assert not session.start()
    assert not session.start()
    assert not session.active
    assert not session.tick(elapsed_time_s=0.0)
    assert "stop" not in backend.events
    session.stop()
    assert session.result(steps=1).exit_reason == "state_timeout"
    assert backend.events[-2:] == ["stop", "close"]


class _StatefulTask(_TestDeployTask):
    def __init__(self, manifest: DeploymentManifest) -> None:
        super().__init__(manifest)
        self.contexts: list[PolicyContext[PlanarVelocityCommand]] = []
        self.reset_contexts: list[PolicyContext[PlanarVelocityCommand]] = []

    def reset(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> None:
        super().reset(state, context)
        self.reset_contexts.append(context)

    def build_observation(self, state: RobotState, context: PolicyContext[PlanarVelocityCommand]) -> FloatArray:
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


@pytest.mark.parametrize("enabling", [False, True])
def test_stateful_session_and_standalone_driver_have_identical_trace_and_reset_semantics(
    manifest_factory: Callable[[], DeploymentManifest],
    enabling: bool,
) -> None:
    manifest = manifest_factory()
    backend_type = EnablingBackend if enabling else FakeRobotInterface
    direct = _session(manifest, backend_type(manifest.robot))
    runtime, driver = _loop(manifest, backend_type(manifest.robot))
    for executor in (direct, driver):
        executor.task = _StatefulTask(manifest)
        executor.policy = _StatefulPolicy()
    assert direct.start()
    for step in range(4):
        assert direct.tick(elapsed_time_s=step * direct.period_s)
    direct.stop()
    direct_result = direct.result(steps=4)
    driver_result = runtime.run(driver, steps=4)
    assert direct_result.success and driver_result.success
    assert direct_result.trace_sha256 == driver_result.trace_sha256
    assert direct.robot.events == driver.robot.events
    for executor in (direct, driver):
        assert executor.policy.reset_count == (2 if enabling else 1)
        assert len(executor.task.reset_contexts) == (3 if enabling else 1)
        assert [context.step for context in executor.task.contexts] == list(range(4))
        assert [context.elapsed_time_s for context in executor.task.contexts] == pytest.approx(
            [step * executor.period_s for step in range(4)]
        )
    for first, second in zip(direct.policy.observations, driver.policy.observations, strict=True):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(direct.robot.commands, driver.robot.commands, strict=True):
        np.testing.assert_array_equal(first.joint_position, second.joint_position)


def test_direct_session_uses_supplied_simulation_relative_context_time(
    manifest_factory: Callable[[], DeploymentManifest],
) -> None:
    manifest = manifest_factory()
    session = _session(manifest, FakeRobotInterface(manifest.robot))
    task = _StatefulTask(manifest)
    session.task = task
    assert session.start()
    assert session.tick(elapsed_time_s=0.125)
    assert session.tick(elapsed_time_s=0.75)
    session.stop()
    assert task.reset_contexts[0].elapsed_time_s == 0.125
    assert [context.elapsed_time_s for context in task.contexts] == [0.125, 0.75]
    assert session.result(steps=2).simulation_time_s == 2 * session.period_s


@pytest.mark.parametrize("phase", ["reset", "wait"])
def test_driver_scheduler_failure_is_recorded_and_cleans_up(
    manifest_factory: Callable[[], DeploymentManifest],
    phase: str,
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    runtime, loop = _loop(manifest, backend)

    def fail(*args) -> None:
        raise RuntimeError("scheduler failed")

    setattr(runtime.scheduler, phase, fail)
    result = runtime.run(loop, steps=1)
    assert result.exit_reason == "backend_error"
    assert result.error == "scheduler failed"
    assert result.completed_steps == 0
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize("prior_failure", [False, True])
def test_cleanup_errors_do_not_replace_existing_failure_and_close_is_attempted(
    manifest_factory: Callable[[], DeploymentManifest],
    prior_failure: bool,
) -> None:
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    session = _session(manifest, backend)
    assert session.start()
    assert session.tick(elapsed_time_s=0.0)

    def fail_stop() -> None:
        backend.events.append("stop-failed")
        raise RuntimeError("cleanup failed")

    backend.stop = fail_stop
    if prior_failure:
        session.fail(LieDownRequestedError("lie down"))
    session.stop()
    session.stop()
    result = session.result(steps=1)
    assert result.exit_reason == ("lie_down" if prior_failure else "cleanup_error")
    assert result.error == ("lie down" if prior_failure else "cleanup failed")
    assert backend.events[-2:] == ["stop-failed", "close"]
