# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot controller execution and runtime safety boundary contracts."""

import time
from dataclasses import replace

import numpy as np
import pytest
from controller_helpers import (
    ConstantPolicy,
    _AlternatingPolicyProcessor,
    _StatefulPolicy,
    _StatefulPolicyProcessor,
    _TestPolicyProcessor,
    _TorquePolicyProcessor,
    runtime_for,
)
from fake_robot import FakeRobotInterface

from motrix_deploy.contracts import (
    JointControlMode,
    JointTorqueCommand,
    RobotCapabilities,
    RobotCommand,
    RobotSpec,
)
from motrix_deploy.errors import LieDownRequestedError, ValidationError
from motrix_deploy.runtime.control import ControlSession, PolicyController
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.runtime.result import LatencySummary, RolloutResult
from motrix_deploy.runtime.scheduler import FixedStepScheduler, RealtimeScheduler
from motrix_env_core.input import CommandBinding, PlanarVelocityCommand


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


@pytest.mark.parametrize("task_type", [_TestPolicyProcessor, _TorquePolicyProcessor, _AlternatingPolicyProcessor])
@pytest.mark.parametrize("requires_enable", [False, True])
def test_runtime_executes_deterministic_controller_commands(manifest_factory, task_type, requires_enable):
    manifest = manifest_factory()
    runs = []
    for _ in range(2):
        backend = EnablingBackend(manifest.robot) if requires_enable else FakeRobotInterface(manifest.robot)
        backend._capabilities = replace(
            backend.capabilities, control_modes=(JointControlMode.SERVO, JointControlMode.TORQUE)
        )
        runtime, controller = runtime_for(manifest, backend, task=task_type(manifest), steps=4)
        result = runtime.run()
        assert result.success and result.completed_steps == 4
        assert len(backend.commands) == 4
        assert backend.events[-2:] == ["stop", "close"]
        if requires_enable:
            assert backend.events.index("enable") < backend.events.index("write")
        if task_type is _TorquePolicyProcessor:
            assert all(command.mode is JointControlMode.TORQUE for command in backend.commands)
            np.testing.assert_array_equal(backend._position, manifest.robot.default_joint_position)
        if task_type is _AlternatingPolicyProcessor:
            assert [command.mode for command in backend.commands] == [
                JointControlMode.SERVO,
                JointControlMode.TORQUE,
            ] * 2
        runs.append((result, controller, backend))
    assert runs[0][0].trace_sha256 == runs[1][0].trace_sha256
    for first, second in zip(runs[0][1].policy.observations, runs[1][1].policy.observations, strict=True):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("task_type,completed", [(_TorquePolicyProcessor, 0), (_AlternatingPolicyProcessor, 1)])
def test_unsupported_mode_is_rejected_before_backend_write(manifest_factory, task_type, completed):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    backend._capabilities = replace(backend.capabilities, control_modes=(JointControlMode.SERVO,))
    runtime, _ = runtime_for(manifest, backend, task=task_type(manifest))
    result = runtime.run()
    assert not result.success
    assert result.completed_steps == completed
    assert len(backend.commands) == completed
    assert "control mode" in result.error
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize("response", [1.0, 0.5])
def test_runtime_accepts_different_measured_robot_responses(manifest_factory, response):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot, response=response)
    runtime, _ = runtime_for(manifest, backend, steps=5)
    result = runtime.run()
    assert result.success
    assert isinstance(result, RolloutResult)
    assert result.completed_steps == 5
    assert result.wall_time_s >= 0
    report = result.to_dict()
    assert report["success"] and report["completed_steps"] == 5
    assert report["wall_time_s"] == result.wall_time_s
    assert isinstance(result.latency["read"], LatencySummary)
    assert report["latency"]["read"]["mean_ms"] == result.latency["read"].mean_ms
    assert len(backend.commands) == 5
    assert set(result.latency) >= {"input", "read", "observation", "inference", "action", "write", "loop"}


def test_runtime_reads_singleton_commands_for_policy_ticks(manifest_factory):
    manifest = manifest_factory()
    binding = _TrackingBinding()
    runtime, _ = runtime_for(manifest, FakeRobotInterface(manifest.robot), binding=binding)
    assert runtime.run().success
    assert all(size == 1 for size in binding.batch_sizes)
    assert len(binding.batch_sizes) >= 3


def test_unbounded_controller_runs_until_input_interrupt(manifest_factory):
    manifest = manifest_factory()
    runtime, _ = runtime_for(manifest, FakeRobotInterface(manifest.robot), steps=None, binding=_InterruptingBinding(3))
    result = runtime.run()
    assert not result.success
    assert result.exit_reason == "interrupted"
    assert result.completed_steps == 3


def test_input_failure_is_structured_and_cleans_backend(manifest_factory):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    runtime, _ = runtime_for(manifest, backend, binding=_FailingBinding())
    result = runtime.run()
    assert result.exit_reason == "input_error"
    assert result.error == "input disconnected"
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize(
    "options,reason,count",
    [
        ({"fail_read_at": 0}, "state_timeout", 0),
        ({"unhealthy_at": 1}, "backend_unhealthy", 0),
        ({"fail_write_at": 2}, "backend_write_error", 2),
    ],
)
def test_backend_failure_preserves_result_and_cleanup(manifest_factory, options, reason, count):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot, **options)
    runtime, _ = runtime_for(manifest, backend, steps=5)
    result = runtime.run()
    assert not result.success
    assert result.exit_reason == reason
    assert result.completed_steps == count
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("invalid", "invalid_state"),
        ("stale", "state_timeout"),
        ("backwards", "invalid_state"),
    ],
)
def test_runtime_validates_state_contract_and_freshness(manifest_factory, kind, reason):
    manifest = manifest_factory()

    class StateBackend(FakeRobotInterface):
        def read_state(self, timeout_s):
            state = super().read_state(timeout_s)
            if kind == "invalid":
                return None
            if kind == "stale":
                return replace(state, receive_time_ns=time.monotonic_ns() - 1_000_000_000)
            if self._read_count > 1:
                return replace(state, sample_time_ns=0)
            return replace(state, sample_time_ns=10)

    backend = StateBackend(manifest.robot)
    runtime, _ = runtime_for(manifest, backend)
    result = runtime.run()
    assert not result.success and result.exit_reason == reason
    assert backend.events[-2:] == ["stop", "close"]


def test_open_validation_failure_cleans_before_first_command(manifest_factory):
    manifest = manifest_factory()

    class InvalidRobot(FakeRobotInterface):
        def open(self):
            self.events.append("open")
            raise ValidationError("backend.joint_names", "matching joints", "missing")

    backend = InvalidRobot(manifest.robot)
    runtime, _ = runtime_for(manifest, backend)
    result = runtime.run()
    assert result.exit_reason == "validation_error"
    assert result.completed_steps == 0
    assert backend.events == ["open", "stop", "close"]


@pytest.mark.parametrize(
    "error,reason", [(KeyboardInterrupt(), "interrupted"), (LieDownRequestedError("lie down"), "lie_down")]
)
def test_runtime_interrupts_reach_backend_cleanup(manifest_factory, error, reason):
    manifest = manifest_factory()

    class InterruptedRobot(FakeRobotInterface):
        def read_state(self, timeout_s):
            raise error

    backend = InterruptedRobot(manifest.robot)
    runtime, _ = runtime_for(manifest, backend)
    assert runtime.run().exit_reason == reason
    assert backend.events[-2:] == ["stop", "close"]


def test_host_running_check_failure_stops_before_command_and_cleans_up(manifest_factory, monkeypatch):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    runtime, _ = runtime_for(manifest, backend)

    def reject_interval():
        raise RuntimeError("host stopped")

    monkeypatch.setattr(runtime, "_check_running", reject_interval)
    result = runtime.run()
    assert result.exit_reason == "backend_error"
    assert result.error == "host stopped"
    assert not backend.commands
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize("phase", ["reset", "wait"])
def test_scheduler_failures_preserve_commands_and_cleanup(manifest_factory, phase):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    scheduler = FixedStepScheduler(manifest.control.period_s)

    def fail(*args):
        raise RuntimeError("scheduler failed")

    setattr(scheduler, phase, fail)
    runtime, _ = runtime_for(manifest, backend, scheduler=scheduler)
    result = runtime.run()
    assert result.exit_reason == "backend_error"
    assert result.error == "scheduler failed"
    assert len(backend.commands) == (1 if phase == "wait" else 0)
    assert backend.events[-2:] == ["stop", "close"]


@pytest.mark.parametrize("prior_failure", [False, True])
def test_cleanup_error_preserves_prior_failure_and_attempts_close(manifest_factory, prior_failure):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot, fail_write_at=0 if prior_failure else None)

    def fail_stop():
        backend.events.append("stop-failed")
        raise RuntimeError("cleanup failed")

    backend.stop = fail_stop
    runtime, _ = runtime_for(manifest, backend, steps=1)
    result = runtime.run()
    assert result.exit_reason == ("backend_write_error" if prior_failure else "cleanup_error")
    assert backend.events[-2:] == ["stop-failed", "close"]


def test_finite_controller_validates_final_measured_state(manifest_factory):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    task = _TestPolicyProcessor(manifest)
    task.check_termination = lambda state: (
        "final state unsafe" if state.sample_time_ns >= 2 * backend._period_ns else None
    )
    runtime, _ = runtime_for(manifest, backend, task=task, steps=2)
    result = runtime.run()
    assert not result.success and result.exit_reason == "task_terminated"
    assert result.completed_steps == 2
    assert len(backend.commands) == 2
    assert backend.events[-2:] == ["stop", "close"]


def test_controller_receives_post_enable_measurements_and_resets_history(manifest_factory):
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot)
    task = _StatefulPolicyProcessor(manifest)
    model = _StatefulPolicy()
    runtime, _ = runtime_for(manifest, backend, task=task, model=model)
    result = runtime.run()
    assert result.success
    assert backend.events.index("enable") < backend.events.index("write")
    assert model.reset_count >= 1
    assert [context.step for context in task.contexts] == [0, 1, 2]
    assert [context.elapsed_time_s for context in task.contexts] == pytest.approx([0, 0.02, 0.04])


def test_realtime_scheduler_uses_absolute_deadlines():
    now = [1_000_000_000]
    sleeps = []

    def sleep(duration):
        sleeps.append(duration)
        now[0] += round(duration * 1e9)

    scheduler = RealtimeScheduler(0.02, clock_ns=lambda: now[0], sleep=sleep)
    scheduler.reset()
    scheduler.wait(1)
    now[0] += 5_000_000
    scheduler.wait(2)
    assert sleeps == pytest.approx([0.02, 0.015])
    assert scheduler.elapsed_time_s(2) == pytest.approx(0.04)
    now[0] += 25_000_000
    scheduler.wait(3)
    assert scheduler.overrun_count == 1
    scheduler.reset()
    assert scheduler.overrun_count == 0


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("enable", "backend_enable_error"),
        ("termination", "task_termination_error"),
        ("observation", "invalid_observation"),
        ("inference", "policy_error"),
        ("action", "invalid_action"),
    ],
)
def test_policy_and_enable_failures_keep_structured_error(manifest_factory, fault, reason):
    manifest = manifest_factory()
    backend = EnablingBackend(manifest.robot)
    task = _TestPolicyProcessor(manifest)
    model = ConstantPolicy(4, np.zeros(2, np.float32))

    def fail(*args):
        raise RuntimeError("injected boundary failure")

    target, method = {
        "enable": (backend, "enable"),
        "termination": (task, "check_termination"),
        "observation": (task, "build_observation"),
        "inference": (model, "infer"),
        "action": (task, "process_action"),
    }[fault]
    setattr(target, method, fail)
    runtime, _ = runtime_for(manifest, backend, task=task, model=model)
    result = runtime.run()
    assert not result.success and result.exit_reason == reason
    assert result.error == "injected boundary failure"
    assert result.completed_steps == 0
    assert backend.commands == []
    assert backend.events[-2:] == ["stop", "close"]


class DeterministicController:
    """A deterministic controller that does not contain a model or task adapter."""

    def reset(self, state):
        self.position = state.joint_position.copy()
        self.calls = 0
        self.stop_requested = False

    def request_stop(self):
        self.stop_requested = True

    def step(self, state, context):
        self.command = context.command
        if self.stop_requested or self.calls == 2:
            return ControllerStep(complete=True)
        self.calls += 1
        return ControllerStep(command=JointTorqueCommand(np.zeros_like(state.joint_position)))


def test_missing_required_command_fails_before_inference_and_write(manifest_factory):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    model = ConstantPolicy(4, np.zeros(2, np.float32))
    session = ControlSession(
        robot=backend,
        controller=PolicyController(_TestPolicyProcessor(manifest), model, steps=1),
        period_s=0.02,
        state_timeout_s=0.1,
    )
    runtime = HardwareRuntime(backend, realtime=False)
    runtime.bind_session(session)
    result = runtime.run()
    assert result.exit_reason == "input_error"
    assert not model.observations
    assert not backend.commands
    assert backend.events[-2:] == ["stop", "close"]


def test_control_executes_model_free_controller_and_reports_generic_completion(manifest_factory):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    controller = DeterministicController()
    session = ControlSession(
        controller=controller,
        robot=backend,
        period_s=0.02,
        state_timeout_s=0.1,
    )
    runtime = HardwareRuntime(backend, realtime=False)
    runtime.bind_session(session)
    result = runtime.run()
    assert result.success
    assert result.completed_steps == 0  # Deterministic commands are not model ticks.
    assert len(backend.commands) == 2
    assert controller.calls == 2
    assert controller.command is None
    assert backend.events[-2:] == ["stop", "close"]


def test_control_requested_stop_is_controller_owned_and_cleanup_is_idempotent(manifest_factory):
    manifest = manifest_factory()
    backend = FakeRobotInterface(manifest.robot)
    controller = DeterministicController()
    session = ControlSession(
        controller=controller,
        robot=backend,
        period_s=0.02,
        state_timeout_s=0.1,
    )
    assert session.start()
    assert session.tick(elapsed_time_s=0.0)
    session.request_stop()
    assert not session.tick(elapsed_time_s=0.02)
    session.stop()
    events = list(backend.events)
    session.stop()
    assert backend.events == events
    assert session.result().success
    assert len(backend.commands) == 1
