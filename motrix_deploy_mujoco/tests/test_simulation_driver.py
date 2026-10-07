# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Physics-owned control cadence, pacing, and shared lifecycle integration."""

import numpy as np
import pytest
from control_helpers import DummyServoTask

from motrix_deploy.errors import ValidationError
from motrix_deploy.policy import NoOpPolicyRuntime
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy_mujoco.runtime import MujocoRuntime
from motrix_env_core.input import ConstantPlanarVelocityBinding


@pytest.fixture
def driver(control_scene_config, control_gains):
    config = control_scene_config
    simulation = MujocoRuntime(config, control_period_s=0.02)
    task = DummyServoTask(simulation.robot.spec, *control_gains)
    interface = simulation.robot
    control = ControlSession(
        robot=interface,
        task=task,
        policy=NoOpPolicyRuntime(interface.spec.joint_count),
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=simulation.control_period_s,
        state_timeout_s=0.1,
    )
    with simulation:
        yield simulation, control, interface, task


def test_physics_substeps_and_control_sample_times(driver, monkeypatch):
    simulation, control, interface, task = driver
    events = []
    samples = []
    contexts = []
    apply = simulation.robot.apply_control
    step = simulation.mj.mj_step
    observe = task.build_observation

    def record_apply():
        events.append("pd")
        apply()

    def record_step(model, data):
        events.append("physics")
        step(model, data)

    def record_observation(state, context):
        events.append("tick")
        samples.append(state.sample_time_ns)
        contexts.append(context.elapsed_time_s)
        return observe(state, context)

    monkeypatch.setattr(simulation.robot, "apply_control", record_apply)
    monkeypatch.setattr(simulation.mj, "mj_step", record_step)
    monkeypatch.setattr(task, "build_observation", record_observation)
    # The driver must not delegate a full physics period to a control callback.
    monkeypatch.setattr(simulation, "advance_control_period", lambda: pytest.fail("legacy advance called"))
    result = simulation.run(control, steps=3)
    assert result.success and result.completed_steps == 3
    assert samples == [0, 20_000_000, 40_000_000]
    assert contexts == pytest.approx([0.0, 0.02, 0.04])
    assert events == (["tick"] + ["pd", "physics"] * simulation.physics_substeps) * 3
    assert result.simulation_time_s == pytest.approx(0.06)
    assert simulation.data.time == pytest.approx(0.06)
    assert simulation.model is not None and not interface.health().healthy
    np.testing.assert_array_equal(simulation.data.ctrl, np.zeros(simulation.model.nu))


def test_torque_control_session_preserves_control_and_physics_cadence(driver, monkeypatch):
    from motrix_deploy.contracts import JointTorqueCommand

    simulation, _, interface, task = driver
    samples = []
    torques = []
    observe = task.build_observation
    step = simulation.mj.mj_step
    desired = np.linspace(-1, 1, interface.spec.joint_count, dtype=np.float32)

    def record_observation(state, context):
        samples.append(state.sample_time_ns)
        return observe(state, context)

    def record_step(model, data):
        torques.append(data.ctrl[simulation.robot._actuator_indices].copy())
        step(model, data)

    monkeypatch.setattr(task, "build_observation", record_observation)
    monkeypatch.setattr(task, "process_action", lambda action: JointTorqueCommand(desired))
    monkeypatch.setattr(simulation.mj, "mj_step", record_step)
    control = ControlSession(
        robot=interface,
        task=task,
        policy=NoOpPolicyRuntime(interface.spec.joint_count),
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=simulation.control_period_s,
        state_timeout_s=0.1,
    )
    result = simulation.run(control, steps=3)
    assert result.success and result.completed_steps == 3
    assert samples == [0, 20_000_000, 40_000_000]
    assert len(torques) == 3 * simulation.physics_substeps
    for torque in torques:
        np.testing.assert_array_equal(torque, desired)
    assert result.simulation_time_s == pytest.approx(0.06)
    np.testing.assert_array_equal(simulation.data.ctrl, 0)


def test_shared_tick_never_advances_physics(driver):
    simulation, control, _, _ = driver
    try:
        assert control.start()
        before = simulation.data.time
        assert control.tick(elapsed_time_s=0.0)
        assert simulation.data.time == before
    finally:
        control.stop()


def test_headless_driver_is_deterministic_and_never_paces(monkeypatch, control_scene_config, control_gains):
    from motrix_deploy_mujoco import runtime

    monkeypatch.setattr(runtime, "RealtimeScheduler", lambda *args: pytest.fail("headless pacing"))
    config = control_scene_config
    traces = []
    for _ in range(2):
        simulation = MujocoRuntime(config, control_period_s=0.02)
        task = DummyServoTask(simulation.robot.spec, *control_gains)
        interface = simulation.robot
        with simulation:
            control = ControlSession(
                robot=interface,
                task=task,
                policy=NoOpPolicyRuntime(interface.spec.joint_count),
                command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
                period_s=simulation.control_period_s,
                state_timeout_s=0.1,
            )
            result = simulation.run(control, steps=4)
            assert result.success and result.overrun_count == 0
            traces.append(result.trace_sha256)
    assert traces[0] == traces[1]


def test_realtime_paces_every_complete_interval_without_extra_tick(driver, monkeypatch):
    from motrix_deploy_mujoco import runtime

    simulation, control, _, _ = driver
    waits = []

    class Scheduler:
        overrun_count = 2

        def __init__(self, period_s):
            assert period_s == simulation.control_period_s

        def reset(self):
            waits.append("reset")

        def wait(self, step):
            waits.append(step)

    monkeypatch.setattr(runtime, "RealtimeScheduler", Scheduler)
    result = simulation.run(control, steps=3, realtime=True)
    assert result.success and result.completed_steps == 3
    assert waits == ["reset", 1, 2, 3]
    assert result.overrun_count == 2


def test_physics_failure_cleans_robot_but_keeps_world_and_reports_actual_time(driver, monkeypatch):
    simulation, control, interface, _ = driver
    step = simulation.mj.mj_step
    calls = 0

    def fail_mid_interval(model, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("physics failed")
        step(model, data)

    monkeypatch.setattr(simulation.mj, "mj_step", fail_mid_interval)
    result = simulation.run(control, steps=3)
    assert not result.success and result.exit_reason == "backend_error"
    assert result.error == "physics failed"
    assert result.completed_steps == 1
    assert result.simulation_time_s == pytest.approx(2 * simulation.config.physics.dt)
    assert result.real_time_factor == pytest.approx(result.simulation_time_s / result.wall_time_s)
    assert simulation.model is not None and simulation.data is not None
    assert not interface.health().healthy and not control.active
    np.testing.assert_array_equal(simulation.data.ctrl, np.zeros(simulation.model.nu))
    control.stop()  # Shared cleanup remains idempotent after driver failure.


def test_start_exception_still_cleans_shared_control(driver, monkeypatch):
    simulation, control, interface, _ = driver

    def failed_start():
        raise RuntimeError("start failed")

    monkeypatch.setattr(control, "start", failed_start)
    result = simulation.run(control, steps=3)
    assert result.exit_reason == "backend_error" and result.error == "start failed"
    assert result.completed_steps == 0 and result.simulation_time_s == 0.0
    assert simulation.model is not None and not interface.health().healthy


def test_elapsed_control_time_is_relative_to_this_run(driver, monkeypatch):
    simulation, control, _, task = driver
    simulation.advance_control_period()
    before = simulation.data.time
    contexts = []
    observe = task.build_observation

    def record_context(state, context):
        contexts.append(context.elapsed_time_s)
        return observe(state, context)

    monkeypatch.setattr(task, "build_observation", record_context)
    result = simulation.run(control, steps=2)
    assert result.success
    assert contexts == pytest.approx([0.0, simulation.control_period_s])
    assert result.simulation_time_s == pytest.approx(2 * simulation.control_period_s)
    assert simulation.data.time - before == pytest.approx(result.simulation_time_s)


def test_start_termination_does_not_advance_physics(driver, monkeypatch):
    simulation, control, interface, task = driver
    monkeypatch.setattr(task, "check_termination", lambda state: "done")
    result = simulation.run(control, steps=3)
    assert not result.success and result.exit_reason == "task_terminated"
    assert result.completed_steps == 0 and result.simulation_time_s == 0.0
    assert simulation.data.time == 0.0 and not interface.health().healthy


def test_tick_termination_has_no_extra_physics_interval(driver, monkeypatch):
    simulation, control, _, task = driver
    monkeypatch.setattr(task, "check_termination", lambda state: "done" if state.sample_time_ns > 0 else None)
    result = simulation.run(control, steps=3)
    assert result.exit_reason == "task_terminated" and result.completed_steps == 1
    assert result.simulation_time_s == pytest.approx(simulation.control_period_s)


def test_viewer_interrupt_uses_shared_cleanup(driver, monkeypatch):
    simulation, control, interface, _ = driver

    def interrupt():
        raise KeyboardInterrupt("viewer closed")

    monkeypatch.setattr(simulation, "check_viewer_running", interrupt)
    result = simulation.run(control, steps=3)
    assert result.exit_reason == "interrupted" and result.completed_steps == 0
    assert result.simulation_time_s == 0.0 and not interface.health().healthy
    assert simulation.model is not None


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_run_rejects_invalid_tick_count(driver, steps):
    simulation, control, _, _ = driver
    with pytest.raises(ValidationError, match="steps"):
        simulation.run(control, steps=steps)


def test_run_uses_supplied_control_session_period(driver):
    simulation, _, interface, task = driver
    control = ControlSession(
        robot=interface,
        task=task,
        policy=NoOpPolicyRuntime(interface.spec.joint_count),
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=0.01,
        state_timeout_s=0.1,
    )
    result = simulation.run(control, steps=2)
    assert result.success
    assert simulation.control_period_s == control.period_s
    assert result.simulation_time_s == pytest.approx(2 * control.period_s)


def test_binding_rejects_nonintegral_period_without_binding_control(driver):
    from types import SimpleNamespace

    simulation, _, _, _ = driver
    period_s = simulation.control_period_s
    substeps = simulation.physics_substeps
    invalid = SimpleNamespace(period_s=simulation.sim.dt * 1.5, robot=simulation.robot)
    with pytest.raises(ValidationError, match="control.period_s"):
        simulation.bind_control_session(invalid)
    assert simulation.control is None
    assert simulation.control_period_s == period_s and simulation.physics_substeps == substeps


def test_run_requires_integral_control_period(driver):
    simulation, _, _, _ = driver
    # Physics cannot resample a control period that does not span whole substeps.
    from types import SimpleNamespace

    mismatched = SimpleNamespace(period_s=simulation.sim.dt * 1.5, robot=simulation.robot)
    with pytest.raises(ValidationError, match="control.period_s"):
        simulation.run(mismatched, steps=1)
