# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Physics-owned session cadence, pacing, and shared lifecycle integration."""

import numpy as np
import pytest
from control_helpers import DummyServoPolicyProcessor

from motrix_deploy.errors import ValidationError
from motrix_deploy.policy import NoOpPolicy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import ControlSession, PolicyController
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.runtime.result import LatencySummary, SimulationRolloutResult
from motrix_deploy.runtime.scheduler import FixedStepScheduler
from motrix_deploy_mujoco.runtime import MujocoRuntime
from motrix_env_core.input import ConstantPlanarVelocityBinding


def bind_task(simulation, task, *, steps=3, period_s=None):
    period_s = simulation.control_period_s if period_s is None else period_s
    controller = PolicyController(task, NoOpPolicy(simulation.robot.spec.joint_count), steps=steps)
    session = ControlSession(
        controller=controller,
        robot=simulation.robot,
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=period_s,
        state_timeout_s=0.1,
    )
    simulation.bind_session(session)
    return controller


@pytest.fixture
def driver(control_scene_config, control_gains):
    simulation = MujocoRuntime(control_scene_config, control_period_s=0.02)
    task = DummyServoPolicyProcessor(simulation.robot.spec, *control_gains)
    controller = bind_task(simulation, task)
    with simulation:
        yield simulation, controller, simulation.robot, task


def test_physics_substeps_and_control_sample_times(driver, monkeypatch):
    simulation, session, interface, task = driver
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
    result = simulation.run()
    assert result.success and result.completed_steps == 3
    assert samples == [0, 20_000_000, 40_000_000]
    assert contexts == pytest.approx([0.0, 0.02, 0.04])
    assert events == (["tick"] + ["pd", "physics"] * simulation.physics_substeps) * 3
    assert isinstance(result, SimulationRolloutResult)
    assert result.policy_simulation_time_s == pytest.approx(0.06)
    assert isinstance(result.latency["read"], LatencySummary)
    report = result.to_dict()
    assert report["policy_simulation_time_s"] == pytest.approx(0.06)
    assert report["latency"]["read"]["mean_ms"] == result.latency["read"].mean_ms
    assert simulation.data.time == pytest.approx(0.06)
    assert simulation.model is not None and not interface.health().healthy
    np.testing.assert_array_equal(simulation.data.ctrl, np.zeros(simulation.model.nu))


def test_simulation_duration_excludes_nonpolicy_command_intervals(driver, monkeypatch):
    simulation, controller, _, _ = driver
    step = controller.step
    decisions = 0

    def with_ready_interval(state, context):
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            # A deterministic ready command has physical duration but no model tick.
            return ControllerStep(
                command=controller.policy_io.process_action(
                    np.zeros(simulation.robot.spec.joint_count, dtype=np.float32)
                )
            )
        return step(state, context)

    monkeypatch.setattr(controller, "step", with_ready_interval)
    result = simulation.run()
    assert result.success and result.completed_steps == 3
    assert simulation.data.time == pytest.approx(0.08)
    assert result.policy_simulation_time_s == pytest.approx(0.06)


def test_torque_controller_preserves_control_and_physics_cadence(driver, monkeypatch):
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
    result = simulation.run()
    assert result.success and result.completed_steps == 3
    assert samples == [0, 20_000_000, 40_000_000]
    assert len(torques) == 3 * simulation.physics_substeps
    for torque in torques:
        np.testing.assert_array_equal(torque, desired)
    assert result.policy_simulation_time_s == pytest.approx(0.06)
    np.testing.assert_array_equal(simulation.data.ctrl, 0)


def test_controller_step_never_writes_or_advances_physics(driver):
    simulation, controller, robot, _ = driver
    state = robot.read_state(0.1)
    context = ControlContext(0, 0.0, ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)).read_command(), dt_s=0.02)
    before_time = simulation.data.time
    before_control = simulation.data.ctrl.copy()
    controller.reset(state)
    decision = controller.step(state, context)
    assert decision.command is not None and not decision.complete
    assert simulation.data.time == before_time
    np.testing.assert_array_equal(simulation.data.ctrl, before_control)


def test_headless_driver_is_deterministic_and_never_paces(monkeypatch, control_scene_config, control_gains):
    from motrix_deploy_mujoco import runtime

    monkeypatch.setattr(runtime, "RealtimeScheduler", lambda *args: pytest.fail("headless pacing"))
    config = control_scene_config
    traces = []
    for _ in range(2):
        simulation = MujocoRuntime(config, control_period_s=0.02)
        task = DummyServoPolicyProcessor(simulation.robot.spec, *control_gains)
        with simulation:
            bind_task(simulation, task, steps=4)
            result = simulation.run()
            assert result.success
            traces.append(result.trace_sha256)
    assert traces[0] == traces[1]


def test_realtime_paces_every_complete_interval_without_extra_tick(driver, monkeypatch):
    from motrix_deploy_mujoco import runtime

    simulation, session, _, _ = driver
    waits = []

    class Scheduler(FixedStepScheduler):
        def __init__(self, period_s):
            super().__init__(period_s)
            assert period_s == simulation.control_period_s

        def reset(self):
            waits.append("reset")

        def wait(self, step):
            waits.append(step)

    monkeypatch.setattr(runtime, "RealtimeScheduler", Scheduler)
    result = simulation.run(realtime=True)
    assert result.success and result.completed_steps == 3
    assert waits == ["reset", 1, 2, 3]


def test_physics_failure_cleans_robot_but_keeps_world_and_reports_actual_time(driver, monkeypatch):
    simulation, session, interface, _ = driver
    step = simulation.mj.mj_step
    calls = 0

    def fail_mid_interval(model, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("physics failed")
        step(model, data)

    monkeypatch.setattr(simulation.mj, "mj_step", fail_mid_interval)
    result = simulation.run()
    assert not result.success and result.exit_reason == "backend_error"
    assert result.error == "physics failed"
    assert result.completed_steps == 1
    assert result.policy_simulation_time_s == pytest.approx(2 * simulation.config.physics.dt)
    assert result.real_time_factor == pytest.approx(result.policy_simulation_time_s / result.wall_time_s)
    assert simulation.model is not None and simulation.data is not None
    assert not interface.health().healthy
    np.testing.assert_array_equal(simulation.data.ctrl, np.zeros(simulation.model.nu))
    interface.stop()  # Backend cleanup remains idempotent after driver failure.


def test_controller_reset_exception_still_cleans_robot(driver, monkeypatch):
    simulation, session, interface, _ = driver

    def failed_reset(state):
        raise RuntimeError("start failed")

    monkeypatch.setattr(session, "reset", failed_reset)
    result = simulation.run()
    assert result.exit_reason == "backend_error" and result.error == "start failed"
    assert result.completed_steps == 0 and result.policy_simulation_time_s == 0.0
    assert simulation.model is not None and not interface.health().healthy


def test_elapsed_control_time_is_relative_to_this_run(driver, monkeypatch):
    simulation, session, _, task = driver
    simulation.advance_control_period()
    before = simulation.data.time
    contexts = []
    observe = task.build_observation

    def record_context(state, context):
        contexts.append(context.elapsed_time_s)
        return observe(state, context)

    monkeypatch.setattr(task, "build_observation", record_context)
    result = simulation.run()
    assert result.success
    assert contexts == pytest.approx([0.0, simulation.control_period_s, 2 * simulation.control_period_s])
    assert result.policy_simulation_time_s == pytest.approx(3 * simulation.control_period_s)
    assert simulation.data.time - before == pytest.approx(result.policy_simulation_time_s)


def test_start_termination_does_not_advance_physics(driver, monkeypatch):
    simulation, session, interface, task = driver
    monkeypatch.setattr(task, "check_termination", lambda state: "done")
    result = simulation.run()
    assert not result.success and result.exit_reason == "task_terminated"
    assert result.completed_steps == 0 and result.policy_simulation_time_s == 0.0
    assert simulation.data.time == 0.0 and not interface.health().healthy


def test_tick_termination_has_no_extra_physics_interval(driver, monkeypatch):
    simulation, session, _, task = driver
    monkeypatch.setattr(task, "check_termination", lambda state: "done" if state.sample_time_ns > 0 else None)
    result = simulation.run()
    assert result.exit_reason == "task_terminated" and result.completed_steps == 1
    assert result.policy_simulation_time_s == pytest.approx(simulation.control_period_s)


def test_viewer_interrupt_uses_shared_cleanup(driver, monkeypatch):
    simulation, session, interface, _ = driver

    def interrupt():
        raise KeyboardInterrupt("viewer closed")

    monkeypatch.setattr(simulation, "check_viewer_running", interrupt)
    result = simulation.run()
    assert result.exit_reason == "interrupted" and result.completed_steps == 0
    assert result.policy_simulation_time_s == 0.0 and not interface.health().healthy
    assert simulation.model is not None


def test_bound_controller_period_controls_physics_cadence(driver):
    simulation, _, _, task = driver
    # A fresh runtime can bind a different integral session period.
    simulation.close()
    with MujocoRuntime(simulation.config) as runtime:
        bind_task(runtime, task, steps=2, period_s=0.01)
        result = runtime.run()
        assert result.success
        assert runtime.control_period_s == 0.01
        assert result.policy_simulation_time_s == pytest.approx(0.02)


def test_binding_rejects_nonintegral_period_without_changing_cadence(driver):
    simulation, controller, _, _ = driver
    period_s = simulation.control_period_s
    substeps = simulation.physics_substeps
    invalid_period = simulation.sim.dt * 1.5
    fresh = MujocoRuntime(simulation.config, control_period_s=period_s)
    invalid = ControlSession(
        controller=PolicyController(controller.policy_io, controller.policy, steps=3),
        robot=fresh.robot,
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=invalid_period,
        state_timeout_s=0.1,
    )
    # An unbound host rejects non-integral cadence before claiming the session.
    bound = simulation.session
    with pytest.raises(ValidationError, match="session.period_s"):
        fresh.bind_session(invalid)
    assert fresh.session is None
    assert simulation.session is bound
    fresh.close()
    assert simulation.control_period_s == period_s and simulation.physics_substeps == substeps


def test_runtime_binding_rejects_foreign_robot_and_conflicting_control(driver):
    simulation, controller, robot, _ = driver
    bound = simulation.session
    conflict = ControlSession(
        controller=controller,
        robot=robot,
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=bound.period_s,
        state_timeout_s=0.1,
    )
    with pytest.raises(ValidationError, match="bound"):
        simulation.bind_session(conflict)
    foreign_runtime = MujocoRuntime(simulation.config, control_period_s=bound.period_s)
    try:
        with pytest.raises(ValidationError, match="robot"):
            foreign_runtime.bind_session(bound)
        assert foreign_runtime.session is None
        assert simulation.session is bound
    finally:
        foreign_runtime.close()


def test_final_advanced_state_is_checked_before_controller_completion(driver, monkeypatch):
    simulation, _, _, task = driver
    monkeypatch.setattr(
        task, "check_termination", lambda state: "last interval failed" if state.sample_time_ns >= 60_000_000 else None
    )
    result = simulation.run()
    assert not result.success
    assert result.exit_reason == "task_terminated"
    assert result.completed_steps == 3
    assert result.policy_simulation_time_s == pytest.approx(0.06)
