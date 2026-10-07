# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests for the standalone MuJoCo deployment example."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from control_helpers import DummyServoTask

ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples"
EXAMPLE_SCRIPT = EXAMPLES / "deploy_to_sim.py"


@pytest.fixture
def onnx_policy(tmp_path, control_scene_config, control_gains):
    """Generate raw ONNX locally; never depend on a trained checkout policy."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from motrix_deploy.task import create_task
    from motrix_deploy_mujoco.runtime import MujocoRuntime
    from motrix_deploy_tasks.tasks.go2_walk import Go2WalkTaskSpec

    runtime = MujocoRuntime(control_scene_config, control_period_s=0.02)
    robot = runtime.robot.spec
    kp, kd = control_gains
    task_spec = Go2WalkTaskSpec(
        action_scale=[0.1] * robot.joint_count,
        command_lower=[-1.0] * 3,
        command_upper=[1.0] * 3,
        command_scale=[1.0] * 3,
        feet_phase_offsets=[0.0, 0.5, 0.5, 0.0],
        gait_frequency_hz=1.0,
        standing_threshold=0.05,
        termination_min_up_z=-1.0,
        termination_min_base_height=None,
        kp=kp.tolist(),
        kd=kd.tolist(),
        action_lower=[-1.0] * robot.joint_count,
        action_upper=[1.0] * robot.joint_count,
    )
    task = create_task(task_spec, robot)
    from motrix_deploy.runtime.context import PolicyContext
    from motrix_env_core.input import ConstantPlanarVelocityBinding

    with runtime:
        runtime.robot.open()
        context = PolicyContext(
            step=0,
            elapsed_time_s=0.0,
            command=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)).read_command(),
        )
        state = runtime.robot.read_state(0.1)
        task.reset(state, context)
        observation_size = task.build_observation(state, context).size
        runtime.robot.close()
    weight = numpy_helper.from_array(np.zeros((observation_size, robot.joint_count), dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "weight"], ["actions"])],
        "deterministic_example_fixture",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, observation_size])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, robot.joint_count])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    path = tmp_path / "fixture.onnx"
    onnx.save(model, path)
    return path


def _result(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_example_builders_are_import_safe() -> None:
    namespace = runpy.run_path(str(EXAMPLE_SCRIPT), run_name="motrix_deploy_mujoco_example_test")
    assert callable(namespace["main"])
    assert callable(namespace["ExampleSceneObjsCfg"])


@pytest.mark.parametrize("backend", ["mujoco"])
def test_programmatic_example_uses_shared_control_execution(tmp_path, onnx_policy, backend) -> None:
    command = [
        sys.executable,
        "-c",
        "import runpy, sys; from pathlib import Path; "
        "namespace = runpy.run_path(sys.argv.pop(1)); "
        "namespace['main'].__globals__['POLICY_PATH'] = Path(sys.argv.pop(1)); "
        "raise SystemExit(namespace['main']())",
        str(EXAMPLE_SCRIPT),
        str(onnx_policy),
        "--backend",
        backend,
        "--headless",
        "--steps",
        "3",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    result = _result(completed.stdout)
    assert result["success"] is True
    assert result["completed_steps"] == 3
    assert result["exit_reason"] == "completed"
    assert result["error"] is None
    assert result["simulation_time_s"] == pytest.approx(0.06)
    assert "latency" in result  # Shared control execution report, not a separate preview loop.
    repeated = subprocess.run(command, check=False, capture_output=True, text=True, cwd=tmp_path)
    assert repeated.returncode == 0, repeated.stderr
    assert _result(repeated.stdout)["trace_sha256"] == result["trace_sha256"]


def test_simulation_factory_supports_explicit_bound_control_and_world_ownership(
    control_scene_config, control_gains
) -> None:
    from motrix_deploy.policy import NoOpPolicyRuntime
    from motrix_deploy.robot.interface import RobotInterface
    from motrix_deploy.runtime.config import SimulationRuntimeConfig
    from motrix_deploy.runtime.control import ControlSession
    from motrix_deploy.runtime.factory import create_simulation_runtime
    from motrix_env_core.input import ConstantPlanarVelocityBinding

    config = SimulationRuntimeConfig(
        scene=control_scene_config.scene,
        physics=control_scene_config.physics,
        sensor_bindings=control_scene_config.sensor_bindings,
    )
    runtime = create_simulation_runtime("mujoco", config)
    robot = runtime.robot
    assert isinstance(robot, RobotInterface)
    assert not runtime.opened
    assert runtime.model is None and runtime.data is None
    control = ControlSession(
        robot=robot,
        task=DummyServoTask(robot.spec, *control_gains),
        policy=NoOpPolicyRuntime(robot.spec.joint_count),
        command_binding=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)),
        period_s=0.012,
        state_timeout_s=0.1,
    )
    runtime.bind_control_session(control)
    assert runtime.control_period_s == control.period_s
    assert runtime.physics_substeps == 6
    assert runtime.control is control and control.robot is robot
    assert not runtime.opened
    assert runtime.model is None and runtime.data is None

    with runtime:
        result = runtime.run(steps=2)
        assert result.success and result.completed_steps == 2
        assert runtime.robot is robot and runtime.control is control
        assert not control.active
        # Control shutdown releases robot I/O, not the simulation world's resources.
        assert runtime.opened and runtime.model is not None and runtime.data is not None
        before = runtime.data.time
        runtime.advance_control_period()
        assert runtime.data.time > before
    assert not runtime.opened
    assert runtime.model is None and runtime.data is None


def test_robot_commands_do_not_advance_or_own_the_world(control_scene_config, control_gains) -> None:
    from motrix_deploy.runtime.context import PolicyContext
    from motrix_deploy_mujoco.runtime import MujocoRuntime
    from motrix_env_core.input import ConstantPlanarVelocityBinding

    config = control_scene_config
    simulation = MujocoRuntime(config, control_period_s=0.02)
    spec = simulation.robot.spec
    task = DummyServoTask(spec, *control_gains)
    interface = simulation.robot
    assert interface.spec is spec
    with simulation:
        interface.open()
        before = interface.read_state(0.1)
        context = PolicyContext(
            step=0,
            elapsed_time_s=0.0,
            command=ConstantPlanarVelocityBinding((0.0, 0.0, 0.0)).read_command(),
        )
        observation = task.build_observation(before, context)
        np.testing.assert_array_equal(observation, np.zeros(spec.joint_count, dtype=np.float32))
        assert observation.dtype == np.float32
        interface.write_command(task.process_action(np.zeros(spec.joint_count, dtype=np.float32)))
        after_write = interface.read_state(0.1)
        assert after_write.sample_time_ns == before.sample_time_ns
        np.testing.assert_array_equal(after_write.joint_position, before.joint_position)
        simulation.advance_control_period()
        advanced = interface.read_state(0.1)
        assert advanced.sample_time_ns > before.sample_time_ns
        interface.stop()
        interface.close()
        # Closing the robot port releases control, not the shared simulation.
        assert simulation.model is not None and simulation.data is not None
        time_before = simulation.data.time
        simulation.advance_control_period()
        assert simulation.data.time > time_before
        np.testing.assert_array_equal(simulation.data.ctrl, np.zeros(simulation.model.nu))
    assert simulation.model is None and simulation.data is None
