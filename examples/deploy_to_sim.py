# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Programmatically assemble a Go2 scene, walking task, ONNX policy and control session."""

import argparse
import json
from pathlib import Path

from motrix_robots.unitree import UnitreeGo2Robot

from motrix_deploy.contracts import TensorSpec
from motrix_deploy.policy import OnnxPolicyRuntime
from motrix_deploy.robot.interface import KeyboardDeviceProvider
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkDeployTaskV1, Go2WalkTaskSpec
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg, SceneVisualCfg, SystemCameraCfg
from motrix_env_core.config.scene.light import LightCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.input import ConstantPlanarVelocityBinding, KeyboardPlanarVelocityBinding

POLICY_PATH = Path(__file__).resolve().parent / "assets/go2-walk-flat.onnx"


@configclass
class ExampleSceneObjsCfg(SceneObjsCfg):
    """Objects used by this example's flat scene."""

    floor: FlatTerrainCfg | None = None
    sun: LightCfg = LightCfg(color=(0.7, 0.7, 0.7), illuminance=10_000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco",), default="mujoco", help="Simulation runtime plugin")
    parser.add_argument("--headless", action="store_true", help="No window; run fixed-step without wall-clock pacing")
    parser.add_argument("--steps", type=int, help="Headless control ticks; GUI mode runs until the window closes")
    args = parser.parse_args()
    if args.headless and (args.steps is None or args.steps <= 0):
        parser.error("--steps must be a positive integer in --headless mode")
    if not args.headless and args.steps is not None:
        parser.error("--steps is only valid in --headless mode; GUI mode runs until the window closes")

    policy = OnnxPolicyRuntime(
        POLICY_PATH,
        input_spec=TensorSpec(name="obs", shape=(1, 49)),
        output_spec=TensorSpec(name="actions", shape=(1, 12)),
    )

    scene = SceneCfg(
        objs=ExampleSceneObjsCfg(
            # Placement is an attach transform; the model root is already at z=0.445.
            robot=UnitreeGo2Robot(translation=(0.0, 0.0, -0.114), rotation=(0.0, 0.0, 0.0, 1.0)),
            floor=FlatTerrainCfg(friction=(0.6, 0.005, 0.0001)),
        ),
        # Match StandardSceneCfg lighting without importing the training environment package.
        visual=SceneVisualCfg(
            ambient_light_color=(0.3, 0.3, 0.3),
            ambient_light_brightness=1_000.0,
            head_light_color=(0.6, 0.6, 0.6),
            head_light_luminous_power=1_000.0,
            haze=(0.1, 0.1, 0.1, 1.0),
            tone_mapping="none",
        ),
        system_camera=SystemCameraCfg(distance=3.0, elevation=-20.0, azimuth=90.0),
    )
    config = SimulationRuntimeConfig(
        scene=scene,
        physics=SimCfg(dt=0.002, solver_iterations=100),
        render=not args.headless,
        sensor_bindings=SensorBindings(
            base_angular_velocity="gyro",
            base_linear_acceleration="accelerometer",
            base_linear_velocity="global_linvel",
        ),
    )

    runtime = create_simulation_runtime(args.backend, config)
    robot = runtime.robot
    # Explicit preprocessing/action semantics for this walking policy. These must
    # match its training configuration; a raw ONNX file does not describe them.
    action_scale = 0.25
    task = Go2WalkDeployTaskV1(
        Go2WalkTaskSpec(
            action_scale=[action_scale] * robot.spec.joint_count,
            kp=[35.0] * robot.spec.joint_count,
            kd=[0.5] * robot.spec.joint_count,
            action_lower=((robot.spec.position_lower - robot.spec.default_joint_position) / action_scale).tolist(),
            action_upper=((robot.spec.position_upper - robot.spec.default_joint_position) / action_scale).tolist(),
            command_lower=[-1.0, -0.5, -0.5],
            command_upper=[1.0, 0.5, 0.5],
            command_scale=[1.0, 1.0, 1.0],
            feet_phase_offsets=[0.0, 0.5, 0.5, 0.0],
            gait_frequency_hz=2.0,
            standing_threshold=0.05,
            termination_min_up_z=0.5,
            termination_min_base_height=None,
        ),
        robot.spec,
    )
    if not args.headless and isinstance(runtime, KeyboardDeviceProvider):
        command_binding = KeyboardPlanarVelocityBinding(
            runtime.get_keyboard_device(),
            command_lower=task.command_lower,
            command_upper=task.command_upper,
        )
    else:
        command_binding = ConstantPlanarVelocityBinding((0.0, 0.0, 0.0))
        task.validate_command(command_binding.read_command())
    control = ControlSession(
        robot=robot,
        task=task,
        policy=policy,
        command_binding=command_binding,
        period_s=0.02,
        state_timeout_s=0.1,
    )
    runtime.bind_control_session(control)
    with runtime:
        result = runtime.run(steps=args.steps)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
