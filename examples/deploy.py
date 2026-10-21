# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Programmatically assemble a Go2 ONNX walking controller for simulation or hardware.

No artifact or deployment CLI is used. Robot/task settings must match the ONNX model.
Real execution moves all legs: use load-bearing support and an independent stop.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_deploy.contracts import RobotSpec, TensorSpec
from motrix_deploy.policy import OnnxPolicy
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkDeployTask, Go2WalkTaskSpec
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg, SceneVisualCfg, SystemCameraCfg
from motrix_env_core.config.scene.light import LightCfg
from motrix_env_core.config.sim import SimCfg

POLICY_PATH = Path(__file__).resolve().parent / "assets/go2-walk-flat.onnx"


@configclass
class ExampleSceneObjsCfg(SceneObjsCfg):
    """Objects used by this example's flat scene."""

    floor: FlatTerrainCfg | None = None
    sun: LightCfg = LightCfg(color=(0.7, 0.7, 0.7), illuminance=10_000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Run on hardware instead of simulation")
    parser.add_argument("--backend", choices=("mujoco", "motrixsim"), help="Simulation plugin (default: mujoco)")
    parser.add_argument("--network-interface", help="Required wired interface for real execution")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH, help="ONNX matching the declared task settings")
    parser.add_argument("--headless", action="store_true", help="No window; run fixed-step without wall-clock pacing")
    parser.add_argument("--steps", type=int, help="Policy-step budget for headless simulation or real execution")
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be a positive integer")
    if args.real:
        if not args.network_interface:
            parser.error("--network-interface is required for --real")
        if args.backend is not None or args.headless:
            parser.error("--backend and --headless are only valid for simulation")
    else:
        if args.network_interface is not None:
            parser.error("--network-interface is only valid for --real")
        if args.headless and args.steps is None:
            parser.error("--steps is required in --headless mode")
        if not args.headless and args.steps is not None:
            parser.error("--steps is only valid in headless simulation; GUI mode runs until the window closes")

    model = OnnxPolicy(
        args.policy,
        input_spec=TensorSpec(name="obs", shape=(1, 49)),
        output_spec=TensorSpec(name="actions", shape=(1, 12)),
    )

    robot_cfg = UnitreeGo2Robot()
    # Explicit policy limits match the bundled model's position actuators; these
    # are independent of the hardware backend's mechanical command limits.
    spec = RobotSpec(
        base_link_name=robot_cfg.base_link_name,
        joint_names=tuple(robot_cfg.key_pose.joint_names),
        default_joint_position=np.asarray(robot_cfg.key_pose.poses["default"], dtype=np.float32),
        position_lower=np.asarray([-0.9472, -1.4, -2.6227] * 4, dtype=np.float32),
        position_upper=np.asarray([0.9472, 2.5, -0.84776] * 4, dtype=np.float32),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )
    period_s, state_timeout_s = 0.02, 0.1
    if not args.real:
        # Placement is an attach transform; the model root is already at z=0.445.
        robot_cfg.translation = (0.0, 0.0, -0.114)
        robot_cfg.rotation = (0.0, 0.0, 0.0, 1.0)
        scene = SceneCfg(
            objs=ExampleSceneObjsCfg(
                robot=robot_cfg,
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
            system_camera=SystemCameraCfg(lookat=(0.0, 0.0, 0.3), distance=3.0, elevation=-20.0, azimuth=90.0),
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

        runtime = create_simulation_runtime(args.backend or "mujoco", config)
        spec.validate_compatible(runtime.robot.spec)
        device = runtime.keyboard
    else:
        # Lazy imports keep simulation independent of the optional hardware SDK.
        from motrix_deploy.runtime.hardware import HardwareRuntime
        from motrix_deploy_unitree.config.control import load_control_config
        from motrix_deploy_unitree.config.runtime import UnitreeTransportConfig
        from motrix_deploy_unitree.go2.robot import UnitreeGo2Robot as Go2HardwareRobot

        robot = Go2HardwareRobot(
            UnitreeTransportConfig(network_interface=args.network_interface),
            load_control_config("go2"),
            spec=spec,
            control_period_s=period_s,
            state_timeout_s=state_timeout_s,
        )
        runtime = HardwareRuntime(robot)
        device = runtime.gamepad
    robot = runtime.robot
    # Explicit preprocessing/action semantics for this walking controller. These must
    # match its training configuration; a raw ONNX file does not describe them.
    action_scale = 0.25
    task = Go2WalkDeployTask(
        Go2WalkTaskSpec(
            action_scale=[action_scale] * spec.joint_count,
            kp=[35.0] * spec.joint_count,
            kd=[0.5] * spec.joint_count,
            action_lower=((spec.position_lower - spec.default_joint_position) / action_scale).tolist(),
            action_upper=((spec.position_upper - spec.default_joint_position) / action_scale).tolist(),
            command_lower=[-1.0, -0.5, -0.5],
            command_upper=[1.0, 0.5, 0.5],
            command_scale=[1.0, 1.0, 1.0],
            feet_phase_offsets=[0.0, 0.5, 0.5, 0.0],
            gait_frequency_hz=2.0,
            standing_threshold=0.05,
            termination_min_up_z=0.5,
        ),
        spec,
        model,
        steps=args.steps,
    )
    command_binding = task.prepare(simulation=not args.real, device=device)
    session = ControlSession(
        controller=task,
        robot=robot,
        command_binding=command_binding,
        period_s=period_s,
        state_timeout_s=state_timeout_s,
    )
    runtime.bind_session(session)
    with runtime:
        result = runtime.run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
