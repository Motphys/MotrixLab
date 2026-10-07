# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hydra deployment rollout and artifact inspection entry points."""

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from motrix_deploy import __version__
from motrix_deploy.artifact.io import inspect_artifact, read_artifact
from motrix_deploy.config import DeployRunConfig
from motrix_deploy.errors import ArtifactError, ValidationError
from motrix_deploy.robot.interface import GamePadDeviceProvider, KeyboardDeviceProvider
from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_deploy.runtime.factory import RuntimeCreateContext, create_hardware_runtime, create_simulation_runtime
from motrix_deploy.task import create_task


def run(cfg: DeployRunConfig) -> int:
    """Run one deployment rollout from typed application configuration."""
    from motrix_deploy.policy import OnnxPolicyRuntime
    from motrix_deploy.runtime.control import ControlSession
    from motrix_env_core.input import (
        BoundedGamePadPlanarVelocityBinding,
        ConstantPlanarVelocityBinding,
        KeyboardPlanarVelocityBinding,
    )

    artifact = read_artifact(Path(cfg.artifact))
    manifest = artifact.manifest
    steps = _duration_steps(cfg.duration_s, manifest.control.period_s)
    task = create_task(manifest.task, manifest.robot)
    simulation = cfg.runtime["kind"] == "simulation"
    viewer = cfg.runtime["viewer"] if simulation else False
    realtime = cfg.runtime["realtime"] if simulation else True
    if realtime is None:
        realtime = viewer
    runtime_options = cfg.runtime_options
    if simulation:
        from motrix_deploy.env import assemble_deploy_scene

        for field in ("deploy_env_id", "robot_id"):
            if not isinstance(runtime_options.get(field), str) or not runtime_options[field]:
                raise ValidationError(f"runtime.{field}", "a non-empty registered ID", runtime_options.get(field))
        deploy_env_id = runtime_options.pop("deploy_env_id")
        robot_id = runtime_options.pop("robot_id")
        scene = assemble_deploy_scene(deploy_env_id, robot_id)
        for field in ("translation", "rotation"):
            option = f"robot_{field}"
            if option in runtime_options:
                value = runtime_options.pop(option)
                setattr(scene.objs.robot, field, None if value is None else tuple(value))
        runtime_options["scene"] = scene
        runtime_config = SimulationRuntimeConfig.from_mapping(runtime_options, render=viewer, realtime=realtime)
        runtime = create_simulation_runtime(cfg.backend_name, runtime_config)
    else:
        runtime = create_hardware_runtime(
            cfg.backend_name,
            runtime_options,
            RuntimeCreateContext(
                robot=manifest.robot,
                control=manifest.control,
            ),
        )
    backend = runtime.robot
    try:
        manifest.robot.validate_compatible(backend.spec)
    except ValidationError:
        runtime.close()
        raise
    command_lower = getattr(task, "command_lower", None)
    command_upper = getattr(task, "command_upper", None)
    if command_lower is None or command_upper is None:
        raise ValueError(f"Task {manifest.task.name!r} does not provide planar velocity command bounds")
    command_config = cfg.command or {}
    command_source = command_config.get("source", "constant")
    if viewer and isinstance(runtime, KeyboardDeviceProvider):
        command_binding = KeyboardPlanarVelocityBinding(
            runtime.get_keyboard_device(),
            command_lower=command_lower,
            command_upper=command_upper,
        )
    elif command_source == "gamepad":
        if not isinstance(backend, GamePadDeviceProvider):
            raise ValueError(f"Backend {cfg.backend_name!r} does not provide gamepad input")
        gamepad = command_config.get("gamepad", {})
        if not isinstance(gamepad, dict):
            raise ValueError("command.gamepad must be a mapping")
        command_binding = BoundedGamePadPlanarVelocityBinding(
            backend.get_gamepad_device(),
            linear_x_axis=gamepad.get("linear_x_axis", "ly"),
            linear_y_axis=gamepad.get("linear_y_axis", "lx"),
            yaw_axis=gamepad.get("yaw_axis", "rx"),
            command_lower=command_lower,
            command_upper=command_upper,
            deadzone=gamepad.get("deadzone", 0.1),
            range_scale=gamepad.get("range_scale", [1.0, 1.0, 1.0]),
            invert_linear_x=gamepad.get("invert_linear_x", False),
            invert_linear_y=gamepad.get("invert_linear_y", False),
            invert_yaw=gamepad.get("invert_yaw", False),
            deadman_button=gamepad.get("deadman_button", "L1"),
        )
    elif command_source == "constant" and "velocity" in command_config:
        command_binding = ConstantPlanarVelocityBinding(command_config["velocity"])
        task.validate_command(command_binding.read_command())
    else:
        raise ValueError(f"Unsupported command source {command_source!r}; use gamepad or configure command.velocity")
    control = ControlSession(
        robot=backend,
        task=task,
        policy=OnnxPolicyRuntime(artifact.policy_path, manifest.policy.input, manifest.policy.output),
        command_binding=command_binding,
        period_s=manifest.control.period_s,
        state_timeout_s=manifest.control.state_timeout_s,
    )
    runtime.bind_control_session(control)
    with runtime:
        result = runtime.run(steps=steps)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def _duration_steps(duration_s: float | None, control_period_s: float) -> int | None:
    """Convert a control-time duration to a tick budget, not a wall-clock timeout."""
    if duration_s is None:
        return None
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise ValueError(f"duration_s must be a positive finite number, got {duration_s!r}")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"duration_s must be a positive finite number, got {duration_s!r}")
    return math.ceil(duration_s / control_period_s)


def _run_config(cfg: DictConfig) -> None:
    try:
        typed_cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DeployRunConfig), cfg))
        if not isinstance(typed_cfg, DeployRunConfig):
            raise TypeError(f"Expected DeployRunConfig, got {type(typed_cfg).__name__}")
        exit_code = run(typed_cfg)
    except (ArtifactError, OmegaConfBaseException, RuntimeError, TypeError, ValidationError, ValueError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from error
    if exit_code:
        raise SystemExit(exit_code)


_hydra_main = hydra.main(version_base=None, config_path="pkg://motrix_deploy.config", config_name="deploy")(_run_config)


def inspect(cfg: DictConfig) -> None:
    """Validate one deployment artifact without opening a backend."""
    print(json.dumps(inspect_artifact(Path(cfg.artifact)), indent=2, sort_keys=True))


@hydra.main(version_base=None, config_path="pkg://motrix_deploy.config", config_name="inspect")
def _hydra_inspect_main(cfg: DictConfig) -> None:
    try:
        inspect(cfg)
    except (ArtifactError, OmegaConfBaseException, RuntimeError, ValidationError, ValueError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from error


def main(argv: Sequence[str] | None = None) -> int | None:
    """Inspect an artifact or run a Hydra-configured deployment rollout."""
    if argv is not None:
        original_argv = sys.argv
        sys.argv = ["motrix-deploy", *argv]
        try:
            return main()
        finally:
            sys.argv = original_argv

    args = sys.argv[1:]
    if args[:1] == ["inspect"]:
        del sys.argv[1]
        _hydra_inspect_main()
        return None
    if args == ["--version"]:
        print(f"motrix-deploy {__version__}")
        return 0
    if not args:
        sys.argv.append("--help")
    _hydra_main()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
