# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hydra deployment rollout and artifact inspection entry points."""

import json
import math
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from motrix_deploy.artifact.io import inspect_artifact, read_artifact
from motrix_deploy.config import DeployRunConfig, ProgramRunConfig, SimRuntimeCfg
from motrix_deploy.errors import ArtifactError, ValidationError
from motrix_deploy.operator import OperatorCommandBinding
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import RuntimeCreateContext, create_hardware_runtime, create_simulation_runtime
from motrix_deploy.task import create_task, load_task_type


def run(cfg: DeployRunConfig) -> int:
    """Run one deployment rollout from typed application configuration."""
    from motrix_deploy.policy import OnnxPolicy

    artifact = read_artifact(Path(cfg.artifact))
    manifest = artifact.manifest
    task_type = load_task_type(manifest.task.name)
    default_duration_s = task_type.default_duration_s(manifest.task, manifest.control.period_s)
    duration_s = cfg.duration_s if cfg.duration_s is not None else default_duration_s
    steps = _duration_steps(duration_s, manifest.control.period_s)
    if default_duration_s is not None and duration_s > default_duration_s + 1e-9:
        raise ValueError("duration_s exceeds the task's finite playback budget")
    policy = OnnxPolicy(artifact.policy_path, manifest.policy.input, manifest.policy.output)
    task = create_task(manifest.task, manifest.robot, policy, steps=steps, artifact=artifact)
    simulation = isinstance(cfg.runtime, SimRuntimeCfg)
    spawn_pose = cfg.runtime.robot_init_key_pose if simulation else None
    if simulation:
        runtime = create_simulation_runtime(cfg.runtime.backend, cfg.runtime.build_runtime_config())
    else:
        runtime = create_hardware_runtime(
            cfg.runtime.backend,
            cfg.runtime,
            RuntimeCreateContext(
                robot=manifest.robot,
                control=manifest.control,
            ),
        )
    robot = runtime.robot
    device = runtime.keyboard or runtime.gamepad
    try:
        manifest.robot.validate_compatible(robot.spec)
        task.configure_operator(getattr(cfg.runtime, "operator", None))
        command_binding = task.prepare(simulation=simulation, device=device, spawn_pose=spawn_pose)
        session = ControlSession(
            controller=task,
            robot=robot,
            command_binding=command_binding,
            period_s=manifest.control.period_s,
            state_timeout_s=manifest.control.state_timeout_s,
        )
        runtime.bind_session(session)
    except (ValidationError, ValueError):
        runtime.close()
        raise
    if task.operator_hint:
        print(task.operator_hint)
    with runtime:
        result = runtime.run()
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


def run_program(cfg: ProgramRunConfig) -> int:
    """Run one persistent multi-policy program on the selected backend."""
    from motrix_deploy.program import create_program

    program = create_program(cfg.program)
    simulation = isinstance(cfg.runtime, SimRuntimeCfg)
    if simulation:
        runtime = create_simulation_runtime(cfg.runtime.backend, cfg.runtime.build_runtime_config())
    else:
        runtime = create_hardware_runtime(
            cfg.runtime.backend,
            cfg.runtime,
            RuntimeCreateContext(robot=program.robot, control=program.control),
        )
    try:
        program.robot.validate_compatible(runtime.robot.spec)
        profile = getattr(cfg.runtime, "operator", None) or program.operator
        command_binding = (
            OperatorCommandBinding(
                runtime.keyboard or runtime.gamepad,
                profile,
                command_lower=program.command_lower,
                command_upper=program.command_upper,
            )
            if (runtime.keyboard or runtime.gamepad) is not None
            else program.prepare(None)
        )
        session = ControlSession(
            controller=program.controller,
            robot=runtime.robot,
            command_binding=command_binding,
            period_s=program.control.period_s,
            state_timeout_s=program.control.state_timeout_s,
        )
        runtime.bind_session(session)
    except (ValidationError, ValueError):
        runtime.close()
        raise
    with runtime:
        result = runtime.run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


def _run_program_config(cfg: DictConfig) -> None:
    try:
        typed_cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ProgramRunConfig), cfg))
        if not isinstance(typed_cfg, ProgramRunConfig):
            raise TypeError(f"Expected ProgramRunConfig, got {type(typed_cfg).__name__}")
        exit_code = run_program(typed_cfg)
    except (ArtifactError, OmegaConfBaseException, RuntimeError, TypeError, ValidationError, ValueError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from error
    if exit_code:
        raise SystemExit(exit_code)


_hydra_program_main = hydra.main(version_base=None, config_path="pkg://motrix_deploy.config", config_name="program")(
    _run_program_config
)

_hydra_program_real_main = hydra.main(
    version_base=None, config_path="pkg://motrix_deploy.config", config_name="program-real"
)(_run_program_config)

# deploy=<preset> aliases: one flag selects a whole application configuration.
# The simulation application keeps the generic program entry point, so its preset
# also selects the program group option.
_DEPLOY_PRESETS = {
    "g1-sim": (_hydra_program_main, ("program=g1-hot-swapping",)),
    "g1-real": (_hydra_program_real_main, ()),
}


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


def main() -> None:
    """Adapt ``mxlab deploy`` arguments to the deployment Hydra commands."""
    args = sys.argv[1:]
    if args[:1] == ["inspect"]:
        sys.argv = [f"{sys.argv[0]} inspect", *args[1:]]
        _hydra_inspect_main()
        return

    preset = next((argument.split("=", 1)[1] for argument in args if argument.startswith("deploy=")), None)
    if preset is not None:
        if preset not in _DEPLOY_PRESETS:
            raise SystemExit(f"unknown deploy preset {preset!r}; choose from {sorted(_DEPLOY_PRESETS)}")
        del sys.argv[1 + args.index(f"deploy={preset}")]
        entry_point, implicit_arguments = _DEPLOY_PRESETS[preset]
        sys.argv.extend(implicit_arguments)
        entry_point()
        return

    if any(argument == "program" or argument.startswith("program=") for argument in args):
        if args[:1] == ["program"]:
            del sys.argv[1]
        _hydra_program_main()
        return

    if not args:
        sys.argv.append("--help")
    _hydra_main()
