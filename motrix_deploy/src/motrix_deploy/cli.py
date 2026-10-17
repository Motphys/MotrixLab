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
from motrix_deploy.config import DeployRunConfig, SimRuntimeCfg
from motrix_deploy.errors import ArtifactError, ValidationError
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
        sys.argv = ["mxdeploy", *argv]
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
        print(f"mxdeploy {__version__}")
        return 0
    if not args:
        sys.argv.append("--help")
    _hydra_main()
    return None


if __name__ == "__main__":
    raise SystemExit(main())
