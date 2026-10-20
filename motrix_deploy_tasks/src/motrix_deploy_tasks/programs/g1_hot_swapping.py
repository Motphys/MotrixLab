# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Load and validate the G1 Walk/WBT hot-swapping program."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from hydra.core.config_store import ConfigStore

from motrix_deploy.artifact.io import read_artifact
from motrix_deploy.contracts import RobotSpec
from motrix_deploy.errors import ValidationError
from motrix_deploy.operator import ButtonGestureCfg, OperatorProfileCfg, PlanarVelocityOperatorCfg
from motrix_deploy.policy import OnnxPolicy
from motrix_deploy.program import DeploymentProgram
from motrix_deploy.robot.g1_standing import resolve_g1_standing_gains
from motrix_deploy.runtime.preparation import SkillPreparationSpec
from motrix_deploy_tasks.state_machines.g1_states import (
    G1HotSwappingController,
    G1TransitionSettings,
    G1WalkState,
    G1WbtState,
)
from motrix_deploy_tasks.tasks.g1_walk import G1WalkTaskSpec
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtTaskSpec, load_g1_wbt_motion


def _default_operator() -> OperatorProfileCfg:
    return OperatorProfileCfg(
        intents={
            "request_fixstand": ButtonGestureCfg(pressed="1"),
            "request_walk": ButtonGestureCfg(pressed="2"),
            "request_wbt": ButtonGestureCfg(pressed="3"),
            "request_passive": ButtonGestureCfg(pressed="0"),
            "request_emergency": ButtonGestureCfg(pressed="x"),
        },
        planar_velocity=PlanarVelocityOperatorCfg(),
    )


@dataclass
class G1HotSwappingProgramCfg:
    """Artifact paths, transition timing and backend-neutral operator defaults."""

    name: str = "g1-hot-swapping"
    walk_artifact: str = "artifacts/g1-flat.deploy"
    wbt_artifact: str = "artifacts/g1-wbt-dance.deploy"
    policy_transition_duration_s: float = 0.5
    passive_transition_min_s: float = 0.5
    passive_transition_max_s: float = 3.0
    passive_max_joint_speed_rad_s: float = 0.5
    passive_gain_ramp_s: float = 0.5
    passive_kd: float = 3.0
    initial_state: str = "fixstand"
    operator: OperatorProfileCfg = field(default_factory=_default_operator)

    def __post_init__(self) -> None:
        if self.name != "g1-hot-swapping":
            raise ValueError(f"unsupported program name: {self.name!r}")
        if self.initial_state not in {"passive", "fixstand"}:
            raise ValueError("initial_state must be passive or fixstand")
        if not self.walk_artifact or not self.wbt_artifact:
            raise ValueError("walk_artifact and wbt_artifact must be non-empty paths")
        for name in (
            "policy_transition_duration_s",
            "passive_transition_min_s",
            "passive_transition_max_s",
            "passive_max_joint_speed_rad_s",
            "passive_gain_ramp_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.passive_transition_min_s > self.passive_transition_max_s:
            raise ValueError("passive transition minimum must not exceed maximum")
        if self.passive_kd < 0:
            raise ValueError("passive damping gain must be non-negative")


def install_program_schema() -> None:
    ConfigStore.instance().store(group="program", name="_g1_hot_swapping_schema", node=G1HotSwappingProgramCfg)


def _indices(source: tuple[str, ...], target: tuple[str, ...]) -> np.ndarray:
    if set(source) != set(target):
        raise ValidationError("program.robot.joint_names", f"joint set {sorted(target)}", sorted(source))
    return np.asarray([source.index(name) for name in target], dtype=np.int64)


def _validate_robot_compatibility(canonical: RobotSpec, candidate: RobotSpec) -> None:
    if canonical.base_link_name != candidate.base_link_name:
        raise ValidationError("program.robot.base_link_name", canonical.base_link_name, candidate.base_link_name)
    order = _indices(candidate.joint_names, canonical.joint_names)
    for name in ("default_joint_position", "position_lower", "position_upper", "torque_limit"):
        expected = getattr(canonical, name)
        actual = getattr(candidate, name)[order]
        if not np.allclose(expected, actual, atol=1e-6, rtol=0):
            raise ValidationError(f"program.robot.{name}", expected.tolist(), actual.tolist())


def _reorder(values: object, source: tuple[str, ...], target: tuple[str, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)[_indices(source, target)].astype(np.float32)


def create_program(config: G1HotSwappingProgramCfg) -> DeploymentProgram:
    """Preload both artifacts and construct one backend-neutral persistent controller."""
    if not isinstance(config, G1HotSwappingProgramCfg):
        raise TypeError(f"expected G1HotSwappingProgramCfg, got {type(config).__name__}")
    walk_artifact = read_artifact(Path(config.walk_artifact))
    wbt_artifact = read_artifact(Path(config.wbt_artifact))
    walk_manifest = walk_artifact.manifest
    wbt_manifest = wbt_artifact.manifest
    if not isinstance(walk_manifest.task, G1WalkTaskSpec):
        raise ValidationError("program.walk_artifact.task", G1WalkTaskSpec.task_name, walk_manifest.task.name)
    if not isinstance(wbt_manifest.task, G1WbtTaskSpec):
        raise ValidationError("program.wbt_artifact.task", G1WbtTaskSpec.task_name, wbt_manifest.task.name)
    if walk_manifest.control != wbt_manifest.control:
        raise ValidationError("program.control", walk_manifest.control, wbt_manifest.control)
    robot = walk_manifest.robot
    _validate_robot_compatibility(robot, wbt_manifest.robot)

    walk_policy = OnnxPolicy(walk_artifact.policy_path, walk_manifest.policy.input, walk_manifest.policy.output)
    wbt_policy = OnnxPolicy(wbt_artifact.policy_path, wbt_manifest.policy.input, wbt_manifest.policy.output)
    motion = load_g1_wbt_motion(wbt_manifest.task, wbt_artifact)
    walk = G1WalkState(walk_manifest.task, robot, walk_policy, robot.joint_names)
    wbt = G1WbtState(
        wbt_manifest.task,
        wbt_manifest.robot,
        motion,
        wbt_policy,
        robot.joint_names,
        takeover_hold_steps=max(1, round(config.policy_transition_duration_s / walk_manifest.control.period_s)),
    )
    wbt_spec = wbt_manifest.task
    wbt_robot = wbt_manifest.robot
    fixstand_position = _reorder(
        wbt_robot.default_joint_position
        if wbt_spec.preparation_joint_position is None
        else wbt_spec.preparation_joint_position,
        wbt_robot.joint_names,
        robot.joint_names,
    )
    # FixStand is a fixed posture owned by the deploy, not by any policy.
    fixstand_kp, fixstand_kd = resolve_g1_standing_gains(robot.joint_names)
    # Skill preparation stays per-artifact: every skill ships its own stand-up recipe.
    wbt_preparation_kp = _reorder(
        wbt_spec.kp if wbt_spec.preparation_kp is None else wbt_spec.preparation_kp,
        wbt_robot.joint_names,
        robot.joint_names,
    )
    wbt_preparation_kd = _reorder(
        wbt_spec.kd if wbt_spec.preparation_kd is None else wbt_spec.preparation_kd,
        wbt_robot.joint_names,
        robot.joint_names,
    )
    settings = G1TransitionSettings(
        policy_duration_s=config.policy_transition_duration_s,
        passive_duration_min_s=config.passive_transition_min_s,
        passive_duration_max_s=config.passive_transition_max_s,
        passive_max_joint_speed_rad_s=config.passive_max_joint_speed_rad_s,
        passive_gain_ramp_s=config.passive_gain_ramp_s,
    )
    controller = G1HotSwappingController(
        robot=robot,
        walk=walk,
        wbt=wbt,
        fixstand_joint_position=fixstand_position,
        fixstand_kp=fixstand_kp,
        fixstand_kd=fixstand_kd,
        passive_kd=np.full(robot.joint_count, config.passive_kd, dtype=np.float32),
        wbt_preparation=SkillPreparationSpec(
            joint_position=fixstand_position,
            kp=wbt_preparation_kp,
            kd=wbt_preparation_kd,
            ramp_duration_s=wbt_spec.preparation_ramp_duration_s,
            settle_duration_s=wbt_spec.preparation_settle_duration_s,
            timeout_s=wbt_spec.preparation_timeout_s,
        ),
        initial_state=config.initial_state,
        settings=settings,
    )
    return DeploymentProgram(
        robot=robot,
        control=walk_manifest.control,
        controller=controller,
        command_lower=walk.processor.command_lower,
        command_upper=walk.processor.command_upper,
        operator=config.operator,
    )


__all__ = ["G1HotSwappingProgramCfg", "create_program", "install_program_schema"]
