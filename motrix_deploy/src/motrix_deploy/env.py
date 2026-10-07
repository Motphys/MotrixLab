# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Installed robot-independent deployment world factories."""

from importlib import metadata

from motrix_env_core import registry
from motrix_env_core.config.scene import SceneCfg

DEPLOY_ENV_ENTRY_POINT_GROUP = "motrix_deploy.envs"


def available_deploy_envs() -> tuple[str, ...]:
    """List installed deployment environment IDs without loading factories."""
    return tuple(sorted({entry.name for entry in metadata.entry_points(group=DEPLOY_ENV_ENTRY_POINT_GROUP)}))


def create_deploy_env(deploy_env_id: str) -> SceneCfg:
    """Create a fresh robot-free scene from one installed environment factory."""
    entries = tuple(metadata.entry_points(group=DEPLOY_ENV_ENTRY_POINT_GROUP))
    matches = [entry for entry in entries if entry.name == deploy_env_id]
    if not matches:
        available = ", ".join(sorted({entry.name for entry in entries})) or "none"
        raise ValueError(f"Unknown deployment environment {deploy_env_id!r}; available environments: {available}")
    if len(matches) != 1:
        raise ValueError(f"Multiple deployment environment plugins provide {deploy_env_id!r}")
    scene = matches[0].load()()
    if not isinstance(scene, SceneCfg):
        raise TypeError(f"Deployment environment {deploy_env_id!r} must return SceneCfg, got {type(scene).__name__}")
    if scene.objs.robot is not None:
        raise ValueError(f"Deployment environment {deploy_env_id!r} must leave scene.objs.robot empty")
    return scene


def assemble_deploy_scene(deploy_env_id: str, robot_id: str) -> SceneCfg:
    """Assemble an independently selected world and registered robot for deployment."""
    scene = create_deploy_env(deploy_env_id)
    scene.objs.robot = registry.make_robot_config(robot_id)
    return scene
