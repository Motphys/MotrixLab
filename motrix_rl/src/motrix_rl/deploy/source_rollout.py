# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Optional MotrixSim source rollout for a deployment artifact."""

import hashlib
from pathlib import Path

import numpy as np

from motrix_deploy.artifact import read_artifact
from motrix_deploy.policy import OnnxPolicy
from motrix_env_core import registry
from motrix_rl.deploy.api import SourceRolloutResult


def validate_motrixsim_source_rollout(
    artifact_path: str | Path,
    *,
    env_name: str,
    steps: int,
    seed: int,
    command: np.ndarray,
) -> SourceRolloutResult:
    """Run an exported deterministic policy in its MotrixSim task implementation."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError(f"command must contain three finite values, got {command}")
    artifact = read_artifact(artifact_path)
    policy = OnnxPolicy(artifact.policy_path, artifact.manifest.policy.input, artifact.manifest.policy.output)
    np.random.seed(seed)
    env = registry.make(env_name, num_envs=1)
    env.cfg.noise_config.level = 0.0
    env.cfg.spawn_xy_range = 0.0
    state = env.init_state()
    # Walk-flavor tasks keep their velocity commands in the episode-scoped
    # ``_commands`` buffer; pin row 0 so the rollout drives a fixed command.
    env._commands[0] = command
    observation = np.asarray(state.obs.policy[0], dtype=np.float32)
    reset_observation = observation.tolist()
    first_outputs: list[list[float]] = []
    trace = hashlib.sha256()
    completed = 0
    exit_reason = "completed"
    for _ in range(steps):
        action = policy.infer(observation)
        if len(first_outputs) < 4:
            first_outputs.append(action.tolist())
        trace.update(observation.tobytes(order="C"))
        trace.update(action.tobytes(order="C"))
        # The deployment runtime intentionally returns immutable arrays, while the direct frontend keeps
        # the action in mutable episode state for partial resets.
        state = env.step(action[None, :].copy())
        completed += 1
        if bool(state.terminated[0]):
            exit_reason = "terminated"
            break
        env._commands[0] = command
        observation = np.asarray(state.obs.policy[0], dtype=np.float32)
        if not np.isfinite(observation).all():
            exit_reason = "invalid_observation"
            break
    return SourceRolloutResult(
        success=exit_reason == "completed" and completed == steps,
        completed_steps=completed,
        exit_reason=exit_reason,
        reset_observation=reset_observation,
        first_policy_outputs=first_outputs,
        trace_sha256=trace.hexdigest(),
    )
