"""Evaluate a Go2 policy under reproducible random velocity commands.

This is deliberately separate from ``play_walk.py``: fixed-command playback
and its output format remain unchanged and this file can be removed to revert
the random-command evaluator.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from motrix_rl_rslrl.torch import wrap_env
from omegaconf import DictConfig
from play import _play_render_config, _resolve_play_target
from rsl_rl.runners import OnPolicyRunner

import motrix_envs  # noqa: F401 register environments
from motrix_env_core import registry as env_registry
from motrix_rl import runner
from motrix_rl.config import PlayConfig
from play_walk import _set_straight_command

logger = logging.getLogger(__name__)


@dataclass
class RandomWalkEvalConfig(PlayConfig):
    duration_seconds: float = 120.0
    warmup_seconds: float = 2.0
    eval_steps: int | None = None
    command_interval_seconds: float = 2.0
    command_seed: int = 1
    vx_range: tuple[float, float] = (-0.5, 0.5)
    vy_range: tuple[float, float] = (-0.5, 0.5)
    yaw_range: tuple[float, float] = (-0.5, 0.5)
    analysis_dir: str | None = None


ConfigStore.instance().store(name="eval_walk_random_schema", node=RandomWalkEvalConfig)


def _sample_command(rng: np.random.Generator, cfg: RandomWalkEvalConfig) -> np.ndarray:
    return np.asarray(
        [
            rng.uniform(*cfg.vx_range),
            rng.uniform(*cfg.vy_range),
            rng.uniform(*cfg.yaw_range),
        ],
        dtype=np.float32,
    )


def _run(cfg: RandomWalkEvalConfig) -> None:
    if cfg.duration_seconds <= 0 or cfg.warmup_seconds < 0:
        raise ValueError("duration_seconds must be positive and warmup_seconds non-negative")
    if cfg.eval_steps is not None and cfg.eval_steps <= 0:
        raise ValueError("eval_steps must be positive")
    if cfg.command_interval_seconds <= 0:
        raise ValueError("command_interval_seconds must be positive")

    target = _resolve_play_target(cfg)
    render = _play_render_config(target, cfg)
    handle = runner.create_run_handle(
        target.run,
        cfg_override=cfg.rl,
        play_num_envs=cfg.num_envs,
        seed=cfg.seed,
        randomize_seed=cfg.rand_seed,
        render=render,
    )
    trainer = handle.trainer
    if not hasattr(trainer, "_create_rslrl_config"):
        raise TypeError("eval_walk_random.py requires an RSLRL PPO policy/run")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer._set_seed(trainer._context.seed)
    env = env_registry.make(
        trainer._env_name,
        sim=trainer._sim,
        num_envs=trainer._context.play_num_envs,
        mode="play",
        seed=trainer._context.seed,
    )
    vec_env = wrap_env(env, device, render=render)
    policy_runner = OnPolicyRunner(vec_env, trainer._create_rslrl_config(), log_dir=None, device=device)
    policy_runner.load(str(target.policy_path), map_location=str(device))
    policy = policy_runner.get_inference_policy(device=device)

    obs, _ = vec_env.reset()
    rng = np.random.default_rng(cfg.command_seed)
    command = _sample_command(rng, cfg)
    _set_straight_command(env, command)
    ctrl_dt = float(getattr(env.cfg, "ctrl_dt", 0.02))
    interval_steps = max(1, round(cfg.command_interval_seconds / ctrl_dt))
    total_steps = cfg.eval_steps if cfg.eval_steps is not None else round(cfg.duration_seconds / ctrl_dt)
    warmup_steps = round(cfg.warmup_seconds / ctrl_dt)
    velocities, gyros, commands, orientations, heights = [], [], [], [], []
    yaw_rate_deltas = []
    previous_eval_gyro = None
    fall_count = 0
    command_count = 1
    started = time.monotonic()
    try:
        for step in range(total_steps):
            if step > 0 and step % interval_steps == 0:
                command = _sample_command(rng, cfg)
                command_count += 1
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            _set_straight_command(env, command)
            done = bool(torch.as_tensor(dones).detach().cpu().numpy().reshape(-1)[0])
            fall_count += int(done)
            if step >= warmup_steps and not done:
                velocities.append(np.asarray(env.get_local_linvel()[0], dtype=np.float64).copy())
                current_gyro = np.asarray(env.get_gyro()[0], dtype=np.float64).copy()
                gyros.append(current_gyro)
                if previous_eval_gyro is not None:
                    yaw_rate_deltas.append(float((current_gyro[2] - previous_eval_gyro[2]) / ctrl_dt))
                previous_eval_gyro = current_gyro
                commands.append(command.astype(np.float64).copy())
                gravity = np.asarray(env.get_gravity()[0], dtype=np.float64)
                orientations.append(float(np.linalg.norm(gravity[:2])))
                heights.append(float(env.sim_data["base_pos"][0, 2]))
            else:
                previous_eval_gyro = None
            if vec_env.render() is False:
                break
    finally:
        vec_env.close()

    output = Path(cfg.analysis_dir) if cfg.analysis_dir else target.run.run_dir / "walk_analysis_random"
    output.mkdir(parents=True, exist_ok=True)
    vel = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
    gyro = np.asarray(gyros, dtype=np.float64).reshape(-1, 3)
    cmd = np.asarray(commands, dtype=np.float64).reshape(-1, 3)
    if len(vel):
        command_velocity = np.column_stack((vel[:, 0], vel[:, 1], gyro[:, 2]))
        command_error = command_velocity - cmd
        linear_target = np.column_stack((cmd[:, :2], np.zeros(len(cmd), dtype=np.float64)))
        angular_target = np.column_stack((np.zeros((len(cmd), 2), dtype=np.float64), cmd[:, 2]))
        linear_error = vel - linear_target
        angular_error = gyro - angular_target
        yaw_rate_deltas_array = np.asarray(yaw_rate_deltas, dtype=np.float64)
        xy_norm = np.linalg.norm(cmd[:, :2], axis=1)
        moving = xy_norm > 1e-8
        orthogonal = np.linalg.norm(vel[:, :2], axis=1)
        if np.any(moving):
            perpendicular = np.column_stack((-cmd[moving, 1], cmd[moving, 0])) / xy_norm[moving, None]
            orthogonal[moving] = np.abs(np.sum(vel[moving, :2] * perpendicular, axis=1))
        metrics = {
            "metrics_schema_version": 2,
            "command_mode": "random_uniform",
            "command_axes": ["vx", "vy", "yaw_rate"],
            "command_ranges": {"vx": list(cfg.vx_range), "vy": list(cfg.vy_range), "yaw": list(cfg.yaw_range)},
            "command_interval_seconds": cfg.command_interval_seconds,
            "command_seed": cfg.command_seed,
            "command_count": command_count,
            "mean_command": cmd.mean(axis=0).tolist(),
            "mean_velocity": command_velocity.mean(axis=0).tolist(),
            "steady_state_error": command_error.mean(axis=0).tolist(),
            "rmse": np.sqrt(np.mean(np.square(command_error), axis=0)).tolist(),
            "p95_error": np.percentile(np.abs(command_error), 95, axis=0).tolist(),
            "mean_linear_velocity": vel.mean(axis=0).tolist(),
            "linear_velocity_rmse": np.sqrt(np.mean(np.square(linear_error), axis=0)).tolist(),
            "linear_velocity_p95_error": np.percentile(np.abs(linear_error), 95, axis=0).tolist(),
            "mean_angular_velocity": gyro.mean(axis=0).tolist(),
            "angular_velocity_rmse": np.sqrt(np.mean(np.square(angular_error), axis=0)).tolist(),
            "angular_velocity_p95_error": np.percentile(np.abs(angular_error), 95, axis=0).tolist(),
            "yaw_rate_error_mean": float(np.mean(angular_error[:, 2])),
            "yaw_rate_rmse": float(np.sqrt(np.mean(np.square(angular_error[:, 2])))),
            "yaw_rate_p95_error": float(np.percentile(np.abs(angular_error[:, 2]), 95)),
            "yaw_rate_delta_rms": (
                float(np.sqrt(np.mean(np.square(yaw_rate_deltas_array))))
                if yaw_rate_deltas_array.size
                else 0.0
            ),
            "yaw_rate_delta_samples": int(yaw_rate_deltas_array.size),
            "roll_pitch_rate_rms": float(np.sqrt(np.mean(np.sum(np.square(gyro[:, :2]), axis=1)))),
            "orthogonal_drift": float(np.mean(orthogonal)),
            "fall_rate": float(fall_count / max(total_steps, 1)),
            "termination_count": int(fall_count),
            "termination_events_per_minute": float(fall_count * 60.0 / (total_steps * ctrl_dt)),
            "orientation_rms": float(np.sqrt(np.mean(np.square(orientations)))),
            "base_height_std": float(np.std(heights)),
            "samples": int(len(vel)),
        }
    else:
        metrics = {
            "metrics_schema_version": 2,
            "command_mode": "random_uniform",
            "command_axes": ["vx", "vy", "yaw_rate"],
            "samples": 0,
            "command_count": command_count,
            "termination_count": int(fall_count),
        }
    path = output / "velocity_metrics_random.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved random-command metrics: %s", path)


@hydra.main(version_base=None, config_path="../configs", config_name="eval_walk_random")
def main(cfg: DictConfig) -> None:
    _run(cfg)


if __name__ == "__main__":
    main()
