# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Run a trained quadruped policy with a fixed straight-line command."""

import csv
import json
import logging
import time
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from motrix_rl_rslrl.torch import wrap_env
from omegaconf import DictConfig
from play import _play_render_config, _resolve_play_target
from rsl_rl.runners import OnPolicyRunner

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry as env_registry
from motrix_rl import runner
from motrix_rl.cli import to_typed_config
from motrix_rl.config import PlayConfig

logger = logging.getLogger(__name__)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class WalkPlayConfig(PlayConfig):
    """Configuration for bounded, straight-line quadruped validation."""

    duration_seconds: float = 100.0
    warmup_seconds: float = 2.0
    eval_steps: int | None = None
    forward_velocity: float = 0.5
    lateral_velocity: float = 0.0
    yaw_velocity: float = 0.0
    eval_env: str | None = None
    """Optional environment used for cross-environment policy evaluation."""
    analysis_dir: str | None = None
    analysis_env_index: int = 0
    metrics_path: str | None = None
    no_render: bool = False


ConfigStore.instance().store(name="play_walk_schema", node=WalkPlayConfig)


def _set_straight_command(env, command: np.ndarray) -> None:
    """Pin the walk task's episode command for every parallel environment."""

    commands = getattr(env, "_commands", None)
    if commands is None:
        raise TypeError(
            f"Environment {type(env).__name__} does not expose the quadruped walk command buffer '_commands'. "
            "Use a locomotion walk environment for play_walk.py."
        )

    if isinstance(commands, torch.Tensor):
        value = torch.as_tensor(command, dtype=commands.dtype, device=commands.device)
    else:
        value = np.asarray(command, dtype=np.float32)
    commands[...] = value


_FOOT_NAMES = ("FL", "FR", "RL", "RR")
_FOOT_COLORS = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#ff7f0e", "RR": "#2ca02c"}


def _compute_motion_metrics(
    velocities: np.ndarray,
    gyros: np.ndarray,
    command: np.ndarray,
    *,
    yaw_rate_deltas: np.ndarray,
    vz_deltas: np.ndarray,
    orientation_samples: list[float],
    height_samples: list[float],
    termination_count: int,
    total_steps: int,
    ctrl_dt: float,
) -> dict[str, object]:
    """Compute fixed-command motion metrics with explicit linear/angular axes."""

    command64 = np.asarray(command, dtype=np.float64)
    linear_command = np.array([command64[0], command64[1], 0.0], dtype=np.float64)
    angular_command = np.array([0.0, 0.0, command64[2]], dtype=np.float64)
    linear_error = velocities - linear_command
    angular_error = gyros - angular_command

    # Preserve the original compact fields, but make their axes match the
    # external command contract: [vx, vy, yaw_rate]. Version 1 accidentally
    # used [vx, vy, vz] while subtracting a yaw command from vz.
    command_velocity = np.column_stack((velocities[:, 0], velocities[:, 1], gyros[:, 2]))
    command_error = command_velocity - command64

    # ``vz`` is deliberately reported independently from yaw.  It is the
    # vertical linear velocity (m/s), whose target is always zero for the
    # flat-walk commands.  The delta metric is an optional high-frequency
    # indicator (m/s^2) for up/down bouncing.
    vz_error = linear_error[:, 2]

    command_xy = command64[:2]
    command_xy_norm = float(np.linalg.norm(command_xy))
    if command_xy_norm > 1e-8:
        perpendicular = np.array([-command_xy[1], command_xy[0]], dtype=np.float64) / command_xy_norm
        orthogonal_drift = float(np.mean(np.abs(velocities[:, :2] @ perpendicular)))
    else:
        orthogonal_drift = float(np.mean(np.linalg.norm(velocities[:, :2], axis=1)))

    evaluated_seconds = total_steps * ctrl_dt
    actual_speed_xy = np.linalg.norm(velocities[:, :2], axis=1)
    command_speed_xy = float(np.linalg.norm(command64[:2]))
    return {
        "metrics_schema_version": 2,
        "command": command64.tolist(),
        "command_axes": ["vx", "vy", "yaw_rate"],
        "mean_velocity": np.mean(command_velocity, axis=0).tolist(),
        "steady_state_error": np.mean(command_error, axis=0).tolist(),
        # Named fields make the JSON self-describing for scripts and humans.
        "actual_velocity": {
            "vx": float(np.mean(velocities[:, 0])),
            "vy": float(np.mean(velocities[:, 1])),
            "vz": float(np.mean(velocities[:, 2])),
            "yaw_rate": float(np.mean(gyros[:, 2])),
        },
        "tracking_error": {
            "vx": float(np.mean(command_error[:, 0])),
            "vy": float(np.mean(command_error[:, 1])),
            "yaw_rate": float(np.mean(command_error[:, 2])),
        },
        "tracking_mae": np.mean(np.abs(command_error), axis=0).tolist(),
        "actual_speed_xy_mean": float(np.mean(actual_speed_xy)),
        "actual_speed_xy_rmse": float(np.sqrt(np.mean(np.square(actual_speed_xy)))),
        "command_speed_xy": command_speed_xy,
        "rmse": np.sqrt(np.mean(np.square(command_error), axis=0)).tolist(),
        "p95_error": np.percentile(np.abs(command_error), 95, axis=0).tolist(),
        "mean_linear_velocity": np.mean(velocities, axis=0).tolist(),
        "linear_velocity_rmse": np.sqrt(np.mean(np.square(linear_error), axis=0)).tolist(),
        "linear_velocity_p95_error": np.percentile(np.abs(linear_error), 95, axis=0).tolist(),
        "vz_mean": float(np.mean(velocities[:, 2])),
        "vz_error_mean": float(np.mean(vz_error)),
        "vz_rms": float(np.sqrt(np.mean(np.square(vz_error)))),
        "vz_p95_abs": float(np.percentile(np.abs(vz_error), 95)),
        "vz_std": float(np.std(velocities[:, 2])),
        "vz_delta_rms": float(np.sqrt(np.mean(np.square(vz_deltas)))) if vz_deltas.size else 0.0,
        "vz_delta_samples": int(vz_deltas.size),
        "mean_angular_velocity": np.mean(gyros, axis=0).tolist(),
        "angular_velocity_rmse": np.sqrt(np.mean(np.square(angular_error), axis=0)).tolist(),
        "angular_velocity_p95_error": np.percentile(np.abs(angular_error), 95, axis=0).tolist(),
        "yaw_rate_error_mean": float(np.mean(angular_error[:, 2])),
        "yaw_rate_rmse": float(np.sqrt(np.mean(np.square(angular_error[:, 2])))),
        "yaw_rate_p95_error": float(np.percentile(np.abs(angular_error[:, 2]), 95)),
        "yaw_rate_delta_rms": (
            float(np.sqrt(np.mean(np.square(yaw_rate_deltas)))) if yaw_rate_deltas.size else 0.0
        ),
        "yaw_rate_delta_samples": int(yaw_rate_deltas.size),
        "roll_pitch_rate_rms": float(np.sqrt(np.mean(np.sum(np.square(gyros[:, :2]), axis=1)))),
        "orthogonal_drift": orthogonal_drift,
        # Kept as a compatibility alias. This is a termination-event rate per
        # control step, not an episode-level probability.
        "fall_rate": float(termination_count / max(total_steps, 1)),
        "termination_count": int(termination_count),
        "termination_events_per_minute": (
            float(termination_count * 60.0 / evaluated_seconds) if evaluated_seconds > 0.0 else 0.0
        ),
        "orientation_rms": (
            float(np.sqrt(np.mean(np.square(orientation_samples)))) if orientation_samples else 0.0
        ),
        "base_height_std": float(np.std(height_samples)) if height_samples else 0.0,
        "samples": int(len(velocities)),
    }


def _compute_trajectory_metrics(
    base_positions: np.ndarray,
    yaw_samples: np.ndarray,
    command: np.ndarray,
    segment_ids: np.ndarray | None = None,
) -> dict[str, float | int | str | None]:
    """Compute reset-aware world-frame drift for a nominal straight command."""

    if len(base_positions) < 2 or len(yaw_samples) < 2:
        return {"trajectory_metrics_available": 0}
    base_positions = np.asarray(base_positions, dtype=np.float64)
    yaw_samples = np.asarray(yaw_samples, dtype=np.float64)
    if segment_ids is None:
        segment_ids = np.zeros(len(base_positions), dtype=np.int64)
    else:
        segment_ids = np.asarray(segment_ids, dtype=np.int64)
    if len(segment_ids) != len(base_positions):
        raise ValueError("segment_ids must match base_positions")

    yaw_drift_parts: list[np.ndarray] = []
    cross_track_parts: list[np.ndarray] = []
    segment_final_yaw: list[float] = []
    segment_final_cross_track: list[float] = []
    segment_along_track: list[float] = []
    initial_yaws: list[float] = []
    for segment_id in np.unique(segment_ids):
        selected = segment_ids == segment_id
        if np.count_nonzero(selected) < 2:
            continue
        positions = base_positions[selected]
        yaw = np.unwrap(yaw_samples[selected])
        initial_yaw = float(yaw[0])
        initial_yaws.append(initial_yaw)
        yaw_drift = yaw - initial_yaw
        displacement = positions[:, :2] - positions[0, :2]
        forward = np.array([np.cos(initial_yaw), np.sin(initial_yaw)], dtype=np.float64)
        lateral = np.array([-np.sin(initial_yaw), np.cos(initial_yaw)], dtype=np.float64)
        cross_track = displacement @ lateral
        along_track = displacement @ forward
        yaw_drift_parts.append(yaw_drift)
        cross_track_parts.append(cross_track)
        segment_final_yaw.append(float(yaw_drift[-1]))
        segment_final_cross_track.append(float(cross_track[-1]))
        segment_along_track.append(float(along_track[-1]))

    if not yaw_drift_parts:
        return {"trajectory_metrics_available": 0}
    yaw_drift = np.concatenate(yaw_drift_parts)
    cross_track = np.concatenate(cross_track_parts)
    # Reverse commands produce negative signed along-track displacement, but
    # travelled distance must remain positive for drift-per-metre metrics.
    total_along_track = float(np.sum(np.abs(segment_along_track)))
    total_abs_final_cross_track = float(np.sum(np.abs(segment_final_cross_track)))
    # These accumulated metrics are meaningful for straight commands only.
    if abs(float(command[2])) > 1e-6:
        mode = "nonzero_yaw_command"
    else:
        mode = "straight"
    return {
        "trajectory_metrics_available": 1,
        "trajectory_command_mode": mode,
        "initial_yaw_rad": initial_yaws[0],
        "final_yaw_drift_rad": segment_final_yaw[-1],
        "max_abs_yaw_drift_rad": float(np.max(np.abs(yaw_drift))),
        "yaw_drift_rms_rad": float(np.sqrt(np.mean(np.square(yaw_drift)))),
        "final_cross_track_error_m": segment_final_cross_track[-1],
        "max_abs_cross_track_error_m": float(np.max(np.abs(cross_track))),
        "cross_track_error_rms_m": float(np.sqrt(np.mean(np.square(cross_track)))),
        "final_along_track_distance_m": segment_along_track[-1],
        "trajectory_segment_count": int(len(segment_final_cross_track)),
        "mean_abs_segment_final_yaw_drift_rad": float(np.mean(np.abs(segment_final_yaw))),
        "mean_abs_segment_final_cross_track_error_m": float(
            np.mean(np.abs(segment_final_cross_track))
        ),
        "total_segment_along_track_distance_m": total_along_track,
        "cross_track_error_per_meter": (
            total_abs_final_cross_track / total_along_track if total_along_track > 1e-8 else 0.0
        ),
        "trajectory_samples": int(len(base_positions)),
    }


def _write_foot_landings_csv(path: Path, landings: dict[str, list[tuple[int, float, float, float]]]) -> None:
    lines = ["step,time_s,foot,x,y"]
    for foot in _FOOT_NAMES:
        for step, time_s, x, y in landings[foot]:
            lines.append(f"{step},{time_s:.6f},{foot},{x:.6f},{y:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_velocity_trace_csv(path: Path, samples: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    """Write one row per evaluated control step for plotting and diagnosis."""

    header = (
        "time_s,command_vx,command_vy,command_yaw_rate,"
        "actual_vx,actual_vy,actual_vz,actual_yaw_rate,"
        "error_vx,error_vy,error_yaw_rate,actual_speed_xy\n"
    )
    rows = [header]
    for time_s, actual_linear, actual_gyro, command in samples:
        error = np.array([actual_linear[0], actual_linear[1], actual_gyro[2]], dtype=np.float64) - command
        rows.append(
            f"{time_s:.6f},{command[0]:.8f},{command[1]:.8f},{command[2]:.8f},"
            f"{actual_linear[0]:.8f},{actual_linear[1]:.8f},{actual_linear[2]:.8f},{actual_gyro[2]:.8f},"
            f"{error[0]:.8f},{error[1]:.8f},{error[2]:.8f},"
            f"{np.linalg.norm(actual_linear[:2]):.8f}\n"
        )
    path.write_text("".join(rows), encoding="utf-8")


def _write_foot_landings_svg(path: Path, landings: dict[str, list[tuple[int, float, float, float]]]) -> None:
    """Write a dependency-free planar landing-position plot as SVG."""

    points = [(x, y) for foot in _FOOT_NAMES for _, _, x, y in landings[foot]]
    if not points:
        points = [(0.0, 0.0)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)
    padding_x = max(0.15 * span_x, 0.03)
    padding_y = max(0.15 * span_y, 0.03)
    min_x -= padding_x
    max_x += padding_x
    min_y -= padding_y
    max_y += padding_y

    width, height = 960, 720
    margin_left, margin_top, margin_right, margin_bottom = 90, 55, 35, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def project(x: float, y: float) -> tuple[float, float]:
        px = margin_left + (x - min_x) / (max_x - min_x) * plot_width
        py = margin_top + (max_y - y) / (max_y - min_y) * plot_height
        return px, py

    def tick_values(low: float, high: float, count: int = 6) -> list[float]:
        return [low + (high - low) * i / (count - 1) for i in range(count)]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" '
        'font-weight="bold">Go2 foot landing positions (body frame)</text>',
    ]

    for value in tick_values(min_x, max_x):
        x, _ = project(value, min_y)
        svg.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" '
                   'stroke="#e5e5e5"/>')
        svg.append(f'<text x="{x:.2f}" y="{height - 42}" text-anchor="middle" font-family="sans-serif" '
                   f'font-size="12">{value:.3f}</text>')
    for value in tick_values(min_y, max_y):
        _, y = project(min_x, value)
        svg.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" '
                   'stroke="#e5e5e5"/>')
        svg.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" '
                   f'font-size="12">{value:.3f}</text>')

    svg.extend(
        [
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
            'fill="none" stroke="#333"/>',
            f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 12}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14">x (forward, m)</text>',
            f'<text x="18" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14" transform="rotate(-90 18 '
            f'{margin_top + plot_height / 2:.2f})">y (left, m)</text>',
        ]
    )

    for foot in _FOOT_NAMES:
        foot_points = landings[foot]
        if not foot_points:
            continue
        color = _FOOT_COLORS[foot]
        projected = [project(x, y) for _, _, x, y in foot_points]
        path_data = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f},{y:.2f}" for index, (x, y) in enumerate(projected)
        )
        svg.append(
            f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2" opacity="0.75"/>'
        )
        for index, (x, y) in enumerate(projected):
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>{escape(foot)} '
                f"landing {index + 1}</title></circle>"
            )

    legend_x, legend_y = margin_left + 12, margin_top + 18
    for index, foot in enumerate(_FOOT_NAMES):
        y = legend_y + index * 22
        color = _FOOT_COLORS[foot]
        svg.append(f'<circle cx="{legend_x}" cy="{y - 4}" r="5" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 12}" y="{y}" font-family="sans-serif" font-size="13">{foot}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _save_foot_analysis(
    output_dir: Path,
    landings: dict[str, list[tuple[int, float, float, float]]],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "foot_landings.csv"
    svg_path = output_dir / "foot_landings.svg"
    _write_foot_landings_csv(csv_path, landings)
    _write_foot_landings_svg(svg_path, landings)
    return csv_path, svg_path


def _write_left_right_bias_analysis(
    output_dir: Path,
    landings: dict[str, list[tuple[int, float, float, float]]],
) -> tuple[Path, Path]:
    """Write left/right bias metrics and a diagnostic figure in body coordinates."""

    pairs = (("front", "FL", "FR"), ("rear", "RL", "RR"))
    metrics: list[dict[str, float | str]] = []
    for leg, left, right in pairs:
        left_points = landings[left]
        right_points = landings[right]
        left_xy = np.asarray([(point[2], point[3]) for point in left_points], dtype=np.float64).reshape(-1, 2)
        right_xy = np.asarray([(point[2], point[3]) for point in right_points], dtype=np.float64).reshape(-1, 2)
        if left_xy.size and right_xy.size:
            delta_x = float(left_xy[:, 0].mean() - right_xy[:, 0].mean())
            mirror_y = float(left_xy[:, 1].mean() + right_xy[:, 1].mean())
        else:
            delta_x = float("nan")
            mirror_y = float("nan")
        metrics.extend(
            [
                {"leg": leg, "metric": "left_minus_right_x", "value_m": delta_x},
                {"leg": leg, "metric": "left_plus_right_y", "value_m": mirror_y},
            ]
        )

    csv_path = output_dir / "left_right_bias.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("leg", "metric", "value_m"))
        writer.writeheader()
        writer.writerows(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"FL": "#d62728", "FR": "#1f77b4", "RL": "#ff7f0e", "RR": "#2ca02c"}

    ax = axes[0, 0]
    for foot in _FOOT_NAMES:
        points = np.asarray([(point[2], point[3]) for point in landings[foot]], dtype=np.float64).reshape(-1, 2)
        if points.size:
            ax.scatter(points[:, 0], points[:, 1], label=foot, color=colors[foot], alpha=0.8)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    ax.set_title("Landing points in body frame")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(ncol=4)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    x_data = [
        [point[2] for point in landings[foot]] or [np.nan]
        for foot in _FOOT_NAMES
    ]
    box = ax.boxplot(x_data, tick_labels=_FOOT_NAMES, patch_artist=True)
    for patch, foot in zip(box["boxes"], _FOOT_NAMES):
        patch.set_facecolor(colors[foot])
        patch.set_alpha(0.45)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.set_title("Forward landing position by foot")
    ax.set_ylabel("x forward (m)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    left_y = [point[3] for foot in ("FL", "RL") for point in landings[foot]]
    mirrored_right_y = [-point[3] for foot in ("FR", "RR") for point in landings[foot]]
    y_data = [left_y or [np.nan], mirrored_right_y or [np.nan]]
    box = ax.boxplot(y_data, tick_labels=("left y", "-right y"), patch_artist=True)
    for patch, color in zip(box["boxes"], ("#d62728", "#1f77b4")):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.set_title("Lateral mirror check")
    ax.set_ylabel("mirrored lateral position (m)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    labels = ["front Δx", "rear Δx", "front εy", "rear εy"]
    values = [
        float(metrics[0]["value_m"]),
        float(metrics[2]["value_m"]),
        float(metrics[1]["value_m"]),
        float(metrics[3]["value_m"]),
    ]
    bars = ax.bar(labels, values, color=("#8c564b", "#9467bd", "#e377c2", "#7f7f7f"))
    ax.axhline(0.0, color="0.2", linewidth=0.9)
    ax.set_title("Left/right bias metrics")
    ax.set_ylabel("meters")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.3f}", ha="center", va="bottom")

    fig.suptitle("Go2 left/right footfall symmetry", fontsize=15)
    png_path = output_dir / "left_right_bias.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path


def _run_walk(cfg: WalkPlayConfig) -> None:
    if cfg.duration_seconds <= 0.0:
        raise ValueError(f"duration_seconds must be positive, got {cfg.duration_seconds}")
    if cfg.analysis_env_index < 0:
        raise ValueError(f"analysis_env_index must be non-negative, got {cfg.analysis_env_index}")
    if cfg.warmup_seconds < 0.0 or cfg.warmup_seconds >= cfg.duration_seconds:
        if cfg.eval_steps is None:
            raise ValueError("warmup_seconds must be non-negative and shorter than duration_seconds")
    if cfg.eval_steps is not None and cfg.eval_steps <= 0:
        raise ValueError("eval_steps must be positive when provided")

    target = _resolve_play_target(cfg)
    render_cfg = replace(cfg, record_seconds=cfg.duration_seconds)
    render = None if cfg.no_render else _play_render_config(target, render_cfg)
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
        raise TypeError("play_walk.py requires an RSLRL PPO policy/run")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer._set_seed(trainer._context.seed)
    eval_env_name = cfg.eval_env or trainer._env_name
    if cfg.eval_env is not None:
        logger.info("Evaluating policy trained for %s on %s", trainer._env_name, eval_env_name)
    env = env_registry.make(
        eval_env_name,
        sim=trainer._sim,
        num_envs=trainer._context.play_num_envs,
        mode="play",
        seed=trainer._context.seed,
    )
    vec_env = wrap_env(env, device, render=render)
    if cfg.analysis_env_index >= env.num_envs:
        vec_env.close()
        raise ValueError(f"analysis_env_index {cfg.analysis_env_index} is outside num_envs={env.num_envs}")
    policy_runner = OnPolicyRunner(
        vec_env,
        trainer._create_rslrl_config(),
        log_dir=None,
        device=device,
    )
    policy_runner.load(str(target.policy_path), map_location=str(device))
    policy = policy_runner.get_inference_policy(device=device)

    command = np.asarray(
        [cfg.forward_velocity, cfg.lateral_velocity, cfg.yaw_velocity],
        dtype=np.float32,
    )
    obs, _ = vec_env.reset()
    _set_straight_command(env, command)
    started = time.monotonic()
    recording = cfg.record_video
    fps = 60.0
    analysis_dir = Path(cfg.analysis_dir) if cfg.analysis_dir else target.run.run_dir / "walk_analysis"
    landings: dict[str, list[tuple[int, float, float, float]]] = {foot: [] for foot in _FOOT_NAMES}
    previous_contact = np.asarray(env.feet_contact[cfg.analysis_env_index], dtype=bool).copy()
    contact_initialized = False
    step_index = 0
    ctrl_dt = float(getattr(env.cfg, "ctrl_dt", 0.02))
    velocity_samples: list[np.ndarray] = []
    gyro_samples: list[np.ndarray] = []
    velocity_trace_samples: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    yaw_rate_delta_samples: list[float] = []
    vz_delta_samples: list[float] = []
    previous_eval_gyro: np.ndarray | None = None
    previous_eval_vz: float | None = None
    orientation_samples: list[float] = []
    height_samples: list[float] = []
    base_position_samples: list[np.ndarray] = []
    yaw_samples: list[float] = []
    trajectory_segment_ids: list[int] = []
    trajectory_segment_id = 0
    reward_term_samples: dict[str, list[float]] = {}
    # Keep rear-compatible fields while also exposing all four feet so a
    # rear-leg regularizer cannot hide a compensating front-leg excursion.
    swing_abs_y_samples = {foot: [] for foot in _FOOT_NAMES}
    swing_lateral_speed_samples = {foot: [] for foot in _FOOT_NAMES}
    previous_eval_feet_pos = np.asarray(env.feet_pos[cfg.analysis_env_index], dtype=np.float32).copy()
    fall_count = 0

    logger.info(
        "Running straight walk for %.2fs with command [%.3f, %.3f, %.3f]",
        cfg.duration_seconds,
        *command,
    )
    try:
        while (
            step_index < cfg.eval_steps
            if cfg.eval_steps is not None
            else time.monotonic() - started < cfg.duration_seconds
        ):
            frame_started = time.monotonic()
            previous_step_feet_pos = previous_eval_feet_pos.copy()
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            current_contact = np.asarray(env.feet_contact[cfg.analysis_env_index], dtype=bool).copy()
            done_array = torch.as_tensor(dones).detach().cpu().numpy().reshape(-1)
            done = bool(done_array[cfg.analysis_env_index])
            current_step_feet_pos = np.asarray(env.feet_pos[cfg.analysis_env_index], dtype=np.float32).copy()
            elapsed = step_index * ctrl_dt
            if elapsed >= cfg.warmup_seconds and not done:
                current_gyro = np.asarray(env.get_gyro()[cfg.analysis_env_index], dtype=np.float32).copy()
                current_velocity = np.asarray(
                    env.get_local_linvel()[cfg.analysis_env_index], dtype=np.float32
                ).copy()
                velocity_samples.append(current_velocity)
                gyro_samples.append(current_gyro)
                velocity_trace_samples.append(
                    (float((step_index + 1) * ctrl_dt), current_velocity.copy(), current_gyro.copy(), command.copy())
                )
                if previous_eval_gyro is not None:
                    yaw_rate_delta_samples.append(float((current_gyro[2] - previous_eval_gyro[2]) / ctrl_dt))
                previous_eval_gyro = current_gyro
                current_vz = float(current_velocity[2])
                if previous_eval_vz is not None:
                    vz_delta_samples.append((current_vz - previous_eval_vz) / ctrl_dt)
                previous_eval_vz = current_vz
                gravity = np.asarray(env.get_gravity()[cfg.analysis_env_index], dtype=np.float32)
                orientation_samples.append(float(np.linalg.norm(gravity[:2])))
                height_samples.append(float(env.sim_data["base_pos"][cfg.analysis_env_index, 2]))
                base_position_samples.append(
                    np.asarray(env.sim_data["base_pos"][cfg.analysis_env_index, :3], dtype=np.float64).copy()
                )
                root_quat = np.asarray(env.sim_data["root_quat"][cfg.analysis_env_index], dtype=np.float64)
                if root_quat.shape[-1] != 4:
                    raise ValueError(f"Expected root_quat shape (..., 4), got {root_quat.shape}")
                qx, qy, qz, qw = root_quat[:4]
                yaw_samples.append(float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))))
                trajectory_segment_ids.append(trajectory_segment_id)
                for name, values in getattr(env._state, "reward_terms", {}).items():
                    reward_term_samples.setdefault(name, []).append(float(values[cfg.analysis_env_index]))
                current_feet_pos = current_step_feet_pos
                swing_mask = np.asarray(env._feet_phase[cfg.analysis_env_index] >= 0.6, dtype=bool)
                swing_abs_y = np.abs(current_feet_pos[:, 1])
                swing_speed_y = np.abs((current_feet_pos[:, 1] - previous_step_feet_pos[:, 1]) / ctrl_dt)
                for foot_index, foot in enumerate(_FOOT_NAMES):
                    if swing_mask[foot_index]:
                        swing_abs_y_samples[foot].append(float(swing_abs_y[foot_index]))
                        swing_lateral_speed_samples[foot].append(float(swing_speed_y[foot_index]))
            else:
                # Do not measure an angular-rate jump across warmup, skipped
                # samples, or an auto-reset boundary.
                previous_eval_gyro = None
                previous_eval_vz = None
                if done:
                    trajectory_segment_id += 1
            # Advance every control step, including warmup and reset steps, so
            # the first measured lateral speed is a true one-step difference.
            previous_eval_feet_pos = current_step_feet_pos
            fall_count += int(done)
            if not done and contact_initialized:
                landed = current_contact & ~previous_contact
                foot_positions = np.asarray(env.feet_pos[cfg.analysis_env_index], dtype=np.float32)
                for foot_index, foot in enumerate(_FOOT_NAMES):
                    if landed[foot_index]:
                        x, y = foot_positions[foot_index, :2]
                        landings[foot].append((step_index, step_index * ctrl_dt, float(x), float(y)))
            previous_contact = current_contact
            contact_initialized = not done
            step_index += 1
            # Re-pin in case a task resamples commands during stepping.
            _set_straight_command(env, command)
            if vec_env.render() is False:
                break

            if not recording and not cfg.no_render:
                elapsed = time.monotonic() - frame_started
                if elapsed < 1.0 / fps:
                    time.sleep(1.0 / fps - elapsed)
    finally:
        vec_env.close()
        csv_path, svg_path = _save_foot_analysis(analysis_dir, landings)
        bias_csv_path, bias_png_path = _write_left_right_bias_analysis(analysis_dir, landings)
        logger.info("Saved foot landing positions: %s", csv_path)
        logger.info("Saved planar foot landing plot: %s", svg_path)
        logger.info("Saved left/right bias metrics: %s", bias_csv_path)
        logger.info("Saved left/right bias plot: %s", bias_png_path)
        velocities = np.asarray(velocity_samples, dtype=np.float64).reshape(-1, 3)
        gyros = np.asarray(gyro_samples, dtype=np.float64).reshape(-1, 3)
        if velocities.size:
            metrics = _compute_motion_metrics(
                velocities,
                gyros,
                command,
                yaw_rate_deltas=np.asarray(yaw_rate_delta_samples, dtype=np.float64),
                vz_deltas=np.asarray(vz_delta_samples, dtype=np.float64),
                orientation_samples=orientation_samples,
                height_samples=height_samples,
                termination_count=fall_count,
                total_steps=step_index,
                ctrl_dt=ctrl_dt,
            )
            metrics.update(
                {
                    **_compute_trajectory_metrics(
                        np.asarray(base_position_samples, dtype=np.float64),
                        np.asarray(yaw_samples, dtype=np.float64),
                        command,
                        np.asarray(trajectory_segment_ids, dtype=np.int64),
                    ),
                    "reward_terms": {
                        name: float(np.mean(values)) for name, values in reward_term_samples.items()
                    },
                    "rear_swing_max_abs_y": {
                        foot: float(max(swing_abs_y_samples[foot])) if swing_abs_y_samples[foot] else 0.0
                        for foot in ("RL", "RR")
                    },
                    "rear_swing_p95_abs_y": {
                        foot: float(np.percentile(swing_abs_y_samples[foot], 95))
                        if swing_abs_y_samples[foot]
                        else 0.0
                        for foot in ("RL", "RR")
                    },
                    "rear_swing_peak_lateral_speed": {
                        foot: float(max(swing_lateral_speed_samples[foot]))
                        if swing_lateral_speed_samples[foot]
                        else 0.0
                        for foot in ("RL", "RR")
                    },
                    "front_swing_max_abs_y": {
                        foot: float(max(swing_abs_y_samples[foot])) if swing_abs_y_samples[foot] else 0.0
                        for foot in ("FL", "FR")
                    },
                    "front_swing_p95_abs_y": {
                        foot: float(np.percentile(swing_abs_y_samples[foot], 95))
                        if swing_abs_y_samples[foot]
                        else 0.0
                        for foot in ("FL", "FR")
                    },
                    "front_swing_peak_lateral_speed": {
                        foot: float(max(swing_lateral_speed_samples[foot]))
                        if swing_lateral_speed_samples[foot]
                        else 0.0
                        for foot in ("FL", "FR")
                    },
                }
            )
        else:
            metrics = {
                "metrics_schema_version": 2,
                "command": command.tolist(),
                "command_axes": ["vx", "vy", "yaw_rate"],
                "samples": 0,
                "fall_rate": float(fall_count / max(step_index, 1)),
                "termination_count": int(fall_count),
            }
        metrics_path = Path(cfg.metrics_path) if cfg.metrics_path else analysis_dir / "velocity_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path = metrics_path.parent / "velocity_trace.csv"
        _write_velocity_trace_csv(trace_path, velocity_trace_samples)
        metrics["velocity_trace_csv"] = str(trace_path)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved velocity metrics: %s", metrics_path)
        logger.info(
            "Velocity summary: target_xy=%.3f m/s, actual_xy=%.3f m/s, "
            "vx_mae=%.3f, vy_mae=%.3f, yaw_mae=%.3f rad/s",
            metrics.get("command_speed_xy", 0.0),
            metrics.get("actual_speed_xy_mean", 0.0),
            metrics.get("tracking_mae", [0.0, 0.0, 0.0])[0],
            metrics.get("tracking_mae", [0.0, 0.0, 0.0])[1],
            metrics.get("tracking_mae", [0.0, 0.0, 0.0])[2],
        )


@hydra.main(version_base=None, config_path="../configs", config_name="play_walk")
def main(cfg: DictConfig) -> None:
    try:
        _run_walk(to_typed_config(cfg, WalkPlayConfig))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        logger.error("Error: %s", exc)
        logger.error("Please provide an RSLRL PPO policy from a metadata-backed run")


if __name__ == "__main__":
    main()
