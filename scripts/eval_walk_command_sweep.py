# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Run reproducible single-axis walk-command sweeps and summarize stability."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_VX = "-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1"
DEFAULT_VY = "-0.5,-0.25,0,0.25,0.5"
DEFAULT_YAW = "-0.5,-0.25,0,0.25,0.5"


def _parse_values(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("velocity value list must not be empty")
    return sorted(set(values))


def _value_tag(value: float) -> str:
    prefix = "m" if value < 0.0 else "p"
    return f"{prefix}{abs(round(value * 100)):03d}"


def _command(axis: str, value: float) -> tuple[float, float, float]:
    result = [0.0, 0.0, 0.0]
    result[{"vx": 0, "vy": 1, "yaw": 2}[axis]] = value
    return tuple(result)


def _classify(metrics: dict[str, Any], axis: str, value: float, args: argparse.Namespace) -> dict[str, Any]:
    index = {"vx": 0, "vy": 1, "yaw": 2}[axis]
    errors = metrics.get("steady_state_error", [float("inf")] * 3)
    rmses = metrics.get("rmse", [float("inf")] * 3)
    p95 = metrics.get("p95_error", [float("inf")] * 3)
    reasons: list[str] = []

    if metrics.get("termination_count", 0) != 0:
        reasons.append("termination")
    if value == 0.0:
        if max(abs(float(error)) for error in errors) > args.zero_error:
            reasons.append("zero_mean_error")
        if max(float(rmse) for rmse in rmses) > args.zero_rmse:
            reasons.append("zero_rmse")
    else:
        if abs(float(errors[index])) > args.tracking_mean_error:
            reasons.append("target_mean_error")
        if float(rmses[index]) > args.tracking_rmse:
            reasons.append("target_rmse")
        if float(p95[index]) > args.tracking_p95:
            reasons.append("target_p95")
    tracking_pass = not reasons

    quality_reasons = list(reasons)
    if float(metrics.get("orthogonal_drift", float("inf"))) > args.orthogonal_drift:
        quality_reasons.append("orthogonal_drift")
    if float(metrics.get("vz_rms", float("inf"))) > args.vz_rms:
        quality_reasons.append("vz_rms")
    if float(metrics.get("orientation_rms", float("inf"))) > args.orientation_rms:
        quality_reasons.append("orientation_rms")
    if float(metrics.get("base_height_std", float("inf"))) > args.base_height_std:
        quality_reasons.append("base_height_std")
    if axis == "vx" and value != 0.0:
        if float(metrics.get("cross_track_error_per_meter", float("inf"))) > args.cross_track_per_meter:
            quality_reasons.append("cross_track_per_meter")
        if float(metrics.get("mean_abs_segment_final_yaw_drift_rad", float("inf"))) > args.yaw_drift:
            quality_reasons.append("yaw_drift")

    # At a true zero command there may be no swing samples; zero-valued swing
    # summaries must not be interpreted as feet placed at the body centre.
    if value != 0.0:
        for group in ("front", "rear"):
            widths = metrics.get(f"{group}_swing_p95_abs_y", {})
            peaks = metrics.get(f"{group}_swing_peak_lateral_speed", {})
            if any(not args.foot_width_min <= float(item) <= args.foot_width_max for item in widths.values()):
                quality_reasons.append(f"{group}_swing_width")
            if any(float(item) > args.foot_lateral_speed for item in peaks.values()):
                quality_reasons.append(f"{group}_swing_speed")

    return {
        "tracking_pass": tracking_pass,
        "quality_pass": not quality_reasons,
        "tracking_failure_reasons": reasons,
        "quality_failure_reasons": sorted(set(quality_reasons)),
    }


def _row(metrics: dict[str, Any], axis: str, value: float, seed: int, result_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    index = {"vx": 0, "vy": 1, "yaw": 2}[axis]
    return {
        "axis": axis,
        "command": value,
        "seed": seed,
        "mean_target": metrics.get("mean_velocity", [None] * 3)[index],
        "steady_state_error": metrics.get("steady_state_error", [None] * 3)[index],
        "target_rmse": metrics.get("rmse", [None] * 3)[index],
        "target_p95_error": metrics.get("p95_error", [None] * 3)[index],
        "mean_vx": metrics.get("mean_velocity", [None] * 3)[0],
        "mean_vy": metrics.get("mean_velocity", [None] * 3)[1],
        "mean_yaw_rate": metrics.get("mean_velocity", [None] * 3)[2],
        "orthogonal_drift": metrics.get("orthogonal_drift"),
        "vz_rms": metrics.get("vz_rms"),
        "vz_delta_rms": metrics.get("vz_delta_rms"),
        "yaw_rate_rmse": metrics.get("yaw_rate_rmse"),
        "roll_pitch_rate_rms": metrics.get("roll_pitch_rate_rms"),
        "orientation_rms": metrics.get("orientation_rms"),
        "base_height_std": metrics.get("base_height_std"),
        "termination_count": metrics.get("termination_count"),
        "cross_track_error_per_meter": metrics.get("cross_track_error_per_meter"),
        "mean_abs_segment_final_cross_track_error_m": metrics.get(
            "mean_abs_segment_final_cross_track_error_m"
        ),
        "mean_abs_segment_final_yaw_drift_rad": metrics.get("mean_abs_segment_final_yaw_drift_rad"),
        "tracking_pass": result["tracking_pass"],
        "quality_pass": result["quality_pass"],
        "tracking_failure_reasons": ";".join(result["tracking_failure_reasons"]),
        "quality_failure_reasons": ";".join(result["quality_failure_reasons"]),
        "result_dir": str(result_dir),
    }


def _sampled_interval(rows: list[dict[str, Any]], pass_key: str) -> dict[str, Any]:
    by_value: dict[float, list[bool]] = defaultdict(list)
    for row in rows:
        by_value[float(row["command"])].append(bool(row[pass_key]))
    passed = {value: all(results) for value, results in by_value.items()}
    if not passed.get(0.0, False):
        return {"min": None, "max": None, "sampled_pass_values": []}
    low = high = 0.0
    for value in sorted((value for value in passed if value < 0.0), reverse=True):
        if not passed[value]:
            break
        low = value
    for value in sorted(value for value in passed if value > 0.0):
        if not passed[value]:
            break
        high = value
    return {
        "min": low,
        "max": high,
        "sampled_pass_values": sorted(value for value, ok in passed.items() if ok),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--env", default="go2-walk-flat")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--axes", nargs="+", choices=("vx", "vy", "yaw"), default=["vx", "vy", "yaw"])
    parser.add_argument("--vx-values", type=_parse_values, default=_parse_values(DEFAULT_VX))
    parser.add_argument("--vy-values", type=_parse_values, default=_parse_values(DEFAULT_VY))
    parser.add_argument("--yaw-values", type=_parse_values, default=_parse_values(DEFAULT_YAW))
    parser.add_argument("--seeds", type=_parse_values, default=[1.0])
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--eval-steps", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tracking-mean-error", type=float, default=0.10)
    parser.add_argument("--tracking-rmse", type=float, default=0.10)
    parser.add_argument("--tracking-p95", type=float, default=0.20)
    parser.add_argument("--zero-error", type=float, default=0.05)
    parser.add_argument("--zero-rmse", type=float, default=0.05)
    parser.add_argument("--orthogonal-drift", type=float, default=0.05)
    parser.add_argument("--vz-rms", type=float, default=0.10)
    parser.add_argument("--orientation-rms", type=float, default=0.03)
    parser.add_argument("--base-height-std", type=float, default=0.005)
    parser.add_argument("--cross-track-per-meter", type=float, default=0.01)
    parser.add_argument("--yaw-drift", type=float, default=0.05)
    parser.add_argument("--foot-width-min", type=float, default=0.12)
    parser.add_argument("--foot-width-max", type=float, default=0.18)
    parser.add_argument("--foot-lateral-speed", type=float, default=0.80)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    policy = args.policy.expanduser().resolve()
    if not policy.is_file():
        parser.error(f"policy does not exist: {policy}")
    output = (args.output_dir or policy.parent / "command_sweep").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds]
    if any(float(seed) != int(seed) for seed in args.seeds):
        parser.error("seeds must be integers")
    values_by_axis = {"vx": args.vx_values, "vy": args.vy_values, "yaw": args.yaw_values}
    rows: list[dict[str, Any]] = []
    raw_results: list[dict[str, Any]] = []

    for axis in args.axes:
        for value in values_by_axis[axis]:
            command = _command(axis, value)
            for seed in seeds:
                result_dir = output / f"seed_{seed}" / f"{axis}_{_value_tag(value)}"
                metrics_path = result_dir / "velocity_metrics.json"
                command_line = [
                    sys.executable,
                    str(repo / "scripts" / "play_walk.py"),
                    f"env={args.env}",
                    f"policy={policy}",
                    "num_envs=1",
                    f"seed={seed}",
                    f"forward_velocity={command[0]}",
                    f"lateral_velocity={command[1]}",
                    f"yaw_velocity={command[2]}",
                    f"warmup_seconds={args.warmup_seconds}",
                    f"eval_steps={args.eval_steps}",
                    f"analysis_dir={result_dir}",
                    "no_render=true",
                ]
                if args.dry_run:
                    print(" ".join(map(str, command_line)))
                    continue
                if args.overwrite or not metrics_path.is_file():
                    result_dir.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env.setdefault("NUMBA_DISABLE_JIT", "1")
                    subprocess.run(command_line, cwd=repo, env=env, check=True)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                expected = list(command)
                if any(abs(float(actual) - target) > 1e-6 for actual, target in zip(metrics["command"], expected)):
                    raise ValueError(f"Command mismatch in {metrics_path}: {metrics['command']} != {expected}")
                classification = _classify(metrics, axis, value, args)
                rows.append(_row(metrics, axis, value, seed, result_dir, classification))
                raw_results.append(
                    {
                        "axis": axis,
                        "command": value,
                        "seed": seed,
                        **classification,
                        "metrics_path": str(metrics_path),
                    }
                )

    if args.dry_run:
        return
    intervals: dict[str, Any] = {}
    for axis in args.axes:
        axis_rows = [row for row in rows if row["axis"] == axis]
        intervals[axis] = {
            "tracking": _sampled_interval(axis_rows, "tracking_pass"),
            "quality": _sampled_interval(axis_rows, "quality_pass"),
        }
    thresholds = {
        name: value
        for name, value in vars(args).items()
        if name
        in {
            "tracking_mean_error", "tracking_rmse", "tracking_p95", "zero_error", "zero_rmse",
            "orthogonal_drift", "vz_rms", "orientation_rms", "base_height_std",
            "cross_track_per_meter", "yaw_drift", "foot_width_min", "foot_width_max",
            "foot_lateral_speed",
        }
    }
    summary = {
        "policy": str(policy),
        "env": args.env,
        "eval_steps": args.eval_steps,
        "warmup_seconds": args.warmup_seconds,
        "seeds": seeds,
        "thresholds": thresholds,
        "sampled_stable_intervals": intervals,
        "results": raw_results,
    }
    _write_csv(output / "command_sweep.csv", rows)
    (output / "command_sweep.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(intervals, indent=2))
    print(f"Saved {output / 'command_sweep.csv'}")
    print(f"Saved {output / 'command_sweep.json'}")


if __name__ == "__main__":
    main()
