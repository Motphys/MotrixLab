# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT = SCRIPTS_DIR / "play_walk.py"
SPEC = importlib.util.spec_from_file_location("motrix_test_play_walk_script", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load {SCRIPT}")
play_walk = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = play_walk
SPEC.loader.exec_module(play_walk)


def test_motion_metrics_use_yaw_rate_as_the_third_command_axis():
    velocities = np.array([[0.5, 0.1, 3.0], [0.5, -0.1, 5.0]], dtype=np.float64)
    gyros = np.array([[0.2, 0.0, 0.3], [0.0, 0.2, 0.5]], dtype=np.float64)

    metrics = play_walk._compute_motion_metrics(
        velocities,
        gyros,
        np.array([0.5, 0.0, 0.4], dtype=np.float64),
        yaw_rate_deltas=np.array([2.0], dtype=np.float64),
        orientation_samples=[0.1, 0.2],
        height_samples=[0.3, 0.4],
        termination_count=1,
        total_steps=100,
        ctrl_dt=0.02,
    )

    assert metrics["metrics_schema_version"] == 2
    np.testing.assert_allclose(metrics["mean_velocity"], [0.5, 0.0, 0.4])
    np.testing.assert_allclose(metrics["mean_linear_velocity"], [0.5, 0.0, 4.0])
    assert metrics["yaw_rate_rmse"] == 0.1
    assert metrics["yaw_rate_delta_rms"] == 2.0
    assert metrics["orthogonal_drift"] == 0.1
    assert metrics["termination_count"] == 1
    assert metrics["termination_events_per_minute"] == 30.0
