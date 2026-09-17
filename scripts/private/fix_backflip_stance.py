# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Fix foot penetration in the G1 backflip motion.

The mjbatch retarget sits ~1 cm too low: during stance frames the lower foot
sole (0.035 m below the ankle_roll_link origin in the training model) is
below ground, so the contact solver ejects the robot at reset and the rollout
diverges deterministically. This script lifts the whole clip per-frame by the
penetration depth (smoothed, flight frames untouched), recomputes the z
component of body linear velocities, and rewrites the npz in place.

Run:
    python scripts/private/fix_backflip_stance.py [--clearance 0.002]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from absl import app, flags

_MOTION = flags.DEFINE_string("motion", None, "Path to the G1 backflip npz.")
_CLEARANCE = flags.DEFINE_float("clearance", 0.002, "Target sole clearance in stance, meters.")
_SMOOTH = flags.DEFINE_integer("smooth", 5, "Smoothing window (frames) for the lift profile.")

_SOLE_OFFSET = np.array([0.0, 0.0, -0.035])  # lowest sole point in ankle_roll frame
_ANKLE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return R @ v


def main(argv):
    del argv
    path = Path(_MOTION.value).expanduser()
    data = dict(np.load(path, allow_pickle=False))
    body_names = [str(n) for n in data["body_names"].tolist()]
    num_frames = data["joint_pos"].shape[0]
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    soles = []
    for name in _ANKLE_BODIES:
        i = body_names.index(name)
        pos, quat = data["body_pos_w"][:, i, :], data["body_quat_w"][:, i, :]
        soles.append(np.array([pos[t, 2] + _quat_rot(quat[t], _SOLE_OFFSET)[2] for t in range(num_frames)]))
    min_sole = np.minimum(soles[0], soles[1])

    lift = np.clip(_CLEARANCE.value - min_sole, 0.0, None)
    kernel = np.ones(_SMOOTH.value) / _SMOOTH.value
    lift = np.convolve(lift, kernel, mode="same")
    print(
        f"sole z range [{min_sole.min():+.4f}, {min_sole.max():+.4f}] m; "
        f"lift range [{lift.min():.4f}, {lift.max():.4f}] m over {int((lift > 1e-4).sum())} frames"
    )

    data["body_pos_w"] = data["body_pos_w"].copy()
    data["body_pos_w"][:, :, 2] += lift[:, None]
    dt = 1.0 / fps
    data["body_lin_vel_w"] = data["body_lin_vel_w"].copy()
    data["body_lin_vel_w"][:, :, 2] = np.gradient(data["body_pos_w"][:, :, 2], dt, axis=0)

    np.savez(path, **data)
    print(f"rewrote {path}")


if __name__ == "__main__":
    app.run(main)
