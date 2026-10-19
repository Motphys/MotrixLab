# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Fixed G1 29-DOF standing (FixStand) servo gains.

FixStand is a posture, not a policy: it must not follow the exported policy
artifact, otherwise every new policy silently changes how the robot stands up.
The persistent program, the skill preparation and the single-artifact hardware
startup all resolve their pull-in gains here, so there is exactly one definition
to adjust by hand.

Values are keyed by joint name so any canonical joint order resolves correctly.
They are the field-validated stand-up gains: stiffer than the policy gains so the
legs carry the body, with waist gains that hold the torso and light arm damping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

G1_29DOF_STANDING_KP: Mapping[str, float] = {
    "left_hip_pitch_joint": 350.0,
    "left_hip_roll_joint": 200.0,
    "left_hip_yaw_joint": 200.0,
    "left_knee_joint": 300.0,
    "left_ankle_pitch_joint": 300.0,
    "left_ankle_roll_joint": 150.0,
    "right_hip_pitch_joint": 350.0,
    "right_hip_roll_joint": 200.0,
    "right_hip_yaw_joint": 200.0,
    "right_knee_joint": 300.0,
    "right_ankle_pitch_joint": 300.0,
    "right_ankle_roll_joint": 150.0,
    "waist_yaw_joint": 200.0,
    "waist_roll_joint": 200.0,
    "waist_pitch_joint": 200.0,
    "left_shoulder_pitch_joint": 40.0,
    "left_shoulder_roll_joint": 40.0,
    "left_shoulder_yaw_joint": 40.0,
    "left_elbow_joint": 40.0,
    "left_wrist_roll_joint": 40.0,
    "left_wrist_pitch_joint": 40.0,
    "left_wrist_yaw_joint": 40.0,
    "right_shoulder_pitch_joint": 40.0,
    "right_shoulder_roll_joint": 40.0,
    "right_shoulder_yaw_joint": 40.0,
    "right_elbow_joint": 40.0,
    "right_wrist_roll_joint": 40.0,
    "right_wrist_pitch_joint": 40.0,
    "right_wrist_yaw_joint": 40.0,
}

G1_29DOF_STANDING_KD: Mapping[str, float] = {
    "left_hip_pitch_joint": 5.0,
    "left_hip_roll_joint": 5.0,
    "left_hip_yaw_joint": 5.0,
    "left_knee_joint": 10.0,
    "left_ankle_pitch_joint": 5.0,
    "left_ankle_roll_joint": 5.0,
    "right_hip_pitch_joint": 5.0,
    "right_hip_roll_joint": 5.0,
    "right_hip_yaw_joint": 5.0,
    "right_knee_joint": 10.0,
    "right_ankle_pitch_joint": 5.0,
    "right_ankle_roll_joint": 5.0,
    "waist_yaw_joint": 5.0,
    "waist_roll_joint": 5.0,
    "waist_pitch_joint": 5.0,
    "left_shoulder_pitch_joint": 3.0,
    "left_shoulder_roll_joint": 3.0,
    "left_shoulder_yaw_joint": 3.0,
    "left_elbow_joint": 3.0,
    "left_wrist_roll_joint": 3.0,
    "left_wrist_pitch_joint": 3.0,
    "left_wrist_yaw_joint": 3.0,
    "right_shoulder_pitch_joint": 3.0,
    "right_shoulder_roll_joint": 3.0,
    "right_shoulder_yaw_joint": 3.0,
    "right_elbow_joint": 3.0,
    "right_wrist_roll_joint": 3.0,
    "right_wrist_pitch_joint": 3.0,
    "right_wrist_yaw_joint": 3.0,
}


def resolve_g1_standing_gains(joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed standing gains in the requested canonical joint order."""
    names = tuple(joint_names)
    known = set(G1_29DOF_STANDING_KP)
    if set(names) != known:
        raise ValueError(
            "G1 standing gains cover exactly the 29 G1 DOF joints; "
            f"missing={sorted(known - set(names))}, unknown={sorted(set(names) - known)}"
        )
    kp = np.asarray([G1_29DOF_STANDING_KP[name] for name in names], dtype=np.float32)
    kd = np.asarray([G1_29DOF_STANDING_KD[name] for name in names], dtype=np.float32)
    return kp, kd


__all__ = ["G1_29DOF_STANDING_KD", "G1_29DOF_STANDING_KP", "resolve_g1_standing_gains"]
