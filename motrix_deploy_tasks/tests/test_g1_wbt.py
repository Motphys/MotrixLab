# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Focused, simulator-free tests of the current G1 WBT actor contract."""

import numpy as np
import pytest

from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtMotion, G1WbtPolicyProcessor, G1WbtTaskSpec


def _state(default):
    return RobotState(
        0,
        0,
        default,
        np.zeros(29, np.float32),
        np.array([0, 0, 0, 1], np.float32),
        np.array([1, 2, 3], np.float32),
        np.zeros(3, np.float32),
    )


def _motion(joint_pos=None, joint_vel=None, reference=None):
    return G1WbtMotion(
        joint_pos if joint_pos is not None else np.arange(58, dtype=np.float32).reshape(2, 29),
        joint_vel if joint_vel is not None else np.arange(58, dtype=np.float32).reshape(2, 29) + 100,
        np.asarray(reference if reference is not None else [[0, 0, 0, 1], [0, 0, 1, 0]], np.float32),
    )


def _spec(scale):
    return G1WbtTaskSpec(
        action_scale=[scale] * 29 if np.isscalar(scale) else scale.tolist(),
        kp=[10] * 29,
        kd=[1] * 29,
        motion_frames=2,
        termination_ref_orientation_threshold=0.4,
        termination_joint_position_threshold=0.5,
        termination_joint_velocity_threshold=10,
    )


def _io(scale=0.25, spec=None, motion=None):
    default = np.linspace(-0.4, 0.7, 29, dtype=np.float32)
    robot = RobotSpec(
        "pelvis",
        tuple(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", *[f"joint_{i}" for i in range(26)]]),
        default,
        np.full(29, -1000, np.float32),  # Servo limits far away: these tests assert unclamped math.
        np.full(29, 1000, np.float32),
        np.full(29, 10, np.float32),
    )
    motion = motion if motion is not None else _motion()
    return G1WbtPolicyProcessor(spec if spec is not None else _spec(scale), robot, motion), default


def _observation(io, default, frame=0, previous_action=None):
    return io.actor_observation(
        frame,
        np.array([0, 0, 0, 1]),
        np.array([1, -2, 3]),
        default,
        np.zeros(29),
        previous_action,
    )


def test_observation_order_frame_selection_and_unscaled_terms():
    spec = _spec(0.25)
    io, default = _io(spec=spec)
    motion = io.motion
    positions = default + np.arange(29, dtype=np.float32) / 10
    velocities = np.linspace(-20, 30, 29)
    previous = np.linspace(-5, 4, 29)
    angular_velocity = np.array([7, -8, 9])
    observation = io.actor_observation(1, np.array([0, 0, 0, 1]), angular_velocity, positions, velocities, previous)
    # 154 and these boundaries are the deployed actor contract, not tuning.
    assert observation.shape == (154,)
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation[:29], motion.joint_pos[1])
    np.testing.assert_array_equal(observation[29:58], motion.joint_vel[1])
    # Frame-1 reference [0,0,1,0] is a 180° Z rotation: relative = diag(-1,-1,1).
    np.testing.assert_array_equal(observation[58:64], [-1, 0, 0, 0, -1, 0])
    np.testing.assert_array_equal(observation[64:67], angular_velocity)
    np.testing.assert_allclose(observation[67:96], positions - default)
    np.testing.assert_allclose(observation[96:125], velocities)
    np.testing.assert_allclose(observation[125:], previous)
    np.testing.assert_array_equal(io.previous_action, np.zeros(29))


def test_noncommuting_relative_orientation_first_two_rows_xyzw():
    torso_angle, reference_angle = 0.7, -1.1
    reference = [0, np.sin(reference_angle / 2), 0, np.cos(reference_angle / 2)]
    io, default = _io(spec=_spec(0.25), motion=_motion(reference=[[0, 0, 0, 1], reference]))
    torso = np.array([np.sin(torso_angle / 2), 0, 0, np.cos(torso_angle / 2)])
    ct, st = np.cos(torso_angle), np.sin(torso_angle)
    cr, sr = np.cos(reference_angle), np.sin(reference_angle)
    torso_matrix = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    reference_matrix = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
    relative = torso_matrix.T @ reference_matrix
    actual = io.actor_observation(1, torso, np.zeros(3), default, np.zeros(29))[58:64]
    np.testing.assert_allclose(actual, relative[:2].reshape(-1), atol=1e-7)
    assert not np.allclose(actual, relative[:, :2].reshape(-1))
    assert not np.allclose(actual, (reference_matrix @ torso_matrix.T)[:2].reshape(-1))
    assert not np.allclose(actual, relative.T[:2].reshape(-1))
    # Quaternion sign and small normalization differences do not change rotation.
    negated = io.actor_observation(1, -torso * 1.0001, np.zeros(3), default, np.zeros(29))[58:64]
    np.testing.assert_allclose(negated, actual, atol=1e-7)


@pytest.mark.parametrize("scale", [0.31, np.linspace(0, 0.8, 29, dtype=np.float32)])
def test_action_offset_scale_raw_history_without_clipping_and_reset(scale):
    io, default = _io(scale)
    np.testing.assert_array_equal(_observation(io, default)[125:], np.zeros(29))
    raw = np.linspace(-9, 8, 29, dtype=np.float32)
    expected_raw = raw.copy()
    target = io.process_action(raw).joint_position
    assert target.dtype == np.float32
    np.testing.assert_allclose(target, default + scale * expected_raw)
    # History is raw, not scaled target, not clamped action, and not caller-owned.
    raw.fill(99)
    target.fill(99)
    np.testing.assert_array_equal(io.previous_action, expected_raw)
    np.testing.assert_array_equal(_observation(io, default)[125:], expected_raw)
    io.previous_action.fill(99)
    np.testing.assert_array_equal(io.previous_action, expected_raw)
    override = np.full(29, -3, dtype=np.float32)
    np.testing.assert_array_equal(_observation(io, default, previous_action=override)[125:], override)
    np.testing.assert_array_equal(io.previous_action, expected_raw)
    next_raw = np.full(29, 2, dtype=np.float32)
    io.process_action(next_raw)
    np.testing.assert_array_equal(_observation(io, default)[125:], next_raw)
    io.reset(
        _state(default),
        ControlContext(0, 0.0, None, dt_s=0.02),
    )
    np.testing.assert_array_equal(_observation(io, default)[125:], np.zeros(29))
