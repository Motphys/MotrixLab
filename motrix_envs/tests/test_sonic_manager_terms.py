# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from motrix_envs.locomotion.sonic import mdp


def _identity_quaternions(shape: tuple[int, ...]) -> np.ndarray:
    quaternions = np.zeros((*shape, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    return quaternions


def test_sonic_config_defaults_match_upstream_base() -> None:
    motion = mdp.SonicMotionCommandCfg(motion_file="unused.npz")
    tracking = mdp.SonicTrackingRewardCfg(weight=2.0)

    assert motion.reward_point_body_names == (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    assert motion.reward_point_body_offsets == (
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    assert mdp.SonicJointPositionActionCfg().simulate_action_latency is False
    assert motion.reference_body_name == "pelvis"
    assert tracking.std == 0.1


def test_sonic_action_scale_uses_policy_actuator_contract() -> None:
    scales = mdp._sonic_policy_action_scale()
    left_hip_pitch = mdp.G1_SONIC_JOINTS.index("left_hip_pitch_joint")
    right_hip_pitch = mdp.G1_SONIC_JOINTS.index("right_hip_pitch_joint")

    assert scales[left_hip_pitch] == pytest.approx(0.25 * 139.0 / 99.098427777)
    assert scales[right_hip_pitch] == pytest.approx(0.25 * 139.0 / 99.098427777)


def test_sonic_model_hip_pitch_dynamics_match_policy_scaling_contract() -> None:
    # Regression: runtime gain overrides left the generic G1's 88 Nm force
    # clamp and smaller armature in place despite scaling actions for 139 Nm.
    from motrix_envs.locomotion.sonic.g1 import make_g1_sonic_cfg

    root = ElementTree.parse(make_g1_sonic_cfg().scene.objs.robot.model.file).getroot()
    joint = root.find(".//default[@class='hip_pitch']/joint")
    for side in ("left", "right"):
        name = f"{side}_hip_pitch_joint"
        kp, kd, effort, armature = mdp._SONIC_ACTUATOR_PARAMETERS[name]
        actuator = root.find(f"actuator/position[@name='{name}']")
        assert float(actuator.attrib["kp"]) == pytest.approx(kp)
        assert float(actuator.attrib["kv"]) == pytest.approx(kd)
        assert tuple(map(float, actuator.attrib["forcerange"].split())) == (-effort, effort)
        assert tuple(map(float, joint.attrib["actuatorfrcrange"].split())) == (-effort, effort)
        assert float(joint.attrib["armature"]) == pytest.approx(armature)


def test_sonic_actuator_dynamics_reset_uses_release_gains() -> None:
    kp = np.empty(len(mdp.G1_SONIC_JOINTS), dtype=np.float32)
    damping = np.empty_like(kp)
    gains = tuple(np.float32(mdp._SONIC_ACTUATOR_PARAMETERS[name][0]) for name in mdp.G1_SONIC_JOINTS)
    dampings = tuple(np.float32(mdp._SONIC_ACTUATOR_PARAMETERS[name][1]) for name in mdp.G1_SONIC_JOINTS)
    mdp._reset_sonic_actuator_dynamics(
        SimpleNamespace(),
        {"kp": kp, "damping": damping},
        gains,
        dampings,
    )

    left_hip_pitch = mdp.G1_SONIC_JOINTS.index("left_hip_pitch_joint")
    assert kp[left_hip_pitch] == pytest.approx(99.098427777)
    assert damping[left_hip_pitch] == pytest.approx(6.308801854)


def test_sonic_reward_point_config_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        mdp.SonicMotionCommandCfg(
            motion_file="unused.npz",
            reward_point_body_names=("torso_link",),
            reward_point_body_offsets=(),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        mdp.SonicTrackingRewardCfg(weight=2.0, std=0.0)


def test_sonic_action_latency_and_pre_action_velocity_snapshot() -> None:
    shape = (2, len(mdp.G1_SONIC_JOINTS))
    current = np.full(shape, 2.0, dtype=np.float32)
    previous = np.full(shape, 3.0, dtype=np.float32)
    foot_indices = np.asarray((13, 14, 17, 18), dtype=np.int64)
    velocities = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    term = mdp.SonicJointPositionAction(
        current=current.copy(),
        previous=previous.copy(),
        default_angles=np.zeros(shape[1], dtype=np.float32),
        joint_lower=np.full(shape[1], -10.0, dtype=np.float32),
        joint_upper=np.full(shape[1], 10.0, dtype=np.float32),
        action_scales=np.ones(shape[1], dtype=np.float32),
        encoder_bias=np.zeros(shape, dtype=np.float32),
        processed=np.zeros(shape, dtype=np.float32),
        joint_velocity_before_action=np.zeros((shape[0], 4), dtype=np.float32),
        foot_joint_policy_indices=foot_indices,
        simulate_action_latency=False,
    )

    term.prepare({"robot_dof_vel": velocities})
    np.testing.assert_array_equal(term.joint_velocity_before_action, velocities[:, foot_indices])
    np.testing.assert_array_equal(term.process(np.full(shape, 4.0, dtype=np.float32)), 4.0)

    delayed = mdp.SonicJointPositionAction(
        current=current.copy(),
        previous=previous.copy(),
        default_angles=term.default_angles,
        joint_lower=term.joint_lower,
        joint_upper=term.joint_upper,
        action_scales=term.action_scales,
        encoder_bias=np.zeros(shape, dtype=np.float32),
        processed=np.zeros(shape, dtype=np.float32),
        joint_velocity_before_action=np.ones((shape[0], 4), dtype=np.float32),
        foot_joint_policy_indices=foot_indices,
        simulate_action_latency=True,
    )
    np.testing.assert_array_equal(delayed.process(np.full(shape, 4.0, dtype=np.float32)), 2.0)
    delayed.reset(np.asarray((0,), dtype=np.int64))
    np.testing.assert_array_equal(delayed.current[0], 0.0)
    np.testing.assert_array_equal(delayed.previous[0], 0.0)
    np.testing.assert_array_equal(delayed.joint_velocity_before_action[0], 0.0)


def test_sonic_actor_observation_uses_lane_local_action_width() -> None:
    joint_count = len(mdp.G1_SONIC_JOINTS)
    history_length = 2
    frame_dim = mdp.SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * mdp.SONIC_VECTOR_DIM
    term = mdp.SonicActorObservation(
        history=np.zeros((2, history_length, frame_dim), dtype=np.float32),
        seen_reset=np.full(2, -1, dtype=np.int64),
    )
    action = SimpleNamespace(
        current=np.arange(joint_count, dtype=np.float32),
        default_angles=np.zeros(joint_count, dtype=np.float32),
    )
    tracked_body_quat = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    context = SimpleNamespace(
        env_id=np.int64(1),
        actions={"joint_position": action},
        commands={"motion": SimpleNamespace(reset_counter=np.asarray((0,), dtype=np.int64))},
        sim={
            "tracked_body_quat": tracked_body_quat,
            "tracked_body_angular_velocity": np.zeros((len(mdp.G1_SONIC_BODY_NAMES), 3), dtype=np.float32),
            "robot_dof_pos": np.zeros(joint_count, dtype=np.float32),
            "robot_dof_vel": np.zeros(joint_count, dtype=np.float32),
        },
    )
    out = np.empty(history_length * frame_dim, dtype=np.float32)

    mdp.SonicActorObservation.compute(context, out, term)

    assert term.seen_reset.tolist() == [-1, 0]
    action_offset = history_length * (mdp.SONIC_VECTOR_DIM + 2 * joint_count)
    np.testing.assert_array_equal(out[action_offset : action_offset + joint_count], action.current)
    np.testing.assert_array_equal(out[action_offset + joint_count : action_offset + 2 * joint_count], action.current)

    previous_action = action.current.copy()
    action.current += 100.0
    mdp.SonicActorObservation.compute(context, out, term)
    np.testing.assert_array_equal(out[action_offset : action_offset + joint_count], previous_action)
    np.testing.assert_array_equal(out[action_offset + joint_count : action_offset + 2 * joint_count], action.current)


def test_sonic_g1_reference_matches_release_term_layout_and_pelvis_frame() -> None:
    joint_count = 2
    num_future_frames = 2
    frame_count = 6
    positions = np.arange(frame_count * joint_count, dtype=np.float32).reshape(frame_count, joint_count)
    velocities = positions + 100.0
    target_quaternions = _identity_quaternions((frame_count, len(mdp.G1_SONIC_BODY_NAMES)))
    quarter_turn_z = np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), dtype=np.float32)
    target_quaternions[:, mdp.SONIC_ROOT_BODY_INDEX] = quarter_turn_z
    tracked_quaternions = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    tracked_quaternions[mdp.SONIC_ROOT_BODY_INDEX] = quarter_turn_z
    # Keep torso different so the test detects accidentally using the old reference body.
    torso = mdp.G1_SONIC_BODY_NAMES.index("torso_link")
    tracked_quaternions[torso] = (0.0, 0.0, 0.0, 1.0)
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(num_future_frames),
        clip=SimpleNamespace(
            joint_pos=positions,
            frame_clip_end=np.full(frame_count, frame_count - 1, np.int64),
            joint_vel=velocities,
            tracked_bodies_quat_w=target_quaternions,
        ),
    )
    output = np.empty(num_future_frames * (2 * joint_count + 6), dtype=np.float32)

    mdp.SonicG1ReferenceObservation.compute(
        SimpleNamespace(commands={"motion": motion}, sim={"tracked_body_quat": tracked_quaternions}),
        output,
    )

    selected = np.asarray((0, 5), dtype=np.int64)
    identity_rotation6 = np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    expected = np.concatenate(
        (
            positions[selected].reshape(-1),
            velocities[selected].reshape(-1),
            np.tile(identity_rotation6, num_future_frames),
        )
    )
    np.testing.assert_allclose(output, expected, atol=1.0e-6)


def test_sonic_smpl_reference_matches_release_local_frame_and_wrist_tail() -> None:
    num_future_frames = 2
    smpl_joint_count = 24
    policy_joint_count = len(mdp.G1_SONIC_JOINTS)
    quarter_turn_z = np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), dtype=np.float32)
    tracked_quaternions = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    tracked_quaternions[mdp.SONIC_ROOT_BODY_INDEX] = quarter_turn_z
    human_quaternions = np.tile(quarter_turn_z, (num_future_frames, 1))
    smpl_joints = np.zeros((num_future_frames, smpl_joint_count, 3), dtype=np.float32)
    smpl_joints[..., 0] = 1.0
    joint_positions = np.arange(num_future_frames * policy_joint_count, dtype=np.float32).reshape(
        num_future_frames, policy_joint_count
    )
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(num_future_frames),
        clip=SimpleNamespace(
            joint_pos=joint_positions,
            frame_clip_end=np.full(num_future_frames, num_future_frames - 1, np.int64),
            smpl_joints=smpl_joints,
            smpl_root_quat=human_quaternions,
        ),
    )
    output = np.empty(num_future_frames * (smpl_joint_count * 3 + 6 + 6), dtype=np.float32)

    mdp.SonicSmplReferenceObservation.compute(
        SimpleNamespace(commands={"motion": motion}, sim={"tracked_body_quat": tracked_quaternions}),
        output,
    )

    local_joints = np.zeros_like(smpl_joints)
    local_joints[..., 1] = -1.0
    identity_rotation6 = np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    human_features = np.concatenate(
        (
            local_joints.reshape(num_future_frames, -1),
            np.tile(identity_rotation6, (num_future_frames, 1)),
        ),
        axis=-1,
    )
    wrists = joint_positions[:, mdp.SONIC_WRIST_POLICY_INDICES]
    expected = np.concatenate((human_features.reshape(-1), wrists.reshape(-1)))
    np.testing.assert_allclose(output, expected, atol=1.0e-6)


def test_sonic_observation_sizes_are_derived_from_runtime_shapes() -> None:
    num_future_frames = np.int64(2)
    joint_count = 7
    body_count = 5
    smpl_joint_count = 24
    motion = SimpleNamespace(
        num_future_frames=num_future_frames,
        clip=SimpleNamespace(
            joint_pos=np.empty((3, joint_count), dtype=np.float32),
            tracked_bodies_pos_w=np.empty((3, body_count, mdp.SONIC_VECTOR_DIM), dtype=np.float32),
            smpl_joints=np.empty((3, smpl_joint_count, mdp.SONIC_VECTOR_DIM), dtype=np.float32),
        ),
    )
    env = SimpleNamespace(
        num_envs=4,
        model=SimpleNamespace(
            bodies={"robot": SimpleNamespace(joint_names=tuple(f"joint_{i}" for i in range(joint_count)))}
        ),
        command_terms={"motion": motion},
    )

    actor_frame_dim = mdp.SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * mdp.SONIC_VECTOR_DIM
    assert mdp.SonicActorObservationCfg(history_length=3)(env).size == 3 * actor_frame_dim
    assert mdp.SonicG1ReferenceObservationCfg()(env).size == num_future_frames * (
        2 * joint_count + mdp.SONIC_ROTATION_REPRESENTATION_DIM
    )
    assert mdp.SonicSmplReferenceObservationCfg()(env).size == num_future_frames * (
        smpl_joint_count * mdp.SONIC_VECTOR_DIM
        + mdp.SONIC_ROTATION_REPRESENTATION_DIM
        + len(mdp.SONIC_WRIST_POLICY_INDICES)
    )
    current_frame_dim = 9 + body_count * 9
    future_frame_dim = 2 * joint_count + actor_frame_dim
    assert mdp.SonicCriticObservationCfg()(env).size == current_frame_dim + num_future_frames * future_frame_dim


def test_sonic_anti_shake_uses_three_dimensional_thresholded_norm() -> None:
    velocity = np.zeros((3, 3), dtype=np.float32)
    velocity[0] = (1.2, 0.9, 0.0)  # norm is exactly the threshold
    velocity[1] = (2.0, 0.0, 0.0)
    velocity[2] = (0.0, 0.0, 2.0)
    term = mdp.SonicAntiShakeReward(np.asarray((0, 1, 2), dtype=np.int64))

    actual = mdp.SonicAntiShakeReward.compute(
        SimpleNamespace(sim={"tracked_body_angular_velocity": velocity}),
        term,
    )

    assert actual == pytest.approx((0.0 + 0.25 + 0.25) / 3.0)


def test_sonic_anchor_orientation_threshold_uses_squared_angle_error() -> None:
    target = _identity_quaternions((1,))
    tracked = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        reference_index=np.int64(mdp.SONIC_ROOT_BODY_INDEX),
        clip=SimpleNamespace(root_body_quat_w=target),
    )
    context = SimpleNamespace(
        commands={"motion": motion},
        sim={"tracked_body_quat": tracked},
    )
    term = mdp.SonicAnchorOriTermination(np.float32(0.2))

    tracked[mdp.SONIC_ROOT_BODY_INDEX] = (np.sin(0.15), 0.0, 0.0, np.cos(0.15))
    assert not mdp.SonicAnchorOriTermination.compute(context, term)

    tracked[mdp.SONIC_ROOT_BODY_INDEX] = (np.sin(0.25), 0.0, 0.0, np.cos(0.25))
    assert mdp.SonicAnchorOriTermination.compute(context, term)


def test_sonic_feet_acc_matches_upstream_sum_of_squared_acceleration() -> None:
    foot_indices = np.asarray((13, 14, 17, 18), dtype=np.int64)
    velocity = np.zeros(len(mdp.G1_SONIC_JOINTS), dtype=np.float32)
    velocity[foot_indices] = (0.02, 0.04, 0.06, 0.08)
    action = SimpleNamespace(
        joint_velocity_before_action=np.zeros(4, dtype=np.float32),
        foot_joint_policy_indices=foot_indices,
    )
    context = SimpleNamespace(
        actions={"joint_position": action},
        sim={"robot_dof_vel": velocity},
    )

    actual = mdp.SonicFeetAccelerationReward.compute(context, np.float32(0.02))

    assert actual == pytest.approx(30.0)


def test_sonic_tracking_reward_uses_configured_anchor_local_points() -> None:
    body_count = len(mdp.G1_SONIC_BODY_NAMES)
    target_position = np.zeros((1, body_count, 3), dtype=np.float32)
    target_quaternion = _identity_quaternions((1, body_count))
    robot_position = np.zeros((body_count, 3), dtype=np.float32)
    robot_quaternion = _identity_quaternions((body_count,))
    torso = mdp.G1_SONIC_BODY_NAMES.index("torso_link")
    left_wrist = mdp.G1_SONIC_BODY_NAMES.index("left_wrist_yaw_link")
    right_wrist = mdp.G1_SONIC_BODY_NAMES.index("right_wrist_yaw_link")
    robot_position[left_wrist, 0] = 0.1
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        reference_index=np.int64(torso),
        clip=SimpleNamespace(
            tracked_bodies_pos_w=target_position,
            tracked_bodies_quat_w=target_quaternion,
        ),
    )
    scratch = np.zeros((1, 3), dtype=np.float32)
    term = mdp.SonicTrackingReward(
        np.asarray((torso, left_wrist, right_wrist), dtype=np.int64),
        np.asarray(((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=np.float32),
        np.float32(0.1),
        scratch.copy(),
        scratch.copy(),
        scratch.copy(),
        scratch.copy(),
    )
    context = SimpleNamespace(
        env_id=np.int64(0),
        commands={"motion": motion},
        sim={"tracked_body_pos": robot_position, "tracked_body_quat": robot_quaternion},
    )

    actual = mdp.SonicTrackingReward.compute(context, term)

    assert actual == pytest.approx(np.exp(-1.0 / 3.0), rel=1.0e-6)


def test_sonic_clip_boundary_clamps_reference_and_ends_without_teleporting() -> None:
    motion = SimpleNamespace(
        steps=np.asarray((1,), np.int64),
        num_future_frames=np.int64(2),
        clip=SimpleNamespace(
            frame_clip_end=np.asarray([1, 1, 3, 3]),
            joint_pos=np.asarray([[0.0], [1.0], [100.0], [101.0]], np.float32),
            joint_vel=np.zeros((4, 1), np.float32),
            tracked_bodies_quat_w=_identity_quaternions((4, 1)),
        ),
        clip_ended=np.asarray((False,)),
        hold_at_clip_end=False,
    )
    ctx = SimpleNamespace(commands={"motion": motion}, sim={"tracked_body_quat": _identity_quaternions((1,))})
    out = np.empty(16, np.float32)
    mdp.SonicG1ReferenceObservation.compute(ctx, out)
    np.testing.assert_array_equal(out[:2], [1.0, 1.0])
    assert mdp._sonic_clip_end_termination(ctx)
    motion.hold_at_clip_end = True
    assert not mdp._sonic_clip_end_termination(ctx)
    motion.steps[0] = 0
    mdp.SonicMotionCommand.advance(motion, SimpleNamespace())
    assert motion.steps[0] == 1


def test_sonic_critic_observes_robot_state_and_resets_history() -> None:
    n, joints, bodies = 2, 2, 1
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(n),
        reference_index=np.int64(0),
        reset_counter=np.asarray([0]),
        clip=SimpleNamespace(
            frame_clip_end=np.asarray([1, 1]),
            joint_pos=np.zeros((2, joints), np.float32),
            joint_vel=np.zeros((2, joints), np.float32),
            reference_body_pos_w=np.zeros((2, 3), np.float32),
            reference_body_quat_w=_identity_quaternions((2,)),
            tracked_bodies_pos_w=np.zeros((2, bodies, 3), np.float32),
        ),
    )
    sim = {
        "tracked_body_pos": np.zeros((bodies, 3), np.float32),
        "tracked_body_quat": _identity_quaternions((bodies,)),
        "tracked_body_linear_velocity": np.asarray([[1, 2, 3]], np.float32),
        "tracked_body_angular_velocity": np.zeros((bodies, 3), np.float32),
        "robot_dof_pos": np.zeros(joints, np.float32),
        "robot_dof_vel": np.zeros(joints, np.float32),
    }
    ctx = SimpleNamespace(
        env_id=0,
        commands={"motion": motion},
        sim=sim,
        actions={
            "joint_position": SimpleNamespace(
                current=np.zeros(joints, np.float32), default_angles=np.zeros(joints, np.float32)
            )
        },
    )
    term = mdp.SonicCriticObservation(np.zeros((2, n, 6 + 3 * joints), np.float32), np.full(2, -1, np.int64))
    out = np.full(2 * n * joints + 9 + 9 * bodies + n * (6 + 3 * joints), np.nan, np.float32)
    mdp.SonicCriticObservation.compute(ctx, out, term)
    assert np.isfinite(out).all()
    history_start = 2 * n * joints + 9 + 9 * bodies
    np.testing.assert_array_equal(out[history_start : history_start + 6], [1, 2, 3, 1, 2, 3])
    sim["tracked_body_linear_velocity"][:] = 4
    mdp.SonicCriticObservation.compute(ctx, out, term)
    np.testing.assert_array_equal(out[history_start : history_start + 6], [1, 2, 3, 4, 4, 4])
    motion.reset_counter[0] += 1
    mdp.SonicCriticObservation.compute(ctx, out, term)
    np.testing.assert_array_equal(out[history_start : history_start + 6], [4, 4, 4, 4, 4, 4])
    np.testing.assert_array_equal(term.history[1], 0)
