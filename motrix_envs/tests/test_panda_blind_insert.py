# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from motrix_env_core import registry
from motrix_env_core.math import quaternion
from motrix_env_core.sim.write import BodyJointPositionWrite, KinematicBodyPositionWrite, KinematicBodyRotationWrite
from motrix_envs.manipulation.panda_blind_insert.cfg import TerminationConfig


def _make(num_envs: int = 2):
    return registry.make("panda-blind-insertion", num_envs=num_envs, headless=True)


def test_panda_blind_insert_direct_env_contract() -> None:
    env = _make()
    state = env.init_state()
    assert tuple(state.obs.policy.shape) == (2, 1260)  # 30 x 42
    assert state.obs.value is not None
    assert tuple(state.obs.value.shape) == (2, 52)  # 42 + 10 privileged

    state = env.step(torch.zeros((2, 7), dtype=torch.float32))
    assert tuple(state.obs.policy.shape) == (2, 1260)
    assert torch.isfinite(state.obs.policy).all()
    assert torch.isfinite(state.obs.value).all()
    assert torch.isfinite(state.reward).all()


def test_panda_blind_insert_reset_places_peg_near_hole() -> None:
    env = _make(num_envs=8)
    env.init_state()
    metrics = env._metrics()
    peg = env.sim_data["peg_pos"]
    gap_mm = (peg[:, 2] - 0.040 - 0.040) * 1000.0
    assert torch.all((gap_mm > 0.0) & (gap_mm < 8.0))
    assert torch.all(metrics["xy"] <= env.cfg.reset.offset_cap)
    assert torch.quantile(metrics["tilt"], 0.95) < 8.0


def test_panda_blind_insert_closed_hold_survives_one_hundred_steps() -> None:
    env = _make(num_envs=4)
    env.init_state()
    initial = env.sim_data["peg_pos"].clone()
    for _ in range(100):
        state = env.step(torch.zeros((4, 7), dtype=torch.float32))
        assert not state.done.any()
    final = env.sim_data["peg_pos"]
    drift_mm = (final[:, 2] - initial[:, 2]).abs() * 1000.0
    assert torch.quantile(drift_mm, 0.95) < 2.5


def test_partial_reset_does_not_advance_other_rows() -> None:
    env = _make()
    env.step(torch.full((2, 7), 0.5))
    names = ("robot_joint_pos", "robot_joint_vel", "peg_pos", "socket_pos")
    live_before = {name: env.sim_data[name][1].clone() for name in names}
    reset_socket_before = env.sim_data["socket_pos"][0].clone()
    assert live_before["robot_joint_vel"].abs().max() > 0

    env.reset(torch.tensor([0]))
    env.sim_data.execute()  # Read live physics, including the unreset row.

    for name, before in live_before.items():
        torch.testing.assert_close(env.sim_data[name][1], before, rtol=0, atol=0)
    assert not torch.equal(env.sim_data["socket_pos"][0], reset_socket_before)


def test_truncation_preserves_final_policy_history_and_refills_reset_row() -> None:
    env = _make()
    state = env.init_state()
    history_len = env.observation_spec.policy.history_length
    initial = state.obs.policy.unflatten(1, (history_len, -1)).clone()
    state.episode_steps[0] = env.cfg.max_episode_steps - 1

    state = env.step(torch.full((2, 7), 0.5))

    assert state.truncated.tolist() == [True, False]
    assert state.final_obs.env_ids.tolist() == [0]
    final = state.final_obs.obs.policy.unflatten(1, (history_len, -1))[0]
    current = state.obs.policy.unflatten(1, (history_len, -1))
    torch.testing.assert_close(final[:-1], initial[0, 1:])
    assert not torch.equal(final[-1], initial[0, -1])
    torch.testing.assert_close(current[0], current[0, -1].expand_as(current[0]))
    torch.testing.assert_close(current[1, :-1], initial[1, 1:])


def test_wall_contact_detects_square_peg_overlap_before_success() -> None:
    env = registry.make(
        "panda-blind-insertion", {"termination": TerminationConfig(hold_steps=1)}, num_envs=4, headless=True
    )
    state = env.init_state()
    peg = env.sim_data["peg_pos"]
    # Align the socket's up axis opposite the downward-pointing peg axis.
    socket_quat = quaternion.mul(env.sim_data["peg_quat"], torch.tensor([0, 1, 0, 0], dtype=torch.float32))
    half_yaw = math.radians(2.5)
    socket_quat[3] = quaternion.mul(
        socket_quat[3], torch.tensor([0, 0, math.sin(half_yaw), math.cos(half_yaw)], dtype=torch.float32)
    )
    # Diagonal in-hole clearance, 0.4 mm straight-wall overlap, above the
    # hole at the same offset, and a centered peg with yaw-induced overlap.
    relative = torch.tensor(
        [[0.0008, 0.0008, 0.062], [0.0014, 0, 0.062], [0.0014, 0, 0.087], [0, 0, 0.062]], dtype=torch.float32
    )
    writes = env.sim.write_compiler.compile(
        {
            "robot": BodyJointPositionWrite("link0"),
            "socket_pos": KinematicBodyPositionWrite(("blind_socket",)),
            "socket_quat": KinematicBodyRotationWrite(("blind_socket",)),
        },
        reset=True,
    )
    writes.buffer("robot").copy_(env.sim_data["robot_joint_pos"])
    writes.buffer("socket_pos")[:, 0] = peg - quaternion.rotate_vector(socket_quat, relative)
    writes.buffer("socket_quat")[:, 0] = socket_quat
    writes.execute()
    env.sim.step(1)

    state = env.compute_transition(state)

    assert state.metrics["penetration"].tolist() == pytest.approx([0, 0.0004, 0, 0.00042], abs=0.00002)
    assert state.metrics["contact"][[1, 3]].all()
    assert (state.metrics["depth"][[1, 3]] >= env.cfg.termination.target_depth).all()
    assert (state.metrics["xy_error"][[1, 3]] <= env.cfg.termination.xy_tolerance).all()
    assert not state.metrics["success"][[1, 3]].any()
