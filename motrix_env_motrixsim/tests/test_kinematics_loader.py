# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Loader parity: the extracted KinematicTree against simulator FK.

Random root states and joint positions are pushed through the backend writes;
link positions, rotations and velocities sampled from the simulator must match
the tree's differentiable torch evaluation.
"""

import numpy as np
import pytest
import torch

from motrix_env_core.base import SimCfg
from motrix_env_core.math.quaternion import to_matrix
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyJointPositionWrite,
    BodyJointVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_env_core.sim.registry import create_sim_backend
from motrix_env_motrixsim.kinematics import kinematic_tree


@pytest.fixture(scope="module")
def g1():
    from motrix_env_core import registry

    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    backend = create_sim_backend("motrixsim")(cfg.scene, SimCfg(), 1)
    base_link = cfg.scene.objs.robot.resolved_base_link_name
    tree = kinematic_tree(backend._model, base_link)
    return backend, tree, base_link


def _euler_to_quat_xyzw(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )


def test_tree_matches_simulator_fk(g1):
    backend, tree, base_link = g1
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, size=(3,)) + np.array([0.0, 0.0, 1.0])
    quat = _euler_to_quat_xyzw(*rng.uniform(-np.pi, np.pi, size=(3,)))
    v = rng.uniform(-1, 1, size=(3,))
    w = rng.uniform(-1, 1, size=(3,))
    q = rng.uniform(-0.5, 0.5, size=len(tree.joint_names))
    qd = rng.uniform(-1, 1, size=len(tree.joint_names))

    program = backend.write_compiler.compile(
        {
            "body_position": BodyPositionWrite((base_link,)),
            "body_rotation": BodyRotationWrite((base_link,)),
            "body_linear_velocity": BodyLinearVelocityWrite((base_link,)),
            "body_angular_velocity": BodyAngularVelocityWrite((base_link,)),
            "joint_position": BodyJointPositionWrite(base_link),
            "joint_velocity": BodyJointVelocityWrite(base_link),
        }
    )
    reads = {}
    for name in tree.link_names:
        reads[f"pos_{name}"] = LinkPositionQuery(link=name)
        reads[f"rot_{name}"] = LinkQuaternionQuery(link=name)
        reads[f"linvel_{name}"] = LinkLinearVelocityQuery(link=name)
        reads[f"angvel_{name}"] = LinkAngularVelocityQuery(link=name)
    read = backend.compile_reads(reads)

    program.buffer("body_position")[0, 0] = torch.tensor(p)
    program.buffer("body_rotation")[0, 0] = torch.tensor(quat)
    program.buffer("body_linear_velocity")[0, 0] = torch.tensor(v)
    program.buffer("body_angular_velocity")[0, 0] = torch.tensor(w)
    program.buffer("joint_position")[0, : len(q)] = torch.tensor(q)
    program.buffer("joint_velocity")[0, : len(qd)] = torch.tensor(qd)

    program.execute(torch.tensor([0], dtype=torch.int64))
    read.execute(torch.tensor([0], dtype=torch.int64))

    P, R, V, W = tree.forward(
        torch.tensor(p, dtype=torch.float32).unsqueeze(0),
        to_matrix(torch.tensor(quat, dtype=torch.float32).unsqueeze(0)),
        torch.tensor(v, dtype=torch.float32).unsqueeze(0),
        torch.tensor(w, dtype=torch.float32).unsqueeze(0),
        torch.tensor(q, dtype=torch.float32).unsqueeze(0),
        torch.tensor(qd, dtype=torch.float32).unsqueeze(0),
    )
    for i, name in enumerate(tree.link_names):
        np.testing.assert_allclose(read[f"pos_{name}"][0], P[0, i].numpy(), atol=1e-4)
        np.testing.assert_allclose(read[f"angvel_{name}"][0], W[0, i].numpy(), atol=1e-4)
        R_sim = to_matrix(torch.tensor(np.asarray(read[f"rot_{name}"][0], dtype=np.float32)))
        np.testing.assert_allclose(R_sim.numpy(), R[0, i].numpy(), atol=1e-4)
        np.testing.assert_allclose(read[f"linvel_{name}"][0], V[0, i].numpy(), atol=1e-4)


def test_joint_names_cover_every_moving_link(g1):
    _, tree, _ = g1
    assert tree.link_names[0]  # root row present
    assert len(tree.joint_names) == len([j for j in tree.joint_idx if j >= 0])
    assert len(set(tree.joint_names)) == len(tree.joint_names)


def test_kinematic_tree_query_resolves_backend_neutral_tree(g1):
    backend, _, base_link = g1
    from motrix_env_core.sim import KinematicTreeQuery

    model = backend.compile_model({"tree": KinematicTreeQuery(body=base_link)})
    tree = model.others["tree"]
    assert tree.link_names[0] == base_link
    assert tree.joint_idx[0] == -1
    # Same tree the loader produces: parity is already pinned by the test above.
    assert len(tree.joint_names) == 29
