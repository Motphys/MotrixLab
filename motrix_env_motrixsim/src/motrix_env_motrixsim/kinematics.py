# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Build a differentiable :class:`KinematicTree` from a compiled MotrixSim model.

The simulator model is the single source of truth: link topology, fixed
offsets and joint axes are read from the compiled scene, not re-parsed from
files. The resulting tree consumes a root state plus a joint-position vector
ordered as ``tree.joint_names``.
"""

from __future__ import annotations

import motrixsim as mtx
import numpy as np

from motrix_env_core.math.kinematics import KinematicTree
from motrix_env_core.math.quaternion import mul as quat_mul
from motrix_env_core.math.quaternion import rotate_vector


def kinematic_tree(model: mtx.SceneModel, base_link: str, *, device: str = "cpu") -> KinematicTree:
    """Extract the moving-link tree rooted at ``base_link`` from a compiled model.

    Fixed links are folded into the offsets of their nearest moving
    descendant; each moving link must carry exactly one 1-DoF joint (hinge).
    The root row takes the floating-base state directly, so ``base_link``
    itself must be joint-free.

    Args:
        model: A compiled scene model.
        base_link: Name of the robot's base link (usually the body name).
        device: Device for the tree's constant buffers.

    Returns:
        A :class:`KinematicTree` whose ``joint_names`` give the q-vector order.
    """
    links = [link for link in model.links if link.name == base_link]
    if not links:
        raise KeyError(f"Unknown base link {base_link!r}.")
    root = links[0]
    if root.num_joints != 0:
        raise ValueError(f"Base link {base_link!r} carries joints; the root row takes the floating-base state.")

    parent: list[int] = []
    joint_idx: list[int] = []
    axes: list[np.ndarray] = []
    off_p: list[np.ndarray] = []
    off_q: list[np.ndarray] = []
    link_names: list[str] = []
    joint_names: list[str] = []

    def add_moving(link, moving_idx: int, p_offset, q_offset) -> None:
        pose = link.local_pose  # [x, y, z, i, j, k, w] in the parent link frame
        p = p_offset + rotate_vector(q_offset, pose[:3])
        q = quat_mul(q_offset, pose[3:])
        parent.append(moving_idx)
        joint_idx.append(len(joint_names))
        axes.append(np.asarray(link.joints()[0].axis))
        off_p.append(p)
        off_q.append(q)
        joint_names.append(link.joints()[0].name)
        link_names.append(link.name)
        # Children are relative to this link's frame; offset accumulation only
        # spans fixed intermediate links, so it resets at every moving link.
        recurse(link, len(parent) - 1, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    def recurse(link, moving_idx: int, p_offset, q_offset) -> None:
        for child in link.children:
            if child.num_joints == 0:
                pose = child.local_pose
                recurse(
                    child,
                    moving_idx,
                    p_offset + rotate_vector(q_offset, pose[:3]),
                    quat_mul(q_offset, pose[3:]),
                )
            elif child.num_joints == 1 and child.joints()[0].num_dof_pos == 1:
                add_moving(child, moving_idx, p_offset, q_offset)
            else:
                raise NotImplementedError(
                    f"Link {child.name!r} needs single 1-DoF hinge joints for FK; found {child.num_joints} joint(s)."
                )

    # Root row: dummy axis/offset entries, never consumed by the recursion.
    parent.append(-1)
    joint_idx.append(-1)
    axes.append(np.zeros(3))
    off_p.append(np.zeros(3))
    off_q.append(np.array([0.0, 0.0, 0.0, 1.0]))
    link_names.append(root.name)
    recurse(root, 0, np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))

    tree = KinematicTree(
        parent,
        joint_idx,
        np.stack(axes),
        np.stack(off_p),
        np.stack(off_q),
        link_names,
        joint_names,
        device=device,
    )
    return tree
