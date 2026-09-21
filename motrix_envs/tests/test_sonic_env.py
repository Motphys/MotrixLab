# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""End-to-end contract test for the bundled SONIC smoke task."""

from pathlib import Path

import numpy as np

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_envs.locomotion.sonic import mdp


def test_sonic_environment_compiles_and_steps() -> None:
    store = Path(__file__).resolve().parents[2] / "data" / "sonic" / "lafan1-pack-smoke"
    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.packed_store = str(store)
    motion = cfg.commands.motion
    joints = len(motion.joint_names)
    bodies = len(motion.tracked_body_names)
    frames = motion.num_future_frames
    smpl_joints = np.load(store / "smpl_joints.npy", mmap_mode="r").shape[1]
    policy_width = (
        frames * mdp._sonic_actor_frame_dim(joints)
        + frames * mdp._g1_reference_frame_dim(joints)
        + frames * mdp._smpl_reference_frame_dim(smpl_joints)
        + mdp.SONIC_ENCODER_COUNT
    )
    value_width = 9 + 9 * bodies + frames * (2 * joints + mdp._sonic_actor_frame_dim(joints))

    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=1)

    state = env.init_state()
    assert state.obs.policy.shape == (2, policy_width)
    assert state.obs.value.shape == (2, value_width)

    next_state = env.step(np.zeros((2, joints), dtype=np.float32))
    assert np.isfinite(next_state.obs.policy).all()
    assert np.isfinite(next_state.obs.value).all()
    assert np.isfinite(next_state.reward).all()
