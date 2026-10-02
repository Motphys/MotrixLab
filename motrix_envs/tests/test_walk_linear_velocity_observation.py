# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Optional actor velocity layout and compiled body-frame observation contract."""

from dataclasses import replace

import numpy as np
import pytest

from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.observations import BodyLinearVelocityObsCfg
from motrix_envs.locomotion.humanoid.g1_stairs import make_g129dof_walk_mixed_cfg


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_actor_linear_velocity_compiles_body_frame_and_normalization(enabled):
    torch = pytest.importorskip("torch")
    from motrix_rl.fastsac.buffer import EmpiricalNormalization

    cfg = make_g129dof_walk_mixed_cfg()
    cfg.observations.policy.base_lin_vel = BodyLinearVelocityObsCfg(scale=2.0) if enabled else None
    # Test inputs ensure nonzero horizontal velocity distinguishes body/world
    # frames without changing robot masses or any training factory defaults.
    cfg.sim_reset.humanoid_state.randomization = replace(
        cfg.sim_reset.humanoid_state.randomization,
        enabled=True,
        root_velocity_range=(0.2, 0.6),
        curriculum_steps=0,
    )
    cfg = replace(cfg)  # Reassemble model queries for the enabled reset inputs.
    resolved = cfg.observation_cfgs()
    assert ("base_lin_vel" in resolved["policy"]) is enabled
    env = ManagerEnv(cfg, num_envs=2, seed=17)
    try:
        state = env.init_state()
        env.compile()
        policy_group = env.observation_groups["policy"]
        value_group = env.observation_groups["value"]
        policy_size = sum(entry.size for entry in policy_group.terms)
        value_size = sum(entry.size for entry in value_group.terms)
        assert state.obs.policy.shape == (2, policy_size)
        assert state.obs.value.shape == (2, value_size)
        assert env.policy_observation_space.shape == (policy_size,)
        assert env.value_observation_space.shape == (value_size,)
        assert value_size - policy_size == (0 if enabled else 3)
        assert ("base_lin_vel" in {entry.name for entry in policy_group.terms}) is enabled

        # Use real simulator reads after a transition; read-program views are
        # immutable. Independently invert the full XYZW quaternion in NumPy,
        # rather than reusing the observation dispatch's rotation helper.
        state = env.step(np.zeros((2, env.num_actuators), dtype=np.float32))
        velocity_entry = next(entry for entry in value_group.terms if entry.name == "base_lin_vel")
        quaternion_query, velocity_query = velocity_entry.term.args[:2]
        quaternion = env.sim_data[env._term_query_keys[quaternion_query]]
        velocity = env.sim_data[env._term_query_keys[velocity_query]]
        inverse_xyz = -quaternion[:, :3]
        twice_cross = 2.0 * np.cross(inverse_xyz, velocity)
        expected = (velocity + quaternion[:, 3:4] * twice_cross + np.cross(inverse_xyz, twice_cross)) * 2.0
        assert not np.allclose(expected, velocity * 2.0, atol=1e-6)
        for name, group, obs in (
            ("policy", policy_group, state.obs.policy),
            ("value", value_group, state.obs.value),
        ):
            normalizer = EmpiricalNormalization(shape=group.size, device=torch.device("cpu"))
            normalized = normalizer(torch.as_tensor(obs.copy()))
            assert tuple(normalized.shape) == (2, group.size)
            assert tuple(normalizer.state_dict()["_mean"].shape) == (1, group.size)
            assert tuple(normalizer.state_dict()["_var"].shape) == (1, group.size)
            assert tuple(normalizer.state_dict()["_std"].shape) == (1, group.size)
            assert torch.isfinite(normalized).all()
            print(f"{enabled=}, {name}: observation/normalizer shape={obs.shape}")
            offset = 0
            for entry in group.terms:
                if entry.name == "base_lin_vel":
                    np.testing.assert_allclose(obs[:, offset : offset + entry.size], expected, atol=1e-6)
                    np.testing.assert_allclose(
                        normalizer.state_dict()["_mean"].numpy()[:, offset : offset + entry.size],
                        expected.mean(axis=0, keepdims=True),
                        atol=1e-6,
                    )
                    print(f"{enabled=}, {name}: shape={obs.shape}, body-frame mean={expected.mean(axis=0)}")
                offset += entry.size

        next_state = env.step(np.zeros((2, env.num_actuators), dtype=np.float32))
        assert next_state.obs.policy.shape == (2, policy_size)
        assert next_state.obs.value.shape == (2, value_size)
        assert np.all(np.isfinite(next_state.obs.policy))
        assert np.all(np.isfinite(next_state.obs.value))
    finally:
        del env
