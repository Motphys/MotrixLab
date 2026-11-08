# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compact reset contract for the ball-balance simulator reset terms.

Sim reset terms run on the reset pass's compact rows: ``ctx.rand`` draws the
whole batch but the term must gather ``ctx.reset_ids`` before writing the
``(K, ...)`` staging buffers. These tests reproduce the partial-reset shape
failure directly at the term-function level, without a simulator.
"""

import pytest
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.rand import RandValue, draw_uniform, initialize_rand_states
from motrix_envs.locomotion.ball_balance.mdp.reset import (
    DofResetParams,
    _reset_ball,
    _reset_body_dof_pos,
    _reset_body_lin_vel,
    _reset_body_pos,
    _reset_body_rot,
    _reset_body_rot_vel,
)

_NUM_ENVS = 4
_RESET_IDS = torch.tensor([1, 3])
_SEED = 11


def _ctx() -> ManagerContext:
    return ManagerContext(
        env_id=torch.arange(_NUM_ENVS, dtype=torch.int64),
        actions=Map({}),
        commands=Map({}),
        metrics=Map({}),
        rand=RandValue(initialize_rand_states(_NUM_ENVS, _SEED)),
        sim=Map({}),
        dt=0.02,
        sim_reset_requested=torch.zeros((_NUM_ENVS, 1), dtype=torch.bool),
        reset_ids=_RESET_IDS,
        reset_episode=torch.ones(len(_RESET_IDS), dtype=torch.bool),
        terminated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
    )


def _fresh_draws(count: int) -> torch.Tensor:
    """Reference draws with the same seed, gathered on the reset lanes."""
    state = initialize_rand_states(_NUM_ENVS, _SEED)
    samples, _ = draw_uniform(state, count)
    return samples[_RESET_IDS]


def _writes(spec: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    return {name: torch.empty((len(_RESET_IDS), *shape)) for name, shape in spec.items()}


AMP = torch.tensor([0.1, 0.1, 0.05])
VEL_AMP = torch.tensor([0.05, 0.05, 0.0])
SPAWN = torch.tensor([0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("term", "writes_spec", "args"),
    [
        (_reset_body_pos, {"position": (1, 3)}, (SPAWN, AMP)),
        (_reset_body_lin_vel, {"linear_velocity": (1, 3)}, (AMP,)),
        (_reset_body_rot_vel, {"angular_velocity": (1, 3)}, (AMP,)),
    ],
)
def test_partial_reset_noise_terms_write_compact_rows(term, writes_spec, args) -> None:
    ctx = _ctx()
    writes = _writes(writes_spec)
    term(ctx, Map(writes), *args)
    buffer = next(iter(writes.values()))
    noise = _fresh_draws(3)
    offset = SPAWN if "position" in writes else torch.zeros(3)
    torch.testing.assert_close(buffer[:, 0], offset + noise * AMP)


def test_partial_reset_body_rot_normalizes_compact_quaternions() -> None:
    ctx = _ctx()
    writes = _writes({"rotation": (1, 4)})
    _reset_body_rot(ctx, Map(writes), AMP)
    quat = writes["rotation"][:, 0]
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones(len(_RESET_IDS)))
    # Reference via the shared euler helper on the same gathered draws.
    from motrix_env_core.math.quaternion import from_euler

    noise = _fresh_draws(3)
    expected = from_euler(noise * AMP)
    torch.testing.assert_close(quat, expected / expected.norm(dim=-1, keepdim=True))


def test_partial_reset_dof_positions_write_compact_rows() -> None:
    dof = 5
    params = DofResetParams(
        default_pos=torch.linspace(-0.1, 0.1, dof),
        joint_lower=torch.full((dof,), -1.0),
        joint_upper=torch.full((dof,), 1.0),
        noise_scale=0.05,
    )
    ctx = _ctx()
    writes = _writes({"position": (dof,), "velocity": (dof,)})
    _reset_body_dof_pos(ctx, Map(writes), params)
    noise = _fresh_draws(dof)
    torch.testing.assert_close(writes["position"], params.default_pos + noise * params.noise_scale)
    assert not writes["velocity"].any()


def test_partial_reset_ball_interleaves_position_and_velocity_draws() -> None:
    ctx = _ctx()
    writes = _writes({"position": (1, 3), "linear_velocity": (1, 3), "angular_velocity": (3,)})
    _reset_ball(ctx, Map(writes), SPAWN, AMP, VEL_AMP)
    samples = _fresh_draws(6)
    torch.testing.assert_close(writes["position"][:, 0], SPAWN + samples[:, 0::2] * AMP)
    torch.testing.assert_close(writes["linear_velocity"][:, 0], samples[:, 1::2] * VEL_AMP)
    assert not writes["angular_velocity"].any()
