# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager randomness follows the seeded default Torch stream, including compilation."""

import pytest
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.random import RandomSource
from motrix_env_core.mdp.observations import add_uniform_noise


def _context() -> ManagerContext:
    return ManagerContext(
        env_id=torch.arange(4),
        actions=dict({}),
        commands=dict({}),
        metrics=dict({}),
        sim=dict({}),
        dt=0.02,
        sim_reset_requested=torch.zeros(4, dtype=torch.bool),
        terminated=torch.zeros(4, dtype=torch.bool),
        reward=torch.zeros(4),
        rand=RandomSource(),
    )


def _noise(ctx: ManagerContext, out: torch.Tensor) -> torch.Tensor:
    out.add_(1.0)
    add_uniform_noise(ctx, out, 0.25)
    return out.mul_(2.0)


@pytest.mark.parametrize("compiled", [False, True])
def test_uniform_observation_noise_is_seeded_bounded_and_advances(compiled: bool) -> None:
    sample = torch.compile(_noise, backend="eager", fullgraph=True) if compiled else _noise
    torch.manual_seed(123)  # The training entry point owns seeding, not the context.
    ctx = _context()
    first = sample(ctx, torch.zeros(1024, 4))
    second = sample(ctx, torch.zeros(1024, 4))
    assert bool((first >= 1.5).all())
    assert bool((first < 2.5).all())
    assert not torch.equal(first, second)
    assert abs(float(first.mean()) - 2.0) < 0.02
    torch.manual_seed(123)
    reproduced = sample(_context(), torch.zeros(1024, 4))
    torch.testing.assert_close(first, reproduced, rtol=0, atol=0)


def test_random_sources_share_default_torch_draws_and_preserve_layout() -> None:
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    first, second = RandomSource(), RandomSource()
    assert torch.equal(before, torch.get_rng_state())
    value = torch.zeros(2, 3, dtype=torch.float64).T
    sample = first.uniform((3, 2), dtype=torch.float64, device=value.device)
    interleaved = torch.rand(11)
    like = second.uniform_like(value)
    after = torch.get_rng_state().clone()
    assert not torch.equal(before, after)
    assert like.shape == value.shape and like.dtype == value.dtype
    assert like.device == value.device and like.stride() == value.stride()
    torch.manual_seed(7)
    torch.testing.assert_close(sample, torch.rand((3, 2), dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(interleaved, torch.rand(11), rtol=0, atol=0)
    torch.testing.assert_close(like, torch.rand_like(value), rtol=0, atol=0)
    assert torch.equal(after, torch.get_rng_state())


@pytest.mark.parametrize("compiled", [False, True])
def test_zero_noise_preserves_values_and_default_random_state(compiled: bool) -> None:
    def zero_noise(ctx: ManagerContext, out: torch.Tensor) -> torch.Tensor:
        add_uniform_noise(ctx, out, 0.0)
        return out

    sample = torch.compile(zero_noise, backend="eager", fullgraph=True) if compiled else zero_noise
    ctx = _context()
    before = torch.get_rng_state().clone()
    out = sample(ctx, torch.ones(4, 3))
    torch.testing.assert_close(out, torch.ones(4, 3))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("backend", ["eager", pytest.param("inductor", marks=pytest.mark.slow)])
def test_fullgraph_random_multi_term_loop_repeats_in_same_execution_mode(backend: str) -> None:
    def uniform_term(ctx: ManagerContext, out: torch.Tensor) -> torch.Tensor:
        return out + ctx.rand.uniform(out.shape, dtype=out.dtype, device=out.device)

    def noise_term(ctx: ManagerContext, out: torch.Tensor) -> torch.Tensor:
        return _noise(ctx, out).square()

    def schedule(ctx: ManagerContext, out: torch.Tensor) -> torch.Tensor:
        # The executor loops over fixed terms, not over environments.
        for term in (uniform_term, noise_term, uniform_term):
            out = term(ctx, out)
        return out

    sample = torch.compile(schedule, backend=backend, fullgraph=True)

    def rollout() -> tuple[list[torch.Tensor], torch.Tensor]:
        torch.manual_seed(7)
        ctx = _context()
        values = [sample(ctx, torch.zeros(count, 3)) for count in (2, 4, 2, 2)]
        return values, torch.get_rng_state().clone()

    actual, after = rollout()
    reproduced, reproduced_after = rollout()
    for value, repeated in zip(actual, reproduced, strict=True):
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(value, repeated, rtol=0, atol=0)
    assert not torch.equal(actual[2], actual[3])
    assert torch.equal(after, reproduced_after)
    # Inductor may use a different RNG algorithm; no cross-mode equality promise.
