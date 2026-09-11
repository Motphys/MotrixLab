# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from motrix_rl.fastsac.async_impl.shm import WeightSnapshot
from motrix_rl.fastsac.buffer import EmpiricalNormalization
from motrix_rl.fastsac.sonic import SonicActor, SonicAuxiliaryConfig, SonicModelConfig


def _packed(cfg: SonicModelConfig, batch: int = 4) -> torch.Tensor:
    g1 = torch.randn(batch, cfg.g1_input_dim)
    smpl = torch.randn(batch, cfg.smpl_input_dim)
    obs = torch.randn(batch, cfg.actor_obs_dim)
    # Reference terms use the manager's term-major layout: command/body terms
    # first, then the 6D orientation/wrist terms.
    index = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])[:batch]
    return torch.cat((obs, g1, smpl, index), dim=-1)


def test_sonic_profiles_have_expected_packed_dimensions() -> None:
    assert SonicModelConfig.from_profile("release").packed_obs_dim == 2412
    assert SonicModelConfig.from_profile("lafan").packed_obs_dim == 966
    assert SonicModelConfig.from_profile("smoke").packed_obs_dim == 243


def test_sonic_actor_observation_width_is_derived_from_action_width() -> None:
    cfg = SonicModelConfig(
        num_future_frames=2,
        num_tokens=1,
        token_dim=4,
        fsq_levels=8,
        action_dim=7,
    )
    assert cfg.actor_obs_frame_dim == 27
    assert cfg.actor_obs_dim == 54
    assert cfg.g1_frame_dim == 20
    assert cfg.g1_input_dim == 40
    assert cfg.smpl_frame_dim == 84
    assert cfg.packed_obs_dim == 264


def test_sonic_actor_forward_and_auxiliary_backward() -> None:
    cfg = SonicModelConfig.from_profile("smoke")
    actor = SonicActor(cfg, SonicAuxiliaryConfig(), log_std_min=-5.0, log_std_max=0.0, device="cpu")
    actions, logp, aux = actor.get_actions_and_log_probs_with_aux(_packed(cfg))
    assert actions.shape == (4, cfg.action_dim)
    assert logp.shape == (4,)
    assert torch.isfinite(actions).all() and torch.isfinite(logp).all()
    assert torch.isfinite(aux["total"])
    (actions.square().mean() + aux["total"]).backward()
    assert actor.backbone.encoders["g1"].module[0].weight.grad is not None
    assert actor.backbone.encoders["smpl"].module[0].weight.grad is not None


def test_sonic_deterministic_action_is_bounded() -> None:
    cfg = SonicModelConfig.from_profile("smoke")
    actor = SonicActor(cfg, SonicAuxiliaryConfig(), log_std_min=-5.0, log_std_max=0.0, device="cpu")
    actions = actor.explore(_packed(cfg, 2), deterministic=True)
    assert actions.shape == (2, cfg.action_dim)
    assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0)


def test_sonic_normalization_preserves_encoder_selector() -> None:
    cfg = SonicModelConfig.from_profile("smoke")
    observations = _packed(cfg)
    normalizer = EmpiricalNormalization(cfg.packed_obs_dim, device="cpu", passthrough_dims=2)

    normalized = normalizer(observations, update=True)

    torch.testing.assert_close(normalized[:, -2:], observations[:, -2:])
    actor = SonicActor(cfg, SonicAuxiliaryConfig(), log_std_min=-5.0, log_std_max=0.0, device="cpu")
    assert actor.explore(normalized).shape == (4, cfg.action_dim)


def test_sonic_ignores_non_finite_inactive_reference() -> None:
    cfg = SonicModelConfig.from_profile("smoke")
    observations = _packed(cfg)
    smpl_start = cfg.actor_obs_dim + cfg.g1_input_dim
    observations[:, smpl_start : smpl_start + cfg.smpl_input_dim] = float("nan")
    observations[:, -2:] = torch.tensor((1.0, 0.0))
    actor = SonicActor(cfg, SonicAuxiliaryConfig(), log_std_min=-5.0, log_std_max=0.0, device="cpu")

    actions, logp, auxiliary = actor.get_actions_and_log_probs_with_aux(observations)

    assert torch.isfinite(actions).all()
    assert torch.isfinite(logp).all()
    assert all(torch.isfinite(loss) for loss in auxiliary.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SONIC snapshot test requires a GPU")
def test_cuda_sonic_actor_buffers_publish_to_cpu_snapshot() -> None:
    cfg = SonicModelConfig.from_profile("smoke")
    actor = SonicActor(cfg, SonicAuxiliaryConfig(), log_std_min=-5.0, log_std_max=0.0, device="cuda:0")
    normalizer = EmpiricalNormalization(cfg.packed_obs_dim, device="cuda:0", passthrough_dims=2)
    buffer_numel = sum(buffer.numel() for buffer in actor.buffers() if buffer.numel())
    weights = WeightSnapshot(
        sum(parameter.numel() for parameter in actor.parameters()),
        cfg.packed_obs_dim,
        buffer_numel,
    )

    assert all(buffer.device.type == "cuda" for buffer in actor.buffers())
    weights.publish(actor, normalizer)

    assert weights.version == 1
