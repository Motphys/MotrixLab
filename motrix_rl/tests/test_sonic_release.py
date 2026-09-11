# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from motrix_rl.fastsac.sonic import SonicAuxiliaryConfig, SonicBackbone, SonicModelConfig
from motrix_rl.fastsac.sonic_release import SonicReleaseActor, load_sonic_release_checkpoint


class _UnsupportedCheckpointMetadata:
    pass


def _release_state(model: SonicBackbone) -> dict[str, torch.Tensor]:
    return {f"actor_module.{key}": value.detach().clone() for key, value in model.state_dict().items()}


def test_load_sonic_release_checkpoint_requires_complete_compatible_backbone(tmp_path) -> None:
    config = SonicModelConfig.from_profile("smoke")
    source = SonicBackbone(config, SonicAuxiliaryConfig(), device="cpu")
    state = _release_state(source)
    state["actor_module.encoders.teleop.module.0.weight"] = torch.randn(2, 2)
    checkpoint = tmp_path / "sonic.pt"
    torch.save({"policy_state_dict": state}, checkpoint)

    target = SonicBackbone(config, SonicAuxiliaryConfig(), device="cpu")
    report = load_sonic_release_checkpoint(target, checkpoint)

    assert report.loaded
    assert "actor_module.encoders.teleop.module.0.weight" in report.ignored
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)

    state["actor_module.encoders.g1.module.0.weight"] = torch.randn(1, 1)
    torch.save({"actor_model_state_dict": state}, checkpoint)
    with pytest.raises(ValueError, match="shape_mismatch"):
        load_sonic_release_checkpoint(target, checkpoint)


def test_load_sonic_release_checkpoint_rejects_unknown_pickle_globals(tmp_path) -> None:
    checkpoint = tmp_path / "unsafe.pt"
    torch.save({"policy_state_dict": {}, "metadata": _UnsupportedCheckpointMetadata()}, checkpoint)

    config = SonicModelConfig.from_profile("smoke")
    model = SonicBackbone(config, SonicAuxiliaryConfig(), device="cpu")
    with pytest.raises(RuntimeError, match="unsupported globals"):
        load_sonic_release_checkpoint(model, checkpoint)


def test_sonic_release_actor_returns_raw_control_decoder_mean() -> None:
    config = SonicModelConfig.from_profile("smoke")
    actor = SonicReleaseActor(config, torch.device("cpu"))
    observations = torch.randn(2, config.packed_obs_dim)
    observations[:, -2:] = torch.tensor((1.0, 0.0))

    actor_obs, g1_reference, smpl_reference, encoder_index = actor._unpack(observations)
    output = actor.backbone(
        actor_obs,
        g1_reference,
        smpl_reference,
        encoder_index,
        compute_auxiliary=False,
    )
    expected = actor.backbone.decoders["g1_dyn"].module[-1](output.action_features)

    torch.testing.assert_close(actor(observations), expected)
