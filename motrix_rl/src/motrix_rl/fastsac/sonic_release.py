# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Playback support for the official SONIC PPO release checkpoint."""

from __future__ import annotations

import pickle
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch._weights_only_unpickler import _get_allowed_globals

from motrix_env_core import registry as env_registry
from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl.fastsac.sonic import SonicAuxiliaryConfig, SonicBackbone, SonicModelConfig
from motrix_rl.fastsac.wrap import FastSacEnvWrap
from motrix_rl.fastsac.wrap_np import FastSacNpEnvWrap
from motrix_rl.fastsac.wrap_torch import FastSacTorchEnvWrap

SONIC_RELEASE_ENV_NAME = "g1-sonic"
SONIC_RELEASE_ACTION_SCALE = 0.25
_TERM_6D_DIM = 6
_SELECTED_PREFIXES = ("encoders.g1.", "encoders.smpl.", "decoders.g1_dyn.", "decoders.g1_kin.")
_UPSTREAM_METADATA_GLOBALS = frozenset(
    {
        "accelerate.state.PartialState",
        "accelerate.utils.dataclasses.DistributedType",
        "transformers.trainer_pt_utils.AcceleratorConfig",
        "transformers.trainer_utils.HubStrategy",
        "transformers.trainer_utils.IntervalStrategy",
        "transformers.trainer_utils.SaveStrategy",
        "transformers.trainer_utils.SchedulerType",
        "transformers.training_args.OptimizerNames",
        "trl.trainer.ppo_config.PPOConfig",
        "trl.trainer.utils.OnlineTrainerState",
    }
)


class _CheckpointMetadata:
    def __new__(cls, *args: Any, **kwargs: Any) -> _CheckpointMetadata:
        del args, kwargs
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _RestrictedCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        qualified_name = f"{module}.{name}"
        allowed = _get_allowed_globals()
        if qualified_name in allowed:
            return allowed[qualified_name]
        if qualified_name == "collections.deque":
            return deque
        if qualified_name in _UPSTREAM_METADATA_GLOBALS:
            return _CheckpointMetadata
        raise pickle.UnpicklingError(f"Unsupported checkpoint global: {qualified_name}")


class _RestrictedCheckpointPickle:
    __name__ = "motrixlab_sonic_restricted_pickle"
    Unpickler = _RestrictedCheckpointUnpickler


@dataclass(frozen=True)
class SonicReleaseCheckpointReport:
    loaded: tuple[str, ...]
    ignored: tuple[str, ...]


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
        expected = _UPSTREAM_METADATA_GLOBALS | {"collections.deque"}
        unexpected = unsafe - expected
        if unexpected:
            raise RuntimeError(f"SONIC checkpoint contains unsupported globals: {sorted(unexpected)}") from None
        return torch.load(path, map_location="cpu", pickle_module=_RestrictedCheckpointPickle, weights_only=False)


def load_sonic_release_checkpoint(
    model: SonicBackbone,
    checkpoint_path: str | Path,
) -> SonicReleaseCheckpointReport:
    """Load the complete checkpoint-compatible SONIC backbone."""

    checkpoint = _load_checkpoint(Path(checkpoint_path))
    if not isinstance(checkpoint, dict):
        raise ValueError("SONIC release checkpoint must contain a dictionary")
    state = checkpoint.get("actor_model_state_dict", checkpoint.get("policy_state_dict"))
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint is not a SONIC release checkpoint")
    normalized = {
        key.removeprefix("actor_module."): value for key, value in state.items() if isinstance(value, torch.Tensor)
    }
    target = model.state_dict()
    required = {key: value for key, value in target.items() if key.startswith(_SELECTED_PREFIXES)}
    missing = sorted(set(required) - set(normalized))
    mismatched = sorted(
        key
        for key, value in required.items()
        if key in normalized and tuple(normalized[key].shape) != tuple(value.shape)
    )
    if missing or mismatched:
        raise ValueError(
            "SONIC release checkpoint is incompatible with the release model "
            f"(missing={missing}, shape_mismatch={mismatched})"
        )
    compatible = {key: normalized[key] for key in required}
    model.load_state_dict(compatible, strict=False)
    loaded_source_keys = {f"actor_module.{key}" if f"actor_module.{key}" in state else key for key in compatible}
    return SonicReleaseCheckpointReport(
        loaded=tuple(sorted(compatible)),
        ignored=tuple(sorted(set(state) - loaded_source_keys)),
    )


class SonicReleaseActor(nn.Module):
    """Official PPO policy whose action mean is stored in the ``g1_dyn`` decoder."""

    def __init__(self, config: SonicModelConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.backbone = SonicBackbone(config, SonicAuxiliaryConfig(), device=device)

    def _unpack(self, observations: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cfg = self.config
        if observations.ndim != 2 or observations.shape[1] != cfg.packed_obs_dim:
            raise ValueError(
                f"SONIC release actor input must have shape (B, {cfg.packed_obs_dim}), got {tuple(observations.shape)}"
            )
        actor_end = cfg.actor_obs_dim
        g1_end = actor_end + cfg.g1_input_dim
        smpl_end = g1_end + cfg.smpl_input_dim
        actor_obs = observations[:, :actor_end]
        g1_flat = observations[:, actor_end:g1_end]
        smpl_flat = observations[:, g1_end:smpl_end]
        g1_command_dim = cfg.g1_frame_dim - _TERM_6D_DIM
        g1_command_width = cfg.num_future_frames * g1_command_dim
        g1_reference = torch.cat(
            (
                g1_flat[:, :g1_command_width].reshape(-1, cfg.num_future_frames, g1_command_dim),
                g1_flat[:, g1_command_width:].reshape(-1, cfg.num_future_frames, _TERM_6D_DIM),
            ),
            dim=-1,
        )
        smpl_human_dim = cfg.smpl_frame_dim - _TERM_6D_DIM
        smpl_human_width = cfg.num_future_frames * smpl_human_dim
        smpl_reference = torch.cat(
            (
                smpl_flat[:, :smpl_human_width].reshape(-1, cfg.num_future_frames, smpl_human_dim),
                smpl_flat[:, smpl_human_width:].reshape(-1, cfg.num_future_frames, _TERM_6D_DIM),
            ),
            dim=-1,
        )
        return actor_obs, g1_reference, smpl_reference, observations[:, smpl_end : smpl_end + 2]

    @torch.no_grad()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actor_obs, g1_reference, smpl_reference, encoder_index = self._unpack(observations)
        output = self.backbone(
            actor_obs,
            g1_reference,
            smpl_reference,
            encoder_index,
            compute_auxiliary=False,
        )
        return self.backbone.decoders["g1_dyn"].module[-1](output.action_features)


def _make_release_env(
    *,
    env_name: str,
    sim: str | None,
    num_envs: int,
    seed: int | None,
    device: torch.device,
    render: RenderConfig,
) -> FastSacEnvWrap:
    if env_name != SONIC_RELEASE_ENV_NAME:
        raise ValueError(f"SONIC release playback requires env={SONIC_RELEASE_ENV_NAME}")
    env_cfg = env_registry.make_env_config(env_name, mode="play")
    env_cfg.max_episode_seconds = None
    env_cfg.actions.joint_position.scale = SONIC_RELEASE_ACTION_SCALE
    env_cfg.commands.motion.packed_clip_limit = 1
    env_cfg.commands.motion.start_at_timestep_zero_prob = 1.0
    env_cfg.commands.motion.hold_at_clip_end = True
    env_cfg.terminations.anchor_pos_z.threshold = 0.25
    env_cfg.terminations.anchor_ori.threshold = 1.0
    env_cfg.terminations.ee_body_pos_z.threshold = 0.25
    # This manager API requires every declared term to remain a
    # ``TerminationTermCfg``.  An infinite threshold is equivalent to the
    # disabled feet-position term in the upstream release-play config.
    env_cfg.terminations.feet_pos.threshold = float("inf")
    env = env_registry.resolve(env_name, env_cfg=env_cfg, sim=sim).make(num_envs=num_envs, seed=seed)
    if isinstance(env, TorchEnv):
        return FastSacTorchEnvWrap(env, device, render=render, clip_actions=False)
    if isinstance(env, ArrayEnv):
        return FastSacNpEnvWrap(env, device, render=render, clip_actions=False)
    raise TypeError(f"SONIC release playback does not support environment type '{type(env).__name__}'")


def play_sonic_release(
    checkpoint_path: str | Path,
    *,
    env_name: str,
    sim: str | None,
    num_envs: int,
    seed: int | None,
    render: RenderConfig,
) -> None:
    """Play an official SONIC checkpoint without MotrixLab run metadata."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _make_release_env(
        env_name=env_name,
        sim=sim,
        num_envs=num_envs,
        seed=seed,
        device=device,
        render=render,
    )
    config = SonicModelConfig.from_profile("release")
    if env.env.policy_observation_space.shape[-1] != config.packed_obs_dim:
        env.close()
        raise ValueError(
            f"SONIC release expects policy observation dim {config.packed_obs_dim}, "
            f"got {env.env.policy_observation_space.shape[-1]}"
        )
    actor = SonicReleaseActor(config, device)
    load_sonic_release_checkpoint(actor.backbone, checkpoint_path)
    actor.eval()
    observations, _ = env.reset()
    try:
        with torch.inference_mode():
            while True:
                observations, _, _, _, _ = env.step(actor(observations))
                if env.render() is False:
                    break
    finally:
        env.close()


__all__ = [
    "SONIC_RELEASE_ACTION_SCALE",
    "SONIC_RELEASE_ENV_NAME",
    "SonicReleaseActor",
    "SonicReleaseCheckpointReport",
    "load_sonic_release_checkpoint",
    "play_sonic_release",
]
