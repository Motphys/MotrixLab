# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SONIC actor used by the Motrix FastSAC provider.

The module deliberately keeps the environment-facing contract small: a SONIC
actor consumes the packed policy observation emitted by the manager and
returns the same ``(actions, log_probs)`` pair as the generic FastSAC actor.
The G1/SMPL tokenizer and decoders mirror the upstream SONIC graph while the
policy head uses the local FastSAC Normal-Tanh sampling convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from motrix_rl.fastsac.config import SonicAuxiliaryConfig

try:  # optional at source-import time; installed by motrix-rl dependencies
    from vector_quantize_pytorch import FSQ
except ImportError:  # pragma: no cover - gives an actionable error on bare source checkouts
    FSQ = None  # type: ignore[assignment,misc]


SONIC_VECTOR_DIM = 3
SONIC_ROTATION_REPRESENTATION_DIM = 2 * SONIC_VECTOR_DIM
SONIC_ACTOR_JOINT_FEATURES = 3
SONIC_ENCODER_COUNT = 2
SONIC_SMPL_JOINT_COUNT = 24
SONIC_SMPL_END_EFFECTOR_COUNT = 2


@dataclass(frozen=True)
class SonicModelConfig:
    num_future_frames: int
    num_tokens: int
    token_dim: int
    fsq_levels: int
    action_dim: int = 29
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 512
    g1_encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    smpl_encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    g1_motion_decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    g1_control_decoder_hidden_dims: tuple[int, ...] = (2048, 2048, 1024, 1024, 512, 512)

    @property
    def actor_obs_dim(self) -> int:
        return self.expected_actor_obs_dim

    @property
    def expected_actor_obs_dim(self) -> int:
        """Proprioceptive history width implied by this model configuration."""
        return self.num_future_frames * self.actor_obs_frame_dim

    @property
    def actor_obs_frame_dim(self) -> int:
        """Width of one proprioceptive history frame."""
        return SONIC_ACTOR_JOINT_FEATURES * self.action_dim + 2 * SONIC_VECTOR_DIM

    @property
    def g1_frame_dim(self) -> int:
        return 2 * self.action_dim + SONIC_ROTATION_REPRESENTATION_DIM

    @property
    def smpl_end_effector_dim(self) -> int:
        return SONIC_SMPL_END_EFFECTOR_COUNT * SONIC_VECTOR_DIM

    @property
    def smpl_frame_dim(self) -> int:
        return (
            SONIC_SMPL_JOINT_COUNT * SONIC_VECTOR_DIM + SONIC_ROTATION_REPRESENTATION_DIM + self.smpl_end_effector_dim
        )

    @property
    def g1_input_dim(self) -> int:
        return self.num_future_frames * self.g1_frame_dim

    @property
    def smpl_input_dim(self) -> int:
        return self.num_future_frames * self.smpl_frame_dim

    @property
    def packed_obs_dim(self) -> int:
        return self.actor_obs_dim + self.g1_input_dim + self.smpl_input_dim + SONIC_ENCODER_COUNT

    @property
    def token_total_dim(self) -> int:
        return self.num_tokens * self.token_dim

    @classmethod
    def from_profile(cls, profile: str) -> SonicModelConfig:
        profiles = {
            "release": dict(
                num_future_frames=10,
                num_tokens=2,
                token_dim=32,
                fsq_levels=32,
                actor_hidden_dim=512,
                critic_hidden_dim=512,
            ),
            "lafan": dict(
                num_future_frames=4,
                num_tokens=2,
                token_dim=8,
                fsq_levels=32,
                actor_hidden_dim=256,
                critic_hidden_dim=256,
                g1_encoder_hidden_dims=(512, 256),
                smpl_encoder_hidden_dims=(512, 256),
                g1_motion_decoder_hidden_dims=(512, 256),
                g1_control_decoder_hidden_dims=(1024, 512),
            ),
            "smoke": dict(
                num_future_frames=1,
                num_tokens=1,
                token_dim=8,
                fsq_levels=16,
                actor_hidden_dim=128,
                critic_hidden_dim=128,
                g1_encoder_hidden_dims=(256, 128),
                smpl_encoder_hidden_dims=(256, 128),
                g1_motion_decoder_hidden_dims=(256, 128),
                g1_control_decoder_hidden_dims=(512, 256),
            ),
        }
        try:
            return cls(**profiles[profile])
        except KeyError as exc:
            raise ValueError(f"unknown SONIC model profile {profile!r}; expected release, lafan, or smoke") from exc

    @classmethod
    def from_mapping(cls, values: dict) -> SonicModelConfig:
        """Build from the Hydra ``sonic_model`` group while preserving tuples."""
        allowed = {
            "num_future_frames",
            "num_tokens",
            "token_dim",
            "fsq_levels",
            "action_dim",
            "actor_hidden_dim",
            "critic_hidden_dim",
            "g1_encoder_hidden_dims",
            "smpl_encoder_hidden_dims",
            "g1_motion_decoder_hidden_dims",
            "g1_control_decoder_hidden_dims",
        }
        data = {key: values[key] for key in allowed if key in values}
        for key in (
            "g1_encoder_hidden_dims",
            "smpl_encoder_hidden_dims",
            "g1_motion_decoder_hidden_dims",
            "g1_control_decoder_hidden_dims",
        ):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


class SonicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, device=None):
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for i, (source, target) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(source, target, device=device))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.module = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


@dataclass
class SonicBackboneOutput:
    selected_tokens: torch.Tensor
    action_features: torch.Tensor
    auxiliary_losses: dict[str, torch.Tensor]


class SonicBackbone(nn.Module):
    def __init__(self, config: SonicModelConfig, auxiliary: SonicAuxiliaryConfig, device=None):
        super().__init__()
        if FSQ is None:
            raise RuntimeError("SONIC requires the 'vector-quantize-pytorch' package")
        self.config = config
        self.auxiliary_config = auxiliary
        self.encoders = nn.ModuleDict(
            {
                "g1": SonicMLP(
                    config.g1_input_dim, config.g1_encoder_hidden_dims, config.token_total_dim, device=device
                ),
                "smpl": SonicMLP(
                    config.smpl_input_dim, config.smpl_encoder_hidden_dims, config.token_total_dim, device=device
                ),
            }
        )
        self.quantizer = FSQ(levels=[config.fsq_levels] * config.token_dim, return_indices=False).to(device)
        self.decoders = nn.ModuleDict(
            {
                "g1_dyn": SonicMLP(
                    config.token_total_dim + config.actor_obs_dim,
                    config.g1_control_decoder_hidden_dims,
                    config.action_dim,
                    device=device,
                ),
                "g1_kin": SonicMLP(
                    config.token_total_dim,
                    config.g1_motion_decoder_hidden_dims,
                    config.g1_input_dim,
                    device=device,
                ),
            }
        )

    def _encode(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        batch = reference.shape[0]
        latent = self.encoders[name](reference.flatten(start_dim=1))
        return latent.reshape(batch, self.config.num_tokens, self.config.token_dim)

    def forward(
        self,
        actor_obs: torch.Tensor,
        g1_reference: torch.Tensor,
        smpl_reference: torch.Tensor,
        encoder_index: torch.Tensor,
        *,
        compute_auxiliary: bool = False,
    ) -> SonicBackboneOutput:
        cfg = self.config
        if actor_obs.ndim != 2 or actor_obs.shape[-1] != cfg.actor_obs_dim:
            raise ValueError(f"SONIC actor observation must have width {cfg.actor_obs_dim}")
        if tuple(g1_reference.shape[1:]) != (cfg.num_future_frames, cfg.g1_frame_dim):
            raise ValueError("invalid SONIC G1 reference shape")
        if tuple(smpl_reference.shape[1:]) != (cfg.num_future_frames, cfg.smpl_frame_dim):
            raise ValueError("invalid SONIC SMPL reference shape")
        if encoder_index.shape != (actor_obs.shape[0], SONIC_ENCODER_COUNT):
            raise ValueError(f"SONIC encoder_index must have shape (B, {SONIC_ENCODER_COUNT})")
        if not torch.compiler.is_compiling():
            if torch.any((encoder_index != 0) & (encoder_index != 1)) or torch.any(encoder_index.sum(-1) == 0):
                raise ValueError("SONIC encoder_index must contain binary masks and activate an encoder")
        g1_mask = encoder_index[:, 0].bool()
        smpl_mask = encoder_index[:, 1].bool()
        g1_required = torch.ones_like(g1_mask) if compute_auxiliary else g1_mask
        g1_input = torch.where(g1_required[:, None, None], g1_reference, torch.zeros_like(g1_reference))
        smpl_input = torch.where(smpl_mask[:, None, None], smpl_reference, torch.zeros_like(smpl_reference))
        g1_encoded = self._encode("g1", g1_input)
        smpl_encoded = self._encode("smpl", smpl_input)
        g1_latent = g1_encoded * g1_mask[:, None, None]
        smpl_latent = smpl_encoded * smpl_mask[:, None, None]
        g1_tokens = self.quantizer(g1_latent)[0].contiguous()
        smpl_tokens = self.quantizer(smpl_latent)[0].contiguous()
        selected = torch.where(smpl_mask[:, None, None], smpl_tokens, g1_tokens)
        decoder_input = torch.cat((selected.flatten(start_dim=1), actor_obs), dim=-1)
        action_features = self.decoders["g1_dyn"].module[:-1](decoder_input)
        losses: dict[str, torch.Tensor] = {}
        if compute_auxiliary:
            reconstruction = self.decoders["g1_kin"](selected.flatten(start_dim=1)).reshape(
                -1, cfg.num_future_frames, cfg.g1_frame_dim
            )
            recon_loss = F.mse_loss(reconstruction, g1_reference)
            smpl_weight = smpl_mask.to(g1_encoded.dtype)
            smpl_count = smpl_weight.sum().clamp_min(1.0)
            align_per_sample = (smpl_encoded - g1_encoded).square().mean(dim=(1, 2))
            cycle_per_sample = (self._encode("g1", reconstruction) - g1_encoded).square().mean(dim=(1, 2))
            align = (align_per_sample * smpl_weight).sum() / smpl_count
            cycle = (cycle_per_sample * smpl_weight).sum() / smpl_count
            total = (
                self.auxiliary_config.reconstruction * recon_loss
                + self.auxiliary_config.latent_alignment * align
                + self.auxiliary_config.cycle_consistency * cycle
            )
            losses = {
                "reconstruction": recon_loss,
                "latent_alignment": align,
                "cycle_consistency": cycle,
                "total": total,
            }
        return SonicBackboneOutput(selected, action_features, losses)


class UnitLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, device=None):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False, device=device)
        nn.init.orthogonal_(self.w.weight)

    def forward(self, x):
        return self.w(x)

    def normalize_parameters(self):
        with torch.no_grad():
            self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.bias = nn.Parameter(torch.zeros(dim, device=device))
        self.register_buffer("running_mean", torch.zeros(dim, device=device))
        self.register_buffer("running_var", torch.ones(dim, device=device))
        self.momentum, self.eps = momentum, eps

    def forward(self, x, training: bool):
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias, training, self.momentum, self.eps
        )

    def normalize_parameters(self):
        with torch.no_grad():
            norm = (self.weight.square().sum() + self.bias.square().sum() + 1e-8).rsqrt() * self.weight.numel() ** 0.5
            self.weight.mul_(norm)
            self.bias.mul_(norm)


class UnitRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight

    def normalize_parameters(self):
        with torch.no_grad():
            self.weight.mul_(self.weight.numel() ** 0.5 / (self.weight.square().sum() + 1e-8).sqrt())


class SonicPolicyBlock(nn.Module):
    def __init__(self, dim: int, device=None):
        super().__init__()
        self.w1 = UnitLinear(dim, dim * 4, device=device)
        self.n1 = UnitBatchNorm(dim * 4, device=device)
        self.w2 = UnitLinear(dim * 4, dim, device=device)
        self.n2 = UnitBatchNorm(dim, device=device)

    def forward(self, x, training: bool):
        residual = x
        x = F.relu(self.n1(self.w1(x), training))
        x = F.relu(self.n2(self.w2(x), training))
        return x + residual


class SonicActor(nn.Module):
    """SONIC policy with the FastSAC actor protocol."""

    def __init__(
        self,
        config: SonicModelConfig,
        auxiliary: SonicAuxiliaryConfig,
        *,
        log_std_min,
        log_std_max,
        actor_num_blocks: int = 2,
        device,
    ):
        super().__init__()
        self.config = config
        self.backbone = SonicBackbone(config, auxiliary, device=device)
        policy_input = config.actor_obs_dim + config.token_total_dim
        self.embed_norm = UnitBatchNorm(policy_input, device=device)
        self.embed = UnitLinear(policy_input, config.actor_hidden_dim, device=device)
        self.blocks = nn.ModuleList(
            [SonicPolicyBlock(config.actor_hidden_dim, device=device) for _ in range(actor_num_blocks)]
        )
        self.post_norm = UnitRMSNorm(config.actor_hidden_dim, device=device)
        self.mean = UnitLinear(config.actor_hidden_dim, config.action_dim, device=device)
        self.mean_bias = nn.Parameter(torch.zeros(config.action_dim, device=device))
        self.std = UnitLinear(config.actor_hidden_dim, config.action_dim, device=device)
        self.std_bias = nn.Parameter(torch.zeros(config.action_dim, device=device))
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        self.n_act = config.action_dim
        self.register_buffer("action_scale", torch.ones(config.action_dim, device=device))
        self.register_buffer("action_bias", torch.zeros(config.action_dim, device=device))

    def _policy(self, obs, training: bool):
        cfg = self.config
        a = obs[:, : cfg.actor_obs_dim]
        g1_start = cfg.actor_obs_dim
        g1_end = g1_start + cfg.g1_input_dim
        smpl_end = g1_end + cfg.smpl_input_dim
        g1_flat = obs[:, g1_start:g1_end]
        smpl_flat = obs[:, g1_end:smpl_end]
        # The manager flattens observation terms term-major: all command
        # frames precede the orientation term (and all SMPL body features
        # precede the wrist term). Rebuild the per-frame tokenizer layout.
        g1_command_frame_dim = cfg.g1_frame_dim - SONIC_ROTATION_REPRESENTATION_DIM
        g1_command_width = cfg.num_future_frames * g1_command_frame_dim
        g1 = torch.cat(
            (
                g1_flat[:, :g1_command_width].reshape(-1, cfg.num_future_frames, g1_command_frame_dim),
                g1_flat[:, g1_command_width:].reshape(-1, cfg.num_future_frames, SONIC_ROTATION_REPRESENTATION_DIM),
            ),
            dim=-1,
        )
        smpl_body_frame_dim = cfg.smpl_frame_dim - cfg.smpl_end_effector_dim
        smpl_body_width = cfg.num_future_frames * smpl_body_frame_dim
        smpl = torch.cat(
            (
                smpl_flat[:, :smpl_body_width].reshape(-1, cfg.num_future_frames, smpl_body_frame_dim),
                smpl_flat[:, smpl_body_width:].reshape(-1, cfg.num_future_frames, cfg.smpl_end_effector_dim),
            ),
            dim=-1,
        )
        index = obs[:, smpl_end : smpl_end + SONIC_ENCODER_COUNT]
        output = self.backbone(a, g1, smpl, index, compute_auxiliary=training)
        x = torch.cat((a, output.selected_tokens.flatten(start_dim=1)), dim=-1)
        x = self.embed(self.embed_norm(x, training=training))
        for block in self.blocks:
            x = block(x, training)
        x = self.post_norm(x)
        mean_features = self.mean(x)
        std_features = self.std(x)
        mean = mean_features + self.mean_bias.to(mean_features.dtype)
        raw_log_std = std_features + self.std_bias.to(std_features.dtype)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(raw_log_std) + 1)
        return mean, log_std, output.auxiliary_losses

    def forward(self, obs):
        mean, _log_std, _ = self._policy(obs, training=self.training)
        return torch.tanh(mean) * self.action_scale + self.action_bias, mean, _log_std

    def get_actions_and_log_probs(self, obs):
        mean, log_std, _ = self._policy(obs, training=False)
        dist = torch.distributions.Normal(mean, log_std.exp(), validate_args=False)
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        logp = (dist.log_prob(raw) - torch.log(1 - squashed.square() + 1e-6)).sum(-1)
        return squashed * self.action_scale + self.action_bias, logp

    def get_actions_and_log_probs_with_aux(self, obs):
        mean, log_std, aux = self._policy(obs, training=True)
        dist = torch.distributions.Normal(mean, log_std.exp(), validate_args=False)
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        logp = (dist.log_prob(raw) - torch.log(1 - squashed.square() + 1e-6)).sum(-1)
        return squashed * self.action_scale + self.action_bias, logp, aux

    @torch.no_grad()
    def explore(self, obs, deterministic=False):
        mean, log_std, _ = self._policy(obs, training=False)
        if deterministic:
            return torch.tanh(mean) * self.action_scale + self.action_bias
        raw = torch.distributions.Normal(mean, log_std.exp(), validate_args=False).sample()
        return torch.tanh(raw) * self.action_scale + self.action_bias


__all__ = ["SonicActor", "SonicAuxiliaryConfig", "SonicBackbone", "SonicModelConfig"]
