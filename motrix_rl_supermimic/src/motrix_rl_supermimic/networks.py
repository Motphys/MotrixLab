# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""World model, policy (goal encoder + shared decoder), and critic.

State semantics and dimensions come from ``WorldState``; action and observation
dimensions come from the assembled task spaces. No robot constants live here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from motrix_env_core.observations import ObservationGroupSpec
from motrix_rl_supermimic.config import AgentCfg
from motrix_rl_supermimic.world_state import WorldState

__all__ = [
    "Critic",
    "NormStatsView",
    "Policy",
    "RunningNorm",
    "WorldModel",
    "build_policy",
    "mlp",
    "resolve_amp_dtype",
    "run_mlp",
]


def mlp(
    inp: int, hidden: list[int], out: int, act: type[nn.Module] = nn.ELU, layer_norm: bool = False
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if x.shape[0] == 0 or self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)


def resolve_amp_dtype(amp: str) -> torch.dtype | None:
    """Resolve the user-facing precision mode at model construction boundaries."""
    if amp == "none":
        return None
    if amp == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown AMP mode {amp!r}; expected 'none' or 'bf16'")


def run_mlp(net: nn.Module, x: torch.Tensor, amp_dtype: torch.dtype | None = None) -> torch.Tensor:
    # Autocast only MLP matmuls; state integration, FK and rewards stay fp32.
    if amp_dtype is None or x.device.type != "cuda":
        return net(x)
    with torch.autocast("cuda", dtype=amp_dtype):
        return net(x).float()


class WorldModel(nn.Module):
    """Learned dynamics over flat states with task-owned state semantics.

    The network predicts normalized codec targets; ``WorldState`` defines feature
    extraction, regression targets, integration and the state-error metric.
    """

    def __init__(
        self,
        world_state: WorldState,
        context_dim: int,
        hidden: tuple[int, ...] = (2048, 2048, 2048),
        structured: bool = False,
        dt: float = 0.02,
        amp_dtype: torch.dtype | None = None,
        *,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.world_state = world_state
        self.amp_dtype = amp_dtype
        self.structured, self.dt = structured, dt
        self.in_dim = world_state.feature_dim + action_dim + context_dim
        self.out_dim = world_state.prediction_dim
        self.net = mlp(self.in_dim, list(hidden), self.out_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        # Plain codec increments scale the loss independently of structured targets.
        self.inc_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.world_state.features(state), action, context), -1)

    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self.world_state.increments(state, next_state)

    def targets(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self.world_state.targets(state, next_state, structured=self.structured, dt=self.dt)

    @torch.no_grad()
    def update_norms(
        self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor, context: torch.Tensor
    ) -> None:
        """Share transition decoding across the three world-model normalizers."""
        statistics = self.world_state.statistics(state, next_state, structured=self.structured, dt=self.dt)
        self.in_norm.update(torch.cat((statistics.features, action, context), -1))
        self.out_norm.update(statistics.targets)
        self.inc_norm.update(statistics.increments)

    def step(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        computation = self.world_state.prepare(state)
        inputs = torch.cat((computation.features(), action, context), -1)
        output = run_mlp(self.net, self.in_norm(inputs), self.amp_dtype)
        prediction = output * self.out_norm.std() + self.out_norm.mean
        return computation.integrate(prediction, structured=self.structured, dt=self.dt)

    def forward(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.step(state, action, context)

    def state_error(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.world_state.error(prediction, truth)


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-h+1:t}); deterministic, latent z bounded by tanh."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        hist: int,
        z_dim: int = 32,
        hidden: tuple[int, ...] = (2048, 1024, 512, 256),
        reference_action: nn.Module | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hist = hist
        self.amp_dtype = amp_dtype
        # Task-supplied frozen base-action module; the policy output becomes a
        # residual on top of it. Its buffers follow .to(device/dtype).
        self.reference_action = reference_action
        self.enc = mlp(goal_dim, list(hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * hist, list(hidden), action_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Encode one or many goals; the last axis is the goal feature width."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal), self.amp_dtype))

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = self.encode_goal(goal) if z is None else z
        o = self.obs_norm(obs_hist).flatten(-2)
        correction = run_mlp(self.dec, torch.cat((z, o), -1), self.amp_dtype)
        if self.reference_action is None:
            return correction
        return self.reference_action(goal) + correction


def build_policy(
    cfg: AgentCfg,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    device: str,
    *,
    observation_spec: ObservationGroupSpec,
    reference_action: nn.Module | None = None,
    amp_dtype: torch.dtype | None = None,
) -> Policy:
    """Build a policy from its dimensions and explicit reference-action dependency."""
    kw = {"hidden": (cfg.hidden,) * 2} if cfg.hidden else {}
    return Policy(
        obs_dim,
        goal_dim,
        action_dim,
        hist=observation_spec.history_length,
        z_dim=cfg.z_dim,
        reference_action=reference_action,
        amp_dtype=amp_dtype,
        **kw,
    ).to(device)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (256, 128), amp_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.amp_dtype = amp_dtype
        self.net = mlp(in_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(x), self.amp_dtype).squeeze(-1)


class NormStatsView:
    """Transport adapter mapping :class:`RunningNorm` to the weight channel's
    ``_mean``/``_std``/``_var``/``count`` attribute contract.

    ``_mean``/``_var`` are live buffer views (writes land in the normalizer);
    ``_std`` is derived, so the receiver's ``_std.copy_`` is intentionally a
    no-op — inference recomputes it from ``var``.
    """

    def __init__(self, norm: RunningNorm) -> None:
        self._norm = norm
        # Receiver writes derived stats through this throwaway slot; inference
        # recomputes std from var, so its content is never read.
        self._std_slot = torch.empty(1, norm.mean.shape[-1])

    @property
    def _mean(self) -> torch.Tensor:
        return self._norm.mean.reshape(1, -1)

    @property
    def _std(self) -> torch.Tensor:
        return self._std_slot

    @property
    def _var(self) -> torch.Tensor:
        return self._norm.var.reshape(1, -1)

    @property
    def count(self) -> torch.Tensor:
        return self._norm.count
