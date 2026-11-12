# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""World model, policy (goal encoder + shared decoder), and critic.

Every dimension comes from the task's ``StateSpec``; no robot constants
live here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from motrix_rl_supermimic import state as st
from motrix_rl_supermimic.state import mm3, mv3
from motrix_rl_supermimic.task_mdp import StateSpec

__all__ = ["Critic", "NormStatsView", "Policy", "RunningNorm", "WorldModel", "mlp", "run_mlp"]


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
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

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
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

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None or x.device.type != "cuda":
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    ``structured``: the network predicts velocity
    increments plus small position corrections on top of the semi-implicit drift
    implied by the velocities, so rollout positions no longer compound raw
    increment errors.
    """

    # Weight of the new velocity in the position drift: n semi-implicit
    # substeps with a linearly changing velocity give (n + 1) / (2 n); n = 4.
    DRIFT_W = 0.625

    def __init__(
        self,
        spec: StateSpec,
        hidden=(2048, 2048, 2048),
        structured: bool = False,
        contact_modes: bool = False,
        dt: float = 0.02,
    ):
        super().__init__()
        self.spec = spec
        self.structured, self.dt = structured, dt
        self.contact_modes = contact_modes  # set before use; flag param added below
        n_j = st.num_joints(spec)
        self.num_j = n_j
        self.in_dim = st.inv_dim(spec) + spec.action_dim + spec.context_dim
        self.out_dim = 3 + 3 + 3 + 3 + 2 * n_j
        self.net = mlp(self.in_dim, list(hidden), self.out_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        # Scale of the plain frame-local increments: normalizes the training
        # loss and errors regardless of what the network outputs represent.
        self.inc_norm = RunningNorm(self.out_dim, eps=1e-4)

    def inputs(self, S, a, xi_n):
        return torch.cat((st.invariant_features(self.spec, S), a, xi_n), -1)

    def _drift(self, S, v1, w1, qd1):
        """Semi-implicit position update implied by the old and new velocities."""
        p, R, v, w, q, qd = S
        H = st.yaw_mat(R)
        a, b = self.dt * (1 - self.DRIFT_W), self.dt * self.DRIFT_W
        return (
            mv3(H.transpose(-1, -2), a * v + b * v1),
            mv3(R.transpose(-1, -2), a * w + b * w1),
            a * qd + b * qd1,
        )

    @classmethod
    def increments(cls, S, S1):
        """Frame-local increments from S to S1."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = st.yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                st.so3_log(mm3(Rt, R1)),
                mv3(Rt, v1 - v),
                mv3(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def targets(self, S, S1):
        """Regression targets: plain increments, or velocity increments plus
        position corrections on the semi-implicit drift (``structured``)."""
        inc = self.increments(S, S1)
        if not self.structured:
            return inc
        dp, dth, dv, dw, dq, dqd = inc.split((3, 3, 3, 3, self.num_j, self.num_j), -1)
        gp, gth, gq = self._drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    def step(self, S, a, xi_n):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n)))
        if self.contact_modes:
            cl, cr = torch.sigmoid(y[..., -2:]).unbind(-1)
            mode_w = torch.stack(((1 - cl) * (1 - cr), cl * (1 - cr), (1 - cl) * cr, cl * cr), -1)
            y = (y[..., :-2].unflatten(-1, (4, self.out_dim)) * mode_w.unsqueeze(-1)).sum(-2)
        d = y * self.out_norm.std() + self.out_norm.mean
        H = st.yaw_mat(R)
        n_j = q.shape[-1]
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, n_j, n_j), -1)
        v1 = v + mv3(R, dv)
        w1 = w + mv3(R, dw)
        qd1 = qd + dqd
        if self.structured:
            gp, gth, gq = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        p1 = p + mv3(H, dp)
        R1 = R @ st.so3_exp(dth)
        return (p1, R1, v1, w1, q + dq, qd1)

    def forward(self, S, a, xi_n):
        return self.step(S, a, xi_n)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = st.yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                st.so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-h+1:t}); deterministic, latent z bounded by tanh."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        action_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        hidden=(2048, 1024, 512, 256),
        reference_action: nn.Module | None = None,
    ):
        super().__init__()
        self.hist = hist
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
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # The reference-action module is task-owned and rebuilt from the task;
        # a policy without it drops the checkpoint's reference-action keys so
        # it can load a reference-skip checkpoint strictly.
        if getattr(self, "reference_action", None) is None:
            for key in [k for k in list(state_dict) if k.startswith(prefix + "reference_action.")]:
                del state_dict[key]
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = self.encode_goal(goal) if z is None else z
        o = self.obs_norm(obs_hist).flatten(-2)
        correction = run_mlp(self.dec, torch.cat((z, o), -1))
        if self.reference_action is None:
            return correction
        return self.reference_action(goal) + correction


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128)):
        super().__init__()
        self.net = mlp(in_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


class NormStatsView:
    """Transport adapter mapping :class:`RunningNorm` to the weight channel's
    ``_mean``/``_std``/``_var``/``count`` attribute contract.

    ``_mean``/``_var`` are live buffer views (writes land in the normalizer);
    ``_std`` is derived, so the receiver's ``_std.copy_`` is intentionally a
    no-op — inference recomputes it from ``var``.
    """

    def __init__(self, norm: RunningNorm):
        self._norm = norm
        # Receiver writes derived stats through this throwaway slot; inference
        # recomputes std from var, so its content is never read.
        self._std_slot = torch.empty(1, norm.mean.shape[-1])

    @property
    def _mean(self):
        return self._norm.mean.reshape(1, -1)

    @property
    def _std(self):
        return self._std_slot

    @property
    def _var(self):
        return self._norm.var.reshape(1, -1)

    @property
    def count(self):
        return self._norm.count
