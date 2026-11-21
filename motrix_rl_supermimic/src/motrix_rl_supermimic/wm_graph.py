# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph world-model full-window loss.

The graph owns only the tensor-only loss forward. Sampling, running-normalizer
updates, backward/gradient clipping and the optimizer step remain in the
learner loop. The network's prediction de-normalization uses the ``out_norm``
statistics while the loss scale uses the ``inc_norm`` statistics; both are
snapshotted separately and refreshed on every call.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from motrix_rl_supermimic.networks import WorldModel, run_mlp

__all__ = ["FlattenFullLoss", "WMLossGraph"]


class FlattenFullLoss(nn.Module):
    """Flattened WM loss with static snapshots of mutable normalizer state."""

    def __init__(
        self,
        wm: WorldModel,
        err_floor: torch.Tensor,
        window: int,
        batch: int,
        normalize_context: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.wm = wm
        self.window, self.batch = window, batch
        self.normalize_context = normalize_context
        self.register_buffer("err_floor", err_floor.detach().clone())
        self.register_buffer("in_mean", wm.in_norm.mean.detach().clone())
        self.register_buffer("in_var", wm.in_norm.var.detach().clone())
        self.register_buffer("out_mean", wm.out_norm.mean.detach().clone())
        self.register_buffer("out_var", wm.out_norm.var.detach().clone())
        self.register_buffer("inc_var", wm.inc_norm.var.detach().clone())
        self.in_eps, self.out_eps = wm.in_norm.eps, wm.out_norm.eps

    @torch.no_grad()
    def sync_norms(self, err_floor: torch.Tensor) -> None:
        self.in_mean.copy_(self.wm.in_norm.mean)
        self.in_var.copy_(self.wm.in_norm.var)
        self.out_mean.copy_(self.wm.out_norm.mean)
        self.out_var.copy_(self.wm.out_norm.var)
        self.inc_var.copy_(self.wm.inc_norm.var)
        self.err_floor.copy_(err_floor)

    def forward(
        self,
        initial_flat: torch.Tensor,
        actions: torch.Tensor,
        xis: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        state = initial_flat
        predictions = []
        out_std = torch.sqrt(self.out_var + self.out_eps**2)
        for j in range(self.window):
            xi_n = self.normalize_context(xis[:, j])
            computation = self.wm.world_state.prepare(state)
            inp = torch.cat((computation.features(), actions[:, j], xi_n), -1)
            inp = (inp - self.in_mean) / torch.sqrt(self.in_var + self.in_eps**2)
            y = run_mlp(self.wm.net, inp, self.wm.amp_dtype)
            prediction = y * out_std + self.out_mean
            state = computation.integrate(prediction, structured=self.wm.structured, dt=self.wm.dt)
            predictions.append(state)
        states = torch.stack(predictions, 1)
        std = torch.maximum(torch.sqrt(self.inc_var + self.wm.inc_norm.eps**2), self.err_floor)
        error = (self.wm.state_error(states, targets) / std).square().mean(-1)
        return (error * mask).sum() / self.batch


class WMLossGraph:
    """Lazily captured WM loss callable; intentionally not an ``nn.Module``.

    Capture must not consume training randomness beyond the warmup runs on the
    provided sample arguments; the input signature is fixed after capture and
    shape changes are an error, never a silent eager fallback.
    """

    def __init__(
        self,
        wm: WorldModel,
        err_floor: torch.Tensor,
        normalize_context: Callable[[torch.Tensor], torch.Tensor],
        window: int,
        batch: int,
        compile_kernel: bool = False,
    ) -> None:
        self.wm = wm
        self.err_floor = err_floor
        self.normalize_context = normalize_context
        self.window, self.batch = window, batch
        self.compile_kernel = compile_kernel
        self.module: FlattenFullLoss | None = None
        self.callable: Callable[..., torch.Tensor] | None = None
        self.key: tuple[tuple[tuple[int, ...], torch.dtype, torch.device, bool], ...] | None = None

    def __call__(
        self,
        initial_flat: torch.Tensor,
        actions: torch.Tensor,
        xis: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        args = tuple(x.contiguous() for x in (initial_flat, actions, xis, targets, mask))
        key = tuple((tuple(x.shape), x.dtype, x.device, x.requires_grad) for x in args)
        if self.callable is None:
            self.module = FlattenFullLoss(self.wm, self.err_floor, self.window, self.batch, self.normalize_context)
            self.module.sync_norms(self.err_floor)
            if self.compile_kernel:
                self.module.forward = torch.compile(self.module.forward, dynamic=False)
            rng_cpu = torch.get_rng_state().clone()
            rng_cuda = torch.cuda.get_rng_state_all()
            try:
                graph = torch.cuda.make_graphed_callables(
                    self.module, sample_args=args, num_warmup_iters=3, allow_unused_input=True
                )
                torch.cuda.synchronize()
            finally:
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state_all(rng_cuda)
            self.callable, self.key = graph, key
        if key != self.key:
            raise RuntimeError("WM loss input signature changed; recapture disabled")
        self.module.sync_norms(self.err_floor)
        return self.callable(*args)
