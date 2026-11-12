"""First-order real cumprod(dim=0), native operation order without host checks.

Matches ATen ReduceOps.cpp::cumprod_backward's zero-aware O(n) branch.
Both zero-free and zero-containing inputs always take the tensor-only branches;
higher-order gradients, complex inputs and tensor subclasses are unsupported.

Ported from the validated task implementation: ATen's ``cumprod`` backward is
not CUDA-graph capturable, while this tensor-only formulation is.
"""
from __future__ import annotations

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.autograd.function import once_differentiable


class _SurvivalCumprod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.ndim == 0:
            raise ValueError("survival_cumprod expects a horizon dimension")
        if not x.is_floating_point():
            raise TypeError("survival_cumprod requires a real floating tensor")
        y = torch.cumprod(x, dim=0)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, y = ctx.saved_tensors
        if x.numel() <= 1 or x.shape[0] == 1:
            return grad
        w = y * grad
        is_zero = x == 0
        count = is_zero.cumsum(0)
        before_mask = count == 0
        before = w.masked_fill(~before_mask, 0.).flip(0).cumsum(0).flip(0)
        before.div_(x)
        result = torch.where(before_mask, before, torch.zeros_like(grad))
        range_mask = count == 1
        index = range_mask.max(0, keepdim=True).indices
        first_zero_mask = range_mask.logical_and(is_zero)
        mid_mask = range_mask & ~first_zero_mask
        at_zero = x.masked_fill(~mid_mask, 1.).cumprod(0)
        grad_masked = grad.masked_fill(count != 1, 0.)
        output_before = y.gather(0, (index - 1).relu_()).masked_fill_(index == 0, 1.)
        at_zero = at_zero.mul_(grad_masked).sum(0, keepdim=True).mul_(output_before)
        return torch.where(first_zero_mask, at_zero, result)


def survival_cumprod(x: torch.Tensor) -> torch.Tensor:
    """Native-order, zero-aware survival product with first-order autograd only."""
    return _SurvivalCumprod.apply(x)


__all__ = ["survival_cumprod"]
