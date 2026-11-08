# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def raw_action_to_close_ratio(raw_gripper_action: torch.Tensor, use_sigmoid: bool) -> torch.Tensor:
    raw = raw_gripper_action.to(torch.float32)
    if bool(use_sigmoid):
        return 1.0 / (1.0 + torch.exp(-raw))
    return ((raw + 1.0) * 0.5).clamp(0.0, 1.0)


def binary_hysteresis_step(
    *,
    close_ratio: torch.Tensor,
    prev_closed: torch.Tensor,
    steps_since_switch: torch.Tensor,
    close_on_threshold: float,
    open_off_threshold: float,
    min_switch_interval_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    close_ratio = close_ratio.to(torch.float32)
    prev_closed = prev_closed.to(torch.bool)
    steps_since_switch = steps_since_switch.to(torch.int32)

    close_on = float(min(max(close_on_threshold, 0.0), 1.0))
    open_off = float(min(max(open_off_threshold, 0.0), 1.0))
    if open_off > close_on:
        open_off = close_on
    min_steps = max(int(min_switch_interval_steps), 0)

    can_switch = steps_since_switch >= min_steps
    want_close = close_ratio > close_on
    want_open = close_ratio < open_off

    next_closed = prev_closed.clone()
    next_closed = torch.where(~prev_closed & can_switch & want_close, True, next_closed)
    next_closed = torch.where(prev_closed & can_switch & want_open, False, next_closed)
    switched = next_closed != prev_closed
    return next_closed, switched
