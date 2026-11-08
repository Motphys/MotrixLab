# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure tensor joint-position processing and selected-lane action-history reset."""

from __future__ import annotations

import torch


def process_joint_position_actions(
    action_queue: torch.Tensor,
    action_ptr: torch.Tensor,
    actions: torch.Tensor,
    delay_steps: torch.Tensor,
    action_scales: torch.Tensor,
    default_angles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance raw history and map delayed policy actions to position targets.

    Args:
        action_queue: Raw policy-action ring buffer ``(N, W, A)``, ``W >= 2``.
        action_ptr: Shared ``(1,)`` int64 cursor pointing at the latest action.
        actions: New raw policy actions ``(N, A)``.
        delay_steps: Per-lane int64 delay ``(N,)`` in ``[0, W - 1]``. Use
            zeros when delay is disabled.
        action_scales: Per-actuator target scales ``(A,)``.
        default_angles: Per-actuator default positions ``(A,)``.

    Returns:
        ``(targets, new_queue, new_ptr, current, previous)``. Targets, current,
        and previous have shape ``(N, A)``; history and cursor retain their
        input shapes. Current and previous are *raw*, not delayed or scaled.

    Matches ``ActionTerm.process`` followed by
    ``JointPositionActionTerm._process``: advance the shared cursor, write the
    new actions, then read the delayed slot. Inputs are not mutated. Cursor
    arithmetic and indexing stay on tensors for full-graph compilation.
    """
    num_envs, width, num_actions = action_queue.shape
    new_ptr = (action_ptr + 1) % width
    current_indices = new_ptr.reshape(1, 1, 1).expand(num_envs, 1, num_actions)
    new_queue = action_queue.scatter(1, current_indices, actions.unsqueeze(1))
    delayed_indices = ((new_ptr - delay_steps) % width).reshape(num_envs, 1, 1).expand(-1, 1, num_actions)
    delayed = new_queue.gather(1, delayed_indices).squeeze(1)
    current = new_queue.gather(1, current_indices).squeeze(1)
    previous_indices = ((new_ptr - 1) % width).reshape(1, 1, 1).expand(num_envs, 1, num_actions)
    previous = new_queue.gather(1, previous_indices).squeeze(1)
    targets = delayed * action_scales + default_angles
    return targets, new_queue, new_ptr, current, previous


def reset_action_history(action_queue: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
    """Return history with selected lanes zeroed, without changing the cursor.

    ``reset_mask`` is a fixed-shape ``(N,)`` boolean tensor, not dynamic lane
    indices. Unselected lanes retain all ``(W, A)`` entries. This pure helper
    does not resample delays; callers own reset-time randomization separately.
    """
    return torch.where(reset_mask[:, None, None], torch.zeros_like(action_queue), action_queue)


__all__ = ["process_joint_position_actions", "reset_action_history"]
