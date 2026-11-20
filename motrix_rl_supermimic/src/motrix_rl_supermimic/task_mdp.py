# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Numerical task head compilation.

The analytic MDP contract itself is :class:`~motrix_rl_supermimic.manager_task.ManagerTaskMDP`:
the learner independently builds one to evaluate predicted states for
world-model rollouts, while real sampling state belongs to the configured
command term and transition recording belongs to the environment. Numerical heads have no
env/simulator handles and remain device-agnostic and ``torch.compile``-friendly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motrix_rl_supermimic.manager_task import ManagerTaskMDP

import torch

__all__ = ["compile_task_heads"]


def compile_task_heads(task_mdp: ManagerTaskMDP) -> None:
    """Compile the required tensor-only TaskMDP heads in place, once per owner.

    The environment constructor and learner each own a separately constructed
    task and call this once when compilation is enabled. Simulator I/O and
    lifecycle bookkeeping stay eager across the backend boundary; optional
    task hooks remain untouched.
    """
    task_mdp.observation = torch.compile(task_mdp.observation, dynamic=False)
    task_mdp.reward = torch.compile(task_mdp.reward, dynamic=False)
    task_mdp.soft_termination = torch.compile(task_mdp.soft_termination, dynamic=False)
    task_mdp.critic_observation = torch.compile(task_mdp.critic_observation, dynamic=False)
    task_mdp.goal = torch.compile(task_mdp.goal, dynamic=False)
    task_mdp.normalize_context = torch.compile(task_mdp.normalize_context, dynamic=False)
