# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MbrlEnvBase: DirectEnv lifecycle with reward/termination/observation delegated to TaskMDP.

Subclasses write only the simulator wiring: action writes, the per-step read
program, the canonical ``state()`` projection, the reset distribution, and
any task-specific hard termination. Reward, soft termination and observation
come from the shared TaskMDP, so the simulator-labeled reward and the
imagined reward are the same function by construction.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv, DirectEnvCfg
from motrix_env_core.sim.backend import SimBackend  # noqa: F401  (type docs)
from motrix_rl_supermimic.task_mdp import StateSpec, TaskMDP

__all__ = ["MbrlEnvBase", "compile_task_heads"]


_COMPILE_TASK_HEADS = (
    "observation",
    "add_observation_noise",
    "reward",
    "soft_termination",
    "critic_observation",
    "command",
    "normalize_context",
)


def compile_task_heads(task_mdp: TaskMDP) -> None:
    """Compile tensor-only TaskMDP heads in place.

    Simulator reads/writes and lifecycle bookkeeping stay eager because they
    cross the backend boundary.  These heads are pure tensor functions and
    are the same functions used by both real transitions and imagination, so
    compiling them gives both paths one optimized implementation.
    """
    if getattr(task_mdp, "_motrix_compile_task_heads", False):
        return
    for name in _COMPILE_TASK_HEADS:
        function = getattr(task_mdp, name, None)
        if function is not None:
            setattr(task_mdp, name, torch.compile(function, dynamic=False))
    setattr(task_mdp, "_motrix_compile_task_heads", True)


class MbrlEnvBase(DirectEnv[DirectEnvCfg]):
    """Direct environment whose evaluation heads come from a TaskMDP.

    The base owns the per-environment task tensors — ``frame`` (reference
    frame index, long), ``context`` (task context), and the one-step action
    history ``prev_action`` — plus the full ``ArrayEnv`` lifecycle template:

    - ``apply_action`` records the action and delegates simulator writes to
      ``_write_action``;
    - ``compute_transition`` refreshes simulator state through
      ``_refresh_sim_state``, then fills ``reward``/``terminated`` from the
      TaskMDP (hard termination through the ``_termination`` hook) and
      rotates the action history;
    - ``compute_observation`` builds the noise-free TaskMDP observation and
      applies :meth:`TaskMDP.add_observation_noise`;
    - ``reset`` delegates row writes to ``_reset_rows`` and clears the action
      history of the reset rows.
    """

    @classmethod
    def build_task(cls, cfg: DirectEnvCfg, device: str) -> TaskMDP:
        """Construct the TaskMDP on ``device`` without any simulator state.

        The async learner needs the task's analytic heads (plus any curriculum
        tensors the task exposes) on the learner device, but must not spawn a
        second physics instance. Tasks override this with their own builder;
        there is no generic default because constructing a task is task
        knowledge (paths, motion libraries, frozen constants).
        """
        raise NotImplementedError(
            f"{cls.__name__} must override build_task(cfg, device) for physics-free learner construction"
        )

    def __init__(
        self,
        cfg: DirectEnvCfg,
        task_mdp: TaskMDP,
        num_envs: int = 1,
        backend: str | None = None,
        compile_task: bool = False,
    ):
        super().__init__(cfg, num_envs, backend)
        if compile_task:
            compile_task_heads(task_mdp)
        spec = task_mdp.spec
        self.task_mdp = task_mdp
        self.frame = torch.zeros(num_envs, dtype=torch.long)
        self.context = torch.zeros(num_envs, spec.context_dim)
        self.prev_action = torch.zeros(num_envs, spec.action_dim)
        self._action = torch.zeros(num_envs, spec.action_dim)
        self._transition: dict[str, torch.Tensor] = {}
        # Keep the last simulator projection across lifecycle hooks.  The
        # generic ArrayEnv path otherwise asks a concrete task for the same
        # post-step state once in compute_transition and again in
        # compute_observation; task state projections may include quaternion
        # conversion and simulator read-buffer materialization.
        self._cached_state: torch.Tensor | None = None

    @property
    def spec(self) -> StateSpec:
        return self.task_mdp.spec

    @property
    def episode_steps(self) -> torch.Tensor:
        """Per-environment episode step count of the current ArrayEnv state."""
        return self._state.episode_steps

    @property
    def action_dt(self) -> float:
        """Control-step duration from the environment configuration."""
        return float(self.cfg.ctrl_dt)

    @property
    def observation_space(self) -> gym.spaces.Box:
        spec = self.spec
        return gym.spaces.Box(-torch.inf, torch.inf, (spec.observation_dim,))

    @property
    def action_space(self) -> gym.spaces.Box:
        spec = self.spec
        return gym.spaces.Box(-1.0, 1.0, (spec.action_dim,))

    # ------------------------------------------------------------------
    # Lifecycle template (ArrayEnv hooks)
    # ------------------------------------------------------------------

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        self._action = actions
        # Pre-step state and frame: the simulator still holds the transition's
        # start (reference semantics — replay stores the pre-step frame).
        self._transition = {
            "s": self.state().detach(),
            "f": self.frame.detach().clone(),
            "a": actions.detach().clone(),
            "a_prev": self.prev_action.detach().clone(),
            "xi": self.context.detach().clone(),
        }
        self._write_action(actions)
        return state

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self._refresh_sim_state()
        s = self.state()
        self._cached_state = s
        state.reward.copy_(self.task_mdp.reward(s, self.frame, self._action, self.prev_action))
        state.terminated.copy_(self._termination(s))
        self.prev_action.copy_(self._action)
        # s1 is the physical endpoint: reset writes have not run yet. The
        # reward's frame is the post-refresh one (reference evaluates against
        # the next reference frame).
        self._transition.update(
            s1=s.detach(),
            r=state.reward.detach().clone(),
            ep_step=state.episode_steps.detach().clone() + 1,
        )
        return state

    @property
    def last_transition(self) -> dict[str, torch.Tensor]:
        """The most recent transition, finalized with post-bookkeeping done flags.

        Keys: ``s``/``s1`` (canonical states), ``a``/``a_prev``, ``xi``
        (context), ``r``, ``f`` (frame), ``ep_step``, ``done``
        (terminated | truncated, read at access time).
        """
        transition = dict(self._transition)
        transition["done"] = self._state.terminated | self._state.truncated
        return transition

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        current = self._cached_state
        if current is None:
            current = self.state()
        obs = self.task_mdp.observation(current, self.prev_action, self.context, self.frame)
        state.obs = self.task_mdp.add_observation_noise(obs)
        self._cached_state = None
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        self._reset_rows(env_ids)
        self.prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Task/simulator wiring (subclass responsibilities)
    # ------------------------------------------------------------------

    def state(self) -> torch.Tensor:
        """Canonical flat state ``(..., state_dim)`` of all environments.

        Read from the simulator data refreshed by ``_refresh_sim_state``;
        reset rows carry their post-reset state.
        """
        raise NotImplementedError

    def _write_action(self, actions: torch.Tensor) -> None:
        """Push actions into the simulator (compile/write program call)."""
        raise NotImplementedError

    def _refresh_sim_state(self) -> None:
        """Execute the per-step read program before ``state()`` is consumed."""
        raise NotImplementedError

    def _reset_rows(self, env_ids: torch.Tensor) -> None:
        """Write reset rows: simulator state, frame and context distributions."""
        raise NotImplementedError

    def _termination(self, state: torch.Tensor) -> torch.Tensor:
        """Hard termination of ``state``; the TaskMDP only defines the soft variant."""
        return torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
