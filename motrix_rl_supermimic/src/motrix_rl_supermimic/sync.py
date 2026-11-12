# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Sync trainer bridging the provider learner into the motrix_rl control plane."""

from __future__ import annotations

from pathlib import Path

import torch

from motrix_env_core import registry as env_registry
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl_supermimic.config import SuperMimicCfg
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.mbrl_env import MbrlEnvBase
from motrix_rl_supermimic.networks import Policy


class Trainer(TrainerBase):
    def __init__(self, *, context: TrainerContext[SuperMimicCfg]):
        self._context = context
        self._rlcfg = context.rl_cfg
        self._env_spec = env_registry.resolve(context.env_name, sim=context.sim)

    def _device(self) -> torch.device:
        if self._rlcfg.device:
            return torch.device(self._rlcfg.device)
        return resolve_default_device()

    def _metrics_sink(self, interval: int):
        metrics = self._context.metrics

        def sink(iteration: int, values: dict[str, float]) -> None:
            if metrics.enabled:
                metrics.scalars(values, step=iteration, env_steps=None)

        return sink if interval > 0 else None

    def _learner(self, env: MbrlEnvBase, output: Path, device: str) -> Learner:
        cfg = self._rlcfg
        interval = getattr(self._context.checkpoint, "interval", 0) or cfg.trainer.ckpt_every
        cfg.trainer.ckpt_every = interval
        learner = Learner(cfg, env, device, output, metrics=self._metrics_sink(cfg.trainer.log_every))
        resume = self._context.resume_from
        if resume:
            learner.load(resume)
        return learner

    def train(self) -> None:
        if self._context.seed is not None:
            torch.manual_seed(self._context.seed)
        env = self._env_spec.make(num_envs=self._context.num_envs, mode="train", seed=self._context.seed)
        learner = self._learner(env, self._context.checkpoint_dir, str(self._device()))
        try:
            learner.train()
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    def play(self, policy: str) -> None:
        import time

        cfg = self._rlcfg
        env = self._env_spec.make(num_envs=self._context.play_num_envs, mode="play")
        device = self._device()
        task_mdp = env.task_mdp
        spec = env.spec
        goal_dim = spec.command_dim * cfg.trainer.goal_frames
        pi_kw = {"hidden": (cfg.agent.hidden,) * 2} if cfg.agent.hidden else {}
        pi = Policy(
            spec.observation_dim, goal_dim, spec.action_dim, hist=cfg.agent.hist, z_dim=cfg.agent.z_dim, **pi_kw
        ).to(device)
        ck = torch.load(policy, map_location=device, weights_only=False)
        pi.load_state_dict(ck["pi"])
        pi.eval()
        history = torch.zeros(env.num_envs, cfg.agent.hist, spec.observation_dim, device=device)
        env.init_state()
        with torch.inference_mode():
            while True:
                obs = task_mdp.add_observation_noise(
                    task_mdp.observation(env.state(), env.prev_action, env.context, env.frame)
                ).to(device)
                shifted = torch.cat((history[:, 1:], obs[:, None]), 1)
                history = torch.where((env.episode_steps == 0)[:, None, None], obs[:, None].expand_as(history), shifted)
                goal = task_mdp.command(env.frame, cfg.trainer.goal_frames).flatten(-2)
                env.step(pi(history, goal).cpu())
                time.sleep(0.02)
