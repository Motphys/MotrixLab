# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""PPO Trainer for RSLRL integration."""

import logging
import random
import sys

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import RenderConfig
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl_rslrl.cfg import RslrlCfg
from motrix_rl_rslrl.torch import wrap_env
from motrix_rl_rslrl.torch.train.execution import run_sync
from motrix_rl_rslrl.torch.train.logging import RslrlTrainingLogger

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    """RSLRL PPO Trainer.

    Uses native RSLRL PPO primitives with MotrixLab synchronous scheduling
    and the shared training console; playback retains OnPolicyRunner.
    """

    _env_name: str
    _rlcfg: RslrlCfg
    _render: RenderConfig | None

    def __init__(
        self,
        *,
        context: TrainerContext[RslrlCfg],
    ) -> None:
        """Initialize the RSLRL PPO trainer.

        Args:
            context: Trainer runtime context.
        """
        if context.resume_from is not None:
            raise ValueError("Resume is not implemented for RSLRL PPO Trainer yet.")
        env_name = context.env_name
        self._rlcfg = context.rl_cfg
        self._env_name = env_name
        self._sim = context.sim
        self._render = context.render
        self._context = context

    def train(self) -> None:
        """Start training the agent.

        Creates the environment, wraps it for RSLRL, and runs the training loop.
        """
        # Determine device for training
        device = resolve_default_device()

        # Set random seed
        self._set_seed(self._context.seed)

        # Create environment
        env = env_registry.make(
            self._env_name,
            sim=self._sim,
            num_envs=self._context.num_envs,
            mode="train",
            seed=self._context.seed,
        )

        vec_env = training_log = None
        try:
            vec_env = wrap_env(env, device, render=self._render)
            rslrl_cfg = self._create_rslrl_config()
            # Reuse native algorithm construction/distributed setup and checkpoint
            # helpers, not its monolithic learn() or console renderer.
            runner = OnPolicyRunner(vec_env, rslrl_cfg, log_dir=None, device=str(device))
            training_log = RslrlTrainingLogger(
                log_dir=str(self._context.run_dir),
                cfg=rslrl_cfg,
                env_cfg=vec_env.cfg,
                num_envs=vec_env.num_envs,
                is_distributed=runner.is_distributed,
                gpu_world_size=runner.gpu_world_size,
                gpu_global_rank=runner.gpu_global_rank,
                device=str(device),
            )
            runner.logger = training_log
            run_sync(runner, self._context)
        finally:
            # Preserve training exceptions/cancellation while attempting all cleanup.
            active_error = sys.exc_info()[1]
            cleanup_error = None
            closers = []
            if training_log is not None:
                closers.append(training_log.close)
            if vec_env is not None:
                closers.append(vec_env.close)
            for close in closers:
                try:
                    close()
                except BaseException as exc:
                    logger.warning("RSLRL training cleanup failed", exc_info=True)
                    if cleanup_error is None:
                        cleanup_error = exc
            if active_error is None and cleanup_error is not None:
                raise cleanup_error

    def play(self, policy_path: str) -> None:
        """Evaluate a trained policy.

        Args:
            policy_path: Path to the saved policy file
        """
        import time

        # Determine device for evaluation
        device = resolve_default_device()

        # Set random seed
        self._set_seed(self._context.seed)

        # Create environment with play_num_envs
        env = env_registry.make(
            self._env_name,
            sim=self._sim,
            num_envs=self._context.play_num_envs,
            mode="play",
            seed=self._context.seed,
        )

        # Wrap environment for RSLRL
        vec_env = wrap_env(env, device, render=self._render)

        # Create RSLRL config (minimal for evaluation)
        rslrl_cfg = self._create_rslrl_config()

        # Create RSLRL runner with log_dir=None to disable logging (no git diff storage in play mode)
        runner = OnPolicyRunner(vec_env, rslrl_cfg, log_dir=None, device=device)

        # Load policy
        logger.info(f"Loading policy from {policy_path}")
        runner.load(policy_path)

        # Run evaluation loop
        logger.info("Starting evaluation loop...")
        logger.info("Press Ctrl+C to stop")
        obs, _ = vec_env.reset()
        recording = self._render is not None and self._render.headless
        fps = self._render.fps if recording else 60

        try:
            while True:
                t = time.time()

                # Get actions from policy
                with torch.no_grad():
                    policy = runner.get_inference_policy(device=device)
                    # MLPModel is callable, returns distribution mean for deterministic evaluation
                    actions = policy(obs)

                # Step environment
                obs, rewards, dones, infos = vec_env.step(actions)

                # Render the environment
                if vec_env.render() is False:
                    break

                delta_time = time.time() - t
                if not recording and delta_time < 1.0 / fps:
                    time.sleep(1.0 / fps - delta_time)

        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
        finally:
            vec_env.close()

    def _create_rslrl_config(self) -> dict:
        return add_runtime_config(self._rlcfg.to_dict(), self._context.logging, self._context.checkpoint)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def add_runtime_config(
    cfg: dict,
    logging: LoggingConfig,
    checkpoint: CheckpointConfig,
) -> dict:
    """Add framework-neutral runtime settings expected by RSLRL."""
    supported_loggers = {"tensorboard", "neptune", "wandb"}
    if logging.backend not in supported_loggers:
        raise ValueError(f"RSLRL logging backend must be one of {sorted(supported_loggers)}, got {logging.backend!r}.")
    cfg["logger"] = logging.backend
    cfg["save_interval"] = checkpoint.interval if checkpoint.interval > 0 else cfg["max_iterations"] + 1
    return cfg
