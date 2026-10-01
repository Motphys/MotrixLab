# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""PPO Trainer for RSLRL integration."""

import logging
import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import RenderConfig
from motrix_rl import checkpoints
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl_rslrl.cfg import RslrlCfg
from motrix_rl_rslrl.torch import wrap_env

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    """RSLRL PPO Trainer.

    This class wraps RSLRL's OnPolicyRunner to provide a training interface
    consistent with the SKRL trainer implementation.
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        logger.info(f"Using device: {device}")

        # Wrap environment for RSLRL
        vec_env = wrap_env(env, device, render=self._render)

        # Create RSLRL config - use to_dict() method
        rslrl_cfg = self._create_rslrl_config()

        run_dir = self._context.run_dir
        runner = OnPolicyRunner(
            vec_env,
            rslrl_cfg,
            log_dir=str(run_dir),
            device=device,
        )

        # Start training
        logger.info(f"Starting training for {self._env_name}")
        logger.info(f"Number of environments: {self._context.num_envs}")

        # Get max_iterations from config
        total_iterations = rslrl_cfg["max_iterations"]
        logger.info(f"Number of learning iterations: {total_iterations}")

        runner.learn(num_learning_iterations=total_iterations)
        checkpoint_path = checkpoints.final_checkpoint_path(self._context.checkpoint_format, run_dir)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        runner.save(str(checkpoint_path))
        checkpoints.record_checkpoint_artifact(
            run_dir,
            checkpoints.BEST_POLICY,
            checkpoint_path,
            checkpoints.POLICY,
            checkpoint_format=self._context.checkpoint_format,
        )
        checkpoints.record_checkpoint_artifact(
            run_dir,
            checkpoints.LATEST_TRAINING_STATE,
            checkpoint_path,
            checkpoints.TRAINING_STATE,
            checkpoint_format=self._context.checkpoint_format,
        )

        logger.info("Training completed")

    def play(self, policy_path: str) -> None:
        """Evaluate a trained policy.

        Args:
            policy_path: Path to the saved policy file
        """
        import time

        # Determine device for evaluation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
