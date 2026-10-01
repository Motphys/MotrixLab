# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

from motrix_env_core.renderer import RenderConfig
from motrix_rl import checkpoints
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl_skrl.config import SkrlCfg


class SkrlEnvProtocol(Protocol):
    observation_space: Any
    state_space: Any
    action_space: Any
    device: Any
    num_envs: int

    def reset(self) -> tuple[Any, dict]: ...

    def state(self) -> Any: ...

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]: ...

    def render(self) -> Any: ...

    def close(self) -> None: ...


class SkrlAgentProtocol(Protocol):
    experiment_dir: str

    def load(self, path: str) -> None: ...

    def save(self, path: str) -> None: ...

    def act(self, obs: Any, state: Any, timestep: int, timesteps: int) -> tuple[Any, dict[str, Any]]: ...


class SkrlSequentialTrainerProtocol(Protocol):
    def __init__(self, cfg: dict[str, Any], env: SkrlEnvProtocol, agents: SkrlAgentProtocol) -> None: ...

    def train(self) -> None: ...


def add_runtime_config(
    cfg: dict[str, Any],
    env: SkrlEnvProtocol,
    scheduler_cls: type,
    scaler_cls: type,
    logging: LoggingConfig,
    checkpoint: CheckpointConfig,
    run_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    remove_mixed_precision: bool = False,
) -> dict:
    """Add runtime-specific SKRL PPO configuration."""
    if run_dir is not None and log_dir is not None:
        raise ValueError("Pass either run_dir or log_dir, not both.")
    if logging.backend != "tensorboard":
        raise ValueError("SKRL supports only the 'tensorboard' logging backend.")

    if cfg.get("learning_rate_scheduler") == "KLAdaptiveLR":
        cfg["learning_rate_scheduler"] = scheduler_cls

    if remove_mixed_precision:
        cfg.pop("mixed_precision", None)

    if cfg.get("rewards_shaper_scale", 1.0) != 1.0:
        scale = cfg.pop("rewards_shaper_scale")
        cfg["rewards_shaper"] = lambda reward, timestep, timesteps: reward * scale
    else:
        cfg.pop("rewards_shaper_scale", None)
        cfg["rewards_shaper"] = None

    cfg["observation_preprocessor"] = scaler_cls
    cfg["observation_preprocessor_kwargs"] = {
        "size": env.observation_space,
        "device": env.device,
    }
    cfg["state_preprocessor"] = scaler_cls
    cfg["state_preprocessor_kwargs"] = {
        "size": env.state_space,
        "device": env.device,
    }
    cfg["value_preprocessor"] = scaler_cls
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

    if log_dir:
        cfg["experiment"] = {
            "directory": str(log_dir),
            "experiment_name": "",
            "write_interval": logging.interval,
            "checkpoint_interval": checkpoint.interval,
        }
    elif run_dir:
        run_dir = Path(run_dir)
        cfg["experiment"] = {
            "directory": str(run_dir.parent),
            "experiment_name": run_dir.name,
            "write_interval": logging.interval,
            "checkpoint_interval": checkpoint.interval,
        }
    else:
        cfg["experiment"] = {
            "directory": "",
            "experiment_name": "",
            "write_interval": 0,
            "checkpoint_interval": 0,
        }

    return cfg


def ppo_memory_size(memory_size: int, ppo_cfg: dict[str, Any]) -> int:
    if memory_size == -1:
        return ppo_cfg["rollouts"]
    return memory_size


class SkrlPpoTrainerBase(TrainerBase):
    train_backend: str
    checkpoint_format: str
    scheduler_cls: type
    scaler_cls: type
    trainer_cls: type[SkrlSequentialTrainerProtocol]
    remove_mixed_precision: bool = False

    _env_name: str
    _rlcfg: SkrlCfg
    _render: RenderConfig | None

    def __init__(
        self,
        *,
        context: TrainerContext[SkrlCfg],
    ) -> None:
        if context.resume_from is not None:
            raise ValueError("Resume is not implemented for SKRL PPO Trainer yet.")
        env_name = context.env_name
        self._rlcfg = context.rl_cfg
        self._env_name = env_name
        self._sim = context.sim
        self._render = context.render
        self._context = context

    def train(self) -> None:
        rlcfg = self._rlcfg
        self._set_seed(self._context.seed)
        env = self._make_env(num_envs=self._context.num_envs, mode="train")
        skrl_env = self._wrap_env(env, self._render)
        models = self._make_model(skrl_env, rlcfg)
        ppo_cfg = rlcfg.agent.to_dict()
        add_runtime_config(
            ppo_cfg,
            skrl_env,
            scheduler_cls=type(self).scheduler_cls,
            scaler_cls=type(self).scaler_cls,
            run_dir=self._context.run_dir,
            logging=self._context.logging,
            checkpoint=self._context.checkpoint,
            remove_mixed_precision=self.remove_mixed_precision,
        )
        agent = self._make_agent(models, skrl_env, ppo_cfg, rlcfg.memory)
        cfg_trainer = {
            "timesteps": rlcfg.trainer.timesteps,
            "headless": self._render is None or self._render.headless,
        }
        trainer = self.trainer_cls(cfg=cfg_trainer, env=skrl_env, agents=agent)
        trainer.train()
        policy_path = checkpoints.final_checkpoint_path(self._context.checkpoint_format, self._context.run_dir)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(policy_path))
        checkpoints.record_checkpoint_artifact(
            self._context.run_dir,
            checkpoints.BEST_POLICY,
            policy_path,
            checkpoints.POLICY,
            checkpoint_format=self._context.checkpoint_format,
        )
        checkpoints.record_checkpoint_artifact(
            self._context.run_dir,
            checkpoints.LATEST_TRAINING_STATE,
            policy_path,
            checkpoints.TRAINING_STATE,
            checkpoint_format=self._context.checkpoint_format,
        )

    def play(self, policy: str) -> None:
        rlcfg = self._rlcfg
        self._set_seed(self._context.seed)
        env = self._make_env(num_envs=self._context.play_num_envs, mode="play")
        env = self._wrap_env(env, self._render)
        models = self._make_model(env, rlcfg)
        ppo_cfg = rlcfg.agent.to_dict()
        add_runtime_config(
            ppo_cfg,
            env,
            scheduler_cls=type(self).scheduler_cls,
            scaler_cls=type(self).scaler_cls,
            logging=self._context.logging,
            checkpoint=self._context.checkpoint,
            remove_mixed_precision=self.remove_mixed_precision,
        )
        agent = self._make_agent(models, env, ppo_cfg, rlcfg.memory)
        agent.load(policy)
        obs, _ = env.reset()
        state = env.state()

        recording = self._render is not None and self._render.headless
        fps = self._render.fps if recording else 60
        try:
            with self._play_context():
                while True:
                    t = time.time()
                    actions, outputs = agent.act(obs, state, timestep=0, timesteps=0)
                    actions = outputs.get("mean_actions", actions)
                    obs, _, _, _, _ = env.step(actions)
                    state = env.state()
                    if env.render() is False:
                        break
                    delta_time = time.time() - t
                    if not recording and delta_time < 1.0 / fps:
                        time.sleep(1.0 / fps - delta_time)
        finally:
            env.close()

    def _make_env(self, num_envs: int, mode: str = "train") -> Any:
        raise NotImplementedError

    def _wrap_env(self, env: Any, render: RenderConfig | None) -> SkrlEnvProtocol:
        raise NotImplementedError

    def _set_seed(self, seed: int | None) -> None:
        raise NotImplementedError

    def _make_model(self, env: SkrlEnvProtocol, rlcfg: SkrlCfg) -> dict[str, Any]:
        raise NotImplementedError

    def _make_agent(
        self,
        models: dict[str, Any],
        env: SkrlEnvProtocol,
        ppo_cfg: dict[str, Any],
        memory_cfg: Any,
    ) -> SkrlAgentProtocol:
        raise NotImplementedError

    def _play_context(self):
        return nullcontext()
