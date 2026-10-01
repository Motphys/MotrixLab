# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from motrix_rl import runner, runs
from motrix_rl.plugins import load_plugins


def _compose_task_option(env, rllib, algo, *, overrides):
    load_plugins()
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[2] / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={env}/{rllib}.{algo}", *overrides])
    return OmegaConf.to_object(cfg)


def test_motrix_fastsac_asynchronous_switches_trainer(tmp_path):
    from motrix_rl_builtin.fastsac.async_impl.train import Trainer as AsyncTrainer
    from motrix_rl_builtin.fastsac.config import FastSacAsyncOptionsCfg
    from motrix_rl_builtin.fastsac.sync.train import Trainer as SyncTrainer

    import motrix_envs  # noqa: F401 registers environments used by the trainer

    async_cfg = _compose_task_option(
        "g1-walk-flat",
        "motrix",
        "fastsac",
        overrides=["algo.trainer.async_options.utd_mode=learner_bound"],
    )
    sync_cfg = _compose_task_option(
        "g1-walk-flat",
        "motrix",
        "fastsac",
        overrides=["algo.asynchronous=false"],
    )

    assert async_cfg.algo.asynchronous is True
    assert sync_cfg.algo.asynchronous is False
    assert isinstance(async_cfg.algo.trainer.async_options, FastSacAsyncOptionsCfg)
    assert async_cfg.algo.trainer.async_options.utd_mode == "learner_bound"
    assert async_cfg.algo.trainer.async_options.collector_inference_device == "cuda"
    assert async_cfg.algo.trainer.async_options.collector_compile is True
    assert async_cfg.algo.trainer.async_options.collector_amp is True

    for index, (task_cfg, trainer_type) in enumerate(((sync_cfg, SyncTrainer), (async_cfg, AsyncTrainer))):
        run = runs.create_run_context(
            env_name=task_cfg.task.env,
            rllib=task_cfg.task.rllib,
            train_backend="torch",
            algo=task_cfg.task.algo,
            seed=task_cfg.seed,
            checkpoint_format="pt",
            runs_root=tmp_path / str(index),
        )
        runs.write_task_config(run.run_dir, task_cfg)

        assert isinstance(runner.create_run_handle(run).trainer, trainer_type)
