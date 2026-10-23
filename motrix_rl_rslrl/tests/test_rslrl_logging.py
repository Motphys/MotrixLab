# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Metric/backend contracts for RSLRL without a real logging service."""

import sys
from types import ModuleType

import pytest
import torch
from motrix_rl_rslrl.torch.train.logging import RslrlTrainingLogger


class RecordingWriter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scalars = {}
        self.videos = []
        self.models = []
        self.configs = []
        self.files = []
        self.flush_count = 0
        self.close_count = 0
        self.stop_count = 0

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (float(value), step)

    def save_video(self, video, step):
        self.videos.append((video, step))

    def save_model(self, path, step):
        self.models.append((path, step))

    def store_config(self, *configs):
        self.configs.append(configs)

    def save_file(self, path):
        self.files.append(path)

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1

    def stop(self):
        self.stop_count += 1


def make_logger(tmp_path, *, backend="tensorboard", rnd=False, distributed=False):
    logger = RslrlTrainingLogger(
        log_dir=None,
        cfg={"num_steps_per_env": 4, "algorithm": {"rnd_cfg": {"weight": 0.25} if rnd else None}, "logger": backend},
        env_cfg={"task": "test"},
        num_envs=2,
        is_distributed=distributed,
        gpu_world_size=2,
        gpu_global_rank=1 if distributed else 0,
        device="cpu",
    )
    # The real logger's code-state snapshot is unrelated to writer behavior.
    logger.log_dir = str(tmp_path)
    logger.writer = RecordingWriter()
    logger.logger_type = backend
    return logger


def log_iteration(logger):
    return logger.log(
        it=3,
        start_it=0,
        total_it=10,
        collect_time=0.75,
        learn_time=0.5,
        loss_dict={"surrogate": 0.2, "value": 0.3},
        learning_rate=0.001,
        action_std=torch.tensor([0.2, 0.4]),
        rnd_weight=0.25,
        print_minimal=True,
        width=20,
        pad=10,
    )


def test_log_preserves_scalar_tags_and_episode_buffers_without_printing(tmp_path, capsys):
    logger = make_logger(tmp_path)
    logger.process_env_step(
        torch.tensor([2.0, 4.0]),
        torch.tensor([1, 0]),
        {"episode": {"score": 2.0, "Rewards/alive": torch.tensor(1.0)}},
    )
    logger.process_env_step(
        torch.tensor([6.0, 8.0]),
        torch.tensor([0, 1]),
        {"log": {"score": torch.tensor([4.0, 6.0])}},
    )

    assert log_iteration(logger) is None

    scalars = logger.writer.scalars
    assert set(scalars) == {
        "Episode/score",
        "Rewards/alive",
        "Loss/surrogate",
        "Loss/value",
        "Loss/learning_rate",
        "Policy/mean_noise_std",
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Train/mean_reward/time",
        "Train/mean_episode_length/time",
    }
    assert scalars["Episode/score"] == (4.0, 3)
    assert scalars["Rewards/alive"] == (1.0, 3)
    assert scalars["Loss/surrogate"] == (0.2, 3)
    assert scalars["Loss/value"] == (0.3, 3)
    assert scalars["Loss/learning_rate"] == (0.001, 3)
    assert scalars["Policy/mean_noise_std"][0] == pytest.approx(0.3)
    assert scalars["Perf/total_fps"] == (12.0, 3)
    assert scalars["Perf/collection_time"] == (0.75, 3)
    assert scalars["Perf/learning_time"] == (0.5, 3)
    assert scalars["Train/mean_reward"] == (7.0, 3)
    assert scalars["Train/mean_episode_length"] == (1.5, 3)
    assert scalars["Train/mean_reward/time"] == (7.0, 1)
    assert scalars["Train/mean_episode_length/time"] == (1.5, 1)
    assert list(logger.rewbuffer) == [2.0, 12.0]
    assert list(logger.lenbuffer) == [1.0, 2.0]
    assert logger.tot_timesteps == 16
    assert logger.tot_time == 1.25
    assert logger.ep_extras == []
    assert capsys.readouterr().out == ""


def test_rnd_and_wandb_video_semantics(tmp_path, capsys):
    logger = make_logger(tmp_path, backend="wandb", rnd=True)
    logger.process_env_step(
        torch.tensor([2.0, 4.0]),
        torch.ones(2),
        {},
        intrinsic_rewards=torch.tensor([1.0, 3.0]),
    )
    video = tmp_path / "videos" / "episode.mp4"
    video.parent.mkdir()
    video.touch()

    log_iteration(logger)

    assert logger.writer.scalars["Rnd/mean_extrinsic_reward"] == (3.0, 3)
    assert logger.writer.scalars["Rnd/mean_intrinsic_reward"] == (2.0, 3)
    assert logger.writer.scalars["Rnd/weight"] == (0.25, 3)
    assert logger.writer.scalars["Train/mean_reward"] == (5.0, 3)
    assert "Train/mean_reward/time" not in logger.writer.scalars
    assert "Train/mean_episode_length/time" not in logger.writer.scalars
    assert logger.writer.videos == [(video, 3)]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("backend", ["tensorboard", "wandb", "neptune"])
def test_inherited_writer_initialization_model_upload_and_close(tmp_path, monkeypatch, backend):
    modules = {
        "tensorboard": ("torch.utils.tensorboard", "SummaryWriter"),
        "wandb": ("rsl_rl.utils.wandb_utils", "WandbSummaryWriter"),
        "neptune": ("rsl_rl.utils.neptune_utils", "NeptuneSummaryWriter"),
    }
    module_name, writer_name = modules[backend]
    module = ModuleType(module_name)
    setattr(module, writer_name, RecordingWriter)
    monkeypatch.setitem(sys.modules, module_name, module)
    logger = make_logger(tmp_path, backend=backend)
    logger.files_to_upload = ["snapshot.diff"]

    logger.init_logging_writer()
    writer = logger.writer
    logger.save_model("model.pt", 3)
    logger.close()
    logger.close()

    assert writer.kwargs["log_dir"] == str(tmp_path)
    assert writer.kwargs["flush_secs"] == 10
    if backend == "tensorboard":
        assert writer.flush_count == 1
        assert writer.close_count == 1
        assert writer.stop_count == 0
        assert writer.models == []
    else:
        assert writer.stop_count == 1
        assert writer.close_count == 0
        assert writer.configs == [(logger.env_cfg, logger.cfg)]
        assert writer.files == ["snapshot.diff"]
        assert writer.models == [("model.pt", 3)]
    assert logger.writer is None


@pytest.mark.parametrize("disabled", ["no_log_dir", "distributed_worker"])
def test_disabled_logging_is_quiet_and_close_is_safe(tmp_path, capsys, disabled):
    logger = make_logger(tmp_path, distributed=disabled == "distributed_worker")
    if disabled == "no_log_dir":
        logger.log_dir = None
    logger.init_logging_writer()

    log_iteration(logger)
    logger.close()

    assert logger.writer is None
    assert logger.tot_timesteps == 0
    assert capsys.readouterr().out == ""


def test_close_before_initialization(tmp_path):
    logger = make_logger(tmp_path)
    del logger.writer
    logger.close()


@pytest.mark.parametrize("backend,method", [("tensorboard", "flush"), ("tensorboard", "close"), ("wandb", "stop")])
def test_close_propagates_writer_failure(tmp_path, monkeypatch, backend, method):
    logger = make_logger(tmp_path, backend=backend)
    writer = logger.writer
    failure = RuntimeError("writer failure")

    def fail():
        raise failure

    monkeypatch.setattr(writer, method, fail)
    with pytest.raises(RuntimeError) as exc:
        logger.close()

    assert exc.value is failure
    assert logger.writer is None
    if method == "flush":
        assert writer.close_count == 1
