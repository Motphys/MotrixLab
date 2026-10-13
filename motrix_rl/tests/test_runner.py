# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from motrix_env_core import registry as env_registry
from motrix_rl import frameworks, runner, runs, utils
from motrix_rl.config import CheckpointConfig, LoggingConfig, TaskMeta, TrainConfig
from motrix_rl.frameworks import RlFramework
from motrix_rl.metrics import JSONL_FILENAME

RUNNING_FIXTURE = Path(__file__).parent / "fixtures" / "metrics_v1" / "motrixlab_running.json"


@dataclass
class DemoCfg:
    value: int = 1


@pytest.fixture
def demo_framework(monkeypatch):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    monkeypatch.setattr(env_registry, "contains", lambda env_name: env_name == "demo-env")
    monkeypatch.setattr(utils, "get_device_supports", lambda: utils.DeviceSupports(torch=True))
    monkeypatch.delenv("MOTRIX_METRICS_URL", raising=False)
    monkeypatch.delenv("MOTRIX_METRICS_JSONL", raising=False)
    monkeypatch.delenv("MOTRIX_METRICS_RUN_ID", raising=False)
    provider = _RunnerProvider()
    frameworks.register_framework(_RunnerFramework("external", (provider,)))
    return provider


def _train_request(runs_root):
    return runner.TrainRequest(
        config=TrainConfig(
            task=TaskMeta(env="demo-env", rllib="external", algo="demo", train_backend="custom"),
            num_envs=2,
            play_num_envs=3,
            seed=7,
            algo=DemoCfg(),
            logging=LoggingConfig(backend="tensorboard", interval=20),
            checkpoint=CheckpointConfig(interval=50),
        ),
        runs_root=runs_root,
    )


def test_runner_train_creates_run_and_returns_result(demo_framework, tmp_path):
    provider = demo_framework
    result = runner.train(_train_request(tmp_path))

    assert provider.trainer is not None
    assert provider.trainer.train_called
    assert result.run.metadata.env_name == "demo-env"
    assert result.run.metadata.rllib == "external"
    assert result.run.metadata.train_backend == "custom"
    assert result.run.metadata.algo == "demo"
    assert result.run.metadata.checkpoint_format == "pt"
    assert result.run.metadata.seed == 7
    assert result.run.run_dir.parent == tmp_path / "demo-env" / "external" / "custom" / "demo"
    assert provider.trainer.context.rl_cfg == DemoCfg()
    assert provider.trainer.context.num_envs == 2
    assert provider.trainer.context.play_num_envs == 3
    assert provider.trainer.context.seed == 7
    assert provider.trainer.context.logging == LoggingConfig(backend="tensorboard", interval=20)
    assert provider.trainer.context.checkpoint == CheckpointConfig(interval=50)
    assert runs.task_config_path(result.run.run_dir).is_file()

    handle = runner.create_run_handle(
        result.run,
        cfg_override={"value": 2},
        play_num_envs=6,
    )
    assert handle.trainer.context.num_envs == 2
    assert handle.trainer.context.play_num_envs == 6
    assert handle.trainer.context.rl_cfg == DemoCfg(value=2)
    assert handle.trainer.context.logging == LoggingConfig(backend="tensorboard", interval=20)
    assert handle.trainer.context.checkpoint == CheckpointConfig(interval=50)
    assert not provider.trainer.context.metrics.enabled
    assert not (result.run.run_dir / JSONL_FILENAME).exists()


def test_training_reports_live_metrics_but_play_does_not(demo_framework, monkeypatch, tmp_path):
    received = []

    class Ingest(BaseHTTPRequestHandler):
        def do_POST(self):
            body = gzip.decompress(self.rfile.read(int(self.headers["Content-Length"])))
            received.extend(json.loads(body)["records"])
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Ingest)
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    monkeypatch.setenv("MOTRIX_METRICS_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("MOTRIX_METRICS_JSONL", "1")
    monkeypatch.setenv("MOTRIX_METRICS_RUN_ID", "demo-run")
    monkeypatch.setenv("MOTRIX_METRICS_PROJECT", "demo")
    monkeypatch.setenv("MOTRIX_METRICS_GROUP", "demo__1")
    monkeypatch.setenv("MOTRIX_METRICS_META", '{"flow_run_id": "1"}')
    try:
        result = runner.train(_train_request(tmp_path))
        metadata = (result.run.run_dir / runs.METADATA_FILENAME).read_text()
        train_metrics = demo_framework.trainer.context.metrics
        train_metrics.join(5)  # training closed the logger without waiting for the sinks
        play = runner.create_run_handle(result.run)
    finally:
        server.shutdown()
        server.server_close()

    records = [json.loads(line) for line in (result.run.run_dir / JSONL_FILENAME).read_text().splitlines()]
    assert records == received
    assert [record["data"]["state"] for record in records] == ["running", "finished"]
    running = records[0]["data"]
    expected = json.loads(RUNNING_FIXTURE.read_text())["body"]["records"][0]["data"]
    assert set(running) == set(expected)
    assert running["config"] == runs.task_config_snapshot(_train_request(tmp_path).config)
    assert running["run_dir"] == str(result.run.run_dir)
    assert train_metrics.run_id == "demo-run"
    assert json.loads(metadata)["live"] == {"run_id": "demo-run"}

    assert not play.metrics.enabled
    assert not play.trainer.context.metrics.enabled
    assert (result.run.run_dir / runs.METADATA_FILENAME).read_text() == metadata


class _RunnerTrainer(frameworks.TrainerBase):
    def __init__(self, context: frameworks.TrainerContext[DemoCfg]) -> None:
        self.context = context
        self.train_called = False

    def train(self) -> None:
        self.train_called = True

    def play(self, policy: str) -> None:
        raise NotImplementedError


class _RunnerFramework(RlFramework):
    def __init__(self, name: str, providers) -> None:
        self._name = name
        super().__init__(providers)

    @property
    def name(self) -> str:
        return self._name


@dataclass
class _RunnerProvider(frameworks.AgentProvider[DemoCfg]):
    config_type = DemoCfg

    trainer: _RunnerTrainer | None = None

    @property
    def train_backend(self) -> str:
        return "custom"

    @property
    def agent_name(self) -> str:
        return "demo"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(
        self,
        context: frameworks.TrainerContext[DemoCfg],
    ) -> frameworks.TrainerBase:
        self.trainer = _RunnerTrainer(context)
        return self.trainer
