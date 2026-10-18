# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command boundary contracts independent of Hydra and simulator runtime."""

import sys

import pytest

from motrix_rl.cli import run_cli


def test_cli_uses_local_configs_and_restores_arguments(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    original = ["mxlab train", "task=example/ppo"]
    monkeypatch.setattr(sys, "argv", original)
    observed = []
    run_cli(lambda: observed.append(sys.argv.copy()))
    assert observed == [["mxlab train", "--config-dir", str(tmp_path / "configs"), "task=example/ppo"]]
    assert sys.argv is original


@pytest.mark.parametrize("args", [["--config-dir", "/other"], ["--config-dir=/other"], ["-cp", "/other"]])
def test_explicit_hydra_directory_bypasses_local_default(monkeypatch, tmp_path, args):
    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    original = ["mxlab train", *args]
    monkeypatch.setattr(sys, "argv", original)
    observed = []
    run_cli(lambda: observed.append(sys.argv.copy()))
    assert observed == [original]


def test_cli_does_not_search_parent_directories(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    original = ["mxlab play", "--cfg", "job"]
    monkeypatch.setattr(sys, "argv", original)
    observed = []
    run_cli(lambda: observed.append(sys.argv.copy()))
    assert observed == [original]
