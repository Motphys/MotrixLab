# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI routing, argument ownership, and lazy help contracts."""

import sys
from types import SimpleNamespace

import pytest

from motrixlab import cli


@pytest.mark.parametrize(
    "command,module",
    [
        ("train", "motrix_rl.commands.train"),
        ("play", "motrix_rl.commands.play"),
        ("view", "motrixlab.commands.view"),
        ("export-onnx", "motrix_rl.commands.export_onnx"),
        ("export-deploy", "motrix_rl.commands.export_deploy"),
        ("deploy", "motrix_deploy.cli"),
    ],
)
def test_dispatch_forwards_arguments_and_exit_status(monkeypatch, command, module):
    original = ["mxlab", command, "task=example/ppo", "--config-dir", "/project/configs"]
    monkeypatch.setattr(sys, "argv", original)
    observed = []

    def run():
        observed.append(sys.argv.copy())
        return 7

    def load(name):
        assert name == module
        return SimpleNamespace(main=run)

    monkeypatch.setattr(cli, "import_module", load)
    assert cli.main() == 7
    assert observed == [[f"mxlab {command}", *original[2:]]]
    assert sys.argv is original


def test_dispatch_restores_arguments_on_failure(monkeypatch):
    original = ["mxlab", "export-onnx", "run_dir=example"]
    monkeypatch.setattr(sys, "argv", original)

    def run():
        raise SystemExit(2)

    monkeypatch.setattr(cli, "import_module", lambda name: SimpleNamespace(main=run))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert sys.argv is original


@pytest.mark.parametrize("args,code", [([], 0), (["--help"], 0), (["--version"], 0), (["unknown"], 2)])
def test_top_level_information_never_loads_components(monkeypatch, capsys, args, code):
    monkeypatch.setattr(sys, "argv", ["mxlab", *args])
    monkeypatch.setattr(cli, "version", lambda name: "1.2.3")

    def load(name):
        pytest.fail(f"Top-level command imported {name}")

    monkeypatch.setattr(cli, "import_module", load)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == code
    output = capsys.readouterr()
    assert "mxlab" in output.out + output.err


def test_subcommand_help_belongs_to_component(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mxlab", "train", "--help"])
    observed = []
    monkeypatch.setattr(
        cli, "import_module", lambda name: SimpleNamespace(main=lambda: observed.append(sys.argv.copy()))
    )
    cli.main()
    assert observed == [["mxlab train", "--help"]]
