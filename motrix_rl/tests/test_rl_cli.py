# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command boundary contracts independent of Hydra and simulator runtime."""

import sys

import pytest

from motrix_rl.cli import import_env_modules, run_cli
from motrix_rl.commands.export import main as export_main


@pytest.mark.parametrize("format_name", ["onnx", "deploy"])
def test_export_dispatch_preserves_hydra_arguments_and_restores_argv(monkeypatch, format_name):
    import importlib

    module = importlib.import_module(f"motrix_rl.commands.export_{format_name}")
    observed = []

    def command():
        observed.append(sys.argv.copy())
        raise RuntimeError("export failed")

    monkeypatch.setattr(module, "main", command)
    original = ["mxexport", format_name, "run_dir=example", "--config-dir", "/project/configs"]
    monkeypatch.setattr(sys, "argv", original)
    with pytest.raises(RuntimeError, match="export failed"):
        export_main()
    assert observed == [[f"mxexport {format_name}", *original[2:]]]
    assert sys.argv is original


@pytest.mark.parametrize("args,code", [([], 0), (["--help"], 0), (["invalid"], 2)])
def test_export_requires_an_explicit_format(monkeypatch, capsys, args, code):
    monkeypatch.setattr(sys, "argv", ["mxexport", *args])
    with pytest.raises(SystemExit) as error:
        export_main()
    assert error.value.code == code
    output = capsys.readouterr()
    assert "onnx" in output.out + output.err
    assert "deploy" in output.out + output.err


def test_cli_uses_local_configs_and_restores_arguments(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    original = ["mxtrain", "task=example/ppo"]
    monkeypatch.setattr(sys, "argv", original)
    observed = []
    run_cli(lambda: observed.append(sys.argv.copy()))
    assert observed == [["mxtrain", "--config-dir", str(tmp_path / "configs"), "task=example/ppo"]]
    assert sys.argv is original


@pytest.mark.parametrize("args", [["--config-dir", "/other"], ["--config-dir=/other"], ["-cp", "/other"]])
def test_explicit_hydra_directory_bypasses_local_default(monkeypatch, tmp_path, args):
    (tmp_path / "configs").mkdir()
    monkeypatch.chdir(tmp_path)
    original = ["mxtrain", *args]
    monkeypatch.setattr(sys, "argv", original)
    observed = []
    run_cli(lambda: observed.append(sys.argv.copy()))
    assert observed == [original]


def test_cli_does_not_search_parent_directories(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    original = ["mxplay", "--cfg", "job"]
    monkeypatch.setattr(sys, "argv", original)
    observed = []
    run_cli(lambda: observed.append(sys.argv.copy()))
    assert observed == [original]


def test_environment_module_imports_register_and_propagate_failures(monkeypatch, tmp_path):
    (tmp_path / "external_env_registration.py").write_text("registered = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        import_env_modules(["external_env_registration"])
        assert sys.modules["external_env_registration"].registered
        with pytest.raises(ModuleNotFoundError, match="missing_env_registration"):
            import_env_modules(["missing_env_registration"])
    finally:
        sys.modules.pop("external_env_registration", None)
