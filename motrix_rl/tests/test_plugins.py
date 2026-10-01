# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from types import SimpleNamespace

import pytest

from motrix_rl import plugins


def test_core_import_does_not_load_algorithm_plugins():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import motrix_rl; "
            "assert not any(n in sys.modules for n in "
            "('motrix_rl_builtin', 'motrix_rl_skrl', 'motrix_rl_rslrl'))",
        ],
        check=True,
    )


def test_plugin_loading_is_idempotent(monkeypatch):
    calls = []
    entry = SimpleNamespace(name="example", load=lambda: lambda: calls.append("registered"))
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: (entry,))
    monkeypatch.setattr(plugins, "_loaded", set())

    plugins.load_plugins()
    plugins.load_plugins()
    assert calls == ["registered"]


def test_failed_plugin_can_be_retried(monkeypatch):
    calls = []

    def register():
        calls.append("attempt")
        if len(calls) == 1:
            raise RuntimeError("plugin failed")

    entry = SimpleNamespace(name="example", load=lambda: register)
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: (entry,))
    monkeypatch.setattr(plugins, "_loaded", set())

    with pytest.raises(RuntimeError, match="plugin failed"):
        plugins.load_plugins()
    plugins.load_plugins()
    assert calls == ["attempt", "attempt"]
