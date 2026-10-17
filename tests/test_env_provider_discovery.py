# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Installed-style environment discovery across fresh and spawned interpreters."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.integration
def test_external_environment_provider_is_discovered_in_fresh_and_spawned_python(tmp_path: Path) -> None:
    # Real distribution metadata exercises importlib discovery without an installer
    # or importing/registering the provider in the pytest process.
    (tmp_path / "external_env_provider.py").write_text(
        textwrap.dedent(
            """\
            from motrix_env_core import registry
            from motrix_env_core.base import EnvCfg
            from motrix_env_core.config import configclass
            from motrix_env_core.config.scene import SceneCfg

            calls = 0

            @configclass
            class ExternalEnvCfg(EnvCfg):
                scene: SceneCfg = SceneCfg()
                provider_value: str = "discovered externally"

            def register():
                global calls
                calls += 1
                registry.register_env_config("external-provider-discovery-test", ExternalEnvCfg)
                # Public queries are legal during a registration callback and
                # must not recursively invoke the provider again.
                assert registry.contains("external-provider-discovery-test")
            """
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "external_env_provider-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: external-env-provider\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata / "entry_points.txt").write_text(
        "[motrix_env_core.envs]\nexternal-discovery-test = external_env_provider:register\n",
        encoding="utf-8",
    )
    script = tmp_path / "probe_discovery.py"
    script.write_text(
        textwrap.dedent(
            """\
            import json
            import multiprocessing
            import os
            import sys

            def probe(first_query):
                assert "external_env_provider" not in sys.modules
                from motrix_env_core import registry
                assert "external_env_provider" not in sys.modules
                name = "external-provider-discovery-test"
                if first_query == "contains":
                    assert registry.contains(name)
                else:
                    assert registry.make_env_config(name).provider_value == "discovered externally"
                cfg = registry.make_env_config(name)
                assert registry.contains(name)
                assert registry.list_registered_envs()[name]["config_class"] == type(cfg).__name__
                assert cfg.provider_value == "discovered externally"
                calls = sys.modules["external_env_provider"].calls
                assert calls == 1
                return {"pid": os.getpid(), "value": cfg.provider_value, "calls": calls}

            def child(connection):
                connection.send(probe("make_env_config"))
                connection.close()

            if __name__ == "__main__":
                parent = probe("contains")
                context = multiprocessing.get_context("spawn")
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=child, args=(sender,))
                process.start()
                sender.close()
                try:
                    assert receiver.poll(45), "spawned child did not report discovery"
                    spawned = receiver.recv()
                    process.join(10)
                    assert process.exitcode == 0
                finally:
                    if process.is_alive():
                        process.terminate()
                    process.join()
                    receiver.close()
                print(json.dumps({"parent": parent, "spawned": spawned}))
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["parent"]["pid"] != report["spawned"]["pid"]
    for process_report in report.values():
        assert process_report["value"] == "discovered externally"
        assert process_report["calls"] == 1
