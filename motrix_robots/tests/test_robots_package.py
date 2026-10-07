# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulation model package boundaries and installed asset closure."""

import subprocess
import sys
import textwrap


def test_model_imports_do_not_read_or_scan_assets() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                def check_asset_access(event, args):
                    if event in ('open', 'os.scandir', 'os.listdir'):
                        path = str(args[0]).replace('\\\\', '/')
                        assert '/motrix_robots/assets' not in path, (event, path)
                        assert '/motrix_robots/common' not in path, (event, path)

                sys.addaudithook(check_asset_access)
                from motrix_env_core import registry
                import motrix_robots
                from motrix_robots.unitree import UnitreeGo2Robot
                import motrix_robots.registration
                UnitreeGo2Robot()
                assert not registry._robots
                assert not registry._robots_discovered
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_builtin_import_does_not_load_training_or_simulators() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from motrix_robots.unitree import UnitreeGo2Robot; "
            "import sys; "
            "assert not any(name in sys.modules for name in "
            "('motrix_envs', 'motrixsim', 'mujoco', 'torch', 'motrix_deploy', 'unitree_sdk2py')); "
            "UnitreeGo2Robot().validate('robot')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_installed_plugin_discovers_builtin_models_without_explicit_imports() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                from motrix_env_core import registry
                assert 'motrix_robots' not in sys.modules
                expected_types = {
                    'anymal_c': 'AnymalC',
                    'dex-evt': 'DexEvt',
                    'g1-29dof': 'UnitreeG129Dof',
                    'go1': 'UnitreeGo1Robot',
                    'go2': 'UnitreeGo2Robot',
                    'k1': 'BoosterK1',
                    'microduck': 'Microduck',
                }
                # A make lookup alone must discover the installed wheel entry point.
                first = registry.make_robot_config('go2')
                registered = registry.list_registered_robots()
                assert expected_types.keys() <= registered.keys()
                for name, expected_type in expected_types.items():
                    robot = registry.make_robot_config(name)
                    assert type(robot).__name__ == expected_type
                    assert registered[name]['config_class'] == expected_type
                    assert registry.make_robot_config(name) is not robot
                assert registry.make_robot_config('go2') is not first
                assert not any(name in sys.modules for name in (
                    'motrix_envs', 'motrixsim', 'mujoco', 'torch',
                    'motrix_deploy', 'unitree_sdk2py',
                ))
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
