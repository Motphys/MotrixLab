# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Installed deployment profile compiler discovery and loading contracts."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from motrix_deploy import profile as profile_module
from motrix_deploy.profile import (
    DeploymentProfile,
    build_deployment_profile,
    registered_profile_compilers,
)


@dataclass
class FakeEntryPoint:
    name: str
    value: object
    load_error: Exception | None = None
    loads: int = 0

    def load(self) -> object:
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error
        return self.value


@pytest.fixture
def installed_compilers(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def install(*entries: FakeEntryPoint) -> None:
        def entry_points(*, group: str) -> tuple[FakeEntryPoint, ...]:
            assert group == "motrix_deploy.profiles"
            return entries

        monkeypatch.setattr(profile_module.metadata, "entry_points", entry_points)

    return install


def test_discovery_and_build_load_only_exact_selected_compiler(installed_compilers) -> None:
    expected = cast(DeploymentProfile, object())
    calls: list[str] = []

    def compile_profile(env_name: str) -> DeploymentProfile:
        calls.append(env_name)
        return expected

    selected = FakeEntryPoint("test-profile", compile_profile)
    unrelated = FakeEntryPoint("test-profile-other", None, ImportError("optional dependency missing"))
    installed_compilers(unrelated, selected)

    assert registered_profile_compilers() == ("test-profile", "test-profile-other")
    assert selected.loads == unrelated.loads == 0
    assert build_deployment_profile("test-profile") is expected
    assert calls == ["test-profile"]
    assert selected.loads == 1
    assert unrelated.loads == 0


def test_duplicate_selected_compilers_fail_before_loading(installed_compilers) -> None:
    first = FakeEntryPoint("duplicate-profile", None)
    second = FakeEntryPoint("duplicate-profile", None)
    installed_compilers(first, second)

    with pytest.raises(ValueError, match="Multiple deployment profile compiler plugins provide 'duplicate-profile'"):
        build_deployment_profile("duplicate-profile")

    assert first.loads == second.loads == 0


@pytest.mark.parametrize("names, supported", [((), "none"), (("profile-z", "profile-a"), "profile-a, profile-z")])
def test_unknown_profile_reports_installed_environments(installed_compilers, names, supported) -> None:
    entries = [FakeEntryPoint(name, None) for name in names]
    installed_compilers(*entries)

    with pytest.raises(ValueError) as raised:
        build_deployment_profile("unknown-profile")

    assert str(raised.value) == (
        f"No deployment profile compiler for environment 'unknown-profile'; supported environments: {supported}"
    )
    assert all(entry.loads == 0 for entry in entries)


def test_selected_plugin_load_failure_propagates(installed_compilers) -> None:
    error = ImportError("compiler dependency missing")
    installed_compilers(FakeEntryPoint("broken-profile", None, error))

    with pytest.raises(ImportError) as raised:
        build_deployment_profile("broken-profile")

    assert raised.value is error


def test_compiler_failure_propagates(installed_compilers) -> None:
    error = RuntimeError("invalid training configuration")

    def compile_profile(env_name: str) -> DeploymentProfile:
        raise error

    installed_compilers(FakeEntryPoint("broken-profile", compile_profile))

    with pytest.raises(RuntimeError) as raised:
        build_deployment_profile("broken-profile")

    assert raised.value is error


def test_profile_plugin_must_provide_callable(installed_compilers) -> None:
    installed_compilers(FakeEntryPoint("invalid-profile", object()))

    with pytest.raises(TypeError, match="'invalid-profile' must provide a callable"):
        build_deployment_profile("invalid-profile")
