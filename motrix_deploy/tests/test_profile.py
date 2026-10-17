# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Installed package provider discovery and profile compiler registry contracts."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest

from motrix_deploy import profile as profile_module
from motrix_deploy.profile import (
    DeploymentProfile,
    build_deployment_profile,
    register_profile_compiler,
    registered_profile_compilers,
)


@dataclass
class FakeEntryPoint:
    name: str
    callback: object
    value: str = "test_provider:register"
    load_error: Exception | None = None
    loads: int = 0

    def load(self) -> object:
        self.loads += 1
        if self.load_error is not None:
            raise self.load_error
        return self.callback


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace every registry/discovery state field, preserving any preexisting
    # process state and preventing installed package providers from leaking in.
    monkeypatch.setattr(profile_module, "_profile_compilers", {})
    monkeypatch.setattr(profile_module, "_profile_compilers_discovered", False)
    monkeypatch.setattr(profile_module, "_profile_compilers_discovering", False)
    monkeypatch.setattr(profile_module, "_profile_entry_points", None)
    monkeypatch.setattr(profile_module, "_completed_profile_entry_points", 0)
    monkeypatch.setattr(profile_module.metadata, "entry_points", lambda *, group: ())


@pytest.fixture
def installed_providers(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def install(*entries: FakeEntryPoint) -> None:
        def entry_points(*, group: str) -> tuple[FakeEntryPoint, ...]:
            assert group == "motrix_deploy.profiles"
            return entries

        monkeypatch.setattr(profile_module.metadata, "entry_points", entry_points)

    return install


@pytest.mark.parametrize("first_query", ["list", "build"])
def test_providers_register_multiple_mappings_and_only_selected_compiler_runs(installed_providers, first_query) -> None:
    expected = cast(DeploymentProfile, object())
    calls: list[str] = []
    callbacks: list[str] = []

    def compile_profile(env_name: str) -> DeploymentProfile:
        calls.append(env_name)
        return expected

    def unselected_compiler(env_name: str) -> DeploymentProfile:
        pytest.fail(f"Unselected compiler executed for {env_name}")

    def register_first() -> None:
        callbacks.append("first")
        register_profile_compiler("test-profile", compile_profile)
        register_profile_compiler("test-profile-variant", compile_profile)

    def register_other() -> None:
        callbacks.append("other")
        register_profile_compiler("test-profile-other", unselected_compiler)

    selected = FakeEntryPoint("package-a", register_first)
    unrelated = FakeEntryPoint("package-b", register_other)
    installed_providers(unrelated, selected)

    if first_query == "list":
        assert registered_profile_compilers() == ("test-profile", "test-profile-other", "test-profile-variant")
        assert calls == []
    assert build_deployment_profile("test-profile") is expected
    assert calls == ["test-profile"]
    assert callbacks == ["first", "other"]
    assert registered_profile_compilers() == ("test-profile", "test-profile-other", "test-profile-variant")
    assert build_deployment_profile("test-profile-variant") is expected
    assert calls == ["test-profile", "test-profile-variant"]
    assert selected.loads == unrelated.loads == 1
    assert callbacks == ["first", "other"]


def test_registration_does_not_discover_providers(installed_providers) -> None:
    provider = FakeEntryPoint("not-loaded", None, load_error=ImportError("must stay lazy"))
    installed_providers(provider)

    assert register_profile_compiler("manual-profile", lambda name: cast(DeploymentProfile, object())) is None
    assert provider.loads == 0


def test_reentrant_queries_see_current_registrations_without_rediscovery(installed_providers) -> None:
    expected = cast(DeploymentProfile, object())
    calls: list[str] = []

    def register() -> None:
        calls.append("register")
        register_profile_compiler("reentrant-profile", lambda name: expected)
        assert registered_profile_compilers() == ("reentrant-profile",)
        assert build_deployment_profile("reentrant-profile") is expected

    provider = FakeEntryPoint("reentrant-package", register)
    installed_providers(provider)

    assert registered_profile_compilers() == ("reentrant-profile",)
    assert registered_profile_compilers() == ("reentrant-profile",)
    assert calls == ["register"]
    assert provider.loads == 1


@pytest.mark.parametrize("query", [registered_profile_compilers, lambda: build_deployment_profile("unknown")])
def test_duplicate_provider_names_fail_before_any_provider_load(installed_providers, query) -> None:
    unrelated = FakeEntryPoint("a-package", None)
    first = FakeEntryPoint("duplicate-package", None, value="first:register")
    second = FakeEntryPoint("duplicate-package", None, value="second:register")
    installed_providers(unrelated, first, second)

    with pytest.raises(ValueError, match="Duplicate deployment profile provider name 'duplicate-package'"):
        query()

    assert unrelated.loads == first.loads == second.loads == 0


def test_duplicate_compiler_registration_does_not_overwrite() -> None:
    expected = cast(DeploymentProfile, object())

    def compiler(name: str) -> DeploymentProfile:
        return expected

    register_profile_compiler("duplicate-profile", compiler)

    # Even registering the same callable again is a duplicate.
    for duplicate in (compiler, lambda name: cast(DeploymentProfile, object())):
        with pytest.raises(ValueError, match="'duplicate-profile' is already registered"):
            register_profile_compiler("duplicate-profile", duplicate)

    assert build_deployment_profile("duplicate-profile") is expected


def test_duplicate_compilers_from_different_providers_fail(installed_providers) -> None:
    def register() -> None:
        register_profile_compiler("shared-profile", lambda name: cast(DeploymentProfile, object()))

    installed_providers(FakeEntryPoint("package-a", register), FakeEntryPoint("package-b", register))

    with pytest.raises(ValueError, match="'shared-profile' is already registered"):
        registered_profile_compilers()


@pytest.mark.parametrize("names, supported", [((), "none"), (("profile-z", "profile-a"), "profile-a, profile-z")])
def test_unknown_profile_reports_registered_environments(installed_providers, names, supported) -> None:
    def register() -> None:
        for name in names:
            register_profile_compiler(name, lambda name: cast(DeploymentProfile, object()))

    installed_providers(FakeEntryPoint("package-provider", register))

    with pytest.raises(ValueError) as raised:
        build_deployment_profile("profile")

    assert str(raised.value) == (
        f"No deployment profile compiler for environment 'profile'; registered environments: {supported}"
    )


@pytest.mark.parametrize("failure", ["load", "callback"])
def test_discovery_retries_failed_provider_without_repeating_completed_callbacks(installed_providers, failure) -> None:
    error = ImportError("provider dependency missing")
    completed_calls: list[str] = []
    failing_calls: list[str] = []
    recovered = False

    def register_completed() -> None:
        completed_calls.append("completed")
        register_profile_compiler("completed-profile", lambda name: cast(DeploymentProfile, object()))

    def register_retrying() -> None:
        failing_calls.append("retrying")
        if failure == "callback" and not recovered:
            # Match env discovery: failed callback side effects remain visible.
            register_profile_compiler("partial-profile", lambda name: cast(DeploymentProfile, object()))
            raise error
        register_profile_compiler("recovered-profile", lambda name: cast(DeploymentProfile, object()))

    first = FakeEntryPoint("a-completed", register_completed)
    second = FakeEntryPoint("b-retrying", register_retrying, load_error=error if failure == "load" else None)
    installed_providers(second, first)

    with pytest.raises(ImportError) as raised:
        registered_profile_compilers()
    assert raised.value is error
    assert completed_calls == ["completed"]
    assert first.loads == second.loads == 1

    recovered = True
    second.load_error = None
    names = registered_profile_compilers()
    expected = ("completed-profile", "recovered-profile")
    if failure == "callback":
        expected = ("completed-profile", "partial-profile", "recovered-profile")
    assert names == expected
    assert registered_profile_compilers() == expected
    assert completed_calls == ["completed"]
    assert len(failing_calls) == (2 if failure == "callback" else 1)
    assert first.loads == 1
    assert second.loads == 2


def test_compiler_failure_propagates_without_rediscovering_provider(installed_providers) -> None:
    error = RuntimeError("invalid training configuration")

    def compile_profile(env_name: str) -> DeploymentProfile:
        raise error

    def register() -> None:
        register_profile_compiler("broken-profile", compile_profile)

    provider = FakeEntryPoint("package-provider", register)
    installed_providers(provider)

    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            build_deployment_profile("broken-profile")
        assert raised.value is error
    assert provider.loads == 1


def test_provider_entry_point_must_load_callable_and_can_retry(installed_providers) -> None:
    provider = FakeEntryPoint("invalid-provider", object())
    installed_providers(provider)

    with pytest.raises(TypeError, match="'invalid-provider' must load a callable"):
        registered_profile_compilers()

    provider.callback = lambda: None
    assert registered_profile_compilers() == ()
    assert provider.loads == 2


def test_registered_compiler_must_be_callable() -> None:
    with pytest.raises(TypeError, match="'invalid-profile' must be callable"):
        register_profile_compiler("invalid-profile", cast(Callable[[str], DeploymentProfile], object()))

    assert registered_profile_compilers() == ()
