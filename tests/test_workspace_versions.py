# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Workspace release version contracts."""

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def _project_field(path: Path, field: str) -> str:
    match = re.search(rf'^{field} = "(.+)"$', path.read_text(), re.MULTILINE)
    assert match is not None, f"Missing {field} in {path}"
    return match.group(1)


def _workspace_pyprojects() -> list[tuple[Path, str, str]]:
    root_pyproject = ROOT / "pyproject.toml"
    members_match = re.search(
        r"\[tool\.uv\.workspace\]\s*members\s*=\s*\[(.*?)\]",
        root_pyproject.read_text(),
        re.DOTALL,
    )
    assert members_match is not None, "Missing [tool.uv.workspace] members in root pyproject.toml"
    members = re.findall(r'"([^"]+)"', members_match.group(1))
    projects = [
        (
            root_pyproject,
            _project_field(root_pyproject, "name"),
            _project_field(root_pyproject, "version"),
        )
    ]
    for member in members:
        pyproject = ROOT / member / "pyproject.toml"
        projects.append((pyproject, _project_field(pyproject, "name"), _project_field(pyproject, "version")))
    return projects


def _workspace_projects() -> dict[str, str]:
    return {name: version for _, name, version in _workspace_pyprojects()}


def _lock_versions() -> dict[str, str]:
    lock = (ROOT / "uv.lock").read_text()
    return dict(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock))


def _toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomlkit

        return tomlkit.loads(path.read_text())
    with path.open("rb") as file:
        return tomllib.load(file)


def test_workspace_packages_share_one_version():
    versions = set(_workspace_projects().values())
    assert len(versions) == 1, f"Workspace package versions diverged: {sorted(versions)}"


def test_lockfile_records_workspace_versions():
    lock_versions = _lock_versions()
    for name, version in _workspace_projects().items():
        assert lock_versions.get(name) == version, (
            f"uv.lock has {name} {lock_versions.get(name)!r}, pyproject.toml declares {version!r}; run `uv lock`"
        )


def test_workspace_member_dependencies_are_pinned_to_workspace_version():
    projects = _workspace_projects()
    workspace_names = {canonicalize_name(name) for name in projects}

    for pyproject, project_name, version in _workspace_pyprojects():
        document = _toml(pyproject)
        project = document["project"]
        requirements = list(project.get("dependencies", []))
        for optional in project.get("optional-dependencies", {}).values():
            requirements.extend(optional)
        for group in document.get("dependency-groups", {}).values():
            requirements.extend(item for item in group if isinstance(item, str))

        for requirement_text in requirements:
            requirement = Requirement(requirement_text)
            requirement_name = canonicalize_name(requirement.name)
            if requirement_name == canonicalize_name(project_name):
                # Self-dependencies only activate another extra of the same
                # distribution and do not need a version specifier.
                continue
            if requirement_name not in workspace_names:
                continue
            assert str(requirement.specifier) == f"=={version}", (
                f"{pyproject} must pin {requirement.name} to the workspace version "
                f"{version}; got {requirement.specifier}"
            )
