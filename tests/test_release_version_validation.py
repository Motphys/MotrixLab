# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Release workflow version-selection contracts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/validate_release_version.py"
SPEC = spec_from_file_location("validate_release_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validate_release_version = module_from_spec(SPEC)
SPEC.loader.exec_module(validate_release_version)


def test_normal_release_accepts_strictly_newer_prerelease() -> None:
    assert str(validate_release_version.validate("0.4.0", "0.5.0b0", "normal")) == "0.5.0b0"


def test_stable_patch_requires_exact_next_final_patch() -> None:
    assert str(validate_release_version.validate("0.4.0", "0.4.1", "stable-patch")) == "0.4.1"


@pytest.mark.parametrize(
    ("current", "target", "release_kind"),
    [
        ("0.4.0", "0.4.0", "normal"),
        ("0.4.0", "0.4.0.dev1", "normal"),
        ("0.4.0", "0.5.0+local", "normal"),
        ("0.4.0", "0.4.0b0", "stable-patch"),
        ("0.4.0", "0.4.2", "stable-patch"),
        ("0.4.0", "0.5.0", "stable-patch"),
    ],
)
def test_invalid_release_targets_are_rejected(current: str, target: str, release_kind: str) -> None:
    with pytest.raises(ValueError):
        validate_release_version.validate(current, target, release_kind)
