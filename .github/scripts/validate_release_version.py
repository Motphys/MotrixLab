# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Validate a release target selected by the release workflow."""

from argparse import ArgumentParser
from typing import Literal

from packaging.version import InvalidVersion, Version

ReleaseKind = Literal["normal", "stable-patch"]


def parse_version(value: str, label: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as error:
        message = f"{label} version {value!r} is not a valid PEP 440 version"
        raise ValueError(message) from error


def validate(current_value: str, target_value: str, release_kind: ReleaseKind) -> Version:
    current = parse_version(current_value, "current")
    target = parse_version(target_value, "target")

    if current.local is not None or current.dev is not None:
        message = f"current release version must not be a dev or local version: {current}"
        raise ValueError(message)
    if target.local is not None or target.dev is not None:
        message = f"target release version must not be a dev or local version: {target}"
        raise ValueError(message)
    if target <= current:
        message = f"target version {target} must be newer than current version {current}"
        raise ValueError(message)

    if release_kind == "stable-patch":
        release = current.release
        if len(release) != 3:
            message = f"stable branch patch releases require a three-part current version: {current}"
            raise ValueError(message)
        expected = Version(f"{release[0]}.{release[1]}.{release[2] + 1}")
        if target != expected or target.pre is not None or target.post is not None:
            message = (
                "stable branch patch releases must use the exact next final patch version: "
                f"expected {expected}, got {target}"
            )
            raise ValueError(message)

    return target


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--release-kind", choices=("normal", "stable-patch"), required=True)
    arguments = parser.parse_args()

    try:
        target = validate(arguments.current, arguments.target, arguments.release_kind)
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")
    print(target)


if __name__ == "__main__":
    main()
