# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Errors shared by deployment contracts and artifact validation."""


class ValidationError(ValueError):
    """A validation failure tied to a stable artifact field path."""

    def __init__(self, path: str, expected: str, actual: object) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(f"{path}: expected {expected}, got {actual!r}")


class ArtifactError(RuntimeError):
    """An artifact cannot be read or written safely."""


class EmergencyStopError(RuntimeError):
    """A physical emergency-stop input requested the backend's safe stop."""


class LieDownRequestedError(RuntimeError):
    """A physical input requested the backend's configured lie-down shutdown."""


class ControlFailure(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)
