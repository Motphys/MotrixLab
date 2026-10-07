# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed application configuration for the deployment CLI."""

from dataclasses import dataclass
from typing import Any


@dataclass
class DeployRunConfig:
    """Resolved execution settings with one selected runtime configuration."""

    artifact: str
    runtime: dict[str, Any]
    duration_s: float | None = None
    command: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.artifact:
            raise ValueError("artifact must be a non-empty path")
        kind = self.runtime.get("kind")
        if kind not in {"simulation", "hardware"}:
            raise ValueError(f"runtime.kind must be simulation or hardware, got {kind!r}")
        if kind == "simulation":
            if not isinstance(self.runtime.get("viewer"), bool):
                raise ValueError("runtime.viewer must be bool")
            realtime = self.runtime.get("realtime")
            if realtime is not None and not isinstance(realtime, bool):
                raise ValueError("runtime.realtime must be bool or null")
        backend_name = self.runtime.get("backend")
        if not isinstance(backend_name, str) or not backend_name:
            raise ValueError(f"runtime.backend must be a non-empty string, got {backend_name!r}")

    @property
    def backend_name(self) -> str:
        return self.runtime["backend"]

    @property
    def runtime_options(self) -> dict[str, Any]:
        execution_fields = {"kind", "backend"}
        if self.runtime["kind"] == "simulation":
            execution_fields |= {"viewer", "realtime"}
        return {name: value for name, value in self.runtime.items() if name not in execution_fields}


__all__ = ["DeployRunConfig"]
