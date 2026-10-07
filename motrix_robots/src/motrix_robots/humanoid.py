# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import RobotCfg


@configclass(kw_only=True)
class HumanoidRobotCfg(RobotCfg):
    """A model-file-backed robot with left and right foot-link semantics."""

    left_foot_link_name: str
    right_foot_link_name: str

    def validate(self, name: str) -> None:
        super().validate(name)
        foot_link_names = (self.left_foot_link_name, self.right_foot_link_name)
        if not all(foot_link_names):
            raise ValueError("Humanoid foot link names must not be empty")
        if len(set(foot_link_names)) != len(foot_link_names):
            raise ValueError("Humanoid foot link names must be unique")

    @property
    def resolved_foot_link_names(self) -> tuple[str, str]:
        """Left and right foot link names after applying the robot name affixes."""

        return (
            self.resolve_name(self.left_foot_link_name),
            self.resolve_name(self.right_foot_link_name),
        )


__all__ = ["HumanoidRobotCfg"]
