# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import RobotCfg


@configclass(kw_only=True)
class QuadrupedLegCfg:
    """Model-local contact reference for one quadruped leg."""

    contact_geom_name: str | None = None

    def validate(self) -> None:
        if self.contact_geom_name == "":
            raise ValueError("Quadruped leg contact geom name must be None or non-empty")


@configclass(kw_only=True)
class QuadrupedLegsCfg:
    """Four named legs in the canonical FL, FR, RL, RR order."""

    front_left: QuadrupedLegCfg
    front_right: QuadrupedLegCfg
    rear_left: QuadrupedLegCfg
    rear_right: QuadrupedLegCfg

    def validate(self) -> None:
        legs = (self.front_left, self.front_right, self.rear_left, self.rear_right)
        for leg in legs:
            leg.validate()

        contact_geom_names = [leg.contact_geom_name for leg in legs if leg.contact_geom_name is not None]
        if len(set(contact_geom_names)) != len(contact_geom_names):
            raise ValueError("Quadruped leg contact geom names must be unique")


@configclass(kw_only=True)
class QuadrupedRobotCfg(RobotCfg):
    """A model-file-backed robot with source-independent quadruped semantics."""

    legs: QuadrupedLegsCfg

    def validate(self, name: str) -> None:
        super().validate(name)
        self.legs.validate()

    @property
    def foot_contact_geom_names(self) -> tuple[str | None, ...]:
        return tuple(
            self.resolve_name(name) if name is not None else None
            for name in (
                self.legs.front_left.contact_geom_name,
                self.legs.front_right.contact_geom_name,
                self.legs.rear_left.contact_geom_name,
                self.legs.rear_right.contact_geom_name,
            )
        )


__all__ = ["QuadrupedLegCfg", "QuadrupedLegsCfg", "QuadrupedRobotCfg"]
