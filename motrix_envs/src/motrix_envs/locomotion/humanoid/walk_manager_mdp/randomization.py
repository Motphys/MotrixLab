# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Domain-randomization settings for the humanoid velocity-tracking task."""

from motrix_env_core.config import configclass


@configclass
class WalkRandomizationCfg:
    """Reset-time dynamics randomization ranges for humanoid walking.

    Every enabled item is resampled per lane at each episode reset through the
    sim write program, and the backend keeps the written overrides between
    resets. A degenerate range (``min == max``, or ``(1.0, 1.0)`` / zero
    width) disables that item; when randomization is enabled, all write keys
    are declared so the Numba map schema remains stable, while disabled items
    perform no writes.

    Attributes:
        enabled: Master switch; ``False`` keeps the reset term free of any
            randomization writes.
        kp_scale_range: Multiplicative ``(min, max)`` scale on every
            actuator's nominal kp.
        damping_scale_range: Multiplicative ``(min, max)`` scale on every
            actuator's nominal damping.
        sliding_friction_range: Multiplicative ``(min, max)`` scale on the
            ground geom's nominal friction; ``None`` disables the item.
        link_mass_scale_range: Multiplicative ``(min, max)`` scale for
            non-base link masses.
        base_mass_offset_range: Additive ``(min, max)`` kilogram offset on
            the base link mass.
        base_com_offset_noise: Uniform per-axis offset width ``(x, y, z)``
            in m added to the base link's nominal com.
        joint_pos_scale_range: Multiplicative ``(min, max)`` range for
            reset-time default joint positions.
        root_velocity_range: Uniform ``(min, max)`` range for each reset-time
            root linear and angular velocity component.
        curriculum_steps: Control steps over which the initial-state ranges
            widen from ``curriculum_start`` fraction of their target width to
            the full width; 0 keeps the target ranges from the start.
        curriculum_start: Fraction of the target range width at step 0 when
            ``curriculum_steps`` is positive; must lie within (0, 1].
    """

    enabled: bool = False
    kp_scale_range: tuple[float, float] = (1.0, 1.0)
    damping_scale_range: tuple[float, float] = (1.0, 1.0)
    sliding_friction_range: tuple[float, float] | None = None
    link_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    base_mass_offset_range: tuple[float, float] = (0.0, 0.0)
    base_com_offset_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_pos_scale_range: tuple[float, float] = (1.0, 1.0)
    root_velocity_range: tuple[float, float] = (0.0, 0.0)
    curriculum_steps: int = 0
    curriculum_start: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "kp_scale_range",
            "damping_scale_range",
            "link_mass_scale_range",
            "base_mass_offset_range",
            "joint_pos_scale_range",
            "root_velocity_range",
        ):
            lo, hi = getattr(self, name)
            if not lo <= hi:
                raise ValueError(f"WalkRandomizationCfg.{name} must be (min, max) with min <= max, got {lo, hi}")
        if self.sliding_friction_range is not None:
            lo, hi = self.sliding_friction_range
            if not 0.0 < lo <= hi:
                raise ValueError(f"WalkRandomizationCfg.sliding_friction_range must be 0 < min <= max, got {lo, hi}")
        if any(w < 0.0 for w in self.base_com_offset_noise):
            raise ValueError("WalkRandomizationCfg.base_com_offset_noise widths must be non-negative")
        if self.curriculum_steps < 0:
            raise ValueError(
                f"WalkRandomizationCfg.curriculum_steps must be non-negative, got {self.curriculum_steps!r}"
            )
        if self.curriculum_steps and not 0.0 < self.curriculum_start <= 1.0:
            raise ValueError(
                f"WalkRandomizationCfg.curriculum_start must lie within (0, 1] when curriculum_steps is "
                f"positive, got {self.curriculum_start!r}"
            )
