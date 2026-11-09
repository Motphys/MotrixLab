# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Payload contract for the learner -> collector inference-weight channel.

An :class:`InferenceBundleSpec` declares what one published inference bundle
contains: named flat-parameter sections (one per inference module the
collector runs) and named normalizer stat sections (``mean/std/var/count`` per
normalizer). Like the transition schema, the bundle layout
is a provider-level decision: the seqlock weight channel transports whatever
the spec declares, addressed by name.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Stat buffers an EmpiricalNormalization-style module exposes; a normalizer
#: section is published exactly when the module carries all of them.
NORM_KEYS = ("_mean", "_std", "_var", "count")


@dataclass(frozen=True)
class ModuleSection:
    """One inference module's parameters, transported as one flat segment.

    Args:
        name: Section name; ``publish`` addresses modules by these names.
        param_numel: Total parameter element count of the module.
    """

    name: str
    param_numel: int


@dataclass(frozen=True)
class NormalizerSection:
    """One normalizer's running stats (mean/std/var/count), transported per element."""

    name: str
    dim: int


@dataclass(frozen=True)
class InferenceBundleSpec:
    """The declared payload of one inference-bundle publish.

    Module sections share one fused flat-parameter vector per slot (a single
    copy per publish); normalizer sections share one fused stat row. Section
    order in the tuples is the layout order — consumers that need positional
    access use the spec's ``*_layout`` mappings.
    """

    modules: tuple[ModuleSection, ...]
    normalizers: tuple[NormalizerSection, ...] = ()

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("InferenceBundleSpec requires at least one module section")
        names = [m.name for m in self.modules] + [n.name for n in self.normalizers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"InferenceBundleSpec section names must be unique, got duplicates: {duplicates}")
        for module in self.modules:
            if module.param_numel < 0:
                raise ValueError(f"module section '{module.name}' needs a non-negative param count")
        for normalizer in self.normalizers:
            if normalizer.dim < 1:
                raise ValueError(f"normalizer section '{normalizer.name}' needs a positive dim, got {normalizer.dim}")

    @property
    def params_numel(self) -> int:
        """Fused flat-parameter vector length (sum over module sections)."""
        return sum(module.param_numel for module in self.modules)

    @property
    def normalizer_dim(self) -> int:
        """Fused normalizer stat row length (sum over normalizer sections)."""
        return sum(normalizer.dim for normalizer in self.normalizers)

    @property
    def module_layout(self) -> dict[str, tuple[int, int]]:
        """``name -> (begin, end)`` offsets into the fused parameter vector."""
        layout, offset = {}, 0
        for module in self.modules:
            layout[module.name] = (offset, offset + module.param_numel)
            offset += module.param_numel
        return layout

    @property
    def normalizer_layout(self) -> dict[str, tuple[int, int]]:
        """``name -> (begin, end)`` offsets into the fused stat row."""
        layout, offset = {}, 0
        for normalizer in self.normalizers:
            layout[normalizer.name] = (offset, offset + normalizer.dim)
            offset += normalizer.dim
        return layout
