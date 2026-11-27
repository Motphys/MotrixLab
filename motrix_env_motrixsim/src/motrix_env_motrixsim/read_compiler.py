# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile-time model lookups shared by both MotrixSim read programs."""

from collections.abc import Mapping

import motrixsim as mtx

from motrix_env_core.sim.read import SimDataQuery, SimDataQueryCompiler


def canonical_reads(queries: Mapping[str, SimDataQuery]) -> tuple[dict[str, SimDataQuery], dict[str, str]]:
    canonical_queries: dict[str, SimDataQuery] = {}
    canonical_keys: dict[str, str] = {}
    for key, query in queries.items():
        duplicate = next((c for c, q in canonical_queries.items() if q == query), None)
        canonical_keys[key] = key if duplicate is None else duplicate
        if duplicate is None:
            canonical_queries[key] = query
    return canonical_queries, canonical_keys


class MotrixReadCompiler(SimDataQueryCompiler):
    _model: mtx.SceneModel

    def _link_indices(self, key: str, names: tuple[str, ...]) -> tuple[int, ...]:
        link_names = self._validate_names(names, key, "links")
        return tuple(self._get_link_index(name, key) for name in link_names)

    @staticmethod
    def _validate_name(name: str, key: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Simulator query {key!r} names must be non-empty strings.")
        return name

    @classmethod
    def _validate_names(
        cls,
        names: tuple[str, ...],
        key: str,
        kind: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[str, ...]:
        if not isinstance(names, tuple):
            raise TypeError(f"Simulator query {key!r} {kind} must be tuple[str, ...].")
        if not names and not allow_empty:
            raise ValueError(f"Simulator query {key!r} must contain at least one {kind} entry.")
        for name in names:
            cls._validate_name(name, key)
        if len(set(names)) != len(names):
            raise ValueError(f"Simulator query {key!r} contains duplicate {kind}: {names}.")
        return names

    @staticmethod
    def _validate_name_pairs(pairs: tuple[tuple[str, str], ...], key: str) -> tuple[tuple[str, str], ...]:
        if not isinstance(pairs, tuple):
            raise TypeError(f"Simulator query {key!r} geom pairs must be tuple[tuple[str, str], ...].")
        if not pairs:
            raise ValueError(f"Simulator query {key!r} must contain at least one geom pair entry.")
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"Simulator query {key!r} geom pairs must contain two-name tuples.")
            if any(not isinstance(name, str) or not name for name in pair):
                raise TypeError(f"Simulator query {key!r} geom pairs must contain non-empty string names.")
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"Simulator query {key!r} contains duplicate geom pairs: {pairs}.")
        return pairs

    def _one_dof_joints(self, key: str, names: tuple[str, ...], *, position: bool) -> tuple[mtx.Joint, ...]:
        joint_names = self._validate_names(names, key, "joints")
        joints = []
        for name in joint_names:
            joint = self._model.get_joint(name)
            if joint is None:
                raise KeyError(f"Simulator query {key!r} references unknown joint {name!r}.")
            width = joint.num_dof_pos if position else joint.num_dof_vel
            if width != 1:
                quantity = "position" if position else "velocity"
                raise ValueError(
                    f"Simulator query {key!r} joint {name!r} has {width} {quantity} DOFs; exactly one is required."
                )
            joints.append(joint)
        return tuple(joints)

    def _get_body(self, name: str, key: str) -> mtx.Body:
        body = self._model.get_body(name)
        if body is None:
            raise KeyError(f"Simulator query {key!r} references unknown body {name!r}.")
        return body

    def _get_site(self, name: str, key: str) -> mtx.Site:
        site = self._model.get_site(name)
        if site is None:
            raise KeyError(f"Simulator query {key!r} references unknown site {name!r}.")
        return site

    def _get_geom(self, name: str, key: str) -> mtx.Geom:
        geom = self._model.get_geom(name)
        if geom is None:
            raise KeyError(f"Simulator query {key!r} references unknown geom {name!r}.")
        return geom

    def _get_link_index(self, name: str, key: str) -> int:
        link = self._model.get_link(name)
        if link is None:
            raise KeyError(f"Simulator query {key!r} references unknown link {name!r}.")
        return link.index
