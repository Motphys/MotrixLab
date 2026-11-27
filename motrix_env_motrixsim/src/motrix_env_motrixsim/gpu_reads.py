# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile CUDA reads into live views and captured derived projections."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
import warp as wp

from motrix_env_core.sim.read import PhysicsReadProgram, SimDataQuery
from motrix_env_motrixsim._gpu_primitives import GpuContactProjector
from motrix_env_motrixsim.read_compiler import MotrixReadCompiler, canonical_reads

if TYPE_CHECKING:
    from motrix_env_motrixsim.gpu_runtime import MotrixCudaBackend

Projection = Callable[[tuple[torch.Tensor, ...]], torch.Tensor]


def _column_slice(indices) -> slice | None:
    columns = tuple(int(index) for index in indices)
    if not columns:
        return slice(0, 0)
    first = columns[0]
    if columns == tuple(range(first, first + len(columns))):
        return slice(first, first + len(columns))
    return None


class CudaReadCompiler(MotrixReadCompiler):
    """Resolve names and bind device storage once for one SceneData instance."""

    def __init__(self, backend: MotrixCudaBackend, queries: Mapping[str, SimDataQuery]):
        self.backend = backend
        self._model = backend.host_model
        self.queries = dict(queries)
        self.fields: dict[str, Projection] = {}
        self.views: dict[str, torch.Tensor] = {}
        self.sources: list[Callable] = []
        self.sensors: dict[str, Projection] = {}
        self.native_sensors: dict[str, torch.Tensor] = {}
        self.poses: dict[tuple[str, str], torch.Tensor] = {}
        self.contacts = GpuContactProjector(backend)
        self.link_poses = None
        self.velocity_sources = []

    def build(self) -> PhysicsReadProgram:
        queries, self.aliases = canonical_reads(self.queries)
        with self.backend.physics_stream():
            return self.compile(queries)

    def _build_program(self) -> PhysicsReadProgram:
        return CudaReads(self)

    def _source(self, fetch: Callable) -> Projection:
        index = len(self.sources)
        self.sources.append(fetch)
        return lambda values: values[index]

    def _joint_columns(self, key, indices, target):
        span = _column_slice(indices)
        if span is not None:
            self.views[key] = target[:, span]
        else:
            columns = torch.as_tensor(indices, dtype=torch.long, device=target.device)
            self.fields[key] = lambda values: target[:, columns]

    def compile_body_joint_position(self, key: str, body: str) -> None:
        obj = self._get_body(self._validate_name(body, key), key)
        self._joint_columns(key, obj.get_dof_pos_indices(False), self.backend.position)

    def compile_body_joint_velocity(self, key: str, body: str) -> None:
        obj = self._get_body(self._validate_name(body, key), key)
        self._joint_columns(key, obj.get_dof_vel_indices(False), self.backend.velocity)

    def compile_joints_position(self, key: str, joints: tuple[str, ...]) -> None:
        resolved = self._one_dof_joints(key, joints, position=True)
        self._joint_columns(key, [joint.dof_pos_index for joint in resolved], self.backend.position)

    def compile_joints_velocity(self, key: str, joints: tuple[str, ...]) -> None:
        resolved = self._one_dof_joints(key, joints, position=False)
        self._joint_columns(key, [joint.dof_vel_index for joint in resolved], self.backend.velocity)

    def _link_pose(self, key, links, components):
        if self.link_poses is None:
            self.link_poses = wp.to_torch(self.backend.model.get_link_poses(self.backend.data))
        poses = self.link_poses
        if isinstance(links, str):
            index = self._get_link_index(self._validate_name(links, key), key)
            self.views[key] = poses[:, index, components]
        else:
            indices = self._link_indices(key, links)
            span = _column_slice(indices)
            if span is not None:
                self.views[key] = poses[:, span, components]
            else:
                index = torch.tensor(indices, device=self.backend.device)
                self.fields[key] = lambda values: poses[:, index, components]

    def compile_links_position(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_pose(key, links, slice(3))

    def compile_links_quaternion(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_pose(key, links, slice(3, 7))

    def _link_velocity(self, key, links, method):
        single = isinstance(links, str)
        names = (links,) if single else links
        self._link_indices(key, names)
        arrays = tuple(method(self.backend.model.get_link(name))(self.backend.data) for name in names)
        self.velocity_sources.extend(arrays)
        velocities = tuple(wp.to_torch(array) for array in arrays)
        if single:
            self.views[key] = velocities[0]
        elif len(velocities) == 1:
            self.views[key] = velocities[0][:, None, :]
        else:
            # SDK views can be slices of one packed per-world allocation. Use
            # the SDK's actual pointers/strides to distinguish those from
            # independently materialized per-link results.
            first, second = arrays[:2]
            link_stride = second.ptr - first.ptr
            row_stride, component_stride = first.strides
            span = (len(arrays) - 1) * link_stride + velocities[0].shape[-1] * component_stride
            if (
                link_stride > 0
                and span <= row_stride
                and all(
                    array.strides == first.strides and array.ptr == first.ptr + i * link_stride
                    for i, array in enumerate(arrays)
                )
            ):
                packed = wp.array(
                    ptr=first.ptr,
                    shape=(first.shape[0], len(arrays), first.shape[1]),
                    strides=(row_stride, link_stride, component_stride),
                    dtype=first.dtype,
                    device=first.device,
                )
                self.views[key] = wp.to_torch(packed)
            else:
                self.fields[key] = lambda values: torch.stack(velocities, dim=1)

    def compile_links_linear_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_velocity(key, links, lambda link: link.get_linear_velocity)

    def compile_links_angular_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_velocity(key, links, lambda link: link.get_angular_velocity)

    def compile_links_net_contact_force(self, key: str, links: str | tuple[str, ...]) -> None:
        single = isinstance(links, str)
        indices = self._link_indices(key, (links,) if single else links)
        source = self._source(self.contacts.link_contact_reader(indices))
        self.fields[key] = (lambda values: source(values)[:, 0]) if single else source

    def compile_body_link_net_contact_force(self, key: str, body: str, exclude_links: tuple[str, ...]) -> None:
        body = self._get_body(self._validate_name(body, key), key)
        excluded = set(self._validate_names(exclude_links, key, "excluded links", allow_empty=True))
        links = body.links
        unknown = excluded.difference(link.name for link in links)
        if unknown:
            raise KeyError(f"Simulator query {key!r} excludes unknown body links: {sorted(unknown)}.")
        self.fields[key] = self._source(
            self.contacts.link_contact_reader([link.index for link in links if link.name not in excluded])
        )

    def compile_sensor_values(self, key: str, sensors: tuple[str, ...]) -> None:
        for name in self._validate_names(sensors, key, "sensors"):
            if name not in self.sensors:
                fetch = self.contacts.sensor_reader(name)
                if fetch is None:
                    if name not in self._model.sensors.names:
                        raise KeyError(f"Simulator query {key!r} references unknown sensor {name!r}.")
                    view = wp.to_torch(self.backend.model.get_sensor_value(name, self.backend.data))
                    self.native_sensors[name] = view
                    self.sensors[name] = lambda values, view=view: view
                else:
                    self.sensors[name] = self._source(fetch)
        if len(sensors) == 1 and sensors[0] in self.native_sensors:
            self.views[key] = self.native_sensors[sensors[0]]
            return
        sources = tuple(self.sensors[name] for name in sensors)
        self.fields[key] = (
            sources[0] if len(sources) == 1 else lambda values: torch.cat([s(values) for s in sources], dim=1)
        )

    def _pose(self, key, kind, name, components):
        token = (kind, name)
        if token not in self.poses:
            if kind == "geom":
                self._get_geom(self._validate_name(name, key), key)
                obj = self.backend.model.get_geom(name)
            else:
                self._get_site(self._validate_name(name, key), key)
                obj = self.backend.model.get_site(name)
            self.poses[token] = wp.to_torch(obj.get_pose(self.backend.data))
        pose = self.poses[token]
        self.views[key] = pose[:, components]

    def compile_site_position(self, key: str, site: str) -> None:
        self._pose(key, "site", site, slice(3))

    def compile_site_quaternion(self, key: str, site: str) -> None:
        self._pose(key, "site", site, slice(3, 7))

    def compile_geom_position(self, key: str, geom: str) -> None:
        self._pose(key, "geom", geom, slice(3))

    def compile_geom_quaternion(self, key: str, geom: str) -> None:
        self._pose(key, "geom", geom, slice(3, 7))

    def compile_geom_linear_velocity(self, key: str, geom: str) -> None:
        self._get_geom(self._validate_name(geom, key), key)
        obj = self.backend.model.get_geom(geom)
        velocity = wp.to_torch(obj.get_linear_velocity(self.backend.data))
        self.views[key] = velocity

    def compile_geom_pair_colliding(self, key: str, pairs: tuple[tuple[str, str], ...]) -> None:
        def colliders(name):
            return self.contacts.geom_colliders(self._get_geom(name, key).index)

        groups = [
            [(a, b) for a in colliders(left) for b in colliders(right)]
            for left, right in self._validate_name_pairs(pairs, key)
        ]
        self.fields[key] = self._source(self.contacts.pair_reader(groups))

    def compile_dof_position(self, key: str) -> None:
        self.views[key] = self.backend.position

    def compile_dof_velocity(self, key: str) -> None:
        self.views[key] = self.backend.velocity

    def compile_actuator_ctrl(self, key: str) -> None:
        self.views[key] = self.backend.control


class CudaReads(PhysicsReadProgram):
    def __init__(self, compiler: CudaReadCompiler):
        backend = compiler.backend
        self._backend = backend
        self._queries = compiler.queries
        self._velocity_sources = tuple(compiler.velocity_sources)
        self._world_mask = torch.ones_like(backend._all_worlds)
        full_mask = wp.from_torch(backend._all_worlds)
        masked = wp.from_torch(self._world_mask)
        sources = tuple(compiler.sources)
        prepare_contacts = compiler.contacts.prepare_rows

        def fetch(mask):
            if prepare_contacts is not None:
                prepare_contacts(mask)
            return tuple(source(mask) for source in sources)

        values = fetch(full_mask)
        outputs = {key: project(values).to(torch.float32).clone() for key, project in compiler.fields.items()}
        self._arena_bytes = sum(out.numel() * out.element_size() for out in outputs.values())
        views = compiler.views | outputs
        self._outputs = {key: views[canonical] for key, canonical in compiler.aliases.items()}
        fields = tuple((outputs[key], project) for key, project in compiler.fields.items())
        # CUDA graphs retain raw addresses. Keep the indexed projections and
        # the Warp launch closures alive along with their input tensors.
        self._graph_inputs = (fields, fetch, values, full_mask, masked)

        def publish(values):
            for output, project in fields:
                output.copy_(project(values))

        def publish_masked(values, mask):
            for output, project in fields:
                rows = mask.reshape((-1,) + (1,) * (output.ndim - 1))
                output.copy_(torch.where(rows, project(values), output))

        self._graph = None
        self._masked_graph = None
        if fields:
            options = {"triton.cudagraphs": False}
            publish = torch.compile(publish, dynamic=False, options=options)
            publish_masked = torch.compile(publish_masked, dynamic=False, options=options)
            publish(values)
            publish_masked(values, self._world_mask)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=backend._torch_stream):
                fetch(full_mask)
                publish(values)
            self._masked_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._masked_graph, stream=backend._torch_stream):
                fetch(masked)
                publish_masked(values, self._world_mask)

    @property
    def arena_bytes(self) -> int:
        return self._arena_bytes

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._queries)

    def query(self, key: str) -> SimDataQuery:
        return self._queries[key]

    def view(self, key: str) -> torch.Tensor:
        return self._outputs[key]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            if (
                not isinstance(env_ids, torch.Tensor)
                or env_ids.ndim != 1
                or env_ids.dtype not in (torch.int32, torch.int64)
            ):
                raise TypeError("Partial simulator read env_ids must be a one-dimensional integer tensor.")
            if env_ids.numel() == 0:
                return
            env_ids = env_ids.to(self._backend.device, dtype=torch.long)
            with self._backend.physics_stream():
                self.execute_masked(self._backend.world_mask(env_ids))
        elif self._graph is not None:
            with self._backend.physics_stream():
                self._graph.replay()

    def execute_masked(self, world_mask: torch.Tensor) -> None:
        if self._masked_graph is not None:
            with self._backend.physics_stream():
                self._world_mask.copy_(world_mask)
                self._masked_graph.replay()
