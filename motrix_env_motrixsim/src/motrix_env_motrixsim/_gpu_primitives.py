# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Version-specific MotrixSim GPU 1.0.0 runtime primitives."""

from typing import Any

import numpy as np
import torch
import warp as wp


@wp.struct
class ResetBuffer:
    values: wp.array2d(dtype=wp.uint8)
    fill: wp.uint8


@wp.kernel
def reset_solver_rows(
    mask: wp.array(dtype=wp.bool),
    position: wp.array2d(dtype=wp.float32),
    initial_position: wp.array(dtype=wp.float32),
    velocity: wp.array2d(dtype=wp.float32),
    control: wp.array2d(dtype=wp.float32),
    buffers: wp.array(dtype=ResetBuffer),
):
    world, lane = wp.tid()
    if not mask[world]:
        return
    for column in range(lane, position.shape[1], 32):
        position[world, column] = initial_position[column]
    for column in range(lane, velocity.shape[1], 32):
        velocity[world, column] = 0.0
    for column in range(lane, control.shape[1], 32):
        control[world, column] = 0.0
    for i in range(buffers.shape[0]):
        entry = buffers[i]
        buf = entry.values
        for column in range(lane, buf.shape[1], 32):
            buf[world, column] = entry.fill


class GpuSolverResetter:
    """Reset step inputs and both warm-start banks for the selected worlds."""

    def __init__(self, model, data, device):
        self._device = device
        self._shape = (data.num_worlds, 32)
        domains = data.low.domains
        # The scene model supplies the defaults independently of live world 0.
        self._initial_position = wp.array(model.compute_init_dof_pos(), dtype=wp.float32, device=device)
        solver = domains["solver"]
        normal = solver.persistent_normal_lambdas
        collision = domains["rigid_collision"]
        warm = solver.persistent_lambdas
        zero_arrays = [
            normal.equality_mask,
            normal.unilateral_mask,
            collision.contact_count,
            collision.overflow,
            solver.constraint_lambdas.equality,
            solver.constraint_lambdas.unilateral,
        ]
        if warm is not None:
            zero_arrays.extend((warm.counts, warm.segment_counts, warm.row_counter))
        descriptors = []
        for array in zero_arrays:
            if array.size:
                descriptors.append(self._descriptor(array, data.num_worlds, 0))
        if warm is not None and warm.tags.size:
            # SDK 1.0.0 initializes both int32 tag banks to -1 (invalid pair).
            descriptors.append(self._descriptor(warm.tags, data.num_worlds, 0xFF))
        self._buffers = wp.array(descriptors, dtype=ResetBuffer, device=device)
        self._position = data.dof_pos
        self._velocity = data.dof_vel
        self._control = data.actuator_ctrls

    def _descriptor(self, array, num_worlds: int, fill: int):
        row_bytes = array.strides[0]
        descriptor = ResetBuffer()
        descriptor.values = wp.array(ptr=array.ptr, shape=(num_worlds, row_bytes), dtype=wp.uint8, device=self._device)
        descriptor.fill = fill
        return descriptor

    def execute(self, warp_mask) -> None:
        wp.launch(
            reset_solver_rows,
            self._shape,
            inputs=[
                warp_mask,
                self._position,
                self._initial_position,
                self._velocity,
                self._control,
                self._buffers,
            ],
            device=self._device,
        )


class GpuStepStatus:
    """Advance physics and publish per-world engine status codes."""

    def __init__(self, backend):
        self._backend = backend
        self._overflow = wp.to_torch(backend.data.low.domains["rigid_collision"].overflow)

    def step(self, substeps: int, result: torch.Tensor | None) -> None:
        backend = self._backend
        for _ in range(substeps):
            backend.model.step(backend.data, stream=backend._stream)
        # Bound magnitudes well below float32 square-overflow as well as NaN/Inf.
        good = (backend.position.abs() < 1e4).all(dim=1) & (backend.velocity.abs() < 1e4).all(dim=1)
        failed = ~good | self._overflow.ne(0)
        if result is not None:
            result.copy_(failed)
        elif failed.any():
            raise RuntimeError("MotrixSim CUDA encountered a non-finite state or contact overflow")


def joint_actuator_slots(model):
    return model.low.actuator_model.joint_slot_by_ctrl_index


@wp.kernel
def recompute_link_rbi(
    world_mask: wp.array(dtype=wp.bool),
    model_rbi: wp.array2d(dtype=wp.float32),
    high_offsets: wp.array(dtype=wp.int32),
    high_indices: wp.array(dtype=wp.int32),
    transforms_r: wp.array(dtype=wp.mat33f),
    transforms_t: wp.array(dtype=wp.vec3f),
    default_mass: wp.array(dtype=wp.float32),
    default_h: wp.array(dtype=wp.vec3f),
    default_shift: wp.array(dtype=wp.mat33f),
    apparent_mass: wp.array2d(dtype=wp.float32),
    apparent_com: wp.array3d(dtype=wp.float32),
    link_rbi: wp.array3d(dtype=wp.float32),
):
    world, low_link = wp.tid()
    if not world_mask[world]:
        return
    start = high_offsets[low_link]
    end = high_offsets[low_link + 1]
    if start == end:
        return
    mass = model_rbi[low_link, 0]
    h = wp.vec3(model_rbi[low_link, 1], model_rbi[low_link, 2], model_rbi[low_link, 3])
    inertia = wp.mat33(
        model_rbi[low_link, 4],
        model_rbi[low_link, 5],
        model_rbi[low_link, 6],
        model_rbi[low_link, 7],
        model_rbi[low_link, 8],
        model_rbi[low_link, 9],
        model_rbi[low_link, 10],
        model_rbi[low_link, 11],
        model_rbi[low_link, 12],
    )
    for k in range(start, end):
        hi = high_indices[k]
        m = apparent_mass[world, hi]
        c_app = wp.vec3(apparent_com[world, hi, 0], apparent_com[world, hi, 1], apparent_com[world, hi, 2])
        c = transforms_r[k] * c_app + transforms_t[k]
        mass = mass + (m - default_mass[k])
        h = h + (m * c - default_h[k])
        c_sq = wp.dot(c, c)
        shift = m * (wp.mat33(c_sq, 0.0, 0.0, 0.0, c_sq, 0.0, 0.0, 0.0, c_sq) - wp.outer(c, c))
        inertia = inertia + (shift - default_shift[k])
    link_rbi[world, low_link, 0] = mass
    link_rbi[world, low_link, 1] = h[0]
    link_rbi[world, low_link, 2] = h[1]
    link_rbi[world, low_link, 3] = h[2]
    for row in range(3):
        for col in range(3):
            link_rbi[world, low_link, 4 + row * 3 + col] = inertia[row, col]


class GpuParameterOverrides:
    """Device-resident link inertia and collider sliding-friction overrides."""

    def __init__(self, backend):
        self._backend = backend
        self._device = backend._warp_device
        self._overrides = backend.data.low.domains["overrides"]
        self._num_worlds = backend.data.num_worlds
        self._link_bound = False
        self._friction_bound = False

    def bind_link_inertia(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._link_bound:
            backend = self._backend
            ov = self._overrides
            model = backend.model
            self._mass_table = wp.to_torch(ov.apparent_link_mass)
            self._com_table = wp.to_torch(ov.apparent_link_com)
            self._default_mass = torch.as_tensor(ov._default_apparent_link_mass, device=backend.device)
            self._default_com = torch.as_tensor(ov._default_apparent_link_com, device=backend.device)
            self._num_low_links = int(ov._model_link_rbi.shape[0])
            if self._num_low_links:
                # RigidBodyInertia packs (mass: f32, h: vec3f, inertia: mat33f) = 13 float32s.
                # Concrete float32 views avoid late dtype=Any template reloads after CUDAGraph capture.
                self._model_rbi = wp.array(
                    ptr=ov._model_link_rbi.ptr,
                    shape=(self._num_low_links, 13),
                    dtype=wp.float32,
                    device=self._device,
                )
                self._link_rbi = wp.array(
                    ptr=ov.link_rbi.ptr,
                    shape=(self._num_worlds, self._num_low_links, 13),
                    dtype=wp.float32,
                    device=self._device,
                )
            groups: list[list[int]] = [[] for _ in range(self._num_low_links)]
            for hi in range(len(model.links)):
                low_idx = model._low_link_index_for_high_link(hi)
                if low_idx is not None:
                    groups[low_idx].append(hi)
            offsets = [0]
            indices: list[int] = []
            rs, ts, dm, dh, ds = [], [], [], [], []
            for low_idx in range(self._num_low_links):
                for hi in groups[low_idx]:
                    rot, pos = model._apparent_to_low_transform(hi)
                    rot64 = np.asarray(rot, dtype=np.float64)
                    t64 = -rot64 @ np.asarray(pos, dtype=np.float64)
                    m0 = float(ov._default_apparent_link_mass[hi])
                    c0 = rot64 @ np.asarray(ov._default_apparent_link_com[hi], dtype=np.float64) + t64
                    s0 = m0 * (np.dot(c0, c0) * np.eye(3) - np.outer(c0, c0))
                    indices.append(hi)
                    rs.append(rot64.astype(np.float32))
                    ts.append(t64.astype(np.float32))
                    dm.append(np.float32(m0))
                    dh.append((m0 * c0).astype(np.float32))
                    ds.append(s0.astype(np.float32))
                offsets.append(len(indices))
            self._num_mapped_high = len(indices)
            if self._num_mapped_high:
                self._high_offsets = wp.array(offsets, dtype=wp.int32, device=self._device)
                self._high_indices = wp.array(indices, dtype=wp.int32, device=self._device)
                self._transforms_r = wp.array(rs, dtype=wp.mat33f, device=self._device)
                self._transforms_t = wp.array(ts, dtype=wp.vec3f, device=self._device)
                self._default_rbi_mass = wp.array(dm, dtype=wp.float32, device=self._device)
                self._default_rbi_h = wp.array(dh, dtype=wp.vec3f, device=self._device)
                self._default_rbi_shift = wp.array(ds, dtype=wp.mat33f, device=self._device)
            self._link_bound = True
        return self._mass_table, self._com_table

    def bind_geom_friction(self, geom_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backend = self._backend
        if not self._friction_bound:
            ov = self._overrides
            self._collider_friction = wp.to_torch(ov.collider_friction)
            self._default_sliding_friction = torch.as_tensor(
                ov._default_collider_friction_values[:, 0], device=backend.device
            )
            self._friction_bound = True
        source_cols: list[int] = []
        collider_cols: list[int] = []
        low_model = backend.host_model.low
        for column, geom_index in enumerate(geom_indices):
            for collider in low_model.get_collider_indices(geom_index) or ():
                source_cols.append(column)
                collider_cols.append(int(collider))
        return (
            self._collider_friction,
            torch.tensor(source_cols, dtype=torch.long, device=backend.device),
            torch.tensor(collider_cols, dtype=torch.long, device=backend.device),
        )

    def restore_defaults(self, mask: torch.Tensor, warp_mask, *, recompute_inertia: bool) -> None:
        if self._friction_bound:
            sliding = self._collider_friction[..., 0]
            sliding.copy_(torch.where(mask[:, None], self._default_sliding_friction, sliding))
        if self._link_bound:
            self._mass_table.copy_(torch.where(mask[:, None], self._default_mass, self._mass_table))
            self._com_table.copy_(torch.where(mask[:, None, None], self._default_com, self._com_table))
            if recompute_inertia:
                self.commit_link_inertia(warp_mask)

    def commit_link_inertia(self, warp_mask) -> None:
        if not self._link_bound or self._num_low_links == 0 or self._num_mapped_high == 0:
            return
        wp.launch(
            recompute_link_rbi,
            (self._num_worlds, self._num_low_links),
            inputs=[
                warp_mask,
                self._model_rbi,
                self._high_offsets,
                self._high_indices,
                self._transforms_r,
                self._transforms_t,
                self._default_rbi_mass,
                self._default_rbi_h,
                self._default_rbi_shift,
                self._overrides.apparent_link_mass,
                self._overrides.apparent_link_com,
                self._link_rbi,
            ],
            device=self._device,
        )
        if self._overrides.needs_link_spatial_inertia:
            self._backend.model._materialize_link_spatial_inertia(self._backend.data)


@wp.kernel
def read_geom_contacts(
    world_mask: wp.array(dtype=wp.bool),
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    limit: int,
    collider_pairs: wp.array2d(dtype=wp.int32),
    normal_report: bool,
    result: wp.array2d(dtype=wp.float32),
):
    world = wp.tid()
    if not world_mask[world]:
        return
    found = wp.int32(0)
    normal = wp.vec3()
    for i in range(wp.min(counts[world], limit)):
        pair = contacts[world, i].pair_index
        for j in range(collider_pairs.shape[0]):
            a = collider_pairs[j, 0]
            b = collider_pairs[j, 1]
            if (pair[0] == a and pair[1] == b) or (pair[0] == b and pair[1] == a):
                if normal_report and found == 0:
                    normal = contacts[world, i].normal
                    if pair[0] != a:
                        normal = -normal
                found += 1
                break
    if normal_report:
        for axis in range(3):
            result[world, axis] = normal[axis]
    else:
        result[world, 0] = float(found)


@wp.kernel
def dense_contact_rows(
    world_mask: wp.array(dtype=wp.bool),
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    slots: wp.array2d(dtype=wp.int32),
    records: wp.array2d(dtype=Any),
    record_counts: wp.array(dtype=wp.int32),
    rows: wp.array3d(dtype=wp.int32),
):
    world, i = wp.tid()
    if not world_mask[world]:
        return
    if i >= counts[world]:
        return
    contact = contacts[world, i]
    for axis in range(6):
        rows[world, i, axis] = -1
    slot = slots[world, i]
    for j in range(wp.min(record_counts[world], records.shape[1])):
        r = records[world, j]
        if int(r.pos) == slot + 1 and r.feature_id == contact.feature_id and int(r.midx) == contact.index_in_manifold:
            rows[world, i, int(r.axis)] = int(r.lineno)


@wp.kernel
def link_contact_wrenches(
    world_mask: wp.array(dtype=wp.bool),
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    rows: wp.array3d(dtype=wp.int32),
    impulses: wp.array2d(dtype=wp.float32),
    poses: wp.array3d(dtype=wp.float32),
    local_com: wp.array3d(dtype=wp.float32),
    high_links: wp.array(dtype=wp.int32),
    low_links: wp.array(dtype=wp.int32),
    inv_dt: float,
    output: wp.array3d(dtype=wp.float32),
):
    world, column = wp.tid()
    if not world_mask[world]:
        return
    link = high_links[column]
    low = low_links[column]
    origin = wp.vec3(poses[world, link, 0], poses[world, link, 1], poses[world, link, 2])
    rotation = wp.quat(poses[world, link, 3], poses[world, link, 4], poses[world, link, 5], poses[world, link, 6])
    com = origin + wp.quat_rotate(
        rotation, wp.vec3(local_com[world, link, 0], local_com[world, link, 1], local_com[world, link, 2])
    )
    force = wp.vec3()
    moment = wp.vec3()
    for i in range(wp.min(counts[world], rows.shape[1])):
        c = contacts[world, i]
        if c.link_index[0] != low and c.link_index[1] != low:
            continue
        n = c.normal
        # motrixsim-gpu 1.0.0 resolves tangent impulses in this world-frame basis.
        reference = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(n[0]) >= 0.9:
            reference = wp.vec3(0.0, 1.0, 0.0)
        t0 = wp.normalize(wp.cross(n, reference))
        t1 = wp.cross(n, t0)
        linear = wp.vec3()
        angular = wp.vec3()
        for axis in range(6):
            row = rows[world, i, axis]
            if row >= 0 and row < impulses.shape[1]:
                direction = n
                if axis == 1 or axis == 4:
                    direction = t0
                elif axis == 2 or axis == 5:
                    direction = t1
                value = direction * impulses[world, row]
                if axis < 3:
                    linear += value
                else:
                    angular += value
        if c.link_index[0] == low:
            force -= linear
            moment -= angular + wp.cross(c.point - com, linear)
        if c.link_index[1] == low:
            force += linear
            moment += angular + wp.cross(c.point + n * c.penetration_depth - com, linear)
    for j in range(3):
        output[world, column, j] = force[j] * inv_dt
        output[world, column, 3 + j] = moment[j] * inv_dt


@wp.kernel
def touch_force(
    world_mask: wp.array(dtype=wp.bool),
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    rows: wp.array3d(dtype=wp.int32),
    impulses: wp.array2d(dtype=wp.float32),
    frames: wp.array(dtype=Any),
    frame_count: int,
    frame_id: int,
    colliders: wp.array(dtype=wp.int32),
    size: wp.vec3,
    box: bool,
    inv_dt: float,
    output: wp.array2d(dtype=wp.float32),
):
    world = wp.tid()
    if not world_mask[world]:
        return
    frame = frames[world * frame_count + frame_id]
    total = wp.float32(0.0)
    for i in range(wp.min(counts[world], rows.shape[1])):
        c = contacts[world, i]
        row = rows[world, i, 0]
        if row < 0 or row >= impulses.shape[1]:
            continue
        for j in range(colliders.shape[0]):
            collider = colliders[j]
            if collider != c.pair_index[0] and collider != c.pair_index[1]:
                continue
            point = c.point
            if collider == c.pair_index[1]:
                point = point + c.normal * c.penetration_depth
            local = wp.transpose(frame.rotation) * (point - frame.translation)
            inside = wp.length_sq(local) <= size[0] * size[0]
            if box:
                inside = wp.abs(local[0]) <= size[0] and wp.abs(local[1]) <= size[1] and wp.abs(local[2]) <= size[2]
            if inside:
                total += wp.abs(impulses[world, row])
                break
    output[world, 0] = total * inv_dt


@wp.kernel
def geom_pair_hits(
    world_mask: wp.array(dtype=wp.bool),
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    limit: int,
    pairs: wp.array2d(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.float32),
):
    world, group = wp.tid()
    if not world_mask[world]:
        return
    for i in range(wp.min(counts[world], limit)):
        pair = contacts[world, i].pair_index
        for j in range(offsets[group], offsets[group + 1]):
            a, b = pairs[j, 0], pairs[j, 1]
            if (pair[0] == a and pair[1] == b) or (pair[0] == b and pair[1] == a):
                output[world, group] = 1.0
                return
    output[world, group] = 0.0


class GpuContactProjector:
    """Bind solver storage and share one dense row lookup per snapshot refresh."""

    def __init__(self, backend):
        self.backend = backend
        self._domains = backend.data.low.domains
        low_model = backend.host_model.low
        self._contact_sensors = {sensor.name: sensor for sensor in low_model.low_contact_sensors}
        self._touch_sensors = {sensor.name: sensor for sensor in low_model.low_touch_sensors}
        self._rows = None
        self.prepare_rows = None

    def geom_colliders(self, index):
        return self.backend.host_model.low.get_collider_indices(index) or ()

    def pair_reader(self, groups):
        offsets = [0]
        flat = []
        for group in groups:
            flat.extend(group)
            offsets.append(len(flat))
        backend = self.backend
        if not flat:
            result = backend.position.new_zeros((backend.data.num_worlds, len(groups)))
            return lambda mask: result
        device = backend._warp_device
        pairs = wp.array(flat, dtype=wp.int32, device=device).reshape((-1, 2))
        bounds = wp.array(offsets, dtype=wp.int32, device=device)
        shape = (backend.data.num_worlds, len(groups))
        output = wp.empty(shape, dtype=wp.float32, device=device)
        result = wp.to_torch(output)
        collision = self._domains["rigid_collision"]
        limit = min(int(collision.max_contact_count), int(collision.contacts.shape[1]))

        def fetch(mask):
            wp.launch(
                geom_pair_hits,
                shape,
                inputs=[mask, collision.contacts, collision.contact_count, limit, pairs, bounds, output],
                device=device,
            )
            return result

        return fetch

    def sensor_reader(self, name):
        if name in self._touch_sensors:
            return self.touch_reader(self._touch_sensors[name])
        sensor = self._contact_sensors.get(name)
        if sensor is None:
            return None
        report = sensor.report
        if (
            report.normal
            and not any((report.found, report.force, report.torque, report.pos, report.tangent, report.dist))
            and sensor.max_num == 1
            and sensor.reduce == "none"
        ):
            return self.geom_contact_reader(sensor.collider_pairs, normal=True)
        if (
            not report.found
            or any((report.force, report.torque, report.pos, report.normal, report.tangent, report.dist))
            or sensor.max_num != 1
        ):
            raise NotImplementedError(
                f"MotrixSim CUDA cannot provide contact sensor {name!r} with its requested report"
            )
        return self.geom_contact_reader(sensor.collider_pairs)

    def link_contact_reader(self, indices):
        if not indices:
            empty = self.backend.position.new_empty((self.backend.data.num_worlds, 0, 6))
            return lambda mask: empty

        collision = self._domains["rigid_collision"]
        solver = self._domains["solver"]
        maps = self.backend.host_model.low.link_low_maps
        # A high-level link can span virtual joints. Its colliders belong to the final rigid link.
        low = [maps[i].index + maps[i].num - 1 for i in indices]
        high_ids = wp.array(indices, dtype=wp.int32, device=self.backend._warp_device)
        low_ids = wp.array(low, dtype=wp.int32, device=self.backend._warp_device)
        output = wp.zeros(
            (self.backend.data.num_worlds, len(indices), 6), dtype=wp.float32, device=self.backend._warp_device
        )
        result = wp.to_torch(output)

        rows = self._contact_rows()
        poses = self.backend.model.get_link_poses(self.backend.data)

        def fetch(mask):
            wp.launch(
                link_contact_wrenches,
                (self.backend.data.num_worlds, len(indices)),
                inputs=[
                    mask,
                    collision.contacts,
                    collision.contact_count,
                    rows,
                    solver.constraint_lambdas.unilateral,
                    poses,
                    self._domains["overrides"].apparent_link_com,
                    high_ids,
                    low_ids,
                    1.0 / self.backend.model.timestep,
                    output,
                ],
                device=self.backend._warp_device,
            )
            return result

        return fetch

    def _contact_rows(self):
        if self._rows is not None:
            return self._rows
        collision = self._domains["rigid_collision"]
        solver = self._domains["solver"]
        sparse = solver.sparse_jacobian
        if sparse is not None:
            self._rows = sparse.contact_rows.rows
        else:
            limit = min(int(collision.max_contact_count), int(collision.contacts.shape[1]))
            row_shape = (collision.contacts.shape[0], limit)
            rows = wp.empty((*row_shape, 6), dtype=wp.int32, device=self.backend._warp_device)
            warm = solver.persistent_lambdas

            def prepare(mask):
                wp.launch(
                    dense_contact_rows,
                    row_shape,
                    inputs=[
                        mask,
                        collision.contacts,
                        collision.contact_count,
                        warm.slots,
                        warm.rows,
                        warm.row_counter,
                        rows,
                    ],
                    device=self.backend._warp_device,
                )

            self._rows = rows
            self.prepare_rows = prepare
        return self._rows

    def touch_reader(self, sensor):
        shape = sensor.shape.shape_type
        if shape not in ("sphere", "box") or sensor.loc.location_type != "Frame":
            raise NotImplementedError(f"Unsupported touch sensor geometry for {sensor.name!r}: {shape}")
        collision = self._domains["rigid_collision"]
        impulses = self._domains["solver"].constraint_lambdas.unilateral
        sensors = self._domains["sensors"]
        colliders = wp.array(sensor.colliders, dtype=wp.int32, device=self.backend._warp_device)
        output = wp.zeros((self.backend.data.num_worlds, 1), dtype=wp.float32, device=self.backend._warp_device)
        result = wp.to_torch(output)
        rows = self._contact_rows()
        size = wp.vec3(*sensor.size)

        def fetch(mask):
            wp.launch(
                touch_force,
                self.backend.data.num_worlds,
                inputs=[
                    mask,
                    collision.contacts,
                    collision.contact_count,
                    rows,
                    impulses,
                    sensors.base_to_frames,
                    sensors.frame_len,
                    sensor.loc.location_id,
                    colliders,
                    size,
                    shape == "box",
                    1.0 / self.backend.model.timestep,
                    output,
                ],
                device=self.backend._warp_device,
            )
            return result

        return fetch

    def geom_contact_reader(self, collider_pairs, *, normal=False):
        collision = self._domains["rigid_collision"]
        if not collider_pairs:
            empty = self.backend.position.new_zeros((self.backend.data.num_worlds, 3 if normal else 1))
            return lambda mask: empty
        pairs = wp.array(collider_pairs, dtype=wp.int32, device=self.backend._warp_device).reshape((-1, 2))
        output = wp.zeros(
            (self.backend.data.num_worlds, 3 if normal else 1), dtype=wp.float32, device=self.backend._warp_device
        )

        result = wp.to_torch(output)
        limit = min(int(collision.max_contact_count), int(collision.contacts.shape[1]))

        def fetch(mask):
            wp.launch(
                read_geom_contacts,
                self.backend.data.num_worlds,
                inputs=[mask, collision.contacts, collision.contact_count, limit, pairs, normal, output],
                device=self.backend._warp_device,
            )
            return result

        return fetch
