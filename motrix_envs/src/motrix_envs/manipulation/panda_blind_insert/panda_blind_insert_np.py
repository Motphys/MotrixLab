# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyJointPositionLimitsQuery,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyJointVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    GeomPairCollidingQuery,
    GeomPositionQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SitePositionQuery,
    SiteQuaternionQuery,
)
from motrix_env_core.sim.write import (
    CtrlTargetsWrite,
    KinematicBodyPositionWrite,
    KinematicBodyRotationWrite,
)

from .cfg import PandaBlindInsertEnvCfg

_ACTUATORS = (
    "actuator1",
    "actuator2",
    "actuator3",
    "actuator4",
    "actuator5",
    "actuator6",
    "actuator7",
    "actuator8",
)
_NUM_ARM = 7
_PEG_RADIUS = 0.012
_PEG_HALF = 0.040
_CONTACT_PAIRS = tuple(
    [("peg_collision", "socket_floor")]
    + [("peg_collision", f"socket_wall_{side}") for side in ("xp", "xn", "yp", "yn")]
    + [("peg_collision", "socket_top")]
)

_SIM_DATA_QUERIES = {
    "robot_joint_pos": BodyJointPositionQuery(body="link0"),
    "robot_joint_vel": BodyJointVelocityQuery(body="link0"),
    "gripper_pos": SitePositionQuery(site="gripper"),
    "gripper_quat": SiteQuaternionQuery(site="gripper"),
    "gripper_lin_vel": LinkLinearVelocityQuery(link="hand"),
    "left_finger_pos": GeomPositionQuery(geom="left_finger_pad"),
    "right_finger_pos": GeomPositionQuery(geom="right_finger_pad"),
    "peg_pos": LinkPositionQuery(link="peg"),
    "peg_quat": LinkQuaternionQuery(link="peg"),
    "socket_pos": LinkPositionQuery(link="blind_socket"),
    "contact": GeomPairCollidingQuery(pairs=_CONTACT_PAIRS),
}

_SIM_MODEL_QUERIES = {
    "joint_limits": BodyJointPositionLimitsQuery(body="link0"),
}


@registry.env("panda-blind-insertion")
@registry.env("panda-insert")
class PandaBlindInsertEnv(DirectEnv):
    """Open-asset blind insertion: Panda + parametric 24-segment socket.

    Semantics mirror the validated blind-insertion recipe: stratified
    near-hole resets, watermark alignment kernel, symmetric depth potential,
    impedance-residual observation (joint_target - qpos exposes the hole
    mouth without a force sensor), qpos-rebased integrator with anti-windup.
    """

    _cfg: PandaBlindInsertEnvCfg

    def __init__(self, cfg: PandaBlindInsertEnvCfg, num_envs=1, backend: str | None = None, seed: int | None = None):
        super().__init__(cfg, num_envs=num_envs, backend=backend)
        # DirectEnv does not forward the registry seed; the task accepts it
        # explicitly so frozen-seed evals sample genuinely different resets.
        self._rng = np.random.default_rng(0 if seed is None else int(seed))
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite(_ACTUATORS)})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "robot_position": BodyJointPositionWrite("link0"),
                "robot_velocity": BodyJointVelocityWrite("link0"),
                "peg_position": BodyPositionWrite(("peg",)),
                "peg_rotation": BodyRotationWrite(("peg",)),
                "peg_linear_velocity": BodyLinearVelocityWrite(("peg",)),
                "peg_angular_velocity": BodyAngularVelocityWrite(("peg",)),
                "socket_position": KinematicBodyPositionWrite(("blind_socket",)),
                "socket_rotation": KinematicBodyRotationWrite(("blind_socket",)),
            },
            reset=True,
        )

        self._default_joint_pos = np.asarray(cfg.init_state.default_joint_pos, dtype=np.float32)
        limits = self.model.others["joint_limits"]
        if isinstance(limits[0], torch.Tensor):
            limits = tuple(t.detach().cpu().numpy() for t in limits)
        self._joint_limits = tuple(np.asarray(item, dtype=np.float32) for item in limits)
        self._action_dim = _NUM_ARM
        # 42-dim frame: qpos-default(9) qvel(9) ee_pos(3) ee_quat(4)
        # last_action(7) tracking_residual(7) ee_lin_vel(3)
        self._frame_obs_dim = 42
        self._history_len = int(cfg.reset.history_len)
        self._policy_obs_dim = self._frame_obs_dim * self._history_len
        self._privileged_obs_dim = 10
        self._value_obs_dim = self._frame_obs_dim + self._privileged_obs_dim
        self._action_space = gym.spaces.Box(-1.0, 1.0, (self._action_dim,), dtype=np.float32)
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-np.inf, np.inf, (self._policy_obs_dim,), dtype=np.float32),
            value=gym.spaces.Box(-np.inf, np.inf, (self._value_obs_dim,), dtype=np.float32),
        )

        self._current_actions = np.zeros((num_envs, self._action_dim), dtype=np.float32)
        self._last_actions = np.zeros_like(self._current_actions)
        self._joint_targets = np.zeros((num_envs, _NUM_ARM), dtype=np.float32)
        self._history = np.zeros((num_envs, self._history_len, self._frame_obs_dim), dtype=np.float32)
        self._history_initialized = np.zeros(num_envs, dtype=bool)
        self._physics_failed = np.zeros(num_envs, dtype=bool)
        self._grasped = np.zeros(num_envs, dtype=bool)
        self._contacted = np.zeros(num_envs, dtype=bool)
        self._mouth_seen = np.zeros(num_envs, dtype=bool)
        self._hold_count = np.zeros(num_envs, dtype=np.int32)
        self._prev_xy = np.zeros(num_envs, dtype=np.float32)
        self._prev_depth = np.zeros(num_envs, dtype=np.float32)
        self._prev_height = np.zeros(num_envs, dtype=np.float32)
        self._xy_watermark = np.full(num_envs, np.inf, dtype=np.float32)
        self._socket_xy = np.zeros((num_envs, 2), dtype=np.float32)
        self._initial_peg_ee_rel = np.zeros((num_envs, 3), dtype=np.float32)
        self._initial_peg_ee_quat = np.zeros((num_envs, 4), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # ---------- helpers ----------
    def _np(self, value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _write_rows(self, env_ids: torch.Tensor, *, robot_pos, peg_pos, socket_pos) -> None:
        row_ids = env_ids.detach().cpu().to(torch.int64)
        count = int(row_ids.numel())
        device = self._reset_program.buffer("robot_position").device
        self._reset_program.buffer("robot_position")[row_ids] = torch.as_tensor(
            np.asarray(robot_pos, dtype=np.float32), device=device
        )
        self._reset_program.buffer("robot_velocity")[row_ids] = 0.0
        peg = torch.zeros((count, 1, 3), dtype=torch.float32, device=device)
        peg[:, 0, :] = torch.as_tensor(np.asarray(peg_pos, dtype=np.float32), device=device)
        self._reset_program.buffer("peg_position")[row_ids] = peg
        quat = torch.zeros((count, 1, 4), dtype=torch.float32, device=device)
        quat[:, 0, 3] = 1.0
        self._reset_program.buffer("peg_rotation")[row_ids] = quat
        self._reset_program.buffer("peg_linear_velocity")[row_ids] = 0.0
        self._reset_program.buffer("peg_angular_velocity")[row_ids] = 0.0
        sock = torch.zeros((count, 1, 3), dtype=torch.float32, device=device)
        sock[:, 0, :] = torch.as_tensor(np.asarray(socket_pos, dtype=np.float32), device=device)
        self._reset_program.buffer("socket_position")[row_ids] = sock
        self._reset_program.buffer("socket_rotation")[row_ids] = quat
        self._reset_program.execute(row_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        row_ids = env_ids.detach().cpu().to(torch.int64)
        np_rows = row_ids.numpy()
        count = int(np_rows.size)
        if count == 0:
            return

        noise = self._rng.uniform(
            -self._cfg.init_state.joint_pos_reset_noise_scale,
            self._cfg.init_state.joint_pos_reset_noise_scale,
            size=(count, self._default_joint_pos.size),
        ).astype(np.float32)
        robot_pos = self._default_joint_pos[None, :] + noise
        robot_pos = np.clip(robot_pos, self._joint_limits[0], self._joint_limits[1])

        # Place the arm first; seed the peg between the finger pads.
        zero = np.zeros((count, 3), dtype=np.float32)
        self._write_rows(row_ids, robot_pos=robot_pos, peg_pos=zero, socket_pos=zero)
        self.sim_data.execute(row_ids)
        # Hold the peg centered between the finger pads, measured from the
        # live state for a repeatable grasp.
        left = self._np(self.sim_data["left_finger_pos"][row_ids])
        right = self._np(self.sim_data["right_finger_pos"][row_ids])
        pad_center = 0.5 * (left + right)
        peg_pos = np.zeros((count, 3), dtype=np.float32)
        peg_pos[:, :2] = pad_center[:, :2]
        peg_pos[:, 2] = pad_center[:, 2] - _PEG_HALF + self._cfg.reset.peg_hold_height

        # Stratified socket placement around the peg start.
        angles = self._rng.uniform(0.0, 2.0 * np.pi, count)
        w_near, w_mid = self._cfg.reset.bucket_weights
        bucket = self._rng.uniform(0.0, 1.0, count)
        cap = self._cfg.reset.offset_cap
        r_in = self._rng.uniform(0.0, 1.0, count)
        radii = np.where(
            bucket < w_near,
            0.3 * cap * r_in,
            np.where(bucket < w_near + w_mid, cap * (0.3 + 0.3 * r_in), cap * (0.6 + 0.4 * r_in)),
        ).astype(np.float32)
        offset = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1).astype(np.float32)
        socket_pos = np.column_stack(
            [
                peg_pos[:, 0] - offset[:, 0],
                peg_pos[:, 1] - offset[:, 1],
                np.zeros(count, dtype=np.float32),
            ]
        ).astype(np.float32)
        self._write_rows(row_ids, robot_pos=robot_pos, peg_pos=peg_pos, socket_pos=socket_pos)
        device = self._ctrl_writes.buffer("ctrl").device
        ctrl = torch.zeros((count, len(_ACTUATORS)), dtype=torch.float32, device=device)
        ctrl[:, :_NUM_ARM] = torch.as_tensor(robot_pos[:, :_NUM_ARM], device=device)
        ctrl[:, _NUM_ARM] = self._cfg.action.gripper_close_ctrl
        self._ctrl_writes.buffer("ctrl")[row_ids] = ctrl
        self._ctrl_writes.execute(row_ids)
        self.sim_data.execute(row_ids)

        self._current_actions[np_rows] = 0.0
        self._last_actions[np_rows] = 0.0
        self._joint_targets[np_rows] = robot_pos[:, :_NUM_ARM]
        self._socket_xy[np_rows] = socket_pos[:, :2]
        self._grasped[np_rows] = True
        self._contacted[np_rows] = False
        self._mouth_seen[np_rows] = False
        self._hold_count[np_rows] = 0
        self._xy_watermark[np_rows] = np.inf
        self._history_initialized[np_rows] = False
        self._physics_failed[np_rows] = False

        # Settle the freshly placed grasp: advance physics while holding the
        # reset ctrl targets so the peg slides to its contact equilibrium
        # INSIDE the fingers (the placement pose is not an equilibrium —
        # measured drift reaches ~17 mm within 3 control steps of placement).
        # The drop anchor and obs/watermark baselines are re-taken from the
        # settled state so that transient is never booked as policy drift.
        for _ in range(25):
            self.physics_step()
        self.sim_data.execute(row_ids)
        metrics = self._metrics(row_ids)
        gripper_pos = self._np(self.sim_data["gripper_pos"][row_ids])
        gripper_quat = self._np(self.sim_data["gripper_quat"][row_ids])
        peg_quat = self._np(self.sim_data["peg_quat"][row_ids])
        self._initial_peg_ee_rel[np_rows] = quaternion.rotate_inverse(gripper_quat, metrics["peg"] - gripper_pos)
        self._initial_peg_ee_quat[np_rows] = quaternion.mul(quaternion.inverse(gripper_quat), peg_quat)
        self._prev_xy[np_rows] = metrics["xy"]
        self._prev_depth[np_rows] = metrics["depth"]
        reset_height = np.maximum(metrics["peg"][:, 2] - _PEG_HALF - self._cfg.reset.socket_top_z, 0.0)
        self._prev_height[np_rows] = reset_height

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        acts = self._np(actions).astype(np.float32)
        acts = np.clip(acts, -1.0, 1.0)
        self._last_actions[:] = self._current_actions
        self._current_actions[:] = acts
        q = self._np(self.sim_data["robot_joint_pos"])[:, :_NUM_ARM]
        target = q + self._cfg.action.action_scale * acts
        target = np.clip(target, self._joint_limits[0][:_NUM_ARM], self._joint_limits[1][:_NUM_ARM])
        max_err = self._cfg.action.max_tracking_error
        target = np.clip(target, q - max_err, q + max_err)
        self._joint_targets[:] = target
        n = target.shape[0]
        ctrl = np.zeros((n, len(_ACTUATORS)), dtype=np.float32)
        ctrl[:, :_NUM_ARM] = target
        ctrl[:, _NUM_ARM] = self._cfg.action.gripper_close_ctrl
        device = self._ctrl_writes.buffer("ctrl").device
        self._ctrl_writes.buffer("ctrl")[:] = torch.as_tensor(ctrl, device=device)
        self._ctrl_writes.execute()
        return state

    def physics_step(self) -> None:
        """Advance physics, converting batch solver failures to terminations."""
        self._physics_failed.fill(False)
        try:
            super().physics_step()
        except RuntimeError:
            self._physics_failed[:] = True

    def _metrics(self, rows=slice(None)) -> dict[str, np.ndarray]:
        peg_pos = self._np(self.sim_data["peg_pos"][rows])
        peg_quat = self._np(self.sim_data["peg_quat"][rows])
        socket_pos = self._np(self.sim_data["socket_pos"][rows])
        axis = quaternion.rotate_vector(peg_quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        bottom = peg_pos[:, 2] - _PEG_HALF * np.abs(axis[:, 2])
        xy = np.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], axis=-1)
        inside_outer = xy <= self._cfg.reset.socket_outer_radius
        depth = np.where(
            inside_outer, np.clip(socket_pos[:, 2] + _PEG_HALF + 0.040 - bottom, 0.0, 0.040), 0.0
        )
        # The socket top surface sits at z = socket_top_z (cylinder height
        # 0.040 standing on z=0): depth measured from that face.
        depth = np.where(inside_outer, np.clip(self._cfg.reset.socket_top_z - bottom, 0.0, 0.040), 0.0)
        tilt = np.degrees(np.arccos(np.clip(np.abs(axis[:, 2]), -1.0, 1.0)))
        contacts = self._np(self.sim_data["contact"][rows]).astype(bool).any(axis=-1)
        penetration = np.where(
            (depth > 0.002) & inside_outer & contacts,
            np.maximum(xy - self._cfg.reset.hole_clearance, 0.0),
            0.0,
        )
        return {
            "peg": peg_pos,
            "socket": socket_pos,
            "xy": xy,
            "depth": depth,
            "tilt": tilt,
            "penetration": penetration,
            "contact": contacts,
        }

    def compute_transition(self, state: ArrayEnvState):
        if self._physics_failed.any():
            failed = torch.as_tensor(self._physics_failed.copy())
            state.reward = torch.where(failed, -1.0, 0.0).float()
            state.terminated = failed
            state.reward_terms = {"physics_failure": failed.float()}
            state.metrics = {"physics_failure": failed}
            return state
        self.sim_data.execute()
        metrics = self._metrics()
        gripper_pos = self._np(self.sim_data["gripper_pos"])
        gripper_quat = self._np(self.sim_data["gripper_quat"])
        peg_quat = self._np(self.sim_data["peg_quat"])
        peg_ee_rel = quaternion.rotate_inverse(gripper_quat, metrics["peg"] - gripper_pos)
        peg_ee_quat = quaternion.mul(quaternion.inverse(gripper_quat), peg_quat)
        peg_ee_drift = np.linalg.norm(peg_ee_rel - self._initial_peg_ee_rel, axis=-1)
        peg_ee_angle = quaternion.rotation_distance(peg_ee_quat, self._initial_peg_ee_quat)
        drop = self._grasped & (
            (peg_ee_drift > self._cfg.termination.drop_drift)
            | (np.degrees(peg_ee_angle) > self._cfg.termination.drop_angle_deg)
        )
        self._grasped[drop] = False
        new_contact = metrics["contact"] & ~self._contacted
        self._contacted |= metrics["contact"]
        mouth = (metrics["depth"] > 0.0015) & (metrics["xy"] < 0.003) & (metrics["tilt"] < 10.0)
        new_mouth = mouth & ~self._mouth_seen
        self._mouth_seen |= mouth
        success_now = (
            (metrics["depth"] >= self._cfg.termination.target_depth)
            & (metrics["xy"] <= self._cfg.termination.xy_tolerance)
            & (metrics["tilt"] <= self._cfg.termination.tilt_tolerance_deg)
            & (metrics["penetration"] <= self._cfg.termination.success_penetration_witness)
            & self._contacted
            & self._grasped
        )
        self._hold_count = np.where(success_now, self._hold_count + 1, 0)
        penetration_fail = metrics["penetration"] > self._cfg.termination.penetration_limit
        success = self._hold_count >= self._cfg.termination.hold_steps
        terminated = success | penetration_fail | drop

        reward_cfg = self._cfg.reward
        xy_progress = self._prev_xy - metrics["xy"]
        depth_progress = metrics["depth"] - self._prev_depth
        grasp_mask = self._grasped.astype(np.float32)
        pre_contact = (~self._contacted).astype(np.float32)
        height_above_panel = np.maximum(
            metrics["peg"][:, 2] - _PEG_HALF - self._cfg.reset.socket_top_z, 0.0
        )
        descent_drop = np.maximum(self._prev_height - height_above_panel, 0.0)
        kernel_at = lambda d: np.exp(-80.0 * d)  # noqa: E731
        improved = self._contacted & (metrics["xy"] < self._xy_watermark)
        xy_pay = np.zeros(self.num_envs, dtype=np.float32)
        if improved.any():
            xy_pay[improved] = kernel_at(metrics["xy"][improved]) - kernel_at(self._xy_watermark[improved])
            self._xy_watermark[improved] = metrics["xy"][improved]
        depth_gain = depth_progress
        depth_fraction = np.clip(metrics["depth"] / self._cfg.termination.target_depth, 0.0, 1.0)
        in_mouth = metrics["xy"] <= self._cfg.termination.xy_tolerance * 2.0

        reward = reward_cfg.reach_scale * np.where(
            self._contacted, np.maximum(xy_progress, -0.5 * self._prev_xy), 0.0
        ) * grasp_mask
        reward += reward_cfg.depth_scale * depth_gain * grasp_mask
        reward += reward_cfg.xy_scale * xy_pay * grasp_mask
        reward += reward_cfg.descent_scale * descent_drop * pre_contact * grasp_mask
        reward += reward_cfg.contact_bonus * new_contact * grasp_mask
        reward += reward_cfg.contact_maintenance * metrics["contact"] * grasp_mask
        reward += reward_cfg.contact_residency * metrics["contact"] * in_mouth * grasp_mask
        reward += reward_cfg.insert_residency * depth_fraction * grasp_mask
        reward += reward_cfg.mouth_bonus * new_mouth * grasp_mask
        reward += reward_cfg.success_bonus * success * grasp_mask
        reward -= reward_cfg.drop_penalty * drop
        reward -= reward_cfg.action_penalty * np.square(self._current_actions).sum(axis=-1)
        self._prev_xy[:] = metrics["xy"]
        self._prev_depth[:] = metrics["depth"]
        self._prev_height[:] = height_above_panel

        t = lambda x: torch.as_tensor(np.asarray(x))  # noqa: E731
        state.reward = t(reward.astype(np.float32))
        state.terminated = t(terminated)
        state.reward_terms = {
            "reach": t(reward_cfg.reach_scale * np.where(self._contacted, xy_progress, 0.0)),
            "descent": t(reward_cfg.descent_scale * descent_drop * pre_contact),
            "xy_kernel": t(reward_cfg.xy_scale * xy_pay),
            "depth": t(reward_cfg.depth_scale * depth_gain),
            "insert_hold": t(depth_fraction * reward_cfg.insert_residency),
            "contact": t(
                new_contact.astype(np.float32) * reward_cfg.contact_bonus
                + metrics["contact"] * in_mouth * reward_cfg.contact_residency
                + metrics["contact"] * reward_cfg.contact_maintenance
            ),
            "success": t(success.astype(np.float32) * reward_cfg.success_bonus),
            "penetration_fail": t(penetration_fail.astype(np.float32)),
            "drop": t(drop.astype(np.float32)),
        }
        state.metrics = {
            "success": t(success.astype(bool)),
            "contact": t(metrics["contact"]),
            "depth": t(metrics["depth"]),
            "xy_error": t(metrics["xy"]),
            "tilt_deg": t(metrics["tilt"]),
            "penetration": t(metrics["penetration"]),
            "peg_ee_gap": t(np.linalg.norm(metrics["peg"] - gripper_pos, axis=-1)),
            "peg_ee_drift": t(peg_ee_drift),
            "peg_ee_angle_deg": t(np.degrees(peg_ee_angle)),
            "grasped": t(self._grasped),
        }
        return state

    def compute_observation(self, state: ArrayEnvState):
        joints = self._np(self.sim_data["robot_joint_pos"])
        velocities = self._np(self.sim_data["robot_joint_vel"])
        gripper_pos = self._np(self.sim_data["gripper_pos"])
        gripper_quat = self._np(self.sim_data["gripper_quat"])
        ee_vel = self._np(self.sim_data["gripper_lin_vel"])
        tracking_residual = self._joint_targets - joints[:, :_NUM_ARM]
        frame = np.concatenate(
            [
                joints - self._default_joint_pos,
                velocities,
                gripper_pos,
                gripper_quat,
                self._current_actions,
                tracking_residual,
                ee_vel,
            ],
            axis=-1,
        ).astype(np.float32)
        fresh = ~self._history_initialized
        self._history[fresh] = frame[fresh, None, :]
        live = ~fresh
        self._history[live, :-1] = self._history[live, 1:]
        self._history[live, -1] = frame[live]
        self._history_initialized[:] = True
        policy = self._history.reshape(self.num_envs, -1)
        metrics = self._metrics()
        value = np.concatenate(
            [
                frame,
                metrics["peg"] - metrics["socket"],
                metrics["socket"] - gripper_pos,
                metrics["depth"][:, None],
                metrics["xy"][:, None],
                metrics["tilt"][:, None],
                self._grasped[:, None],
            ],
            axis=-1,
        ).astype(np.float32)
        setattr(state, "obs", EnvObs(policy=torch.as_tensor(policy), value=torch.as_tensor(value)))
        return state
