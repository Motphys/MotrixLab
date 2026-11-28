# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import gymnasium as gym
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, EnvObs, FinalObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec, append_observation
from motrix_env_core.renderer import RecordConfig
from motrix_env_core.sim import (
    BodyJointPositionLimitsQuery,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyJointVelocityWrite,
    GeomPairCollidingQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SensorValuesQuery,
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
_PEG_HALF_LENGTH = 0.040
_SOCKET_SIDES = ("xp", "xn", "yp", "yn")
_WALL_SENSORS = tuple(f"{section}_{side}" for section in ("wall", "cham") for side in _SOCKET_SIDES)
_CONTACT_PAIRS = (
    ("peg_collision", "socket_floor"),
    *(("peg_collision", f"socket_top_{side}") for side in _SOCKET_SIDES),
    *(("peg_collision", f"socket_{wall}") for wall in _WALL_SENSORS),
)

_SIM_DATA_QUERIES = {
    "robot_joint_pos": BodyJointPositionQuery(body="link0"),
    "robot_joint_vel": BodyJointVelocityQuery(body="link0"),
    "gripper_pos": SitePositionQuery(site="gripper"),
    "gripper_quat": SiteQuaternionQuery(site="gripper"),
    "gripper_lin_vel": LinkLinearVelocityQuery(link="hand"),
    "peg_pos": LinkPositionQuery(link="peg"),
    "peg_quat": LinkQuaternionQuery(link="peg"),
    "socket_pos": LinkPositionQuery(link="blind_socket"),
    "contact": GeomPairCollidingQuery(pairs=_CONTACT_PAIRS),
    "wall_distance": SensorValuesQuery(sensors=_WALL_SENSORS),
}

_SIM_MODEL_QUERIES = {
    "joint_limits": BodyJointPositionLimitsQuery(body="link0"),
}


@registry.env("panda-blind-insertion")
@registry.env("panda-insert")
class PandaBlindInsertEnv(DirectEnv):
    """Proprioceptive square-peg insertion with a rigidly attached peg."""

    _cfg: PandaBlindInsertEnvCfg

    def __init__(
        self,
        cfg: PandaBlindInsertEnvCfg,
        num_envs: int = 1,
        backend: str | None = None,
        seed: int | None = None,
        *,
        headless: bool = False,
        record: RecordConfig | None = None,
    ):
        super().__init__(cfg, num_envs=num_envs, backend=backend, headless=headless, record=record)
        # DirectEnv does not forward the registry seed; the task accepts it
        # explicitly so frozen-seed evals sample genuinely different resets.
        self._rng = torch.Generator().manual_seed(0 if seed is None else seed)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite(_ACTUATORS)})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "robot_position": BodyJointPositionWrite("link0"),
                "robot_velocity": BodyJointVelocityWrite("link0"),
                "socket_position": KinematicBodyPositionWrite(("blind_socket",)),
                "socket_rotation": KinematicBodyRotationWrite(("blind_socket",)),
            },
            reset=True,
        )

        self._default_joint_pos = torch.tensor(cfg.init_state.default_joint_pos, dtype=torch.float32)
        self._settled_joint_pos = torch.tensor(cfg.init_state.settled_joint_pos, dtype=torch.float32)
        self._joint_limits = self.model.others["joint_limits"]
        self._arm_limits = tuple(limit[:_NUM_ARM] for limit in self._joint_limits)
        self._vertical = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self._socket_rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._action_dim = _NUM_ARM
        # 42-dim frame: qpos-default(9) qvel(9) ee_pos(3) ee_quat(4)
        # last_action(7) tracking_residual(7) ee_lin_vel(3)
        self._frame_obs_dim = 42
        self._history_len = int(cfg.reset.history_len)
        self._privileged_obs_dim = 10
        self._value_obs_dim = self._frame_obs_dim + self._privileged_obs_dim
        self._action_space = gym.spaces.Box(-1.0, 1.0, (self._action_dim,), dtype="float32")
        proprioception = gym.spaces.Box(-math.inf, math.inf, (self._frame_obs_dim,), dtype="float32")
        privileged = gym.spaces.Box(-math.inf, math.inf, (self._value_obs_dim,), dtype="float32")
        self._observation_spec = ObservationSpec(
            {
                "policy": ObservationGroupSpec(proprioception, self._history_len),
                "value": ObservationGroupSpec(privileged),
            }
        )

        self._current_actions = torch.zeros((num_envs, self._action_dim), dtype=torch.float32)
        self._joint_targets = torch.zeros((num_envs, _NUM_ARM), dtype=torch.float32)
        self._history = torch.zeros((num_envs, self._history_len, self._frame_obs_dim), dtype=torch.float32)
        self._history_needs_reset = torch.ones(num_envs, dtype=torch.bool)
        self._grasped = torch.zeros(num_envs, dtype=torch.bool)
        self._contacted = torch.zeros(num_envs, dtype=torch.bool)
        self._mouth_seen = torch.zeros(num_envs, dtype=torch.bool)
        self._hold_count = torch.zeros(num_envs, dtype=torch.int32)
        self._prev_xy = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_depth = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_height = torch.zeros(num_envs, dtype=torch.float32)
        self._xy_watermark = torch.full((num_envs,), math.inf, dtype=torch.float32)
        self._initial_peg_ee_rel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._initial_peg_ee_quat = torch.zeros((num_envs, 4), dtype=torch.float32)

    @property
    def observation_spec(self) -> ObservationSpec:
        return self._observation_spec

    @property
    def action_space(self):
        return self._action_space

    def _write_rows(self, env_ids: torch.Tensor, robot_pos: torch.Tensor, socket_pos: torch.Tensor) -> None:
        self._reset_program.buffer("robot_position")[env_ids] = robot_pos
        self._reset_program.buffer("robot_velocity")[env_ids] = 0.0
        self._reset_program.buffer("socket_position")[env_ids, 0] = socket_pos
        self._reset_program.buffer("socket_rotation")[env_ids, 0] = self._socket_rotation
        self._reset_program.execute(env_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        noise_scale = self._cfg.init_state.joint_pos_reset_noise_scale
        noise = torch.empty((count, _NUM_ARM), dtype=torch.float32).uniform_(
            -noise_scale, noise_scale, generator=self._rng
        )
        robot_pos = self._settled_joint_pos.expand(count, -1).clone()
        robot_pos[:, :_NUM_ARM] += noise
        robot_pos.clamp_(*self._joint_limits)

        # Refresh the settled pose before centering the socket distribution on the peg.
        socket_pos = robot_pos.new_zeros((count, 3))
        self._write_rows(env_ids, robot_pos, socket_pos)
        self.sim_data.execute(env_ids)
        peg_pos = self.sim_data["peg_pos"][env_ids]
        angles, bucket, unit_radius = torch.rand((3, count), generator=self._rng, dtype=torch.float32)
        angles *= math.tau
        w_near, w_mid = self._cfg.reset.bucket_weights
        cap = self._cfg.reset.offset_cap
        radii = torch.where(
            bucket < w_near,
            0.3 * cap * unit_radius,
            torch.where(bucket < w_near + w_mid, cap * (0.3 + 0.3 * unit_radius), cap * (0.6 + 0.4 * unit_radius)),
        )
        offset = radii[:, None] * torch.stack((angles.cos(), angles.sin()), dim=-1)
        socket_pos[:, :2] = peg_pos[:, :2] - offset
        self._write_rows(env_ids, robot_pos, socket_pos)
        target = (self._default_joint_pos[:_NUM_ARM] + noise).clamp(*self._arm_limits)
        self._joint_targets[env_ids] = target
        self._write_controls(target, env_ids)
        self.sim_data.execute(env_ids)

        self._current_actions[env_ids] = 0.0
        self._grasped[env_ids] = True
        self._contacted[env_ids] = False
        self._mouth_seen[env_ids] = False
        self._hold_count[env_ids] = 0
        self._xy_watermark[env_ids] = math.inf
        self._history_needs_reset[env_ids] = True

        metrics = self._metrics(env_ids)
        gripper_pos = self.sim_data["gripper_pos"][env_ids]
        gripper_quat = self.sim_data["gripper_quat"][env_ids]
        peg_quat = self.sim_data["peg_quat"][env_ids]
        self._initial_peg_ee_rel[env_ids] = quaternion.rotate_inverse(gripper_quat, metrics["peg"] - gripper_pos)
        self._initial_peg_ee_quat[env_ids] = quaternion.mul(quaternion.inverse(gripper_quat), peg_quat)
        self._prev_xy[env_ids] = metrics["xy"]
        self._prev_depth[env_ids] = metrics["depth"]
        self._prev_height[env_ids] = self._height_above_socket(metrics["peg"], metrics["socket"])

    def _write_controls(self, target: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        ctrl = self._ctrl_writes.buffer("ctrl")
        rows = slice(None) if env_ids is None else env_ids
        ctrl[rows, :_NUM_ARM] = target
        ctrl[rows, _NUM_ARM] = self._cfg.action.gripper_close_ctrl
        self._ctrl_writes.execute(env_ids)

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        acts = actions.clamp(-1.0, 1.0)
        self._current_actions.copy_(acts)
        # Accumulating the target allows the PD controller to maintain force
        # against the socket, with a bound on the tracking error.
        target = self._joint_targets + self._cfg.action.action_scale * acts
        target.clamp_(*self._arm_limits)
        q = self.sim_data["robot_joint_pos"][:, :_NUM_ARM]
        max_err = self._cfg.action.max_tracking_error
        target.clamp_(q - max_err, q + max_err)
        self._joint_targets[:] = target
        self._write_controls(target)
        return state

    def _height_above_socket(self, peg: torch.Tensor, socket: torch.Tensor) -> torch.Tensor:
        surface = socket[:, 2] + self._cfg.reset.socket_top_z
        return (peg[:, 2] - _PEG_HALF_LENGTH - surface).clamp_min(0.0)

    def _metrics(self, rows: slice | torch.Tensor = slice(None)) -> dict[str, torch.Tensor]:
        peg_pos = self.sim_data["peg_pos"][rows]
        peg_quat = self.sim_data["peg_quat"][rows]
        socket_pos = self.sim_data["socket_pos"][rows]
        axis = quaternion.rotate_vector(peg_quat, self._vertical)
        bottom = peg_pos[:, 2] - _PEG_HALF_LENGTH * axis[:, 2].abs()
        xy = torch.linalg.vector_norm(peg_pos[:, :2] - socket_pos[:, :2], dim=-1)
        inside_outer = xy <= self._cfg.reset.socket_outer_radius
        surface = socket_pos[:, 2] + self._cfg.reset.socket_top_z
        depth = torch.where(inside_outer, (surface - bottom).clamp(0.0, self._cfg.reset.socket_top_z), 0.0)
        tilt = axis[:, 2].abs().clamp_max(1.0).acos().rad2deg()
        contacts = self.sim_data["contact"][rows].any(dim=-1)
        # Signed distances from the deepest contact on each wall include the
        # peg's actual orientation and both sections of the socket. The top
        # plate and floor count as contact for rewards but are not wall overlap.
        wall_distance = self.sim_data["wall_distance"][rows]
        penetration = (-wall_distance.amin(dim=-1)).clamp_min(0.0)
        return {
            "peg": peg_pos,
            "socket": socket_pos,
            "xy": xy,
            "depth": depth,
            "tilt": tilt,
            "penetration": penetration,
            "contact": contacts,
        }

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        metrics = self._metrics()
        gripper_pos = self.sim_data["gripper_pos"]
        gripper_quat = self.sim_data["gripper_quat"]
        peg_quat = self.sim_data["peg_quat"]
        peg_ee_rel = quaternion.rotate_inverse(gripper_quat, metrics["peg"] - gripper_pos)
        peg_ee_quat = quaternion.mul(quaternion.inverse(gripper_quat), peg_quat)
        peg_ee_drift = torch.linalg.vector_norm(peg_ee_rel - self._initial_peg_ee_rel, dim=-1)
        peg_ee_angle = quaternion.rotation_distance(peg_ee_quat, self._initial_peg_ee_quat)
        # Welded peg: grasp failure is structurally impossible; drop stays
        # False and termination is governed by success/penetration/tilt.
        drop = torch.zeros_like(self._grasped)
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
        self._hold_count = torch.where(success_now, self._hold_count + 1, 0)
        penetration_fail = metrics["penetration"] > self._cfg.termination.penetration_limit
        success = self._hold_count >= self._cfg.termination.hold_steps
        terminated = success | penetration_fail | drop

        reward_cfg = self._cfg.reward
        xy_progress = self._prev_xy - metrics["xy"]
        depth_progress = metrics["depth"] - self._prev_depth
        grasp_mask = self._grasped.float()
        pre_contact = (~self._contacted).float()
        height_above_panel = self._height_above_socket(metrics["peg"], metrics["socket"])
        descent_drop = (self._prev_height - height_above_panel).clamp_min(0.0)
        xy = metrics["xy"]
        improved = self._contacted & (xy < self._xy_watermark)
        xy_pay = torch.where(improved, torch.exp(-80.0 * xy) - torch.exp(-80.0 * self._xy_watermark), 0.0)
        self._xy_watermark = torch.where(improved, xy, self._xy_watermark)
        # Quadratic depth fraction: residency pay is negligible at the 1-2 mm
        # mouth (where passive sag parks the peg) and only becomes real
        # income toward the 15 mm target.
        depth_fraction = (metrics["depth"] / self._cfg.termination.target_depth).clamp(0.0, 1.0).square()
        # Mouth residency is gated on actual insertion depth: paying it for
        # mere top-plate proximity let the stage-1 policy farm ~4/episode by
        # parking at the hole mouth without ever pressing in.
        in_mouth = (metrics["xy"] <= self._cfg.termination.xy_tolerance * 2.0) & (metrics["depth"] > 0.001)

        reward = (
            reward_cfg.reach_scale
            * torch.where(self._contacted, torch.maximum(xy_progress, -0.5 * self._prev_xy), 0.0)
            * grasp_mask
        )
        reward += reward_cfg.depth_scale * depth_progress * grasp_mask
        reward += reward_cfg.xy_scale * xy_pay * grasp_mask
        reward += reward_cfg.descent_scale * descent_drop * pre_contact * grasp_mask
        reward += reward_cfg.contact_bonus * new_contact * grasp_mask
        reward += reward_cfg.contact_maintenance * metrics["contact"] * grasp_mask
        reward += reward_cfg.contact_residency * metrics["contact"] * in_mouth * grasp_mask
        reward += reward_cfg.insert_residency * depth_fraction * grasp_mask
        reward += reward_cfg.mouth_bonus * new_mouth * grasp_mask
        reward += reward_cfg.success_bonus * success * grasp_mask
        reward -= reward_cfg.drop_penalty * drop
        reward -= reward_cfg.action_penalty * self._current_actions.square().sum(dim=-1)
        self._prev_xy[:] = metrics["xy"]
        self._prev_depth[:] = metrics["depth"]
        self._prev_height[:] = height_above_panel

        state.reward = reward
        state.terminated = terminated
        state.reward_terms = {
            "reach": reward_cfg.reach_scale * torch.where(self._contacted, xy_progress, 0.0),
            "descent": reward_cfg.descent_scale * descent_drop * pre_contact,
            "xy_kernel": reward_cfg.xy_scale * xy_pay,
            "depth": reward_cfg.depth_scale * depth_progress,
            "insert_hold": depth_fraction * reward_cfg.insert_residency,
            "contact": (
                new_contact.float() * reward_cfg.contact_bonus
                + metrics["contact"] * in_mouth * reward_cfg.contact_residency
                + metrics["contact"] * reward_cfg.contact_maintenance
            ),
            "success": success.float() * reward_cfg.success_bonus,
            "penetration_fail": penetration_fail.float(),
            "drop": drop.float(),
        }
        state.metrics = {
            "success": success,
            "contact": metrics["contact"],
            "depth": metrics["depth"],
            "xy_error": metrics["xy"],
            "tilt_deg": metrics["tilt"],
            "penetration": metrics["penetration"],
            "peg_ee_gap": torch.linalg.vector_norm(metrics["peg"] - gripper_pos, dim=-1),
            "peg_ee_drift": peg_ee_drift,
            "peg_ee_angle_deg": peg_ee_angle.rad2deg(),
            "grasped": self._grasped,
        }
        return state

    def _observation_frames(self, rows: slice | torch.Tensor = slice(None)) -> tuple[torch.Tensor, torch.Tensor]:
        joints = self.sim_data["robot_joint_pos"][rows]
        velocities = self.sim_data["robot_joint_vel"][rows]
        gripper_pos = self.sim_data["gripper_pos"][rows]
        gripper_quat = self.sim_data["gripper_quat"][rows]
        ee_vel = self.sim_data["gripper_lin_vel"][rows]
        tracking_residual = self._joint_targets[rows] - joints[:, :_NUM_ARM]
        frame = torch.cat(
            [
                joints - self._default_joint_pos,
                velocities,
                gripper_pos,
                gripper_quat,
                self._current_actions[rows],
                tracking_residual,
                ee_vel,
            ],
            dim=-1,
        )
        metrics = self._metrics(rows)
        value = torch.cat(
            [
                frame,
                metrics["peg"] - metrics["socket"],
                metrics["socket"] - gripper_pos,
                metrics["depth"][:, None],
                metrics["xy"][:, None],
                metrics["tilt"][:, None],
                self._grasped[rows, None],
            ],
            dim=-1,
        )
        return frame, value

    def _capture_final_obs(self, env_ids: torch.Tensor) -> FinalObs:
        frame, value = self._observation_frames(env_ids)
        history = append_observation(self._history[env_ids], frame)
        return FinalObs(env_ids, EnvObs({"policy": history.flatten(1), "value": value}))

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        frame, value = self._observation_frames()
        self._history = append_observation(self._history, frame, self._history_needs_reset)
        self._history_needs_reset.zero_()
        setattr(state, "obs", EnvObs({"policy": self._history.flatten(1), "value": value}))
        return state
