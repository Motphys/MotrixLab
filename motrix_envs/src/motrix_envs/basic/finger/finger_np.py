# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofPositionLimitsQuery,
    GeomPairCollidingQuery,
    GeomSpecsQuery,
    JointPositionQuery,
    JointPositionWrite,
    JointVelocityQuery,
    LinkPositionQuery,
    SensorValuesQuery,
    SitePositionQuery,
)
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite
from motrix_envs.basic.finger.cfg import FingerBaseCfg

_FINGER_COLLIDABLE_GEOMS = (
    "ground",
    "proximal_decoration",
    "proximal",
    "fingertip",
    "cap1",
    "cap2",
    "spinner_decoration",
)
_FINGER_COLLISION_PAIRS = tuple(
    (first, second)
    for index, first in enumerate(_FINGER_COLLIDABLE_GEOMS)
    for second in _FINGER_COLLIDABLE_GEOMS[index + 1 :]
)

_SIM_DATA_QUERIES = {
    "joint_pos": JointPositionQuery(joints=("proximal", "distal", "hinge")),
    "joint_vel": JointVelocityQuery(joints=("proximal", "distal", "hinge")),
    "actuator_ctrls": ActuatorCtrlQuery(),
    "spinner_pos": LinkPositionQuery(link="spinner"),
    "tip_pos": SitePositionQuery(site="tip"),
    "touchtop_pos": SitePositionQuery(site="touchtop"),
    "touchbottom_pos": SitePositionQuery(site="touchbottom"),
    "touch": SensorValuesQuery(sensors=("touchtop", "touchbottom")),
    "colliding": GeomPairCollidingQuery(pairs=_FINGER_COLLISION_PAIRS),
}


def _sim_model_queries(cfg: FingerBaseCfg):
    geom_names = ("cap1",) if cfg.task == "spin" else ("cap1", "target_geom")
    return {
        "dof_position_limits": DofPositionLimitsQuery(),
        "geoms": GeomSpecsQuery(names=geom_names),
    }


def _sanitize_joint_limits(low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.where(torch.isfinite(low), low, torch.full_like(low, -np.pi))
    high = torch.where(torch.isfinite(high), high, torch.full_like(high, np.pi))
    return low, high


class FingerEnv(DirectEnv):
    _cfg: FingerBaseCfg
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: FingerBaseCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_sim_model_queries(cfg))
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._cfg = cfg

        dof_lower, dof_upper = self.model.others["dof_position_limits"]
        self._joint_limit_low, self._joint_limit_high = _sanitize_joint_limits(dof_lower, dof_upper)

        self._joint_resets = self.sim.write_compiler.compile(
            {
                "joints_position": JointPositionWrite(("proximal", "distal", "hinge")),
                "joints_velocity": JointVelocityWrite(("proximal", "distal", "hinge")),
            },
            reset=True,
        )
        self._target_reset = (
            self.sim.write_compiler.compile(
                {
                    "target_position": BodyPositionWrite(("target_vis",)),
                    "target_rotation": BodyRotationWrite(("target_vis",)),
                    "target_linear_velocity": BodyLinearVelocityWrite(("target_vis",)),
                    "target_angular_velocity": BodyAngularVelocityWrite(("target_vis",)),
                }
            )
            if "target_geom" in self.model.others["geoms"]
            else None
        )

        self._target_xyz = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._target_radius = float(cfg.target_radius)
        self._spin_vel_threshold = float(cfg.spin_velocity_threshold)

        self._init_obs_space()
        self._init_action_space()

    def _init_obs_space(self):
        raise NotImplementedError

    def _init_action_space(self):
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._action_space = gym.spaces.Box(
            ctrl_ranges[:, 0], ctrl_ranges[:, 1], (self.num_actuators,), dtype=np.float32
        )

        # Episode-scoped action history: full-batch buffers, reset writes the
        # done rows in place.
        self._actions = torch.zeros((self._num_envs, self.num_actuators), dtype=torch.float32)
        self._last_actions = torch.zeros((self._num_envs, self.num_actuators), dtype=torch.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        # Keep track of actions for reward shaping (e.g., smoothness penalties)
        self._last_actions.copy_(self._actions)
        self._actions.copy_(actions)
        self._ctrl_writes.buffer("ctrl").copy_(actions)
        self._ctrl_writes.execute()
        return state

    def _dist_to_target(self, rows) -> torch.Tensor:
        # Signed distance to the target surface. Negative means inside.
        tip_xyz = self.sim_data["tip_pos"][rows]
        dist = torch.linalg.norm((self._target_xyz[rows] - tip_xyz)[:, [0, 2]], dim=-1)
        return dist - self._target_radius

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        raise NotImplementedError

    def _reset_target(self, env_ids: torch.Tensor, positions: torch.Tensor) -> None:
        if self._target_reset is None:
            return
        self._target_reset.buffer("target_position")[env_ids, 0] = positions
        self._target_reset.buffer("target_rotation")[env_ids, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self._target_reset.buffer("target_linear_velocity")[env_ids, 0] = 0.0
        self._target_reset.buffer("target_angular_velocity")[env_ids, 0] = 0.0
        self._target_reset.execute(env_ids)

    def _reset_collision_free_joint_angles(self, env_ids: torch.Tensor):
        # Randomize joint angles with a collision-free rejection sampler (dm_control-style).
        # The joint bounds are per-joint, so we explicitly fill each DOF.
        num = env_ids.numel()
        max_attempts = int(getattr(self._cfg, "reset_collision_free_attempts", 200))
        positions = self._joint_resets.buffer("joints_position")
        velocities = self._joint_resets.buffer("joints_velocity")
        velocities[env_ids] = 0.0
        pending = torch.ones((num,), dtype=torch.bool)
        for _ in range(max_attempts):
            if not bool(pending.any()):
                break

            num_pending = int(pending.sum())
            for column in range(2):
                low = float(self._joint_limit_low[column])
                high = float(self._joint_limit_high[column])
                positions[env_ids[pending], column] = (torch.rand(num_pending, dtype=torch.float32) * 2.0 - 1.0) * (
                    (high - low) / 2.0
                ) + (high + low) / 2.0

            positions[env_ids[pending], 2] = (torch.rand(num_pending, dtype=torch.float32) * 2.0 - 1.0) * np.pi

            self._joint_resets.execute(env_ids)
            if self._target_reset is not None:
                self._reset_target(env_ids, torch.tile(torch.tensor([0.0, 0.0, 0.4]), (num, 1)))
            self.sim_data.execute(env_ids)
            # Contact iff any declared collidable geom pair is colliding; this
            # reproduces the legacy global ``num_contacts > 0`` check exactly.
            pending = self.sim_data["colliding"][env_ids].max(dim=-1).values > 0

    def reset(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError


@registry.env("dm-finger-spin")
class FingerSpinEnv(FingerEnv):
    def _init_obs_space(self):
        # Match dm_control's observation dict, but flatten into a vector.
        # Spin: position(4) + velocity(3) + touch(2) = 9
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        joint_pos = inputs["joint_pos"]
        tip_xz = (inputs["tip_pos"] - inputs["spinner_pos"])[:, [0, 2]]
        position = torch.cat([joint_pos[:, :2], tip_xz], dim=-1)
        velocity = inputs["joint_vel"]
        touch = torch.log1p(inputs["touch"])
        obs = torch.cat([position, velocity, touch], dim=-1)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        terminated = inputs["joint_pos"].isnan().any(dim=-1) | inputs["joint_vel"].isnan().any(dim=-1)

        hinge_velocity = inputs["joint_vel"][:, 2]
        spin_sparse = (hinge_velocity <= -self._spin_vel_threshold).float()

        if self._cfg.reward_mode == "shaped":
            # Dense reward to help PPO learn: encourage fast negative hinge velocity.
            # Range: [0, 1] roughly, with 1 around reaching the threshold.
            spin = torch.clamp((-hinge_velocity) / self._spin_vel_threshold, 0.0, 1.0)
            if self._cfg.shaped_reward_beta != 1.0:
                spin = spin.pow(self._cfg.shaped_reward_beta)
        else:
            spin = spin_sparse

        touch_raw = torch.zeros((self._num_envs,), dtype=torch.float32)
        touch_bonus = torch.zeros((self._num_envs,), dtype=torch.float32)
        approach_dist = torch.zeros((self._num_envs,), dtype=torch.float32)
        approach_reward = torch.zeros((self._num_envs,), dtype=torch.float32)

        if self._cfg.reward_mode == "shaped":
            if float(getattr(self._cfg, "spin_touch_bonus_scale", 0.0)) > 0.0:
                touch_values = inputs["touch"]
                touch_raw = touch_values[:, 0] + touch_values[:, 1]
                touch_bonus = float(self._cfg.spin_touch_bonus_scale) * torch.tanh(
                    touch_raw / float(max(self._cfg.spin_touch_bonus_tanh_scale, 1e-6))
                )

            if float(getattr(self._cfg, "spin_approach_reward_scale", 0.0)) > 0.0:
                spinner_xyz = inputs["spinner_pos"]
                top_xyz = inputs["touchtop_pos"]
                bottom_xyz = inputs["touchbottom_pos"]
                top_dist = torch.linalg.norm((top_xyz - spinner_xyz)[:, [0, 2]], dim=-1)
                bottom_dist = torch.linalg.norm((bottom_xyz - spinner_xyz)[:, [0, 2]], dim=-1)
                approach_dist = torch.minimum(top_dist, bottom_dist)
                sigma = float(max(self._cfg.spin_approach_sigma, 1e-6))
                approach_reward = float(self._cfg.spin_approach_reward_scale) * torch.exp(-approach_dist / sigma)

            spin = torch.clamp(spin + touch_bonus + approach_reward, 0.0, 1.0)

        rwd = spin
        state.reward_terms = {
            "hinge_velocity": hinge_velocity.clone(),
            "spin": spin.clone(),
            "spin_sparse": spin_sparse.clone(),
            "touch_raw": touch_raw.clone(),
            "touch_bonus": touch_bonus.clone(),
            "approach_dist": approach_dist.clone(),
            "approach_reward": approach_reward.clone(),
        }

        rwd[terminated] = 0.0
        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        self._reset_collision_free_joint_angles(env_ids)

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self.sim_data.execute(env_ids)


@registry.env("dm-finger-turn-easy")
@registry.env("dm-finger-turn-hard")
class FingerTurnEnv(FingerEnv):
    def _init_obs_space(self):
        # Match dm_control's observation dict, but flatten into a vector.
        # Turn: position(4) + velocity(3) + touch(2) + target_position(2) + dist_to_target(1) = 12
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        joint_pos = inputs["joint_pos"]
        tip_xz = (inputs["tip_pos"] - inputs["spinner_pos"])[:, [0, 2]]
        position = torch.cat([joint_pos[:, :2], tip_xz], dim=-1)
        velocity = inputs["joint_vel"]
        touch = torch.log1p(inputs["touch"])
        target_position = (self._target_xyz - inputs["spinner_pos"])[:, [0, 2]]
        dist_to_target = self._dist_to_target(slice(None)).reshape(-1, 1)
        obs = torch.cat([position, velocity, touch, target_position, dist_to_target], dim=-1)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        terminated = inputs["joint_pos"].isnan().any(dim=-1) | inputs["joint_vel"].isnan().any(dim=-1)

        dist_to_target = self._dist_to_target(slice(None))
        turn_sparse = (dist_to_target <= 0.0).float()

        touch_raw = torch.zeros((self._num_envs,), dtype=torch.float32)
        touch_bonus = torch.zeros((self._num_envs,), dtype=torch.float32)
        approach_dist = torch.zeros((self._num_envs,), dtype=torch.float32)
        approach_reward = torch.zeros((self._num_envs,), dtype=torch.float32)
        action_l2 = torch.zeros((self._num_envs,), dtype=torch.float32)
        action_delta_l2 = torch.zeros((self._num_envs,), dtype=torch.float32)

        if self._cfg.reward_mode == "shaped":
            # Encourage approaching the spinner so the agent actually makes contact and can rotate it.
            spinner_xyz = inputs["spinner_pos"]
            top_xyz = inputs["touchtop_pos"]
            bottom_xyz = inputs["touchbottom_pos"]
            top_dist = torch.linalg.norm((top_xyz - spinner_xyz)[:, [0, 2]], dim=-1)
            bottom_dist = torch.linalg.norm((bottom_xyz - spinner_xyz)[:, [0, 2]], dim=-1)
            approach_dist = torch.minimum(top_dist, bottom_dist)
            sigma = max(float(self._cfg.turn_approach_sigma), 1e-6)
            approach_reward = self._cfg.turn_approach_reward_scale * torch.exp(-approach_dist / sigma)

            dist_pos = torch.clamp(dist_to_target, min=0.0)
            if getattr(self._cfg, "turn_reward_shape", "linear") == "exp":
                sigma = float(
                    max(self._cfg.turn_reward_sigma_scale * self._target_radius, self._cfg.turn_reward_sigma_min)
                )
                sigma = max(sigma, 1e-6)
                turn = torch.exp(-dist_pos / sigma)
            else:
                margin = float(
                    max(self._cfg.turn_reward_margin_scale * self._target_radius, self._cfg.turn_reward_min_margin)
                )
                margin = max(margin, 1e-6)
                # Dense reward: 1 inside target sphere, decays to 0 at `margin` outside.
                turn = torch.clamp(1.0 - dist_pos / margin, 0.0, 1.0)
            if self._cfg.turn_shaped_reward_beta != 1.0:
                turn = turn.pow(self._cfg.turn_shaped_reward_beta)

            # Encourage making contact (to actually be able to rotate the spinner)
            touch_values = inputs["touch"]
            touch_raw = touch_values[:, 0] + touch_values[:, 1]
            touch_bonus = self._cfg.turn_touch_bonus_scale * torch.tanh(
                touch_raw / self._cfg.turn_touch_bonus_tanh_scale
            )

            # Reduce jitter: penalize large actions and action changes
            actions = self._actions
            last_actions = self._last_actions
            action_l2 = torch.mean(torch.square(actions), dim=-1)
            action_delta_l2 = torch.mean(torch.square(actions - last_actions), dim=-1)

            turn = (
                turn
                + approach_reward
                + touch_bonus
                - self._cfg.turn_action_l2_penalty_scale * action_l2
                - self._cfg.turn_action_delta_l2_penalty_scale * action_delta_l2
            )
            turn = torch.clamp(turn, 0.0, 1.0)
        else:
            turn = turn_sparse

        rwd = turn
        state.reward_terms = {
            "dist_to_target": dist_to_target.clone(),
            "turn": turn.clone(),
            "turn_sparse": turn_sparse.clone(),
            "touch_raw": touch_raw.clone(),
            "touch_bonus": touch_bonus.clone(),
            "approach_dist": approach_dist.clone(),
            "approach_reward": approach_reward.clone(),
            "action_l2": action_l2.clone(),
            "action_delta_l2": action_delta_l2.clone(),
        }

        rwd[terminated] = 0.0
        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num = env_ids.numel()
        self._reset_collision_free_joint_angles(env_ids)

        self.sim_data.execute(env_ids)
        hinge_xyz = self.sim_data["spinner_pos"][env_ids]
        # Match dm_control: radius = cap1.geom_size.sum() for capsule (radius + half-length).
        cap1_size = self.model.others["geoms"]["cap1"].size
        radius = float(sum(cap1_size[:2]))
        target_angle = (torch.rand(num, dtype=torch.float32) * 2.0 - 1.0) * np.pi
        target_x = hinge_xyz[:, 0] + radius * torch.sin(target_angle)
        target_z = hinge_xyz[:, 2] + radius * torch.cos(target_angle)
        self._target_xyz[env_ids] = torch.stack([target_x, hinge_xyz[:, 1], target_z], dim=-1)

        # NOTE: the legacy num_envs==1 visualization hack mutated the shared
        # "target" site's local position and marker size on the simulator
        # model; that debug rendering tweak (wrapped in try/except, purely
        # cosmetic, never read back by observations or physics) was dropped
        # with the DirectEnv migration since static site mutation is outside
        # the sim contract.

        # If we have a freejoint-backed visual target (geom), set its pose in the state.
        if self._target_reset is not None:
            self._reset_target(env_ids, self._target_xyz[env_ids])
            self.sim_data.execute(env_ids)

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
