# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    BatchLinkQuaternionQuery,
    BodyJointPositionLimitsQuery,
    BodyJointPositionWrite,
    DofPositionQuery,
    DofVelocityQuery,
    LinkPositionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite
from motrix_env_core.torch import reward
from motrix_envs.basic.walker.cfg import WalkerEnvCfg

_WALKER_LINKS = ("torso", "right_thigh", "right_leg", "right_foot", "left_thigh", "left_leg", "left_foot")

_SIM_DATA_QUERIES = {
    "link_quats": BatchLinkQuaternionQuery(links=_WALKER_LINKS),
    "torso_pos": LinkPositionQuery(link="torso"),
    "torso_subtreelinvel": SensorValuesQuery(sensors=("torso_subtreelinvel",)),
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
}
_SIM_MODEL_QUERIES = {"joint_position_limits": BodyJointPositionLimitsQuery(body="torso")}


def _quat_to_rotation_mats(quats: torch.Tensor) -> torch.Tensor:
    """Convert (x, y, z, w) quaternions, shape (..., 4), to rotation mats (..., 3, 3)."""
    x, y, z, w = quats.unbind(-1)
    entries = (
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )
    mats = torch.stack(entries, dim=-1).reshape(quats.shape[:-1] + (3, 3))
    return mats


@registry.env("dm-walker")
@registry.env("dm-runner")
@registry.env("dm-stander")
class Walker2DEnv(DirectEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: WalkerEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {"walker_position": BodyJointPositionWrite("torso"), "walker_velocity": BodyJointVelocityWrite("torso")},
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("walker_position")
        self._reset_velocity = self._reset_program.buffer("walker_velocity")
        self._init_obs_space()
        self._init_action_space()
        self._num_links = len(_WALKER_LINKS)
        self._move_speed = cfg.move_speed
        self._joint_pos_lower, self._joint_pos_upper = self.model.others["joint_position_limits"]
        self._stand_height = cfg.stand_height

    def _init_obs_space(self):
        num = 0
        num += (len(_WALKER_LINKS) - 1) * 2  # planar rotation (x,z) per link except root
        num += 1  # torso height
        num += self.num_dof_vel
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num,), dtype=np.float32)

    def _init_action_space(self):
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._action_space = gym.spaces.Box(
            ctrl_ranges[:, 0],
            ctrl_ranges[:, 1],
            (self.num_actuators,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        self._ctrl_writes.buffer("ctrl").copy_(actions)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        num_env = self._num_envs
        link_rotations = _quat_to_rotation_mats(inputs["link_quats"])
        dof_vel = inputs["dof_vel"]
        up_right = link_rotations[:, 0, 2, 2].reshape(num_env, 1)  # 1
        orientations = link_rotations[:, 1:, [0, 0], [2, 0]].reshape(num_env, -1)  # (num_links - 1) * 2
        obs = torch.cat([orientations, up_right, dof_vel], dim=-1)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        # === compute termination inputs ===
        dof_vel = inputs["dof_vel"]

        torso_upright = _quat_to_rotation_mats(inputs["link_quats"][:, 0])[:, 2, 2]
        torso_height = inputs["torso_pos"][:, 2]
        torso_vel = inputs["torso_subtreelinvel"]
        horizontal_velocity = torso_vel[:, 0]

        # ==== compute terminated
        terminated = inputs["dof_pos"].isnan().any(dim=-1) | dof_vel.isnan().any(dim=-1)

        # ==== compute reward
        rwd_height = torch.as_tensor(
            reward.tolerance(
                torso_height,
                bounds=(self._stand_height, float("inf")),
                margin=self._stand_height * 4 / 5,
            ),
            dtype=torch.float32,
        )
        rwd_upright = (1 + torso_upright) / 2
        rwd_stand = (3 * rwd_height + 1 * rwd_upright) / 4

        rwd = rwd_stand

        reward_terms = {
            "height": rwd_height,
            "upright": rwd_upright,
            "stand": rwd_stand,
        }

        if self._move_speed > 0.0:
            rwd_move = torch.as_tensor(
                reward.tolerance(
                    horizontal_velocity,
                    bounds=(self._move_speed, float("inf")),
                    margin=self._move_speed / 2,
                    value_at_margin=0.5,
                    sigmoid="linear",
                ),
                dtype=torch.float32,
            )
            reward_terms["move"] = rwd_move
            rwd = rwd_stand * (5 * rwd_move + 1) / 6

        state.reward_terms = reward_terms
        rwd[terminated] = 0.0

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()

        dof_pos = torch.zeros((num_reset, self._reset_position.shape[1]), dtype=torch.float32)
        dof_pos[:, 2] = (torch.rand(num_reset, dtype=torch.float32) * 2.0 - 1.0) * np.pi  # randomize root yaw
        lower = self._joint_pos_lower[3:]
        upper = self._joint_pos_upper[3:]
        dof_pos[:, 3:] = (
            torch.rand((num_reset, self._reset_position.shape[1] - 3), dtype=torch.float32) * (upper - lower) + lower
        )  # randomize other joint angles
        dof_vel = torch.zeros((num_reset, self._reset_velocity.shape[1]), dtype=torch.float32)

        self._reset_position[env_ids] = dof_pos
        self._reset_velocity[env_ids] = dof_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
