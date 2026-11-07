# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct import reward
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    DofPositionQuery,
    DofVelocityQuery,
    JointPositionWrite,
    LinkPositionQuery,
)
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite
from motrix_envs.basic.reacher.cfg import ReacherEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "finger_pos": LinkPositionQuery(link="finger"),
    "target_pos": LinkPositionQuery(link="target"),
}
_SIM_MODEL_QUERIES = {}


@registry.env("dm-reacher")
class Reacher2DEnv(DirectEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: ReacherEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "arm_position": JointPositionWrite(("shoulder", "wrist")),
                "arm_velocity": JointVelocityWrite(("shoulder", "wrist")),
                "target_position": JointPositionWrite(("target_x", "target_y")),
                "target_velocity": JointVelocityWrite(("target_x", "target_y")),
            },
            reset=True,
        )
        self._arm_position = self._reset_program.buffer("arm_position")
        self._arm_velocity = self._reset_program.buffer("arm_velocity")
        self._target_position = self._reset_program.buffer("target_position")
        self._target_velocity = self._reset_program.buffer("target_velocity")

        self._target_size = cfg.target_size
        self._target_xyz = np.zeros((num_envs, 3), dtype=np.float32)
        self._init_obs_space()
        self._init_action_space()

    def _init_obs_space(self):
        num_obs = self.num_dof_pos + 2 + self.num_dof_vel
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._action_space = gym.spaces.Box(
            ctrl_ranges[:, 0], ctrl_ranges[:, 1], (self.num_actuators,), dtype=np.float32
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        actions = actions.detach().cpu().numpy()
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = np.asarray(actions, dtype=np.float32)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        qpos = inputs["dof_pos"]
        qvel = inputs["dof_vel"]
        finger_xy = inputs["finger_pos"][:, :2]
        to_target = self._target_xyz[:, :2] - finger_xy
        return state.replace(obs=np.concatenate([qpos, to_target, qvel], axis=-1))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        finger_xy = self.sim_data["finger_pos"][:, :2]
        dist = np.linalg.norm(self._target_xyz[:, :2] - finger_xy, axis=-1)
        rwd = reward.tolerance(
            dist, bounds=(0.0, self._target_size), margin=self._target_size, value_at_margin=0.0, sigmoid="linear"
        )
        terminated = (
            np.isnan(self.sim_data["dof_pos"]).any(axis=-1)
            | np.isnan(self.sim_data["dof_vel"]).any(axis=-1)
            | np.isnan(finger_xy).any(axis=-1)
            | np.isnan(rwd)
        )
        rwd[terminated] = 0.0

        state.reward_terms = {"distance": dist, "tolerance": rwd.copy()}

        return state.replace(reward=rwd, terminated=terminated)

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.detach().cpu().numpy()
        """Reset environment with randomized target position in xy plane (z=0)."""
        num_reset = len(env_ids)

        arm_position = np.zeros((num_reset, 2), dtype=np.float32)
        arm_position[:, 0] = np.random.uniform(-np.pi, np.pi, size=(num_reset,))
        arm_position[:, 1] = np.random.uniform(-np.pi, np.pi, size=(num_reset,))

        target_x = np.random.uniform(-0.15, 0.15, size=(num_reset,))
        target_y = np.random.uniform(0.15, 0.15, size=(num_reset,))

        target_position = np.stack([target_x, target_y], axis=-1).astype(np.float32)
        arm_velocity = np.zeros((num_reset, 2), dtype=np.float32)
        target_velocity = np.zeros((num_reset, 2), dtype=np.float32)

        self._arm_position[env_ids] = arm_position
        self._arm_velocity[env_ids] = arm_velocity
        self._target_position[env_ids] = target_position
        self._target_velocity[env_ids] = target_velocity
        self._reset_program.execute(env_ids)
        self.sim_data.execute(np.asarray(env_ids, np.int64))

        target_pose = self.sim_data["target_pos"][env_ids]
        self._target_xyz[env_ids] = target_pose
        self._target_xyz[env_ids, 2] = 0.0
