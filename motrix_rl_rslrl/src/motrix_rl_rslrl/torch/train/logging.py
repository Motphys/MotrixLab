# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RSLRL metric logging without its native training console output."""

import statistics
from pathlib import Path

import torch
from rsl_rl.utils.logger import Logger


class RslrlTrainingLogger(Logger):
    """Keep RSLRL's writers and episode buffers, leaving console output to the trainer."""

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std: torch.Tensor,
        rnd_weight: float | None,
        print_minimal: bool = False,
        width: int = 80,
        pad: int = 40,
        *,
        num_iterations: int = 1,
    ) -> None:
        """Write the upstream scalar tags and videos, without rendering a progress table.

        Console-only arguments remain accepted so this is interchangeable with
        :meth:`rsl_rl.utils.logger.Logger.log`. For window publication, callers
        supply summed durations, averaged losses and the number of updates in
        ``num_iterations`` so throughput and cumulative steps stay accurate.
        """
        if not self.writer:
            return

        collection_size = num_iterations * self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += collect_time + learn_time

        if self.ep_extras:
            for key in self.ep_extras[0]:
                values = torch.tensor([], device=self.device)
                for ep_info in self.ep_extras:
                    if key not in ep_info:
                        continue
                    value = ep_info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.Tensor([value])
                    if value.ndim == 0:
                        value = value.unsqueeze(0)
                    values = torch.cat((values, value.to(self.device)))
                mean = values.mean()
                tag = key if "/" in key else f"Episode/{key}"
                self.writer.add_scalar(tag, mean, it)

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", learning_rate, it)
        self.writer.add_scalar("Policy/mean_noise_std", action_std.mean().item(), it)

        fps = int(collection_size / (collect_time + learn_time))
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collect_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if self.rewbuffer:
            if self.cfg["algorithm"]["rnd_cfg"]:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(self.erewbuffer), it)
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(self.irewbuffer), it)
                self.writer.add_scalar("Rnd/weight", rnd_weight, it)
            mean_reward = statistics.mean(self.rewbuffer)
            mean_length = statistics.mean(self.lenbuffer)
            self.writer.add_scalar("Train/mean_reward", mean_reward, it)
            self.writer.add_scalar("Train/mean_episode_length", mean_length, it)
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", mean_reward, int(self.tot_time))
                self.writer.add_scalar("Train/mean_episode_length/time", mean_length, int(self.tot_time))

        if self.logger_type == "wandb":
            for video in Path(self.log_dir).rglob("*.mp4"):
                self.writer.save_video(video, it)

        self.ep_extras.clear()

    def close(self) -> None:
        """Release the writer, propagating failures for the trainer to handle.

        TensorBoard needs an explicit flush and close; external backends use
        RSLRL's own shutdown hook. This also works before writer initialization.
        """
        writer = getattr(self, "writer", None)
        if writer is None:
            return
        try:
            if self.logger_type == "tensorboard":
                try:
                    writer.flush()
                finally:
                    writer.close()
            else:
                self.stop_logging_writer()
        finally:
            self.writer = None
