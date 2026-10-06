# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained g1-walk-slope policy pinned at one terrain difficulty row.

Loads an RSLRL run directory, rebuilds the slope environment with every lane
pinned to a single difficulty row (both the up-slope and down-slope columns),
and rolls out deterministic episodes while measuring:

- termination ratio (collision / bad-dof failures vs 20 s timeouts),
- fraction of lanes that leave the central spawn platform onto the slope
  (base radius from the tile origin beyond the platform edge),
- platform spinning: mean |yaw rate| while still on the platform.

Usage:
    python scripts/private/eval_g1_slope.py run_dir=<runs/g1-walk-slope/...> \
        [row=9] [num_envs=64] [rounds=5]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from motrix_env_core import registry  # noqa: E402
from motrix_envs.locomotion.humanoid.g1_slope import make_g129dof_walk_slope_cfg  # noqa: E402
from motrix_rl.config import CheckpointConfig, LoggingConfig  # noqa: E402
from motrix_rl.rslrl.torch import wrap_env  # noqa: E402
from motrix_rl.rslrl.torch.train.ppo import add_runtime_config  # noqa: E402

MAX_EPISODE_STEPS = 1000  # 20 s at 0.02 s control dt.
PLATFORM_EDGE = 1.0  # Platform half-width in meters (platform_width=2.0).
PLATFORM_EXIT_RADIUS = 1.5  # Safely onto the slope from the tile center.
DEEP_SLOPE_RADIUS = 2.5
TILE_EDGE_RADIUS = 3.5

_RUNNER_KEYS = (
    "num_steps_per_env",
    "max_iterations",
    "obs_groups",
    "actor",
    "critic",
    "algorithm",
)

# Evaluated policy: copied next to this script and pinned here so the eval
# invocation stays argument-free. Refresh the copy when a better checkpoint
# is confirmed.
_POLICY_PATH = Path(__file__).resolve().parent / "slope_policy.pt"


def _build_runner_cfg(run_dir: Path) -> dict:
    task_cfg = yaml.safe_load((run_dir / "task_config.yaml").read_text())
    algo = task_cfg["algo"]
    cfg = {key: algo[key] for key in _RUNNER_KEYS if key in algo}
    cfg["max_iterations"] = 1
    return add_runtime_config(cfg, LoggingConfig(backend="tensorboard"), CheckpointConfig(interval=0))


def _yaw(quats: np.ndarray) -> np.ndarray:
    x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--row", type=int, default=9)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=5, help="full episode batches")
    parser.add_argument(
        "--framework",
        default=None,
        choices=("rslrl", "fastsac"),
        help="trainer framework (default: infer from run dir)",
    )
    parser.add_argument("--record", default=None, help="mp4 path: record the first episodes offscreen")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    policy_path = _POLICY_PATH
    framework = args.framework or ("fastsac" if "/fastsac/" in str(run_dir) else "rslrl")

    cfg = make_g129dof_walk_slope_cfg()
    cfg.commands.walk.terrain_min_level = args.row
    cfg.commands.walk.terrain_max_level = args.row
    cfg.commands.walk.terrain_balanced_columns = False
    env = registry.resolve("g1-walk-slope", env_cfg=cfg).make(num_envs=args.num_envs, mode="play", seed=99)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if framework == "fastsac":
        from motrix_rl.fastsac.agent import FastSacAgent
        from motrix_rl.fastsac.config import FastSacAgentCfg
        from motrix_rl.fastsac.wrap_np import FastSacNpEnvWrap

        render = None
        if args.record:
            from pathlib import Path as _P

            from motrix_env_core.renderer import RenderConfig

            render = RenderConfig(
                headless=True,
                path=_P(args.record),
                fps=60,
                num_frames=900,
                width=1280,
                height=720,
            )
        vec_env = FastSacNpEnvWrap(env, device, render=render)
        agent_cfg = FastSacAgentCfg(**yaml.safe_load((run_dir / "task_config.yaml").read_text())["algo"]["agent"])
        low, high = vec_env.action_low, vec_env.action_high
        agent = FastSacAgent(
            obs_dim=env.observation_space.policy.shape[0],
            critic_obs_dim=env.observation_space.value.shape[0],
            act_dim=env.action_space.shape[0],
            num_envs=args.num_envs,
            cfg=agent_cfg,
            device=device,
            action_scale=(high - low) / 2.0,
            action_bias=(high + low) / 2.0,
        )
        ckpt = torch.load(policy_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt)
        agent.actor.eval()
        if hasattr(agent.obs_normalizer, "eval"):
            agent.obs_normalizer.eval()

        def _step(obs):
            actions = agent.act(obs, deterministic=True)
            out = vec_env.step(actions)
            if render is not None and vec_env.render() is False:
                raise SystemExit("recording finished")
            return out

        def _terminated_of(step_out):
            return step_out[3].cpu().numpy()

        def _done_of(step_out):
            return (step_out[3] | step_out[4]).cpu().numpy()
    else:
        from rsl_rl.runners import OnPolicyRunner

        vec_env = wrap_env(env, device)
        runner = OnPolicyRunner(vec_env, _build_runner_cfg(run_dir), log_dir=None, device=device)
        runner.load(str(policy_path))
        policy = runner.get_inference_policy(device=device)

        def _step(obs):
            with torch.no_grad():
                return vec_env.step(policy(obs))

        def _terminated_of(step_out):
            return np.asarray(vec_env._state.terminated)

        def _done_of(step_out):
            return np.asarray(vec_env._state.done)

    walk = env.command_terms["walk"]
    n = args.num_envs
    cols = walk.terrain_cols[:, 0].copy()
    origins = np.asarray(walk.terrain_origin_grid).reshape(-1, 3)[args.row * walk.terrain_cols_count + cols][:, :2]

    # Foot attitude diagnostics: resolve the feet pos/quat queries the reward
    # terms registered (look them up by links to stay robust across layouts).
    foot_pos_key = foot_quat_key = None
    for query, key in env._term_query_keys.items():
        links = getattr(query, "links", None)
        if links and "ankle_roll_link" in str(links[0]):
            if query.__class__.__name__ == "BatchLinkPositionQuery":
                foot_pos_key = key
            elif query.__class__.__name__ == "BatchLinkQuaternionQuery":
                foot_quat_key = key
    foot_pitch = {True: [], False: []}  # stance -> attitude-error samples (deg)
    from motrix_env_core.mdp.terrain import heightfield_lookup
    from motrix_env_core.numba.math.quaternion import rotate_inverse_components
    from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid

    robot = env.cfg.scene.objs.robot
    body = env.model.bodies["robot"]
    # Reference gravity per foot: gravity seen in the foot frame at the init
    # key pose (the flat-footed crouch) -- the same quantity penalty_feet_ori
    # penalizes deviation from.
    ref_gravity = np.stack(
        [
            rotate_inverse_components(body.init_link_quats[body.link_names.index(link)], (0.0, 0.0, -1.0))
            for link in robot.resolved_foot_link_names
        ]
    )

    hfield = ground_height_grid(env, "floor")

    def _bilinear_height(grid, x, y):
        """Vector-free bilinear lookup mirroring heightfield_lookup."""
        nrow, ncol = grid.heights.shape
        fx = min(max((x - grid.origin[0]) / grid.spacing[0], 0.0), ncol - 1.0 - 1e-6)
        fy = min(max((y - grid.origin[1]) / grid.spacing[1], 0.0), nrow - 1.0 - 1e-6)
        col, row = int(fx), int(fy)
        tx, ty = fx - col, fy - row
        top = grid.heights[row, col] * (1 - tx) + grid.heights[row, col + 1] * tx
        bottom = grid.heights[row + 1, col] * (1 - tx) + grid.heights[row + 1, col + 1] * tx
        return grid.z0[0] + top * (1 - ty) + bottom * ty

    def _record_foot_pitches() -> None:
        """Record stance/swing foot attitude error vs the flat reference."""
        if foot_pos_key is None:
            return
        fpos = np.asarray(env.sim_data[foot_pos_key])  # (num_envs, 2, 3)
        fquat = np.asarray(env.sim_data[foot_quat_key])  # (num_envs, 2, 4)
        for env_i in range(fpos.shape[0]):
            for foot in range(2):
                ground = _bilinear_height(hfield, fpos[env_i, foot, 0], fpos[env_i, foot, 1])
                clearance = fpos[env_i, foot, 2] - ground
                live = rotate_inverse_components(fquat[env_i, foot], (0.0, 0.0, -1.0))
                dot = float(np.clip(np.dot(live, ref_gravity[foot]), -1.0, 1.0))
                foot_pitch[clearance <= 0.07].append(math.degrees(math.acos(dot)))

    # Per-lane episode tracking.
    max_radius = np.zeros(n, dtype=np.float64)
    platform_steps = np.zeros(n, dtype=np.int64)
    platform_yaw_travel = np.zeros(n, dtype=np.float64)
    prev_yaw = None
    prev_pos = None
    slope_speed_sum = np.zeros(n, dtype=np.float64)
    slope_speed_steps = np.zeros(n, dtype=np.int64)
    cmd_speed_sum = np.zeros(n, dtype=np.float64)
    cmd_speed_steps = np.zeros(n, dtype=np.int64)
    results = []  # (terminated, max_radius, platform_steps, platform_yaw_travel, col, slope_speed, cmd_speed)

    obs = vec_env.reset()[0]
    base_pos = np.asarray(env.sim_data["terrain_curriculum_base_pos"])
    prev_yaw = _yaw(np.asarray(env.sim_data["terrain_curriculum_base_quat"]))

    for round_index in range(args.rounds):
        done_count = 0
        step = 0
        while done_count < n and step < MAX_EPISODE_STEPS + 10:
            step_out = _step(obs)
            obs = step_out[0]
            done_mask = _done_of(step_out).astype(bool)
            terminated_mask = _terminated_of(step_out).astype(bool)
            base_pos = np.asarray(env.sim_data["terrain_curriculum_base_pos"])
            base_quat = np.asarray(env.sim_data["terrain_curriculum_base_quat"])
            _record_foot_pitches()
            yaw = _yaw(base_quat)
            radius = np.linalg.norm(base_pos[:, :2] - origins, axis=1)
            max_radius = np.maximum(max_radius, radius)
            on_platform = radius <= PLATFORM_EXIT_RADIUS
            platform_steps += on_platform
            d_yaw = np.abs((yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi)
            platform_yaw_travel += np.where(on_platform, d_yaw, 0.0)
            if prev_pos is not None:
                speed = np.linalg.norm(base_pos[:, :2] - prev_pos, axis=1) / 0.02
                off = ~on_platform
                slope_speed_sum += np.where(off, speed, 0.0)
                slope_speed_steps += off
            cmd_speed = np.linalg.norm(np.asarray(walk.command)[:, :2], axis=1)
            has_cmd = cmd_speed > 0.05
            cmd_speed_sum += np.where(has_cmd, cmd_speed, 0.0)
            cmd_speed_steps += has_cmd
            fall_pos = prev_pos  # live position at the end of the previous step
            prev_pos = base_pos[:, :2].copy()
            prev_yaw = yaw
            if done_mask.any():
                # base_pos after the step already reflects the auto-reset
                # spawn; the previous step's position is the fall site.
                terminal_radius = np.linalg.norm(fall_pos - origins, axis=1) if fall_pos is not None else np.zeros(n)
                reason_masks = {
                    name: np.asarray(env._state.metrics[name]).astype(bool)
                    for name in ("colliding", "bad_dof_velocity")
                    if name in env._state.metrics
                }
                for lane in np.flatnonzero(done_mask):
                    term_reason = "+".join(name for name, mask in reason_masks.items() if mask[lane])
                    results.append(
                        (
                            bool(terminated_mask[lane]),
                            float(max_radius[lane]),
                            float(terminal_radius[lane]) if terminated_mask[lane] else float("nan"),
                            int(platform_steps[lane]),
                            float(platform_yaw_travel[lane]),
                            int(cols[lane]),
                            float(slope_speed_sum[lane] / max(slope_speed_steps[lane], 1)),
                            float(cmd_speed_sum[lane] / max(cmd_speed_steps[lane], 1)),
                            term_reason if terminated_mask[lane] else "",
                        )
                    )
                max_radius[done_mask] = 0.0
                platform_steps[done_mask] = 0
                platform_yaw_travel[done_mask] = 0.0
                slope_speed_sum[done_mask] = 0.0
                slope_speed_steps[done_mask] = 0
                cmd_speed_sum[done_mask] = 0.0
                cmd_speed_steps[done_mask] = 0
                done_count += int(done_mask.sum())
            step += 1
        # Force-start the next round for any stragglers.
        if round_index + 1 < args.rounds:
            obs, _ = vec_env.reset()
            base_pos = np.asarray(env.sim_data["terrain_curriculum_base_pos"])
            prev_yaw = _yaw(np.asarray(env.sim_data["terrain_curriculum_base_quat"]))
            prev_pos = None
            max_radius[:] = 0.0
            platform_steps[:] = 0
            platform_yaw_travel[:] = 0.0
            slope_speed_sum[:] = 0.0
            slope_speed_steps[:] = 0
            cmd_speed_sum[:] = 0.0
            cmd_speed_steps[:] = 0
            # Drop the in-flight partial episodes.
            results = results[-n * round_index :] if round_index else results

    for stance, label in ((True, "stance"), (False, "swing")):
        if foot_pitch[stance]:
            pitches = np.array(foot_pitch[stance])
            print(
                f"  {label} foot pitch deg: med={np.median(pitches):.1f}"
                f" p25={np.percentile(pitches, 25):.1f} p75={np.percentile(pitches, 75):.1f}"
                f" >15deg share={(pitches > 15).mean() * 100:.0f}%"
            )
    _report(results, args.row)


def _report(results: list[tuple], row: int) -> None:
    total = len(results)
    if total == 0:
        print("No completed episodes.")
        return
    terminated = np.array([r[0] for r in results])
    max_radius = np.array([r[1] for r in results])
    terminal_radius = np.array([r[2] for r in results])
    platform_steps = np.array([r[3] for r in results])
    yaw_travel = np.array([r[4] for r in results])
    cols = np.array([r[5] for r in results])
    slope_speed = np.array([r[6] for r in results])
    cmd_speed = np.array([r[7] for r in results])

    def block(mask: np.ndarray, label: str) -> None:
        count = int(mask.sum())
        if count == 0:
            print(f"  [{label}] no episodes")
            return
        print(
            f"  [{label}] n={count}"
            f" term={terminated[mask].mean() * 100:.1f}%"
            f" exit> {PLATFORM_EXIT_RADIUS}m: {(max_radius[mask] > PLATFORM_EXIT_RADIUS).mean() * 100:.1f}%"
            f" deep> {DEEP_SLOPE_RADIUS}m: {(max_radius[mask] > DEEP_SLOPE_RADIUS).mean() * 100:.1f}%"
            f" edge> {TILE_EDGE_RADIUS}m: {(max_radius[mask] > TILE_EDGE_RADIUS).mean() * 100:.1f}%"
            f" med_max_r={np.median(max_radius[mask]):.2f}m"
            f" platform_steps_med={np.median(platform_steps[mask]):.0f}"
            f" platform_yaw_travel_med={np.median(yaw_travel[mask]) / math.pi:.1f}pi"
            f" slope_speed/med_cmd={np.median(slope_speed[mask]) / max(np.median(cmd_speed[mask]), 1e-6):.2f}"
        )

    print(f"\n=== g1-walk-slope eval @ row {row} (hardest) — {total} episodes ===")
    print(
        f"  overall: termination={terminated.mean() * 100:.1f}%"
        f" exit_rate={(max_radius > PLATFORM_EXIT_RADIUS).mean() * 100:.1f}%"
    )
    fall_r = terminal_radius[terminated & ~np.isnan(terminal_radius)]
    if fall_r.size:
        hist, edges = np.histogram(fall_r, bins=(0.0, 0.5, 0.8, 1.0, 1.5, 2.5, 4.5))
        zones = ["pad(<0.5)", "crest(0.5-0.8)", "crest2(0.8-1)", "slope(1-1.5)", "slope(1.5-2.5)", "far(>2.5)"]
        print("  fall sites: " + ", ".join(f"{z}:{h}" for z, h in zip(zones, hist)))
        fall_steps = np.array([r[3] for r in results if r[0]]) / 50.0
        print(
            f"  fall time s: med={np.median(fall_steps):.1f}"
            f" p25={np.percentile(fall_steps, 25):.1f} p75={np.percentile(fall_steps, 75):.1f}"
        )
        reasons = {}
        for r in results:
            if r[0]:
                reasons[r[8]] = reasons.get(r[8], 0) + 1
        print(f"  fall reasons: {reasons}")
    block(cols == 0, "slope (descend)")
    block(cols == 1, "slope_inv (ascend)")
    goal_term = terminated.mean() < 0.10
    goal_exit = (max_radius > PLATFORM_EXIT_RADIUS).mean() > 0.5
    print(
        f"  goal: termination<10% [{'PASS' if goal_term else 'FAIL'}],"
        f" majority-leaves-platform [{'PASS' if goal_exit else 'FAIL'}]"
    )


if __name__ == "__main__":
    main()
