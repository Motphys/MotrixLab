# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""
Run all training demos with both JAX and PyTorch backends.

This script iterates through all available demo environments and runs
training for each one using both training backends.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# List of all available demo environments
ALL_DEMOS = [
    "cartpole",
    "dm-cheetah",
    "dm-hopper-stand",
    "dm-hopper-hop",
    "dm-reacher",
    "dm-stander",
    "dm-walker",
    "dm-runner",
    "bounce_ball",
    "go1-walk-flat",
    "go1-walk-rough",
    "go1-stairs-terrain-walk",
    "anymal_c_navigation_flat",
    "franka-lift-cube",
    "franka-open-cabinet",
]

# Available training backends
BACKENDS = ["jax", "torch"]


def run_command(env: str, backend: str, extra_args: list[str]) -> int:
    """Run a single training command."""
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    backend_task = config_dir / "task" / env / f"skrl.ppo.{backend}.yaml"
    task_option = f"{env}/skrl.ppo.{backend}" if backend_task.is_file() else f"{env}/skrl.ppo"
    cmd = [
        sys.executable,
        "-m",
        "motrix_rl.commands.train",
        "--config-dir",
        str(config_dir),
        f"task={task_option}",
    ] + extra_args
    if not backend_task.is_file():
        cmd.append(f"task.train_backend={backend}")

    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Error: Training failed for {env} with {backend} backend")
        return result.returncode

    print(f"\n✅ Success: Training completed for {env} with {backend} backend")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run training demos with both JAX and PyTorch backends")
    parser.add_argument(
        "--demos",
        nargs="+",
        choices=ALL_DEMOS + ["all"],
        default=["all"],
        help="List of demos to run (default: all)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS + ["all"],
        default=["all"],
        help="Training backends to use (default: all)",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments to pass to mxtrain",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution if any training fails",
    )

    args = parser.parse_args()

    # Determine which demos to run
    if "all" in args.demos:
        demos_to_run = ALL_DEMOS
    else:
        demos_to_run = args.demos

    # Determine which backends to use
    if "all" in args.backends:
        backends_to_run = BACKENDS
    else:
        backends_to_run = args.backends

    print(f"Will run {len(demos_to_run)} demos with {len(backends_to_run)} backend(s)")
    print(f"Demos: {demos_to_run}")
    print(f"Backends: {backends_to_run}")
    print(f"Total training runs: {len(demos_to_run) * len(backends_to_run)}")

    failed_runs = []

    for demo in demos_to_run:
        for backend in backends_to_run:
            returncode = run_command(demo, backend, args.extra_args)

            if returncode != 0:
                failed_runs.append((demo, backend))
                if args.stop_on_error:
                    print("\n⚠️  Stopping due to error as requested")
                    sys.exit(1)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    total_runs = len(demos_to_run) * len(backends_to_run)
    successful_runs = total_runs - len(failed_runs)

    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {len(failed_runs)}")

    if failed_runs:
        print("\nFailed runs:")
        for demo, backend in failed_runs:
            print(f"  - {demo} ({backend})")
        sys.exit(1)
    else:
        print("\n✅ All training runs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
