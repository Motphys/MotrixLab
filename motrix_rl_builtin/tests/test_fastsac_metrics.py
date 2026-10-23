# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl_builtin.fastsac.async_impl.metrics import FastSacMetrics

from motrix_rl.execution.training import WorkerReports


def collector(collector_id=0):
    return {
        "collector_id": collector_id,
        "return": 3.0,
        "ep_len": 4.0,
        "episodes": 1,
        "reward_terms": {"task": 1.0},
        "env_metrics": {"success": 0.5},
        "policy_lag": 2,
        "timing_ms": {"collect": 1.0},
    }


def learner(rank=0, *, step=1, learning_iteration=1, started=True):
    return {
        "rank": rank,
        "step": step,
        "learning_iteration": learning_iteration,
        "update_idx": 7 + rank,
        "updates": 7 + rank,
        "training_started": started,
        "metrics": {"critic/loss": 2.0 + rank},
        "learn_ms": 2.0,
        "learn_pct": 10.0,
        "learner_timing": {},
        "buffer_size": 4,
        "buffer_capacity": 8,
        "checkpoint_path": None,
    }


def report(*messages):
    return WorkerReports({str(i): (message,) for i, message in enumerate(messages)})


def test_first_update_gate_and_short_run_override():
    metrics = FastSacMetrics(
        learning_starts=2,
        logging_interval=10,
        resume_step=0,
        resume_learning_iteration=0,
        num_envs=4,
        num_collectors=1,
        num_learners=1,
        budget=10,
    )
    metrics.consume(
        report(
            {"type": "collector_stats", "stats": collector()},
            {"type": "learner_payload", "payload": learner(started=False)},
        )
    )
    assert metrics.frame(1.0, collector_steps=1).valid is False

    short = FastSacMetrics(
        learning_starts=100,
        logging_interval=10,
        resume_step=0,
        resume_learning_iteration=0,
        num_envs=4,
        num_collectors=1,
        num_learners=1,
        budget=1,
    )
    short.consume(
        report(
            {"type": "collector_stats", "stats": collector()},
            {"type": "learner_payload", "payload": learner(step=1, started=False)},
        )
    )
    assert short.frame(1.0, collector_steps=1).valid is True


def test_rank_zero_coordinates_scalars_and_force_final():
    metrics = FastSacMetrics(
        learning_starts=2,
        logging_interval=100,
        resume_step=0,
        resume_learning_iteration=0,
        num_envs=4,
        num_collectors=2,
        num_learners=2,
        budget=10,
    )
    metrics.consume(
        WorkerReports(
            {
                "collector-0": ({"type": "collector_stats", "stats": collector(0)},),
                "collector-1": ({"type": "collector_stats", "stats": collector(1)},),
                "learner-0": ({"type": "learner_payload", "payload": learner(0, step=3, learning_iteration=3)},),
                "learner-1": ({"type": "learner_payload", "payload": learner(1, step=3, learning_iteration=99)},),
            }
        )
    )
    frame = metrics.frame(2.0, collector_steps=6, force=True)
    assert frame.valid is True
    assert frame.panel.learning_iteration == 3
    assert frame.panel.update_idx == 7
    assert frame.step == 3
    assert frame.env_steps == 12
    assert frame.scalars["progress/learning_iteration"] == 3
    assert frame.panel.elapsed_seconds == 0.0
    assert frame.panel.buffer_size == 4
    assert frame.scalars["async/policy_lag_collector1"] == 2
    assert frame.scalars["perf/collect_ms_per_batch"] == 1.0
    assert frame.scalars["perf/learn_ms_total"] == 2.0
    metrics.consume(
        WorkerReports(
            {
                "learner-0": (
                    {
                        "type": "learner_payload",
                        "payload": {
                            "rank": 0,
                            "learner_counters": {
                                "learning_iteration": 4,
                                "update_idx": 9,
                                "updates": 9,
                                "metrics": None,
                            },
                        },
                    },
                )
            }
        )
    )
    final = metrics.frame(3.0, collector_steps=7, force=True)
    assert final.step == 4
    assert final.env_steps == 14
    assert final.panel.buffer_size == 4
    assert final.panel.elapsed_seconds == 1.0
    assert final.panel.update_idx == 9
    assert "train/critic/loss" not in final.scalars
    assert final.scalars["perf/updates_per_s"] == 2.0


def test_checkpoint_event_and_warmup_event():
    metrics = FastSacMetrics(
        learning_starts=2,
        logging_interval=10,
        resume_step=0,
        resume_learning_iteration=0,
        num_envs=2,
        num_collectors=1,
        num_learners=1,
        budget=10,
    )
    metrics.consume(
        WorkerReports(
            {
                "c": ({"type": "collector_stats", "stats": collector()},),
                "l": (
                    {"type": "learner_payload", "payload": learner(step=1, learning_iteration=1)},
                    {"type": "checkpoint_saved", "step": 1, "path": "/tmp/model.pt"},
                ),
            }
        )
    )
    first = metrics.frame(1.0, collector_steps=1, force=True)
    assert any(event.name == "checkpoint_saved" for event in first.events)

    metrics.consume(
        WorkerReports({"l": ({"type": "learner_payload", "payload": learner(step=2, learning_iteration=2)},)})
    )
    second = metrics.frame(2.0, collector_steps=2, force=True)
    assert any(event.name == "warmup_done" for event in second.events)
