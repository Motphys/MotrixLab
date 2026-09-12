# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_env_snapshot.py"
MODULE_SPEC = importlib.util.spec_from_file_location("generate_env_snapshot", SCRIPT_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
generate_env_snapshot = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(generate_env_snapshot)


class _FakeSnapshotRenderer:
    def __init__(self):
        self.render_calls = 0
        self.capture_count = 0
        self.close_calls = 0

    def render(self):
        self.render_calls += 1

    def capture(self):
        self.capture_count += 1
        return f"frame-{self.capture_count}"

    def close(self):
        self.close_calls += 1


def test_save_snapshot_captures_one_overview_with_env_camera(monkeypatch, tmp_path):
    renderer = _FakeSnapshotRenderer()
    created = []

    def fake_create_renderer(config):
        created.append(config)
        return renderer

    written = []
    monkeypatch.setattr(generate_env_snapshot.imageio, "imwrite", lambda out, frame: written.append((out, frame)))
    monkeypatch.setattr(generate_env_snapshot.time, "sleep", lambda _: None)

    env = SimpleNamespace(
        cfg=SimpleNamespace(
            scene=SimpleNamespace(
                system_camera=SimpleNamespace(
                    lookat=(1.0, 2.0, 3.0),
                    distance=4.0,
                    elevation=-15.0,
                    azimuth=135.0,
                )
            )
        ),
        create_renderer=fake_create_renderer,
    )
    path = tmp_path / "cartpole.jpg"

    result = generate_env_snapshot._save_snapshot(env, path, width=960, height=540, warmup_seconds=1.0)

    assert len(created) == 1
    config = created[0]
    assert config.headless is True
    assert (config.width, config.height) == (960, 540)
    assert config.camera_lookat == (1.0, 2.0, 3.0)
    assert config.camera_distance == 4.0
    assert config.camera_elevation == -15.0
    assert config.camera_azimuth == 135.0
    assert renderer.render_calls == 1
    assert renderer.capture_count == 2
    assert result == path
    assert written == [(str(path), "frame-2")]
    assert renderer.close_calls == 1


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 480, 280, 1.0), "num_envs must be positive"),
        ((16, 0, 280, 1.0), "width must be positive"),
        ((16, 480, 0, 1.0), "height must be positive"),
        ((16, 480, 280, -1.0), "warmup_seconds must be non-negative"),
    ],
)
def test_validate_options_rejects_invalid_values(args, message):
    with pytest.raises(ValueError, match=message):
        generate_env_snapshot._validate_options(*args)


def test_generate_snapshot_initializes_registered_np_env(monkeypatch, tmp_path):
    class FakeArrayEnv:
        pass

    env = FakeArrayEnv()
    env.init_state = lambda: setattr(env, "initialized", True)
    monkeypatch.setattr(generate_env_snapshot, "ArrayEnv", FakeArrayEnv)
    monkeypatch.setattr(generate_env_snapshot.registry, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(generate_env_snapshot, "_save_snapshot", lambda candidate, path, **kwargs: path)

    path = generate_env_snapshot.generate_snapshot("cartpole", output_dir=tmp_path)

    assert env.initialized is True
    assert path == tmp_path / "cartpole.jpg"


def test_generate_snapshot_requires_force_to_overwrite_existing_image(monkeypatch, tmp_path):
    path = tmp_path / "cartpole.jpg"
    path.write_bytes(b"existing")
    make_calls = []
    monkeypatch.setattr(
        generate_env_snapshot.registry,
        "make",
        lambda *args, **kwargs: make_calls.append((args, kwargs)),
    )

    assert generate_env_snapshot.generate_snapshot("cartpole", output_dir=tmp_path) == path
    assert make_calls == []

    class FakeArrayEnv:
        def init_state(self):
            pass

    env = FakeArrayEnv()
    monkeypatch.setattr(generate_env_snapshot, "ArrayEnv", FakeArrayEnv)
    monkeypatch.setattr(generate_env_snapshot.registry, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(generate_env_snapshot, "_save_snapshot", lambda candidate, output, **kwargs: output)

    assert generate_env_snapshot.generate_snapshot("cartpole", output_dir=tmp_path, force=True) == path


def test_generate_missing_snapshots_skips_existing_env_ids(monkeypatch, tmp_path):
    existing = tmp_path / "env-b.jpg"
    existing.write_bytes(b"existing")
    monkeypatch.setattr(
        generate_env_snapshot.registry,
        "list_registered_envs",
        lambda: {"env-c": {}, "env-a": {}, "env-b": {}},
    )
    calls = []

    def generate(env_name, **kwargs):
        calls.append((env_name, kwargs))
        return kwargs["output_dir"] / f"{env_name}.jpg"

    monkeypatch.setattr(generate_env_snapshot, "generate_snapshot", generate)

    generated, skipped = generate_env_snapshot.generate_missing_snapshots(output_dir=tmp_path)

    assert [name for name, _ in calls] == ["env-a", "env-c"]
    assert generated == [tmp_path / "env-a.jpg", tmp_path / "env-c.jpg"]
    assert skipped == [existing]
    assert all(kwargs["output_dir"] == tmp_path for _, kwargs in calls)
