# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate registered-robot tables and optional robot screenshots."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import motrixsim as mtx
from motrixsim.render import RenderApp, RenderSettings

import motrix_envs  # noqa: F401 registers built-in robots
from motrix_env_core import registry
from motrix_env_core.config.scene import MjcfFileCfg, SystemCameraCfg, UrdfFileCfg
from motrix_env_motrixsim.compiler import build_scene_model
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg

REPO_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "robots"
START_MARKER = "<!-- ROBOT_TABLE_START -->"
END_MARKER = "<!-- ROBOT_TABLE_END -->"
SCREENSHOT_WIDTH = 800
SCREENSHOT_HEIGHT = 600
DEFAULT_SCREENSHOT_WARMUP_SECONDS = 0.5

_ROBOT_METADATA = {
    "anymal_c": {
        "kind": "quadruped",
        "lookat": (0.0, 0.0, 0.4),
        "distance": 1.6,
        "screenshot_warmup_seconds": 5.0,
    },
    "dex-evt": {"kind": "humanoid", "lookat": (0.0, 0.0, 0.9), "distance": 2.3},
    "g1-29dof": {"kind": "humanoid", "lookat": (0.0, 0.0, 0.9), "distance": 2.3},
    "go1": {"kind": "quadruped", "lookat": (0.0, 0.0, 0.3), "distance": 1.4},
    "go2": {"kind": "quadruped", "lookat": (0.0, 0.0, 0.3), "distance": 1.4},
    "k1": {"kind": "humanoid", "lookat": (0.0, 0.0, 0.8), "distance": 2.1},
    "microduck": {"kind": "humanoid", "lookat": (0.0, 0.0, 0.15), "distance": 0.8},
}

_DOC_CONFIGS = {
    "en": {
        "path": REPO_ROOT / "docs" / "source" / "en" / "user_guide" / "robots.md",
        "headers": ("Screenshot", "Registry name", "Configuration class", "Type", "Model format", "DoF"),
        "kinds": {"humanoid": "Humanoid", "quadruped": "Quadruped"},
    },
    "zh_CN": {
        "path": REPO_ROOT / "docs" / "source" / "zh_CN" / "user_guide" / "robots.md",
        "headers": ("截图", "Registry 名称", "配置类", "类型", "模型格式", "自由度"),
        "kinds": {"humanoid": "人形机器人", "quadruped": "四足机器人"},
    },
}


def _model_format(robot_cfg: object) -> str:
    model = robot_cfg.model
    if isinstance(model, MjcfFileCfg):
        return "MJCF"
    if isinstance(model, UrdfFileCfg):
        return "URDF"
    return type(model).__name__


def _robot_dof(robot_cfg: object) -> int:
    """Number of robot joint DoF, read from the built scene model.

    Sum the velocity DoF of every joint under the robot's base body, so the
    floating base (if any) and the rest of the scene are excluded.
    """
    scene = StandardSceneCfg(objs=StandardSceneObjsCfg(robot=robot_cfg))
    model = build_scene_model(scene)
    body = model.get_body(robot_cfg.resolved_base_link_name)
    return sum(joint.num_dof_vel for joint in body.joints)


def _registered_robot_rows(language: str) -> list[tuple[str, ...]]:
    registered = registry.list_registered_robots()
    if set(registered) != set(_ROBOT_METADATA):
        missing = sorted(set(registered) - set(_ROBOT_METADATA))
        removed = sorted(set(_ROBOT_METADATA) - set(registered))
        raise RuntimeError(f"Robot documentation metadata is stale: missing={missing}, removed={removed}")

    doc_config = _DOC_CONFIGS[language]
    kinds = doc_config["kinds"]

    rows = []
    for name, robot_meta in sorted(registered.items()):
        robot_cfg = registry.make_robot_config(name)
        metadata = _ROBOT_METADATA[name]
        screenshot = f'<img src="../_static/images/robots/{name}.png" alt="{name}" width="180">'
        rows.append(
            (
                screenshot,
                f"`{name}`",
                f"`{robot_meta['config_class']}`",
                kinds[metadata["kind"]],
                _model_format(robot_cfg),
                str(_robot_dof(robot_cfg)),
            )
        )
    return rows


def render_robot_table(language: str) -> str:
    """Render the registered robots as a localized Markdown table."""
    headers = _DOC_CONFIGS[language]["headers"]
    rows = _registered_robot_rows(language)
    lines = [
        "<!-- This table is generated; do not edit this block manually. -->",
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
    ]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return "\n".join(lines)


def _replace_generated_table(content: str, table: str, path: Path) -> str:
    if content.count(START_MARKER) != 1 or content.count(END_MARKER) != 1:
        raise RuntimeError(f"{path} must contain exactly one generated robot table marker pair")
    prefix, remainder = content.split(START_MARKER, maxsplit=1)
    _, suffix = remainder.split(END_MARKER, maxsplit=1)
    return f"{prefix}{START_MARKER}\n\n{table}\n\n{END_MARKER}{suffix}"


def generate_tables(*, check: bool) -> list[Path]:
    """Update robot tables, or return stale documents without changing them."""
    stale = []
    for language, doc_config in _DOC_CONFIGS.items():
        path = doc_config["path"]
        content = path.read_text(encoding="utf-8")
        generated = _replace_generated_table(content, render_robot_table(language), path)
        if generated == content:
            continue
        stale.append(path)
        if not check:
            path.write_text(generated, encoding="utf-8")
    return stale


def _resolve_screenshot_robot_names(robot_names: list[str] | None) -> list[str]:
    registered = set(registry.list_registered_robots())
    if not robot_names:
        return sorted(registered)

    unknown = sorted(set(robot_names) - registered)
    if unknown:
        raise ValueError(f"Unknown robot registry names: {unknown}")
    return list(dict.fromkeys(robot_names))


def generate_screenshots(robot_names: list[str] | None = None) -> None:
    """Render consistently framed screenshots for all or selected registered robots."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for name in _resolve_screenshot_robot_names(robot_names):
        metadata = _ROBOT_METADATA[name]
        camera = SystemCameraCfg(
            lookat=metadata["lookat"],
            distance=metadata["distance"],
            elevation=-15.0,
            azimuth=135.0,
        )
        scene = StandardSceneCfg(
            system_camera=camera,
            objs=StandardSceneObjsCfg(robot=registry.make_robot_config(name)),
        )
        model = build_scene_model(scene)
        data = mtx.SceneData(model, batch=[1])
        data.reset(model)
        model.forward_kinematic(data)
        model.cameras.set_system_render_target("image", SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT)

        renderer = RenderApp(headless=True)
        try:
            settings = RenderSettings.performance()
            settings.enable_shadow = True
            renderer.launch(
                model,
                batch=1,
                render_offset=[[0.0, 0.0, 0.0]],
                render_settings=settings,
            )
            renderer.system_camera.set_view(camera.lookat, camera.distance, camera.elevation, camera.azimuth)
            renderer.system_camera.active = True
            renderer.sync(data=data, wait=True)
            time.sleep(metadata.get("screenshot_warmup_seconds", DEFAULT_SCREENSHOT_WARMUP_SECONDS))
            capture = renderer.system_camera.capture()
            renderer.sync(data=data, wait=True)
            image = capture.take_image()
            if image is None:
                raise RuntimeError(f"Screenshot capture for robot {name!r} did not return an image")
            image.save_to_disk(str(SCREENSHOT_DIR / f"{name}.png"))
        finally:
            renderer.__exit__(None, None, None)


def _missing_screenshots() -> list[Path]:
    return [
        SCREENSHOT_DIR / f"{name}.png"
        for name in sorted(_ROBOT_METADATA)
        if not (SCREENSHOT_DIR / f"{name}.png").is_file()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated documentation is stale")
    parser.add_argument(
        "--screenshots",
        nargs="*",
        metavar="ROBOT",
        help="regenerate all robot screenshots, or only the listed registry names",
    )
    args = parser.parse_args()

    if args.check and args.screenshots is not None:
        parser.error("--check and --screenshots are mutually exclusive")
    if args.screenshots is not None:
        try:
            generate_screenshots(args.screenshots)
        except ValueError as exc:
            parser.error(str(exc))

    stale_docs = generate_tables(check=args.check)
    missing_screenshots = _missing_screenshots()
    if args.check and (stale_docs or missing_screenshots):
        for path in stale_docs:
            print(f"stale generated robot table: {path.relative_to(REPO_ROOT)}")
        for path in missing_screenshots:
            print(f"missing robot screenshot: {path.relative_to(REPO_ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
