# Motion File Format

WBT uses MotrixLab Motion NPZ v1 for reference motions. One `.npz` contains one clip, represented by explicitly named
NumPy arrays for per-frame joint state, body pose, and velocity. Name-based binding allows motion columns to use a different
order from the robot model.

## Coordinates and timing

-   The current `schema_version` is `1`.
-   The world frame is right-handed and Z-up.
-   Quaternions use **xyzw** order, not wxyz.
-   The loader converts per-frame floating-point arrays to `float32`.
-   `fps` is the default playback rate used by `replay.py`.

The WBT environment advances one stored motion frame per control step and does not currently resample from `fps`. To preserve
the source motion speed, use

$$
fps=\frac{1}{ctrl\_dt}.
$$

Built-in WBT configs use `ctrl_dt=0.02` s, and all bundled motions are therefore 50 FPS. Resample other frame rates during
conversion or adjust the task `ctrl_dt` consistently.

## Required fields

Let $T=num\_frames$, $N=len(joint\_names)$, and $B=len(body\_names)$:

| Field            | Shape            | Type    | Meaning                                                                                   |
| ---------------- | ---------------- | ------- | ----------------------------------------------------------------------------------------- |
| `schema_version` | Scalar or `(1,)` | integer | Schema version, currently `1`                                                             |
| `fps`            | Scalar or `(1,)` | integer | Motion frame rate; must be positive                                                       |
| `num_frames`     | Scalar or `(1,)` | integer | Frame count; the generic loader accepts 1 or more, while WBT training requires at least 2 |
| `joint_names`    | `(N,)`           | string  | Column names for `joint_pos` and `joint_vel`                                              |
| `body_names`     | `(B,)`           | string  | Body-column names for every `body_*` array                                                |
| `joint_pos`      | `(T,N)`          | float   | Joint position                                                                            |
| `joint_vel`      | `(T,N)`          | float   | Joint velocity                                                                            |
| `body_pos_w`     | `(T,B,3)`        | float   | Body position in world coordinates                                                        |
| `body_quat_w`    | `(T,B,4)`        | float   | Body orientation in world coordinates, xyzw                                               |
| `body_lin_vel_w` | `(T,B,3)`        | float   | Body linear velocity in world coordinates                                                 |
| `body_ang_vel_w` | `(T,B,3)`        | float   | Body angular velocity in world coordinates                                                |

Every quaternion in `body_quat_w` must be normalized. The loader's default norm tolerance is `1e-3`.

## Optional fields

| Field                 | Meaning                                                                  |
| --------------------- | ------------------------------------------------------------------------ |
| `tracked_body_names`  | Motion-suggested tracked-body subset                                     |
| `reference_body_name` | Suggested reference body                                                 |
| `root_body_name`      | Body corresponding to the floating root                                  |
| `clip_name`           | Human-readable motion name                                               |
| `ext_*`               | Extension arrays, exposed through `extensions` without the `ext_` prefix |

WBT training treats `WbtManagerEnvCfg.tracked_body_names`, `reference_body_name`, and the robot `base_link_name` as authoritative;
optional fields in the NPZ do not change task semantics. If `root_body_name` is absent, `replay.py` uses `body_names[0]`.

## Name binding and validation

The motion must contain every controlled joint in the target model, plus the tracked bodies, reference body, and robot base
link required by the WBT config. Names are case-sensitive and underscore-sensitive. Moving columns without updating
`joint_names` or `body_names` produces incorrect bindings.

Load a file directly for schema validation:

```python
from motrix_envs.motion import MotrixMotion

motion = MotrixMotion("/path/to/motion.npz")
print(motion.fps, motion.num_frames)
print(motion.joint_names)
print(motion.body_names)
```

`MotrixMotion` checks required fields, scalar ranges, array shapes, and quaternion norms. After schema validation, use the
target robot to check the floating-root layout, model joint set, and actual kinematics:

```bash
python scripts/motion/replay.py --robot g1-29dof --motion /path/to/motion.npz
```

Supported replay `--robot` values are `g1-29dof`, `dex-evt`, and `k1`.

## Convert public LAFAN data

List available G1 clips:

```bash
python scripts/motion/download_lafan.py --list
```

Download and convert one clip to 50 FPS:

```bash
python scripts/motion/download_lafan.py \
  --motion dance1_subject1 \
  --output motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject1.npz \
  --output-fps 50
```

You can also convert an existing G1 CSV:

```bash
python scripts/motion/convert.py \
  --from lafan \
  --input /path/to/dance1_subject1.csv \
  --output /path/to/dance1_subject1.npz \
  --input-fps 30 --output-fps 50
```

Use `--start-sec` and `--end-sec` to trim the clip. The converter runs forward kinematics with the target robot model to
generate the `body_*` arrays required by WBT. LAFAN1 uses the CC BY-NC-ND 4.0 license; verify that its terms suit your use case.

## Pre-training checks

-   `MotrixMotion` loads the file, and the WBT clip contains at least 2 frames.
-   Quaternions use xyzw order and are normalized.
-   `fps` equals the task's `1 / ctrl_dt`.
-   The root trajectory is continuous, without jumps, flips, or unexpected drift.
-   Joint and body names exactly match the target robot and WBT config.
-   Limb sides, joint directions, and contact locations look correct in replay.
-   Motion joint ranges do not obviously exceed model joint limits.

After validation, follow [Adding a WBT Training Task](adding_wbt_task.md) to register the Env ID and training configs.
