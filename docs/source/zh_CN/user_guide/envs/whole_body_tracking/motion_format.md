# Motion 文件格式

WBT 使用 MotrixLab Motion NPZ v1 作为参考动作格式。一个 `.npz` 文件只包含一个 clip，以名称明确的 NumPy
数组保存逐帧关节状态、身体位姿和速度。名称绑定使 motion 的数组列顺序可以不同于机器人模型顺序。

## 坐标与时间约定

-   `schema_version` 当前必须为 `1`。
-   世界坐标系为右手 Z-up。
-   四元数顺序为 **xyzw**，不是 wxyz。
-   Loader 将逐帧浮点数组转换为 `float32`。
-   `fps` 是 `replay.py` 的默认播放帧率。

WBT 环境在每个控制周期推进一个已存储的 motion 帧，当前不会根据 `fps` 自动重采样。为了保持原始动作速度，motion 的
帧率应满足

$$
fps=\frac{1}{ctrl\_dt}.
$$

内置 WBT 配置使用 `ctrl_dt=0.02` s，因此内置 motion 均为 50 FPS。其他帧率应在转换阶段重采样，或与任务的
`ctrl_dt` 同步调整。

## 必需字段

令 $T=num\_frames$、$N=len(joint\_names)$、$B=len(body\_names)$：

| 字段             | 形状          | 类型    | 含义                                                     |
| ---------------- | ------------- | ------- | -------------------------------------------------------- |
| `schema_version` | 标量或 `(1,)` | integer | Schema 版本，当前为 `1`                                  |
| `fps`            | 标量或 `(1,)` | integer | Motion 帧率，必须大于 0                                  |
| `num_frames`     | 标量或 `(1,)` | integer | 总帧数；通用 loader 要求至少 1 帧，WBT 训练要求至少 2 帧 |
| `joint_names`    | `(N,)`        | string  | `joint_pos`、`joint_vel` 的列名                          |
| `body_names`     | `(B,)`        | string  | 所有 `body_*` 数组的身体列名                             |
| `joint_pos`      | `(T,N)`       | float   | 关节位置                                                 |
| `joint_vel`      | `(T,N)`       | float   | 关节速度                                                 |
| `body_pos_w`     | `(T,B,3)`     | float   | 世界坐标系中的身体位置                                   |
| `body_quat_w`    | `(T,B,4)`     | float   | 世界坐标系中的身体姿态，xyzw                             |
| `body_lin_vel_w` | `(T,B,3)`     | float   | 世界坐标系中的身体线速度                                 |
| `body_ang_vel_w` | `(T,B,3)`     | float   | 世界坐标系中的身体角速度                                 |

`body_quat_w` 中每个四元数都必须归一化；loader 默认允许的范数误差为 `1e-3`。

## 可选字段

| 字段                  | 含义                                                   |
| --------------------- | ------------------------------------------------------ |
| `tracked_body_names`  | Motion 建议的跟踪身体子集                              |
| `reference_body_name` | 建议的参考身体名称                                     |
| `root_body_name`      | 浮动根对应的身体名称                                   |
| `clip_name`           | 人类可读的动作名称                                     |
| `ext_*`               | 扩展数组；loader 以去掉 `ext_` 的名称放入 `extensions` |

WBT 训练以 `WbtManagerEnvCfg.tracked_body_names`、`reference_body_name` 和机器人 `base_link_name` 为最终配置来源，不会
因为 NPZ 中存在同名可选字段而修改任务语义。`replay.py` 在 `root_body_name` 缺失时使用 `body_names[0]`。

## 名称绑定与验证

Motion 必须包含目标模型的所有受控 joint，以及 WBT 配置需要的 tracked body、reference body 和机器人 base link。
名称区分大小写和下划线。只移动数组列而不同时更新 `joint_names` 或 `body_names` 会产生错误绑定。

可以直接加载文件完成 schema 检查：

```python
from motrix_envs.motion import MotrixMotion

motion = MotrixMotion("/path/to/motion.npz")
print(motion.fps, motion.num_frames)
print(motion.joint_names)
print(motion.body_names)
```

`MotrixMotion` 检查必需字段、标量范围、数组形状和四元数范数。Schema 通过后，再使用目标机器人检查浮动根布局、
模型 joint 集合和实际运动学效果：

```bash
python scripts/motion/replay.py --robot g1-29dof --motion /path/to/motion.npz
```

Replay 支持的 `--robot` 值为 `g1-29dof`、`dex-evt` 和 `k1`。

## 转换公开 LAFAN 数据

查看可用的 G1 clip：

```bash
python scripts/motion/download_lafan.py --list
```

下载并转换到 50 FPS：

```bash
python scripts/motion/download_lafan.py \
  --motion dance1_subject1 \
  --output motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject1.npz \
  --output-fps 50
```

也可以转换已经下载的 G1 CSV：

```bash
python scripts/motion/convert.py \
  --from lafan \
  --input /path/to/dance1_subject1.csv \
  --output /path/to/dance1_subject1.npz \
  --input-fps 30 --output-fps 50
```

`--start-sec` 和 `--end-sec` 用于裁剪片段。Converter 使用目标机器人模型运行正向运动学，生成 WBT 所需的
`body_*` 数组。LAFAN1 数据集采用 CC BY-NC-ND 4.0 许可，使用前应确认许可条件适合目标场景。

## 接入训练前检查

-   `MotrixMotion` 可以加载文件，且 WBT clip 至少包含 2 帧。
-   四元数使用 xyzw 顺序并归一化。
-   `fps` 与任务的 `1 / ctrl_dt` 一致。
-   根轨迹连续，没有突然跳变、翻转或明显漂移。
-   Joint/body 名称与目标机器人和 WBT 配置完全一致。
-   Replay 中左右肢体、关节方向和接触位置正确。
-   Motion 关节范围没有明显超出模型 joint limit。

完成检查后，按照[新增 WBT 训练任务](adding_wbt_task.md)注册 Env ID 和训练配置。
