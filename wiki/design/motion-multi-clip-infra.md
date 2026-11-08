# 多 Motion Clip 训练基础设施设计

## 摘要

把 `WbtMotionCommand` 从"单 clip 假设"中解耦，建立一套支撑多 motion clip 跟踪训练的通用基础设施：
数据平面 `MotionLibrary` 把多个 MotrixLab motion npz v1 文件拼接为带逐帧 clip 边界的统一运行时表示，
时间线平面由 `WbtMotionCommand` 原地泛化承担,课程平面第一版冻结现状。多 clip + 附加参考表示 + 到尾换段
的自适应跟踪任务（如 SONIC 类）此后作为 infra 的新消费者接入，而不是复制一份 command。

## 背景与动机

main 上唯一的 motion 消费者是 `WbtMotionCommand`，"单 clip"假设同时硬编码在三个位置：

- 数据：`WbtMotionClip` 由单个 MotrixMotion npz 构建，clip 边界是隐式的 `num_frames` 标量；
- 时间线：`advance()` 以 `steps >= num_frames` 判定到尾。值得注意的是到尾语义本身已经是多 clip 就绪的——
  `hold_at_clip_end` 钳末帧与 wrap（就地重采 + `sim_reset_requested` 的 sim-only 重置，不结束回合）两种策略
  都与"clip 内容"无关；
- 课程：失败计数 EMA + 全局 bin + CDF 的起始分布只覆盖单 clip 的帧轴。

[MotrixLab Motion NPZ Schema](./motrixlab-motion-npz-schema.md) 已为多 clip 预留扩展槽位：
"一个文件 = 一个 clip，dataset 层管多文件"。本设计实现该 dataset 层，并沿 schema 的 `ext_*` 机制承载
附加参考通道。

## 已确认的决策

1. `WbtMotionCommand` **原地泛化**为多 clip 时间线（不另建平行基类，历史包袱最小）；
2. 附加参考数据以**任务声明的 `ext_` 通道**进入，library 按需装载；
3. 多 clip npz **遵循现有 MotrixLab motion schema**（一文件一 clip），不发明新格式、不引入打包步骤；
   成对 retarget 等异构源经现有 converter 模式转为 schema v1 文件后进入；
4. 第一版**不修改 main 现有 sampling 设定**（计数 EMA + 全局 bin + CDF 原样运行于拼接语料），
   不新增投机字段（如 episode 起始记录等留待需要时再加）。

## 三层抽象

### 数据平面：`MotionLibrary`

位置 `motrix_envs/src/motrix_envs/motion/`，kernel-data dataclass 纯数值对象（torch tensor 字段）。

- 输入：有序的 MotrixMotion npz v1 文件列表（cfg 显式列表或目录展开，顺序固定可复现）；
- 逐文件：`MotrixMotion` 加载校验（schema、fps、finite、单位四元数）→ 按任务 joint/body 顺序重排 →
  装载任务声明的 `ext_` 通道（未声明的不装载）；
- 跨文件契约：fps 与控制步进一致、joint/body 名称集合一致、通道 shape 一致；
- 输出（拼接后的全局帧轴表示）：任务序的 `joint_pos/joint_vel` 与 tracked-body 四件套、
  root/reference 切片、**逐帧 `frame_clip_end`**（clip 边界的唯一运行时载体，per-clip 元数据可由它差分恢复，组装期的 lengths/offsets 不进入 clip）、
  声明的 `ext_` 通道拼接数组。

单 clip 即 N=1 特例（`frame_clip_end ≡ T-1`），现有 WBT 数据路径语义不变。

### 时间线平面：`WbtMotionCommand` 原地泛化

- `WbtMotionClip` 增加 `frame_clip_end` 字段；单文件构建时为常量末帧；
- `advance()` 边界测试从 `steps >= num_frames` 改为 `steps > frame_clip_end[step]`（末帧已被跟踪一步后才
  越界），`hold_at_clip_end` 与 wrap 的语义、`sim_reset_requested` 的 sim-only 重置路径保持 main 现状；
- 多 clip 下 wrap 的重采目标自然扩展为全语料 CDF；
- kernel 侧复用单元保持为模块级 `@njit` helper（起帧抽样、边界推进），`@dispatch` 方法是薄组合；
  子类可自由增加字段（manager 编译器按具体类编译）。

### 课程平面：v1 冻结现状

现有计数 EMA + 全局 bin + CDF 原样运行于拼接语料。已知的多 clip 统计缺口（裸计数的自增强偏置、
全局 bin 跨 clip 边界、无 per-clip 等权、平滑跨 clip 泄漏、缺访问归一）**不在本版解决**；
后续演进作为独立设计另立文档，届时以可替换的 host 侧 sampler 组件形态进入，kernel 只消费 CDF 的
契约在泛化时即成立。

## 通道声明

任务 cfg 声明所需 `ext_` 通道名列表（如 `("smpl_joints", "smpl_root_quat")`），library 按需装载并校验
shape 一致。SMPL 类双参考表示即此机制的应用：任务侧不再需要 motion clip 数据子类，也不需要专用
打包/存储格式。

`WbtMotionCommandCfg` 的动作源与通道声明写法（统一走 `motion_files`，单文件即长度为 1 的列表）：

```python
commands.motion.motion_files = (                     # 有序文件/目录列表（目录按排序展开）
    "assets/motion/g1/dance_a.npz",
    "assets/motion/g1/dance_b.npz",
)
commands.motion.extension_channels = ("smpl_joints", "smpl_root_quat")
```

kernel 侧经 `clip.extensions["smpl_joints"].data` 按帧索引读取（通道随命令进入 kernel 输入，
kernel 编译按声明的通道键集合区分）；未声明的通道不装载。

## 验收门

- **WBT 行为不变**：同 seed 下起始帧抽样序列与 main 逐位一致（`g1-wbt-dance` 回归）；
- **单 clip 等价**：`frame_clip_end ≡ T-1` 时泛化后的 advance/reset 与旧实现路径等价；
- 新增多 clip 单测：多文件 fixture 的边界/重排/通道装载/跨文件契约负向校验。

## 未来消费者示例（SONIC 类任务）

一个"多 clip + 双参考表示 + 到尾换段 + 自适应起始"的任务在此 infra 上的全部工作：
提供多文件 library（声明 SMPL 通道）、沿用内建的 wrap 换段语义、定义任务字段与重播种
（如编码器选择器、足加速度状态、低位姿放宽 EMA）及其终止/奖励项。没有 command 复制、
没有 clip 数据子类、没有打包脚本。

## 非目标

- 不改 `motrix_env_core`（motion 概念留在 `motrix_envs`）；
- 不修改 sampling 统计口径（v1 冻结，另立后续设计）；
- 不新增 npz 格式、不引入 packed/mmap store；
- 不做跨机器人 retarget、实时录制（沿用 schema 设计的排除项）。
