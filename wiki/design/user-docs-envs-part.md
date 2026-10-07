# 用户文档 Envs 栏目设计

## 摘要

本文定义 MotrixLab 用户文档中独立 `Envs` 栏目的信息架构、内容边界、页面模板和渐进迁移方式。`Envs` 位于 `Tutorials` 之后，以任务主题组织环境，其中 Whole Body Tracking、Humanoid Locomotion 和 Quadruped Locomotion 是三个独立主题；其首页根据 Env Registry 和 Hydra Task 配置自动生成包含 Env ID、本地化描述和当前支持训练算法的 overview 表格。第一阶段基于现有 `docs/source/{zh_CN,en}/user_guide` 继续演进：把当前平铺在 `demo/` 下的 24 个环境介绍组织成可发现的环境目录，并把现有 Whole Body Tracking 专题整体迁入对应环境主题，同时保持入门、其余训练教程和 Robot 参考的既有职责。设计先建立导航和主题，再统一页面内容，最后补充旧 URL 重定向，避免一次性重做整个用户文档。

## 1. 目标与范围

### 1.1 目标

`Envs` 是用户指南中的一级栏目，主要回答以下问题：

1. MotrixLab 当前提供哪些可直接使用的环境；
2. 每个环境解决什么任务，使用哪个环境 ID；
3. 环境的 observation、action、reward、termination 和 reset 语义是什么；
4. 如何预览环境，以及当前支持哪些训练算法和 Training Task；
5. 环境实现、配置和相关 Robot 文档位于哪里。

该栏目应让用户从用户指南首页出发，在两次点击内到达任意内置环境页面，并能根据任务领域选择合适的环境。

### 1.2 第一阶段范围

第一阶段只整理现有环境介绍，不同时重构整套用户文档：

- 新增 `Envs` 一级导航、栏目首页和主题页；
- 自动生成包含全部已注册 Env ID、描述和当前支持训练算法的 overview 表格；
- 对现有 24 个 `demo/*.md` 页面建立完整主题归属；
- 定义统一的环境页面内容模板；
- 更新入门文档中指向“训练示例”的入口；
- 中文和英文目录保持相同结构；
- 保留现有安装、其余教程和 Robot 页面位置，并把 WBT 专题整体迁入 `envs/whole_body_tracking/`。

### 1.3 非目标

以下内容不在第一阶段处理：

- 不重新设计安装和快速入门文档；
- 除 WBT 专题内现有训练流程外，不迁移其他训练配置、训练执行、runs/checkpoint 等教程；
- 不把 Robot 页面合并进 `Envs`；
- 不在本阶段设计完整的自定义环境开发教程；
- 不调整 Sphinx 的多语言构建机制；
- 不以代码包目录机械替代面向用户的环境主题。

## 2. 概念与内容边界

MotrixLab 的环境定义层与 RL 框架保持解耦，具体架构见[项目整体架构设计](./architecture.md)。用户文档也应保持相同的语义边界。

| 概念 | 文档职责 | 不负责的内容 |
| --- | --- | --- |
| Environment | 定义一个可通过 registry 创建的仿真任务，包括 scene、observation、action、reward、termination 和 reset | 不定义训练算法和算法超参数 |
| Robot | 可复用的机器人模型与默认配置，可被多个 Environment 组合使用 | 不代表一个可训练任务，不拥有任务 marker、reward 或 termination |
| Training Task | 通过 Hydra 组合 Environment、RL framework、算法和训练参数的可复现训练配方 | 不改变 Environment 本身的任务语义 |
| Tutorial workflow | 串联数据准备、Environment 配置和训练步骤的操作教程 | 不替代 Environment 的任务语义与 Env ID 目录 |

Scene、Robot、asset 和任务场景对象的配置边界见 [Scene / Robot 配置设计](./scene-model-config.md)。`Envs` 页面可以链接相关 Robot 和 Training Task，但不应把三者描述成同一个对象。

Whole Body Tracking 的环境说明和操作流程集中放在 `envs/whole_body_tracking/`。该主题既介绍当前有哪些 WBT Environment、每个 Env ID 对应的 robot、motion 变体和任务语义，也保留 motion 准备、格式说明、新增 WBT 任务和训练操作步骤。`tutorial/` 下不再保留重复的 WBT 入口。

## 3. 信息架构

### 3.1 用户指南一级导航

在现有 `user_guide` 结构中增加独立的 `Envs` part，导航顺序为：

```text
入门指南 / Getting Started
使用教程 / Tutorials
Envs / Environments
```

当前“训练示例”中的页面实际主要描述环境的任务定义、动作空间、奖励和运行方式，因此其导航职责由 `Envs` 接管。训练教程仍留在 `tutorial/`，两者不混合。

### 3.2 目标目录结构

最终目标是在两个语言目录下建立完全对称的结构：

```text
docs/source/<language>/user_guide/
├── index.md
├── getting_started/
├── envs/
│   ├── index.md
│   ├── basic/
│   │   ├── index.md
│   │   ├── acrobot.md
│   │   ├── cartpole.md
│   │   ├── pendulum.md
│   │   └── stewart.md
│   ├── dm_control/
│   │   ├── index.md
│   │   ├── dm_cheetah.md
│   │   ├── dm_finger.md
│   │   ├── dm_hopper.md
│   │   ├── dm_humanoid.md
│   │   ├── dm_lqr.md
│   │   ├── dm_point_mass.md
│   │   ├── dm_quadruped.md
│   │   ├── dm_reacher.md
│   │   ├── dm_walker.md
│   │   └── bring_ball.md
│   ├── whole_body_tracking/
│   │   ├── index.md
│   │   ├── overview.md
│   │   ├── motion_format.md
│   │   └── adding_wbt_task.md
│   ├── humanoid_locomotion/
│   │   ├── index.md
│   │   ├── g1.md
│   │   ├── dex_evt.md
│   │   └── k1.md
│   ├── quadruped_locomotion/
│   │   ├── index.md
│   │   ├── go1.md
│   │   ├── go1_rough_terrain.md
│   │   ├── go2.md
│   │   └── anymal_c.md
│   └── manipulation/
│       ├── index.md
│       ├── bounce_ball.md
│       ├── franka_lift_cube.md
│       ├── franka_open_cabinet.md
│       ├── rm65_insert_peg.md
│       ├── rm65_open_cabinet.md
│       └── shadow_hand_repose.md
├── tutorial/
└── robots.md
```

目录名采用稳定的英文 slug，中文和英文站点只翻译页面标题与导航文案，不翻译路径。

### 3.3 主题原则

环境按用户理解的任务主题组织，而不是严格照搬 `motrix_envs` 的 Python package：

- `basic`：小型控制、平衡和系统稳定化任务，适合作为入门或算法验证环境；
- `dm_control`：移植自 DeepMind Control Suite 的环境，便于按同一 benchmark 家族查找；
- `whole_body_tracking`：由参考 motion 驱动的全身动作跟踪环境，包括 G1、Dex-EVT 和 K1 变体；
- `humanoid_locomotion`：人形机器人的速度跟踪、平地移动和复杂地形移动环境；
- `quadruped_locomotion`：四足机器人的速度跟踪、导航、粗糙地形和台阶环境；
- `manipulation`：机械臂、灵巧手与物体交互任务。

每个 Env ID 有且只有一个主主题。跨主题的 Robot 复用和相近任务通过普通链接表达，不在多个 `toctree` 中重复收录同一个详情页面。WBT 的主主题由任务目标决定，因此 `g1-wbt-dance` 属于 Whole Body Tracking，而不是 Humanoid Locomotion。

DeepMind Control Suite 的 Humanoid 和 Quadruped 仍归入 `dm_control`。它们用于 benchmark 兼容与算法验证，不与 G1、Dex-EVT、K1、Go1、Go2、ANYmal 等机器人任务混在 Humanoid/Quadruped Locomotion 主题中。

### 3.4 现有页面归类

| 主题 | 现有页面或新增内容 |
| --- | --- |
| Basic | `acrobot`、`cartpole`、`pendulum`、`stewart` |
| DM Control | `dm_walker`、`dm_cheetah`、`dm_hopper`、`dm_reacher`、`dm_lqr`、`dm_finger`、`dm_humanoid`、`dm_quadruped`、`dm_point_mass`、`bring_ball` |
| Whole Body Tracking | 将现有 `tutorial/whole_body_tracking/` 整体迁入该主题，覆盖内置 G1、Dex-EVT、K1 任务、motion 格式和新增 WBT Task 流程 |
| Humanoid Locomotion | `locomotion_unitree_g1`；后续补充 Dex-EVT、K1 环境页 |
| Quadruped Locomotion | `locomotion_unitree_go1`、`locomotion_unitree_go1_rough_terrain`、`anymal_c`；后续补充 Go2 环境页 |
| Manipulation | `bounce_ball`、`franka_lift_cube`、`franka_open_cabinet`、`rm65_open_cabinet`、`rm65_insert_peg`、`shadow_hand_repose` |

`bring_ball` 归入 DM Control，因为用户首先需要识别其 benchmark 家族；`shadow_hand_repose` 归入 Manipulation，不为单个页面单独建立 Dexterous 主题。未来同类环境数量增加后，可以在不改变页面语义的前提下拆出新主题。

三个机器人任务主题对应的主要 Env ID 为：

| 主题 | Env ID 范围 |
| --- | --- |
| Whole Body Tracking | `g1-29dof-wbt-*`、`dex-evt-wbt-*`、`k1-wbt-*` |
| Humanoid Locomotion | `g1-walk-*`、`dex-evt-walk-*`、`k1-walk-*` |
| Quadruped Locomotion | `go1-*`、`go2-*`、`anymalc-*`、`anymal_c_*` |

## 4. Envs 栏目首页

`envs/index.md` 不是 24 个链接的简单列表，应承担环境发现入口的职责，包含以下内容：

1. Environment、Robot 和 Training Task 的简短区别；
2. 使用 `uv run scripts/view.py env=<env-id>` 预览环境的统一方式；
3. 六个环境主题的卡片入口；
4. 自动生成的 Env overview 表格；
5. 指向环境开发教程、Robot 列表和训练教程的交叉链接。

### 4.1 Overview 表格

overview 表格保留三个稳定字段：

| 字段 | 含义 |
| --- | --- |
| Env ID | Env Registry 中可传给 `env=...` 的完整注册名称 |
| Description | 对该 ID 所代表任务或变体的一句话描述 |
| Training Algorithms | 该 Env 当前存在 Hydra Task 配方的 `rllib.algo` 集合 |

表格按 Env ID 字典序排列，每个注册 ID 独占一行。多个 Env ID 可以共用同一个环境详情页，但 description 必须说明变体差异，例如 walk、run、rough terrain 或 WBT motion。表格枚举 registry 中全部公共内置 Env ID，不以现有 24 个详情页是否完整为过滤条件。

`Training Algorithms` 使用 Task 选项中的 `rllib.algo` 形式，例如 `skrl.ppo`、`rslrl.ppo` 和 `motrix.fastsac`。同一算法的 backend-specific Task，例如 `skrl.ppo.jax` 与 `skrl.ppo.torch`，在 overview 中合并为一个 `skrl.ppo`；overview 展示算法支持，不展开训练后端或同步/异步执行拓扑。一个 Env 没有任何 Task 配方时显示 `—`，不能因为某个 framework 全局注册了算法就声称该 Env 已支持训练。

### 4.2 自动生成机制

新增 `docs/scripts/generate_env_docs.py`，采用现有 `generate_robot_docs.py` 的标记区块模式。中英文 `envs/index.md` 各保留一对标记：

```markdown
<!-- ENV_OVERVIEW_TABLE_START -->
<!-- generated content -->
<!-- ENV_OVERVIEW_TABLE_END -->
```

生成流程为：

1. 导入 `motrix_envs`，触发全部内置 Environment 注册；
2. 调用 `registry.list_registered_envs()` 获取 Env ID，以及从注册 provider docstring 自动提取的双语 description；
3. 校验每个已注册环境都具有英文首行和 `zh_CN:` 行；
4. 使用与 Hydra Task 测试一致的发现规则枚举 `configs/task/<env-id>/` 下的 `rllib.algo[.backend]` 选项；
5. 将 Task 选项投影为去除 backend 后缀的唯一 `rllib.algo` 集合，并按名称排序；
6. 拒绝指向未注册 Env ID 的 Task 目录；
7. 按 Env ID 排序，为 `zh_CN` 和 `en` 渲染三列表格；
8. 只替换两个 marker 之间的生成内容，保留 `index.md` 中的人工说明和主题入口。

环境 description 与 Python 环境定义放在一起，不在生成脚本中维护 Env ID 清单。注册 class 或 factory 使用标准 docstring：首个非空行是英文 description，另加一个 `zh_CN:` 行提供中文 description。`registry.envcfg` 在注册时自动提取这两项，`list_registered_envs()` 通过 `description` 字段暴露给文档工具。普通 `#` 注释不会保留在运行时，因此不作为事实来源。

```python
@registry.envcfg("cartpole")
@configclass
class CartPoleEnvCfg(EnvCfg):
    """Move a cart to keep an inverted pendulum upright.

    zh_CN: 移动小车以保持倒立摆直立。
    """
```

Training Algorithms 不写入 Env 定义，而是完全根据当前 Hydra Task 选项生成，避免产生第二份训练能力清单。新增、重命名或删除 Env 后，只需同步 Python provider docstring 和真实 Task 配置；生成脚本本身不需要修改。缺少任一语言描述时，生成器必须直接报错，不能静默遗漏。

生成器提供两种模式：

```bash
# 更新中英文 index.md 中的 overview 表格
uv run python docs/scripts/generate_env_docs.py

# 检查已提交表格是否与 registry description 和 Task 配置一致，不写文件
uv run python docs/scripts/generate_env_docs.py --check
```

文档构建入口在运行 Sphinx 前自动执行更新模式，使 overview 表格无需人工编辑；CI 或提交前检查执行 `--check`，防止生成结果未提交。直接编辑 marker 区块中的内容不属于支持的工作流。

## 5. 主题首页

每个主题目录都有自己的 `index.md`，包含：

- 该主题环境适合解决的问题；
- 推荐从哪个环境开始；
- 主题内环境对比表；
- 该主题全部页面的隐藏或可见 `toctree`。

主题页应帮助用户选择环境，而不是重复每个环境页面的完整内容。例如 Humanoid Locomotion 可以比较 G1、Dex-EVT 和 K1 的自由度、地形和控制方式；Quadruped Locomotion 可以比较 Go1、Go2 和 ANYmal 的速度跟踪、导航与地形能力；Whole Body Tracking 可以比较 robot、motion 和 tracking 目标。reward 公式仍由具体环境页面负责。

## 6. 单个环境页面模板

所有环境页面使用相同的信息顺序：

```text
# <环境名称>

一句话说明任务目标。

## 环境信息
Env ID、主题、Robot/Model、仿真后端、控制方式、可用 Training Task

## 效果预览
视频或图片

## 任务定义
场景、目标和 episode 成功条件

## Observation
字段、形状和语义

## Action
字段、形状、范围和控制方式

## Reward
各 reward term 的语义和组合方式

## Reset 与 Termination
初始状态、随机化、成功/失败/超时条件

## 运行环境
view 命令，以及最小 train/play 命令

## 源码与配置
环境实现、EnvCfg、场景/资产和 Training Task 路径

## 相关文档
Robot、训练教程、专题工作流或相近环境
```

页面以环境语义为主体。训练命令用于验证环境可运行，但算法原理、超参数解释、checkpoint 管理和训练结果分析仍由训练教程负责。

同一页面对应多个 Env ID 时，在“环境信息”中逐项列出各变体及差异。例如 `dm_quadruped` 页面可以同时描述 walk、run、escape 和 fetch，但不能只写一个模糊的总称。

## 7. 导航与链接规则

### 7.1 Toctree

`user_guide/index.md` 使用带 `环境 / Environments` caption 的独立 `toctree`，直接收录环境概览和六个主题首页。`envs/index.md` 使用普通 Markdown 链接指向六个主题，不再建立第二层 `toctree`；各主题首页继续收录具体环境页面：

```text
user_guide/index
└── 环境 / Environments（toctree caption）
    ├── envs/index（环境概览 / Environment Overview）
    ├── envs/basic/index
    ├── envs/dm_control/index
    ├── envs/whole_body_tracking/index
    ├── envs/humanoid_locomotion/index
    ├── envs/quadruped_locomotion/index
    └── envs/manipulation/index
```

caption 只负责划分侧栏 part，不对应重复的“环境”页面节点，因此基础环境在侧栏中显示为“环境 → 基础环境”，不会成为 Tutorials 的子项，也不会出现“环境 → 环境 → 基础环境”。新增具体环境时只需更新对应主题，不必继续扩大用户指南根索引。

### 7.2 入口链接

现有入门文档中的“训练示例”链接改为“浏览环境”，目标为 `envs/index.md` 或具体环境页面：

- `getting_started/hello_motrixlab.md` 的 Cartpole 下一步入口；
- `getting_started/container_deployment.md` 的更多任务入口；
- 其他指向 `demo/*.md` 的内部链接。

Robot 页面继续作为独立参考。环境页面只链接其使用的 Robot，不把 Robot-only viewer 或 Robot registry 说明复制到环境正文中。

## 8. 渐进迁移策略

### 8.1 阶段 A：先整理目录与导航

新增 `envs/index.md` 和六个主题首页，把现有 24 个 `demo/*.md` 页面移动到目标 `envs/<topic>/` 目录，并把现有 `tutorial/whole_body_tracking/` 整体迁到 `envs/whole_body_tracking/`。同步中英文 `toctree`、文档链接和 `literalinclude` 相对路径，同时从 `user_guide/index.md` 移除平铺的“训练示例”列表和 Tutorials 下的 WBT 入口。

该阶段只整理目录，不实现 overview 生成器。`envs/index.md` 预留生成区块 marker，后续脚本直接填充。

### 8.2 阶段 B：独立实现 Overview 生成器

按照第 4 节实现 `generate_env_docs.py`，从 Env Registry 中由 Python docstring 提取的双语 description 和 Hydra Task 配置生成 Env ID、Description、Training Algorithms 三列表格，并接入文档构建与 `--check` 检查。

### 8.3 阶段 C：统一环境页面

按第 6 节模板逐页整理现有内容，优先处理：

1. Cartpole，作为模板样例；
2. G1 WBT，统一整理 Whole Body Tracking 主题中的环境语义和操作流程；
3. G1 平地行走，验证 Humanoid Locomotion 页面；
4. Go1 复杂地形，验证 Quadruped Locomotion 页面；
5. Franka Lift Cube，验证 Manipulation 页面；
6. DM Quadruped，验证一个页面包含多个 Env ID 的情况；
7. 其余页面按主题批量迁移。

本阶段不要求所有页面拥有完全相同篇幅，但环境信息、Env ID、Observation、Action、Reward、Reset/Termination 和运行入口必须可定位。

### 8.4 阶段 D：补充旧 URL 重定向

目录迁移会把页面 URL 从 `/user_guide/demo/<name>.html` 改为 `/user_guide/envs/<topic>/<name>.html`。对外发布该目录结构前，应为旧 URL 建立到新页面的显式重定向，并验证：

- 24 个旧页面 URL 都有唯一目标；
- 中文和英文构建使用相同的路径映射；
- 重定向目标参与内部链接检查；
- 不保留重复的旧正文页面。

“Demo”可以继续作为首页中的视觉展示文案，但不再作为文档类型或目录分类。

## 9. 双语维护

`zh_CN` 与 `en` 使用相同的目录、文件名和 `toctree` 层级。每个阶段都同时完成两种语言的结构变更，避免先移动中文、后补英文造成导航漂移。

双语内容允许翻译进度不同，但必须满足：

- 页面路径一一对应；
- Env ID、命令、配置字段和源码路径完全一致；
- 主题归属一致；
- 不把英文页面链接到中文页面，反之亦然。

## 10. 可维护性

### 10.1 事实来源

- Env ID 和 EnvCfg：`motrix_env_core.registry` 及 `motrix_envs` 注册代码；
- Env description：注册 class 或 factory 的 Python docstring，由 Env Registry 自动提取；
- Training Algorithms：`configs/task/<env-id>/` 中实际存在的 Hydra Task 选项；
- Robot：`motrix_env_core.registry` 的 Robot Registry 与 `motrix_robots` 公共配置 API；
- 页面视频和封面：`docs/source/_static/`。

`envs/index.md` 的 overview 表格由生成器维护，不人工添加、删除或调整行。Env ID 和双语 description 直接来自 registry；description 的事实来源是注册 provider 的 Python docstring；Training Algorithms 直接来自 Hydra Task 发现结果。环境详情页中的长篇任务描述、任务语义和选型建议继续人工维护。

### 10.2 新增环境规则

新增内置环境时，用户文档至少需要：

1. 在一个主题下增加环境页面，或补充到已有的多变体页面；
2. 更新该主题的 `index.md`；
3. 在注册 class 或 factory 的 docstring 中补充英文摘要和 `zh_CN:` 描述，再重新生成 overview 表格；
4. 如果该 Env 支持训练，在 `configs/task/<env-id>/` 增加真实 Task 配方；overview 不单独维护算法字段；
5. 同步中文和英文结构；
6. 验证 Env ID、view 命令和声明存在的 Training Task；
7. 完成 Sphinx 构建和内部链接检查。

## 11. 验收标准

`Envs` part 完成时应满足：

1. 用户指南侧栏在 Tutorials 之后显示独立的“环境 / Environments”入口；
2. Whole Body Tracking、Humanoid Locomotion、Quadruped Locomotion 分别作为独立主题出现；
3. 24 个现有环境页面全部且仅归入一个主主题；
4. 用户从 `envs/index.md` 最多两次点击可到达任一环境页面；
5. 根 `user_guide/index.md` 不再平铺 24 个页面；
6. 首页 overview 表格由生成器产生，包含 Env ID、Description 和 Training Algorithms 三列，并覆盖 registry 中全部公共内置 Env ID；
7. 每个 Env ID 的注册 provider docstring 都包含英文和中文 description；
8. 环境、Robot 和 Training Task 的概念边界在首页明确说明；
9. WBT 环境说明和操作流程统一位于 `envs/whole_body_tracking/`，`tutorial/` 下没有重复入口；
10. 中文和英文导航结构一致；
11. Sphinx 中英文构建不新增 warning，内部链接和媒体路径有效；
12. 对外发布新目录前，为所有旧 `/user_guide/demo/*.html` URL 提供重定向；
13. `generate_env_docs.py --check` 能发现缺少 description、未知 Task Env 和过期的算法列表或生成表格。
