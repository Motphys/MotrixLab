# MotrixSim GPU 能力调查

## 基线与调查入口

从 `e98888a5` 开始，先确定现有 CPU 契约，再用小场景验证 GPU SDK 能否表达相同操作。下表列出基线中可直接查阅的入口，以及需要从实验取得的证据。

| 入口 | 已有依据 | 需要回答的问题 |
| --- | --- | --- |
| [环境构造](../../motrix_env_core/src/motrix_env_core/registry.py) | `registry.make` 接收 `sim`、环境数和运行模式，签名中没有独立的仿真设备参数。 | Direct 和 Manager 分别在哪里分配动作、索引、随机数和持久状态？用户的设备选择如何到达这些分配点？ |
| [后端契约](../../motrix_env_core/src/motrix_env_core/sim/backend.py) | 查询按声明编译；`step` 可将逐行状态写入调用方提供的张量。 | SDK 能提供哪些状态或错误信息？写入后何时可读，批次内单行异常如何报告？ |
| [CPU 查询](../../motrix_env_motrixsim/src/motrix_env_motrixsim/sim_data.py) 与 [CPU 写入](../../motrix_env_motrixsim/src/motrix_env_motrixsim/write_compiler.py) | 可从现有实现核对通道布局、读数与参数写入语义。 | GPU 侧的坐标系、单位、分量顺序，以及局部写入后的生效时刻是否一致？ |
| [插件依赖](../../motrix_env_motrixsim/pyproject.toml) | 插件依赖精确锁定的 `motrixsim-core===1.0.0`。 | 以 GPU SDK 1.0.0 为首个调查对象，核实可安装的平台、依赖组合和驱动要求；分别记录解析、安装与真实设备运行结果。 |
| [FastSAC 拓扑](../../motrix_rl/src/motrix_rl/topology.py) | 可从现有拓扑追踪 collector 与 learner 的归属。 | 分片在哪张卡推进？推理、样本交接与终态观测分别在哪个边界跨设备？ |

## 最小实验

1. **读写与流顺序**。在小场景中给两行写入可区分的关节状态和控制量，分开记录写后、单步后和多子步后的读数。用相同初态和输入对照 CPU，并检查 Torch 消费者在正常调用顺序下可见最新值。
2. **接触与局部 reset**。让批次中的两行都进入持续接触，只复位其中一行。将目标行与相同初态的新环境对照，将其余行与未复位的对照批次比较。分别观察复位当下及下一步的位姿、速度和接触响应，以识别残留的求解器状态。
3. **随机化与传感器**。用不同环境行验证摩擦等参数写入的作用范围。对 Raycast Camera 设置可区分的几何体，分别检查过滤、采样周期与复位后首次更新。记录具体缺失的能力和受影响的任务。
4. **训练数据传输**。分别采集 CPU 仿真配 GPU 推理、同卡 CUDA 仿真与推理、显式跨卡放置的运行轨迹。区分普通控制步与结束回合的控制步，标出动作、观测、transition 和终态观测的传输方向及同步点。拷贝次数与耗时从实测轨迹取得。

记录 SDK 与驱动版本、场景、输入、对照值、误差容限及其依据。单步可对拍的字段与长时间接触轨迹分别评价。将确认过的能力和限制带入[设计](../design/motrixsim-gpu.md)，再按[实施计划](../plan/motrixsim-gpu.md)扩展到任务和训练矩阵。

## 本机实测（2026-11-25）

CPU 基线为 `e98888a5`。本机使用 i5-14600KF、单个 NUMA 节点和 RTX 5060 Ti，Python 3.10、Torch 2.7.0+cu128、MotrixSim Core/GPU 1.0.0。CPU 环境仅安装 Core。两端分别用独立进程构造每个注册任务，seed 183，2 行环境，初始化后执行零动作、单独复位第 1 行、再次运行零动作。CPU 基线和候选各通过全部 56 项。

| RTX 5060 Ti 阶段性验证结果 | 数量 | 注册任务 |
| --- | ---: | --- |
| 通过 | 42 | `acrobot`、`anymal_c_navigation_flat`、`anymalc-walk-flat`、`anymalc-walk-rough`、`bounce_ball`、`cartpole`、`dex-evt-walk-flat`、`dex-evt-walk-rough`、`dm-cheetah`、`dm-humanoid-run`、`dm-humanoid-stand`、`dm-humanoid-walk`、`dm-lqr-2-1`、`dm-lqr-6-2`、`dm-quadruped-escape`、`dm-quadruped-fetch`、`dm-quadruped-run`、`dm-quadruped-walk`、`dm-reacher`、`dm-runner`、`dm-stander`、`dm-walker`、`franka-lift-cube`、`franka-open-cabinet`、`g1-walk-rough`、`go1-walk-flat`、`go1-walk-rough`、`go2-walk-flat`、`go2-walk-rough`、`k1-walk-flat`、`k1-walk-rough`、`microduck-walk-flat`、`microduck-walk-rough`、`peg-insert`、`pendulum`、`point_mass`、`rm65-open-cabinet`、`rm65_insert_peg`、`shadow-hand-repose`、`stewart`、`stewart-disturb-xy`、`stewart-static` |
| 通过（后端适配 link 净接触力） | 6 | `dex-evt-wbt-dance`、`g1-29dof-wbt-largebox`、`g1-wbt-backflip`、`g1-wbt-dance`、`k1-wbt-freekick`、`microduck-ball-balance` |
| 初测缺少 touch 读数（后续已加入适配） | 6 | `dm-finger-spin`、`dm-finger-turn-easy`、`dm-finger-turn-hard`、`dm-hopper-hop`、`dm-hopper-stand`、`dm-manipulator-bring-ball` |
| 只支持滑动摩擦覆盖 | 1 | `g1-walk-flat` |
| 初测缺少接触传感器请求的力报告 | 1 | `go1-stairs-terrain-walk` |

接触力适配器按物理步长将每个接触点的约束冲量换成力，输出世界坐标系下的合力和绕 link 质心的合力矩。1 kg 球体静止承重时，两端均测得 9.81 N。偏心球体在横向重力作用下，两端均测得横向力 −1.00 N、竖向力 9.81 N 和绕 y 轴力矩 −0.831 N·m。仅复位一行后，该行读数清零，保留行读数保持不变。原生 GPU 查询可用后，替换插件读取实现即可。

球体在重力下接触平面 100 个物理步后，只对中间一行设置位置和速度。以未复位的独立批次对照其余两行，以全新批次对照中间行。复位后立即采样，再依次推进 1、10、100 步。CPU 和 CUDA 各自的位置、速度、几何体接触状态都在绝对和相对误差 1e-5 内通过。GPU SDK 的高层碰撞查询在该场景一直返回零，按 collider pair 读取碰撞点后可检测到承重接触。

Cartpole 实测 SKRL PPO 和 RSLRL PPO 短训。Peg-insert 的同步 FastSAC checkpoint 中 critic、actor、温度优化器分别更新 6、2、6 次。异步 FastSAC 的两个 collector 在同一张 GPU 采样，learner 收到样本并更新。Cartpole checkpoint 分别使用记录的 CUDA 设备和显式 CPU 覆盖完成短视频回放。

单卡 Cartpole 的环境步吞吐在 64、256、1024、4096 行时分别约为 9.74 万、23.71 万、98.10 万、396.27 万步/秒。主机绑定 CPU 0、2、4、6，预热 50 个控制步，计时 500 步，窗口两端同步 GPU。该窗口不含 learner，单次计时，且在本机回归测试期间执行。CPU 基线在 1024 行时，绑定 1、2、4 个物理核分别为 31.22 万、58.37 万、101.73 万步/秒。

相同 CPU 绑定和环境数下，候选的 1024 行吞吐分别为 27.92 万、44.67 万、92.07 万步/秒。单次结果低于基线约 11%、23%、10%，尚需在隔离负载下重复计时。Raycast Camera 的 4 × 3 小场景在 CUDA 创建并推进后，GPU SDK 以名称读取原始通道时报错 `sensor 'camera' not found`。本版本将相机查询标为受限。

### 查询发布与流交接

CPU 原生批量查询把各字段发布到共享的 host arena。CPU Torch 直接读取 arena 的视图。Cartpole 接 CUDA policy 输入的单独测量中，8 次整批 policy observation 上传产生 8 次 H2D，字段数不参与这次传输。

CUDA 仿真在同一条非默认流上执行一个控制步。Warp 与 Torch 在控制步边界建立调用流依赖，内部查询和物理步复用该流。固定 DOF 与控制量的查询绑定设备视图，编译快照发布后捕获 CUDA Graph。含 SDK getter 或接触 kernel 的查询先获取当前设备数据，再执行编译后的投影。局部刷新只更新所选行。物理步使用 SDK 自带的 Graph。[MotrixSim 性能优化](https://motrixsim.readthedocs.io/zh-cn/latest/user_guide/gpu_version/performance_tuning.html)

RTX 5060 Ti、1024 行 Cartpole、CPU 绑定 0/2/4/6、预热 50 个控制步后计时 500 步。各阶段单次测量，计时窗口两端同步，吞吐按环境行计算。

| 实现 | 万环境步/秒 | 毫秒/控制步 |
| --- | ---: | ---: |
| 修改前工作树 | 86.7 | 1.18 |
| 仅合并流交接 | 98.0 | 1.05 |
| 流交接与编译查询 | 90.0 | 1.14 |
| 固定地址查询图重放 | 99.0 | 1.03 |

CUPTI 13.0 驱动 API 计数显示，8 个普通 CUDA 控制步共提交 24 次图启动和 16 次跨流事件等待，记录到 16 次 D2H、32 次 D2D 和 0 次 H2D。回合调度仍在主机判断截断和结束状态。使用已有编译缓存时，独立实例从构造到完成前 10 个控制步为 3.95 秒。该时间包含 SDK 初始化、编译查询、图捕获与步进。

分段采样中，Cartpole 单次物理调用提交 2 次图启动，单次全批查询提交 1 次。Microduck 的 Manager 任务完成 500 个控制步与单行强制截断，保存的终态观测在下一步保持稳定。Inductor 对含 CPU 输入或动态命令更新的子图报告跳过捕获，因此 Manager 的图重放覆盖范围随 term 调度而变。

### Manager 的 Warp 交接与 term 图重放

以同机 `microduck-walk-flat` 对照上一轮查询图实现。环境数 64、seed 183、零动作、CPU 绑定 0/2/4/6。独立进程预热 15 个控制步，测量 250 步，窗口两端同步；另在预热后的 8 步上用 CUPTI 13 计数。各阶段各运行一次。

| 指标 | 上一轮实现 | 本轮实现 |
| --- | ---: | ---: |
| 环境步/秒 | 22,479 | 24,472 |
| 流同步调用 | 56 | 40 |
| D2H / H2D | 48 / 8 | 32 / 8 |
| 创建 / 销毁 CUDA 事件 | 8 / 8 | 0 / 0 |
| 跨流事件等待 | 16 | 16 |
| 图启动 / 普通 kernel 启动 | 156 / 1,116 | 180 / 936 |

Warp 调用流交接复用预分配事件。Manager 通过 term 声明中的持久张量标记静态地址，允许 Inductor 重放原地更新命令状态的子图。普通 CUDA 步先合并检查回合结束与 sim-only reset 请求，有请求时再选择行并保存终态观测。位置与四元数查询从同一份 SDK 世界坐标 link pose 表投影，按声明保留 link 顺序和重复项。

与 SDK 逐 link 读取对照，验证了全批与非连续行查询、保留行快照和四元数排列。Microduck 通过单行强制截断、终态观测生命周期及 sim-only reset 的回合计数检查。Inductor 仍报告部分含 CPU 标量输入的子图跳过捕获。
