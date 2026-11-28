# APPO 训练效果与性能评测

## 摘要

本文汇总 Motrix APPO 在三项四足任务上的训练对照，以及 learner、transition、rollout carrier、权重与 normalizer 的 GPU 优化实验。

- 单训练 seed 下，Go2 flat/rough 的 APPO 回报高于 RSLRL PPO；ANYmal-C 在训练支持的前进命令下接近，但分布外命令稳定性明显退化。
- Learner 同步消除、编译和 CUDA Graph 将独立 update 从约67ms降至34ms；不能将其近2倍吞吐直接等同于整训练提速。
- IPC 能降低接收/发布局部开销，GPU常驻 chunk 没有显示稳定端到端收益；外部负载和CPU collector仍影响整体耗时。

算法、buffer所有权与当前执行契约见 [APPO设计](../design/appo.md)，随机流和timeout capture约束见[随机性设计](../design/randomness-and-seeding.md)与[终态观测契约](../design/array-env-final-observation.md)。以下是实验结果，不是跨训练seed的统计显著性证明。

## 1. 当前接口与证据范围

公开入口为 `task=<env>/motrix.appo`，由 builtin 默认注册，无独立 APPO extra。三份四足配方默认 CUDA Graph + fused Adam；通用provider和CartPole保留eager以支持CPU。

```sh
mxlab train task=go2-walk-flat/motrix.appo
# 显式CPU路径示例
mxlab train task=cartpole/motrix.appo algo.device=cpu algo.collector_device=cpu
# 分别控制transition和完整参数/统计bundle
mxlab train task=go2-walk-flat/motrix.appo algo.transition_transport=host algo.weight_transport=host
```

当前只暴露 `transition_transport`、`weight_transport`，均为auto/host/cuda_ipc。chunk存储随transition路径选择；weight控制参数传输，normalizer统计保留host共享快照。实验时的 `appo.ppo`、独立 `chunk_device` / `normalizer_transport`、APPO安装extra，以及 `runs/.../appo/torch/ppo/` 路径属于历史证据，不是当前兼容接口。原始脚本与配置快照保留执行时值，不能机械复制历史A/B命令到当前版本。

完整2000-update效果验证发生在host transport阶段；后续IPC实验为各500-update匹配预算，不能拼成“当前全部GPU路径已完成2000-update验证”的结论。各实验比较的是各自同代码/同预算配对，不将不同阶段的最好时间混合比较。

## 2. 公共实验协议

硬件/运行时为RTX5090 32GB、Ryzen9950X、Torch2.9.1+cu128、RSL-RL4.0.1、MotrixSim core1.0.0。训练串行执行，但机器其他负载未隔离。

| 任务 | 环境数×rollout | updates | transitions/算法 |
|---|---|---:|---:|
| Go2 flat | 1024×24 | 2000 | 49,152,000 |
| ANYmal-C flat | 1024×24 | 500 | 12,288,000 |
| Go2 rough | 1024×24 | 1500 | 36,864,000 |

每任务训练seed1，相同环境/reward/DR/课程、512/256/128 ELU actor/critic、std0.5、privileged value、Adam初始lr3e-4、5epochs/4minibatches、entropy0.001、desiredKL0.01。V-trace/GAE、std参数化、初始化和异步随机流等实现差异不强行统一，不将结果单独归因于某个数学项。

评估固定final checkpoint，不按held-out分数选模型；环境seeds101/102/103，确定性mean policy，clean play显式关闭DR/noise。每指令16lanes、1000control steps（20s），动作无额外clamp，reward无bootstrap修正，tracking使用auto-reset前物理速度，包含terminal端点。各command等权。

- Go2 flat：21格训练包络内指令；主回报为1000步horizon累计。
- ANYmal-C：训练只含forward0.5m/s，主评估仅此格，其他8格单列OOD。默认play仍有噪声，clean评估显式关闭。
- Go2 rough：9格合法指令，vx∈[−0.5,1]、vy∈[−0.4,0.4]、yaw∈[−1,1]；保留procedural heightfield、physics/reward/spawn规则，无height scan或terrain curriculum。地图generator固定seed0，环境seed不会改变地图，不是未见地形测试。
- ANYmal/rough主指标截至首次episode结束，首次done后停止计分；continuing1000步另列。tracking须和有效曝光、survival一起解读，不能把早期失败导致的小RMSE当优势。

**计时口径：**RSLRL是collection+learning累计，APPO是trainer wall轴（含startup/循环，至最后更新，最终save/退出不含）。训练日志聚合窗口也不同；曲线只作过程诊断，有限评估才是效果依据。所有实验每算法/模式仅一个训练seed，三环境seed不能替代独立训练；clean play极小方差不代表训练稳定性。

## 3. 任务效果

### 3.1 Go2 flat完整预算

| 指标 | RSLRL PPO | 原版 APPO eager | APPO Graph + fused |
|---|---:|---:|---:|
| 记录训练时间s | 515.571 | 378.995 | 326.815 |
| 固定指令回报 | 108.8406 | 130.2282 | 130.7391 |
| 无termination存活率 | 100% | 100% | 100% |
| vx RMSE m/s | 0.108312 | 0.083251 | 0.091970 |
| vy RMSE m/s | 0.152730 | 0.114999 | 0.119541 |
| yaw RMSE rad/s | 0.175074 | 0.070756 | 0.068629 |

原APPO较RSLRL回报高19.65%，记录吞吐约高36%。Graph较原APPO耗时少13.77%、吞吐高15.97%，回报高0.39%，但vx/vy误差增加10.47%/3.95%、yaw降低3.01%，不称全面无损。原APPO的左横移vy、部分mixed单轴跟踪也劣于RSLRL（详见逐command原始报告），总体回报高不代表每个分项都更好。

![Go2 flat：RSLRL、原版APPO与Graph APPO](./assets/appo-go2-cudagraph-training.png)

[原始两算法训练曲线](./assets/appo-go2-training.png)保留作基础实验参考，主图包含第三列优化模型。

基准均使用相同pre-reset final-observation实现。没有capture的早期RSLRL run约439s、评估105.5681，仅作诊断；capture消费共享噪声流，因此不能拿旧run与新APPO混比。

### 3.2 ANYmal-C与Go2 rough

| 任务/指标 | RSLRL PPO | APPO Graph |
|---|---:|---:|
| ANYmal记录训练时间s | 87.322 | 67.693 |
| ANYmal首episode回报 | 57.2186 | 57.2561 |
| ANYmal survival | 100% | 100% |
| ANYmal vx/vy/yaw RMSE | 0.07188 / 0.02928 / 0.01729 | 0.05140 / 0.01431 / 0.02055 |
| rough记录训练时间s | 353.061 | 275.643 |
| rough首episode回报 | 31.5178 | 39.9555 |
| rough survival | 92.824% | 99.306% |
| rough首episode有效步 | 959.34 | 997.11 |
| rough vx/vy/yaw RMSE | 0.17653 / 0.17673 / 0.10280 | 0.11319 / 0.12029 / 0.09559 |
| rough continuing回报 | 32.7019 | 40.0734 |

ANYmal分布内基本持平，APPO前向/横向误差更低、yaw略高。rough首episode回报高26.77%、存活率高6.48个百分点，continuing回报高22.54%；same-terrain clean测试不等于地形泛化。三任务主评估engine errors均为0，不横比不同任务reward绝对值。

![ANYmal-C flat训练对照](./assets/appo-anymalc-training.png)

![Go2 rough训练对照](./assets/appo-go2-rough-training.png)

### 3.3 ANYmal OOD诊断

8格未训练指令不进入主分数。RSLRL/APPO首回合survival为50%/0%，有效步574/34.25，回报7.2016/−1.1841；APPO逐格14–56步失败，属于明显外推退化。supported/OOD使用相同checkpoint/hash且维度严格匹配，没有恢复错误证据。

两模型command observation索引42:45的训练mean=[0.5,0,0]、std=0，eps0.01时站立vx标准化为−50。这是可能的外推风险背景，不是因果证明；没有为救分修改奖励、normalizer或重训。当前不把OOD列为任务验收目标，但保留该负面证据。

## 4. Learner执行优化

独立benchmark使用Go2规格T24×N1024、obs49/value52/action12、FP32默认精度，无TF32/AMP。原版参考为优化前`c6283ba6`独立源码；合成固定off-policy batch，完整update前后CUDA synchronize，warmup10后测50次，不是学习效果试验。

| 模式 | 稳态update均值ms | 首次update ms | peak allocated MiB |
|---|---:|---:|---:|
| 原版eager | 67.21 | 223.1 | 见原始JSON |
| 优化eager | 41.11 | 195.0 | 140.0 |
| eager + fused Adam | 41.88 | 170.8 | 140.0 |
| compile loss + fused Adam | 34.11 | 4632.0 | 218.0 |
| CUDA Graph + fused Adam | 33.98 | 197.5 | 153.2 |

原版profile记录312次cudaStreamSynchronize和4394次cudaLaunchKernel，CPU同步约17.8ms、launch约11.5ms。device LR/metrics、解析Gaussian loss和整批指标回传是主要收益；fused Adam单独无稳定提升。compile与graph稳态接近；编译缓存部分预热，首次不是全冷成本。固定LR三次CPUupdate参数maxdiff2.98e-8不代表adaptive逐位一致；KL严格0附近的舍入能改变LR分支，不为掩盖差异增加epsilon。

200update真实短跑：原版/优化eager/fused/compile/graph wall分别34.32/35.25/35.12/41.74/38.35s，未出现端到端加速，collector164–170ms/chunk成为瓶颈。长训练的收益见Go2 flat表，不能由独立learner近2倍推断整训练2倍。

反序复测eager/fused/compile/graph为37.46/38.30/30.37/32.85ms，存在负载漂移且一段可能与旧短跑尾部重叠，不作为严格无争用排名。process/V-trace单独编译1.663→1.135ms，首编译4.875s，仅省0.53ms，未增加对应产品分支。未编写自定义Triton/CUDA kernel，融合来自Torch fused Adam/Inductor。

## 5. 传输优化的分阶段对照

以下各阶段固定Go2 flat/CUDA Graph fused/seed1，每次500updates=12,288,000transitions，日志10、checkpoint500，配对顺序反转。一次只变待测路径；不同表不能跨阶段相减归因。表中wall为完整训练记录，collect/ingest/publish为后200updates日志均值；ingest是host copy/enqueue耗时，未额外同步，不是纯device DMA计时。

### 5.1 Transition ring：CPU-staged IPC

此阶段仍是CPU chunk→pinned staging→fused H2D，weight仍host，不是零拷贝。

| 轮次/顺序 | ring | wall s | ingest ms | update+publish ms | collect ms |
|---|---|---:|---:|---:|---:|
| 1先 | host | 99.737 | 1.701 | 39.999 | 222.705 |
| 1后 | IPC | 89.062 | 0.327 | 39.575 | 174.633 |
| 2先 | IPC | 90.671 | 0.339 | 39.737 | 185.820 |
| 2后 | host | 87.495 | 1.681 | 39.637 | 172.470 |

Ingress约1.69→0.33ms（−80%）稳定改善；完整wall一轮快10.7%、另一轮慢3.6%，**没有稳定端到端提速证据**。

### 5.2 GPU常驻chunk与D2D发布

CPU/GPU carrier均预分配，避免把共同去clone/stack收益归因于GPU；weight仍host。

| 轮次 | 路径 | wall s | collect ms | ingest ms | update+publish ms |
|---|---|---:|---:|---:|---:|
| 1 | host / CPU | 95.488 | 198.057 | 1.673 | 40.022 |
| 1 | IPC / CPU | 88.810 | 175.170 | 0.344 | 39.923 |
| 1 | IPC / GPU | 87.622 | 169.287 | 0.328 | 39.736 |
| 2 | IPC / GPU | 93.449 | 180.559 | 0.323 | 40.007 |
| 2 | IPC / CPU | 87.437 | 168.895 | 0.324 | 39.205 |
| 2 | host / CPU | 86.223 | 169.005 | 1.511 | 39.333 |

均值host/CPU IPC/GPU IPC为90.855/88.123/90.536s；GPU carrier较CPU IPC先快1.34%、后慢6.88%，均值慢2.74%，**未证实端到端收益**。仍有CPU环境同步、逐步H2D、strided写入、D2D pack与publication轮询，未逐kernel证明单因素原因。

预先选round1 final的9次eval：host/CPU IPC/GPU IPC回报91.1464/100.4396/100.4396，survival全100%；vx RMSE0.13903/0.09479/0.09479、vy0.20220/0.21569/0.21569、yaw0.28631/0.28340/0.28340。两IPC模型actor/critic/target逐元素相同，三seed评估也相同；只证明该轮没有观察到学习差异，不是异步普遍确定性保证。

### 5.3 参数IPC（统计仍host）

固定GPU transition与GPU chunk，仅改变参数通道。

| 轮次/顺序 | 参数路径 | wall s | publish ms | optimize ms | collect ms |
|---|---|---:|---:|---:|---:|
| 1先 | host | 96.536 | 0.914 | 39.027 | 221.802 |
| 1后 | IPC | 88.533 | 0.685 | 38.696 | 176.589 |
| 2先 | IPC | 87.685 | 0.560 | 38.947 | 172.054 |
| 2后 | host | 88.429 | 0.679 | 34.180 | 172.539 |

发布均值0.796→0.622ms，省约0.17ms（21.8%）；wall均值少4.73%，但首轮collector漂移明显，不能全部归因参数通道。预先round1两final逐元素相同，6次eval相同：回报100.4396、survival100%、RMSE约0.09479/0.21569/0.28340。

### 5.4 参数与normalizer完整GPU bundle（实验，未保留实现）

该方案因绝对收益较小而暂不纳入当前实现，后续场景和验收条件由[issue #197](https://github.com/Motphys/MotrixLab/issues/197)追踪；以下保留实验证据，不表示当前具有GPU统计通道。

固定参数IPC与GPU transition/chunk，仅改变normalizer位置。

| 轮次/顺序 | 统计路径 | wall s | bundle publish ms | optimize ms | collect ms |
|---|---|---:|---:|---:|---:|
| 1先 | host | 95.814 | 0.576 | 38.945 | 193.613 |
| 1后 | GPU | 86.061 | 0.422 | 38.799 | 171.786 |
| 2先 | GPU | 87.642 | 0.502 | 38.659 | 170.157 |
| 2后 | host | 85.494 | 0.569 | 38.643 | 168.571 |

Bundle发布均值0.5725→0.4619ms，省约0.11ms（19.3%）；wall一轮快10.18%、另一轮慢2.51%，**没有稳定端到端收益证据**。两轮各自host/GPU final的actor/critic/target含normalizer buffers均maxdiff0；未额外eval，也未重跑2000updates。count独立int64槽、真实测试超过2^40精确；私有snapshot完成后检查seq，torn/odd重试不触live模型。seq控制面仍host，不称全系统无CPU交互。

## 6. 正确性与验证范围

长期测试保护数据对齐、timeout/terminal/endpoint、pooled moments、target更新、checkpoint/RNG、CPU/GPU路径和跨进程所有权，不把一次性benchmark保留成固定超参测试。

- 真实CUDA验证包含：compile/graph多步parity、counter/RNG及checkpoint重捕获；IPC非默认stream、背压、wrap、bool/shape、写完成后carrier复用；完整参数/统计大count、Identity/空sections与强制重试模型不变。
- 不同阶段分别有CPU/真实CartPole与Go2集成、快速回归、Ruff/pre-commit及wheel构建通过；测试数不相加为一个总数，skip不作为GPU通过证据。
- 初始阶段裸全仓pytest因本地Unitree插件缺失及历史runs脚本旧import收集失败；已明确报告，没有靠缩小测试范围声称全仓通过。后续PR测试矩阵的验证状态以对应commit的CI为准。
- 没有实际执行PyPI发布；依赖解析dry-run、wheel构建和真实安装/训练分别记录，不混为兼容性证明。

## 7. 证据索引与复现

当前代码用法见[APPO包说明](../../motrix_rl_appo/README.md)。完整设计不在research重复维护；旧namespace、已删除A/B字段仅存在于历史配置与原始命令记录，模型和评测数据不改写。

| 阶段 | 原始脚本/JSON/报告目录 |
|---|---|
| Go2基础对比 | `runs/appo-go2-comparison/` |
| learner与Graph完整复验 | `runs/appo-learner-performance/` |
| ANYmal / rough（含逐command与OOD） | `runs/appo-quadruped-comparison/` |
| transition IPC | `runs/appo-ipc-performance/` |
| device chunk | `runs/appo-device-chunk-performance/` |
| 参数IPC | `runs/appo-weight-ipc-performance/` |
| GPU normalizer bundle | `runs/appo-gpu-normalizer-performance/` |

Final checkpoint相对run目录均为`checkpoints/latest.pt`：

| 任务 / 模式 | run目录 |
|---|---|
| Go2 flat RSLRL | `runs/go2-walk-flat/rslrl/torch/ppo/26-11-27_23-56-33-216442/` |
| Go2 flat 原APPO | `runs/go2-walk-flat/appo/torch/ppo/26-11-28_00-05-38-475556/` |
| Go2 flat Graph APPO | `runs/go2-walk-flat/appo/torch/ppo/26-11-28_01-03-28-934502/` |
| ANYmal RSLRL | `runs/anymalc-walk-flat/rslrl/torch/ppo/26-11-28_10-01-10-864844/` |
| ANYmal APPO | `runs/anymalc-walk-flat/appo/torch/ppo/26-11-28_10-03-16-390297/` |
| rough RSLRL | `runs/go2-walk-rough/rslrl/torch/ppo/26-11-28_10-05-02-563160/` |
| rough APPO | `runs/go2-walk-rough/appo/torch/ppo/26-11-28_10-11-30-215167/` |

运行产物由gitignore忽略，未随源码发布；目录中保留任务快照、checkpoint/source SHA256、完整命令和评估器。图片集中在[assets](./assets/index.md)。后续复验应使用当前namespace重新生成配置，不将历史路径重命名为新实验。

## 8. 结论与后续验证

当前证据支持APPO在已测任务/预算下运行正确，以及learner和局部传输开销降低；不支持“所有指标无损”“全GPU后必然更快”或跨任务普遍优势。下一步有效工作是隔离负载、多训练seed、同协议交替计时、未见地形与DR分布robustness，而非累加不同阶段单轮最好时间。
