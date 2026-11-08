# NumbaWbtTask.update_state 远端 96 核扩展性研究(进行中)

## 摘要

`NumbaWbtTask` 在 EPYC 7K62(96 物理核 / 192 逻辑核)上的代码融合和运行时调查
已完成4096 envs回测。最终counter-based并行RNG消除了主kernel中的串行噪声相位;
随后确认生产调用的主要剩余开销来自48线程GNU OpenMP(libgomp)worker在
MotrixSim/Rayon和host串行阶段持续自旋。

固定远端node0的48个物理核、`RAYON_NUM_THREADS=8` / `NUMBA_NUM_THREADS=48`时,
`GOMP_SPINCOUNT=0`将完整`update_state`从15.689ms降到5.454ms,运行时收益
**2.88x**,相对同轮NumPy为**7.34x**。完整`step`也从243.904ms降到
229.048ms,配置收益**1.065x**;相对同进程NumPy的中位加速由1.21x提高到1.23x。
本地8核对应收益为`update_state` **4.12x**、完整`step` **1.030x**。

当前采用的目标服务器启动配置是OpenMP、`GOMP_SPINCOUNT=0`、Numba/Rayon
48/48;这些参数属于进程级runtime,不写入`EnvCfg`。调用侧已把
`_state_quantities()`落地为3-getter实现:复用`get_link_states()`中的root/reference
状态,只保留link states、joint position、joint velocity三次MotrixSim调用。
base world velocity的两次inverse rotation也已进入现有main `prange`,没有新增
parallel launch。正式源码在48/48下的完整`update_state`五进程中位数为2.161ms
(2.102-2.193ms),相对最初48/8基线提升2.52x。当前热点主要是三个MotrixSim getter。

## 背景与目标

分支 `146-numba-env` 引入 `NumbaWbtTask`(PoC):把 WBT 的 post-physics 计算
(motion targets / observation / termination / reward)融合为一个 Numba kernel。
最初现象是同一 `update_state` 在远端 96 物理核 EPYC 7K62 上比本地 8 核 Ryzen 9700X
**更慢**(4096 envs:远端 ~60ms vs 本地 ~4.5ms)。物理部分已在此前
`motphys-articulated-body` 的 dispatch 优化中处理。

目标:

1. 优化 `update_state` 中与 motrixsim 无关的部分,尽量融合为一个 kernel
   (当前被中间的 `_read_kernel_inputs` 拆开)。
2. 评测 numba 在远端 96 核上的性能上限,并给出可复现的对照数据。
3. 测试报告写入 `wiki/research/`。

## 硬件与软件

| 项目 | 本地 | 远端 |
| --- | --- | --- |
| CPU | Ryzen 7 9700X(8C/16T) | EPYC 7K62(2×48C,96 物理 / 192 逻辑) |
| NUMA | - | node0 `0-47,96-143`;node1 `48-95,144-191` |
| numba / numpy | 0.61.2 / 2.2.6 | 0.61.2 / 2.2.6 |
| motrixsim | 0.9.1.dev115457(与远端同一 .so,BuildID 一致) | 同左 |
| 调度器 | - | schedutil(空闲 ~1.5GHz) |

## 优化前基准(commit `c7f37cc`)

该组数据由 commit `c7f37cc` 中的 legacy WBT task-compute 专项脚本产生。该专项脚本和当时用于对照的旧环境配置已删除；当前仓库没有可复现这组历史数据的端到端 A/B 命令，因此下表仅保留为历史记录，不应作为当前基准入口。

### update_state(numba),本地 vs 远端

| num_envs | 本地 ms | 远端 ms | 远端/本地 |
| --- | --- | --- | --- |
| 1 | 0.178 | 6.00 | 33.7x |
| 16 | 0.342 | 14.97 | 43.8x |
| 256 | 0.594 | 21.02 | 35.4x |
| 1024 | 1.324 | 27.96 | 21.1x |
| 4096 | 4.466 | 60.05 | 13.4x |

### 远端 4096 envs 相位分解(优化前)

| 相位 | ms | 占比 |
| --- | --- | --- |
| pre-kernel(motion advance + 噪声,numpy) | 9.59 | 16.4% |
| `_read_kernel_inputs`(MotrixSim 读取) | 43.72 | 74.8% |
| task-kernel(numba) | 4.64 | 7.9% |
| post-kernel(metrics,numpy) | 0.53 | 0.9% |

同参数本地分解:read-inputs 3.02ms、pre-kernel 1.01ms、task-kernel 0.36ms、
post-kernel 0.08ms。

## 根因分析

### 1. MotrixSim getter 每次调用都有并行派发固定开销(远端最大头)

`get_rotation` / `get_link_states` / `get_link_net_contact_forces` 等通过
`data.build_ndarray` → `parallel_for` 派发。远端默认按 192 逻辑核初始化,小读取
的派发开销被放大:

| RAYON_NUM_THREADS | root get_rotation(4096,4) | get_link_states(4096,14,13) |
| --- | --- | --- |
| 1 | 0.144 ms | 6.36 ms |
| 8 | **0.042 ms** | 1.40 ms |
| 48 | 0.109 ms | **0.96 ms** |
| 192(默认) | 0.633 ms | 1.74 ms |

本地默认 16 线程时同类 getter 仅 0.012-0.215 ms。9 个 getter 顺序调用累计
~9.5ms(远端);本地仅 ~0.33ms。这是 motrixsim 侧问题(理想修复在
`motphys-articulated-body`:让状态读取走持久 BatchPool,或按读取规模自适应线程数)。

### 2. numba kernel 线程超卖 + NUMA

远端默认 `NUMBA_NUM_THREADS=192`,task-kernel 4.64ms;`NUMBA_NUM_THREADS=48`
时降到 1.14ms(96t 1.76ms,192t 4.69ms)。48t 最优与单 NUMA 节点内存局部性一致。

### 3. motrixsim 无关的 numpy 后处理可全部融合

- 观测噪声:numpy 4 次 `uniform → astype → 乘幅值`,4096 envs 约 8-9ms。
- `_r_undesired_contacts` 的 `linalg.norm + 阈值 + sum` 约 3.3-4.8ms。
- metrics:12 次 `np.mean` 约 0.5ms(远端放大后更多)。
- `robot_body_*` 非连续视图的 4 次 `ascontiguousarray` 拷贝。

## 已实现优化(commit `39356f6`)

把 motrixsim 无关部分融合进 numba:

1. **pre-kernel**(`WBT_PRE_KERNEL`):并行推进 `motion_steps` + 检测 clip end
   (替代 numpy `_advance_motion`);clip end 的 MotrixSim reset 仍留在 host(罕见路径)。
2. **主 kernel 头部串行噪声相位**:按 numpy 相同的 C-order 绘制顺序生成 4 个噪声数组
   (numba MT19937 与 numpy legacy 流逐位一致);每调用以递增 seed 重播种,
   确定且步间独立,避免共享 RNG 状态。
3. **contact 归约移入 kernel**:直接读取原始
   `_undesired_contact_forces(num_envs, n_links, 6)`,kernel 内算
   `norm(xyz) > threshold` 计数,替代 numpy `_r_undesired_contacts`。
4. **metrics partial 移入 kernel**:kernel 写 `(num_envs, 12)` partial 缓冲,
   host 只做一次 `mean(axis=0)`(替代 12 次 `np.mean`)。
5. **放宽只读输入 contiguity 校验**(`motrix_env_core/numba/kernel.py`):
   buffers/outputs 仍要求 C-contiguous + writable;inputs 允许 strided 视图,
   消除 `robot_body_*` 的 4 次 `ascontiguousarray` 拷贝。

`update_state` 新结构:

```text
pre-kernel(advance+clip) -> [host: clip-end reset, 罕见] -> _read_kernel_inputs(MotrixSim)
-> task-kernel(noise + motion/obs/termination/reward + metrics partial)
-> [host: 1 次 mean + sampler update]
```

### 本地验证(优化后,steps=50)

| num_envs | numpy ms | numba ms | 优化前 numba ms |
| --- | --- | --- | --- |
| 16 | 1.087 | 0.279 | 0.343 |
| 256 | 1.980 | 0.344 | 0.594 |
| 2048 | 9.773 | 0.922 | 2.34 |

测试:`motrix_envs/tests/test_wbt_numba.py` 7 passed;
`motrix_env_core/tests/test_numba_env.py` 3 passed。
噪声语义:新增 bounds/determinism 测试;跨后端 parity 测试改用零噪声 cfg,
因为 numba 现在使用自己的 RNG 流(与 numpy 逐位一致,但状态不同步)。

## 远端阶段性结果

### 基准方法

以下 `update_state` 数据来自远端 EPYC 96 核机器（见文首机器对比表），每项预热后运行 100 次并取中位数。它使用对应 commit 中现已删除的
legacy task-compute 专项入口；当前仓库没有可用的端到端 A/B 命令复现这组历史数据，因此下表仅作为阶段性历史记录。
旧/新代码均使用同一 Python 3.10.20、Numba 0.61.2、NumPy 2.2.6 和 MotrixSim
0.9.1.dev115457,并固定 `RAYON_NUM_THREADS=8` / `NUMBA_NUM_THREADS=48`。

目前每个配置只完成一个进程内的中位数采样,足以作为阶段性代码对照,
但还不是包含多进程重复次数与置信区间的最终性能报告。

### 纯代码收益(同线程配置)

| num_envs | NumPy ms | 优化前 Numba ms | 优化后 Numba ms | 纯代码收益 | 优化后 Numba / NumPy |
| --- | --- | --- | --- | --- | --- |
| 16 | 2.6297 | 1.3023 | 0.5842 | 2.23x | 4.50x |
| 256 | 4.8423 | 1.9315 | 1.1208 | 1.72x | 4.32x |
| 1024 | 11.7286 | 5.7670 | 2.6544 | 2.17x | 4.42x |
| 4096 | 48.5499 | 21.1288 | 8.8313 | 2.39x | 5.50x |

NumPy 列取优化后同轮实测;它与旧代码同轮 NumPy 的最大差异小于 1.3%,
因此不会改变纯代码收益的判断。

### 4096 envs 热路径重分布

| 相位 | 优化前 ms | 优化后 ms | 说明 |
| --- | --- | --- | --- |
| pre-kernel | 5.0037 | 0.0805 | motion advance 和 clip 检测进入并行 kernel,噪声不再由 NumPy 生成 |
| read-inputs | 16.3106 | 3.7579 | 去掉 strided 输入拷贝,contact 归约移出 NumPy |
| task-kernel | 0.9724 | 4.7349 | 主动承接噪声、contact 归约和 metrics partial,因此单项耗时增大 |
| post-kernel | 0.1897 | 0.1539 | host 仅保留一次 mean、sampler 和诊断整理 |
| **完整调用** | **22.1532** | **8.7316** | phase 中位数之和与完整调用中位数有小量采样差异 |

这一重分布符合优化目标:task kernel 变重是因为它接管了原先的 NumPy 工作,
不是回归;热路径从多个 NumPy 扫描和拷贝收敛为 3.76ms MotrixSim 读取 +
4.73ms 融合 kernel。

### 线程配置影响

优化后代码在远端默认 192/192 线程下仍受 fork-join 派发、线程超卖和 NUMA
影响。1024 envs 的 `update_state` 在默认配置下为 26.97ms,而 8/48 配置为
2.65ms,相差 **10.16x**。因此代码收益不能替代运行时线程配置;
高核双 NUMA 机器上不应使用两个线程池各自拉满 192 逻辑核的默认值。

### 完整 `step` 收益

完整 `step` 用优化后代码、8/48 线程和 50 次稳态迭代中位数对比:

| num_envs | NumPy ms | Numba ms | 端到端收益 |
| --- | --- | --- | --- |
| 16 | 4.9547 | 3.0710 | 1.61x |
| 256 | 31.5835 | 17.7131 | 1.78x |
| 1024 | 83.3981 | 62.7314 | 1.33x |

`update_state` 的 4-5x 优势在完整 `step` 中被 physics 耗时稀释,但 1024 envs
内仍有 1.33-1.78x 的环境端收益。4096 envs 测量在进入完整 `step` 后未产生
有效输出,需改为独立后台记录的单点基准再确认。

## 主 `task_kernel` 独立调查(commit `3c18924`)

### 调查方法与证据边界

当前对应入口为 `bench/bench_wbt_kernel.py`,在计时区外构造环境、完成 Numba warm-up,
并且只读取一次真实 kernel inputs。历史调查脚本曾包含以下一次性 stage:

- `main`:只调用生产 `task_kernel`,不包含 pre-kernel、MotrixSim getter 和 host post-processing。
- `noise`:只调用主 kernel 头部的串行 `_sample_noise`。
- `zero-noise`:把4组噪声幅值设为0,只测对应数组清零。
- `zero-noise-main`:运行生产主 kernel,但把噪声幅值设为0。
- `launch`:对4096个 float 执行一次极小 `prange`,测量并行派发/同步下限。

当前入口只保留 `main`、`main-after-read`、`read`、`partial-reset-read`、`read-main`、`step` 和 `launch`；
`noise`、`zero-noise`、`zero-noise-main` 属于该次调查的临时 instrumentation，不再作为稳定 benchmark 维护。

固定参数为4096 envs、Numba 0.61.2、`NUMBA_THREADING_LAYER=omp`、
`OMP_DYNAMIC=false`、`RAYON_NUM_THREADS=8`;每个配置启动5个独立进程,
每进程20次预热后采样200次。表中数值是5个“进程内中位数”的中位数,
括号为5个进程中位数的 min-max。

本地绑定8个物理核和node0内存:

```bash
numactl --physcpubind=0-7 --membind=0 env \
  NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=<threads> \
  OMP_DYNAMIC=false RAYON_NUM_THREADS=8 \
  PYTHONPATH=motrix_env_core/src:motrix_envs/src \
  .venv/bin/python bench/bench_wbt_kernel.py \
  --env g1-wbt-dance --stage <stage> --num-envs 4096 --warmup 20 --steps 200
```

远端只把绑定改为 `--physcpubind=0-47 --membind=0`;CPU `0-47` 是node0的
48个不同物理核,不包含其SMT siblings `96-143`。

### cache-hot 主 kernel 线程扩展

这一基准连续复用同一输入快照,所以代表主 kernel 的 cache-hot 上限,不能直接替代
`getter -> kernel` 交替执行的生产热路径。

| Numba线程 | 本地 main ms | 远端 node0 main ms | 远端/本地 |
| ---: | ---: | ---: | ---: |
| 1 | 2.956(2.945-2.976) | 7.479(7.443-7.590) | 2.53x |
| 2 | 1.724(1.719-1.733) | 4.910(4.809-5.219) | 2.85x |
| 4 | 1.122(1.114-1.126) | 3.866(3.763-3.962) | 3.45x |
| 8 | 0.828(0.826-0.830) | 3.390(3.289-3.517) | 4.09x |
| 16 | - | 3.059(2.992-3.073) | - |
| 24 | - | 3.045(3.036-3.147) | - |
| 32 | - | **2.980**(2.964-3.106) | - |
| 48 | - | 3.138(3.097-3.198) | - |

本地主 kernel 从1核到8核加速3.57x;远端从1核到最佳32核只加速2.51x,
48核为2.38x。最佳点停在16-32线程,继续增加线程已经没有收益。

### 纯 Numba 根因:并行计算扩展良好,串行 RNG 主导完整 main

commit `50831d9` 把生产计算体抽成可独立调用的 `WBT_COMPUTE_KERNEL`,生产
`WBT_TASK_KERNEL` 仍按相同顺序执行 noise + compute;7个 WBT parity/行为测试通过。
下面两列均不调用 MotrixSim getter,也不在计时区内调用 Rayon:

- `compute`:连续只跑主 `prange` 计算体。
- `noise-after-compute`:每次先在计时区外跑一次 compute,再只计时下一轮串行 RNG。

| 线程 | 本地 compute ms | 本地 noise-after-compute ms | 远端 compute ms | 远端 noise-after-compute ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.492 | 0.460 | 6.430 | 1.310 |
| 2 | 1.264 | 0.461 | 3.017 | 1.982 |
| 4 | 0.658 | 0.468 | 1.488 | 2.522 |
| 8 | 0.374 | 0.486 | 0.776 | 2.675 |
| 16 | - | - | 0.419 | 2.665 |
| 24 | - | - | 0.320 | 2.795 |
| 32 | - | - | 0.268 | 2.703 |
| 48 | - | - | **0.231** | **2.891** |

每项仍是5个独立进程、每进程20次预热+200次采样的二级中位数。由此确认:

1. 远端纯 compute 从1线程6.430ms降到48线程0.231ms,加速 **27.8x**,
   48核并行效率57.9%;本地1->8线程加速6.66x、效率83.3%。远端仍有跨CCX、
   工作粒度和共享cache损失,但不是“没有均分任务”。
2. 完整 main 的差扩展来自串行 `_sample_noise`:它独立连续运行时远端约1.31ms,
   但紧跟并行计算运行时随线程数升到2.89ms。48线程时 main约3.14ms,
   其中 compute只有约7.4%,串行 RNG/相位切换约92.1%。
3. `compute + noise-after-compute` 可重建完整 main:远端32线程约2.971ms
   (完整 main 2.980ms),48线程约3.122ms(完整 main 3.138ms)。这条证据链完全排除 Rayon。
4. 4096迭代极小 `prange` 在远端48线程只需0.017ms。默认
   `parallel_chunksize=0` 为3.051ms,显式约85 env/线程为3.043ms;
   `chunksize=1` 反而退化到3.567ms。Numba默认静态均分已经合适,动态调度不是优化方向。
5. `OMP_WAIT_POLICY=ACTIVE` 3.049ms,默认3.030ms,`PASSIVE` 3.811ms;
   `GOMP_SPINCOUNT=0` 3.900ms。改变OpenMP等待策略只能在串行竞争和并行唤醒之间转移成本,
   没有优于默认设置。

纯 Numba 的第一优化候选因此是把 observation noise 改为按env独立、调度无关的
counter-based并行 RNG,直接消除串行相位;候选实现必须保留幅值边界、fresh-env
可复现性和零幅值语义,并增加分布统计测试。若这一候选成立,48线程 compute
0.231ms表明高核扩展仍有实际价值。

### Counter-based并行RNG优化(2026-08-09)

已按上述候选实现调度无关的counter RNG。每个噪声样本只由
`(step seed, env_id, component_id)`决定,不共享或顺序推进随机状态;同seed在1线程和
多线程下逐元素一致,不同seed产生不同序列。随机数不再要求匹配NumPy MT19937,
但保留fresh-env复现、幅值边界、零幅值和均匀分布语义。

生产 `WBT_TASK_KERNEL` 没有增加第二个parallel region:每个env在现有主 `prange`
开头生成自己的噪声,随后立即构造observation。独立 `WBT_NOISE_KERNEL` 只供分层
benchmark和分布测试使用。

固定4096 envs、OpenMP、`RAYON_NUM_THREADS=8`,每项仍为5个独立进程、每进程
20次预热+200次采样的二级中位数。下表只含纯Numba/cache-hot阶段,不调用
MotrixSim getter:

| 机器/线程 | 指标 | baseline ms | final ms | 收益 |
| --- | --- | ---: | ---: | ---: |
| 本地8核/8线程 | 完整main | 0.828 | 0.398 | **2.08x** |
| 本地8核/8线程 | compute-only | 0.374 | 0.374 | 1.00x |
| 本地8核/8线程 | compute后的RNG阶段 | 0.486 | 0.0202 | **24.0x** |
| 远端node0/48线程 | 完整main | 3.138 | 0.269 | **11.65x** |
| 远端node0/48线程 | compute-only | 0.231 | 0.241 | 0.96x |
| 远端node0/48线程 | compute后的RNG阶段 | 2.891 | 0.0581 | **49.8x** |

RNG行均在计时区外先运行一次compute,再计时下一轮RNG,因此baseline/final同口径;
final独立连续RNG另为本地0.0175ms、远端0.0353ms。生产路径将RNG融合进main,
因此完整main才是最终判据。final main与compute-only的差仅为本地0.024ms、
远端0.028ms,说明原先主导远端main的串行RNG/相位切换已基本消除。远端
compute-only的4.5%波动没有改变量级,且完整main相对baseline仍有11.65x收益。

### Rayon边界:只用于解释生产 `getter -> kernel`,不混入纯 Numba 结论

`task_kernel` 本身不调用 Rayon。Rayon来自 `_read_kernel_inputs()` 的 MotrixSim getter。
新增 `main-after-read` stage把getter放在计时区外,只计随后main,远端48线程为
6.694ms,复现了完整 `update_state` 分解中的6.62ms;纯 main则为3.14ms。

进一步拆分得到远端48线程 `compute-after-read=3.490ms`、
`noise-after-read=2.067ms`;轮换1到64份真实输入快照(6.8MB到433MB)只把main
从3.045ms提高到3.399ms,不足以解释差异。getter后延迟0-5ms时compute约
3.5-3.8ms,延迟10ms后可降到0.821ms,支持Rayon与OpenMP线程池活动窗口重叠的判断。

三种 Numba threading layer 的远端单进程筛选结果如下;这是生产组合路径的
候选筛选,不是纯 Numba 48核扩展结论:

| layer | main ms | read ms | main-after-read ms | 连续 read+main ms |
| --- | ---: | ---: | ---: | ---: |
| OpenMP | **3.047** | 2.700 | 6.693 | 13.004 |
| TBB | 4.702 | **2.686** | 5.074 | 8.654 |
| workqueue | 4.264 | 3.227 | **4.424** | **7.635** |

OpenMP单独kernel最快,workqueue在getter组合路径筛选中最好。该表是单进程候选筛选;
后续5进程生产序列复测确认TBB/workqueue能降低交接成本,但都没有优于
`GOMP_SPINCOUNT=0`下的OpenMP total。更换Numba线程后端还会牺牲纯kernel性能。

### 生产 `pre` 阶段的libgomp自旋放大

4096 envs完整生产序列中,本地8线程的`pre`约0.012-0.017ms,而远端node0
48线程约5.9ms。`pre`计时区实际包含`WBT_PRE_KERNEL`、`np.any(clip_ended)`和
条件执行的`_handle_clip_end()`。play配置的`time_step_total=999`,环境从step 0
开始;本轮100/200次诊断中没有一次clip结束,因此`_handle_clip_end()`不在热路径。

分层运行表明,连续cache-hot的纯`pre_kernel`在本地/远端分别只有0.0059ms和
0.0289ms;远端在`main + post`后、不调用MotrixSim getter时也只有0.0347ms。
一旦把getter放回前序混合调用,远端`pre`升到3.20ms,完整生产序列进一步升到
5.65-5.86ms。这说明差距不是4096次motion step加法和clip比较的算术成本,
而是MotrixSim/Rayon、host串行阶段和下一次OpenMP parallel region之间的运行时交接。

线程数和threading layer对照进一步定位到GNU OpenMP(libgomp)的48线程team:

| 远端配置 | pre ms | read ms | main ms | total ms |
| --- | ---: | ---: | ---: | ---: |
| OpenMP 48T / Rayon 8T,默认 | 5.862 | 7.431 | 3.722 | 17.990 |
| OpenMP 48T / Rayon 8T,`GOMP_SPINCOUNT=0` | 0.216 | 3.865 | 0.970 | **5.349** |
| TBB 48T / Rayon 8T | **0.147** | **3.269** | 2.118 | 5.600 |
| workqueue 48T / Rayon 8T | 0.520 | 3.840 | 1.224 | 5.864 |
| OpenMP 8T / Rayon 8T,默认 | 0.048 | 3.804 | 1.863 | 5.983 |

每项为5个独立进程、每进程20次预热和100次采样的进程内中位数,表中再取5个
进程中位数的中位数。另一个单进程线程数对照中,OpenMP 48T/Rayon 1T的`pre`
仍为5.65ms,说明问题不是8个Rayon worker简单抢核;OpenMP降到8T或1T后则分别
降到0.0495ms和0.0413ms。

OpenMP等待策略给出了更直接的根因证据:

| OpenMP 48T等待配置 | pre ms | main ms | total ms |
| --- | ---: | ---: | ---: |
| 默认 | 5.793 | 3.640 | 17.973 |
| `OMP_WAIT_POLICY=ACTIVE` | 5.643 | 1.108 | 14.959 |
| `OMP_WAIT_POLICY=PASSIVE` | 0.224 | 1.001 | 5.476 |
| `GOMP_SPINCOUNT=0` | **0.179** | **0.990** | **4.657** |
| `GOMP_SPINCOUNT=100000000` | 5.652 | 3.172 | 15.122 |

默认libgomp worker在混合调用的串行/异构阶段持续自旋,占满node0物理核并放大
下一次parallel region的team同步成本;计时从`pre`前开始,所以这笔运行时成本被
归入`pre`。让worker立即休眠后,pre、getter和main同时恢复到正常量级。
远端`perf_event_paranoid=4`阻止了内核栈采样,因此目前确认的是libgomp
wait/spin这一根因类别,不声称已定位到某个内部barrier/futex符号。

`GOMP_SPINCOUNT=0`是当前OpenMP 48T生产组合的首选候选,必须在Python进程启动前
设置,不能写入`EnvCfg`。它在纯cache-hot kernel中可能增加worker唤醒成本,所以仍需
以完整`update_state`、完整`step`和最终训练吞吐作为生产采用判据;TBB、workqueue
和降低OpenMP线程数保留为对照方案。

### `GOMP_SPINCOUNT=0`正式A/B

正式A/B使用当前counter RNG实现和当时的 legacy WBT task-compute 专项脚本。该私有边界入口已在 benchmark
收敛时删除；当前公开 step A/B 使用 `bench/bench_env.py`。本地绑定
8个物理核,远端绑定node0的48个物理核;两边均固定OpenMP、`OMP_DYNAMIC=false`和
`RAYON_NUM_THREADS=8`。`update_state`每项启动5个独立进程,每进程采样100次;
完整`step`每项启动3个独立进程,每进程预热3次后采样20次。表中为进程内中位数
的中位数,括号是独立进程的min-max。

基线和候选的历史启动参数如下。对应的 legacy task-compute 专项入口以及旧环境配置均已删除，当前仓库不能使用下面的命令重新生成这些数据；这里保留参数仅用于解释历史表格，不把同一个 Env ID 同时作为 `--env` 和 `--compare-env` 的当前 A/B 对照入口。

-   baseline：本地绑定 `0-7`，移除 `GOMP_SPINCOUNT`，使用 `NUMBA_NUM_THREADS=8`。
-   candidate：本地绑定 `0-7`，在 Python 启动前设置 `GOMP_SPINCOUNT=0`，使用 `NUMBA_NUM_THREADS=8`。

远端只把 CPU 绑定改为 `0-47`、`NUMBA_NUM_THREADS` 改为 48，并使用远端 venv。

#### 完整 `update_state`

| 机器 | libgomp配置 | NumPy ms | Numba ms | NumPy/Numba | 配置收益 |
| --- | --- | ---: | ---: | ---: | ---: |
| 本地8核 | 默认 | 16.609 | 4.685(4.475-4.705) | 3.54x | - |
| 本地8核 | `GOMP_SPINCOUNT=0` | 16.696 | **1.137**(1.112-1.145) | **14.66x** | **4.12x** |
| 远端node0 48核 | 默认 | 40.267 | 15.689(14.555-16.988) | 2.57x | - |
| 远端node0 48核 | `GOMP_SPINCOUNT=0` | 40.180 | **5.454**(5.434-5.640) | **7.34x** | **2.88x** |

同一批运行的生产阶段分解如下。阶段中位数之和与完整调用中位数来自不同采样序列,
因此存在小量差异:

| 机器 | libgomp配置 | pre ms | read ms | main ms | post ms | phase total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 本地8核 | 默认 | 0.0169 | 3.8637 | **0.3527** | **0.0614** | 4.4180 |
| 本地8核 | `GOMP_SPINCOUNT=0` | 0.0241 | **0.6685** | 0.3798 | 0.0660 | **1.1473** |
| 远端node0 48核 | 默认 | 5.6980 | 6.5311 | 3.3778 | **0.1481** | 15.9035 |
| 远端node0 48核 | `GOMP_SPINCOUNT=0` | **0.2435** | **3.9974** | **1.0152** | 0.2404 | **5.4918** |

本地8线程的pre原本已处于微秒量级,`spin=0`的主要收益来自不再让OpenMP worker
干扰MotrixSim getter。远端48线程则同时消除了pre的team同步放大,并改善read和
main。post略有增加但绝对值不足0.1ms,不改变总收益。

当前只比较`GOMP_SPINCOUNT=0`下的`update_state`,远端相对本地的剩余差距为:

| 阶段 | 本地8核 ms | 远端node0 48核 ms | 远端/本地 | 对4.358ms阶段中位数差的贡献 |
| --- | ---: | ---: | ---: | ---: |
| pre | 0.0241 | 0.2435 | 10.10x | 5.0% |
| read | 0.6685 | 3.9974 | 5.98x | **76.4%** |
| main | 0.3798 | 1.0152 | 2.67x | 14.6% |
| post | 0.0660 | 0.2404 | 3.64x | 4.0% |
| **phase total** | **1.1473** | **5.4918** | **4.79x** | **100%** |

`GOMP_SPINCOUNT=0`已经消除了pre的5.9ms病态放大;远端pre仍是本地的10倍,
但绝对差只有0.219ms,不再是主要矛盾。各阶段中位数差的76.4%来自
`_read_kernel_inputs()`的MotrixSim getter,14.6%来自融合main kernel。
所以在只看`update_state`的范围内,下一优化优先级应是read,而不是继续修改pre。

### 下一优化方向:`read`调用合并与线程池再平衡

#### 3-getter状态读取已落地

优化前`_state_quantities()`每步调用:

1. root rotation;
2. 14个tracked links的`get_link_states()`;
3. root linear/angular velocity;
4. reference position/rotation;
5. joint position/velocity。

G1的`tracked_body_names`已经包含root `pelvis`和reference `torso_link`,而
`get_link_states()`的13列正好包含position、quaternion、linear velocity和
angular velocity。因此当前实现从tracked states切片root/reference状态,只保留
`get_link_states + joint_pos + joint_vel`三次MotrixSim调用。初始化时缓存
root/reference在tracked states中的slot;任一link未被track时立即报告配置错误。

4096 envs上对当前实现和派生实现逐项比较,base quaternion、base local
linear/angular velocity、reference pose和全部tracked body states均逐元素一致,
最大绝对误差为0。这一实现不需要修改MotrixSim ABI。4个内置WBT配置均包含所需
link;正式测试同时覆盖直接getter等价、NumPy/Numba计算parity和短rollout parity。

三个getter都支持`out=`预分配数组,但五进程复测没有显示稳定收益,部分进程反而
出现更高延迟。因此当前3-getter实现不使用`out=`预分配。

#### 目标服务器采用48/48线程组合

减少小getter以后可以提高Rayon线程数。Numba/Rayon联合扫描覆盖
Rayon 1/2/4/8/16/24/32/48和Numba 8/16/24/32/48;
五进程复测表明24/24是当前`update_state`的纯性能最佳点,但目标服务器当前采用
48/48作为运行配置。每个进程20次预热、100次采样,进程依次运行。

采用配置必须在Python启动前设置:

```bash
GOMP_SPINCOUNT=0 NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=48 \
OMP_DYNAMIC=false RAYON_NUM_THREADS=48 <python-command>
```

这两个线程池在`update_state`中按getter和kernel阶段交替使用,不是96个线程同时
执行同一阶段。48/48相对24/24仍有额外的线程team同步和唤醒成本。

五进程复测结果如下:

| 远端4096 envs候选 | Numba线程 | Rayon线程 | direct `update_state` ms | phase total ms |
| --- | ---: | ---: | ---: | ---: |
| 当前源码正式基线 | 48 | 8 | 5.454(5.434-5.640) | 5.492 |
| 当前源码,只调线程 | 24 | 24 | 3.977(3.176-4.085) | 3.209 |
| 8-getter源码,只调线程 | 48 | 48 | 3.946(3.900-4.026) | 4.072(3.990-4.128) |
| 3-getter派生原型 | 48 | 24 | 3.715(3.273-3.751) | 3.325 |
| 3-getter正式源码 | 48 | 48 | 3.391(3.259-3.479) | 3.383(3.355-3.552) |
| 3-getter派生原型,性能最佳点 | 24 | 24 | **2.924**(2.843-3.518) | **2.883**(2.862-2.984) |
| **3-getter + rotation融合,采用配置** | **48** | **48** | **2.161**(2.102-2.193) | **2.291**(2.218-2.392) |

direct计时在服务器不同频率状态下出现约3.0/3.5ms两档,所以同时列出更稳定的
phase total。最终48/48正式实现相对48/8正式基线direct提升**2.52x**、耗时降低
约60.4%;rotation融合相对3-getter未融合版本再提升**1.57x**、耗时降低约36.3%。
融合后的48/48也已快于此前24/24原型,所以24/24不再是当前性能上界。

正式融合源码在本地8/8、4096 envs下完成同口径五进程串行复测:direct
`update_state`为0.803ms(0.797-0.816ms),phase total为0.831ms
(0.827-0.852ms),read为0.354ms(0.352-0.359ms)。相对3-getter未融合的
1.038ms direct结果耗时下降约22.6%。仓库正式对照脚本同轮运行NumPy和Numba时,
两者分别为16.666ms(16.634-16.717ms)和0.842ms(0.834-0.846ms),Numba加速19.78x。

融合后本地/远端同为4096 envs、五进程依次运行、每进程20次预热和100次采样;
本地使用8/8,远端node0使用48/48,两者均设置`GOMP_SPINCOUNT=0`:

| 阶段 | 本地8/8 ms | 远端48/48 ms | 远端/本地 |
| --- | ---: | ---: | ---: |
| direct `update_state` | **0.803**(0.797-0.816) | **2.161**(2.102-2.193) | 2.69x |
| pre | 0.0228 | 0.1560 | 6.85x |
| read | 0.3535 | 1.1987 | 3.39x |
| main | 0.3860 | 0.7917 | 2.05x |
| post | 0.0704 | 0.1187 | 1.69x |
| phase total | **0.831**(0.827-0.852) | **2.291**(2.218-2.392) | 2.76x |

采用配置的rotation融合前后五进程阶段中位数为:

| 阶段 | 3-getter 48/48 ms | rotation融合48/48 ms | 收益 |
| --- | ---: | ---: | ---: |
| pre | 0.1801 | **0.1560** | 1.15x |
| read | 2.2524 | **1.1987** | 1.88x |
| main | 0.8340 | **0.7917** | 1.05x |
| post | 0.1257 | **0.1187** | 1.06x |
| **phase total** | **3.3828** | **2.2906** | **1.48x** |

性能最佳点24/24原型的五进程阶段中位数为:

| 阶段 | 当前48/8 ms | 原型24/24 ms | 收益 |
| --- | ---: | ---: | ---: |
| pre | 0.2435 | **0.0976** | 2.50x |
| read | 3.9974 | **1.8491** | 2.16x |
| main | 1.0152 | **0.7971** | 1.27x |
| post | 0.2404 | **0.1093** | 2.20x |
| **phase total** | **5.4918** | **2.8832** | **1.90x** |

#### Base velocity inverse rotation已融合进main kernel

对3-getter、24/24原型继续拆分read,五个进程中位数为:

| read子阶段 | ms | 说明 |
| --- | ---: | --- |
| `get_link_states` | 0.743 | 最大单个MotrixSim getter |
| base linear velocity inverse rotation | 0.527 | NumPy host计算 |
| base angular velocity inverse rotation | 0.332 | NumPy host计算 |
| joint position getter | 0.170 | MotrixSim getter |
| joint velocity getter | 0.121 | MotrixSim getter |
| host assembly | 0.005 | 可忽略 |

两次NumPy `quaternion.rotate_inverse`原先合计约0.859ms,已超过`get_link_states`,
占24/24原型read约44%。当前Numba读取层直接传入base world-frame linear/angular
velocity;主kernel在已有`prange`内调用标量`_rotate_inverse()`,结果写入预分配
local-frame velocity buffers,供observation和host diagnostics共同复用。该结构没有
新增独立kernel launch,NumPy `WbtTask`仍通过共享3-getter读取层在host完成转换。

48/48正式复测的phase total从3.383ms降到2.291ms,实际回收1.092ms,优于此前
约0.78ms估算。read现在主要剩下约1.20ms的三个MotrixSim getter和轻量组装。再往下需要在
`motphys-articulated-body`评估通用状态快照、getter复用持久BatchPool或进一步
合并派发,不应先增加WBT专用后端API。

#### 完整 `step`

| 机器/实现 | Numba/Rayon | libgomp配置 | NumPy step ms | Numba step ms | NumPy/Numba |
| --- | ---: | --- | ---: | ---: | ---: |
| 本地8核,融合前 | 8/8 | 默认 | 121.634 | 108.769(108.228-110.906) | 1.12x |
| 本地8核,融合前 | 8/8 | `GOMP_SPINCOUNT=0` | 121.294 | 105.557(104.715-106.116) | 1.15x |
| **本地8核,rotation融合** | **8/8** | **`GOMP_SPINCOUNT=0`** | **119.751**(119.726-120.096) | **103.482**(103.459-103.525) | **1.16x** |
| 远端node0,融合前 | 48/8 | 默认 | 294.750 | 243.904(242.531-244.578) | 1.21x |
| 远端node0,融合前 | 48/8 | `GOMP_SPINCOUNT=0` | 281.672 | 229.048(228.390-229.217) | 1.23x |
| **远端node0,rotation融合** | **48/48** | **`GOMP_SPINCOUNT=0`** | **110.404**(108.258-112.807) | **49.506**(49.392-49.678) | **2.23x** |

最终48/48融合实现的完整step启动3个独立进程,每进程采样20次;Numba step的
进程中位数范围只有49.392-49.678ms,相对同轮NumPy为2.23x。它与旧48/8行同时包含
Rayon线程配置和代码实现差异,不能把229.048ms到49.506ms全部归因于rotation融合。
`update_state`之外仍由physics主导,最终还需要包含policy forward/backward的训练吞吐验证。

### 恢复检查点(2026-08-10)

本地分支 `146-numba-env` 当前HEAD为 `08bd54a`,相对
`origin/146-numba-env` ahead 13;未push。远端原工作区仍停在 `c7f37cc` 且有其自己的
profiling改动,没有覆盖。最新隔离远端 worktree为
`/tmp/morphos-numba-kernel.79R1Xy`(detached `d84ea69`),Python继续使用
`/home/server/prj/morphos-lab/.venv/bin/python`;3-getter、rotation融合生产源码和
WBT测试已同步到该worktree并通过10项测试。

关键原始JSONL:

- 本地完整main线程曲线:`/tmp/wbt-main-local-3c18924.8RJw4n.jsonl`。
- 远端完整main线程曲线:`/tmp/wbt-main-remote-3c18924.bYXQfk.jsonl`。
- 本地纯compute/RNG线程曲线:`/tmp/wbt-pure-numba-scaling-local-d84ea69.5NV9VH.jsonl`。
- 远端纯compute/RNG线程曲线:`/tmp/wbt-pure-numba-scaling-remote-d84ea69.WEQsnH.jsonl`。
- 远端getter后分解:`/tmp/wbt-after-read-remote-2825f17.HRpfB8.jsonl`。

原始文件只在对应机器 `/tmp`;关键二级中位数已写入本文。代码通过git bundle同步,
没有push到GitLab。恢复时先检查远端worktree和 `/tmp` 文件是否仍存在;若已清理,
按本文固定命令重跑即可。

### 深入调查与优化 TODO

- [x] 建立只测生产主 kernel、不含getter/Rayon的独立进程 benchmark。
- [x] 完成本地和远端纯main、compute、RNG的5进程线程扩展曲线。
- [x] 验证默认静态均分、chunksize、OpenMP等待策略和最小派发成本。
- [x] 分离纯Numba结论与生产getter/Rayon组合路径。
- [x] 实现counter-based并行RNG,完成分布、seed确定性、线程数无关性和零噪声parity测试。
- [x] 完成纯Numba的本地8核/远端48核五进程baseline/final;远端main已从3.138ms降至0.269ms。
- [x] 对生产路径的TBB/workqueue筛选结果做5进程复测,同时记录read+main总成本,避免只转移开销。
- [x] 将`GOMP_SPINCOUNT=0`候选带回完整`update_state`和`step`,按本地/远端统一表报告。
- [x] 拆分read getter,验证root/reference状态可从tracked link states逐元素等价派生。
- [x] 完成Rayon和Numba联合线程扫描,五进程确认3-getter、24/24原型。
- [x] 拆分3-getter原型read,确认host quaternion rotation是下一代码热点。
- [x] 落地3-getter状态读取,验证全部内置WBT配置和NumPy/Numba parity。
- [x] 使用正式3-getter源码完成远端48/48五进程benchmark。
- [x] 把base velocity inverse rotation融合进现有main prange并完成parity与正式benchmark。
- [ ] 本地采集cycles/instructions/cache-misses;远端`perf_event_paranoid=4`时继续以分层bench替代。

## 阶段性结论

1. **Numba 路径值得保留**:最终48/48实现在目标服务器上相对最初48/8基线实现
   2.52x `update_state`代码/配置合计收益,相对同轮NumPy的完整step为2.23x。
2. **融合边界已基本到位**:与MotrixSim无关的motion、noise、base velocity rotation、
   contact归约和metrics partial已进入同一个main kernel;继续在Python层做小修补的收益
   不会再是主量级。
3. **当前下限是“MotrixSim 读取 + 1 个融合 kernel”**:`GOMP_SPINCOUNT=0`下,
   3-getter和rotation融合后的48/48正式源码direct中位数为2.16ms,phase total为
   2.29ms;read中的约1.20ms已主要是三个MotrixSim getter。
4. **生产接入仍需明确线程所有权**:`RAYON_NUM_THREADS=48`、`NUMBA_NUM_THREADS=48`
   和`GOMP_SPINCOUNT=0`是当前采用的目标服务器配置,不应写入`EnvCfg`;训练runtime需
   根据物理核、SMT和NUMA拓扑统一设置两个进程级线程池和等待策略。

## 下一步

1. 在`motphys-articulated-body`评估getter复用持久
   BatchPool或通用状态快照,避免提前增加WBT专用API。
2. 在训练runtime中明确Rayon/Numba线程数、OpenMP等待策略与NUMA绑定的所有权,
   再测一次policy forward/backward在内的training throughput;本文的`step`收益
   不等于训练端到端收益。
