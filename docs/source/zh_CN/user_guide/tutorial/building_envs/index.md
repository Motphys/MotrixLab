# 环境构建总览

本节介绍如何在 MotrixLab 中编写自己的强化学习环境。开始之前，先建立三个整体印象。
## 一个环境由什么组成

一个环境通常由两部分组成，分别注册到环境名上：

- **配置类**：一个 `@configclass` 数据类，声明场景、仿真参数与任务参数，
  通过 `@registry.envcfg("name")` 注册；
- **环境类**：实现任务逻辑，通过 `@registry.env("name")` 注册。

注册完成后，`scripts/view.py`、训练器与回放流程都通过同一个注册表按名字创建环境。


## 生命周期总览

两种工作流共享同一个向量化环境生命周期，`step(actions)` 的固定时序为：

```{image} /_static/images/tutorial/env-lifecycle-light.svg
:alt: ArrayEnv 单个控制步流水线：apply_action → physics_step → compute_transition → 截断判定 → auto-reset → compute_observation
:class: only-light
```

```{image} /_static/images/tutorial/env-lifecycle-dark.svg
:alt: ArrayEnv 单个控制步流水线：apply_action → physics_step → compute_transition → 截断判定 → auto-reset → compute_observation
:class: only-dark
```

所有阶段都由 `ArrayEnv` 基类编排，环境实现只填充钩子，不要重复实现生命周期。语义约定：

- `terminated` 是任务失败等回合终止条件；`truncated` 是达到回合时长上限的时间截断，
  `info["time_outs"]` 标记"截断但未失败"的行；
- done 的环境在每步末尾被自动重置，观测在重置之后重新计算。

各阶段在两种工作流中分别由谁实现：DirectEnv 在
[编写 DirectEnv 环境](direct_env.md#生命周期)中逐钩子展开，ManagerEnv 由
[编写 ManagerEnv 环境](manager_env.md#生命周期)中的配置组驱动。


## 两种工作流怎么选

MotrixLab 提供两种环境工作流，区别在于任务逻辑写在哪里：

| 维度      | DirectEnv（直接工作流）             | ManagerEnv（Manager 工作流）                    |
| --------- | ----------------------------------- | ----------------------------------------------- |
| 配置基类  | `DirectEnvCfg`                      | `ManagerBasedEnvCfg`                            |
| 任务逻辑  | 在环境类钩子中手写                  | 在配置中逐项声明，由 manager 编译为 kernel  |
| 奖励/终止 | `compute_transition` 中手写数组运算 | `rewards` / `terminations` 配置组               |
| 观测      | `compute_observation` 中手写拼接    | `observations` 配置组（`policy`/`value`）       |
| 重置逻辑  | 覆写 `reset(env_ids)` 手写          | `sim_reset` 配置组 + command 重置钩子           |
| 典型场景  | 简单或高度定制的任务                | locomotion 等组合式任务                         |

简单任务的逻辑一目了然，直接工作流最省事；当奖励、观测、终止各项的数量多、需要在多个
环境间复用时，Manager 工作流的声明式组合更易维护。


## 奖励与终止

两种工作流只影响奖励的**写法**，不影响设计思路：

- DirectEnv 在 `compute_transition` 中手写奖励数组运算；ManagerEnv 在 `rewards`
  配置组逐项声明，最终奖励是各项按 `weight` 的加权和。终止条件同理，
  分别对应 `compute_transition` 中的 `terminated` 掩码与 `terminations` 配置组。
- 设计上建议：每个奖励项只负责一个明确的目标；用平滑函数（如指数衰减）
  代替硬阈值；把权重放在配置中，便于逐项调试；检查是否存在"刷分"漏洞。

具体示例见各内置环境的实现与[编写 ManagerEnv 环境](manager_env.md)的
奖励与终止一节。


## 阅读路径

1. [编写 DirectEnv 环境](direct_env.md)：从最小示例理解环境骨架；
2. [编写 ManagerEnv 环境](manager_env.md)：需要声明式组合时切换到 Manager 工作流；
3. [SceneCfg：搭建物理场景](scene.md)：场景文件与仿真参数的细节；
4. [程序化地形生成](terrain.md)：用声明式生成器搭建高度场地形；
5. [SimBackend：与仿真器解耦](sim_backend.md)：环境与仿真器的边界、backend 选择与接入。

```{toctree}
:hidden:

direct_env
manager_env
scene
terrain
sim_backend
```
