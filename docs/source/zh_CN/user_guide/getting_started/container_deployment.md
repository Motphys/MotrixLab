# 容器部署

本文档介绍如何使用 Docker 容器化部署 MotrixLab，以简化环境配置并实现快速部署。

## 前置要求

-   **Docker** 和 **Docker Compose**: [安装文档](https://docs.docker.com/engine/install/)
-   **NVIDIA Container Toolkit**: [安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
-   **NVIDIA GPU**: 支持 CUDA 12.8 的显卡

## 快速开始

### 1. 克隆项目仓库

```bash
git clone https://github.com/Motphys/MotrixLab.git

cd MotrixLab/docker
```

### 2. 使用 Docker Compose 启动

我们提供了完整的 `docker-compose.yml` 配置文件，支持一键启动训练和 TensorBoard 可视化服务。

```bash
# 启动训练和 TensorBoard 服务
docker compose up -d
```

这将启动以下服务：

-   **motrixlab-training**: 训练容器，执行强化学习训练任务
-   **motrixlab-tensorboard**: TensorBoard 可视化服务，通过浏览器访问 http://localhost:6006

### 3. 配置训练参数

您可以通过环境变量自定义训练配置：

```bash
# 设置训练后端（jax 或 torch）
export MOTRIX_TRAIN_BACKEND=jax

# 设置并行环境数量
export MOTRIX_NUM_ENVS=2048

# 设置训练环境名称
export MOTRIX_ENV=cartpole

# 启动服务
docker compose up -d
```

| 环境变量               | 默认值     | 说明                       |
| :--------------------- | :--------- | :------------------------- |
| `MOTRIX_TRAIN_BACKEND` | `jax`      | 训练后端：`jax` 或 `torch` |
| `MOTRIX_NUM_ENVS`      | `2048`     | 并行环境数量               |
| `MOTRIX_ENV`           | `cartpole` | 训练环境名称               |

### 4. 查看训练进度

训练日志会自动保存到 Docker Volume `motrixlab-data` 中。您可以通过以下方式查看：

```bash
# 查看训练容器日志
docker logs -f motrixlab-training

# 访问 TensorBoard
# 浏览器打开: http://localhost:6006
```

### 5. 停止服务

```bash
# 停止所有服务
docker compose down

# 停止服务并删除数据卷
docker compose down -v
```

## 高级用法

### 构建 Docker 镜像

如果您需要自定义镜像，可以从源代码构建：

```bash
# 在项目根目录执行
cd docker
docker build -t motphys/motrixlab:latest .
```

### 运行单个容器

如果您只想运行训练容器而不使用 Docker Compose：

```bash
docker run --gpus all \
    -v $(pwd)/runs:/root/motrixlab/runs \
    motphys/motrixlab:latest \
    mxlab train task=cartpole/skrl.ppo task.train_backend=jax num_envs=2048
```

### 持久化训练结果

默认配置使用 Docker Volume `motrixlab-data` 保存训练结果。您可以将其挂载到主机目录：

```bash
# 修改 docker-compose.yml 中的 volumes 配置
volumes:
  - ./runs:/root/motrixlab/runs
```

## 镜像说明

我们的 Docker 镜像基于 {bdg-primary-line}`NVIDIA CUDA 12.8.1` 运行时环境，预装了以下组件：

-   **UV 包管理器**: 快速、可靠的依赖管理
-   **SKRL**: 支持 JAX 和 PyTorch 后端的强化学习库
-   **TensorBoard**: 训练过程可视化工具
-   **MotrixSim**: 高性能物理仿真引擎
-   **MotrixLab**: 完整的强化学习训练框架

镜像层构建过程：

1.  **基础环境**: NVIDIA CUDA 12.8.1 Runtime + Ubuntu 24.04
2.  **系统依赖**: 安装 UV 包管理器和必要的系统工具
3.  **Python 依赖**: 使用 UV 缓存机制快速安装 Python 包
4.  **项目代码**: 复制 MotrixLab 源代码并完成依赖安装

## 故障排查

### GPU 不可用

如果容器无法访问 GPU：

```bash
# 检查 NVIDIA Docker Runtime
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# 确认 NVIDIA Container Toolkit 已正确安装
which nvidia-container-cli
```

### 存储空间不足

清理 Docker 缓存和未使用的镜像：

```bash
# 清理构建缓存
docker builder prune

# 删除未使用的镜像
docker image prune -a

# 清理所有未使用的资源
docker system prune -a
```

## 性能优化

### 使用 UV 缓存加速构建

Dockerfile 使用了 UV 的缓存挂载功能，可以显著加速重建过程：

```bash
# 利用 UV 缓存重新构建
docker build --cache-from motphys/motrixlab:latest -t motphys/motrixlab:latest .
```

### GPU 资源分配

您可以在 `docker-compose.yml` 中指定 GPU 使用数量：

```yaml
deploy:
    resources:
        reservations:
            devices:
                - driver: nvidia
                  device_ids: ["0", "1"] # 使用 GPU 0 和 1
                  capabilities: [gpu]
```

## 下一步

-   查看 [快速入门教程](hello_motrixlab.md) 了解 MotrixLab 基本用法
-   浏览 [环境](../envs/index.md) 了解更多可用任务
-   探索 [基础框架](../tutorial/basic_frame.md) 深入理解框架架构
