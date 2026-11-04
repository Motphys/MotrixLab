# 使用 Docker 运行 MotrixLab

本文介绍如何使用 MotrixLab 提供的 Docker 镜像运行训练、查看 TensorBoard 日志，以及部署导出的策略。

## 前置要求

-   **Docker** 与 **Docker Compose**
-   使用 CUDA 训练镜像时，安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
-   CUDA 训练需要宿主机 NVIDIA 驱动和对应 GPU；ROCm 训练需要宿主机安装并加载 AMD GPU 内核驱动（通常是 `amdgpu` / `amdgpu-dkms`）

## 选择并运行镜像

如果镜像已经存在于本机或镜像仓库，不需要检出源码，也不需要重新构建。先选择运行 profile，再按后面的示例挂载输出目录并启动容器：

| 镜像 | 用途 | 入口命令 |
| --- | --- | --- |
| `motphys/motrixlab:cuda` | CUDA Torch 训练（最新版本） | `train` |
| `motphys/motrixlab:cuda-jax` | CUDA JAX 训练（最新版本） | `train` |
| `motphys/motrixlab:rocm` | AMD ROCm Torch 训练（最新版本） | `train` |
| `motphys/motrixlab:deploy` | 运行 deployment artifact（最新版本） | `deploy` 参数 |

最小的 CUDA 训练命令如下；容器会将运行结果写入当前目录的 `runs/`：

```bash
mkdir -p runs
sudo chown -R 1000:1000 runs
docker run --rm --gpus all \
    -v "$PWD/runs:/workspace/runs" \
    motphys/motrixlab:cuda \
    train task=cartpole/skrl.ppo num_envs=2048
```

之后可以用 TensorBoard 查看 `runs/cartpole/` 下的日志。若使用 Docker Compose，训练和 TensorBoard 会一起启动，见下文。

请根据宿主机 GPU 选择 CUDA 或 ROCm 镜像；两者面向不同的 GPU 运行时。`latest` 只指向默认 CUDA 训练镜像，不表示包含全部功能。

## 使用 CUDA Compose 训练

在仓库根目录执行：

```bash
MOTRIXLAB_IMAGE=motphys/motrixlab:cuda \
MOTRIX_TASK=cartpole/skrl.ppo \
docker compose -f docker/docker-compose.yml up
```

该配置会启动：

-   **motrixlab-training**：训练容器
-   **motrixlab-tensorboard**：TensorBoard 服务，访问 `http://localhost:6006`

常用环境变量：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MOTRIXLAB_IMAGE` | `motphys/motrixlab:cuda` | 训练和 TensorBoard 使用的镜像；可改为版本标签 |
| `MOTRIX_TASK` | `cartpole/skrl.ppo` | Hydra task 配置 |
| `MOTRIX_NUM_ENVS` | `2048` | 并行环境数量 |
| `MOTRIX_ENV` | `cartpole` | TensorBoard 读取的 run 目录名 |
| `MOTRIX_TENSORBOARD_PORT` | `6006` | TensorBoard 宿主机端口 |

训练结果保存在 `motrixlab-data` Docker volume 中的 `/workspace/runs`。若要绑定宿主机目录，将 Compose 中的 volume 改为：

```yaml
volumes:
    - ./runs:/workspace/runs
```

named volume 会继承镜像中 `/workspace/runs` 的初始权限。若改用宿主机目录，请先让目录可由镜像的 `motrix` 用户（UID/GID `1000:1000`）写入：

```bash
mkdir -p runs
sudo chown -R 1000:1000 runs
```

## 直接运行训练镜像

镜像入口是 `mxlab`，因此 `docker run` 的参数直接传给 CLI：

```bash
docker run --rm --gpus all \
    -v "$PWD/runs:/workspace/runs" \
    motphys/motrixlab:cuda \
    train task=cartpole/skrl.ppo num_envs=2048
```

镜像默认以 UID/GID `1000:1000` 的非 root 用户运行。使用 bind mount 时，宿主机的 `/workspace/runs` 对应目录必须允许该用户写入；不需要从宿主机读取结果时，可以改用 Compose 中的 `motrixlab-data` named volume。

SKRL 会根据 GPU 和已安装 runtime 自动选择训练 backend。需要强制指定时，请按所选 task 的 Hydra 配置覆写对应字段。

CUDA JAX 镜像的用法相同：

```bash
docker run --rm --gpus all \
    -v "$PWD/runs:/workspace/runs" \
    motphys/motrixlab:cuda-jax \
    train task=cartpole/skrl.ppo num_envs=2048
```

## 使用 AMD ROCm 训练

AMD ROCm 不使用 `nvidia-container-toolkit`，也不要在容器内安装 GPU 内核驱动。宿主机必须安装 AMD GPU 内核驱动；容器启动时暴露 AMD compute 和 DRI 设备：

```bash
docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --shm-size 1g \
    -v "$PWD/runs:/workspace/runs" \
    motphys/motrixlab:rocm \
    train task=cartpole/skrl.ppo num_envs=2048
```

-   `/dev/kfd` 是 ROCm compute 接口，必须传入。
-   `/dev/dri` 提供 GPU DRI/render 节点；只暴露第一块 GPU 时可改为 `/dev/dri/renderD128`。
-   `video` 与 `render` 组权限用于让容器内非 root 用户访问设备。
-   `seccomp=unconfined` 是 AMD 推荐的可选项，用于 HPC/memory mapping 场景；若安全策略要求，可先验证训练能否在不开启它的情况下运行。

仓库同时提供 `docker/docker-compose.yml` 和 `docker/docker-compose.rocm.yml`。前者使用 NVIDIA GPU reservation，后者使用 AMD 设备透传。AMD 可以直接用 Compose 训练：

```bash
MOTRIXLAB_IMAGE=motphys/motrixlab:rocm \
MOTRIX_TASK=cartpole/skrl.ppo \
docker compose -f docker/docker-compose.rocm.yml up
```

ROCm Compose 默认透传 `/dev/kfd` 与 `/dev/dri`。若只暴露第一块 GPU，可设置 `MOTRIX_ROCM_RENDER_DEVICE=/dev/dri/renderD128`。不要把默认 CUDA Compose 文件中的 `deploy.resources...driver: nvidia` 用作 AMD 配置。

训练前可在宿主机检查：

```bash
lsmod | grep amdgpu
ls -l /dev/kfd /dev/dri/renderD*
```

容器内可用以下命令确认 ROCm Torch 能否看到 GPU：

```bash
docker run --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --entrypoint /app/.venv/bin/python \
    motphys/motrixlab:rocm \
    -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

更多 AMD 官方容器参数说明见 [ROCm Docker 文档](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.2/how-to/docker.html)。

## 使用部署镜像运行策略

`deploy` 镜像用于运行已经导出的 deployment artifact。先使用训练结果生成 artifact，再挂载 artifact 运行部署镜像：

```bash
mxlab export-deploy run_dir=/path/to/run output=artifacts/policy.deploy
```

```bash
docker run --rm \
    -v "$PWD/artifacts/g1-wbt-dance.deploy:/workspace/artifacts/g1-wbt-dance.deploy:ro" \
    motphys/motrixlab:deploy \
    task=g1-wbt-dance/sim \
    artifact=/workspace/artifacts/g1-wbt-dance.deploy \
    runtime.viewer=false
```

部署镜像入口直接调用 `motrix_deploy.cli:main`，参数语义与 `mxlab deploy` 后续参数相同。硬件部署、Unitree SDK 镜像和实时运行参数见 [Docker 镜像说明](https://github.com/Motphys/MotrixLab/blob/main/docker/README.md)。

## 从源码构建镜像（维护者）

必须在仓库根目录构建，使 workspace manifests、源码和 LFS 资产进入 build context：

```bash
git lfs pull
```

常用的四个运行镜像可以分别构建为：

```bash
docker build -f docker/Dockerfile --target train-cuda \
    -t motphys/motrixlab:0.4.0-cuda .
docker build -f docker/Dockerfile --target train-cuda-jax \
    -t motphys/motrixlab:0.4.0-cuda-jax .
docker build -f docker/Dockerfile --target train-rocm \
    -t motphys/motrixlab:0.4.0-rocm .
docker build -f docker/Dockerfile --target deploy \
    -t motphys/motrixlab:0.4.0-deploy .
```

`train-cuda`、`train-cuda-jax` 和 `train-rocm` 用于训练，`deploy` 用于策略部署。`unitree` 需要额外的 Unitree SDK2 镜像，不能按上述四个 target 直接构建。Dockerfile 先安装锁定的第三方依赖，再复制 workspace 源码和根目录 `configs/`，因此 task 配置会随训练镜像提供，普通代码改动也可以复用重型依赖层。

## 故障排查

### NVIDIA GPU 不可用

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

若命令失败，请检查宿主机 NVIDIA 驱动和 NVIDIA Container Toolkit 配置。

### AMD GPU 不可用

先确认宿主机已加载 `amdgpu`，且存在 `/dev/kfd` 与 `/dev/dri/renderD*`。再确认容器启动参数包含这两个设备路径，并添加了 `video` / `render` 组权限。不要用 `--gpus all` 或 NVIDIA Compose reservation 判断 ROCm 是否可用。

### 机器人资产无效

构建日志出现 LFS pointer 或运行时找不到 mesh 时，执行：

```bash
git lfs install
git lfs pull
```

然后重新构建镜像。

### 存储空间不足

```bash
docker builder prune
docker image prune -a
docker system prune -a
```

## 下一步

-   查看 [快速入门教程](hello_motrixlab.md)
-   浏览 [环境](../envs/index.md)
-   探索 [基础框架](../tutorial/basic_frame.md)
