# Run MotrixLab with Docker

This page explains how to use the MotrixLab Docker images to train policies, view TensorBoard logs, and deploy exported policies.

## Prerequisites

-   **Docker** and **Docker Compose**
-   For CUDA training images, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
-   CUDA training requires a compatible NVIDIA GPU and host driver; ROCm training requires the AMD GPU kernel driver, normally `amdgpu` / `amdgpu-dkms`, to be installed and loaded on the host

## Choose and Run an Image

When an image is already available locally or in a registry, you do not need a source checkout or a local rebuild. Choose a runtime profile, then mount the run directory and start the container using the examples below:

| Image | Purpose | Entry command |
| --- | --- | --- |
| `motphys/motrixlab:cuda` | CUDA Torch training (latest release) | `train` |
| `motphys/motrixlab:cuda-jax` | CUDA JAX training (latest release) | `train` |
| `motphys/motrixlab:rocm` | AMD ROCm Torch training (latest release) | `train` |
| `motphys/motrixlab:deploy` | Run the latest deployment artifact image | `deploy` arguments |

The smallest CUDA training command is:

```bash
mkdir -p runs
sudo chown -R 1000:1000 runs
docker run --rm --gpus all \
    -v "$PWD/runs:/workspace/runs" \
    motphys/motrixlab:cuda \
    train task=cartpole/skrl.ppo num_envs=2048
```

The container writes run data under `runs/cartpole/`. To run training and TensorBoard together, use Docker Compose as described below.

Choose the CUDA or ROCm image that matches the GPU runtime on your host. `latest` points only to the default CUDA training profile; it does not mean “contains everything.”

## Train with CUDA Compose

From the repository root:

```bash
MOTRIXLAB_IMAGE=motphys/motrixlab:cuda \
MOTRIX_TASK=cartpole/skrl.ppo \
docker compose -f docker/docker-compose.yml up
```

The configuration starts:

-   **motrixlab-training**: the training container
-   **motrixlab-tensorboard**: TensorBoard at `http://localhost:6006`

Common environment variables:

| Environment variable | Default | Description |
| --- | --- | --- |
| `MOTRIXLAB_IMAGE` | `motphys/motrixlab:cuda` | Image used by training and TensorBoard; override it with a version tag when needed |
| `MOTRIX_TASK` | `cartpole/skrl.ppo` | Hydra task configuration |
| `MOTRIX_NUM_ENVS` | `2048` | Number of parallel environments |
| `MOTRIX_ENV` | `cartpole` | Run directory read by TensorBoard |
| `MOTRIX_TENSORBOARD_PORT` | `6006` | TensorBoard host port |

Training output is stored under `/workspace/runs` in the `motrixlab-data` Docker volume. To use a host directory instead, change the Compose volume to:

```yaml
volumes:
    - ./runs:/workspace/runs
```

The named volume inherits the initial permissions of `/workspace/runs` from the image. If you use a host directory, make it writable by the image's `motrix` user (UID/GID `1000:1000`) first:

```bash
mkdir -p runs
sudo chown -R 1000:1000 runs
```

## Run a Training Image Directly

The image entrypoint is `mxlab`, so arguments passed to `docker run` go directly to the CLI:

```bash
docker run --rm --gpus all \
    -v "$PWD/runs:/workspace/runs" \
    motphys/motrixlab:cuda \
    train task=cartpole/skrl.ppo num_envs=2048
```

The image runs as the non-root user with UID/GID `1000:1000`. When you bind mount `/workspace/runs`, the corresponding host directory must be writable by that user. If you do not need the run files directly in a host directory, use the `motrixlab-data` named volume from Compose instead.

SKRL selects the training backend automatically from the available GPU and runtime. To force a specific backend, override the corresponding field in the selected task's Hydra configuration.

The CUDA JAX image uses the same interface:

```bash
docker run --rm --gpus all \
    -v "$PWD/runs:/workspace/runs" \
    motphys/motrixlab:cuda-jax \
    train task=cartpole/skrl.ppo num_envs=2048
```

## Train with AMD ROCm

AMD ROCm does not use `nvidia-container-toolkit`, and the GPU kernel driver must not be installed inside the container. Install the AMD GPU kernel driver on the host, then expose the AMD compute and DRI devices when starting the container:

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

-   `/dev/kfd` is the ROCm compute interface and is required.
-   `/dev/dri` provides the GPU DRI/render nodes. To expose only the first GPU, replace it with `/dev/dri/renderD128`.
-   The `video` and `render` group memberships let the non-root container user access the devices.
-   `seccomp=unconfined` is an optional AMD recommendation for HPC/memory mapping workloads. If your security policy requires it, first test whether training works without the option.

The repository provides both `docker/docker-compose.yml` and `docker/docker-compose.rocm.yml`. The former uses an NVIDIA GPU reservation; the latter uses AMD device passthrough. AMD training can also use Compose directly:

```bash
MOTRIXLAB_IMAGE=motphys/motrixlab:rocm \
MOTRIX_TASK=cartpole/skrl.ppo \
docker compose -f docker/docker-compose.rocm.yml up
```

The ROCm Compose file passes `/dev/kfd` and `/dev/dri` by default. To expose only the first GPU, set `MOTRIX_ROCM_RENDER_DEVICE=/dev/dri/renderD128`. Do not use the default CUDA file's `deploy.resources...driver: nvidia` reservation for AMD.

Before training, check on the host:

```bash
lsmod | grep amdgpu
ls -l /dev/kfd /dev/dri/renderD*
```

Use the following command to verify that ROCm Torch can see the GPU inside the container:

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

See the [ROCm Docker documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.2/how-to/docker.html) for AMD's official parameter descriptions.

## Run a Policy with the Deployment Image

The `deploy` image runs an existing deployment artifact. Create the artifact from a training run first, then mount it and run the deployment image:

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

The deployment image entrypoint calls `motrix_deploy.cli:main` directly; its arguments have the same meaning as the arguments after `mxlab deploy`. See the [Docker image guide](https://github.com/Motphys/MotrixLab/blob/main/docker/README.md) for hardware deployment, Unitree SDK images, and real-time runtime settings.

## Build Images from Source (Maintainers)

Build from the repository root so workspace manifests, source, and LFS assets are part of the context:

```bash
git lfs pull
```

The four commonly used runtime images can be built as follows:

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

Use `train-cuda`, `train-cuda-jax`, and `train-rocm` for training, and `deploy` for policy deployment. The `unitree` target requires a separate Unitree SDK2 image and cannot be built directly as one of these four profiles. The Dockerfile installs locked third-party dependencies before copying workspace source and the root `configs/` directory, so task configurations are available in training images and routine source edits reuse the heavyweight dependency layers.

## Troubleshooting

### NVIDIA GPU unavailable

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

If this fails, check the host NVIDIA driver and NVIDIA Container Toolkit configuration.

### AMD GPU unavailable

First confirm that `amdgpu` is loaded on the host and that `/dev/kfd` and `/dev/dri/renderD*` exist. Then confirm that both device paths were passed to the container and that the `video` / `render` group memberships were added. Do not use `--gpus all` or the NVIDIA Compose reservation to determine whether ROCm is available.

### Invalid robot assets

If build logs show LFS pointers or runtime cannot find meshes, run:

```bash
git lfs install
git lfs pull
```

Then rebuild the image.

### Insufficient storage

```bash
docker builder prune
docker image prune -a
docker system prune -a
```

## Next Steps

-   Read the [quick-start tutorial](hello_motrixlab.md)
-   Browse [environments](../envs/index.md)
-   Explore the [framework overview](../tutorial/basic_frame.md)
