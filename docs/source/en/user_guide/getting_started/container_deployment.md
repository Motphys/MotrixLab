# Container Deployment

This document describes how to deploy MotrixLab using Docker containers to simplify environment configuration and enable rapid deployment.

## Prerequisites

-   **Docker** and **Docker Compose**: [Installation Docs](https://docs.docker.com/engine/install/)
-   **NVIDIA Container Toolkit** (for GPU support): [Installation Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
-   **NVIDIA GPU**: Graphics card supporting CUDA 12.8

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Motphys/MotrixLab.git

cd MotrixLab/docker
```

### 2. Start with Docker Compose

We provide a complete `docker-compose.yml` configuration file that supports one-click startup of training and TensorBoard visualization services.

```bash
# Start training and TensorBoard services
docker compose up -d
```

This will start the following services:

-   **motrixlab-training**: Training container that executes reinforcement learning training tasks
-   **motrixlab-tensorboard**: TensorBoard visualization service accessible via browser at http://localhost:6006

### 3. Configure Training Parameters

You can customize training configurations through environment variables:

```bash
# Set training backend (jax or torch)
export MOTRIX_TRAIN_BACKEND=jax

# Set number of parallel environments
export MOTRIX_NUM_ENVS=2048

# Set training environment name
export MOTRIX_ENV=cartpole

# Start services
docker compose up -d
```

| Environment Variable   | Default    | Description                        |
| :--------------------- | :--------- | :--------------------------------- |
| `MOTRIX_TRAIN_BACKEND` | `jax`      | Training backend: `jax` or `torch` |
| `MOTRIX_NUM_ENVS`      | `2048`     | Number of parallel environments    |
| `MOTRIX_ENV`           | `cartpole` | Training environment name          |

### 4. Monitor Training Progress

Training logs are automatically saved to the Docker Volume `motrixlab-data`. You can view them through:

```bash
# View training container logs
docker logs -f motrixlab-training

# Access TensorBoard
# Open in browser: http://localhost:6006
```

### 5. Stop Services

```bash
# Stop all services
docker compose down

# Stop services and remove data volumes
docker compose down -v
```

## Advanced Usage

### Building Docker Images

If you need to customize the image, you can build from source:

```bash
# Execute from project root directory
cd docker
docker build -t motphys/motrixlab:latest .
```

### Running Single Container

If you only want to run the training container without using Docker Compose:

```bash
docker run --gpus all \
    -v $(pwd)/runs:/root/motrixlab/runs \
    motphys/motrixlab:latest \
    mxtrain task=cartpole/skrl.ppo task.train_backend=jax num_envs=2048
```

### Persisting Training Results

The default configuration uses the Docker Volume `motrixlab-data` to save training results. You can mount it to a host directory:

```bash
# Modify volumes configuration in docker-compose.yml
volumes:
  - ./runs:/root/motrixlab/runs
```

## Image Details

Our Docker image is based on {bdg-primary-line}`NVIDIA CUDA 12.8.1` runtime environment and comes pre-installed with the following components:

-   **UV Package Manager**: Fast, reliable dependency management
-   **MotrixLab**: Complete reinforcement learning training framework
-   **SKRL**: Reinforcement learning library supporting both JAX and PyTorch backends
-   **TensorBoard**: Training process visualization tool
-   **MotrixSim**: High-performance physics simulation engine

Image layer build process:

1.  **Base Environment**: NVIDIA CUDA 12.8.1 Runtime + Ubuntu 24.04
2.  **System Dependencies**: Install UV package manager and necessary system tools
3.  **Python Dependencies**: Use UV cache mechanism for fast Python package installation
4.  **Project Code**: Copy MotrixLab source code and complete dependency installation

## Troubleshooting

### GPU Not Available

If the container cannot access the GPU:

```bash
# Check NVIDIA Docker Runtime
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# Verify NVIDIA Container Toolkit is correctly installed
which nvidia-container-cli
```

### Insufficient Storage Space

Clean Docker cache and unused images:

```bash
# Clean build cache
docker builder prune

# Delete unused images
docker image prune -a

# Clean all unused resources
docker system prune -a
```

## Performance Optimization

### Accelerate Builds with UV Cache

The Dockerfile uses UV's cache mount feature, which can significantly speed up rebuilds:

```bash
# Rebuild using UV cache
docker build --cache-from motphys/motrixlab:latest -t motphys/motrixlab:latest .
```

### GPU Resource Allocation

You can specify the number of GPUs to use in `docker-compose.yml`:

```yaml
deploy:
    resources:
        reservations:
            devices:
                - driver: nvidia
                  device_ids: ["0", "1"] # Use GPU 0 and 1
                  capabilities: [gpu]
```

## Next Steps

-   Check out the [Quick Start Tutorial](hello_motrixlab.md) to learn basic MotrixLab usage
-   Browse [Environments](../envs/index.md) for more available tasks
-   Explore [Basic Framework](../tutorial/basic_frame.md) for in-depth understanding of the framework architecture
