# MotrixLab Docker Images

MotrixLab uses one multi-target Dockerfile and several runtime profiles. Images are split by execution capability—not by Python package—because `mxlab` needs the environment, robot, simulator, and RL registries to be present together.

## Build

Build from the repository root:

```bash
git lfs pull

docker build -f docker/Dockerfile --target train-cuda \
  -t motphys/motrixlab:0.4.0-cuda .
```

Available targets are:

| Target | Image suffix | Contents |
| --- | --- | --- |
| `train-cuda` | `-cuda` | Default CUDA training runtime with SKRL Torch, RSLRL, FastSAC, MotrixSim, and TensorBoard |
| `train-cuda-jax` | `-cuda-jax` | CUDA training runtime plus the SKRL JAX stack |
| `train-rocm` | `-rocm` | ROCm Torch training runtime |
| `deploy` | `-deploy` | ONNX deployment runtime with MotrixSim and MuJoCo simulation backends; no Torch/JAX training stack |
| `unitree` | `-unitree` | Unitree SDK2/DDS deployment runtime |

If `--target` is omitted, the final stage selects the default CUDA training profile.

The CUDA and ROCm profiles are built separately because their locked Torch stacks are mutually exclusive.

GPU acceleration still requires the corresponding host driver and container runtime: NVIDIA Container Toolkit for CUDA images, or an AMD GPU kernel driver for ROCm images.

ROCm does not use `nvidia-container-toolkit`. With an AMD host driver, expose `/dev/kfd` and `/dev/dri` to the container, add the `video` and `render` groups, and optionally use `seccomp=unconfined` as recommended by AMD.

Dependency manifests are installed before workspace source is copied. BuildKit's uv cache is also reused across builds, so routine source changes do not redownload Torch, JAX, or MotrixSim.

## Use

Start the training service and TensorBoard with Docker Compose:

```bash
MOTRIXLAB_IMAGE=motphys/motrixlab:0.4.0-cuda \
MOTRIX_TASK=cartpole/skrl.ppo \
docker compose -f docker/docker-compose.yml up
```

Training output is written to the `motrixlab-data` volume. TensorBoard listens on port 6006 by default.

Use the dedicated ROCm Compose file on AMD hosts:

```bash
MOTRIXLAB_IMAGE=motphys/motrixlab:0.4.0-rocm \
MOTRIX_TASK=cartpole/skrl.ppo \
docker compose -f docker/docker-compose.rocm.yml up
```

By default, the ROCm service exposes `/dev/kfd` and `/dev/dri`. Override `MOTRIX_ROCM_COMPUTE_DEVICE` and `MOTRIX_ROCM_RENDER_DEVICE` to select a specific render node, such as `/dev/dri/renderD128`.

For deployment, mount an exported artifact and pass the deployment CLI arguments to the image entrypoint:

```bash
docker run --rm \
  -v "$PWD/artifacts/g1-wbt-dance.deploy:/workspace/artifacts/g1-wbt-dance.deploy:ro" \
  motphys/motrixlab:0.4.0-deploy \
  task=g1-wbt-dance/sim \
  artifact=/workspace/artifacts/g1-wbt-dance.deploy \
  runtime.viewer=false
```

The deployment entrypoint calls `motrix_deploy.cli:main` directly. Its arguments are the same arguments that follow `mxlab deploy`.

## Unitree SDK

By default, the `unitree` target builds the Python deployment stack but skips the native extension because Unitree SDK2 is not bundled. The runtime then reports that native transport is unavailable. This keeps automated image builds possible.

For hardware transport, build a private base image that contains Unitree SDK2 below `/opt/unitree_robotics`, then pass its reference as `UNITREE_SDK_IMAGE`:

```bash
docker build -f docker/Dockerfile --target unitree \
  --build-arg UNITREE_SDK_IMAGE=my-unitree-sdk2 \
  -t my-registry/motrixlab-unitree:0.4.0 .
```

The final image expects the SDK under `/opt/unitree_robotics`; the native extension records the appropriate runtime library path.
