# SONIC G1 Tracking

SONIC is a manager-based Unitree G1 motion-tracking task whose dedicated actor is
built by the FastSAC `sonic` PolicyVariant. The policy combines 10 frames of
proprioceptive history with G1 and SMPL future-reference streams selected by a
two-value encoder mask. This temporal depth is the public contract of
`g1-sonic`; training, LAFAN data, and small-scale validation all use the same
environment.

The task recipe inlines `algo.variant` in
`configs/task/g1-sonic/motrix.fastsac.yaml`. `model.num_future_frames` defines
both the environment policy-observation width and the actor input width;
`auxiliary` contains weights for three named auxiliary losses, which FastSAC
weights uniformly and logs as `aux_*` metrics. `g1_control_decoder_hidden_dims`
does not appear in the training configuration: the training actor omits
that decoder and uses the FastSAC policy head.

## Observations and actions

The `g1-sonic` policy observation has width 2412 and the privileged value
observation has width 1645. Actions are 29 normalized joint-position targets;
the environment action term owns action scaling and offsets, so the FastSAC actor
keeps an identity action affine.

## Data and execution

The environment falls back to the bundled small smoke store so registry checks
and short integration runs work from a clean checkout. Meaningful training
requires the full motion corpus supplied separately. Point the task at a native
packed store:

```bash
source .venv/bin/activate
SONIC_PACKED_STORE=$PWD/data/sonic/lafan1-packed \
  python scripts/train.py task=g1-sonic/motrix.fastsac
```

`SONIC_DATA_ROOT` can instead select a directory containing a single
`sonic.npz`. When both are available, `SONIC_PACKED_STORE` selects the
read-only memory-mapped corpus.

## Build a packed store

The packer consumes paired robot and SMPL NPZ directories. Input quaternions
are `wxyz`; output quaternions are `xyzw`, and joint/body columns are
reordered to the task contract.

```bash
python scripts/motion/pack_sonic_data.py \
  /path/to/robot_filtered \
  /path/to/smpl_filtered \
  data/sonic/lafan1-packed
```

The output format is versioned as `motrixlab_sonic_packed_v1`. The generated
directory is ignored by Git; only the small smoke store is bundled.

## Small-scale validation

From the repository root, the bundled smoke store needs no external motion data.
Small-scale validation uses the same `g1-sonic` recipe and overrides only scale
and duration:

```bash
python scripts/train.py task=g1-sonic/motrix.fastsac \
  num_envs=32 play_num_envs=4 checkpoint.interval=100 \
  algo.trainer.num_learning_iterations=1000
```

See [Training Artifacts](../../tutorial/training/runs_and_checkpoints.md) for playback
of MotrixLab checkpoints.
