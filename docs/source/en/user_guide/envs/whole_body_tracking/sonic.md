# SONIC G1 Tracking

SONIC is a manager-based Unitree G1 motion-tracking task with a dedicated
FastSAC actor. The policy combines proprioceptive history with G1 and SMPL
future-reference streams selected by a two-value encoder mask.

## Environments

| Environment | Temporal profile | Intended use |
| --- | --- | --- |
| `g1-sonic` | 10 frames | Release-capacity training and official checkpoint playback |
| `g1-sonic-lafan` | 4 frames | Intermediate packed-corpus training |
| `g1-sonic-smoke` | 1 frame | Contract and integration tests using the bundled small store |

All three variants fall back to the bundled smoke store so registry checks and
short integration runs work from a clean checkout. Meaningful training with the
release or LAFAN profile requires the full motion corpus supplied separately.
Point the task at a native packed store:

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

## Smoke run

From the repository root, the smoke profile needs no external motion data:

```bash
python scripts/train.py task=g1-sonic-smoke/motrix.fastsac \
  num_envs=4 algo.trainer.num_learning_iterations=2
```

See [Training Artifacts](../../tutorial/runs_and_checkpoints.md) for playback
of MotrixLab checkpoints and official SONIC release checkpoints.
