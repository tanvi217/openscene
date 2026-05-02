# Running OpenScene Evaluation on Unity Cluster

## Overview

[OpenScene](https://pengsongyou.github.io/openscene) is an open-vocabulary 3D scene understanding model. It lifts 2D vision-language features (from OpenSeg / CLIP) onto 3D point clouds and classifies points by comparing their features against CLIP text embeddings — no fixed label set required.

This guide covers running **fusion mode** evaluation (no 3D model needed) on the Unity cluster.

---

## Prerequisites

### 1. Allocate a GPU node

```bash
salloc -p gpu -G 1 -N 1 -c 4 --constraint=vram40 --time=03:59:00 --mem 50000 --qos=short --no-shell
```

Then SSH into it:

```bash
srun --jobid=<JOBID> --pty bash
```

### 2. Activate the conda environment

```bash
module load conda/latest
conda activate repopt
```

### 3. Navigate to the OpenScene directory

```bash
cd ~/repopt-3d/third_party/openscene
```

---

## Running ScanNet Evaluation (Fusion Mode)

> **Note:** `MinkowskiEngine` is not installed in the `repopt` environment.  
> Only **fusion mode** is supported — it uses pre-computed 2D features directly and requires no 3D network.

### Available data (read-only)


| Path                                                                                      | Contents                                                 |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_3d/val/`            | 3 ScanNet val scenes (`.pth`)                            |
| `/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_multiview_openseg/` | Pre-fused 2D OpenSeg features for those 3 scenes (`.pt`) |


### Run command

```bash
mkdir -p out/scannet_openseg && \
PYTHONPATH=. python -u run/evaluate.py \
  --config=config/scannet/ours_openseg_pretrained.yaml \
  feature_type fusion \
  save_folder out/scannet_openseg/result_eval \
  data_root /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_3d \
  data_root_2d_fused_feature /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_multiview_openseg \
  2>&1 | tee out/scannet_openseg/eval.log
```

### Expected output

Results are reported per-class IoU over 5 repeated runs (to account for voxelization randomness):

```
classes          IoU
----------------------------
wall          : 0.510
floor         : 0.691
...
Mean IoU 0.214
Mean Acc 0.292
```

---

## Running Matterport Evaluation (Fusion Mode)

> **Note:** `MinkowskiEngine` is not installed in the `repopt` environment.  
> Only **fusion mode** is supported — it uses pre-computed 2D features directly and requires no 3D network.

### Available data (read-only)

The Matterport data is already available on the cluster — **no download needed**:

| Path | Contents |
| ---- | -------- |
| `/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_3d/` | Matterport3D test point clouds (406 scenes, `.pth`) |
| `/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_multiview_openseg_test/` | Pre-fused 2D OpenSeg features for test scenes (`.pt`) |

### Run command

> **Memory warning:** The default config uses `test_repeats: 5`, which accumulates logit tensors for all 406 scenes × 5 passes before computing IoU. This requires ~100–200 GB RAM and will be OOM-killed on nodes allocated with `--mem 50000`. Use `test_repeats 1` (recommended) or request more memory.

```bash
cd ~/repopt-3d/third_party/openscene && \
mkdir -p out/matterport_openseg && \
PYTHONPATH=. python -u run/evaluate.py \
  --config=config/matterport/ours_openseg_pretrained.yaml \
  feature_type fusion \
  save_folder out/matterport_openseg/result_eval \
  data_root /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_3d \
  data_root_2d_fused_feature /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_multiview_openseg_test \
  test_repeats 1 \
  2>&1 | tee out/matterport_openseg/eval.log
```

| Option | `test_repeats 1` | `test_repeats 5` |
| ------ | ---------------- | ---------------- |
| RAM usage | ~50 GB | ~100–200 GB |
| Runtime | ~10–15 min | ~2–3 hours |
| Accuracy | Single-pass IoU | Averaged over 5 voxelizations (slightly higher) |

> **Runtime note:** The IoU table is printed **after** all 406 scenes are processed, so the log will appear silent for several minutes at the end while tensor aggregation and metric computation happen.

### Expected output

Results are reported per-class IoU over 5 repeated runs (to account for voxelization randomness):

```
classes          IoU
----------------------------
wall          : 0.xxx
floor         : 0.xxx
...
Mean IoU 0.xxx
Mean Acc 0.xxx
```

### Downloading the data yourself (optional)

If you need a personal copy of the data:

**Matterport 3D point clouds (~5.1 GB):**

```bash
mkdir -p /work/pi_adamoneill_umass_edu/$USER/openscene/data
cd /work/pi_adamoneill_umass_edu/$USER/openscene/data
wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_3d.zip
unzip matterport_3d.zip
```

**Matterport test 2D fused features (~66.7 GB)** — submit as a SLURM job (too large for interactive session):

```bash
sbatch --job-name=dl_mp_feat --partition=cpu --mem=8G --time=06:00:00 \
  --output=~/dl_mp_feat_%j.out \
  --wrap="cd /work/pi_adamoneill_umass_edu/$USER/openscene/data && \
          wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_multiview_openseg_test.zip && \
          unzip matterport_multiview_openseg_test.zip"
```

Monitor with: `squeue -u $USER`

---

## Bugs Fixed (already applied to the codebase)

These issues were encountered and patched during setup — no action needed:


| File                    | Bug                                                                 | Fix                                                                       |
| ----------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `models/mink_unet.py`   | `BLOCK = None` — missing `BasicBlock` assignment and ME import      | Added `try/except` ME import; uncommented `BLOCK = BasicBlock/Bottleneck` |
| `models/resnet_base.py` | Same as above                                                       | Same fix                                                                  |
| `run/distill.py`        | `get_model()` tried to instantiate MinkowskiNet even in fusion mode | Returns `nn.Module()` dummy for `feature_type == 'fusion'`                |
| `run/evaluate.py`       | `SparseTensor` created unconditionally even in fusion mode          | Moved inside `distill`/`ensemble` branches with lazy import               |
| `dataset/voxelizer.py`  | `collections.Iterable` removed in Python 3.10+                      | Changed to `collections.abc.Iterable`                                     |


---

## Config Reference


| Config                                           | Dataset      | Classes | Split | Feature extractor |
| ------------------------------------------------ | ------------ | ------- | ----- | ----------------- |
| `config/scannet/ours_openseg_pretrained.yaml`    | ScanNet      | 20      | val   | OpenSeg (768-dim) |
| `config/matterport/ours_openseg_pretrained.yaml` | Matterport3D | 21      | test  | OpenSeg (768-dim) |


---

## Evaluation Modes


| Mode       | Description                           | Requires MinkowskiEngine? |
| ---------- | ------------------------------------- | ------------------------- |
| `fusion`   | Uses pre-fused 2D features directly   | No                        |
| `distill`  | Uses trained 3D UNet (MinkUNet18A)    | **Yes**                   |
| `ensemble` | Combines fusion + distill predictions | **Yes**                   |


