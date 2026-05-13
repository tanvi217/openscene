# Running OpenScene on the Unity Cluster

## Overview

[OpenScene](https://pengsongyou.github.io/openscene) is an open-vocabulary 3D scene understanding model. It lifts 2D vision-language features (from OpenSeg / CLIP) onto 3D point clouds and classifies points by comparing their features against CLIP text embeddings — no fixed label set required.

This guide covers:

1. **Fusion-mode evaluation** with `run/evaluate.py` (no MinkowskiEngine, no 3D UNet).
2. **Optional:** Matterport adapter / H2 pipeline via `scripts/run_h2_pipeline.sh` and `scripts/h2_pipeline_env_unity.sh` (see the repo `README.md` in `third_party/openscene` for experiment details).

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

### 3. Go to the OpenScene tree inside this repo

From your Unity home (same layout as a typical clone):

```bash
cd "${HOME}/repopt-3d/third_party/openscene"
```

If your checkout lives elsewhere, `cd` to that path instead. Most commands below assume this directory is the current working directory and use `PYTHONPATH=.` (or `export PYTHONPATH="$(pwd)"`).

> **Note:** `MinkowskiEngine` is not installed in the `repopt` environment. Only **`feature_type fusion`** (pre-fused 2D features on disk) is supported unless you install MinkowskiEngine yourself.

---

## Shared read-only data (project storage)

These paths are on **`/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/`** and are readable by the group. Use them for ScanNet and for **Matterport fused OpenSeg features** (large `.pt` trees).

### ScanNet (fusion eval — small val subset)

| Path | Contents |
| ---- | -------- |
| `/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_3d/val/` | 3 ScanNet val scenes (`.pth`) |
| `/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_multiview_openseg/` | Pre-fused 2D OpenSeg features for those scenes (`.pt`) |

### Matterport — fused test features only (shared)

| Path | Contents |
| ---- | -------- |
| `/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_multiview_openseg_test/` | Pre-fused OpenSeg features for Matterport **test** scenes (`.pt`) |

Full Matterport **3D** `.pth` train/test trees used by this fork’s adapter configs are kept under **your lab work directory** (next section), not only under the path above — use whichever copy you have access to.

---

## Per-user Matterport 3D layout (`/work/pi_adamoneill_umass_edu/…`)

Adapter configs and `scripts/h2_pipeline_env_unity.sh` expect **Matterport 3D** here (replace `YOUR_UNITY_USERNAME` with your Unity login, e.g. `tanviagarwal_umass_edu`):

| Path | Contents |
| ---- | -------- |
| `/work/pi_adamoneill_umass_edu/YOUR_UNITY_USERNAME/openscene/matterport_3d/` | `train/*.pth`, `test/*.pth` (406 test scenes in the public OpenScene layout) |

**Fused features** for eval/train still come from the shared directory:

`/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_multiview_openseg_test`

If you still have the older **all-in-one** copy under Soorya’s tree, you can instead set `data_root` to:

`/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_3d`

---

## Running ScanNet evaluation (fusion mode)

### Command

```bash
cd "${HOME}/repopt-3d/third_party/openscene"
mkdir -p out/scannet_openseg
PYTHONPATH=. python -u run/evaluate.py \
  --config=config/scannet/ours_openseg_pretrained.yaml \
  feature_type fusion \
  save_folder out/scannet_openseg/result_eval \
  data_root /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_3d \
  data_root_2d_fused_feature /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/scannet_multiview_openseg \
  2>&1 | tee out/scannet_openseg/eval.log
```

### Expected output

Per-class IoU; the upstream config may use multiple voxelization repeats — the log header reflects the effective `test_repeats`.

---

## Running Matterport evaluation (fusion mode)

> **Memory:** With `test_repeats: 5`, logits for all scenes accumulate before IoU and can require **~100–200 GB** RAM. For a **50 GB** allocation, pass **`test_repeats 1`** (recommended below).

### Command (paths aligned with adapter / Unity scripts)

```bash
cd "${HOME}/repopt-3d/third_party/openscene"
mkdir -p out/matterport_openseg
PYTHONPATH=. python -u run/evaluate.py \
  --config=config/matterport/ours_openseg_pretrained.yaml \
  feature_type fusion \
  save_folder out/matterport_openseg/result_eval \
  data_root "/work/pi_adamoneill_umass_edu/${USER}/openscene/matterport_3d" \
  data_root_2d_fused_feature /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_multiview_openseg_test \
  test_repeats 1 \
  2>&1 | tee out/matterport_openseg/eval.log
```

If your Matterport 3D data only exists under the shared Soorya tree, swap `data_root` for:

`/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_3d`

| Option | `test_repeats 1` | `test_repeats 5` |
| ------ | ---------------- | ---------------- |
| RAM usage | ~50 GB | ~100–200 GB |
| Runtime | ~10–15 min | ~2–3 hours |
| Metrics | Single-pass IoU | Averaged over 5 voxelizations (slightly higher variance reduction) |

> **Runtime:** The IoU table is printed **after** all test scenes are processed; the log can look idle for several minutes while metrics are aggregated.

### Fusion baseline + confidence export (H2 / adapter prep)

Uses `config/matterport/eval_fusion_baseline.yaml` (`test_repeats: 1` by default). Override `DATA` paths on the CLI if needed:

```bash
cd "${HOME}/repopt-3d/third_party/openscene"
PYTHONPATH=. python run/evaluate.py \
  --config=config/matterport/eval_fusion_baseline.yaml \
  --save_confidence \
  --confidence_save_dir "/work/pi_adamoneill_umass_edu/${USER}/openscene/matterport_baseline_confidence/test" \
  data_root "/work/pi_adamoneill_umass_edu/${USER}/openscene/matterport_3d" \
  data_root_2d_fused_feature /work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_multiview_openseg_test
```

Put **`--save_confidence` / `--confidence_save_dir` before** any `data_root …` overrides: remaining tokens are parsed as config key/value pairs (`argparse.REMAINDER`).

---

## Adapter / H2 full pipeline (optional)

End-to-end: labeled split → fusion baseline confidences → supervised-only → entropy (H2) trains/evals, with paths driven by env vars.

1. Edit `scripts/h2_pipeline_env_unity.sh` so `MP_ROOT`, `LABELED_OUT`, `CONF_OUT`, and `SAVE_*` use **`YOUR_UNITY_USERNAME`** (defaults in the repo point at `tanviagarwal_umass_edu`). `FUSED_ROOT` should stay on the shared Soorya fused-feature path unless you mirror it yourself.

2. Run:

```bash
cd "${HOME}/repopt-3d/third_party/openscene"
module load conda/latest && conda activate repopt
source scripts/h2_pipeline_env_unity.sh
bash scripts/run_h2_pipeline.sh 2>&1 | tee "${HOME}/h2_pipeline.log"
```

See `scripts/run_h2_pipeline.sh` for skip flags (`SKIP_PREPARE`, `SKIP_RUN0`, …) and optional `RUN_TENT`, `RUN_SPATIAL`, etc.

---

## Downloading data yourself (optional)

Put archives under **`/work/pi_adamoneill_umass_edu/$USER/openscene/`** (not `.../openscene/data` alone — unzip layouts expect a `matterport_3d/` directory next to your other experiment folders).

**Matterport 3D point clouds (~5.1 GB):**

```bash
mkdir -p "/work/pi_adamoneill_umass_edu/${USER}/openscene"
cd "/work/pi_adamoneill_umass_edu/${USER}/openscene"
wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_3d.zip
unzip matterport_3d.zip
```

**Matterport test fused OpenSeg features (~66.7 GB)** — submit as a SLURM job (too large for many interactive nodes):

```bash
sbatch --job-name=dl_mp_feat --partition=cpu --mem=8G --time=06:00:00 \
  --output="${HOME}/dl_mp_feat_%j.out" \
  --wrap="set -e; mkdir -p '/work/pi_adamoneill_umass_edu/${USER}/openscene'; \
          cd '/work/pi_adamoneill_umass_edu/${USER}/openscene'; \
          wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_multiview_openseg_test.zip; \
          unzip matterport_multiview_openseg_test.zip"
```

If you unpack fused features locally, point `data_root_2d_fused_feature` at that directory instead of the shared Soorya path.

Monitor: `squeue -u "$USER"` — cancel: `scancel <JOBID>`.

Upstream helpers in this tree: `bash scripts/download_dataset.sh`, `bash scripts/download_fused_features.sh`.

---

## Bugs fixed in this fork (historical)

Fusion-only code paths were adjusted so evaluation does not require MinkowskiEngine (lazy imports, dummy 3D modules in fusion mode, Python 3.10+ `collections.abc` fixes). If you merge upstream OpenScene, re-check `run/evaluate.py`, `run/distill.py`, and `models/mink_unet.py` / `models/resnet_base.py`.

---

## Config reference

| Config | Dataset | Classes | Split | Feature extractor |
| ------ | ------- | ------- | ----- | ----------------- |
| `config/scannet/ours_openseg_pretrained.yaml` | ScanNet | 20 | val | OpenSeg (768-d) |
| `config/matterport/ours_openseg_pretrained.yaml` | Matterport3D | 21 | test | OpenSeg (768-d) |
| `config/matterport/eval_fusion_baseline.yaml` | Matterport3D | 21 | test | OpenSeg; `test_repeats: 1`; optional `--save_confidence` |
| `config/matterport/adapter_*.yaml` | Matterport3D | 21 | train/test | Adapter training/eval; see root `README.md` |

---

## Evaluation modes

| Mode | Description | Requires MinkowskiEngine? |
| ---- | ----------- | ------------------------- |
| `fusion` | Pre-fused 2D features on disk | No |
| `distill` | Trained 3D UNet (MinkUNet18A) | **Yes** |
| `ensemble` | Fusion + distill combined | **Yes** |
