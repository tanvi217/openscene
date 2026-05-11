# Adapter experiments (Matterport3D, low-label regime)

This document explains how to run the **supervised-only** adapter, the **entropy-regularized** adapter (H2), and the **spatial-smoothing** variant, what each step does, and how the experiment is set up.

---

## What problem this solves

OpenScene builds a 768-dim fused 2D feature per 3D point. Classification is usually done by matching those features to CLIP text prototypes. Here we train a small **residual MLP adapter + linear head** on top of **frozen** fused features so the model can fit the Matterport 21-class label space with **very few supervised points** (5% of valid points per scene, class-balanced).

**Main configs**

| Config | Loss | Purpose |
|--------|------|--------|
| `adapter_sup_only.yaml` | `L_sup` only | Baseline: what supervision alone achieves |
| `adapter_with_entropy.yaml` | `L_sup + λ₂ · L_ent` | H2: **confidence-masked** predictive-entropy on unlabeled points (only `H(pred) < τ`) |
| `adapter_spatial_smooth.yaml` | `L_sup + λ₂ · L_ent + λ_sp · L_spatial` | Same as H2 plus **voxel-space label smoothness**: penalize squared difference of **softmax** predictions on **6-neighbor** edges (×/y/z in the voxel grid), encouraging piecewise-constant predictions over geometry |

Unmasked entropy on all unlabeled points (TENT-style) can hurt noisy low-coverage points; the H2 mask is meant to avoid that. **Spatial loss** targets **local consistency** of the adapter output (not the frozen features); by default edges are restricted to **unlabeled–unlabeled** pairs so labeled pixels are not pulled toward neighbors.

---

## Experimental setup (summary)

- **Dataset:** Matterport3D, **21** semantic classes (same label set as OpenScene Matterport config).
- **Features:** Pre-fused OpenSeg features (768-d), read from disk; **not** recomputed during training.
- **Train split:** `data_root/train/*.pth` — 3D points + labels; fused features match by scene name under `data_root_2d_fused_feature`.
- **Test split:** `data_root/test/*.pth` — full test evaluation (406 scenes in the public OpenScene setup).
- **Low-label protocol:** For each training scene, a **fixed** set of point indices: **5%** of valid (`label ≠ ignore`) points, **class-balanced**, minimum **1** point per present class, **seed 42**. Stored as one `.npy` per scene (see below).
- **Optimizer:** Adam, `lr=1e-4`, `weight_decay=1e-4`, cosine LR schedule over **50** epochs.
- **Checkpoint:** Best training loss → `model_best.pth.tar` under `<save_path>/model/`.

**Entropy branch (H2 and spatial configs):**

- `lambda2`: weight on `L_ent` (default `0.1`).
- `entropy_tau`: predictive-entropy threshold in nats (default `1.52 ≈ 0.5 × log(21)`). Only unlabeled points with `H(pred) < τ` contribute to `L_ent`.
- `entropy_warmup_epochs`: ramp `λ₂` from 0 over the first N epochs (default `5`) so supervision stabilizes first.

**Spatial smoothing (`adapter_spatial_smooth.yaml`):**

- `lambda_spatial` (`λ_sp`): weight on **`L_spatial`** (mean squared difference of softmax probabilities across 6-connected voxel neighbors among **visible** voxels).
- `spatial_smooth_unlabeled_only` (default `True`): drop edges touching labeled voxels so `L_sup` stays the anchor for annotated points.
- `spatial_max_edges`: cap edges per chunk (`0` = no cap).
- `spatial_warmup_epochs`: same idea as entropy warm-up—increase λ_sp gradually (default `5`).

**Evaluation extras:** Optional **H2 diagnostic** buckets test points by quartiles of **baseline fusion confidence** (from a one-time Run-0 pass). That checks whether accuracy improvements concentrate in high- vs low-confidence regions.

---

## Results summary

Reported metric: **Mean IoU (21 Matterport classes)** on the Matterport **test** split. **Run 0:** zero-shot fusion (`run/evaluate.py` with `eval_fusion_baseline.yaml`). **Runs 1–3:** adapter checkpoints via `run/evaluate_adapter.py`, trained with **5% class-balanced labeled points per train scene**, frozen fused OpenSeg features.

| Run | Config | Objective | **mIoU** |
|-----|--------|-------------|-----------|
| 0 *(fusion baseline)* | `eval_fusion_baseline.yaml` | Zero-shot fusion (no adapter) | **40.04%** |
| 1 | `adapter_sup_only.yaml` | `L_sup` only | **48.79%** |
| 2 | `adapter_with_entropy.yaml` (`τ=1.52`, `λ₂=0.1`, warm-up 5 ep) | `L_sup + λ₂ L_ent` (masked entropy) | **48.26%** |
| 3 | `adapter_spatial_smooth.yaml` (same H2 + `λ_sp=0.02`, warm-up 5 ep) | `L_sup + λ₂ L_ent + λ_sp L_spatial` | **48.89%** |

**Takeaway:** Fusion-only baseline **40.04%** mIoU; training the adapter with **5% labels** raises it to **48.79%**. Under the original H2 entropy settings (`entropy_tau ≈ ½ × log(21)`), adding masked entropy alone **slightly hurt** mIoU (−0.53 pp vs supervised-only). Adding **spatial softmax smoothness** on top of H2 recovers and **slightly exceeds** supervised-only (**+0.10 pp**). Diagnosis for the entropy-only run: **`τ` was too lenient**, so nearly all unlabeled points passed the confidence mask (`frac_confident` grew from ~0.72 toward ~**0.98**), so `L_ent` behaved like near **unmasked** entropy across most points—not the intended selective regularization.

**Next steps recorded in docs:** Sweep **stricter τ** (e.g. `adapter_entropy_tau03.yaml`, `tau05.yaml`, `tau07.yaml`) and stabilize the labeled mask (fixed per-scene indices + balanced sampling)—see `.claude/h2-plan.md` / `H2_PLAN.md`.

**Artifacts:** Primary numbers come from **`run/evaluate_adapter.py`** stdout and files under `$save_folder`; TensorBoard curves under `$save_path` show `loss/total`, `L_sup`, `L_ent`, `frac_confident`, and **`L_spatial`** when spatial smoothing is enabled.

---

## Prerequisites

### 1) Download Matterport3D (OpenScene release)

Skip this if you already use **read-only shared data** on the cluster (see `RUNNING_ON_UNITY.md`). Otherwise download from the OpenScene CVG server, unzip into a writable directory, and point `data_root` / `data_root_2d_fused_feature` at the extracted folders (expected layout: `.../matterport_3d/train/`, `.../matterport_3d/test/`, plus fused `.pt` files).

| What | URL |
|------|-----|
| **3D point clouds + labels** (train/test `.pth`) | https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_3d.zip |
| **Test fused OpenSeg features** (large, ~66.7 GB) | https://cvg-data.inf.ethz.ch/openscene/data/matterport_multiview_openseg_test.zip |

Upstream helpers (optional): `bash scripts/download_dataset.sh` (Matterport 3D) and `bash scripts/download_fused_features.sh` (includes test OpenSeg features). Full OpenScene data notes: `README.md` in this repo.

### 2) Layout

`data_root` must look like:

```text
data_root/
  train/*.pth   # training scenes
  test/*.pth    # evaluation scenes
```

`data_root_2d_fused_feature` must contain the fused OpenSeg `.pt` files for **both** splits, keyed consistently with OpenScene (same layout as upstream).

### 3) Labeled split files (required for training)

Generate once from the **train** `.pth` files:

```bash
cd ~/repopt-3d/third_party/openscene
PYTHONPATH=. python scripts/prepare_labeled_split.py \
  --in_dir data/matterport_3d/train \
  --out_dir data/matterport_labeled_indices/train \
  --label_frac 0.05 \
  --seed 42
```

**Why:** Keeps the 5% supervision **reproducible** and **stable across epochs** (same indices per scene), instead of resampling every step.

### 4) Baseline confidence for H2 diagnostic (optional but recommended)

One fusion forward pass saves per-point confidence for test scenes (used at eval for quartile analysis, not for training loss):

```bash
PYTHONPATH=. python run/evaluate.py --config config/matterport/eval_fusion_baseline.yaml \
  --save_confidence --confidence_save_dir data/matterport_baseline_confidence/test
```

Adjust `DATA` paths in `eval_fusion_baseline.yaml` or override on the CLI if your data lives elsewhere.

**Why:** The diagnostic compares adapter accuracy vs **baseline max-softmax** per point, binned by quartile.

---

## Configure paths

Default YAMLs use **repository-relative** paths (`data/...`, `exp/...`). On the cluster, override without editing files:

```bash
PYTHONPATH=. python run/train_adapter.py --config config/matterport/adapter_sup_only.yaml \
  data_root /path/to/matterport_3d \
  data_root_2d_fused_feature /path/to/matterport_multiview_openseg_test \
  save_path /path/to/experiments/adapter_sup_only \
  labeled_indices_dir /path/to/matterport_labeled_indices/train
```

Use the same pattern for `labeled_indices_dir`, `confidence_dir`, and `save_folder` under `evaluate_adapter.py`.

---

## Train

From `openscene/`:

**Supervised only**

```bash
PYTHONPATH=. python run/train_adapter.py --config config/matterport/adapter_sup_only.yaml
```

**Supervised + confidence-masked entropy (H2)**

```bash
PYTHONPATH=. python run/train_adapter.py --config config/matterport/adapter_with_entropy.yaml
```

**H2 + spatial softmax smoothness**

```bash
PYTHONPATH=. python run/train_adapter.py --config config/matterport/adapter_spatial_smooth.yaml
```

TensorBoard logs are written under `<save_path>` (`loss/total`, `L_sup`, `L_ent`, `frac_confident`, and **`L_spatial`** when `lambda_spatial > 0`).

---

## Evaluate

After training, evaluate on the **test** split and optionally run the H2 diagnostic:

```bash
PYTHONPATH=. python run/evaluate_adapter.py \
  --config config/matterport/adapter_sup_only.yaml \
  --model_path exp/adapter_sup_only/model/model_best.pth.tar \
  --save_folder exp/adapter_sup_only/eval
```

Example for spatial smoothing (same pattern for `adapter_with_entropy.yaml`):

```bash
PYTHONPATH=. python run/evaluate_adapter.py \
  --config config/matterport/adapter_spatial_smooth.yaml \
  --model_path exp/adapter_spatial_smooth/model/model_best.pth.tar \
  --save_folder exp/adapter_spatial_smooth/eval
```

Use `adapter_with_entropy.yaml` / `adapter_spatial_smooth.yaml` and matching `--model_path` / `--save_folder` when evaluating those runs (`exp/adapter_with_entropy/...`, `exp/adapter_spatial_smooth/...`). If baseline confidence files are not available:

```bash
PYTHONPATH=. python run/evaluate_adapter.py \
  --config config/matterport/adapter_sup_only.yaml \
  --model_path exp/adapter_sup_only/model/model_best.pth.tar \
  --no_h2
```

The YAML `TEST` section already sets `split: test`, `confidence_dir`, and `do_h2_diagnostic`; CLI flags override when needed.

---

## Ablation configs

Other loss variants live under `config/matterport/adapter_*.yaml` (e.g. `adapter_entropy_tau03.yaml`, `adapter_tent.yaml`, **`adapter_spatial_smooth.yaml`**). They use the same data and training script; only `ADAPTER` hyperparameters differ.

---

## Troubleshooting

- **`0 file is loaded in the point loader`:** `data_root` does not contain a `train/` or `test/` folder with `.pth` files at the expected level, or paths are wrong.
- **OOM on GPU:** Training uses batch size 1; reduce `workers` or scene size is handled by chunked loading—if issues persist, use a GPU with enough VRAM for the voxelized chunk.
- **H2 diagnostic skipped:** Missing `test/<scene>.npy` under `confidence_dir` for some scenes, or `--no_h2` set.

---

## File reference

| Item | Role |
|------|------|
| `run/train_adapter.py` | Training loop, `L_sup` / `L_ent` / optional spatial loss |
| `run/evaluate_adapter.py` | Test mIoU + optional H2 quartile diagnostic |
| `models/adapter.py` | `FeatureAdapter`, entropy helpers |
| `scripts/prepare_labeled_split.py` | Builds per-scene labeled index `.npy` files |
| `config/matterport/eval_fusion_baseline.yaml` | Run-0 fusion + `--save_confidence` |
