# H2 Plan: Confidence-Masked Entropy Minimization for 3D Semantic Segmentation

## 1. Hypothesis

**H2:** The entropy of OpenScene's lifted features correlates with view-coverage quality. Minimizing entropy on *confident* (low-entropy) unlabeled points, while ignoring uncertain ones, will improve accuracy on low-confidence regions more than on high-confidence regions.

**Success criterion:** `(Run2_Q1 ? Run1_Q1) > (Run2_Q4 ? Run1_Q4)`

i.e., the entropy term helps low-confidence points more than high-confidence ones.

---

## 2. Motivation

OpenScene fuses 2D CLIP features from multiple views into 3D point clouds. Points seen from many views get reliable feature representations; points with sparse coverage do not. We hypothesize this manifests as high-entropy predictions on low-coverage points. A lightweight adapter trained with supervised loss on 5% labeled data + confidence-masked entropy minimization on the rest should selectively improve low-confidence regions.

---

## 3. Prerequisites

- OpenScene codebase with `distill.py` and `util/metric.py`
- Matterport3D dataset with pre-fused features
- 21-class label set
- Python environment with PyTorch

---

## 4. Adapter Architecture

**`FeatureAdapter`** ù a 2-layer residual MLP + linear classifier on top of frozen OpenScene features:

```
fc1 (768?768) ? LayerNorm ? ReLU ? fc2 (768?768) + residual ? classifier (768?21)
```

**Design choices:**
- **LayerNorm** (not BatchNorm) ù point cloud chunks have variable sizes
- **Residual connection** ù at initialization (fc2 weights = 0), adapter is identity, so training starts from the OpenScene baseline without disrupting features
- **~1.2M params** ù tiny; regularization comes from capacity constraint + `L_ent`, not dropout

---

## 5. Loss Functions

### `L_sup` ù Supervised cross-entropy
Standard cross-entropy over the 5% labeled subset within each batch.

### `L_ent` ù Confidence-masked entropy

```python
def confidence_masked_entropy(logits, tau):
    # H = entropy of softmax(logits)
    # conf_mask = (H < tau).detach()  ? NO gradient through mask
    # returns mean(H) over confident points only
```

- `tau = 1.52` ? `log(21) * 0.5` ù points below this entropy threshold are considered "confident"
- Mask is detached: the mask selection doesn't backpropagate, only the entropy values do
- `frac_confident` = fraction of unlabeled points below threshold ù monitor this during training

### Combined loss

```
loss = L_sup + lambda2 * L_ent
```

- **Run 1:** `lambda2 = 0.0` (supervised only, no entropy term)
- **Run 2:** `lambda2 = 0.1` (full H2)

---

## 6. Labeled Split Strategy

**Implemented:** `scripts/prepare_labeled_split.py` builds a **fixed** 5% **class-balanced** labeled set per training scene (`seed=42`, `min 1` point per present class). It writes one NumPy file per scene:

`data/matterport_labeled_indices/train/<scene>.npy` ù indices into the **original** point cloud (same indexing as the fusion `.pt` maps).

`run/train_adapter.py` loads that file per scene and maps labels through voxelization (`inds_reconstruct`) onto visible fused features. No random relabeling happens inside the training loop.

---

## 7. Steps

**Step 1:** Generate labeled indices (required before adapter training)

- `python scripts/prepare_labeled_split.py`
- Writes `data/matterport_labeled_indices/train/*.npy` (override with `--in_dir` / `--out_dir`).

**Step 2 (Run 0):** Zero-shot **fusion** baseline + save per-point confidence for quartile binning

- `python run/evaluate.py` with `eval_fusion_baseline.yaml` and `--save_confidence`
- Writes `data/matterport_baseline_confidence/test/<scene>.npy`

**Step 3 (Run 1):** Train adapter supervised-only (`lambda2=0`)

- `python run/train_adapter.py --config config/matterport/adapter_sup_only.yaml`

**Step 4 (Run 2):** Train adapter with H2 loss (`lambda2=0.1`, `entropy_tau` ~ 1.52)

- `python run/train_adapter.py --config config/matterport/adapter_with_entropy.yaml`
- Monitor `L_ent` and `frac_confident` in logs / TensorBoard.

**Step 5 (optional, Run 3):** Vanilla entropy on all unlabeled points (TENT-style)

- `python run/train_adapter.py --config config/matterport/adapter_tent.yaml`

**Step 6:** Evaluate each run with `run/evaluate_adapter.py` and the same baseline confidence directory for quartile accuracy + mIoU.

---

## 8. Data Flow: Training vs. Evaluation

### Critical difference in `FusedFeatureLoader`

**Training mode (`eval_all=False`):**
- `feat_3d` has **M rows** where M = `mask.sum()` ù already the visible subset
- `label_3d` has **V rows** ù one per voxel
- `mask` has V entries ù boolean, which voxels have features
- To align labels with feat_3d: `labels_visible = label_3d[mask]` ? `[M]`
- Do **NOT** do `feat_3d[mask]` ù that double-indexes and is wrong

**Evaluation mode (`eval_all=True`):**
- `feat_3d` has **V rows** ù all voxels, zeros for non-featured
- `mask` has V entries ù boolean
- To get visible features: `feat_visible = feat_3d[mask]` ? `[M]` ù correct here
- After adapter: map back via `logits_full[mask] = logits_visible`

This asymmetry matches `distill.py:321-322` where model output (V rows) is indexed by mask.

---

## 9. Ablation Runs

| Run | Script / config | Loss | Purpose |
|-----|-----------------|------|---------|
| 0 (baseline) | `run/evaluate.py` + `eval_fusion_baseline.yaml` + `--save_confidence` | none | Fusion baseline; per-point confidence `.npy` |
| 1 (sup only) | `run/train_adapter.py` + `adapter_sup_only.yaml` | `L_sup` | Supervised adapter |
| 2 (H2) | `run/train_adapter.py` + `adapter_with_entropy.yaml` | `L_sup + lambda2 * L_ent` | Masked entropy |
| 3 (optional) | `run/train_adapter.py` + `adapter_tent.yaml` | `L_sup + lambda2 * mean(H)` | TENT (unmasked) |

---

## 10. Code map

### New / primary files

| File | Purpose |
|------|---------|
| `models/adapter.py` | `FeatureAdapter`, `get_adapted_features`, losses |
| `run/train_adapter.py` | Training; `save_path/model/model_best.pth.tar`, `model_final.pth.tar` |
| `run/evaluate_adapter.py` | mIoU + H2 quartiles |
| `scripts/prepare_labeled_split.py` | `data/matterport_labeled_indices/train/*.npy` |
| `config/matterport/eval_fusion_baseline.yaml` | Run 0 (`feature_type: fusion`) |
| `config/matterport/adapter_*.yaml` | Runs 1ù3 |

### Modified files

| File | Change |
|------|--------|
| `run/evaluate.py` | `--save_confidence`, `--confidence_save_dir` |
| `util/metric.py` | `evaluate_by_confidence_bin()` |

---

## 11. How to run

Use directory **`third_party/openscene`** and **`export PYTHONPATH=.`**.

### Outputs (where artifacts go)

| Artifact | Location |
|----------|----------|
| 5% labeled indices | `data/matterport_labeled_indices/train/<scene>.npy` (default from `prepare_labeled_split.py`) |
| Run 0 confidence scores | `data/matterport_baseline_confidence/test/<scene>.npy` (from `evaluate.py --save_confidence`) |
| Training checkpoints | `<ADAPTER.save_path in yaml>/model/model_best.pth.tar`, `model_final.pth.tar`, periodic `checkpoint_epoch_*.pth.tar` |
| TensorBoard | `<ADAPTER.save_path>/` (same as `SummaryWriter`) |
| Evaluation | Prints mIoU + quartiles to stdout; `--save_folder` on `evaluate_adapter` only creates that directory (logs optional) |

### 0) Labeled indices (once)

```bash
python scripts/prepare_labeled_split.py \
  --in_dir /path/to/matterport_3d/train \
  --out_dir data/matterport_labeled_indices/train
```

Point `labeled_indices_dir` in adapter YAMLs at this path.

### 1) Run 0 ù fusion baseline + confidence

```bash
python run/evaluate.py \
  --config config/matterport/eval_fusion_baseline.yaml \
  --save_confidence \
  --confidence_save_dir data/matterport_baseline_confidence/test \
  data_root /your/matterport_3d \
  data_root_2d_fused_feature /your/fused_features
```

### 2) Run 1 ù supervised adapter

```bash
python run/train_adapter.py --config config/matterport/adapter_sup_only.yaml

python run/evaluate_adapter.py \
  --config config/matterport/adapter_sup_only.yaml \
  --model_path "<your_save_path>/model/model_best.pth.tar" \
  --confidence_dir data/matterport_baseline_confidence/test
```

### 3) Run 2 ù H2

```bash
python run/train_adapter.py --config config/matterport/adapter_with_entropy.yaml

python run/evaluate_adapter.py \
  --config config/matterport/adapter_with_entropy.yaml \
  --model_path "<your_save_path>/model/model_best.pth.tar" \
  --confidence_dir data/matterport_baseline_confidence/test
```

(`--confidence_load_dir` is an alias for `--confidence_dir`.)

### 4) Optional Run 3 ù TENT

```bash
python run/train_adapter.py --config config/matterport/adapter_tent.yaml
```

**Tip:** Override any yaml value with trailing `opts`, e.g. `save_path /tmp/exp1`.

---

## 12. Reading the Results

### Expected output format

Each evaluation prints two blocks. Here's how to read them:

```
=== Overall Evaluation ===
classes         IoU
wall          : 0.XXX    (XXXXX/XXXXX)   ? IoU for this class; (correctly_predicted / total_gt_points)
floor         : 0.XXX    (XXXXX/XXXXX)
...
Mean IoU 0.XXXX    ? primary metric; compare this across runs
Mean Acc 0.XXXX    ? per-point accuracy (less informative for class-imbalanced scenes)
```

```
=== H2 Diagnostic (confidence from data/matterport_baseline_confidence/test) ===
Quartile  Accuracy  Points
                             ? quartiles are based on Run 0 (baseline) entropy, fixed across all runs
    Q1      0.XXXX   XXXXX  ? lowest-confidence points (high entropy in baseline) ù H2 predicts most gain here
    Q2      0.XXXX   XXXXX
    Q3      0.XXXX   XXXXX
    Q4      0.XXXX   XXXXX  ? highest-confidence points (low entropy in baseline) ù expect less gain here

Final mIoU: 0.XXXX    ? same as Mean IoU above; repeated for convenience
```

> **Note:** Q1 = worst-covered points (fewest views, highest uncertainty). Q4 = best-covered. The quartile boundaries are computed once from Run 0 and reused for Run 1 and Run 2, so comparisons are apples-to-apples.

### Results table to fill in

```
          mIoU    Q1 acc    Q2 acc    Q3 acc    Q4 acc
Run 0     ___     ___       ___       ___       ___     (zero-shot OpenScene baseline ù no adapter)
Run 1     ___     ___       ___       ___       ___     (adapter, supervised loss only)
Run 2     ___     ___       ___       ___       ___     (adapter, supervised + entropy loss)
```

Fill this in as runs complete. Run 0 numbers become the reference; Run 1 shows what supervision alone buys; Run 2 is the H2 result.

### Success criteria

**H2 is confirmed if:** `(Run2_Q1 ? Run1_Q1) > (Run2_Q4 ? Run1_Q4)`

The entropy loss should selectively help Q1 (where the baseline was uncertain) more than Q4 (where it was already confident). A uniform lift across all quartiles means `L_ent` isn't doing anything targeted ù it's just acting like a generic regularizer.

---

## 13. Interpreting Results

**Strong confirmation:** `L_ent` improves overall mIoU AND the gain is concentrated in Q1.

**Partial confirmation:** `L_ent` improves Q1 accuracy but overall mIoU is flat. The failure mode is real but affects a small fraction of points.

**Negative result:** `L_ent` helps uniformly across quartiles, or doesn't help at all. Frozen lifted features are more robust to view coverage than expected.

---

## 14. Directory layout

```
third_party/openscene/
  models/adapter.py
  run/train_adapter.py
  run/evaluate_adapter.py
  run/evaluate.py                    (Run 0: --save_confidence)
  scripts/prepare_labeled_split.py
  config/matterport/
    eval_fusion_baseline.yaml        (Run 0)
    adapter_sup_only.yaml
    adapter_with_entropy.yaml
    adapter_tent.yaml
    adapter_entropy_tau03.yaml      (sweep)
  util/metric.py
  data/matterport_labeled_indices/train/*.npy   (from prepare_labeled_split)
  data/matterport_baseline_confidence/test/*.npy (Run 0)
  <yaml ADAPTER.save_path>/model/*.pth.tar      (training checkpoints + TB logs)
```

---

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `L_ent ? 0` for many epochs (all points above ?) | No entropy signal early in training | Expected ù the curriculum is automatic. Monitor `frac_confident`. If still 0 after epoch 20, raise ?. |
| Chunk-label misalignment | Labeled mask doesn't match `feat_3d` | Align via `mask` tensor, same pattern as `distill.py:321`: `output_3d = output_3d[mask]` |
| Class imbalance in 5% split | Rare classes get 0 labeled examples | Class-balanced sampling with `min(1, ...)` per class per scene |
| Proxy ? view count | Diagnostic doesn't test the exact causal claim | Frame as "feature reliability" broadly. Optionally validate on 5 scenes by re-running fusion with saved `counter`. |
| Adapter overfits to 5% labeled set | mIoU on test drops below baseline | Trust-region penalty (H3) would fix this, but for H2-only experiments, the small adapter capacity + early stopping should suffice |