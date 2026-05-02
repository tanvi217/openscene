# H2 Full Experiment Plan: Confidence-Masked Entropy Minimization

## 1. Research Question

Can confidence-masked entropy minimization on unlabeled points improve 3D semantic segmentation from frozen lifted features in the low-label regime — and does it specifically help points with unreliable features?

---

## 2. The Problem

OpenScene fuses 2D CLIP features into 3D by averaging per-pixel features across all camera views that see a given point:

```
z_i = (1/n_i) * ?_{v ? views(i)} f_v(i)
```

The quality of `z_i` depends directly on `n_i` (how many views contributed). High `n_i` ? stable, reliable feature near a class prototype. Low `n_i` ? noisy, high-variance estimate that may land anywhere in CLIP space.

When we train a lightweight adapter on top of these frozen features with only 5% labels:

- **Labeled points** get direct supervision ? fine
- **Unlabeled points** get nothing ? their predictions are whatever the adapter happens to produce, which may be wrong for noisy-feature points

A tempting fix is entropy minimization (TENT-style): push all unlabeled predictions toward lower entropy (more confident). But this is dangerous — for noisy-feature points, the adapter may already be confidently wrong, and minimizing entropy reinforces that error.

---

## 3. The Hypothesis (H2)

Points visible from few camera views have higher-variance fused features than points supported by many views. If `n_i` denotes the number of views supporting point `p_i`, the reliability of `z_i` decreases as `n_i` decreases (roughly as `1/?n_i`).

**Implication:** not all unlabeled points should be treated equally during adaptation. Entropy minimization should only be applied to points where the model is already sufficiently confident — these are likely the well-covered points whose features are reliable.

---

## 4. The Regularizer: Confidence-Masked Entropy (`L_ent`)

### Math

For each unlabeled point `i`, compute predictive entropy:

```
H(?_i) = -?_k ?_ik * log(?_ik)
```

Define a confidence mask (no gradient flows through this):

```
m_i = 1[ H(?_i) < ? ]
```

The confidence-masked entropy loss:

```
L_ent = (1/|{i : m_i = 1}|) * ?_{i ? U} m_i * H(?_i)
```

### Why the mask matters

| | Confident point (low H) | Uncertain point (high H) |
|---|---|---|
| Likely feature quality | Reliable (many views) | Unreliable (few views or boundary) |
| What TENT does | Sharpens (good) | Sharpens toward possibly wrong class (bad) |
| What we do | Sharpens (good) | Ignores — zero gradient contribution |

### Self-regulating curriculum

Early in training: most points are uncertain ? `L_ent ? 0` ? pure supervised training. As supervised loss sharpens predictions on common classes ? more points cross the `?` threshold ? `L_ent` activates gradually. No manual scheduling needed.

### Gradient flow

```
L_ent
  ? gradient flows through
H(?_i)  [only for points where m_i = 1]
  ?
softmax(logits)
  ?
h_psi (classifier)
  ?
g_theta (adapter)

m_i = 1[H < ?]  ?  .detach()  ?  NO gradient through mask
```

> **Critical:** `conf_mask = (H < tau).detach()`. If the mask carried gradients, high-entropy points would get negative gradient that drives them toward *higher* entropy to escape the mask — pathological behavior.

### Initial hyperparameters

- `? = log(21) * 0.5 = 1.52 nats` — half of maximum entropy for K=21 classes
- `?2 = 0.1` — weight on `L_ent` in the full objective

---

## 5. Full Objective

```
J(?, ?) = L_sup + ?2 * L_ent
```

- `L_sup = CrossEntropy(logits[labeled], gt[labeled])` — standard CE on 5% labeled points
- `L_ent` — confidence-masked entropy on the remaining 95% unlabeled points
- `?` = adapter parameters (2-layer residual MLP), `?` = classifier head parameters
- The frozen fused features `z_i` are never updated

---

## 6. The Diagnostic: How We Verify H2

### The view-count problem

The fusion script (`scripts/feature_fusion/matterport_openseg.py`) computes a `counter` tensor (view count per point) but discards it after averaging — only the averaged feature is saved to `.pt` files. Re-running fusion requires the TF OpenSeg model and is expensive.

### The proxy

Use the max-softmax score of the baseline fusion model as a reliability proxy:

```
confidence_i = max_k (z_i / ||z_i|| · text_features[k])
```

**Why this works:** a point with many views gets a feature that concentrates near a class prototype ? high max cosine similarity. A point with few views gets a noisy feature that may not align well with any prototype ? low max cosine similarity.

**Caveat:** high max-softmax could also mean a point sits near a strong prototype by luck, regardless of view count. The proxy captures "feature reliability" broadly, not "view count" exactly. If time permits, validate by re-running fusion on 5 scenes saving `counter`, then compute Spearman rank correlation between actual `n_i` and max-softmax.

### Diagnostic procedure

1. Run baseline fusion (Run 0) ? save `confidence_i` per point per test scene
2. Bin all test points into quartiles by `confidence_i`:
   - **Q1** = least confident (suspected low view count / unreliable features)
   - **Q4** = most confident (suspected high view count / reliable features)
3. For each experimental run, report accuracy within each quartile

### Success criteria

**H2 is confirmed if:** accuracy gain from Run 1 ? Run 2 is larger in Q1 than in Q4. (`L_ent` helps most on unreliable points — exactly what H2 predicts.)

---

## 7. Adapter Architecture

```
g_theta (adapter body):
  Linear(768, 768) ? LayerNorm(768) ? ReLU ? Linear(768, 768) + residual skip

h_psi (classifier head):
  Linear(768, 21)
```

**Design choices:**

- **LayerNorm** (not BatchNorm) — point cloud chunks have variable sizes
- **Residual connection** — at initialization (fc2 weights = 0), adapter is identity, so training starts from the OpenScene baseline without disrupting features
- **Separate `g_theta` and `h_psi`** — allows ablation: freeze one and train the other
- **~1.2M params** — tiny; regularization comes from capacity constraint + `L_ent`, not dropout

---

## 8. Ablation Runs

All runs use the same 5% labeled split (fixed seed), same optimizer, same data pipeline. The only variable is the loss function.

| Run | Loss | Purpose |
|---|---|---|
| 0 (baseline) | none | Zero-shot OpenScene fusion. Establishes mIoU floor. Saves per-point confidence scores for the H2 diagnostic in all subsequent runs. |
| 1 (sup only) | `L_sup` | Supervised adapter alone improves over zero-shot. Expected: uniform improvement across quartiles (no incentive to treat Q1 differently). |
| 2 (H2) | `L_sup + ?2·L_ent` | Full confidence-masked entropy. Tests H2: does the gain concentrate on Q1? |
| 3 (TENT, optional) | `L_sup + ?2·L_ent_unmasked` | Vanilla entropy minimization (`m_i = 1` for all `i`). Tests whether the confidence mask actually matters. Expected: Run 2 ? Run 3 because TENT reinforces errors on unreliable points. |

### Hyperparameter sweep (small grid, train split only)

| Hyperparameter | Values to try |
|---|---|
| `?2` | `{0.01, 0.1, 0.5}` |
| `?` | `{log(21)*0.3, log(21)*0.5, log(21)*0.7}` ? `{0.91, 1.52, 2.13}` |
| `lr` | `{1e-4, 5e-4}` |

Pick best `(?2, ?, lr)` on train-split mIoU. Report final numbers on test split.

---

## 9. Training Details

**Data:** Matterport3D, 21-class semantic segmentation

- Train: 60 scenes, test: 17 scenes
- Features: 768-dim OpenSeg/CLIP, pre-fused, stored in `.pt` files
- Existing `FusedFeatureLoader` chunks scenes into ~20K points

**5% labeled split:**

- Per scene: find all points with `label != 255`, sample 5% class-balanced (min 1 per class)
- Save to `data/matterport_labeled_indices/train/<scene>.npy`
- Fixed `seed=42` for reproducibility

**Training loop (per epoch):**

1. Load scene chunk via `FusedFeatureLoader` ? `(coords, feats, labels, feat_3d, mask)`
2. Load labeled indices for this scene, intersect with chunk's `mask`
3. Forward: `logits = adapter(feat_3d.float())` — `feat_3d` stays frozen
4. Compute `L_sup` on labeled+visible points
5. Compute `L_ent` on unlabeled+visible points (with confidence mask)
6. Backward on `L_sup + ?2 * L_ent`
7. Log: `L_sup`, `L_ent`, `frac_confident` (fraction of unlabeled points with `H < ?`)

**Optimizer:** Adam, `lr=1e-4`, `weight_decay=1e-4`, cosine annealing, 50 epochs

**Checkpointing:** same convention as `distill.py` ? `<exp_dir>/model/model_best.pth.tar`

---

## 10. Evaluation

**Primary metric:** mIoU (21 classes), computed via existing `util/metric.py`

**H2 diagnostic:** per-quartile accuracy using baseline confidence scores from Run 0

**Expected results table:**

```
         mIoU    Q1 acc  Q2 acc  Q3 acc  Q4 acc
Run 0    X.X     a1      a2      a3      a4      (baseline)
Run 1    X.X+?   b1      b2      b3      b4      (sup only)
Run 2    X.X+?'  c1      c2      c3      c4      (H2 method)

H2 confirmed if: (c1-b1) > (c4-b4)
```

---

## 11. Interpreting Results

**Strong confirmation:** `L_ent` improves overall mIoU AND the gain is concentrated in Q1. The method corrects the right kind of error for the right reason.

**Partial confirmation:** `L_ent` improves Q1 accuracy but overall mIoU is flat. Still useful — the failure mode is real but affects a small fraction of points. Contribution: diagnostic analysis rather than a big mIoU number.

**Negative result (still publishable):** `L_ent` helps uniformly across quartiles, or doesn't help at all. This means frozen lifted features are more robust to view coverage than expected. The confidence mask isn't capturing the right signal. Contribution: evidence that the obvious regularizer doesn't work, saving future researchers the effort.

**TENT wins:** if Run 3 (unmasked entropy) beats Run 2 (masked), the confidence mask is actively harmful — perhaps because excluding high-entropy points also excludes genuinely ambiguous boundary points that benefit from entropy pressure. This would be a surprising and interesting finding.

---

## 12. Code Changes

### New Files

#### `models/adapter.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAdapter(nn.Module):
    """Lightweight adapter: 2-layer residual MLP + linear classifier."""
    def __init__(self, feat_dim=768, num_classes=21):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        out = F.relu(self.norm(self.fc1(x)))
        out = self.fc2(out) + residual
        logits = self.classifier(out)
        return logits

def confidence_masked_entropy(logits, tau):
    probs = F.softmax(logits, dim=-1)
    H = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
    conf_mask = (H < tau).detach()  # CRITICAL: no grad through mask
    if conf_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return (conf_mask.float() * H).sum() / conf_mask.float().sum()
```

#### `scripts/prepare_labeled_split.py`

```python
import os, numpy as np, torch
from glob import glob

SEED = 42
LABEL_FRAC = 0.05
IN_DIR = "data/matterport_3d/train"
OUT_DIR = "data/matterport_labeled_indices/train"
IGNORE_LABEL = 255

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

for path in sorted(glob(f"{IN_DIR}/*.pth")):
    coords, colors, labels = torch.load(path)
    labels = np.array(labels)
    valid_idx = np.where(labels != IGNORE_LABEL)[0]

    labeled = []
    for cls in np.unique(labels[valid_idx]):
        cls_idx = valid_idx[labels[valid_idx] == cls]
        k = max(1, int(len(cls_idx) * LABEL_FRAC))
        labeled.extend(rng.choice(cls_idx, size=k, replace=False).tolist())

    scene_name = os.path.splitext(os.path.basename(path))[0]
    np.save(f"{OUT_DIR}/{scene_name}.npy", np.array(sorted(set(labeled))))
```

#### `run/train_adapter.py`

Training script following `run/distill.py` patterns (config loading, logging, checkpoints). Key training loop:

```python
from dataset.feature_loader import FusedFeatureLoader
from models.adapter import FeatureAdapter, confidence_masked_entropy

# Data loading — reuse FusedFeatureLoader unchanged
train_data = FusedFeatureLoader(
    datapath_prefix=args.data_root,
    datapath_prefix_feat=args.data_root_2d_fused_feature,
    split='train', ...
)

# Per-batch:
labeled_indices = np.load(f"data/matterport_labeled_indices/train/{scene_name}.npy")
point_ids = inds_reconstruct
is_labeled = np.isin(point_ids, labeled_indices)
is_labeled_visible = is_labeled[mask]  # align with feat_3d shape

logits = adapter(feat_3d.float())  # feat_3d stays frozen

L_sup = F.cross_entropy(logits[is_labeled_visible],
                        labels_visible[is_labeled_visible])

logits_u = logits[~is_labeled_visible]
L_ent = confidence_masked_entropy(logits_u, tau=args.entropy_tau)

loss = L_sup + args.lambda2 * L_ent

# Optimizer
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

#### `run/evaluate_adapter.py`

Evaluation with the H2 diagnostic:

```python
# Load adapter, run inference with FusedFeatureLoader (eval_all=True)
# Replace CLIP nearest-neighbor with: logits = adapter(feat_3d); preds = logits.argmax(dim=1)

# H2 diagnostic
conf_scores = np.load(f"data/matterport_baseline_confidence/test/{scene_name}.npy")
quartile_edges = np.quantile(conf_scores, [0.25, 0.5, 0.75])
bins = np.digitize(conf_scores, quartile_edges)
for q in range(4):
    mask_q = (bins == q)
    acc_q = (preds[mask_q] == gt[mask_q]).mean()
    print(f"Q{q+1} accuracy: {acc_q:.4f}  (n={mask_q.sum()})")
```

**Config files:**

- `config/matterport/adapter_sup_only.yaml` — `lambda2: 0.0, entropy_tau: null`
- `config/matterport/adapter_with_entropy.yaml` — `lambda2: 0.1, entropy_tau: 1.52`

### Modified Files

| File | Change |
|---|---|
| `run/evaluate.py` | Add `--save_confidence` flag + code to save `pred_logit.max(dim=1)` per point after line ~296 |
| `util/metric.py` | Add standalone `evaluate_by_confidence_bin()` function (existing `evaluate()` untouched) |

### Files Left Unchanged

| File | Why |
|---|---|
| `dataset/feature_loader.py` | Reused as-is |
| `dataset/point_loader.py` | No changes needed |
| `scripts/feature_fusion/*` | View counts not needed at train time |
| `run/distill.py` | Reference only |

---

## 13. Execution Order

```
Step 1:  python scripts/prepare_labeled_split.py
           ? generates data/matterport_labeled_indices/train/*.npy

Step 2:  Run 0 — baseline eval with --save_confidence
           ? saves data/matterport_baseline_confidence/test/*.npy
           ? records baseline mIoU

Step 3:  Implement + unit-test models/adapter.py
           ? verify output shape, gradient flow, conf_mask has no grad

Step 4:  Run 1 — train with lambda2=0
           ? verify mIoU > baseline (adapter is learning)

Step 5:  Run 2 — train with lambda2=0.1, tau=1.52
           ? monitor L_ent and frac_confident per epoch

Step 6:  (Optional) Run 3 — vanilla TENT (conf_mask = all ones)

Step 7:  Run H2 diagnostic across all runs
           ? per-quartile accuracy table
           ? compare Q1 vs Q4 gain from Run 1 ? Run 2
```

---

## 14. Directory Layout After All Changes

```
openscene/
  models/
    adapter.py                              NEW
  run/
    train_adapter.py                        NEW
    evaluate_adapter.py                     NEW
    evaluate.py                             MODIFIED (+save_confidence flag)
  scripts/
    prepare_labeled_split.py                NEW
  config/matterport/
    adapter_sup_only.yaml                   NEW
    adapter_with_entropy.yaml               NEW
  util/
    metric.py                               MODIFIED (+evaluate_by_confidence_bin)
  data/
    matterport_labeled_indices/train/*.npy  GENERATED (Step 1)
    matterport_baseline_confidence/test/*.npy  GENERATED (Step 2)
```

---

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `L_ent ? 0` for many epochs (all points above `?`) | No entropy signal early in training | Expected — the curriculum is automatic. Monitor `frac_confident`. If still 0 after epoch 20, raise `?`. |
| Chunk-label misalignment | Labeled mask doesn't match `feat_3d` | Align via `mask` tensor, same pattern as `distill.py:321`: `output_3d = output_3d[mask]` |
| Class imbalance in 5% split | Rare classes get 0 labeled examples | Class-balanced sampling with `min(1, ...)` per class per scene |
| Proxy ? view count | Diagnostic doesn't test the exact causal claim | Frame as "feature reliability" broadly. Optionally validate on 5 scenes by re-running fusion with saved `counter`. |
| Adapter overfits to 5% labeled set | mIoU on test drops below baseline | Trust-region penalty (H3) would fix this, but for H2-only experiments, the small adapter capacity + early stopping should suffice. |
