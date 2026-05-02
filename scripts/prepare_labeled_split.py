"""Create a 5% class-balanced labeled split for adapter training.

For each training scene, randomly samples 5% of valid points (label != 255)
using class-balanced sampling (min 1 per class per scene).  Saves the selected
original-point indices as a numpy array:

  <out_dir>/<scene_name>.npy

These index files are consumed by run/train_adapter.py at training time.

Usage (from the openscene/ root):
  python scripts/prepare_labeled_split.py
  python scripts/prepare_labeled_split.py --out_dir /path/to/writable/dir
"""
import os
import argparse
import numpy as np
import torch
from glob import glob

SEED         = 42
LABEL_FRAC   = 0.05
IGNORE_LABEL = 255
DEFAULT_IN_DIR  = "data/matterport_3d/train"
# Plan §14: outputs under openscene/data/matterport_labeled_indices/train
DEFAULT_OUT_DIR = "data/matterport_labeled_indices/train"


def get_args():
    parser = argparse.ArgumentParser(description='Prepare 5% labeled split')
    parser.add_argument('--in_dir',  type=str, default=DEFAULT_IN_DIR,
                        help='directory containing training .pth scene files')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR,
                        help='writable directory to write per-scene .npy index files')
    parser.add_argument('--label_frac', type=float, default=LABEL_FRAC,
                        help='fraction of valid points to label per class (default 0.05)')
    parser.add_argument('--seed', type=int, default=SEED)
    return parser.parse_args()


def main():
    args = get_args()
    IN_DIR  = args.in_dir
    OUT_DIR = args.out_dir

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    paths = sorted(glob(os.path.join(IN_DIR, "*.pth")))
    if not paths:
        raise FileNotFoundError(
            f"No .pth files found in {IN_DIR!r}. "
            "Check that --in_dir points to the Matterport3D train split."
        )

    print(f"Found {len(paths)} training scenes.")
    print(f"Writing index files to: {OUT_DIR}\n")
    total_labeled = 0
    total_valid   = 0

    for path in paths:
        coords, colors, labels = torch.load(path)
        labels = np.array(labels, dtype=np.int32)
        labels[labels == -100] = IGNORE_LABEL

        valid_idx = np.where(labels != IGNORE_LABEL)[0]
        if len(valid_idx) == 0:
            print(f"  [SKIP] {os.path.basename(path)}: no valid points")
            continue

        labeled = []
        for cls in np.unique(labels[valid_idx]):
            cls_idx = valid_idx[labels[valid_idx] == cls]
            k = max(1, int(len(cls_idx) * args.label_frac))
            chosen = rng.choice(cls_idx, size=k, replace=False)
            labeled.extend(chosen.tolist())

        labeled = np.array(sorted(set(labeled)), dtype=np.int64)
        scene_name = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(OUT_DIR, f"{scene_name}.npy"), labeled)

        total_labeled += len(labeled)
        total_valid   += len(valid_idx)
        print(
            f"  {scene_name}: {len(labeled):5d} / {len(valid_idx):6d} labeled "
            f"({100.0 * len(labeled) / len(valid_idx):.1f}%)"
        )

    print(
        f"\nDone. Total labeled: {total_labeled} / {total_valid} "
        f"({100.0 * total_labeled / max(1, total_valid):.2f}%)"
    )
    print(f"Index files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
