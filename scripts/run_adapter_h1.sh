#!/usr/bin/env bash
# Train H1 adapter (supervised + unmasked entropy on unlabeled points).
# Usage (from anywhere):
#   bash scripts/run_adapter_h1.sh
#   bash scripts/run_adapter_h1.sh labeled_indices_dir /work/.../train save_path /work/.../exp/h1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env PYTHONPATH=. python run/train_adapter.py --config config/matterport/adapter_h1.yaml "$@"
