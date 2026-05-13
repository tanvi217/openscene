#!/usr/bin/env bash
# Train adapter: H1 (unmasked) + H2 + H3 + spatial smoothness (see yaml).
# Usage:
#   bash scripts/run_adapter_combined_regularizers.sh
#   bash scripts/run_adapter_combined_regularizers.sh labeled_indices_dir /work/.../train save_path /work/.../exp/combined
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec env PYTHONPATH=. python run/train_adapter.py \
  --config config/matterport/adapter_combined_h2_h3_spatial.yaml "$@"
