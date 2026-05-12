#!/usr/bin/env bash
# Create data_matterport_adapter/matterport_3d/{train,test} symlinks so adapter
# training sees train/*.pth while global `data/` may only expose Matterport test
# (e.g. Soorya read-only tree). Run once from openscene/:
#   bash scripts/setup_matterport_adapter_symlinks.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MP="${ROOT}/data_matterport_adapter/matterport_3d"
TRAIN_SRC="${TRAIN_SRC:-/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/matterport_3d/train}"
TEST_SRC="${TEST_SRC:-/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/matterport_3d/test}"

mkdir -p "$MP"
ln -sfn "$TRAIN_SRC" "${MP}/train"
ln -sfn "$TEST_SRC" "${MP}/test"
echo "OK: data_root should be"
echo "  ${MP}"
ls -la "$MP"
