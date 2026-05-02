#!/usr/bin/env bash
#
# Run the full H2 adapter pipeline (prepare → baseline confidences → train/eval Run 1 → train/eval Run 2).
# Intended cwd: run from anywhere; script cd's to the OpenScene tree root.
#
# Required environment variables:
#   MP_ROOT       Root directory containing train/ and test/ Matterport .pth scenes (matterport_3d layout)
#   FUSED_ROOT    Directory with pre-fused OpenSeg .pt features per scene
#
# Optional (defaults are under the OpenScene directory):
#   LABELED_OUT   Output dir for prepare_labeled_split (default: data/matterport_labeled_indices/train)
#   CONF_OUT      Baseline confidence .npy dir for Run 0 + evaluate_adapter (default: data/matterport_baseline_confidence/test)
#   SAVE_SUP      save_path for Run 1 training (default: experiments/adapter_sup_only)
#   SAVE_H2       save_path for Run 2 training (default: experiments/adapter_with_entropy)
#
# Skip steps (set to 1):
#   SKIP_PREPARE  Skip labeled-index generation
#   SKIP_RUN0     Skip fusion baseline + confidence export
#   SKIP_RUN1     Skip supervised-only train + eval
#   SKIP_RUN2     Skip H2 train + eval
#
# Optional extra train (set to 1):
#   RUN_TENT      After Run 2, train adapter_tent.yaml (TENT ablation)
#
# Unity concrete paths: source scripts/h2_pipeline_env_unity.sh (see that file).
#
# Example:
#   cd third_party/openscene
#   module load conda/latest && conda activate repopt
#   source scripts/h2_pipeline_env_unity.sh
#   bash scripts/run_h2_pipeline.sh 2>&1 | tee ~/h2_pipeline.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENSCENE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${OPENSCENE_ROOT}"
export PYTHONPATH="${OPENSCENE_ROOT}"

if [[ -z "${MP_ROOT:-}" || -z "${FUSED_ROOT:-}" ]]; then
  echo "ERROR: Set MP_ROOT (matterport_3d root) and FUSED_ROOT (fused features dir)." >&2
  echo "Example: export MP_ROOT=/path/to/matterport_3d FUSED_ROOT=/path/to/fused_pt" >&2
  exit 1
fi

LABELED_OUT="${LABELED_OUT:-${OPENSCENE_ROOT}/data/matterport_labeled_indices/train}"
CONF_OUT="${CONF_OUT:-${OPENSCENE_ROOT}/data/matterport_baseline_confidence/test}"
SAVE_SUP="${SAVE_SUP:-${OPENSCENE_ROOT}/experiments/adapter_sup_only}"
SAVE_H2="${SAVE_H2:-${OPENSCENE_ROOT}/experiments/adapter_with_entropy}"

SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_RUN0="${SKIP_RUN0:-0}"
SKIP_RUN1="${SKIP_RUN1:-0}"
SKIP_RUN2="${SKIP_RUN2:-0}"
RUN_TENT="${RUN_TENT:-0}"

echo "==> OpenScene root: ${OPENSCENE_ROOT}"
echo "==> MP_ROOT=${MP_ROOT}"
echo "==> FUSED_ROOT=${FUSED_ROOT}"
echo "==> LABELED_OUT=${LABELED_OUT}"
echo "==> CONF_OUT=${CONF_OUT}"
echo "==> SAVE_SUP=${SAVE_SUP}"
echo "==> SAVE_H2=${SAVE_H2}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo ""
  echo ">>> Step 1: prepare_labeled_split"
  python scripts/prepare_labeled_split.py \
    --in_dir "${MP_ROOT}/train" \
    --out_dir "${LABELED_OUT}"
else
  echo '>>> Skipping prepare_labeled_split; SKIP_PREPARE=1'
fi

if [[ "${SKIP_RUN0}" != "1" ]]; then
  echo ""
  echo ">>> Step 2: Run 0 - fusion baseline + confidence .npy"
  mkdir -p "${CONF_OUT}"
  python run/evaluate.py \
    --config config/matterport/eval_fusion_baseline.yaml \
    --save_confidence \
    --confidence_save_dir "${CONF_OUT}" \
    data_root "${MP_ROOT}" \
    data_root_2d_fused_feature "${FUSED_ROOT}"
else
  echo '>>> Skipping Run 0; SKIP_RUN0=1'
fi

if [[ "${SKIP_RUN1}" != "1" ]]; then
  echo ""
  echo ">>> Step 3: Run 1 - train (supervised only)"
  python run/train_adapter.py \
    --config config/matterport/adapter_sup_only.yaml \
    data_root "${MP_ROOT}" \
    data_root_2d_fused_feature "${FUSED_ROOT}" \
    labeled_indices_dir "${LABELED_OUT}" \
    save_path "${SAVE_SUP}"

  echo ""
  echo ">>> Step 4: Run 1 - evaluate"
  python run/evaluate_adapter.py \
    --config config/matterport/adapter_sup_only.yaml \
    --model_path "${SAVE_SUP}/model/model_best.pth.tar" \
    --confidence_dir "${CONF_OUT}" \
    data_root "${MP_ROOT}" \
    data_root_2d_fused_feature "${FUSED_ROOT}"
else
  echo '>>> Skipping Run 1; SKIP_RUN1=1'
fi

if [[ "${SKIP_RUN2}" != "1" ]]; then
  echo ""
  echo ">>> Step 5: Run 2 - train (H2)"
  python run/train_adapter.py \
    --config config/matterport/adapter_with_entropy.yaml \
    data_root "${MP_ROOT}" \
    data_root_2d_fused_feature "${FUSED_ROOT}" \
    labeled_indices_dir "${LABELED_OUT}" \
    save_path "${SAVE_H2}"

  echo ""
  echo ">>> Step 6: Run 2 - evaluate"
  python run/evaluate_adapter.py \
    --config config/matterport/adapter_with_entropy.yaml \
    --model_path "${SAVE_H2}/model/model_best.pth.tar" \
    --confidence_dir "${CONF_OUT}" \
    data_root "${MP_ROOT}" \
    data_root_2d_fused_feature "${FUSED_ROOT}"
else
  echo '>>> Skipping Run 2; SKIP_RUN2=1'
fi

if [[ "${RUN_TENT}" == "1" ]]; then
  SAVE_TENT="${SAVE_TENT:-${OPENSCENE_ROOT}/experiments/adapter_tent}"
  echo ""
  echo ">>> Optional: TENT train"
  python run/train_adapter.py \
    --config config/matterport/adapter_tent.yaml \
    data_root "${MP_ROOT}" \
    data_root_2d_fused_feature "${FUSED_ROOT}" \
    labeled_indices_dir "${LABELED_OUT}" \
    save_path "${SAVE_TENT}"

  echo ""
  echo ">>> Optional: TENT evaluate"
  python run/evaluate_adapter.py \
    --config config/matterport/adapter_tent.yaml \
    --model_path "${SAVE_TENT}/model/model_best.pth.tar" \
    --confidence_dir "${CONF_OUT}" \
    data_root "${MP_ROOT}" \
    data_root_2d_fused_feature "${FUSED_ROOT}"
fi

echo ""
echo "==> Pipeline finished OK."
