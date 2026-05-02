# Unity (/work) paths for scripts/run_h2_pipeline.sh
# Matches config/matterport/adapter_*.yaml (Matterport 21-class H2 runs).
#
# Usage (from third_party/openscene):
#   module load conda/latest && conda activate repopt
#   source scripts/h2_pipeline_env_unity.sh
#   bash scripts/run_h2_pipeline.sh
#
# Optional: RUN_TENT=1 bash scripts/run_h2_pipeline.sh

export MP_ROOT=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/matterport_3d
export FUSED_ROOT=/work/pi_dagarwal_umass_edu/project_23/soorya/openscene/data/matterport_multiview_openseg_test

export LABELED_OUT=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/matterport_labeled_indices/train
export CONF_OUT=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/matterport_baseline_confidence/test

export SAVE_SUP=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_sup_only
export SAVE_H2=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_with_entropy
export SAVE_TENT=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_tent
