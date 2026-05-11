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
export SAVE_TAU03=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_entropy_tau03
export SAVE_TAU05=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_entropy_tau05
export SAVE_TAU07=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_entropy_tau07
export SAVE_SOFT=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_soft_entropy
export SAVE_PSEUDO=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_pseudo_label
export SAVE_SHARP=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_temp_sharpen
export SAVE_INV=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_inverted_mask
export SAVE_SPATIAL=/work/pi_adamoneill_umass_edu/tanviagarwal_umass_edu/openscene/experiments/adapter_spatial_smooth
