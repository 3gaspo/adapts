#!/bin/bash
# Expand selected shared CatBoost screen winners across objective/shape variants.
# Submit ../../catboost_ablation.slurm.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root

if [ "${EXPERIMENT_MODE:-}" != catboost_ablation ]; then
  log_error "run_catboost_ablation.sh requires EXPERIMENT_MODE=catboost_ablation"
  return 2
fi

CATBOOST_WINNERS_CSV="${CATBOOST_WINNERS_CSV:-}"
if [ -z "$CATBOOST_WINNERS_CSV" ]; then
  log_error "set CATBOOST_WINNERS_CSV to selected gates/retrieval/shared-regressor winners"
  return 2
fi

selected_winners=()
ablation_winners=()
csv_to_array "$CATBOOST_WINNERS_CSV" selected_winners
for entry in "${selected_winners[@]}"; do
  IFS='/' read -r family run method extra <<< "$entry"
  if [ "$family" != gates ] || [ -z "$run" ] || [ -z "$method" ] || [ -n "${extra:-}" ]; then
    log_error "invalid CatBoost winner=$entry expected=gates/retrieval_run/shared_regressor"
    return 2
  fi
  case "$method" in
    catboost_context_regressor_shared)
      candidate=context
      direct=context_forecast
      ;;
    catboost_aggr_y_regressor_shared)
      candidate=aggr_y
      direct=aggr_y
      ;;
    *)
      log_error "CatBoost winner must be a primary shared regressor method=$method"
      return 2
      ;;
  esac
  methods=(
    "$direct"
    "bayes_${candidate}_shared"
    "bayes_${candidate}_horizon"
    "catboost_${candidate}_regressor_shared"
    "catboost_${candidate}_regressor_horizon"
    "catboost_${candidate}_classifier_shared"
    "catboost_${candidate}_classifier_horizon"
    "oracle_${candidate}_shared"
    "oracle_${candidate}_horizon"
  )
  for variant in "${methods[@]}"; do
    ablation_winners+=("gates/$run/$variant")
  done
done

join_csv_values() {
  local IFS=,
  echo "$*"
}

WINNERS_CSV="$(join_csv_values "${ablation_winners[@]}")"
export WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
