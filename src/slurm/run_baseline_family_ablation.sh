#!/bin/bash
# Compare selected shared-ridge screen winners with one controlled family change.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root

case "${EXPERIMENT_FAMILY:-}" in
  horizon_baselines_ablation) family=horizon ;;
  convex_baselines_ablation) family=convex ;;
  delta_baselines_ablation) family=delta ;;
  *)
    log_error "baseline family ablation requires horizon_baselines_ablation, convex_baselines_ablation, or delta_baselines_ablation"
    return 2
    ;;
esac

BASELINE_WINNERS_CSV="${BASELINE_WINNERS_CSV:-}"
if [ -z "$BASELINE_WINNERS_CSV" ]; then
  log_error "set BASELINE_WINNERS_CSV to complete baselines/retrieval/shared-ridge names"
  return 2
fi

selected_winners=()
ablation_winners=()
csv_to_array "$BASELINE_WINNERS_CSV" selected_winners
for entry in "${selected_winners[@]}"; do
  IFS='/' read -r candidate_family run method extra <<< "$entry"
  if [ "$candidate_family" != baselines ] || [ -z "$run" ] || [ -z "$method" ] || [ -n "${extra:-}" ]; then
    log_error "invalid baseline winner=$entry expected=baselines/retrieval_run/shared_ridge_method"
    return 2
  fi
  case "$method" in
    cov_ridge_shared) design=cov ;;
    avgy_ridge_shared) design=avgy ;;
    y_ridge_shared) design=y ;;
    cov_y_ridge_shared) design=cov_y ;;
    cov_avgy_ridge_shared) design=cov_avgy ;;
    residual_ridge_shared) design=residual ;;
    full_ridge_shared) design=full ;;
    *)
      log_error "baseline winner must be a primary shared-ridge method=$method"
      return 2
      ;;
  esac
  case "$family" in
    horizon) variant="${design}_ridge_horizon" ;;
    convex) variant="${design}_convex_shared" ;;
    delta) variant="${design}_delta_ridge_shared" ;;
  esac
  ablation_winners+=("baselines/$run/$method" "baselines/$run/$variant")
done

join_csv_values() {
  local IFS=,
  echo "$*"
}

WINNERS_CSV="$(join_csv_values "${ablation_winners[@]}")"
export WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
