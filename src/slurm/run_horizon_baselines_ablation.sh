#!/bin/bash
# Compare selected shared baseline winners with their per-horizon counterparts.
# Submit ../../horizon_baselines_ablation.slurm.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root

if [ "${EXPERIMENT_MODE:-}" != horizon_baselines_ablation ]; then
  log_error "run_horizon_baselines_ablation.sh requires EXPERIMENT_MODE=horizon_baselines_ablation"
  return 2
fi

SHARED_BASELINE_WINNERS_CSV="${SHARED_BASELINE_WINNERS_CSV:-}"
if [ -z "$SHARED_BASELINE_WINNERS_CSV" ]; then
  log_error "set SHARED_BASELINE_WINNERS_CSV to complete baselines/retrieval/method names"
  return 2
fi

shared_winners=()
ablation_winners=()
csv_to_array "$SHARED_BASELINE_WINNERS_CSV" shared_winners
for entry in "${shared_winners[@]}"; do
  IFS='/' read -r family run method extra <<< "$entry"
  if [ "$family" != baselines ] || [ -z "$run" ] || [ -z "$method" ] || [ -n "${extra:-}" ]; then
    log_error "invalid shared baseline winner=$entry expected=baselines/retrieval_run/shared_method"
    return 2
  fi
  case "$method" in
    avgy_mix_shared) horizon_method=avgy_mix_horizon ;;
    cov_ridge_shared) horizon_method=cov_ridge_horizon ;;
    avgy_ridge_shared) horizon_method=avgy_ridge_horizon ;;
    y_ridge_shared) horizon_method=y_ridge_horizon ;;
    cov_y_ridge_shared) horizon_method=cov_y_ridge_horizon ;;
    cov_avgy_ridge_shared) horizon_method=cov_avgy_ridge_horizon ;;
    residual_ridge_shared) horizon_method=residual_ridge_horizon ;;
    full_ridge_shared) horizon_method=full_ridge_horizon ;;
    *)
      log_error "shared baseline has no per-horizon counterpart method=$method"
      return 2
      ;;
  esac
  ablation_winners+=("baselines/$run/$method" "baselines/$run/$horizon_method")
done

join_csv_values() {
  local IFS=,
  echo "$*"
}

WINNERS_CSV="$(join_csv_values "${ablation_winners[@]}")"
export WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
