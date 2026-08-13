#!/bin/bash
# Run TS-RAG and one project-method control under the same latent neighbors.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

winner_entries=()
csv_to_array "${WINNERS_CSV:-}" winner_entries
if [ "${#winner_entries[@]}" -ne 1 ]; then
  log_error "tsrag.slurm requires exactly one WINNERS_CSV pipeline"
  return 2
fi
IFS='/' read -r candidate_family candidate_run candidate_method candidate_extra <<< "${winner_entries[0]}"
if [ "$candidate_family" != baselines ] || [ -n "${candidate_extra:-}" ]; then
  log_error "tsrag.slurm currently requires one baseline winner"
  return 2
fi

CONTROL_ROOT="${CONTROL_ROOT:-outputs/adaptation/tsrag_controls}"
TSRAG_RESULTS_ROOT="${TSRAG_RESULTS_ROOT:-outputs/adaptation/tsrag}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/tsrag/full}"
IFS=',' read -r -a requested_stages <<< "${STAGES:-evaluate,tables}"
for stage in "${requested_stages[@]}"; do
  case "$stage" in
    evaluate)
      DATASETS_CSV="Electricity,Traffic,Solar,exchange_rate"
      SETTINGS_CSV=512:64
      DISTANCE_SPACES_CSV=tsrag
      DISTANCE_METRICS_CSV=euclidean
      NEIGHBORS_CSV=10
      RETRIEVAL_MODE=fixed
      RETRIEVAL_SCOPE=same_user
      DATASTORE_STRIDE=1
      ADAPT_QUERY_STRIDE=25
      EVAL_QUERY_STRIDE=127
      ALIGN_PERIOD=false
      MAX_STORE_WINDOWS=""
      export DATASETS_CSV SETTINGS_CSV DISTANCE_SPACES_CSV DISTANCE_METRICS_CSV
      export NEIGHBORS_CSV RETRIEVAL_MODE RETRIEVAL_SCOPE DATASTORE_STRIDE
      export ADAPT_QUERY_STRIDE EVAL_QUERY_STRIDE ALIGN_PERIOD MAX_STORE_WINDOWS
      for control_model in chronos2 chronos-bolt; do
        MODELS_CSV="$control_model"
        RESULTS_ROOT="$CONTROL_ROOT"
        BASELINE_METHODS_CSV="$candidate_method"
        export MODELS_CSV RESULTS_ROOT BASELINE_METHODS_CSV
        source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
        source "$PROJECT_ROOT/src/slurm/run_baselines.sh"
      done
      MODELS_CSV=chronos-bolt
      RESULTS_ROOT="$TSRAG_RESULTS_ROOT"
      export MODELS_CSV RESULTS_ROOT
      source "$PROJECT_ROOT/src/slurm/run_tsrag.sh"
      ;;
    tables)
      srun --ntasks=1 python -m src.visu.tsrag_comparison_table \
        --controls-root "$CONTROL_ROOT" \
        --tsrag-root "$TSRAG_RESULTS_ROOT" \
        --output-dir "$REPORT_ROOT"
      ;;
    *) log_error "unknown STAGES entry=$stage expected=evaluate,tables"; return 2 ;;
  esac
done
