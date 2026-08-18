#!/bin/bash
# Evaluate our selected method on the Cross-RAG paper's globally standardized
# data and official test windows. This is not the project's per-window nMSE.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPERIMENT_FAMILY=sota_benchmark
EXPERIMENT_MODE=full
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation/sota_benchmark}"
REPORT_ROOT="${REPORT_ROOT:-outputs/reports/sota_benchmark/full}"
SOTA_CONFIG="${SOTA_CONFIG:-$PROJECT_ROOT/SOTA_BENCHMARK.json}"
export EXPERIMENT_FAMILY EXPERIMENT_MODE RESULTS_ROOT

TABLE_SELECTION_ARGS=(
  --config-policy "${TABLE_CONFIG_POLICY:-distinct}"
  --repeat-policy "${TABLE_REPEAT_POLICY:-selected}"
)
if [ -n "${TABLE_PIPELINE_CONFIGS:-}" ]; then
  for table_config in ${TABLE_PIPELINE_CONFIGS}; do
    TABLE_SELECTION_ARGS+=(--pipeline-config "$table_config")
  done
fi
if [ -n "${TABLE_PURPOSE:-publication}" ]; then
  TABLE_SELECTION_ARGS+=(--purpose "${TABLE_PURPOSE:-publication}")
fi

winner_entries=()
csv_to_array "${WINNERS_CSV:-}" winner_entries
if [ "${#winner_entries[@]}" -ne 1 ]; then
  log_error "sota_benchmark.slurm requires exactly one WINNERS_CSV pipeline"
  return 2
fi
IFS='/' read -r candidate_family candidate_run candidate_method candidate_extra <<< "${winner_entries[0]}"
if [ -n "${candidate_extra:-}" ]; then
  log_error "invalid winner=${winner_entries[0]}"
  return 2
fi
if [ "$candidate_family" != baselines ]; then
  log_error "sota_benchmark.slurm currently requires one baseline winner"
  return 2
fi
IFS='_' read -r candidate_space candidate_metric candidate_k candidate_mode candidate_tail <<< "$candidate_run"
if [ -n "${candidate_tail:-}" ]; then
  log_error "invalid retrieval pipeline=$candidate_run"
  return 2
fi

IFS=',' read -r -a requested_stages <<< "${STAGES:-evaluate,tables}"
for stage in "${requested_stages[@]}"; do
  case "$stage" in
    evaluate)
      mapfile -t dataset_protocols < <(python -c '
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
for values in config["datasets"].values():
    bounds = ",".join(map(str, values.get("project_split_bounds", ())))
    print(f"{values['"'"'project_name'"'"']}|{bounds}|{values['"'"'standardize_train_boundary'"'"']}")
' "$SOTA_CONFIG")
      for protocol in "${dataset_protocols[@]}"; do
        IFS='|' read -r dataset SPLIT_BOUNDS STANDARDIZE_TRAIN_BOUNDARY <<< "$protocol"
        SPLITS=0.3,0.5,0.2
        DATASETS_CSV="$dataset"
        MODELS_CSV=chronos-bolt
        SETTINGS_CSV=512:64
        DISTANCE_SPACES_CSV="$candidate_space"
        DISTANCE_METRICS_CSV="$candidate_metric"
        NEIGHBORS_CSV="$candidate_k"
        RETRIEVAL_MODE="$candidate_mode"
        RETRIEVAL_SCOPE=all
        DATASTORE_STRIDE=25
        ADAPT_QUERY_STRIDE=25
        EVAL_QUERY_STRIDE=1
        ALIGN_PERIOD=false
        MAX_STORE_WINDOWS=30000
        SKIP_COMPLETE=true
        export DATASETS_CSV MODELS_CSV SETTINGS_CSV DISTANCE_SPACES_CSV
        export DISTANCE_METRICS_CSV NEIGHBORS_CSV RETRIEVAL_MODE RETRIEVAL_SCOPE
        export SPLITS SPLIT_BOUNDS STANDARDIZE_TRAIN_BOUNDARY DATASTORE_STRIDE
        export ADAPT_QUERY_STRIDE EVAL_QUERY_STRIDE ALIGN_PERIOD MAX_STORE_WINDOWS
        source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
        BASELINE_METHODS_CSV="$candidate_method"
        export BASELINE_METHODS_CSV
        source "$PROJECT_ROOT/src/slurm/run_baselines.sh"
      done
      ;;
    tables)
      srun --ntasks=1 python -m src.visu.sota_benchmark_table \
        --config "$SOTA_CONFIG" \
        --results-root "$RESULTS_ROOT" \
        --output-dir "$REPORT_ROOT" \
        --formula "$candidate_method" \
        --space "$candidate_space" \
        --distance-metric "$candidate_metric" \
        --neighbors "$candidate_k" \
        --retrieval-mode "$candidate_mode" \
        "${TABLE_SELECTION_ARGS[@]}"
      ;;
    *) log_error "unknown STAGES entry=$stage expected=evaluate,tables"; return 2 ;;
  esac
done
