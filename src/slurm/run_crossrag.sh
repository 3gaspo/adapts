#!/bin/bash
# Evaluate the official pretrained Cross-RAG checkpoint on our fixed held-out T3 payloads.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-crossrag}"
if [ "$EXPERIMENT_FAMILY" != crossrag ]; then
  log_error "crossrag.slurm only supports EXPERIMENT_FAMILY=crossrag"
  return 2
fi
require_experiment_mode
adaptation_profile_defaults
OUT_ROOT="${OUT_ROOT:-outputs/extraction}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation/crossrag}"
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
CHRONOS_BOLT_WEIGHTS_PATH="${CHRONOS_BOLT_WEIGHTS_PATH:-}"
[ -n "$CHRONOS_BOLT_WEIGHTS_PATH" ] || \
  CHRONOS_BOLT_WEIGHTS_PATH="$(find_weight_path chronos-bolt-base)"
CROSSRAG_WEIGHTS_PATH="${CROSSRAG_WEIGHTS_PATH:-}"
[ -n "$CROSSRAG_WEIGHTS_PATH" ] || \
  CROSSRAG_WEIGHTS_PATH="$(find_weight_path cross-rag)"
CROSSRAG_BATCH_SIZE="${CROSSRAG_BATCH_SIZE:-256}"
CROSSRAG_DEVICE="${CROSSRAG_DEVICE:-cuda}"
SKIP_COMPLETE="${SKIP_COMPLETE:-true}"
require_resolved_profile_grid

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$DISTANCE_METRICS_CSV" DISTANCE_METRICS
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS

TASKS=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for space in "${DISTANCE_SPACES[@]}"; do
        for metric in "${DISTANCE_METRICS[@]}"; do
          for neighbors in "${NEIGHBORS[@]}"; do
            TASKS+=("$dataset|$model|$setting|$space|$metric|$neighbors")
          done
        done
      done
    done
  done
done

crossrag_complete() {
  local output="$1"
  [ -s "$output/crossrag_metrics.json" ] &&
    [ -s "$output/crossrag_predictions.pt" ] &&
    [ -s "$output/crossrag_timing.json" ] &&
    [ -s "$output/result_manifest.json" ] &&
    grep -Eq '"format"[[:space:]]*:[[:space:]]*"adaptation_crossrag_result"' \
      "$output/result_manifest.json"
}

log_section "job start kind=crossrag experiment_mode=$EXPERIMENT_MODE tasks=${#TASKS[@]} fixed_setting=512:64 fixed_neighbors=15"
for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
  IFS='|' read -r dataset model setting space metric neighbors <<< "${TASKS[$task_id]}"
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  if [ "$L" -ne 512 ] || [ "$H" -ne 64 ] || [ "$neighbors" -ne 15 ] ||
    [ "$model" != chronos-bolt ] || [ "$space" != minmax ] ||
    [ "$metric" != cosine ]; then
    log_error "Cross-RAG comparison is fixed to chronos-bolt L=512 H=64 K=15 minmax/cosine"
    return 2
  fi
  RETRIEVAL_SETTING="${space}_${metric}_${neighbors}_${RETRIEVAL_MODE}"
  resolve_extraction_run "$dataset" "$L" "$H" "$model" "$space" "$metric" "$neighbors" "$RETRIEVAL_MODE"
  INPUT_DIR="$EXTRACTION_RUN_DIR"
  require_extraction "$INPUT_DIR"
  resolve_extraction_run "$dataset" "$L" "$H" "$model" none none 0 none
  VANILLA_SOURCE="$EXTRACTION_RUN_DIR/vanilla_metrics.json"
  VANILLA_TIMING_SOURCE="$EXTRACTION_RUN_DIR/extraction_timing.json"
  assert_files vanilla-metrics "$VANILLA_SOURCE" "$VANILLA_TIMING_SOURCE" "$INPUT_DIR/extraction_timing.json"
  identity_root="$RESULTS_ROOT/$dataset/${L}_${H}/${model,,}/crossrag/${space,,}/${metric,,}/$neighbors/${RETRIEVAL_MODE,,}"
  model_values=("formula=crossrag" "space=$space" "metric=$metric" "k=$neighbors" "mode=$RETRIEVAL_MODE")
  pipeline_values=("batch_size=$CROSSRAG_BATCH_SIZE")
  ADDITIONAL_INPUTS=(
    "vanilla_manifest=$EXTRACTION_RUN_DIR/manifest.json"
    "chronos_bolt_weights=$CHRONOS_BOLT_WEIGHTS_PATH"
    "crossrag_weights=$CROSSRAG_WEIGHTS_PATH"
  )
  allocate_manifest_run "$identity_root" adaptation/crossrag "$dataset" "$L" "$H" "$model" \
    formula,space,metric,k,mode crossrag formula space,metric,k,mode \
    model_values pipeline_values 1 "$INPUT_DIR/manifest.json"
  unset ADDITIONAL_INPUTS
  OUTPUT_DIR="$ALLOCATED_RUN_DIR"
  if [ "$ALLOCATED_ACTION" = skip ]; then
    log "skip complete family=crossrag dataset=$dataset run=$OUTPUT_DIR"
    continue
  fi
  mark_manifest_running "$OUTPUT_DIR"
  log_section "crossrag start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset retrieval=$RETRIEVAL_SETTING"
  srun --ntasks=1 python -m src.adaptors.cross_rag.evaluate \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --chronos-bolt-weights "$CHRONOS_BOLT_WEIGHTS_PATH" \
    --cross-rag-weights "$CROSSRAG_WEIGHTS_PATH" \
    --batch-size "$CROSSRAG_BATCH_SIZE" \
    --device "$CROSSRAG_DEVICE"
  assert_files crossrag-output \
    "$OUTPUT_DIR/crossrag_metrics.json" \
    "$OUTPUT_DIR/crossrag_predictions.pt" \
    "$OUTPUT_DIR/crossrag_timing.json" \
    "$OUTPUT_DIR/result_manifest.json"
  mark_manifest_ready "$OUTPUT_DIR" crossrag_metrics.json crossrag_predictions.pt crossrag_timing.json result_manifest.json
done
log_section "job done kind=crossrag output=$RESULTS_ROOT"
