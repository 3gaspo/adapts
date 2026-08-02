#!/bin/bash
# Evaluate the official pretrained Cross-RAG checkpoint on our fixed held-out T3 payloads.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-crossrag}"
if [ "$EXPERIMENT_MODE" != crossrag ]; then
  log_error "crossrag.slurm only supports EXPERIMENT_MODE=crossrag"
  return 2
fi
require_experiment_mode
adaptation_profile_defaults
OUT_ROOT="${OUT_ROOT:-outputs/extractions}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation_results/crossrag}"
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
CROSSRAG_ROOT="${CROSSRAG_ROOT:-}"
CROSSRAG_BASE_CHECKPOINT="${CROSSRAG_BASE_CHECKPOINT:-}"
CROSSRAG_CHECKPOINT="${CROSSRAG_CHECKPOINT:-}"
CROSSRAG_BATCH_SIZE="${CROSSRAG_BATCH_SIZE:-256}"
CROSSRAG_DEVICE="${CROSSRAG_DEVICE:-cuda}"
SKIP_COMPLETE="${SKIP_COMPLETE:-true}"
require_resolved_profile_grid

for variable in CROSSRAG_ROOT CROSSRAG_BASE_CHECKPOINT CROSSRAG_CHECKPOINT; do
  if [ -z "${!variable}" ]; then
    log_error "$variable must point to the downloaded official Cross-RAG repository/checkpoint"
    return 2
  fi
done

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
    [ -s "$output/crossrag_timing.json" ]
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
  INPUT_DIR="$OUT_ROOT/$dataset/${L}_${H}/$model/$RETRIEVAL_SETTING/extracted"
  RESULT_RUN_ROOT="$RESULTS_ROOT/$dataset/${L}_${H}/$model/$RETRIEVAL_SETTING"
  OUTPUT_DIR="$RESULT_RUN_ROOT/crossrag"
  require_extraction "$INPUT_DIR"
  VANILLA_SOURCE="$OUT_ROOT/$dataset/${L}_${H}/$model/vanilla/vanilla_metrics.json"
  VANILLA_TIMING_SOURCE="$OUT_ROOT/$dataset/${L}_${H}/$model/vanilla/extraction_timing.json"
  VANILLA_DEST="$RESULTS_ROOT/$dataset/${L}_${H}/$model/vanilla"
  assert_files vanilla-metrics "$VANILLA_SOURCE" "$VANILLA_TIMING_SOURCE" "$INPUT_DIR/extraction_timing.json"
  mkdir -p "$VANILLA_DEST"
  copy_if_needed "$VANILLA_SOURCE" "$VANILLA_DEST/vanilla_metrics.json"
  copy_if_needed "$VANILLA_TIMING_SOURCE" "$VANILLA_DEST/extraction_timing.json"
  mkdir -p "$RESULT_RUN_ROOT"
  copy_if_needed "$INPUT_DIR/extraction_timing.json" "$RESULT_RUN_ROOT/extraction_timing.json"
  if is_true "$SKIP_COMPLETE" && crossrag_complete "$OUTPUT_DIR" &&
    [ "$OUTPUT_DIR/crossrag_metrics.json" -nt "$INPUT_DIR/extraction_manifest.json" ]; then
    log "skip complete family=crossrag dataset=$dataset"
    continue
  fi
  log_section "crossrag start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset retrieval=$RETRIEVAL_SETTING"
  srun --ntasks=1 python -m src.adaptors.cross_rag.evaluate \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --crossrag-root "$CROSSRAG_ROOT" \
    --base-checkpoint "$CROSSRAG_BASE_CHECKPOINT" \
    --checkpoint "$CROSSRAG_CHECKPOINT" \
    --batch-size "$CROSSRAG_BATCH_SIZE" \
    --device "$CROSSRAG_DEVICE"
  assert_files crossrag-output \
    "$OUTPUT_DIR/crossrag_metrics.json" \
    "$OUTPUT_DIR/crossrag_predictions.pt" \
    "$OUTPUT_DIR/crossrag_timing.json"
done
log_section "job done kind=crossrag output=$RESULTS_ROOT"
