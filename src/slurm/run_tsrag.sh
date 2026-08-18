#!/bin/bash
# Evaluate released TS-RAG with its Chronos-T5 latent retrieval defaults.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-tsrag}"
if [ "$EXPERIMENT_FAMILY" != tsrag ]; then
  log_error "tsrag.slurm only supports EXPERIMENT_FAMILY=tsrag"
  return 2
fi
require_experiment_mode
adaptation_profile_defaults
OUT_ROOT="${OUT_ROOT:-outputs/extraction}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation/tsrag}"
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-fixed}"
RETRIEVAL_SCOPE="${RETRIEVAL_SCOPE:-same_user}"
CHRONOS_BOLT_WEIGHTS_PATH="${CHRONOS_BOLT_WEIGHTS_PATH:-}"
[ -n "$CHRONOS_BOLT_WEIGHTS_PATH" ] || \
  CHRONOS_BOLT_WEIGHTS_PATH="$(find_weight_path chronos-bolt-base)"
TSRAG_WEIGHTS_PATH="${TSRAG_WEIGHTS_PATH:-}"
[ -n "$TSRAG_WEIGHTS_PATH" ] || \
  TSRAG_WEIGHTS_PATH="$(find_weight_path ts-rag)"
TSRAG_BATCH_SIZE="${TSRAG_BATCH_SIZE:-256}"
TSRAG_DEVICE="${TSRAG_DEVICE:-cuda}"
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

log_section "job start kind=tsrag_default_retrieval experiment_mode=$EXPERIMENT_MODE tasks=${#TASKS[@]}"
for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
  IFS='|' read -r dataset model setting space metric neighbors <<< "${TASKS[$task_id]}"
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  if [ "$L" -ne 512 ] || [ "$H" -ne 64 ] || [ "$neighbors" -ne 10 ] ||
    [ "$model" != chronos-bolt ] || [ "$space" != tsrag ] ||
    [ "$metric" != euclidean ] || [ "$RETRIEVAL_SCOPE" != same_user ]; then
    log_error "TS-RAG is fixed to chronos-bolt L=512 H=64 K=10 Chronos-T5/euclidean same-user retrieval"
    return 2
  fi
  resolve_extraction_run "$dataset" "$L" "$H" "$model" "$space" "$metric" "$neighbors" "$RETRIEVAL_MODE"
  INPUT_DIR="$EXTRACTION_RUN_DIR"
  require_extraction "$INPUT_DIR"
  identity_root="$RESULTS_ROOT/$dataset/${L}_${H}/${model,,}/tsrag/${space,,}/${metric,,}/$neighbors/${RETRIEVAL_MODE,,}"
  model_values=("formula=tsrag" "space=$space" "metric=$metric" "k=$neighbors" "mode=$RETRIEVAL_MODE")
  pipeline_values=(
    "batch_size=$TSRAG_BATCH_SIZE" "retrieval_protocol=tsrag_default"
    "retrieval_scope=$RETRIEVAL_SCOPE" "metric_set=mse,mae,nmse,nmae"
    "runtime_fields=inference_seconds,inference_ms_per_example,total_parameters"
  )
  ADDITIONAL_INPUTS=(
    "chronos_bolt_weights=$CHRONOS_BOLT_WEIGHTS_PATH"
    "tsrag_weights=$TSRAG_WEIGHTS_PATH"
  )
  allocate_manifest_run "$identity_root" adaptation/tsrag "$dataset" "$L" "$H" "$model" \
    formula,space,metric,k,mode tsrag formula space,metric,k,mode \
    model_values pipeline_values 1 "$INPUT_DIR/manifest.json"
  unset ADDITIONAL_INPUTS
  OUTPUT_DIR="$ALLOCATED_RUN_DIR"
  if [ "$ALLOCATED_ACTION" = skip ]; then
    log "skip complete family=tsrag dataset=$dataset run=$OUTPUT_DIR"
    continue
  fi
  mark_manifest_running "$OUTPUT_DIR"
  log_section "tsrag start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset"
  srun --ntasks=1 python -m src.adaptors.ts_rag.evaluate \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --chronos-bolt-weights "$CHRONOS_BOLT_WEIGHTS_PATH" \
    --ts-rag-weights "$TSRAG_WEIGHTS_PATH" \
    --neighbors "$neighbors" \
    --batch-size "$TSRAG_BATCH_SIZE" \
    --device "$TSRAG_DEVICE"
  assert_files tsrag-output \
    "$OUTPUT_DIR/tsrag_metrics.json" \
    "$OUTPUT_DIR/tsrag_predictions.pt" \
    "$OUTPUT_DIR/tsrag_timing.json" \
    "$OUTPUT_DIR/result_manifest.json"
  mark_manifest_ready "$OUTPUT_DIR" tsrag_metrics.json tsrag_predictions.pt tsrag_timing.json result_manifest.json
done
log_section "job done kind=tsrag_default_retrieval output=$RESULTS_ROOT"
