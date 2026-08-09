#!/bin/bash
# Fit adaptation gates from completed extractions and write gate metrics/artifacts.
# Submit ../../gates.slurm; source this implementation only for local debugging.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT_ROOT="${OUT_ROOT:-outputs/extraction}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
require_experiment_mode
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-gates}"
require_experiment_family
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation/$EXPERIMENT_FAMILY}"
adaptation_profile_defaults
case "$EXPERIMENT_MODE:$EXPERIMENT_FAMILY" in
  test:*)
    DEFAULT_GATE_ITERATIONS=2
    ;;
  *:k_ablation|*:h_ablation|*:l_ablation|*:crossrag)
    DEFAULT_GATE_ITERATIONS=300
    ;;
  *)
    DEFAULT_GATE_ITERATIONS=300
    ;;
esac
DEFAULT_SKIP_COMPLETE=true
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
case "$EXPERIMENT_MODE:$EXPERIMENT_FAMILY" in
  test:*|*:screen|*:gates) DEFAULT_GATE_METHODS_CSV="$PRIMARY_GATE_METHODS_CSV" ;;
  *) DEFAULT_GATE_METHODS_CSV="" ;;
esac
GATE_METHODS_CSV="${GATE_METHODS_CSV:-$DEFAULT_GATE_METHODS_CSV}"
GATE_ITERATIONS="${GATE_ITERATIONS:-$DEFAULT_GATE_ITERATIONS}"
SKIP_COMPLETE="${SKIP_COMPLETE:-$DEFAULT_SKIP_COMPLETE}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
GATE_LEARNING_RATE="${GATE_LEARNING_RATE:-0.03}"
GATE_DEPTH="${GATE_DEPTH:-4}"
GATE_EARLY_STOPPING_ROUNDS="${GATE_EARLY_STOPPING_ROUNDS:-50}"
GATE_TASK_TYPE="${GATE_TASK_TYPE:-CPU}"
GATE_TASK_TYPE="${GATE_TASK_TYPE^^}"
GATE_THREAD_COUNT="${GATE_THREAD_COUNT:-2}"
GATE_DEVICES="${GATE_DEVICES:-}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}"
SEED="${SEED:-1}"
MAX_T1_FIT_SAMPLES="${MAX_T1_FIT_SAMPLES:-}"
MAX_T2_VALID_SAMPLES="${MAX_T2_VALID_SAMPLES:-}"
MAX_ADAPT_REFIT_SAMPLES="${MAX_ADAPT_REFIT_SAMPLES:-}"
require_resolved_profile_grid
require_profile_neighbors "$NEIGHBORS_CSV"
if requires_selected_methods && [ -z "$GATE_METHODS_CSV" ]; then
  log_error "EXPERIMENT_MODE=$EXPERIMENT_MODE requires GATE_METHODS_CSV for the gate candidate run"
  return 2
fi

FIT_SAMPLE_ARGS=(--fit-sample-seed "$SEED")
[ -z "$MAX_T1_FIT_SAMPLES" ] || FIT_SAMPLE_ARGS+=(--max-t1-fit-samples "$MAX_T1_FIT_SAMPLES")
[ -z "$MAX_T2_VALID_SAMPLES" ] || FIT_SAMPLE_ARGS+=(--max-t2-valid-samples "$MAX_T2_VALID_SAMPLES")
[ -z "$MAX_ADAPT_REFIT_SAMPLES" ] || FIT_SAMPLE_ARGS+=(--max-adapt-refit-samples "$MAX_ADAPT_REFIT_SAMPLES")

ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-$GATE_THREAD_COUNT}"
if [ "$GATE_TASK_TYPE" = CPU ] &&
  [ "$GATE_THREAD_COUNT" -gt "$ALLOCATED_CPUS" ]; then
  log_error "gate threads exceed allocated CPUs: $GATE_THREAD_COUNT > $ALLOCATED_CPUS"
  return 2
fi
GATE_EXECUTION_ARGS=(
  --gate-task-type "$GATE_TASK_TYPE"
  --gate-thread-count "$GATE_THREAD_COUNT"
)
[ -z "$GATE_DEVICES" ] || GATE_EXECUTION_ARGS+=(--gate-devices "$GATE_DEVICES")

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$DISTANCE_METRICS_CSV" DISTANCE_METRICS
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS
csv_to_array "$GATE_METHODS_CSV" GATE_METHODS

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

gate_complete() {
  local output="$1"
  [ -s "$output/gate_metrics.csv" ] &&
    [ -s "$output/gate_metrics.json" ] &&
    [ -s "$output/gate_artifacts.json" ] &&
    [ -s "$output/prediction_manifest.json" ] &&
    [ -s "$output/gate_timing.json" ] &&
    [ -s "$output/result_manifest.json" ] &&
    grep -Eq '"format"[[:space:]]*:[[:space:]]*"adaptation_evaluation_result"' \
      "$output/result_manifest.json" &&
    grep -Eq '"family"[[:space:]]*:[[:space:]]*"gates"' \
      "$output/result_manifest.json" &&
    grep -Eq '"format"[[:space:]]*:[[:space:]]*"adaptation_prediction_store"' \
      "$output/prediction_manifest.json"
}

run_task() {
  local task_id="$1" task dataset model setting space metric neighbors method
  local input_manifest vanilla_dir identity_root
  local -a model_values pipeline_values
  task="${TASKS[$task_id]}"
  IFS='|' read -r dataset model setting space metric neighbors <<< "$task"
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  RETRIEVAL_SETTING="${space}_${metric}_${neighbors}_${RETRIEVAL_MODE}"
  resolve_extraction_run "$dataset" "$L" "$H" "$model" "$space" "$metric" "$neighbors" "$RETRIEVAL_MODE"
  INPUT_DIR="$EXTRACTION_RUN_DIR"
  require_extraction "$INPUT_DIR"
  resolve_extraction_run "$dataset" "$L" "$H" "$model" none none 0 none
  vanilla_dir="$EXTRACTION_RUN_DIR"
  VANILLA_SOURCE="$vanilla_dir/vanilla_metrics.json"
  VANILLA_TIMING_SOURCE="$vanilla_dir/extraction_timing.json"
  assert_files vanilla-metrics "$VANILLA_SOURCE" "$VANILLA_TIMING_SOURCE" "$INPUT_DIR/extraction_timing.json"
  input_manifest="$INPUT_DIR/manifest.json"
  for method in "${GATE_METHODS[@]}"; do
    identity_root="$RESULTS_ROOT/$dataset/${L}_${H}/${model,,}/${method,,}/${space,,}/${metric,,}/$neighbors/${RETRIEVAL_MODE,,}"
    model_values=("formula=$method" "space=$space" "metric=$metric" "k=$neighbors" "mode=$RETRIEVAL_MODE")
    pipeline_values=(
      "gate.iterations=$GATE_ITERATIONS" "gate.learning_rate=$GATE_LEARNING_RATE" "gate.depth=$GATE_DEPTH"
      "gate.early_stopping_rounds=$GATE_EARLY_STOPPING_ROUNDS" "validation_fraction=$VALIDATION_FRACTION"
      "max_t1_fit_samples=${MAX_T1_FIT_SAMPLES:-none}" "max_t2_valid_samples=${MAX_T2_VALID_SAMPLES:-none}"
      "max_adapt_refit_samples=${MAX_ADAPT_REFIT_SAMPLES:-none}" "run_seed=$SEED"
    )
    ADDITIONAL_INPUTS=("vanilla_manifest=$vanilla_dir/manifest.json")
    allocate_manifest_run "$identity_root" "adaptation/$EXPERIMENT_FAMILY/gates" "$dataset" "$L" "$H" "$model" \
      formula,space,metric,k,mode "$method" formula space,metric,k,mode \
      model_values pipeline_values "$SEED" "$input_manifest"
    unset ADDITIONAL_INPUTS
    OUTPUT_DIR="$ALLOCATED_RUN_DIR"
    if [ "$ALLOCATED_ACTION" = skip ]; then
      log "skip complete family=gates dataset=$dataset model=$model formula=$method lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING run=$OUTPUT_DIR"
      continue
    fi
    mark_manifest_running "$OUTPUT_DIR"
    log_section "gates start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model formula=$method lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING run=$OUTPUT_DIR computation_signature=$ALLOCATED_SIGNATURE iterations=$GATE_ITERATIONS learning_rate=$GATE_LEARNING_RATE depth=$GATE_DEPTH early_stopping_rounds=$GATE_EARLY_STOPPING_ROUNDS task_type=$GATE_TASK_TYPE thread_count=$GATE_THREAD_COUNT validation_fraction=$VALIDATION_FRACTION seed=$SEED"
    srun --ntasks=1 python -m src.adaptors.baselines.evaluate \
      --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --family gates \
      --gate-iterations "$GATE_ITERATIONS" --gate-learning-rate "$GATE_LEARNING_RATE" \
      --gate-depth "$GATE_DEPTH" --gate-early-stopping-rounds "$GATE_EARLY_STOPPING_ROUNDS" \
      "${GATE_EXECUTION_ARGS[@]}" --validation-fraction "$VALIDATION_FRACTION" \
      "${FIT_SAMPLE_ARGS[@]}" --methods "$method" --seed "$SEED"
    assert_files gate-output \
      "$OUTPUT_DIR/gate_metrics.csv" "$OUTPUT_DIR/gate_metrics.json" \
      "$OUTPUT_DIR/gate_artifacts.json" "$OUTPUT_DIR/prediction_manifest.json" \
      "$OUTPUT_DIR/gate_timing.json" "$OUTPUT_DIR/result_manifest.json"
    mark_manifest_completed "$OUTPUT_DIR" gate_metrics.csv gate_metrics.json gate_artifacts.json prediction_manifest.json gate_timing.json result_manifest.json
    log "gates done configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model formula=$method lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
  done
}

log_section "job start kind=gates family=$EXPERIMENT_FAMILY experiment_mode=$EXPERIMENT_MODE skip_complete=$SKIP_COMPLETE tasks=${#TASKS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV distance_spaces=$DISTANCE_SPACES_CSV distance_metrics=$DISTANCE_METRICS_CSV neighbors=$NEIGHBORS_CSV methods=$GATE_METHODS_CSV task_type=$GATE_TASK_TYPE thread_count=$GATE_THREAD_COUNT horizon_fits=serial allocated_cpus=$ALLOCATED_CPUS results_root=$RESULTS_ROOT"
for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
  run_task "$task_id"
done
log_section "job done kind=gates output=$RESULTS_ROOT"
