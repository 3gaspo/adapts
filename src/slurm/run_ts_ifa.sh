#!/bin/bash
# Train and evaluate TS-IFA from completed extraction artifacts.
# Submit ../../ts_ifa.slurm; source this implementation only for local debugging.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/adaptation}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation_results/${EXPERIMENT_MODE:-test}}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
require_experiment_mode
adaptation_profile_defaults
case "$EXPERIMENT_MODE" in
  test)
    DEFAULT_BRANCH_EPOCHS=2
    DEFAULT_ROOTER_EPOCHS=2
    DEFAULT_VALID_EVAL_FREQ=1
    DEFAULT_LOGGING_EVAL_FREQ=1
    DEFAULT_MAX_TRAIN_SAMPLES=32
    DEFAULT_MAX_VALID_SAMPLES=32
    DEFAULT_SKIP_COMPLETE=false
    ;;
  *)
    DEFAULT_BRANCH_EPOCHS=10000
    DEFAULT_ROOTER_EPOCHS=10000
    DEFAULT_VALID_EVAL_FREQ=1000
    DEFAULT_LOGGING_EVAL_FREQ=1000
    DEFAULT_MAX_TRAIN_SAMPLES=""
    DEFAULT_MAX_VALID_SAMPLES=""
    DEFAULT_SKIP_COMPLETE=true
    ;;
esac
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
BRANCH_EPOCHS="${BRANCH_EPOCHS:-$DEFAULT_BRANCH_EPOCHS}"
ROOTER_EPOCHS="${ROOTER_EPOCHS:-$DEFAULT_ROOTER_EPOCHS}"
VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-$DEFAULT_VALID_EVAL_FREQ}"
LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-$DEFAULT_LOGGING_EVAL_FREQ}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES-$DEFAULT_MAX_TRAIN_SAMPLES}"
MAX_VALID_SAMPLES="${MAX_VALID_SAMPLES-$DEFAULT_MAX_VALID_SAMPLES}"
SKIP_COMPLETE="${SKIP_COMPLETE:-$DEFAULT_SKIP_COMPLETE}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
SEED="${SEED:-1}"

BATCH_SIZE="${BATCH_SIZE:-256}"
BRANCH_LR="${BRANCH_LR:-0.00001}"
ROOTER_LR="${ROOTER_LR:-0.00001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
VANILLA_ANCHOR="${VANILLA_ANCHOR:-0.01}"
COEFFICIENT_L2="${COEFFICIENT_L2:-0.01}"
HORIZON_SMOOTHNESS="${HORIZON_SMOOTHNESS:-0.01}"
RIDGE_ROOTER_ALPHA="${RIDGE_ROOTER_ALPHA:-0.01}"
DROPOUT="${DROPOUT:-0.0}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
ATTENTION_DIM="${ATTENTION_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
TRANSFORMED_HIDDEN_DIM="${TRANSFORMED_HIDDEN_DIM:-$HIDDEN_DIM}"
TRANSFORMED_EXPERT="${TRANSFORMED_EXPERT:-false}"
LEARNABLE_TRANSFORMED_COVARIATE="${LEARNABLE_TRANSFORMED_COVARIATE:-false}"
VANILLA_ANCHORING_INIT="${VANILLA_ANCHORING_INIT:-true}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}"
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

ts_ifa_complete() {
  local output="$1" expected_precomputed=false expected_learnable=false
  if is_true "$TRANSFORMED_EXPERT"; then expected_precomputed=true; fi
  if is_true "$LEARNABLE_TRANSFORMED_COVARIATE"; then expected_learnable=true; fi
    [ -s "$output/ts_ifa.pt" ] &&
    [ -s "$output/branches.pt" ] &&
    [ -s "$output/ridge_rooter.pt" ] &&
    [ -s "$output/training_history.json" ] &&
    [ -s "$output/eval_metrics.json" ] &&
    [ -s "$output/prediction_manifest.json" ] &&
    [ -s "$output/config.json" ] &&
    [ -s "$output/training_nmse.pdf" ] &&
    [ -s "$output/result_manifest.json" ] &&
    grep -Eq '"format"[[:space:]]*:[[:space:]]*"adaptation_ts_ifa_result"' \
      "$output/result_manifest.json" &&
    grep -Eq '"format"[[:space:]]*:[[:space:]]*"adaptation_prediction_store"' \
      "$output/prediction_manifest.json" &&
    grep -Eq '"neural_rooter_coefficients"[[:space:]]*:' \
      "$output/prediction_manifest.json" &&
    grep -Eq "\"precomputed_transformed_expert\"[[:space:]]*:[[:space:]]*$expected_precomputed" \
      "$output/config.json" &&
    grep -Eq "\"learnable_transformed_covariate\"[[:space:]]*:[[:space:]]*$expected_learnable" \
      "$output/config.json"
}

run_task() {
  local task_id="$1" task dataset model setting space metric neighbors
  local optional_args=() init_args=() transformed_args=()
  task="${TASKS[$task_id]}"
  IFS='|' read -r dataset model setting space metric neighbors <<< "$task"
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  RETRIEVAL_SETTING="${space}_${metric}_${neighbors}_${RETRIEVAL_MODE}"
  RUN_DIR="$OUT_ROOT/$dataset/${L}_${H}/$model/$RETRIEVAL_SETTING"
  INPUT_DIR="$RUN_DIR/extracted"
  RESULT_RUN_ROOT="$RESULTS_ROOT/$dataset/${L}_${H}/$model/$RETRIEVAL_SETTING"
  OUTPUT_DIR="$RESULT_RUN_ROOT/ts_ifa/TS-IFA"
  require_extraction "$INPUT_DIR"
  if is_true "$TRANSFORMED_EXPERT"; then
    if ! grep -Eq '"compute_transformed_prediction"[[:space:]]*:[[:space:]]*true' \
      "$INPUT_DIR/extraction_manifest.json"; then
      log_error "precomputed transformed expert requested; rerun extraction with COMPUTE_TRANSFORMED_PREDICTION=true for $INPUT_DIR"
      return 2
    fi
    transformed_args+=(--precomputed-transformed-expert)
  else
    transformed_args+=(--no-precomputed-transformed-expert)
  fi
  if is_true "$LEARNABLE_TRANSFORMED_COVARIATE"; then
    transformed_args+=(--learnable-transformed-covariate)
  else
    transformed_args+=(--no-learnable-transformed-covariate)
  fi
  VANILLA_SOURCE="$OUT_ROOT/$dataset/${L}_${H}/$model/vanilla/vanilla_metrics.json"
  VANILLA_TIMING_SOURCE="$OUT_ROOT/$dataset/${L}_${H}/$model/vanilla/extraction_timing.json"
  VANILLA_DEST="$RESULTS_ROOT/$dataset/${L}_${H}/$model/vanilla"
  assert_files vanilla-metrics "$VANILLA_SOURCE" "$VANILLA_TIMING_SOURCE" "$INPUT_DIR/extraction_timing.json"
  mkdir -p "$VANILLA_DEST"
  cp "$VANILLA_SOURCE" "$VANILLA_DEST/vanilla_metrics.json"
  cp "$VANILLA_TIMING_SOURCE" "$VANILLA_DEST/extraction_timing.json"
  mkdir -p "$RESULT_RUN_ROOT"
  cp "$INPUT_DIR/extraction_timing.json" "$RESULT_RUN_ROOT/extraction_timing.json"
  if is_true "$SKIP_COMPLETE" && ts_ifa_complete "$OUTPUT_DIR" &&
    [ "$OUTPUT_DIR/eval_metrics.json" -nt "$INPUT_DIR/extraction_manifest.json" ]; then
    log "skip complete family=ts_ifa dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
    return
  fi
  [ -z "$MAX_TRAIN_SAMPLES" ] || optional_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
  [ -z "$MAX_VALID_SAMPLES" ] || optional_args+=(--max-valid-samples "$MAX_VALID_SAMPLES")
  if is_true "$VANILLA_ANCHORING_INIT"; then
    init_args+=(--vanilla-anchoring-init)
  else
    init_args+=(--no-vanilla-anchoring-init)
  fi
  log_section "training start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING validation_fraction=$VALIDATION_FRACTION branch_epochs=$BRANCH_EPOCHS rooter_epochs=$ROOTER_EPOCHS batch_size=$BATCH_SIZE branch_lr=$BRANCH_LR rooter_lr=$ROOTER_LR weight_decay=$WEIGHT_DECAY vanilla_anchor=$VANILLA_ANCHOR coefficient_l2=$COEFFICIENT_L2 horizon_smoothness=$HORIZON_SMOOTHNESS ridge_rooter_alpha=$RIDGE_ROOTER_ALPHA dropout=$DROPOUT attention_heads=$ATTENTION_HEADS attention_dim=$ATTENTION_DIM hidden_dim=$HIDDEN_DIM transformed_expert=$TRANSFORMED_EXPERT learnable_transformed_covariate=$LEARNABLE_TRANSFORMED_COVARIATE transformed_hidden_dim=$TRANSFORMED_HIDDEN_DIM vanilla_anchoring_init=$VANILLA_ANCHORING_INIT seed=$SEED"
  srun --ntasks=1 python -m src.adaptors.ts_ifa.train \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --branch-epochs "$BRANCH_EPOCHS" \
    --rooter-epochs "$ROOTER_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --valid-eval-freq "$VALID_EVAL_FREQ" \
    --logging-eval-freq "$LOGGING_EVAL_FREQ" \
    --branch-lr "$BRANCH_LR" \
    --rooter-lr "$ROOTER_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --vanilla-anchor "$VANILLA_ANCHOR" \
    --coefficient-l2 "$COEFFICIENT_L2" \
    --horizon-smoothness "$HORIZON_SMOOTHNESS" \
    --ridge-rooter-alpha "$RIDGE_ROOTER_ALPHA" \
    --dropout "$DROPOUT" \
    --residual-heads "$ATTENTION_HEADS" \
    --memory-heads "$ATTENTION_HEADS" \
    --rooter-heads "$ATTENTION_HEADS" \
    --residual-attn-dim "$ATTENTION_DIM" \
    --memory-attn-dim "$ATTENTION_DIM" \
    --rooter-attn-dim "$ATTENTION_DIM" \
    --residual-hidden "$HIDDEN_DIM" \
    --memory-hidden "$HIDDEN_DIM" \
    --rooter-hidden "$HIDDEN_DIM" \
    --transformed-hidden "$TRANSFORMED_HIDDEN_DIM" \
    --normalization instance \
    --device gpu \
    --seed "$SEED" \
    "${optional_args[@]}" \
    "${transformed_args[@]}" \
    "${init_args[@]}"
  assert_files ts-ifa-output \
    "$OUTPUT_DIR/ts_ifa.pt" \
    "$OUTPUT_DIR/branches.pt" \
    "$OUTPUT_DIR/ridge_rooter.pt" \
    "$OUTPUT_DIR/training_history.json" \
    "$OUTPUT_DIR/eval_metrics.json" \
    "$OUTPUT_DIR/prediction_manifest.json" \
    "$OUTPUT_DIR/config.json" \
    "$OUTPUT_DIR/training_nmse.pdf" \
    "$OUTPUT_DIR/result_manifest.json"
  log "training done configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
}

log_section "job start kind=ts_ifa_training experiment_mode=$EXPERIMENT_MODE skip_complete=$SKIP_COMPLETE tasks=${#TASKS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV distance_spaces=$DISTANCE_SPACES_CSV distance_metrics=$DISTANCE_METRICS_CSV neighbors=$NEIGHBORS_CSV results_root=$RESULTS_ROOT"
for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
  run_task "$task_id"
done
log_section "job done kind=ts_ifa_training output=$RESULTS_ROOT"
