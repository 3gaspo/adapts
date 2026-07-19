#!/bin/bash
# Train and evaluate TS-IFA from completed extraction artifacts.
# Submit ../../ts_ifa.slurm; source this implementation only for local debugging.
set -euo pipefail
source src/slurm/common.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/adaptation}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
require_experiment_mode
case "$EXPERIMENT_MODE" in
  test)
    DEFAULT_DATASETS_CSV="electricity"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_SETTINGS_CSV="168:24"
    DEFAULT_DISTANCE_SPACES_CSV="raw"
    DEFAULT_NEIGHBORS_CSV="3"
    DEFAULT_EPOCHS=2
    DEFAULT_VALID_EVAL_FREQ=1
    DEFAULT_LOGGING_EVAL_FREQ=1
    DEFAULT_MAX_TRAIN_SAMPLES=32
    DEFAULT_MAX_VALID_SAMPLES=32
    DEFAULT_MAX_EVAL_SAMPLES=32
    DEFAULT_SKIP_COMPLETE=false
    ;;
  small)
    DEFAULT_DATASETS_CSV="electricity,solar"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_SETTINGS_CSV="168:24,672:168"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    DEFAULT_EPOCHS=10000
    DEFAULT_VALID_EVAL_FREQ=1000
    DEFAULT_LOGGING_EVAL_FREQ=1000
    DEFAULT_MAX_TRAIN_SAMPLES=""
    DEFAULT_MAX_VALID_SAMPLES=""
    DEFAULT_MAX_EVAL_SAMPLES=""
    DEFAULT_SKIP_COMPLETE=true
    ;;
  large)
    DEFAULT_DATASETS_CSV="electricity,solar"
    DEFAULT_MODELS_CSV="chronos,tabpfnts"
    DEFAULT_SETTINGS_CSV="168:24,672:168"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    DEFAULT_EPOCHS=10000
    DEFAULT_VALID_EVAL_FREQ=1000
    DEFAULT_LOGGING_EVAL_FREQ=1000
    DEFAULT_MAX_TRAIN_SAMPLES=""
    DEFAULT_MAX_VALID_SAMPLES=""
    DEFAULT_MAX_EVAL_SAMPLES=""
    DEFAULT_SKIP_COMPLETE=true
    ;;
esac
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
VALID_EVAL_FREQ="${VALID_EVAL_FREQ:-$DEFAULT_VALID_EVAL_FREQ}"
LOGGING_EVAL_FREQ="${LOGGING_EVAL_FREQ:-$DEFAULT_LOGGING_EVAL_FREQ}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES-$DEFAULT_MAX_TRAIN_SAMPLES}"
MAX_VALID_SAMPLES="${MAX_VALID_SAMPLES-$DEFAULT_MAX_VALID_SAMPLES}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES-$DEFAULT_MAX_EVAL_SAMPLES}"
SKIP_COMPLETE="${SKIP_COMPLETE:-$DEFAULT_SKIP_COMPLETE}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
SEED="${SEED:-1}"

BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.00001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
BETA="${BETA:-0.01}"
GAMMA="${GAMMA:-0.01}"
DROPOUT="${DROPOUT:-0.0}"
ATTENTION_HEADS="${ATTENTION_HEADS:-4}"
ATTENTION_DIM="${ATTENTION_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
MIXTURE_GATE_INIT="${MIXTURE_GATE_INIT:--6.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
RESTORE_BEST_VALIDATION="${RESTORE_BEST_VALIDATION:-true}"

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS

TASKS=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for space in "${DISTANCE_SPACES[@]}"; do
        for neighbors in "${NEIGHBORS[@]}"; do
          TASKS+=("$dataset|$model|$setting|$space|$neighbors")
        done
      done
    done
  done
done

ts_ifa_complete() {
  local output="$1"
  [ -s "$output/ts_ifa.pt" ] &&
    [ -s "$output/training_history.json" ] &&
    [ -s "$output/eval_metrics.json" ] &&
    [ -s "$output/eval_predictions.pt" ] &&
    [ -s "$output/config.json" ] &&
    [ -s "$output/training_nmse.pdf" ]
}

run_task() {
  local task_id="$1" task dataset model setting space neighbors
  local optional_args=() restore_args=()
  task="${TASKS[$task_id]}"
  IFS='|' read -r dataset model setting space neighbors <<< "$task"
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  RETRIEVAL_SETTING="${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
  RUN_DIR="$OUT_ROOT/$dataset/${L}_${H}/$model/$RETRIEVAL_SETTING"
  INPUT_DIR="$RUN_DIR/extracted"
  OUTPUT_DIR="$RUN_DIR/ts_ifa/TS-IFA"
  require_extraction "$INPUT_DIR"
  if is_true "$SKIP_COMPLETE" && ts_ifa_complete "$OUTPUT_DIR" &&
    [ "$OUTPUT_DIR/eval_metrics.json" -nt "$INPUT_DIR/extraction_manifest.json" ]; then
    log "skip complete family=ts_ifa dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
    return
  fi
  [ -z "$MAX_TRAIN_SAMPLES" ] || optional_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
  [ -z "$MAX_VALID_SAMPLES" ] || optional_args+=(--max-valid-samples "$MAX_VALID_SAMPLES")
  [ -z "$MAX_EVAL_SAMPLES" ] || optional_args+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
  is_true "$RESTORE_BEST_VALIDATION" && restore_args+=(--restore-best-validation)
  log_section "training start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING epochs=$EPOCHS batch_size=$BATCH_SIZE learning_rate=$LR weight_decay=$WEIGHT_DECAY beta=$BETA gamma=$GAMMA dropout=$DROPOUT attention_heads=$ATTENTION_HEADS attention_dim=$ATTENTION_DIM hidden_dim=$HIDDEN_DIM mixture_gate_init=$MIXTURE_GATE_INIT seed=$SEED"
  srun --ntasks=1 python -m src.adaptors.ts_ifa.train \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --valid-eval-freq "$VALID_EVAL_FREQ" \
    --logging-eval-freq "$LOGGING_EVAL_FREQ" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --beta "$BETA" \
    --gamma "$GAMMA" \
    --dropout "$DROPOUT" \
    --residual-heads "$ATTENTION_HEADS" \
    --memory-heads "$ATTENTION_HEADS" \
    --mixture-heads "$ATTENTION_HEADS" \
    --residual-attn-dim "$ATTENTION_DIM" \
    --memory-attn-dim "$ATTENTION_DIM" \
    --mixture-attn-dim "$ATTENTION_DIM" \
    --residual-hidden "$HIDDEN_DIM" \
    --memory-hidden "$HIDDEN_DIM" \
    --mixture-hidden "$HIDDEN_DIM" \
    --mixture-gate-init "$MIXTURE_GATE_INIT" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --early-stopping-min-delta "$EARLY_STOPPING_MIN_DELTA" \
    --normalization instance \
    --device gpu \
    --seed "$SEED" \
    "${optional_args[@]}" \
    "${restore_args[@]}"
  assert_files ts-ifa-output \
    "$OUTPUT_DIR/ts_ifa.pt" \
    "$OUTPUT_DIR/training_history.json" \
    "$OUTPUT_DIR/eval_metrics.json" \
    "$OUTPUT_DIR/eval_predictions.pt" \
    "$OUTPUT_DIR/config.json" \
    "$OUTPUT_DIR/training_nmse.pdf"
  log "training done configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
}

log_section "job start kind=ts_ifa_training experiment_mode=$EXPERIMENT_MODE skip_complete=$SKIP_COMPLETE tasks=${#TASKS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV distance_spaces=$DISTANCE_SPACES_CSV neighbors=$NEIGHBORS_CSV"
for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
  run_task "$task_id"
done
log_section "job done kind=ts_ifa_training output=$OUT_ROOT"
