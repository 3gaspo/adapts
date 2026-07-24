#!/bin/bash
# Run direct univariate backbone forecasts used as non-adapted references.
# Submit ../../univariate.slurm; source this implementation only for local debugging.
set -euo pipefail
source src/slurm/common.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

# Set DATA_ROOT/WEIGHTS_ROOT on another machine or edit candidates in common.sh.
: "${DATA_ROOT:=}"
: "${WEIGHTS_ROOT:=}"
OUT_ROOT="${OUT_ROOT:-outputs/univariate}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
require_experiment_mode
case "$EXPERIMENT_MODE" in
  test)
    DEFAULT_DATASETS_CSV="Electricity"
    DEFAULT_SETTINGS_CSV="168:24"
    DEFAULT_EVAL_QUERY_STRIDE=256
    ;;
  small)
    DEFAULT_DATASETS_CSV="Traffic,Electricity,Solar"
    DEFAULT_SETTINGS_CSV="168:24,504:24,504:168,504:504"
    DEFAULT_EVAL_QUERY_STRIDE=128
    ;;
  full|large|ultra)
    DEFAULT_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,Exchange"
    DEFAULT_SETTINGS_CSV="168:24,504:24,504:168,504:504,512:64"
    DEFAULT_EVAL_QUERY_STRIDE=128
    ;;
esac
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
EVAL_QUERY_STRIDE="${EVAL_QUERY_STRIDE:-$DEFAULT_EVAL_QUERY_STRIDE}"
SPLITS="${SPLITS:-0.3,0.5,0.2}"
SEED="${SEED:-1}"
CHRONOS_WEIGHTS_PATH="${CHRONOS_WEIGHTS_PATH:-}"
[ -n "$CHRONOS_WEIGHTS_PATH" ] || CHRONOS_WEIGHTS_PATH="$(find_weight_path chronos2)"
CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future_included\"}"

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$SETTINGS_CSV" SETTINGS
TASKS=()
for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    TASKS+=("$dataset|$setting")
  done
done

run_task() {
  local task_id="$1" dataset setting dataset_dir config
  local data_args=()
  IFS='|' read -r dataset setting <<< "${TASKS[$task_id]}"
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  dataset_dir="$(find_dataset_dir "$dataset")"
  config="$dataset_dir/config.json"
  [ ! -f "$config" ] || data_args+=(--dataset-config "$config")
  if [ "${dataset,,}" = etth1 ]; then data_args+=(--target-cols OT); fi
  SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"
  log_section "univariate start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset lags=$L horizon=$H model=chronos eval_stride=$EVAL_QUERY_STRIDE normalization=instance seed=$SEED"
  srun --ntasks=1 python -m src.experiments.experiment_univariate \
    --csv "$dataset_dir" \
    --dataset-name "$dataset" \
    "${data_args[@]}" \
    --lags "$L" \
    --horizon "$H" \
    --splits "$SPLITS" \
    --eval-stride "$EVAL_QUERY_STRIDE" \
    --model chronos \
    --model-kwargs "$CHRONOS_KWARGS" \
    --normalization instance \
    --device gpu \
    --output-dir "$SETTING_OUT" \
    --save-name chronos \
    --seed "$SEED"
  assert_files univariate-output \
    "$SETTING_OUT/chronos/univariate_losses.csv" \
    "$SETTING_OUT/chronos/univariate_summary.json" \
    "$SETTING_OUT/chronos/univariate_payload.pt"
  log "univariate done configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset lags=$L horizon=$H model=chronos"
}

log_section "job start kind=univariate experiment_mode=$EXPERIMENT_MODE tasks=${#TASKS[@]} datasets=$DATASETS_CSV settings=$SETTINGS_CSV"
for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
  run_task "$task_id"
done
log_section "job done kind=univariate output=$OUT_ROOT"
