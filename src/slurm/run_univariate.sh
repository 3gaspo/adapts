#!/bin/bash
# Run direct univariate backbone forecasts used as non-adapted references.
# Submit run_univariate.slurm; source this implementation only for local debugging.
set -euo pipefail
source src/slurm/common.sh
require_project_root
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# Set DATA_ROOT/WEIGHTS_ROOT on another machine or edit candidates in common.sh.
: "${DATA_ROOT:=}"
: "${WEIGHTS_ROOT:=}"
OUT_ROOT="${OUT_ROOT:-outputs/univariate}"
TEST_MODE="${TEST_MODE:-false}"
if is_true "$TEST_MODE"; then
  DATASETS_CSV="${DATASETS_CSV:-electricity}"
  SETTINGS_CSV="${SETTINGS_CSV:-168:24}"
  EVAL_QUERY_STRIDE="${EVAL_QUERY_STRIDE:-256}"
else
  DATASETS_CSV="${DATASETS_CSV:-electricity,solar}"
  SETTINGS_CSV="${SETTINGS_CSV:-168:24,672:168}"
  EVAL_QUERY_STRIDE="${EVAL_QUERY_STRIDE:-128}"
fi
SPLITS="${SPLITS:-0.3,0.35,0.15,0.2}"
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
  SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"
  log_section "univariate start task=$task_id/${#TASKS[@]} dataset=$dataset lags=$L horizon=$H model=chronos eval_stride=$EVAL_QUERY_STRIDE normalization=instance seed=$SEED"
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
  log "univariate done task=$task_id dataset=$dataset lags=$L horizon=$H model=chronos"
}

log_section "job start kind=univariate test_mode=$TEST_MODE tasks=${#TASKS[@]} datasets=$DATASETS_CSV settings=$SETTINGS_CSV"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
    log "array task outside narrowed sweep; exiting task=$SLURM_ARRAY_TASK_ID tasks=${#TASKS[@]}"
    exit 0
  fi
  run_task "$SLURM_ARRAY_TASK_ID"
else
  for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
    run_task "$task_id"
  done
fi
log_section "job done kind=univariate output=$OUT_ROOT"
