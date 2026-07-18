#!/bin/bash
# Generic extraction grid used by the pilot and multi-backbone Slurm wrappers.

set -euo pipefail
source src/slurm/common.sh
require_project_root
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

# When this repository is run outside the thesis workspace, set DATA_ROOT and
# WEIGHTS_ROOT here or export them before sbatch. Otherwise project-local,
# parent-shared, then thesis-root shared folders are searched in that order.
: "${DATA_ROOT:=}"
: "${WEIGHTS_ROOT:=}"
: "${OUT_ROOT:=outputs/adaptation}"
: "${TEST_MODE:=false}"
: "${SKIP_COMPLETE:=true}"

if is_true "$TEST_MODE"; then
  DATASETS_CSV="${DATASETS_CSV:-electricity}"
  MODELS_CSV="${MODELS_CSV:-chronos}"
  SETTINGS_CSV="${SETTINGS_CSV:-168:24}"
  DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-raw}"
  NEIGHBORS_CSV="${NEIGHBORS_CSV:-3}"
  DATASTORE_STRIDE="${DATASTORE_STRIDE:-168}"
  TRAIN_QUERY_STRIDE="${TRAIN_QUERY_STRIDE:-256}"
  ORACLE_QUERY_STRIDE="${ORACLE_QUERY_STRIDE:-256}"
  EVAL_QUERY_STRIDE="${EVAL_QUERY_STRIDE:-256}"
  MAX_STORE_WINDOWS="${MAX_STORE_WINDOWS:-2048}"
else
  DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
  MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
  SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
  DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-raw,instance}"
  NEIGHBORS_CSV="${NEIGHBORS_CSV:-1,3,10}"
  DATASTORE_STRIDE="${DATASTORE_STRIDE:-24}"
  TRAIN_QUERY_STRIDE="${TRAIN_QUERY_STRIDE:-24}"
  ORACLE_QUERY_STRIDE="${ORACLE_QUERY_STRIDE:-24}"
  EVAL_QUERY_STRIDE="${EVAL_QUERY_STRIDE:-128}"
  MAX_STORE_WINDOWS="${MAX_STORE_WINDOWS:-30000}"
fi

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS

SPLITS="${SPLITS:-0.3,0.35,0.15,0.2}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
PERIOD="${PERIOD:-24}"
SEED="${SEED:-1}"

model_kwargs() {
  local model="$1"
  local weight_path
  case "$model" in
    chronos)
      weight_path="${CHRONOS_WEIGHTS_PATH:-}"
      [ -n "$weight_path" ] || weight_path="$(find_weight_path chronos2)"
      printf '{"weights_path":"%s","device_map":"cuda","context_mode":"future_included"}\n' "$weight_path"
      ;;
    tabpfnts|tabpfn|tabpfn_ts)
      weight_path="${TABPFN_WEIGHTS_PATH:-}"
      [ -n "$weight_path" ] || weight_path="$(find_weight_path tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt)"
      printf '{"weights_path":"%s","device":"cuda","context_mode":"future_included"}\n' "$weight_path"
      ;;
    ts_icl)
      echo "$(date -Is) model ts_icl is reserved for the later implementation and is not registered" >&2
      return 1
      ;;
    *)
      echo "$(date -Is) unknown extraction model=$model" >&2
      return 1
      ;;
  esac
}

SKIP_ARGS=()
is_true "$SKIP_COMPLETE" && SKIP_ARGS+=(--skip-complete)

run_extraction() {
  local dataset="$1" model="$2" lags="$3" horizon="$4" neighbors="$5" space="$6" save_name="$7" output_root="$8"
  local dataset_dir config model_options
  local data_args=()
  dataset_dir="$(find_dataset_dir "$dataset")"
  config="$dataset_dir/config.json"
  [ ! -f "$config" ] || data_args+=(--dataset-config "$config")
  model_options="$(model_kwargs "$model")"
  srun python -m src.experiments.extraction \
    --csv "$dataset_dir" \
    --dataset-name "$dataset" \
    "${data_args[@]}" \
    --lags "$lags" \
    --horizon "$horizon" \
    --splits "$SPLITS" \
    --datastore-stride "$DATASTORE_STRIDE" \
    --train-stride "$TRAIN_QUERY_STRIDE" \
    --oracle-stride "$ORACLE_QUERY_STRIDE" \
    --eval-stride "$EVAL_QUERY_STRIDE" \
    --period "$PERIOD" \
    --neighbors "$neighbors" \
    --distance-space "$space" \
    --distance-metric euclidean \
    --max-store-windows "$MAX_STORE_WINDOWS" \
    --retrieval-mode "$RETRIEVAL_MODE" \
    --model "$model" \
    --model-kwargs "$model_options" \
    --normalization instance \
    --device gpu \
    --output-dir "$output_root" \
    --save-name "$save_name" \
    --seed "$SEED" \
    "${SKIP_ARGS[@]}"
}

TASK_DATASETS=()
TASK_MODELS=()
TASK_SETTINGS=()
TASK_SPACES=()
TASK_NEIGHBORS=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      TASK_DATASETS+=("$dataset")
      TASK_MODELS+=("$model")
      TASK_SETTINGS+=("$setting")
      TASK_SPACES+=(raw)
      TASK_NEIGHBORS+=(0)
      for space in "${DISTANCE_SPACES[@]}"; do
        for neighbors in "${NEIGHBORS[@]}"; do
          TASK_DATASETS+=("$dataset")
          TASK_MODELS+=("$model")
          TASK_SETTINGS+=("$setting")
          TASK_SPACES+=("$space")
          TASK_NEIGHBORS+=("$neighbors")
        done
      done
    done
  done
done

run_task() {
  local task_id="$1"
  local dataset="${TASK_DATASETS[$task_id]}"
  local model="${TASK_MODELS[$task_id]}"
  local setting="${TASK_SETTINGS[$task_id]}"
  local space="${TASK_SPACES[$task_id]}"
  local neighbors="${TASK_NEIGHBORS[$task_id]}"
  local save_name run_root retrieval_setting
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  MODEL_ROOT="$OUT_ROOT/$dataset/${L}_${H}/$model"
  # Resolve before loading a multi-GB model so a missing dataset fails promptly.
  find_dataset_dir "$dataset" >/dev/null
  if [ "$neighbors" -eq 0 ]; then
    save_name=vanilla
    run_root="$MODEL_ROOT"
    retrieval_setting=vanilla
  else
    save_name=extracted
    retrieval_setting="${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
    run_root="$MODEL_ROOT/$retrieval_setting"
  fi
  echo "$(date -Is) extraction start task=$task_id/${#TASK_DATASETS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$retrieval_setting"
  run_extraction "$dataset" "$model" "$L" "$H" "$neighbors" "$space" "$save_name" "$run_root"
  echo "$(date -Is) extraction done task=$task_id dataset=$dataset model=$model lags=$L horizon=$H retrieval=$retrieval_setting"
}

echo "$(date -Is) job start kind=adaptation_extraction test_mode=$TEST_MODE tasks=${#TASK_DATASETS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASK_DATASETS[@]}" ]; then
    echo "$(date -Is) array task outside narrowed grid; exiting task=$SLURM_ARRAY_TASK_ID tasks=${#TASK_DATASETS[@]}"
    exit 0
  fi
  run_task "$SLURM_ARRAY_TASK_ID"
else
  for ((task_id = 0; task_id < ${#TASK_DATASETS[@]}; task_id++)); do
    run_task "$task_id"
  done
fi
echo "$(date -Is) job done kind=adaptation_extraction output=$OUT_ROOT"
