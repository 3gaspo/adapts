#!/bin/bash
# Enumerate and run adaptation extraction configurations sequentially.
# Submit ../../extraction.slurm; source this implementation only for local debugging.

set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

# On another machine, set DATA_ROOT and WEIGHTS_ROOT to the available resource
# directories or edit the candidate paths in common.sh.
: "${DATA_ROOT:=}"
: "${WEIGHTS_ROOT:=}"
: "${OUT_ROOT:=outputs/adaptation}"
: "${EXPERIMENT_MODE:=test}"
require_experiment_mode

: "${SKIP_COMPLETE:=true}"
adaptation_profile_defaults
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
DATASTORE_STRIDE="${DATASTORE_STRIDE:-$DEFAULT_DATASTORE_STRIDE}"
ADAPT_QUERY_STRIDE="${ADAPT_QUERY_STRIDE:-$DEFAULT_ADAPT_QUERY_STRIDE}"
EVAL_QUERY_STRIDE="${EVAL_QUERY_STRIDE:-$DEFAULT_EVAL_QUERY_STRIDE}"
MAX_STORE_WINDOWS="${MAX_STORE_WINDOWS:-$DEFAULT_MAX_STORE_WINDOWS}"
require_resolved_profile_grid

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$DISTANCE_METRICS_CSV" DISTANCE_METRICS
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS

SPLITS="${SPLITS:-0.3,0.5,0.2}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
PERIOD="${PERIOD:-24}"
SEED="${SEED:-1}"

model_kwargs() {
  local model="$1"
  local weight_path
  case "$model" in
    chronos2|chronos)
      weight_path="${CHRONOS2_WEIGHTS_PATH:-${CHRONOS_WEIGHTS_PATH:-}}"
      [ -n "$weight_path" ] || weight_path="$(find_weight_path chronos2)"
      printf '{"weights_path":"%s","device_map":"cuda","context_mode":"future_included"}\n' "$weight_path"
      ;;
    chronos_bolt|chronos-bolt)
      weight_path="${CHRONOS_BOLT_WEIGHTS_PATH:-}"
      [ -n "$weight_path" ] || weight_path="$(find_weight_path chronos-bolt-base)"
      printf '{"weights_path":"%s","device_map":"cuda"}\n' "$weight_path"
      ;;
    tabpfnts|tabpfn|tabpfn_ts)
      weight_path="${TABPFN_WEIGHTS_PATH:-}"
      [ -n "$weight_path" ] || weight_path="$(find_weight_path tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt)"
      printf '{"weights_path":"%s","device":"cuda","context_mode":"future_included"}\n' "$weight_path"
      ;;
    ts_icl)
      log_error "model ts_icl is reserved for a later implementation and is not registered"
      return 1
      ;;
    *)
      log_error "unknown extraction model=$model"
      return 1
      ;;
  esac
}

SKIP_ARGS=()
is_true "$SKIP_COMPLETE" && SKIP_ARGS+=(--skip-complete)

run_extraction() {
  local dataset="$1" model="$2" lags="$3" horizon="$4" neighbors="$5" space="$6" metric="$7" save_name="$8" output_root="$9"
  local dataset_dir config model_options
  local data_args=()
  dataset_dir="$(find_dataset_dir "$dataset")"
  config="$dataset_dir/config.json"
  [ ! -f "$config" ] || data_args+=(--dataset-config "$config")
  if [ "${dataset,,}" = etth1 ]; then data_args+=(--target-cols OT); fi
  model_options="$(model_kwargs "$model")"
  srun --ntasks=1 python -m src.experiments.extraction \
    --csv "$dataset_dir" \
    --dataset-name "$dataset" \
    "${data_args[@]}" \
    --lags "$lags" \
    --horizon "$horizon" \
    --splits "$SPLITS" \
    --datastore-stride "$DATASTORE_STRIDE" \
    --adapt-stride "$ADAPT_QUERY_STRIDE" \
    --eval-stride "$EVAL_QUERY_STRIDE" \
    --period "$PERIOD" \
    --neighbors "$neighbors" \
    --distance-space "$space" \
    --distance-metric "$metric" \
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
TASK_METRICS=()
TASK_NEIGHBORS=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      TASK_DATASETS+=("$dataset")
      TASK_MODELS+=("$model")
      TASK_SETTINGS+=("$setting")
      TASK_SPACES+=(raw)
      TASK_METRICS+=(euclidean)
      TASK_NEIGHBORS+=(0)
      for space in "${DISTANCE_SPACES[@]}"; do
        for metric in "${DISTANCE_METRICS[@]}"; do
          for neighbors in "${NEIGHBORS[@]}"; do
            TASK_DATASETS+=("$dataset")
            TASK_MODELS+=("$model")
            TASK_SETTINGS+=("$setting")
            TASK_SPACES+=("$space")
            TASK_METRICS+=("$metric")
            TASK_NEIGHBORS+=("$neighbors")
          done
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
  local metric="${TASK_METRICS[$task_id]}"
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
    retrieval_setting="${space}_${metric}_${neighbors}_${RETRIEVAL_MODE}"
    run_root="$MODEL_ROOT/$retrieval_setting"
  fi
  log_section "extraction start configuration=$((task_id + 1))/${#TASK_DATASETS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$retrieval_setting datastore_stride=$DATASTORE_STRIDE adapt_stride=$ADAPT_QUERY_STRIDE eval_stride=$EVAL_QUERY_STRIDE max_store_windows=$MAX_STORE_WINDOWS seed=$SEED"
  run_extraction "$dataset" "$model" "$L" "$H" "$neighbors" "$space" "$metric" "$save_name" "$run_root"
  log "extraction done configuration=$((task_id + 1))/${#TASK_DATASETS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$retrieval_setting"
}

log_section "job start kind=adaptation_extraction experiment_mode=$EXPERIMENT_MODE skip_complete=$SKIP_COMPLETE tasks=${#TASK_DATASETS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV distance_spaces=$DISTANCE_SPACES_CSV distance_metrics=$DISTANCE_METRICS_CSV neighbors=$NEIGHBORS_CSV"
for ((task_id = 0; task_id < ${#TASK_DATASETS[@]}; task_id++)); do
  run_task "$task_id"
done
log_section "job done kind=adaptation_extraction output=$OUT_ROOT"
