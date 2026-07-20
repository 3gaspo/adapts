#!/bin/bash
# Build held-out and equal-configuration-average tables from completed sweeps.
# Submit ../../tables.slurm; source this implementation only for local debugging.

set -euo pipefail
source src/slurm/common.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/adaptation}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
require_experiment_mode
DEFAULT_SMALL_DATASETS_CSV="Traffic,Electricity,Solar"
DEFAULT_FULL_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,Exchange"
DEFAULT_SMALL_SETTINGS_CSV="168:24,504:24,504:168,504:504"
DEFAULT_FULL_SETTINGS_CSV="$DEFAULT_SMALL_SETTINGS_CSV,512:64"
case "$EXPERIMENT_MODE" in
  test)
    DEFAULT_PROFILE_DATASETS_CSV="electricity"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_PROFILE_SETTINGS_CSV="168:24"
    DEFAULT_DISTANCE_SPACES_CSV="raw"
    DEFAULT_NEIGHBORS_CSV="3"
    ;;
  small)
    DEFAULT_PROFILE_DATASETS_CSV="$DEFAULT_SMALL_DATASETS_CSV"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_PROFILE_SETTINGS_CSV="$DEFAULT_SMALL_SETTINGS_CSV"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    ;;
  full|large)
    DEFAULT_PROFILE_DATASETS_CSV="$DEFAULT_FULL_DATASETS_CSV"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_PROFILE_SETTINGS_CSV="$DEFAULT_FULL_SETTINGS_CSV"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    ;;
  ultra)
    DEFAULT_PROFILE_DATASETS_CSV="$DEFAULT_FULL_DATASETS_CSV"
    DEFAULT_MODELS_CSV="chronos,tabpfnts"
    DEFAULT_PROFILE_SETTINGS_CSV="$DEFAULT_FULL_SETTINGS_CSV"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    ;;
esac
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_PROFILE_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_PROFILE_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
FAMILIES_CSV="${FAMILIES_CSV:-baselines,gates}"
TABLE_KINDS_CSV="${TABLE_KINDS_CSV:-full,average}"
METRIC="${METRIC:-nmse}"
DECIMALS="${DECIMALS:-2}"

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS
csv_to_array "$FAMILIES_CSV" FAMILIES
csv_to_array "$TABLE_KINDS_CSV" TABLE_KINDS

SETTING_NAMES=()
for setting in "${SETTINGS[@]}"; do
  parse_setting "$setting"
  SETTING_NAMES+=("${SETTING_LAGS}_${SETTING_HORIZON}")
done

join_csv() {
  local IFS=,
  echo "$*"
}

DATASET_ARG="$(join_csv "${DATASETS[@]}")"
SETTING_ARG="$(join_csv "${SETTING_NAMES[@]}")"
SPACE_ARG="$(join_csv "${DISTANCE_SPACES[@]}")"
NEIGHBOR_ARG="$(join_csv "${NEIGHBORS[@]}")"
FAMILY_ARG="$(join_csv "${FAMILIES[@]}")"

log_section "job start kind=adaptation_tables experiment_mode=$EXPERIMENT_MODE datasets=$DATASET_ARG models=$MODELS_CSV settings=$SETTING_ARG families=$FAMILY_ARG metric=$METRIC table_kinds=$TABLE_KINDS_CSV"
for model in "${MODELS[@]}"; do
  # Fail instead of silently averaging an incomplete sweep.
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTING_NAMES[@]}"; do
      VANILLA_ROOT="$OUT_ROOT/$dataset/$setting/$model/vanilla"
      require_extraction "$VANILLA_ROOT"
      assert_files table-input "$VANILLA_ROOT/vanilla_metrics.json"
      for space in "${DISTANCE_SPACES[@]}"; do
        for neighbors in "${NEIGHBORS[@]}"; do
          RUN_ROOT="$OUT_ROOT/$dataset/$setting/$model/${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
          for family in "${FAMILIES[@]}"; do
            case "$family" in
              baselines) assert_files table-input "$RUN_ROOT/baselines/baseline_metrics.json" ;;
              gates) assert_files table-input "$RUN_ROOT/gates/gate_metrics.json" ;;
              ts_ifa) assert_files table-input "$RUN_ROOT/ts_ifa/TS-IFA/eval_metrics.json" ;;
              full)
                assert_files table-input \
                  "$RUN_ROOT/baselines/baseline_metrics.json" \
                  "$RUN_ROOT/gates/gate_metrics.json" \
                  "$RUN_ROOT/ts_ifa/TS-IFA/eval_metrics.json"
                ;;
              *) log_error "unknown table family=$family"; exit 1 ;;
            esac
          done
        done
      done
    done
  done

  for table_kind in "${TABLE_KINDS[@]}"; do
    OUTPUT_DIR="$OUT_ROOT/tables/$model/$table_kind"
    log_section "table start model=$model kind=$table_kind metric=$METRIC split=eval decimals=$DECIMALS output=$OUTPUT_DIR"
    srun --ntasks=1 python -m src.visu.sweep_results_table \
      "$OUT_ROOT" \
      --table-kind "$table_kind" \
      --output-dir "$OUTPUT_DIR" \
      --metric "$METRIC" \
      --split eval \
      --datasets "$DATASET_ARG" \
      --settings "$SETTING_ARG" \
      --models "$model" \
      --families "$FAMILY_ARG" \
      --spaces "$SPACE_ARG" \
      --neighbors "$NEIGHBOR_ARG" \
      --retrieval-mode "$RETRIEVAL_MODE" \
      --decimals "$DECIMALS"
    for family in "${FAMILIES[@]}"; do
      assert_files table-output "$OUTPUT_DIR/${family}_results.tex"
    done
    log "table done model=$model kind=$table_kind output=$OUTPUT_DIR"
  done
done
log_section "job done kind=adaptation_tables output=$OUT_ROOT/tables"
