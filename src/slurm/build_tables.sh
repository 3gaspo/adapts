#!/bin/bash
# Build held-out and equal-configuration-average tables from completed sweeps.
# Submit ../../tables.slurm; source this implementation only for local debugging.

set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/extractions}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation_results/$EXPERIMENT_MODE}"
require_experiment_mode
adaptation_profile_defaults
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
DISTANCE_METRICS_CSV="${DISTANCE_METRICS_CSV:-$DEFAULT_DISTANCE_METRICS_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
if [ "$EXPERIMENT_MODE" = crossrag ]; then
  DEFAULT_FAMILIES_CSV=comparison
else
  DEFAULT_FAMILIES_CSV=baselines,gates
fi
FAMILIES_CSV="${FAMILIES_CSV:-$DEFAULT_FAMILIES_CSV}"
TABLE_KINDS_CSV="${TABLE_KINDS_CSV:-full,average}"
PROFILE_METHODS_CSV=""
case "$EXPERIMENT_MODE" in
  test|screen)
    if [ "$FAMILIES_CSV" = baselines ]; then
      PROFILE_METHODS_CSV="$PRIMARY_BASELINE_METHODS_CSV"
    elif [ "$FAMILIES_CSV" = gates ]; then
      PROFILE_METHODS_CSV="$PRIMARY_GATE_METHODS_CSV"
    elif [ "$FAMILIES_CSV" = baselines,gates ]; then
      PROFILE_METHODS_CSV="$PRIMARY_BASELINE_METHODS_CSV,$PRIMARY_GATE_METHODS_CSV"
    fi
    ;;
esac
if [ -n "${BASELINE_METHODS_CSV:-}" ] && [ -n "${GATE_METHODS_CSV:-}" ]; then
  DEFAULT_METHODS_CSV="${BASELINE_METHODS_CSV},${GATE_METHODS_CSV}"
else
  DEFAULT_METHODS_CSV="${BASELINE_METHODS_CSV:-${GATE_METHODS_CSV:-$PROFILE_METHODS_CSV}}"
fi
METHODS_CSV="${METHODS_CSV:-$DEFAULT_METHODS_CSV}"
PIPELINES_CSV="${PIPELINES_CSV:-}"
METRIC="${METRIC:-nmse}"
DECIMALS="${DECIMALS:-2}"
CANDIDATE_FAMILY="${CANDIDATE_FAMILY:-}"
if [ "$EXPERIMENT_MODE" = crossrag ] && [ -z "$CANDIDATE_FAMILY" ]; then
  if [ -n "${BASELINE_METHODS_CSV:-}" ] && [ -z "${GATE_METHODS_CSV:-}" ]; then
    CANDIDATE_FAMILY=baselines
  elif [ -n "${GATE_METHODS_CSV:-}" ] && [ -z "${BASELINE_METHODS_CSV:-}" ]; then
    CANDIDATE_FAMILY=gates
  else
    log_error "crossrag tables require CANDIDATE_FAMILY=baselines|gates"
    return 2
  fi
fi
require_resolved_profile_grid
if requires_selected_methods && [ -z "$METHODS_CSV" ] && [ "$FAMILIES_CSV" != ts_ifa ]; then
  log_error "EXPERIMENT_MODE=$EXPERIMENT_MODE requires METHODS_CSV for candidate-only tables"
  return 2
fi

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$DISTANCE_METRICS_CSV" DISTANCE_METRICS
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
METRIC_ARG="$(join_csv "${DISTANCE_METRICS[@]}")"
NEIGHBOR_ARG="$(join_csv "${NEIGHBORS[@]}")"
FAMILY_ARG="$(join_csv "${FAMILIES[@]}")"
METHOD_ARGS=()
[ -z "$METHODS_CSV" ] || METHOD_ARGS+=(--variants "$METHODS_CSV")
PIPELINE_ARGS=()
[ -z "$PIPELINES_CSV" ] || PIPELINE_ARGS+=(--pipelines "$PIPELINES_CSV")

log_section "job start kind=adaptation_tables experiment_mode=$EXPERIMENT_MODE datasets=$DATASET_ARG models=$MODELS_CSV settings=$SETTING_ARG families=$FAMILY_ARG metric=$METRIC distance_metrics=$METRIC_ARG methods=${METHODS_CSV:-all} table_kinds=$TABLE_KINDS_CSV results_root=$RESULTS_ROOT"
for model in "${MODELS[@]}"; do
  # Fail instead of silently averaging an incomplete sweep.
  for dataset in "${DATASETS[@]}"; do
    for setting in "${SETTING_NAMES[@]}"; do
      VANILLA_ROOT="$RESULTS_ROOT/$dataset/$setting/$model/vanilla"
      assert_files table-input "$VANILLA_ROOT/vanilla_metrics.json"
      if [ -z "$PIPELINES_CSV" ]; then
      for space in "${DISTANCE_SPACES[@]}"; do
        for distance_metric in "${DISTANCE_METRICS[@]}"; do
          for neighbors in "${NEIGHBORS[@]}"; do
            RUN_ROOT="$RESULTS_ROOT/$dataset/$setting/$model/${space}_${distance_metric}_${neighbors}_${RETRIEVAL_MODE}"
            for family in "${FAMILIES[@]}"; do
              case "$family" in
                baselines) assert_files table-input "$RUN_ROOT/baselines/baseline_metrics.json" ;;
                gates) assert_files table-input "$RUN_ROOT/gates/gate_metrics.json" ;;
                ts_ifa) assert_files table-input "$RUN_ROOT/ts_ifa/TS-IFA/eval_metrics.json" ;;
                crossrag) assert_files table-input "$RUN_ROOT/crossrag/crossrag_metrics.json" ;;
                comparison)
                  assert_files table-input "$RUN_ROOT/crossrag/crossrag_metrics.json"
                  if [ -n "${BASELINE_METHODS_CSV:-}" ]; then
                    assert_files table-input "$RUN_ROOT/baselines/baseline_metrics.json"
                  fi
                  if [ -n "${GATE_METHODS_CSV:-}" ]; then
                    assert_files table-input "$RUN_ROOT/gates/gate_metrics.json"
                  fi
                  ;;
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
      fi
    done
  done

  for table_kind in "${TABLE_KINDS[@]}"; do
    OUTPUT_DIR="$RESULTS_ROOT/tables/$model/$table_kind"
    log_section "table start model=$model kind=$table_kind metric=$METRIC split=eval decimals=$DECIMALS output=$OUTPUT_DIR"
    srun --ntasks=1 python -m src.visu.sweep_results_table \
      "$RESULTS_ROOT" \
      --table-kind "$table_kind" \
      --output-dir "$OUTPUT_DIR" \
      --metric "$METRIC" \
      --split eval \
      --datasets "$DATASET_ARG" \
      --settings "$SETTING_ARG" \
      --models "$model" \
      --families "$FAMILY_ARG" \
      --spaces "$SPACE_ARG" \
      --distance-metrics "$METRIC_ARG" \
      --neighbors "$NEIGHBOR_ARG" \
      --retrieval-mode "$RETRIEVAL_MODE" \
      --decimals "$DECIMALS" \
      "${METHOD_ARGS[@]}" \
      "${PIPELINE_ARGS[@]}"
    for family in "${FAMILIES[@]}"; do
      assert_files table-output "$OUTPUT_DIR/${family}_results.tex"
    done
    if [ "$table_kind" = average ]; then
      for family in "${FAMILIES[@]}"; do
        case "$family" in
          baselines|gates|full|comparison)
            assert_files table-output "$OUTPUT_DIR/positive_windows_results.tex"
            break
            ;;
        esac
      done
    fi
    log "table done model=$model kind=$table_kind output=$OUTPUT_DIR"
  done
done
if [ "$EXPERIMENT_MODE" = crossrag ]; then
  TIMING_OUTPUT="$RESULTS_ROOT/tables/chronos-bolt/timing_comparison.tex"
  srun --ntasks=1 python -m src.visu.timing_table \
    "$RESULTS_ROOT" \
    --output "$TIMING_OUTPUT" \
    --datasets "$DATASET_ARG" \
    --setting 512_64 \
    --model chronos-bolt \
    --candidate-run "$CANDIDATE_RUN" \
    --crossrag-run minmax_cosine_15_online \
    --candidate-family "$CANDIDATE_FAMILY"
  assert_files table-output "$TIMING_OUTPUT"
fi
log_section "job done kind=adaptation_tables output=$RESULTS_ROOT/tables"
