#!/bin/bash
# Run one complete experiment profile inside one sequential Slurm allocation.
set -euo pipefail
source src/slurm/common.sh
source src/slurm/profiles.sh
require_project_root

if [ -z "${EXPERIMENT_MODE:-}" ]; then
  log_error "EXPERIMENT_MODE must be set by the root Slurm front"
  return 2
fi
require_experiment_mode
adaptation_profile_defaults

PROFILE_DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_DATASETS_CSV}"
PROFILE_MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
PROFILE_SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_SETTINGS_CSV}"
PROFILE_K_CSV="$DEFAULT_NEIGHBORS_CSV"
WINNERS_CSV="${WINNERS_CSV:-}"
EXTRACTION_SKIP_COMPLETE="${EXTRACTION_SKIP_COMPLETE:-true}"

run_screen() {
  DATASETS_CSV="$PROFILE_DATASETS_CSV"
  MODELS_CSV="$PROFILE_MODELS_CSV"
  SETTINGS_CSV="$PROFILE_SETTINGS_CSV"
  DISTANCE_SPACES_CSV="$DEFAULT_DISTANCE_SPACES_CSV"
  DISTANCE_METRICS_CSV="$DEFAULT_DISTANCE_METRICS_CSV"
  NEIGHBORS_CSV="$DEFAULT_NEIGHBORS_CSV"
  BASELINE_METHODS_CSV="$PRIMARY_BASELINE_METHODS_CSV"
  GATE_METHODS_CSV="$PRIMARY_GATE_METHODS_CSV"
  SKIP_COMPLETE="$EXTRACTION_SKIP_COMPLETE"
  source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
  SKIP_COMPLETE=true
  source "$PROJECT_ROOT/src/slurm/run_baselines.sh"
  source "$PROJECT_ROOT/src/slurm/run_gates.sh"
  FAMILIES_CSV=baselines,gates
  METHODS_CSV="$PRIMARY_BASELINE_METHODS_CSV,$PRIMARY_GATE_METHODS_CSV"
  PIPELINES_CSV=""
  source "$PROJECT_ROOT/src/slurm/build_tables.sh"
}

append_unique() {
  local -n array_ref="$1"
  local value="$2" existing
  for existing in "${array_ref[@]:-}"; do
    [ "$existing" != "$value" ] || return 0
  done
  array_ref+=("$value")
}

append_method() {
  local key="$1" method="$2" current="${GROUP_METHODS[$1]:-}"
  case ",$current," in
    *",$method,"*) ;;
    *) GROUP_METHODS["$key"]="${current:+$current,}$method" ;;
  esac
}

parse_winners() {
  [ -n "$WINNERS_CSV" ] || {
    log_error "enter winner names from pipeline_ranking.csv in WINNERS_CSV"
    return 2
  }
  local entries=() entry family run method extra
  local space metric neighbors retrieval tail key group_neighbors k_values=()
  csv_to_array "$WINNERS_CSV" entries
  for entry in "${entries[@]}"; do
    IFS='/' read -r family run method extra <<< "$entry"
    if [ -z "$family" ] || [ -z "$run" ] || [ -z "$method" ] || [ -n "${extra:-}" ]; then
      log_error "invalid winner=$entry expected=baselines|gates/retrieval_run/method"
      return 2
    fi
    case "$family" in
      baselines|gates) ;;
      *) log_error "invalid winner family=$family in $entry"; return 2 ;;
    esac
    IFS='_' read -r space metric neighbors retrieval tail <<< "$run"
    if [ -n "${tail:-}" ] || ! [[ "$neighbors" =~ ^[0-9]+$ ]] ||
      { [ "$retrieval" != online ] && [ "$retrieval" != fixed ]; }; then
      log_error "invalid retrieval run=$run in winner=$entry"
      return 2
    fi
    if [ "$EXPERIMENT_MODE" = k_ablation ]; then
      group_neighbors="$PROFILE_K_CSV"
      csv_to_array "$PROFILE_K_CSV" k_values
      for k in "${k_values[@]}"; do
        PIPELINES+=("${space}_${metric}_${k}_${retrieval}/$method")
      done
    elif [ "$EXPERIMENT_MODE" = crossrag ]; then
      group_neighbors=15
      PIPELINES+=("${space}_${metric}_15_${retrieval}/$method")
      CANDIDATE_RUN="${space}_${metric}_15_${retrieval}"
    else
      case "$neighbors" in
        1|3) ;;
        *)
          log_error "selected winner K=$neighbors is outside the primary K={1,3} policy entry=$entry"
          return 2
          ;;
      esac
      group_neighbors="$neighbors"
      PIPELINES+=("$run/$method")
    fi
    key="$family|$space|$metric|$group_neighbors|$retrieval"
    append_method "$key" "$method"
    append_unique SPACES "$space"
    append_unique METRICS "$metric"
    append_unique FAMILIES "$family"
    append_unique METHODS "$method"
    if [ "$EXPERIMENT_MODE" = k_ablation ]; then
      csv_to_array "$PROFILE_K_CSV" k_values
      for k in "${k_values[@]}"; do append_unique SELECTED_NEIGHBORS "$k"; done
    else
      append_unique SELECTED_NEIGHBORS "$group_neighbors"
    fi
    RETRIEVAL_MODE_SELECTED="$retrieval"
  done
}

run_groups() {
  local key family space metric neighbors retrieval methods
  for key in "${!GROUP_METHODS[@]}"; do
    IFS='|' read -r family space metric neighbors retrieval <<< "$key"
    methods="${GROUP_METHODS[$key]}"
    DATASETS_CSV="$PROFILE_DATASETS_CSV"
    MODELS_CSV="$PROFILE_MODELS_CSV"
    SETTINGS_CSV="$PROFILE_SETTINGS_CSV"
    DISTANCE_SPACES_CSV="$space"
    DISTANCE_METRICS_CSV="$metric"
    NEIGHBORS_CSV="$neighbors"
    RETRIEVAL_MODE="$retrieval"
    SKIP_COMPLETE="$EXTRACTION_SKIP_COMPLETE"
    source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
    case "$EXPERIMENT_MODE" in
      full|ultra) SKIP_COMPLETE=true ;;
      *) SKIP_COMPLETE=false ;;
    esac
    if [ "$family" = baselines ]; then
      BASELINE_METHODS_CSV="$methods"
      source "$PROJECT_ROOT/src/slurm/run_baselines.sh"
    else
      GATE_METHODS_CSV="$methods"
      source "$PROJECT_ROOT/src/slurm/run_gates.sh"
    fi
  done
}

if [ "$EXPERIMENT_MODE" = screen ]; then
  run_screen
  return
fi

declare -A GROUP_METHODS=()
SPACES=()
METRICS=()
# Sourced extraction/evaluation scripts own a NEIGHBORS array. Keep the
# orchestrator's complete grid separate so the last executed group cannot
# narrow the final tables to one K value.
SELECTED_NEIGHBORS=()
FAMILIES=()
METHODS=()
PIPELINES=()
RETRIEVAL_MODE_SELECTED=online
parse_winners

if [ "$EXPERIMENT_MODE" = crossrag ] && [ "${#GROUP_METHODS[@]}" -ne 1 ]; then
  log_error "crossrag.slurm accepts exactly one winning pipeline"
  return 2
fi

run_groups

if [ "$EXPERIMENT_MODE" = crossrag ]; then
  # Cross-RAG has its own fixed minmax/cosine retrieval pipeline.
  DATASETS_CSV="$PROFILE_DATASETS_CSV"
  MODELS_CSV="$PROFILE_MODELS_CSV"
  SETTINGS_CSV="$PROFILE_SETTINGS_CSV"
  DISTANCE_SPACES_CSV=minmax
  DISTANCE_METRICS_CSV=cosine
  NEIGHBORS_CSV=15
  RETRIEVAL_MODE=online
  SKIP_COMPLETE=true
  source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
  source "$PROJECT_ROOT/src/slurm/run_crossrag.sh"
  append_unique SPACES minmax
  append_unique METRICS cosine
  append_unique SELECTED_NEIGHBORS 15
  PIPELINES+=("minmax_cosine_15_online/crossrag")
  FAMILIES=(comparison)
  METHODS+=(crossrag)
  CANDIDATE_FAMILY="${!GROUP_METHODS[*]}"
  CANDIDATE_FAMILY="${CANDIDATE_FAMILY%%|*}"
fi

join_csv_values() {
  local IFS=,
  echo "$*"
}

DATASETS_CSV="$PROFILE_DATASETS_CSV"
MODELS_CSV="$PROFILE_MODELS_CSV"
SETTINGS_CSV="$PROFILE_SETTINGS_CSV"
DISTANCE_SPACES_CSV="$(join_csv_values "${SPACES[@]}")"
DISTANCE_METRICS_CSV="$(join_csv_values "${METRICS[@]}")"
NEIGHBORS_CSV="$(join_csv_values "${SELECTED_NEIGHBORS[@]}")"
FAMILIES_CSV="$(join_csv_values "${FAMILIES[@]}")"
METHODS_CSV="$(join_csv_values "${METHODS[@]}")"
PIPELINES_CSV="$(join_csv_values "${PIPELINES[@]}")"
RETRIEVAL_MODE="$RETRIEVAL_MODE_SELECTED"
export DATASETS_CSV MODELS_CSV SETTINGS_CSV DISTANCE_SPACES_CSV
export DISTANCE_METRICS_CSV NEIGHBORS_CSV FAMILIES_CSV METHODS_CSV
export PIPELINES_CSV RETRIEVAL_MODE CANDIDATE_FAMILY
export CANDIDATE_RUN
source "$PROJECT_ROOT/src/slurm/build_tables.sh"
