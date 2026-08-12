#!/bin/bash
# Shared launcher helpers. Source this file from the project root.

# Keep third-party progress bars out of Slurm stderr; application logs use stdout.
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

RUN_CONFLICT_POLICY="${RUN_CONFLICT_POLICY:-overwrite_exact}"
FORCE_RUN="${FORCE_RUN:-false}"
SKIP_COMPLETE="${SKIP_COMPLETE:-true}"
EXPERIMENT_LAUNCH_ID="${EXPERIMENT_LAUNCH_ID:-${SLURM_JOB_ID:-manual_$(date -u '+%Y%m%dT%H%M%SZ')_$$}}"
export EXPERIMENT_LAUNCH_ID

adaptation_on_exit() {
  local status=$?
  trap - EXIT
  if [ "$status" -ne 0 ] && [ -n "${PROJECT_ROOT:-}" ]; then
    python -m experiment_runs interrupt-launch --root "$PROJECT_ROOT/outputs" --launch-id "$EXPERIMENT_LAUNCH_ID" || true
  elif python -m experiment_runs complete-launch --root "$PROJECT_ROOT/outputs" --launch-id "$EXPERIMENT_LAUNCH_ID" >/dev/null; then
    :
  else
    status=$?
  fi
  exit "$status"
}
trap adaptation_on_exit EXIT

log() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

log_section() {
  printf '\n%s %s\n' "$(date -Is)" "$*"
}

log_error() {
  printf '%s %s\n' "$(date -Is)" "$*" >&2
}

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_experiment_mode() {
  case "${EXPERIMENT_MODE:-test}" in
    test|full|ultra) ;;
    *)
      log_error "unknown EXPERIMENT_MODE=${EXPERIMENT_MODE:-}"
      return 2
      ;;
  esac
}

require_experiment_family() {
  case "${EXPERIMENT_FAMILY:-}" in
    extraction|baselines|gates|benchmark|screen|mixed_quantity_ablation|horizon_baselines_ablation|convex_baselines_ablation|delta_baselines_ablation|catboost_ablation|k_ablation|h_ablation|l_ablation|ts_ifa|ts_ifa_h_ablation|ts_ifa_l_ablation|ts_ifa_meta_ridge|ts_ifa_meta_neural|crossrag|tables) ;;
    *) log_error "unknown EXPERIMENT_FAMILY=${EXPERIMENT_FAMILY:-}"; return 2 ;;
  esac
}

require_scale_experiment_mode() {
  case "${EXPERIMENT_MODE:-test}" in
    test|full|ultra) ;;
    *)
      log_error "this front supports only EXPERIMENT_MODE=test, full, or ultra; use the dedicated screen/ablation front for ${EXPERIMENT_MODE:-}"
      return 2
      ;;
  esac
}

require_test_experiment_mode() {
  if [ "${EXPERIMENT_MODE:-test}" != test ]; then
      log_error "this smoke front supports only EXPERIMENT_MODE=test; use benchmark.slurm for the final full/ultra comparison"
      return 2
  fi
}

require_benchmark_experiment_mode() {
  case "${EXPERIMENT_MODE:-full}" in
    full|ultra) ;;
    *)
      log_error "benchmark.slurm supports only EXPERIMENT_MODE=full or ultra"
      return 2
      ;;
  esac
}

result_methods_match() {
  local manifest="$1" expected_csv="$2"
  python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
actual = set(manifest["methods"])
expected = {item.strip() for item in sys.argv[2].replace(";", ",").split(",") if item.strip()}
metric_fields = set(manifest.get("metric_fields", ()))
raise SystemExit(
    0
    if (not expected or actual == expected) and "positive_window_pct" in metric_fields
    else 1
)
' "$manifest" "$expected_csv"
}

activate_project_environment() {
  local activate="${VENV_ACTIVATE:-$PROJECT_ROOT/.venv/bin/activate}"
  if [ -f "$activate" ]; then
    source "$activate"
  elif [ -z "${VIRTUAL_ENV:-}" ]; then
    log_error "no active environment and $activate does not exist"
    return 1
  fi
}

csv_to_array() {
  local raw="${1//;/,}"
  local target_name="$2"
  local -n target="$target_name"
  local values=()
  local item
  IFS=',' read -r -a values <<< "$raw"
  target=()
  for item in "${values[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [ -n "$item" ] && target+=("$item")
  done
  if [ "${#target[@]}" -eq 0 ]; then
    log_error "empty sweep dimension target=$target_name raw=$raw"
    return 1
  fi
}

parse_setting() {
  local setting="${1//:/ }"
  setting="${setting//_/ }"
  setting="${setting//-/ }"
  read -r SETTING_LAGS SETTING_HORIZON SETTING_EXTRA <<< "$setting"
  if [ -z "${SETTING_LAGS:-}" ] || [ -z "${SETTING_HORIZON:-}" ] || [ -n "${SETTING_EXTRA:-}" ]; then
    log_error "invalid setting value=$1 expected=L:H"
    return 1
  fi
}

resource_candidates() {
  local kind="$1"
  printf '%s\n' \
    "$PROJECT_ROOT/$kind" \
    "$PROJECT_ROOT/../$kind" \
    "$PROJECT_ROOT/../../../$kind"
}

find_dataset_dir() {
  local dataset="$1"
  local roots=()
  local root candidate match
  if [ -n "${DATA_ROOT:-}" ]; then
    roots=("$DATA_ROOT")
  else
    mapfile -t roots < <(resource_candidates datasets)
  fi
  for root in "${roots[@]}"; do
    candidate="$root/$dataset"
    if [ -d "$candidate" ] && find "$candidate" -maxdepth 1 -type f -iname "$dataset.csv" -print -quit | grep -q .; then
      (cd "$candidate" && pwd)
      return 0
    fi
    if [ -d "$root" ]; then
      match="$(find "$root" -mindepth 2 -maxdepth 2 -type f -iname "$dataset.csv" -printf '%h\n' -quit)"
      if [ -n "$match" ]; then
        (cd "$match" && pwd)
        return 0
      fi
    fi
  done
  log_error "missing dataset directory dataset=$dataset searched=${roots[*]}"
  return 1
}

find_weight_path() {
  local relative="$1"
  local roots=()
  local root candidate
  if [ -n "${WEIGHTS_ROOT:-}" ]; then
    roots=("$WEIGHTS_ROOT")
  else
    mapfile -t roots < <(resource_candidates weights)
  fi
  for root in "${roots[@]}"; do
    candidate="$root/$relative"
    if [ -e "$candidate" ]; then
      if [ -d "$candidate" ]; then
        (cd "$candidate" && pwd)
      else
        printf '%s/%s\n' "$(cd "$(dirname "$candidate")" && pwd)" "$(basename "$candidate")"
      fi
      return 0
    fi
  done
  log_error "missing weight path relative=$relative searched=${roots[*]}"
  return 1
}

require_project_root() {
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
  if [ ! -f "$PROJECT_ROOT/pyproject.toml" ] || [ ! -d "$PROJECT_ROOT/src" ]; then
    log_error "submit from the adaptation project root or set PROJECT_ROOT path=$PROJECT_ROOT"
    return 1
  fi
  cd "$PROJECT_ROOT"
  mkdir -p logs outputs
}

copy_if_needed() {
  local source="$1"
  local destination="$2"
  local destination_dir temporary
  if [ -f "$destination" ] && cmp -s "$source" "$destination"; then
    return 0
  fi
  destination_dir="$(dirname "$destination")"
  mkdir -p "$destination_dir"
  temporary="$(mktemp "$destination_dir/.${destination##*/}.XXXXXX")"
  if ! cp "$source" "$temporary"; then
    rm -f "$temporary"
    return 1
  fi
  if [ -f "$destination" ] && cmp -s "$temporary" "$destination"; then
    rm -f "$temporary"
    return 0
  fi
  mv -f "$temporary" "$destination"
}

require_extraction() {
  local directory="$1"
  if ! python -m src.experiments.artifacts "$directory"; then
    log_error "extraction is absent, partial, or stale input=$directory"
    log_error "submit extraction first; older payloads must be re-extracted to receive a completion marker"
    return 1
  fi
}

assert_files() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [ ! -s "$path" ]; then
      log_error "missing expected $label path=$path"
      return 1
    fi
  done
}

allocate_manifest_run() {
  local identity_root="$1" workflow="$2" dataset="$3" lags="$4" horizon="$5" backbone="$6"
  local model_order="$7" display_name="$8" row_config="$9" column_config="${10}"
  local model_values_name="${11}" pipeline_values_name="${12}" seed="${13}" input_path="${14:-}"
  local -n model_values_ref="$model_values_name"
  local -n pipeline_values_ref="$pipeline_values_name"
  local purpose pair
  local -a args
  if [ "${EXPERIMENT_MODE:-test}" = test ]; then purpose=smoke; else purpose=publication; fi
  args=(
    --identity-root "$identity_root" --project adaptation --workflow "$workflow"
    --dataset "$dataset" --lookback "$lags" --horizon "$horizon" --backbone "$backbone"
    --model-config-order "$model_order" --purpose "$purpose" --mode "${EXPERIMENT_MODE:-test}"
    --display-name "$display_name" --row-config "$row_config" --column-config "$column_config"
    --runtime-config "slurm.job_id=${SLURM_JOB_ID:-}"
    --policy "$RUN_CONFLICT_POLICY" --skip-completed "$SKIP_COMPLETE" --force "$FORCE_RUN"
    --launch-id "$EXPERIMENT_LAUNCH_ID" --seed "$seed"
  )
  for pair in "${model_values_ref[@]}"; do args+=(--model-config "$pair"); done
  for pair in "${pipeline_values_ref[@]}"; do args+=(--pipeline-config "$pair"); done
  if [ -n "$input_path" ]; then args+=(--input "upstream_manifest=$input_path"); fi
  if declare -p ADDITIONAL_INPUTS >/dev/null 2>&1; then
    for pair in "${ADDITIONAL_INPUTS[@]}"; do args+=(--input "$pair"); done
  fi
  if [ -n "${RUN_INDEX:-}" ]; then args+=(--run-index "$RUN_INDEX"); fi
  IFS=$'\t' read -r ALLOCATED_RUN_DIR ALLOCATED_ACTION ALLOCATED_SIGNATURE < <(
    python -m experiment_runs allocate "${args[@]}"
  )
}

mark_manifest_running() {
  python -m experiment_runs prepare --run-dir "$1" >/dev/null
  python -m experiment_runs status --run-dir "$1" --status running
}

mark_manifest_ready() {
  local run_dir="$1"
  shift
  local artifact
  local -a args=()
  for artifact in "$@"; do args+=(--artifact "$artifact"); done
  python -m experiment_runs ready --run-dir "$run_dir" "${args[@]}"
}

resolve_extraction_run() {
  local dataset="$1" lags="$2" horizon="$3" backbone="$4" space="$5" metric="$6" neighbors="$7" mode="$8"
  local identity_root="$PROJECT_ROOT/outputs/extraction/$dataset/${lags}_${horizon}/${backbone,,}/${space,,}/${metric,,}/$neighbors/${mode,,}"
  local label manifest_id extra pair
  local -a resolve_args=(
    --config-policy distinct
    --repeat-policy selected
    --allow-ready-launch-id "$EXPERIMENT_LAUNCH_ID"
  )
  if [ -n "${EXTRACTION_PIPELINE_CONFIGS:-}" ]; then
    for pair in ${EXTRACTION_PIPELINE_CONFIGS}; do resolve_args+=(--pipeline-config "$pair"); done
  fi
  IFS=$'\t' read -r EXTRACTION_RUN_DIR label manifest_id extra < <(
    python -m experiment_runs resolve --identity-root "$identity_root" "${resolve_args[@]}"
  )
  if [ -z "${EXTRACTION_RUN_DIR:-}" ] || [ -n "${extra:-}" ]; then
    log_error "expected exactly one extraction pipeline identity=$identity_root; set EXTRACTION_PIPELINE_CONFIGS='key=value ...' when multiple pipeline configs exist"
    return 1
  fi
  python -m experiment_runs validate \
    --run-dir "$EXTRACTION_RUN_DIR" \
    --allow-ready-launch-id "$EXPERIMENT_LAUNCH_ID" >/dev/null
  EXTRACTION_MANIFEST_ID="$manifest_id"
}
