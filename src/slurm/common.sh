#!/bin/bash
# Shared Slurm helpers. Source this file from the project root.

is_true() {
  case "${1:-false}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
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
    echo "$(date -Is) empty grid variable target=$target_name raw=$raw" >&2
    return 1
  fi
}

parse_setting() {
  local setting="${1//:/ }"
  setting="${setting//_/ }"
  setting="${setting//-/ }"
  read -r SETTING_LAGS SETTING_HORIZON SETTING_EXTRA <<< "$setting"
  if [ -z "${SETTING_LAGS:-}" ] || [ -z "${SETTING_HORIZON:-}" ] || [ -n "${SETTING_EXTRA:-}" ]; then
    echo "$(date -Is) invalid setting value=$1 expected=L:H" >&2
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
    if [ -d "$candidate" ]; then
      (cd "$candidate" && pwd)
      return 0
    fi
    if [ -d "$root" ]; then
      match="$(find "$root" -mindepth 1 -maxdepth 1 -type d -iname "$dataset" -print -quit)"
      if [ -n "$match" ]; then
        (cd "$match" && pwd)
        return 0
      fi
    fi
  done
  echo "$(date -Is) missing dataset directory dataset=$dataset searched=${roots[*]}" >&2
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
  echo "$(date -Is) missing weight path relative=$relative searched=${roots[*]}" >&2
  return 1
}

require_project_root() {
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
  if [ ! -f "$PROJECT_ROOT/pyproject.toml" ] || [ ! -d "$PROJECT_ROOT/src" ]; then
    echo "$(date -Is) submit from the adaptation project root or set PROJECT_ROOT path=$PROJECT_ROOT" >&2
    return 1
  fi
  cd "$PROJECT_ROOT"
  mkdir -p logs outputs
}

require_extraction() {
  local directory="$1"
  if ! python -m src.experiments.artifacts "$directory"; then
    echo "$(date -Is) extraction is absent, partial, or stale input=$directory" >&2
    echo "Submit extraction first. Legacy payloads need to be re-extracted to receive a completion marker." >&2
    return 1
  fi
}

assert_files() {
  local label="$1"
  shift
  local path
  for path in "$@"; do
    if [ ! -s "$path" ]; then
      echo "$(date -Is) missing expected $label path=$path" >&2
      return 1
    fi
  done
}
