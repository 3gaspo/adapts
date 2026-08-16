#!/bin/bash
# Read the manually curated candidate manifest and return a filtered CSV list.

selected_candidates_csv() {
  local selection="${1:-adaptation}"
  local file="${SELECTED_CANDIDATES_FILE:-$PROJECT_ROOT/SWEEP_CANDIDATES.txt}"
  local entries=()
  local line family run method extra include

  if [ ! -f "$file" ]; then
    printf 'missing selected candidate file: %s\n' "$file" >&2
    return 1
  fi

  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -n "$line" ] || continue

    IFS='/' read -r family run method extra <<< "$line"
    if [ -z "$family" ] || [ -z "$run" ] || [ -z "$method" ] || [ -n "${extra:-}" ]; then
      printf 'invalid selected candidate: %s\n' "$line" >&2
      return 2
    fi
    case "$family" in
      baselines|gates|ts_ifa) ;;
      *) printf 'unknown selected candidate family: %s\n' "$family" >&2; return 2 ;;
    esac

    include=false
    case "$selection" in
      adaptation)
        case "$family" in
          baselines|gates) include=true ;;
        esac
        ;;
      baseline_shared)
        [ "$family" = baselines ] && [[ "$method" = *_ridge_shared ]] && include=true
        ;;
      catboost_shared)
        [ "$family" = gates ] && [[ "$method" = catboost_*_regressor_shared ]] && include=true
        ;;
      ts_ifa)
        [ "$family" = ts_ifa ] && include=true
        ;;
      ts_ifa_ridge)
        [ "$family" = ts_ifa ] && [[ "$method" = joint_ridge_* ]] && include=true
        ;;
      ts_ifa_neural)
        [ "$family" = ts_ifa ] && [[ "$method" = joint_neural_* ]] && include=true
        ;;
      *) printf 'unknown selected candidate filter: %s\n' "$selection" >&2; return 2 ;;
    esac
    [ "$include" = true ] && entries+=("$line")
  done < "$file"

  local IFS=,
  printf '%s' "${entries[*]}"
}

selected_candidate_first() {
  local selection="${1:-adaptation}" candidates
  candidates="$(selected_candidates_csv "$selection")" || return
  if [ -z "$candidates" ]; then
    printf 'no selected candidate for filter: %s\n' "$selection" >&2
    return 1
  fi
  printf '%s' "${candidates%%,*}"
}
