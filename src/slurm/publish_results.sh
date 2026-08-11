#!/bin/bash
# Commit and push only the artifacts handed off by one completed Slurm job.

publish_log() {
  printf '%s publish | %s\n' "$(date -Is)" "$*"
}

publish_enabled() {
  case "${PUBLISH_RESULTS:-true}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

publish_collect_paths() {
  local project_root="$1" producer_job_id="$2" launch_id="$3" handoff="$4"
  local manifest directory relative_path extra
  local -a extra_paths=()
  mkdir -p "$(dirname "$handoff")"
  : > "$handoff"

  printf 'logs/%s_%s.out\n' "${SLURM_JOB_NAME:-job}" "$producer_job_id" >> "$handoff"
  printf 'logs/%s_%s.err\n' "${SLURM_JOB_NAME:-job}" "$producer_job_id" >> "$handoff"

  if [ -d "$project_root/outputs" ]; then
    while IFS= read -r -d '' manifest; do
      if grep -Fq "\"launch_id\": \"$launch_id\"" "$manifest"; then
        directory="$(dirname "$manifest")"
        relative_path="${directory#"$project_root"/}"
        printf '%s\n' "$relative_path" >> "$handoff"
      fi
    done < <(
      find "$project_root/outputs" -type f \
        \( -name manifest.json -o -name report_manifest.json \) -print0
    )
  fi

  if [ -n "${PUBLISH_EXTRA_PATHS:-}" ]; then
    IFS=',' read -r -a extra_paths <<< "$PUBLISH_EXTRA_PATHS"
    for extra in "${extra_paths[@]}"; do
      [ -n "$extra" ] && printf '%s\n' "$extra" >> "$handoff"
    done
  fi
  sort -u "$handoff" -o "$handoff"
}

submit_publish_job() {
  local project_root producer_job_id launch_id handoff partition submission
  local -a submit_args
  publish_enabled || return 0
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    publish_log "not inside Slurm; skipping automatic publisher submission"
    return 0
  fi
  project_root="${PROJECT_ROOT:-${ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}}"
  producer_job_id="$SLURM_JOB_ID"
  launch_id="${EXPERIMENT_LAUNCH_ID:-$producer_job_id}"
  handoff="$project_root/logs/.publish/${producer_job_id}.paths"
  if ! publish_collect_paths "$project_root" "$producer_job_id" "$launch_id" "$handoff"; then
    publish_log "could not create $handoff; run the publisher manually"
    return 0
  fi
  submit_args=(
    --parsable
    "--dependency=afterok:$producer_job_id"
    "--chdir=$project_root"
    "--job-name=${SLURM_JOB_NAME:-job}_publish"
    "--export=ALL,PRODUCER_JOB_ID=$producer_job_id,PRODUCER_JOB_NAME=${SLURM_JOB_NAME:-job}"
  )
  partition="${PUBLISH_PARTITION:-${SLURM_JOB_PARTITION:-}}"
  [ -n "$partition" ] && submit_args+=("--partition=$partition")
  if ! submission="$(sbatch "${submit_args[@]}" "$project_root/publish.slurm")"; then
    publish_log "publisher submission failed; retry manually with --job-id $producer_job_id"
    return 0
  fi
  publish_log "submitted afterok publisher job=$submission producer=$producer_job_id"
}

publish_valid_path() {
  local path="$1"
  case "$path" in
    logs/*|outputs/*) ;;
    *) return 1 ;;
  esac
  case "/$path/" in
    */../*|*/./*) return 1 ;;
  esac
  return 0
}

publish_results_main() {
  local project_root producer_job_id paths_file message path proxy_status
  local proxy_script credentials_file credential_mode had_xtrace=0
  local -a requested_paths=() paths=() exclusions
  declare -A seen=()

  project_root="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
  producer_job_id="${PRODUCER_JOB_ID:-}"
  paths_file="${PUBLISH_PATHS_FILE:-}"
  message="${PUBLISH_COMMIT_MESSAGE:-}"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --project-root) project_root="$2"; shift 2 ;;
      --job-id) producer_job_id="$2"; shift 2 ;;
      --paths-file) paths_file="$2"; shift 2 ;;
      --path) requested_paths+=("$2"); shift 2 ;;
      --message) message="$2"; shift 2 ;;
      *) printf 'unknown publisher argument: %s\n' "$1" >&2; return 2 ;;
    esac
  done
  project_root="$(cd "$project_root" && pwd)"
  if [ -z "$paths_file" ] && [ -n "$producer_job_id" ]; then
    paths_file="$project_root/logs/.publish/${producer_job_id}.paths"
  fi
  if [ -n "$paths_file" ]; then
    [ -f "$paths_file" ] || { printf 'publisher paths file not found: %s\n' "$paths_file" >&2; return 1; }
    while IFS= read -r path || [ -n "$path" ]; do
      [ -n "$path" ] && requested_paths+=("$path")
    done < "$paths_file"
  fi
  [ "${#requested_paths[@]}" -gt 0 ] || { printf 'publisher received no paths\n' >&2; return 1; }

  for path in "${requested_paths[@]}"; do
    path="${path#./}"
    publish_valid_path "$path" || { printf 'publisher rejected path: %s\n' "$path" >&2; return 1; }
    [ -e "$project_root/$path" ] || { printf 'publisher path does not exist: %s\n' "$path" >&2; return 1; }
    if [ -z "${seen[$path]+x}" ]; then
      paths+=("$path")
      seen[$path]=1
    fi
  done

  cd "$project_root"
  [ "$(git rev-parse --show-toplevel)" = "$project_root" ] || {
    printf 'publisher must run at the project Git root: %s\n' "$project_root" >&2
    return 1
  }
  [ "$(git symbolic-ref --short HEAD)" = main ] || {
    printf 'publisher requires the main branch\n' >&2
    return 1
  }
  exclusions=(
    ':(exclude,glob)**/*.pt'
    ':(exclude,glob)**/*.npy'
    ':(exclude,glob)**/*.cbm'
  )
  command -v flock >/dev/null 2>&1 || {
    printf 'publisher requires flock to serialize repository updates\n' >&2
    return 1
  }
  (
    if ! flock -w "${PUBLISH_LOCK_TIMEOUT:-600}" 9; then
      printf 'timed out waiting for the repository publisher lock\n' >&2
      exit 1
    fi
    publish_log "acquired repository publisher lock"
    git add -v -f -- "${paths[@]}" "${exclusions[@]}"
    if ! git diff --cached --quiet -- "${paths[@]}" "${exclusions[@]}"; then
      if [ -z "$message" ]; then
        message="slurm: publish ${PRODUCER_JOB_NAME:-job} ${producer_job_id:-manual}"
      fi
      git commit --only -m "$message" -- "${paths[@]}" "${exclusions[@]}"
    else
      publish_log "no new changes to commit; attempting push"
    fi

    if [ "${USE_PROXY:-true}" != false ]; then
      proxy_script="${PROXY_SCRIPT_PATH:-$HOME/codes/proxy.sh}"
      credentials_file="${PROXY_CREDENTIALS_FILE:-$HOME/codes/.secrets/proxy.credentials}"
      [ -f "$proxy_script" ] || { printf 'proxy script not found: %s\n' "$proxy_script" >&2; return 1; }
      [ -f "$credentials_file" ] || { printf 'proxy credentials not found: %s\n' "$credentials_file" >&2; return 1; }
      credential_mode="$(stat -c '%a' "$credentials_file")"
      case "$credential_mode" in
        400|600) ;;
        *) printf 'proxy credentials must use chmod 600 (or 400): %s\n' "$credentials_file" >&2; return 1 ;;
      esac
      case "$-" in *x*) had_xtrace=1 ;; esac
      set +x
      # shellcheck disable=SC1090
      if . "$proxy_script" --credentials-file "$credentials_file"; then
        proxy_status=0
      else
        proxy_status=$?
      fi
      unset PASS NNI
      [ "$had_xtrace" -eq 1 ] && set -x
      if [ "$proxy_status" -ne 0 ] || [ "${NOEXPORT:-1}" -ne 0 ] || [ -z "${https_proxy:-}" ]; then
        printf 'proxy authentication failed\n' >&2
        return 1
      fi
    fi
    git push origin main
  ) 9>"$project_root/.git/slurm-publish.lock"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  publish_results_main "$@"
fi
