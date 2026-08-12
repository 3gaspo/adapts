#!/bin/bash
# Manually commit and push lightweight experiment artifacts from this project.
set -euo pipefail

usage() {
  printf 'usage: bash publish_job.sh [JOB_ID] [--message TEXT] [--project-root PATH]\n' >&2
}

project_root="$(pwd)"
job_id=""
message=""
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
  job_id="$1"
  shift
fi
while [ "$#" -gt 0 ]; do
  case "$1" in
    --job-id) job_id="$2"; shift 2 ;;
    --message) message="$2"; shift 2 ;;
    --project-root) project_root="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [ -n "$job_id" ] && ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
  usage
  printf 'JOB_ID must be numeric\n' >&2
  exit 2
fi
project_root="$(cd "$project_root" && pwd)"
cd "$project_root"
[ "$(git rev-parse --show-toplevel)" = "$project_root" ] || {
  printf 'run from a project Git root or pass --project-root: %s\n' "$project_root" >&2
  exit 1
}
[ "$(git symbolic-ref --short HEAD)" = main ] || {
  printf 'publisher requires the main branch\n' >&2
  exit 1
}

if [ -n "$job_id" ]; then
  shopt -s nullglob
  out_logs=("$project_root"/logs/*_"$job_id".out)
  err_logs=("$project_root"/logs/*_"$job_id".err)
  shopt -u nullglob
  [ "${#out_logs[@]}" -eq 1 ] || {
    printf 'expected exactly one logs/*_%s.out file; found %s\n' "$job_id" "${#out_logs[@]}" >&2
    exit 1
  }
  [ "${#err_logs[@]}" -eq 1 ] || {
    printf 'expected exactly one logs/*_%s.err file; found %s\n' "$job_id" "${#err_logs[@]}" >&2
    exit 1
  }

  job_name="$(basename "${out_logs[0]}" "_${job_id}.out")"
  paths=(
    "${out_logs[0]#"$project_root"/}"
    "${err_logs[0]#"$project_root"/}"
  )
  while IFS= read -r directory; do
    [ -n "$directory" ] && paths+=("${directory#"$project_root"/}")
  done < <(
    find "$project_root/outputs" -type f \
      \( -name manifest.json -o -name report_manifest.json \) -print0 2>/dev/null |
      while IFS= read -r -d '' manifest; do
        if grep -Eq '"launch_id"[[:space:]]*:[[:space:]]*"'"$job_id"'"' "$manifest"; then
          dirname "$manifest"
        fi
      done | sort -u
  )
  [ -n "$message" ] || message="slurm: publish $job_name $job_id"
else
  paths=(logs outputs)
  [ -d logs ] || { printf 'logs directory not found\n' >&2; exit 1; }
  [ -d outputs ] || { printf 'outputs directory not found\n' >&2; exit 1; }
  [ -n "$message" ] || message="slurm: publish all logs and outputs"
fi

exclusions=(
  ':(exclude,glob)**/*.pt'
  ':(exclude,glob)**/*.npy'
  ':(exclude,glob)**/*.cbm'
)
if [ -n "$job_id" ]; then
  printf 'Publishing job %s paths:\n' "$job_id"
else
  printf 'Publishing all logs and lightweight outputs:\n'
fi
printf '  %s\n' "${paths[@]}"
git add -v -f -- "${paths[@]}" "${exclusions[@]}"
if ! git diff --cached --quiet -- "${paths[@]}" "${exclusions[@]}"; then
  git commit --only -m "$message" -- "${paths[@]}" "${exclusions[@]}"
else
  printf 'No new artifact changes; pushing existing local commits.\n'
fi

proxy_script="${PROXY_SCRIPT_PATH:-$HOME/codes/proxy.sh}"
credentials_file="${PROXY_CREDENTIALS_FILE:-$HOME/codes/.secrets/proxy.credentials}"
[ -f "$proxy_script" ] || { printf 'proxy script not found: %s\n' "$proxy_script" >&2; exit 1; }
[ -f "$credentials_file" ] || { printf 'proxy credentials not found: %s\n' "$credentials_file" >&2; exit 1; }
credential_mode="$(stat -c '%a' "$credentials_file")"
case "$credential_mode" in
  400|600) ;;
  *) printf 'proxy credentials must use chmod 600 (or 400): %s\n' "$credentials_file" >&2; exit 1 ;;
esac

# shellcheck disable=SC1090
. "$proxy_script" --credentials-file "$credentials_file"
unset PASS NNI
if [ "${NOEXPORT:-1}" -ne 0 ] || [ -z "${https_proxy:-}" ]; then
  printf 'proxy authentication failed\n' >&2
  exit 1
fi

# A stale remote VS Code askpass socket cannot answer from a normal shell.
# Let Git use its configured credential helper or prompt in this terminal.
unset GIT_ASKPASS SSH_ASKPASS GIT_TERMINAL_PROMPT VSCODE_GIT_ASKPASS_MAIN \
  VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_EXTRA_ARGS
git push origin main
