#!/bin/bash
# Manually synchronize and publish lightweight experiment artifacts from this project.
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

proxy_script="${PROXY_SCRIPT_PATH:-$HOME/codes/proxy.sh}"
[ -f "$proxy_script" ] || { printf 'proxy script not found: %s\n' "$proxy_script" >&2; exit 1; }

# shellcheck disable=SC1090
. "$proxy_script"
git pull --ff-only origin main

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
git push origin main
