#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-small}"
SBATCH_EXPORT="ALL,EXPERIMENT_MODE=$EXPERIMENT_MODE"

extract="$(sbatch --parsable --export="$SBATCH_EXPORT" extraction.slurm)"
extract="${extract%%;*}"

baseline="$(sbatch --parsable --export="$SBATCH_EXPORT" \
  --dependency="afterok:$extract" baselines.slurm)"
baseline="${baseline%%;*}"

gate="$(sbatch --parsable --export="$SBATCH_EXPORT" \
  --dependency="afterok:$extract" gates.slurm)"
gate="${gate%%;*}"

tables="$(sbatch --parsable --export="$SBATCH_EXPORT" \
  --dependency="afterok:$baseline:$gate" tables.slurm)"
tables="${tables%%;*}"

printf 'Submitted mode=%s extraction=%s baselines=%s gates=%s tables=%s\n' \
  "$EXPERIMENT_MODE" "$extract" "$baseline" "$gate" "$tables"
