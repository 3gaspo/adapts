#!/bin/bash
#SBATCH --job-name=tables
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

OUT_ROOT="outputs/adaptation"
OUTPUT_DIR="$OUT_ROOT/tables/full"

DATASETS=("electricity" "solar")
SETTINGS=("168_24" "672_168")
DISTANCE_SPACES=("raw" "instance")
NEIGHBORS=(1 3 10)
RETRIEVAL_MODE="online"

DATASET_ARG="$(IFS=,; echo "${DATASETS[*]}")"
SETTING_ARG="$(IFS=,; echo "${SETTINGS[*]}")"
SPACE_ARG="$(IFS=,; echo "${DISTANCE_SPACES[*]}")"
NEIGHBOR_ARG="$(IFS=,; echo "${NEIGHBORS[*]}")"

echo "$(date -Is) job start kind=full_tables datasets=$DATASET_ARG settings=$SETTING_ARG spaces=$SPACE_ARG neighbors=$NEIGHBOR_ARG mode=$RETRIEVAL_MODE"
srun python -m src.visu.sweep_results_table \
  "$OUT_ROOT" \
  --table-kind full \
  --output-dir "$OUTPUT_DIR" \
  --metric nmse \
  --split eval \
  --datasets "$DATASET_ARG" \
  --settings "$SETTING_ARG" \
  --spaces "$SPACE_ARG" \
  --neighbors "$NEIGHBOR_ARG" \
  --retrieval-mode "$RETRIEVAL_MODE" \
  --decimals 2
echo "$(date -Is) job done kind=full_tables output=$OUTPUT_DIR"
