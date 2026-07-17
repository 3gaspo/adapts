#!/bin/bash
#SBATCH --job-name=univariate
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SHARED_ROOT="$(cd .. && pwd)"
DATA_ROOT="${SHARED_ROOT}/datasets"
OUT_ROOT="outputs/univariate"
CHRONOS_WEIGHTS_PATH="${SHARED_ROOT}/weights/chronos2"

DATASETS=("electricity" "solar")
SETTINGS=("168 24" "672 168")

SPLITS="0.3,0.35,0.15,0.2"
EVAL_QUERY_STRIDE=128
SEED=1

CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future_included\"}"

echo "$(date -Is) job start kind=univariate"
for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$DATA_ROOT/$dataset"
  DATA_CONFIG="$DATASET_DIR/config.json"
  DATA_ARGS=()
  if [ ! -d "$DATASET_DIR" ]; then
    echo "$(date -Is) missing dataset directory dataset=$dataset path=$DATASET_DIR" >&2
    exit 1
  fi
  if [ ! -f "$DATA_CONFIG" ]; then
    echo "$(date -Is) missing dataset config dataset=$dataset path=$DATA_CONFIG" >&2
    exit 1
  fi
  DATA_ARGS+=(--dataset-config "$DATA_CONFIG")

  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"
    echo "$(date -Is) univariate start dataset=$dataset lags=$L horizon=$H model=chronos"
    srun python -m src.experiments.experiment_univariate \
      --csv "$DATASET_DIR" \
      --dataset-name "$dataset" \
      "${DATA_ARGS[@]}" \
      --lags "$L" \
      --horizon "$H" \
      --splits "$SPLITS" \
      --eval-stride "$EVAL_QUERY_STRIDE" \
      --model chronos \
      --model-kwargs "$CHRONOS_KWARGS" \
      --normalization instance \
      --device gpu \
      --output-dir "$SETTING_OUT" \
      --save-name chronos \
      --seed "$SEED"
    echo "$(date -Is) univariate done dataset=$dataset lags=$L horizon=$H model=chronos"
  done
done
echo "$(date -Is) job done kind=univariate output=$OUT_ROOT"
