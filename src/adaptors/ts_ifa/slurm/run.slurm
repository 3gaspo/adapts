#!/bin/bash
#SBATCH --job-name=tsifa
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

OUT_ROOT="outputs/adaptation"

DATASETS=("electricity" "solar")
SETTINGS=("168 24" "672 168")
DISTANCE_SPACES=("raw" "instance") # raw instance encoder
NEIGHBORS=(1 3 10)
RETRIEVAL_MODE="online"
SEED=1

EPOCHS=10000
BATCH_SIZE=256
LR=0.00001
WEIGHT_DECAY=0.0001
BETA=0.01
GAMMA=0.01
VALID_EVAL_FREQ=1000
LOGGING_EVAL_FREQ=1000

echo "$(date -Is) job start kind=ts_ifa_training"
for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"

    for space in "${DISTANCE_SPACES[@]}"; do
      for neighbors in "${NEIGHBORS[@]}"; do
        RETRIEVAL_SETTING="${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_DIR="$SETTING_OUT/chronos/$RETRIEVAL_SETTING"
        echo "$(date -Is) training start dataset=$dataset lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"

        srun python -m src.adaptors.ts_ifa.train \
          --input-dir "$RUN_DIR/extracted" \
          --output-dir "$RUN_DIR/ts_ifa/TS-IFA" \
          --epochs "$EPOCHS" \
          --batch-size "$BATCH_SIZE" \
          --valid-eval-freq "$VALID_EVAL_FREQ" \
          --logging-eval-freq "$LOGGING_EVAL_FREQ" \
          --lr "$LR" \
          --weight-decay "$WEIGHT_DECAY" \
          --beta "$BETA" \
          --gamma "$GAMMA" \
          --normalization instance \
          --device gpu \
          --seed "$SEED"

        echo "$(date -Is) training done dataset=$dataset lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
      done
    done
  done
done
echo "$(date -Is) job done kind=ts_ifa_training"
