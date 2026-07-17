#!/bin/bash
#SBATCH --job-name=tsifa_gates
#SBATCH --output=logs/tsifa_gates_%j.out
#SBATCH --error=logs/tsifa_gates_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=h100-bis
#SBATCH --cpus-per-task=4
#SBATCH --mem=20000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

OUT_ROOT="outputs/adaptation"

DATASETS=("electricity" "solar")
SETTINGS=("168 24" "672 168")
DISTANCE_SPACES=("raw" "instance")
NEIGHBORS=(1 3 10)
RETRIEVAL_MODE="online"
GATE_ITERATIONS=300
GATE_LEARNING_RATE=0.03
GATE_DEPTH=4
SEED=1

echo "$(date -Is) job start kind=gates"
for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    MODEL_ROOT="$OUT_ROOT/$dataset/${L}_${H}/chronos"

    for space in "${DISTANCE_SPACES[@]}"; do
      for neighbors in "${NEIGHBORS[@]}"; do
        RETRIEVAL_SETTING="${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_ROOT="$MODEL_ROOT/$RETRIEVAL_SETTING"
        echo "$(date -Is) gates start dataset=$dataset lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
        srun python -m src.adaptors.baselines.evaluate \
          --input-dir "$RUN_ROOT/extracted" \
          --output-dir "$RUN_ROOT/gates" \
          --family gates \
          --gate-iterations "$GATE_ITERATIONS" \
          --gate-learning-rate "$GATE_LEARNING_RATE" \
          --gate-depth "$GATE_DEPTH" \
          --seed "$SEED"
        echo "$(date -Is) gates done dataset=$dataset lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
      done
    done
  done
done
echo "$(date -Is) job done kind=gates output=$OUT_ROOT"
