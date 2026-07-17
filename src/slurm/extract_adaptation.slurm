#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --gres=gpu:1
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SHARED_ROOT="$(cd .. && pwd)"
DATA_ROOT="${SHARED_ROOT}/datasets"
OUT_ROOT="outputs/adaptation"
CHRONOS_WEIGHTS_PATH="${SHARED_ROOT}/weights/chronos2"

DATASETS=("electricity") # "solar")
SETTINGS=("168 24") # "672 168")
DISTANCE_SPACES=("raw") # "instance")
NEIGHBORS=(1) # 3)

SPLITS="0.3,0.35,0.15,0.2"
RETRIEVAL_MODE="online"
DATASTORE_STRIDE=24
TRAIN_QUERY_STRIDE=24
ORACLE_QUERY_STRIDE=24
EVAL_QUERY_STRIDE=128
PERIOD=24
MAX_STORE_WINDOWS=30000
SEED=1

CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future_included\"}"

echo "$(date -Is) job start kind=adaptation_extraction"
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
    MODEL_ROOT="$OUT_ROOT/$dataset/${L}_${H}/chronos"
    echo "$(date -Is) setting start dataset=$dataset lags=$L horizon=$H"

    echo "$(date -Is) vanilla extraction start dataset=$dataset lags=$L horizon=$H"
    srun python -m src.experiments.extraction \
      --csv "$DATASET_DIR" \
      --dataset-name "$dataset" \
      "${DATA_ARGS[@]}" \
      --lags "$L" \
      --horizon "$H" \
      --splits "$SPLITS" \
      --datastore-stride "$DATASTORE_STRIDE" \
      --train-stride "$TRAIN_QUERY_STRIDE" \
      --oracle-stride "$ORACLE_QUERY_STRIDE" \
      --eval-stride "$EVAL_QUERY_STRIDE" \
      --period "$PERIOD" \
      --neighbors 0 \
      --distance-space raw \
      --distance-metric euclidean \
      --max-store-windows "$MAX_STORE_WINDOWS" \
      --retrieval-mode "$RETRIEVAL_MODE" \
      --model chronos \
      --model-kwargs "$CHRONOS_KWARGS" \
      --normalization instance \
      --device gpu \
      --output-dir "$MODEL_ROOT" \
      --save-name vanilla \
      --seed "$SEED"
    echo "$(date -Is) vanilla extraction done dataset=$dataset lags=$L horizon=$H"

    for space in "${DISTANCE_SPACES[@]}"; do
      for neighbors in "${NEIGHBORS[@]}"; do
        RETRIEVAL_SETTING="${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_ROOT="$MODEL_ROOT/$RETRIEVAL_SETTING"
        echo "$(date -Is) retrieval extraction start dataset=$dataset lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"

        srun python -m src.experiments.extraction \
          --csv "$DATASET_DIR" \
          --dataset-name "$dataset" \
          "${DATA_ARGS[@]}" \
          --lags "$L" \
          --horizon "$H" \
          --splits "$SPLITS" \
          --datastore-stride "$DATASTORE_STRIDE" \
          --train-stride "$TRAIN_QUERY_STRIDE" \
          --oracle-stride "$ORACLE_QUERY_STRIDE" \
          --eval-stride "$EVAL_QUERY_STRIDE" \
          --period "$PERIOD" \
          --neighbors "$neighbors" \
          --distance-space "$space" \
          --distance-metric euclidean \
          --max-store-windows "$MAX_STORE_WINDOWS" \
          --retrieval-mode "$RETRIEVAL_MODE" \
          --model chronos \
          --model-kwargs "$CHRONOS_KWARGS" \
          --normalization instance \
          --device gpu \
          --output-dir "$RUN_ROOT" \
          --save-name extracted \
          --seed "$SEED"
        echo "$(date -Is) retrieval extraction done dataset=$dataset lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
      done
    done
    echo "$(date -Is) setting done dataset=$dataset lags=$L horizon=$H"
  done
done
echo "$(date -Is) job done kind=adaptation_extraction output=$OUT_ROOT"
