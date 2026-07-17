#!/bin/bash
#SBATCH --job-name=tsifa_extract_multi
#SBATCH --gres=gpu:1
#SBATCH --output=logs/tsifa_extract_multi_%j.out
#SBATCH --error=logs/tsifa_extract_multi_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100-bis
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SHARED_ROOT="$(cd .. && pwd)"
DATA_ROOT="${SHARED_ROOT}/datasets"
OUT_ROOT="outputs/adaptation"
CHRONOS_WEIGHTS_PATH="${SHARED_ROOT}/weights/chronos2"
TABPFN_WEIGHTS_PATH="${SHARED_ROOT}/weights/tabpfnts/tabpfn-v2.5-regressor-v2.5_default.ckpt"

DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "Weather" "Electricity" "Exchange")
# ts_icl must be registered in ts_ifa.models before launching this sweep.
MODELS=("chronos" "tabpfnts" "ts_icl")
SETTINGS=("572 64" "672 24" "672 48" "672 168" "672 336" "672 672" "168 24" "336 24")
DISTANCE_SPACES=("raw" "instance")
NEIGHBORS=(1 3 10)

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
TABPFN_KWARGS="{\"weights_path\":\"$TABPFN_WEIGHTS_PATH\",\"device\":\"cuda\",\"context_mode\":\"future_included\"}"

model_kwargs() {
  local model="$1"
  case "$model" in
    chronos)
      echo "$CHRONOS_KWARGS"
      ;;
    tabpfnts)
      echo "$TABPFN_KWARGS"
      ;;
    ts_icl)
      echo "{}"
      ;;
    *)
      echo "$(date -Is) unknown model=$model" >&2
      exit 1
      ;;
  esac
}

echo "$(date -Is) job start kind=multi_backbone_extraction"
for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$DATA_ROOT/$dataset"
  DATA_CONFIG="$DATASET_DIR/config.json"
  DATA_ARGS=()
  if [ ! -d "$DATASET_DIR" ]; then
    echo "$(date -Is) missing dataset directory dataset=$dataset path=$DATASET_DIR" >&2
    exit 1
  fi
  if [ -f "$DATA_CONFIG" ]; then
    DATA_ARGS+=(--dataset-config "$DATA_CONFIG")
  fi

  for model in "${MODELS[@]}"; do
    MODEL_KWARGS="$(model_kwargs "$model")"
    for setting in "${SETTINGS[@]}"; do
      read -r L H <<< "$setting"
      MODEL_ROOT="$OUT_ROOT/$dataset/${L}_${H}/$model"
      echo "$(date -Is) setting start dataset=$dataset model=$model lags=$L horizon=$H"

      echo "$(date -Is) vanilla extraction start dataset=$dataset model=$model lags=$L horizon=$H"
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
        --model "$model" \
        --model-kwargs "$MODEL_KWARGS" \
        --normalization instance \
        --device gpu \
        --output-dir "$MODEL_ROOT" \
        --save-name vanilla \
        --seed "$SEED"
      echo "$(date -Is) vanilla extraction done dataset=$dataset model=$model lags=$L horizon=$H"

      for space in "${DISTANCE_SPACES[@]}"; do
        for neighbors in "${NEIGHBORS[@]}"; do
          RETRIEVAL_SETTING="${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
          RUN_ROOT="$MODEL_ROOT/$RETRIEVAL_SETTING"
          echo "$(date -Is) retrieval extraction start dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"

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
            --model "$model" \
            --model-kwargs "$MODEL_KWARGS" \
            --normalization instance \
            --device gpu \
            --output-dir "$RUN_ROOT" \
            --save-name extracted \
            --seed "$SEED"
          echo "$(date -Is) retrieval extraction done dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
        done
      done
      echo "$(date -Is) setting done dataset=$dataset model=$model lags=$L horizon=$H"
    done
  done
done
echo "$(date -Is) job done kind=multi_backbone_extraction output=$OUT_ROOT"
