#!/bin/bash
#SBATCH --job-name=gates
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000
#SBATCH --array=0-671%32

set -euo pipefail
source src/slurm/common.sh
require_project_root
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/adaptation}"
TEST_MODE="${TEST_MODE:-false}"
if is_true "$TEST_MODE"; then
  DATASETS_CSV="${DATASETS_CSV:-electricity}"
  MODELS_CSV="${MODELS_CSV:-chronos}"
  SETTINGS_CSV="${SETTINGS_CSV:-168:24}"
  DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-raw}"
  NEIGHBORS_CSV="${NEIGHBORS_CSV:-3}"
  GATE_ITERATIONS="${GATE_ITERATIONS:-2}"
else
  DATASETS_CSV="${DATASETS_CSV:-ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Exchange}"
  MODELS_CSV="${MODELS_CSV:-chronos,tabpfnts}"
  # 572:64 is the intentional Cross-RAG comparison setting.
  SETTINGS_CSV="${SETTINGS_CSV:-572:64,672:24,672:48,672:168,672:336,672:672,168:24,336:24}"
  DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-raw,instance}"
  NEIGHBORS_CSV="${NEIGHBORS_CSV:-1,3,10}"
  GATE_ITERATIONS="${GATE_ITERATIONS:-300}"
fi
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
GATE_LEARNING_RATE="${GATE_LEARNING_RATE:-0.03}"
GATE_DEPTH="${GATE_DEPTH:-4}"
SEED="${SEED:-1}"

csv_to_array "$DATASETS_CSV" DATASETS
csv_to_array "$MODELS_CSV" MODELS
csv_to_array "$SETTINGS_CSV" SETTINGS
csv_to_array "$DISTANCE_SPACES_CSV" DISTANCE_SPACES
csv_to_array "$NEIGHBORS_CSV" NEIGHBORS

TASKS=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for setting in "${SETTINGS[@]}"; do
      for space in "${DISTANCE_SPACES[@]}"; do
        for neighbors in "${NEIGHBORS[@]}"; do
          TASKS+=("$dataset|$model|$setting|$space|$neighbors")
        done
      done
    done
  done
done

run_task() {
  local task_id="$1" task dataset model setting space neighbors
  task="${TASKS[$task_id]}"
  IFS='|' read -r dataset model setting space neighbors <<< "$task"
  parse_setting "$setting"
  L="$SETTING_LAGS"
  H="$SETTING_HORIZON"
  RETRIEVAL_SETTING="${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
  RUN_ROOT="$OUT_ROOT/$dataset/${L}_${H}/$model/$RETRIEVAL_SETTING"
  INPUT_DIR="$RUN_ROOT/extracted"
  OUTPUT_DIR="$RUN_ROOT/gates"
  require_extraction "$INPUT_DIR"
  echo "$(date -Is) gates start task=$task_id/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
  srun python -m src.adaptors.baselines.evaluate \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --family gates \
    --gate-iterations "$GATE_ITERATIONS" \
    --gate-learning-rate "$GATE_LEARNING_RATE" \
    --gate-depth "$GATE_DEPTH" \
    --seed "$SEED"
  assert_files gate-output \
    "$OUTPUT_DIR/gate_metrics.csv" \
    "$OUTPUT_DIR/gate_metrics.json" \
    "$OUTPUT_DIR/gate_artifacts.pt" \
    "$OUTPUT_DIR/visualization_payload.pt"
  echo "$(date -Is) gates done task=$task_id dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
}

echo "$(date -Is) job start kind=gates test_mode=$TEST_MODE tasks=${#TASKS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
    echo "$(date -Is) array task outside narrowed grid; exiting task=$SLURM_ARRAY_TASK_ID tasks=${#TASKS[@]}"
    exit 0
  fi
  run_task "$SLURM_ARRAY_TASK_ID"
else
  for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
    run_task "$task_id"
  done
fi
echo "$(date -Is) job done kind=gates output=$OUT_ROOT"
