#!/bin/bash
# Evaluate non-trainable and fitted adaptation baselines from completed extractions.
# Submit run_baselines.slurm; source this implementation only for local debugging.
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
else
  DATASETS_CSV="${DATASETS_CSV:-ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Exchange}"
  MODELS_CSV="${MODELS_CSV:-chronos,tabpfnts}"
  # 572:64 is the intentional Cross-RAG comparison setting.
  SETTINGS_CSV="${SETTINGS_CSV:-572:64,672:24,672:48,672:168,672:336,672:672,168:24,336:24}"
  DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-raw,instance}"
  NEIGHBORS_CSV="${NEIGHBORS_CSV:-1,3,10}"
fi
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
L2="${L2:-0.001}"
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
  OUTPUT_DIR="$RUN_ROOT/baselines"
  require_extraction "$INPUT_DIR"
  log_section "baselines start task=$task_id/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING family=baselines l2=$L2 fit_baselines_on_eval=true seed=$SEED"
  srun --ntasks=1 python -m src.adaptors.baselines.evaluate \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --family baselines \
    --l2 "$L2" \
    --fit-baselines-on-eval \
    --seed "$SEED"
  assert_files baseline-output \
    "$OUTPUT_DIR/baseline_metrics.csv" \
    "$OUTPUT_DIR/baseline_metrics.json" \
    "$OUTPUT_DIR/baseline_artifacts.pt" \
    "$OUTPUT_DIR/visualization_payload.pt"
  log "baselines done task=$task_id dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
}

log_section "job start kind=baselines test_mode=$TEST_MODE tasks=${#TASKS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV distance_spaces=$DISTANCE_SPACES_CSV neighbors=$NEIGHBORS_CSV"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
    log "array task outside narrowed sweep; exiting task=$SLURM_ARRAY_TASK_ID tasks=${#TASKS[@]}"
    exit 0
  fi
  run_task "$SLURM_ARRAY_TASK_ID"
else
  for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
    run_task "$task_id"
  done
fi
log_section "job done kind=baselines output=$OUT_ROOT"
