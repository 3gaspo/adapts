#!/bin/bash
# Fit adaptation gates from completed extractions and write gate metrics/artifacts.
# Submit ../../gates.slurm; source this implementation only for local debugging.
set -euo pipefail
source src/slurm/common.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/adaptation}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
require_experiment_mode
DEFAULT_SMALL_DATASETS_CSV="Traffic,Electricity,Solar"
DEFAULT_FULL_DATASETS_CSV="ETTh1,Electricity,Traffic,Solar,Weather,Exchange"
DEFAULT_SMALL_SETTINGS_CSV="168:24,504:24,504:168,504:504"
DEFAULT_FULL_SETTINGS_CSV="$DEFAULT_SMALL_SETTINGS_CSV,512:64"
case "$EXPERIMENT_MODE" in
  test)
    DEFAULT_PROFILE_DATASETS_CSV="electricity"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_PROFILE_SETTINGS_CSV="168:24"
    DEFAULT_DISTANCE_SPACES_CSV="raw"
    DEFAULT_NEIGHBORS_CSV="3"
    DEFAULT_GATE_ITERATIONS=2
    DEFAULT_SKIP_COMPLETE=false
    ;;
  small)
    DEFAULT_PROFILE_DATASETS_CSV="$DEFAULT_SMALL_DATASETS_CSV"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_PROFILE_SETTINGS_CSV="$DEFAULT_SMALL_SETTINGS_CSV"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    DEFAULT_GATE_ITERATIONS=300
    DEFAULT_SKIP_COMPLETE=true
    ;;
  full|large)
    DEFAULT_PROFILE_DATASETS_CSV="$DEFAULT_FULL_DATASETS_CSV"
    DEFAULT_MODELS_CSV="chronos"
    DEFAULT_PROFILE_SETTINGS_CSV="$DEFAULT_FULL_SETTINGS_CSV"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    DEFAULT_GATE_ITERATIONS=300
    DEFAULT_SKIP_COMPLETE=true
    ;;
  ultra)
    DEFAULT_PROFILE_DATASETS_CSV="$DEFAULT_FULL_DATASETS_CSV"
    DEFAULT_MODELS_CSV="chronos,tabpfnts"
    DEFAULT_PROFILE_SETTINGS_CSV="$DEFAULT_FULL_SETTINGS_CSV"
    DEFAULT_DISTANCE_SPACES_CSV="raw,instance"
    DEFAULT_NEIGHBORS_CSV="1,3,10"
    DEFAULT_GATE_ITERATIONS=300
    DEFAULT_SKIP_COMPLETE=true
    ;;
esac
DATASETS_CSV="${DATASETS_CSV:-$DEFAULT_PROFILE_DATASETS_CSV}"
MODELS_CSV="${MODELS_CSV:-$DEFAULT_MODELS_CSV}"
SETTINGS_CSV="${SETTINGS_CSV:-$DEFAULT_PROFILE_SETTINGS_CSV}"
DISTANCE_SPACES_CSV="${DISTANCE_SPACES_CSV:-$DEFAULT_DISTANCE_SPACES_CSV}"
NEIGHBORS_CSV="${NEIGHBORS_CSV:-$DEFAULT_NEIGHBORS_CSV}"
GATE_ITERATIONS="${GATE_ITERATIONS:-$DEFAULT_GATE_ITERATIONS}"
SKIP_COMPLETE="${SKIP_COMPLETE:-$DEFAULT_SKIP_COMPLETE}"
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

gate_complete() {
  local output="$1"
  [ -s "$output/gate_metrics.csv" ] &&
    [ -s "$output/gate_metrics.json" ] &&
    [ -s "$output/gate_artifacts.pt" ] &&
    [ -s "$output/visualization_payload.pt" ]
}

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
  if is_true "$SKIP_COMPLETE" && gate_complete "$OUTPUT_DIR" &&
    [ "$OUTPUT_DIR/gate_metrics.json" -nt "$INPUT_DIR/extraction_manifest.json" ]; then
    log "skip complete family=gates dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
    return
  fi
  log_section "gates start configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING family=gates iterations=$GATE_ITERATIONS learning_rate=$GATE_LEARNING_RATE depth=$GATE_DEPTH seed=$SEED"
  srun --ntasks=1 python -m src.adaptors.baselines.evaluate \
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
  log "gates done configuration=$((task_id + 1))/${#TASKS[@]} dataset=$dataset model=$model lags=$L horizon=$H retrieval=$RETRIEVAL_SETTING"
}

log_section "job start kind=gates experiment_mode=$EXPERIMENT_MODE skip_complete=$SKIP_COMPLETE tasks=${#TASKS[@]} datasets=$DATASETS_CSV models=$MODELS_CSV settings=$SETTINGS_CSV distance_spaces=$DISTANCE_SPACES_CSV neighbors=$NEIGHBORS_CSV"
for ((task_id = 0; task_id < ${#TASKS[@]}; task_id++)); do
  run_task "$task_id"
done
log_section "job done kind=gates output=$OUT_ROOT"
