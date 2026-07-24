#!/bin/bash
# Time three CatBoost execution modes on one representative H=504 horizon gate.
set -euo pipefail
source src/slurm/common.sh
require_project_root
activate_project_environment
export PYTHONPATH="$PROJECT_ROOT"

OUT_ROOT="${OUT_ROOT:-outputs/adaptation}"
DATASET="${DATASET:-Traffic}"
MODEL="${MODEL:-chronos}"
SETTING="${SETTING:-504:504}"
DISTANCE_SPACE="${DISTANCE_SPACE:-raw}"
NEIGHBORS="${NEIGHBORS:-1}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-online}"
parse_setting "$SETTING"
L="$SETTING_LAGS"
H="$SETTING_HORIZON"
if [ "$H" -ne 504 ]; then
  log_error "this benchmark requires H=504; got setting=$SETTING"
  return 2
fi

RETRIEVAL_SETTING="${DISTANCE_SPACE}_euclidean_${NEIGHBORS}_${RETRIEVAL_MODE}"
RUN_ROOT="$OUT_ROOT/$DATASET/${L}_${H}/$MODEL/$RETRIEVAL_SETTING"
INPUT_DIR="${INPUT_DIR:-$RUN_ROOT/extracted}"
if [ -s "$INPUT_DIR/adapt_prediction_payload.pt" ]; then
  require_extraction "$INPUT_DIR"
elif [ -s "$INPUT_DIR/train_prediction_payload.pt" ] &&
  [ -s "$INPUT_DIR/oracle_prediction_payload.pt" ]; then
  log "using legacy train+oracle payloads for this timing-only benchmark input=$INPUT_DIR"
else
  log_error "missing current adapt payload or legacy train+oracle payloads input=$INPUT_DIR"
  return 1
fi

JOB_TAG="${SLURM_JOB_ID:-manual_$(date +%Y%m%dT%H%M%S)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-outputs/gate_compute_benchmark}"
OUTPUT_DIR="${OUTPUT_DIR:-$BENCHMARK_ROOT/$DATASET/${L}_${H}/$MODEL/$RETRIEVAL_SETTING/$JOB_TAG}"
mkdir -p "$OUTPUT_DIR"

CPUS="${SLURM_CPUS_PER_TASK:-16}"
CPU_SERIAL_THREADS="${CPU_SERIAL_THREADS:-$CPUS}"
GPU_DEVICES="${GPU_DEVICES:-0}"
GPU_DATA_THREADS="${GPU_DATA_THREADS:-4}"
CPU_PARALLEL_JOBS="${CPU_PARALLEL_JOBS:-8}"
CPU_PARALLEL_THREADS="${CPU_PARALLEL_THREADS:-2}"
GATE_CANDIDATE="${GATE_CANDIDATE:-context}"
GATE_OBJECTIVE="${GATE_OBJECTIVE:-regressor}"
GATE_ITERATIONS="${GATE_ITERATIONS:-100}"
GATE_LEARNING_RATE="${GATE_LEARNING_RATE:-0.03}"
GATE_DEPTH="${GATE_DEPTH:-4}"
GATE_EARLY_STOPPING_ROUNDS="${GATE_EARLY_STOPPING_ROUNDS:-20}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}"
MAX_T1_FIT_SAMPLES="${MAX_T1_FIT_SAMPLES:-50000}"
MAX_T2_VALID_SAMPLES="${MAX_T2_VALID_SAMPLES:-10000}"
MAX_REFIT_SAMPLES="${MAX_REFIT_SAMPLES:-50000}"
SEED="${SEED:-1}"

if [ "$CPU_SERIAL_THREADS" -gt "$CPUS" ]; then
  log_error "CPU_SERIAL_THREADS=$CPU_SERIAL_THREADS exceeds allocated CPUs=$CPUS"
  return 2
fi
if [ $((CPU_PARALLEL_JOBS * CPU_PARALLEL_THREADS)) -gt "$CPUS" ]; then
  log_error "parallel jobs*threads exceeds allocated CPUs: $CPU_PARALLEL_JOBS*$CPU_PARALLEL_THREADS > $CPUS"
  return 2
fi

COMMON_ARGS=(
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --candidate "$GATE_CANDIDATE"
  --objective "$GATE_OBJECTIVE"
  --expected-horizon 504
  --validation-fraction "$VALIDATION_FRACTION"
  --iterations "$GATE_ITERATIONS"
  --learning-rate "$GATE_LEARNING_RATE"
  --depth "$GATE_DEPTH"
  --early-stopping-rounds "$GATE_EARLY_STOPPING_ROUNDS"
  --max-t1-fit-samples "$MAX_T1_FIT_SAMPLES"
  --max-t2-valid-samples "$MAX_T2_VALID_SAMPLES"
  --max-refit-samples "$MAX_REFIT_SAMPLES"
  --seed "$SEED"
  --cpu-serial-threads "$CPU_SERIAL_THREADS"
  --gpu-devices "$GPU_DEVICES"
  --gpu-data-threads "$GPU_DATA_THREADS"
  --cpu-parallel-jobs "$CPU_PARALLEL_JOBS"
  --cpu-parallel-threads "$CPU_PARALLEL_THREADS"
)

log_section "benchmark start dataset=$DATASET model=$MODEL setting=$SETTING retrieval=$RETRIEVAL_SETTING candidate=$GATE_CANDIDATE objective=$GATE_OBJECTIVE iterations=$GATE_ITERATIONS output=$OUTPUT_DIR"
FAILED_CASES=0
for mode in cpu_serial gpu_serial cpu_parallel; do
  log_section "benchmark case start mode=$mode"
  STEP_ARGS=(--ntasks=1 --cpus-per-task="$CPUS")
  [ "$mode" != gpu_serial ] || STEP_ARGS+=(--gres=gpu:1)
  if srun "${STEP_ARGS[@]}" \
      python -m src.experiments.benchmark_gate_compute \
      --mode "$mode" \
      "${COMMON_ARGS[@]}"; then
    log "benchmark case done mode=$mode"
  else
    log_error "benchmark case failed mode=$mode; continuing with remaining modes"
    FAILED_CASES=$((FAILED_CASES + 1))
  fi
done
log_section "benchmark done summary=$OUTPUT_DIR/summary.csv"
if [ -s "$OUTPUT_DIR/summary.csv" ]; then
  sed -n '1,4p' "$OUTPUT_DIR/summary.csv"
else
  log_error "benchmark produced no timing summary"
  return 1
fi
if [ "$FAILED_CASES" -ne 0 ]; then
  log_error "benchmark completed with failed_cases=$FAILED_CASES"
  return 1
fi
