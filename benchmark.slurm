#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Final comparison of complete winners selected by screen.slurm.
# full defaults to Chronos-2; ultra adds TabPFN-TS. Both write to the same
# publication result tree so ultra reuses completed full outputs.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=benchmark
RESULTS_ROOT="${RESULTS_ROOT:-outputs/adaptation/benchmark}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
WINNERS_CSV="${WINNERS_CSV:-$(selected_candidates_csv adaptation)}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY WINNERS_CSV RESULTS_ROOT
source "$PROJECT_ROOT/src/slurm/common.sh"
require_benchmark_experiment_mode || exit $?
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
