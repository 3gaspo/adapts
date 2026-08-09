#!/bin/bash
#SBATCH --job-name=gates
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Smoke-only primary shared-gate/reference front.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
EXPERIMENT_FAMILY=gates
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY
source "$PROJECT_ROOT/src/slurm/common.sh"
require_test_experiment_mode || exit $?
source "$PROJECT_ROOT/src/slurm/run_gates.sh"
