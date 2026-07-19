#!/bin/bash
#SBATCH --job-name=univariate
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000

# Select test, small, or large; this direct reference remains Chronos-only.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE
source "$PROJECT_ROOT/src/slurm/run_univariate.sh"
