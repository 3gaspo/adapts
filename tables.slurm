#!/bin/bash
#SBATCH --job-name=tables
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

# Select the same test, small, or large profile used by upstream jobs.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE
source "$PROJECT_ROOT/src/slurm/build_tables.sh"
