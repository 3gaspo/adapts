#!/bin/bash
#SBATCH --job-name=univariate
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --array=0-0

# Safe default: one smoke task. Full: TEST_MODE=false sbatch --array=0-3%4 univariate.slurm
TEST_MODE="${TEST_MODE:-true}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT TEST_MODE
source "$PROJECT_ROOT/src/slurm/run_univariate.sh"
