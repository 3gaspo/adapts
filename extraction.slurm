#!/bin/bash
#SBATCH --job-name=extraction
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000

# Submit from the project root. Keep true for the smoke run; set false for all configurations.
TEST_MODE="${TEST_MODE:-true}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT TEST_MODE
source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
