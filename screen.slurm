#!/bin/bash
#SBATCH --job-name=screen
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Chronos-2 primary shared-method screen; no winner names are needed here.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=screen
MODELS_CSV=chronos2
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY MODELS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
