#!/bin/bash
#SBATCH --job-name=fourier_retrieval_ablation
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Compare every selected pipeline with the same method, metric, K, and mode
# using standardized Fourier-amplitude retrieval.

EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=fourier_retrieval_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
WINNERS_CSV="${WINNERS_CSV:-$(selected_candidates_csv adaptation)}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
