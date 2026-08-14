#!/bin/bash
#SBATCH --job-name=horizon_baselines
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Compare every selected shared-ridge baseline with its horizon-wise form.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=horizon_baselines_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
BASELINE_WINNERS_CSV="${BASELINE_WINNERS_CSV:-$(selected_candidates_csv baseline_shared)}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY BASELINE_WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_baseline_family_ablation.sh"
