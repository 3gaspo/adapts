#!/bin/bash
#SBATCH --job-name=catboost_ablation
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Hold every selected shared CatBoost regressor fixed while changing its form.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=catboost_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
CATBOOST_WINNERS_CSV="${CATBOOST_WINNERS_CSV:-$(selected_candidates_csv catboost_shared)}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY CATBOOST_WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_catboost_ablation.sh"
