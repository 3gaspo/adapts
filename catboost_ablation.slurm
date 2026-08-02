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

# Select shared-regressor winners from screen. The runner holds their retrieval
# pipeline fixed and expands only CatBoost objective/decision shape.
CATBOOST_WINNERS_CSV="${CATBOOST_WINNERS_CSV:-}"
EXPERIMENT_MODE=catboost_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE CATBOOST_WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_catboost_ablation.sh"
