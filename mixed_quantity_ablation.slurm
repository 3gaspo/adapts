#!/bin/bash
#SBATCH --job-name=mixed_quantity
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# The four original ETT panels and Weather mix unlike physical quantities.
# Copy selected complete winners from screen's pipeline_ranking.csv; only those
# pipelines are transferred to the mixed-quantity datasets.
WINNERS_CSV="${WINNERS_CSV:-}"
EXPERIMENT_MODE=mixed_quantity_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
