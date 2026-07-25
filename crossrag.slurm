#!/bin/bash
#SBATCH --job-name=crossrag
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Copy exactly one complete winner name from screen's pipeline_ranking.csv.
# Its formula and normalization are retained; K is set to 15 for comparison.
WINNERS_CSV="${WINNERS_CSV:-}"

EXPERIMENT_MODE=crossrag
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
