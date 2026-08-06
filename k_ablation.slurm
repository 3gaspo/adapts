#!/bin/bash
#SBATCH --job-name=k_ablation
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Copy complete winner names from screen's pipeline_ranking.csv. The default
# grid is K=1,3,5,10,15,20,100; complete K values are reused on resubmission.
# Example: baselines/instance_euclidean_3_online/y_ridge_shared
WINNERS_CSV="${WINNERS_CSV:-}"

EXPERIMENT_MODE=k_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
