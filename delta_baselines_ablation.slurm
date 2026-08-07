#!/bin/bash
#SBATCH --job-name=delta_baselines
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Selected shared-ridge winners from the completed primary screen. The runner
# compares each ridge control with the shared delta-ridge family built from the
# same variables and retrieval pipeline.
BASELINE_WINNERS_CSV="${BASELINE_WINNERS_CSV:-baselines/instance_euclidean_3_online/full_ridge_shared,baselines/instance_euclidean_3_online/y_ridge_shared,baselines/instance_euclidean_3_online/cov_y_ridge_shared,baselines/instance_euclidean_3_online/residual_ridge_shared,baselines/raw_euclidean_3_online/cov_avgy_ridge_shared,baselines/raw_euclidean_3_online/cov_ridge_shared}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=delta_baselines_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY BASELINE_WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_baseline_family_ablation.sh"
