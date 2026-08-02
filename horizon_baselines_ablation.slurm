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

# Shared winners from the primary screen. The implementation evaluates each
# winner beside its per-horizon counterpart using the same retrieval pipeline.
SHARED_BASELINE_WINNERS_CSV="${SHARED_BASELINE_WINNERS_CSV:-baselines/instance_euclidean_3_online/aggr_y_mix_shared,baselines/instance_euclidean_3_online/aggr_y_ridge_shared,baselines/instance_euclidean_3_online/y_ridge_shared}"
EXPERIMENT_MODE=horizon_baselines_ablation
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE SHARED_BASELINE_WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_horizon_baselines_ablation.sh"
