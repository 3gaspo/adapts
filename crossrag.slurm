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

# The selected winner runs first on Chronos-2 with its own retrieval K. The
# same method then runs on Chronos-Bolt at K=15 beside released Cross-RAG.
# Put the released weights under weights/chronos-bolt-base and weights/cross-rag.
WINNERS_CSV="${WINNERS_CSV:-baselines/instance_euclidean_10_online/full_ridge_shared}"

EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=crossrag
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY WINNERS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
