#!/bin/bash
#SBATCH --job-name=tsrag
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000

# Re-evaluate released TS-RAG on the four project datasets with its default
# Chronos-T5/Euclidean/same-user K=10 retrieval. The released ARM runs only on
# Chronos-Bolt; the selected project method is evaluated on Bolt and Chronos-2.
STAGES="${STAGES:-evaluate,tables}"
EXPERIMENT_MODE=full
EXPERIMENT_FAMILY=tsrag
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
WINNERS_CSV="${WINNERS_CSV:-$(selected_candidate_first baseline_shared)}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY WINNERS_CSV STAGES
source "$PROJECT_ROOT/src/slurm/run_tsrag_experiment.sh"
