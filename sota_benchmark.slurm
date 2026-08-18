#!/bin/bash
#SBATCH --job-name=sota_benchmark
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80000

# Evaluate one selected project method on the Cross-RAG paper's exact test
# windows after the paper's train-split per-channel standardization, then
# combine its standardized-space MSE with the published TS-RAG/Cross-RAG MSEs.
STAGES="${STAGES:-evaluate,tables}"
EXPERIMENT_MODE=full
EXPERIMENT_FAMILY=sota_benchmark
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
SELECTED_CANDIDATES_FILE="${SELECTED_CANDIDATES_FILE:-$PROJECT_ROOT/SECOND_GENERATION_CANDIDATES.txt}"
WINNERS_CSV="${WINNERS_CSV:-$(selected_candidate_first baseline_shared)}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY WINNERS_CSV STAGES
source "$PROJECT_ROOT/src/slurm/run_sota_benchmark.sh"
