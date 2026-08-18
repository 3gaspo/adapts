#!/bin/bash
#SBATCH --job-name=retrieval_scope_ablation
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Compare every selected pipeline under the complete datastore, same-user-only,
# and other-users-only scopes at four datastore caps. At the 10,000-window cap,
# also compare same-user retrieval with a cross-user pool containing exactly one
# candidate per store date, matching its cardinality and temporal coverage.

EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=retrieval_scope_ablation
RETRIEVAL_MAX_STORE_WINDOWS_CSV="${RETRIEVAL_MAX_STORE_WINDOWS_CSV:-10000,20000,30000,50000}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
WINNERS_CSV="${WINNERS_CSV:-$(selected_candidates_csv adaptation)}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY WINNERS_CSV
export RETRIEVAL_MAX_STORE_WINDOWS_CSV
source "$PROJECT_ROOT/src/slurm/run_profile_experiment.sh"
