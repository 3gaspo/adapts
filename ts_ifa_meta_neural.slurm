#!/bin/bash
#SBATCH --job-name=tsifa-mn
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Meta-learning evaluation of selected joint-neural candidates.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=ts_ifa_meta_neural
TS_IFA_GRID=selected
TS_IFA_META_FORM=neural
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/src/slurm/selected_candidates.sh"
TS_IFA_CANDIDATES_CSV="${TS_IFA_CANDIDATES_CSV:-$(selected_candidates_csv ts_ifa_neural)}"
: "${TS_IFA_CANDIDATES_CSV:?add neural TS-IFA entries to SWEEP_CANDIDATES.txt or set TS_IFA_CANDIDATES_CSV}"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY TS_IFA_GRID TS_IFA_META_FORM TS_IFA_CANDIDATES_CSV
source "$PROJECT_ROOT/src/slurm/run_ts_ifa.sh"
