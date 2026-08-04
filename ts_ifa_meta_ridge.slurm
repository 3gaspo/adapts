#!/bin/bash
#SBATCH --job-name=tsifa-mr
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# T1 support/query meta-learning with exact ridge adaptation; exact T2 ridge fit.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
TS_IFA_VARIANT=meta_ridge
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE TS_IFA_VARIANT
source "$PROJECT_ROOT/src/slurm/common.sh"
require_scale_experiment_mode || exit $?
source "$PROJECT_ROOT/src/slurm/run_ts_ifa.sh"
