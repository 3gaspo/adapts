#!/bin/bash
#SBATCH --job-name=tables
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000

# Select the same profile and winner-method environment used by upstream jobs.
# For TS-IFA, set FAMILIES_CSV=ts_ifa and METHODS_CSV to the completed current
# variants (joint_ridge, joint_neural, meta_ridge, and/or meta_neural).
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
EXPERIMENT_FAMILY="${EXPERIMENT_FAMILY:-benchmark}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY
source "$PROJECT_ROOT/src/slurm/common.sh"
require_scale_experiment_mode || exit $?
source "$PROJECT_ROOT/src/slurm/build_tables.sh"
