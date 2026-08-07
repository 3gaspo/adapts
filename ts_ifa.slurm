#!/bin/bash
#SBATCH --job-name=tsifa-grid
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Complete non-meta routing grid. Pilot is Electricity (336,48); full uses the
# complete full dataset/setting profile. Both use raw/instance retrieval with K=3.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-test}"
EXPERIMENT_FAMILY=ts_ifa
TS_IFA_GRID=complete
NEIGHBORS_CSV="${NEIGHBORS_CSV:-3}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY TS_IFA_GRID NEIGHBORS_CSV
case "$EXPERIMENT_MODE" in test|full|ultra) ;; *) echo "ts_ifa.slurm supports test, full, or ultra" >&2; exit 2 ;; esac
source "$PROJECT_ROOT/src/slurm/run_ts_ifa.sh"
