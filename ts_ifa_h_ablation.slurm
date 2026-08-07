#!/bin/bash
#SBATCH --job-name=tsifa-h
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40000

# Horizon ablation at L=504 for manually selected joint candidates.
: "${TS_IFA_CANDIDATES_CSV:?copy candidates from TS_IFA_CANDIDATES.txt into TS_IFA_CANDIDATES_CSV}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-full}"
EXPERIMENT_FAMILY=ts_ifa_h_ablation
TS_IFA_GRID=selected
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT EXPERIMENT_MODE EXPERIMENT_FAMILY TS_IFA_GRID TS_IFA_CANDIDATES_CSV
source "$PROJECT_ROOT/src/slurm/run_ts_ifa.sh"
