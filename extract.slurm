#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --array=0-1%2

# Safe default: a two-task smoke extraction. Override both PROFILE and --array:
# pilot: PROFILE=pilot sbatch --array=0-27%8 extract.slurm
# full:  PROFILE=full sbatch --array=0-783%8 extract.slurm
PROFILE="${PROFILE:-test}" # test, pilot, or full

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT PROFILE
source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
