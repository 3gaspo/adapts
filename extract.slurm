#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --array=0-783%8

# Submit this file from the project root. Edit resources and the extraction profile here.
# test has 2 tasks:  sbatch --array=0-1 extract.slurm
# pilot has 28:     sbatch --array=0-27 extract.slurm
# full has 784 and matches the default array above.
PROFILE="${PROFILE:-full}" # test, pilot, or full

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT PROFILE
source "$PROJECT_ROOT/src/slurm/extract_adaptation.sh"
