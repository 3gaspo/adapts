#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --array=0-27%8

# Pilot wrapper. Set DATA_ROOT/WEIGHTS_ROOT when datasets and weights are not
# project-local, in the parent folder, or in the thesis workspace shared root.
# Override with ``sbatch --array=0-1`` when TEST_MODE=true (vanilla + one retrieval).
DEFAULT_DATASETS_CSV="electricity,solar"
DEFAULT_MODELS_CSV="chronos"
DEFAULT_SETTINGS_CSV="168:24,672:168"

source src/slurm/extraction_grid.sh
