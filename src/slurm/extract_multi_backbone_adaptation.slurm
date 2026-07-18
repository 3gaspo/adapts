#!/bin/bash
#SBATCH --job-name=multiextract
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --array=0-783%8

# Publication grid. Set DATA_ROOT/WEIGHTS_ROOT when datasets and weights are not
# project-local, in the parent folder, or in the thesis workspace shared root.
# Override with ``sbatch --array=0-1`` when TEST_MODE=true or with the matching
# task range when narrowing the CSV grid variables.
DEFAULT_DATASETS_CSV="ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity,Exchange"
# TS-ICL is deliberately excluded until its backbone is implemented and registered.
DEFAULT_MODELS_CSV="chronos,tabpfnts"
# 572:64 is intentional: it provides the comparison with Cross-RAG.
DEFAULT_SETTINGS_CSV="572:64,672:24,672:48,672:168,672:336,672:672,168:24,336:24"

source src/slurm/extraction_grid.sh
