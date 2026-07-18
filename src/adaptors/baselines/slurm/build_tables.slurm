#!/bin/bash
#SBATCH --job-name=tables
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

# Canonical table builder; the copies beside gates and TS-IFA delegate here too.
source src/slurm/build_tables.sh
