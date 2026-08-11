#!/bin/bash
#SBATCH --job-name=publish
#SBATCH --output=logs/publish_%j.out
#SBATCH --error=logs/publish_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT
source "$PROJECT_ROOT/src/slurm/publish_results.sh"
publish_results_main "$@"
