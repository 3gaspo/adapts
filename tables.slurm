#!/bin/bash
#SBATCH --job-name=tables
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

# Submit after the baseline/gate jobs; set TEST_MODE=true after smoke runs.
TEST_MODE="${TEST_MODE:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_MODE
source "$SCRIPT_DIR/src/slurm/build_tables.sh"
