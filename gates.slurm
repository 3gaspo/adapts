#!/bin/bash
#SBATCH --job-name=gates
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20000
#SBATCH --array=0-671%32

# Submit this file from the project root; use TEST_MODE=true with --array=0-0 for smoke.
TEST_MODE="${TEST_MODE:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_MODE
source "$SCRIPT_DIR/src/slurm/run_gates.sh"
