#!/bin/bash
#SBATCH --job-name=tsifa
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40000
#SBATCH --array=0-23%4

# Optional TS-IFA front end; use TEST_MODE=true with --array=0-0 for smoke.
TEST_MODE="${TEST_MODE:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEST_MODE
source "$SCRIPT_DIR/src/slurm/run_ts_ifa.sh"
