#!/bin/bash
#SBATCH --job-name=merge_branch_imax
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_branch_imax_%j.out
#SBATCH --error=logs/merge_branch_imax_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_DIR="${BRANCH_IMAX_OUTPUT_DIR:-$PROJECT/results/detection/branch_plateau_pilot}"
PARTIAL_DIR="${BRANCH_IMAX_PARTIAL_DIR:-$OUTPUT_DIR/partials}"
MANIFEST="${BRANCH_IMAX_MANIFEST:-$OUTPUT_DIR/pilot_conditions.csv}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m population.detection.branch_plateau_pilot merge \
  --manifest "$MANIFEST" \
  --partial-dir "$PARTIAL_DIR" \
  --output-dir "$OUTPUT_DIR"
