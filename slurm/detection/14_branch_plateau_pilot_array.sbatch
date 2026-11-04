#!/bin/bash
#SBATCH --job-name=mrg_branch_imax
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mrg_branch_imax_%A_%a.out
#SBATCH --error=logs/mrg_branch_imax_%A_%a.err

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
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$PARTIAL_DIR"

python -m population.detection.branch_plateau_pilot run-condition \
  --root "$PROJECT" \
  --condition-index "$SLURM_ARRAY_TASK_ID" \
  --manifest "$MANIFEST" \
  --partial-dir "$PARTIAL_DIR"
