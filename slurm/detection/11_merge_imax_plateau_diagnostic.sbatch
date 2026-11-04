#!/bin/bash
#SBATCH --job-name=merge_imax_diag
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_imax_diag_%j.out
#SBATCH --error=logs/merge_imax_diag_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DIAGNOSTIC_ROOT="${IMAX_DIAGNOSTIC_ROOT:-$PROJECT/results/detection/imax_plateau_diagnostic}"

cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

python -m detection.imax_plateau_diagnostic merge \
  --root "$PROJECT" \
  --manifest "$DIAGNOSTIC_ROOT/diagnostic_conditions.csv" \
  --partial-dir "$DIAGNOSTIC_ROOT/partials" \
  --output-dir "$DIAGNOSTIC_ROOT"
