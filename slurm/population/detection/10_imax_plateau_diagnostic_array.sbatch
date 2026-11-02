#!/bin/bash
#SBATCH --job-name=mrg_imax_diag
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mrg_imax_diag_%A_%a.out
#SBATCH --error=logs/mrg_imax_diag_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DIAGNOSTIC_ROOT="${IMAX_DIAGNOSTIC_ROOT:-$PROJECT/results/population/detection/imax_plateau_diagnostic}"
CONDITIONS="$DIAGNOSTIC_ROOT/diagnostic_conditions.csv"
PARTIAL_DIR="$DIAGNOSTIC_ROOT/partials"

cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$PARTIAL_DIR" "$PROJECT/logs"

python -m population.detection.imax_plateau_diagnostic run-condition \
  --root "$PROJECT" \
  --condition-index "$SLURM_ARRAY_TASK_ID" \
  --manifest "$CONDITIONS" \
  --partial-dir "$PARTIAL_DIR" \
  --coarse-min-uA "${IMAX_COARSE_MIN_UA:-200}" \
  --coarse-max-uA "${IMAX_COARSE_MAX_UA:-2000}" \
  --coarse-step-uA "${IMAX_COARSE_STEP_UA:-25}" \
  --fine-step-uA "${IMAX_FINE_STEP_UA:-0.5}" \
  --fine-half-width-uA "${IMAX_FINE_HALF_WIDTH_UA:-25}" \
  --late-transition-min-uA "${IMAX_LATE_TRANSITION_MIN_UA:-500}" \
  --max-late-transitions "${IMAX_MAX_LATE_TRANSITIONS:-4}" \
  --plateau-min-width-uA "${IMAX_PLATEAU_MIN_WIDTH_UA:-100}" \
  --plateau-min-fraction-of-max "${IMAX_PLATEAU_MIN_FRACTION:-0.90}" \
  --n-jobs "$SLURM_CPUS_PER_TASK"
