#!/bin/bash
#SBATCH --job-name=mrg_index_pilot
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mrg_index_pilot_%A_%a.out
#SBATCH --error=logs/mrg_index_pilot_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PILOT_ROOT="${PILOT_ROOT:-$PROJECT/results/population/detection/index_operating_state_pilot}"
PILOT_TOL_UA="${PILOT_TOL_UA:-0.01}"
PILOT_SWEEP_STEP_UA="${PILOT_SWEEP_STEP_UA:-0.01}"
PILOT_SWEEP_HALF_WIDTH_UA="${PILOT_SWEEP_HALF_WIDTH_UA:-0.20}"

cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$PILOT_ROOT/partials" "$PROJECT/logs"

python -m population.detection.index_operating_state_pilot run-condition \
  --root "$PROJECT" \
  --condition-index "$SLURM_ARRAY_TASK_ID" \
  --partial-dir "$PILOT_ROOT/partials" \
  --tol-uA "$PILOT_TOL_UA" \
  --sweep-step-uA "$PILOT_SWEEP_STEP_UA" \
  --sweep-half-width-uA "$PILOT_SWEEP_HALF_WIDTH_UA" \
  --n-jobs "$SLURM_CPUS_PER_TASK"
