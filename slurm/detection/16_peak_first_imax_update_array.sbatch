#!/bin/bash
#SBATCH --job-name=mrg_peak_imax
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mrg_peak_imax_%A_%a.out
#SBATCH --error=logs/mrg_peak_imax_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_DIR="${PEAK_FIRST_IMAX_OUTPUT_DIR:-$PROJECT/results/detection/peak_first_imax_update}"
PARTIAL_DIR="${PEAK_FIRST_IMAX_PARTIAL_DIR:-$OUTPUT_DIR/partials}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$PARTIAL_DIR"

SOURCE_ARGS=()
if [[ -n "${PEAK_FIRST_IMAX_SOURCE_CSV:-}" ]]; then
  SOURCE_ARGS=(--source-csv "$PEAK_FIRST_IMAX_SOURCE_CSV")
fi
BASE_ARGS=()
if [[ -n "${PEAK_FIRST_IMAX_BASE_UPDATES_CSV:-}" ]]; then
  BASE_ARGS=(--base-updates-csv "$PEAK_FIRST_IMAX_BASE_UPDATES_CSV")
fi

python -m detection.stable_imax_update run-condition \
  --root "$PROJECT" \
  --condition-index "$SLURM_ARRAY_TASK_ID" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --partial-dir "$PARTIAL_DIR" \
  "${BASE_ARGS[@]}" \
  "${SOURCE_ARGS[@]}"
