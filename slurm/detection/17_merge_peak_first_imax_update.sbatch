#!/bin/bash
#SBATCH --job-name=merge_peak_imax
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_peak_imax_%j.out
#SBATCH --error=logs/merge_peak_imax_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_DIR="${PEAK_FIRST_IMAX_OUTPUT_DIR:-$PROJECT/results/detection/peak_first_imax_update}"
PARTIAL_DIR="${PEAK_FIRST_IMAX_PARTIAL_DIR:-$OUTPUT_DIR/partials}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_ARGS=()
if [[ -n "${PEAK_FIRST_IMAX_SOURCE_CSV:-}" ]]; then
  SOURCE_ARGS=(--source-csv "$PEAK_FIRST_IMAX_SOURCE_CSV")
fi
BASE_ARGS=()
if [[ -n "${PEAK_FIRST_IMAX_BASE_UPDATES_CSV:-}" ]]; then
  BASE_ARGS=(--base-updates-csv "$PEAK_FIRST_IMAX_BASE_UPDATES_CSV")
fi

python -m population.detection.stable_imax_update merge \
  --root "$PROJECT" \
  --partial-dir "$PARTIAL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "${BASE_ARGS[@]}" \
  "${SOURCE_ARGS[@]}"
