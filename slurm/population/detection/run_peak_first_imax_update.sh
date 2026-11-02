#!/usr/bin/env bash
set -euo pipefail

# Selectively recompute I_max for every repetition, then write an audited staged
# index-regime table. Production data and downstream recruitment are untouched.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MAX_PARALLEL_IMAX="${MAX_PARALLEL_IMAX:-41}"
PEAK_FIRST_IMAX_OUTPUT_DIR="${PEAK_FIRST_IMAX_OUTPUT_DIR:-$PROJECT/results/population/detection/peak_first_imax_update}"
PEAK_FIRST_IMAX_PARTIAL_DIR="${PEAK_FIRST_IMAX_PARTIAL_DIR:-$PEAK_FIRST_IMAX_OUTPUT_DIR/partials}"
export PROJECT MAX_PARALLEL_IMAX PEAK_FIRST_IMAX_OUTPUT_DIR PEAK_FIRST_IMAX_PARTIAL_DIR
if [[ -n "${PEAK_FIRST_IMAX_BASE_UPDATES_CSV:-}" ]]; then
  export PEAK_FIRST_IMAX_BASE_UPDATES_CSV
fi

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$PEAK_FIRST_IMAX_OUTPUT_DIR" "$PEAK_FIRST_IMAX_PARTIAL_DIR"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_ARGS=()
if [[ -n "${PEAK_FIRST_IMAX_SOURCE_CSV:-}" ]]; then
  SOURCE_ARGS=(--source-csv "$PEAK_FIRST_IMAX_SOURCE_CSV")
  export PEAK_FIRST_IMAX_SOURCE_CSV
fi

python -m population.detection.stable_imax_update prepare \
  --root "$PROJECT" --output-dir "$PEAK_FIRST_IMAX_OUTPUT_DIR" \
  "${SOURCE_ARGS[@]}"
N_CONDITIONS=$(python -m population.detection.stable_imax_update count | tail -n 1)
ARRAY_SPEC="${PEAK_FIRST_IMAX_ARRAY_SPEC:-0-$((N_CONDITIONS - 1))}"

UPSTREAM_ARGS=()
if [[ -n "${PIPELINE_DEPENDENCY:-}" ]]; then
  UPSTREAM_ARGS=(--dependency=afterok:"$PIPELINE_DEPENDENCY")
fi
JID_MOD=$(sbatch --parsable "${UPSTREAM_ARGS[@]}" --export=ALL \
  slurm/population/detection/00_compile_mechanisms.sbatch)
JID_MOD="${JID_MOD%%;*}"
JID_IMAX=$(sbatch --parsable --dependency=afterok:"$JID_MOD" \
  --array="${ARRAY_SPEC}%${MAX_PARALLEL_IMAX}" --export=ALL \
  slurm/population/detection/16_peak_first_imax_update_array.sbatch)
JID_IMAX="${JID_IMAX%%;*}"
JID_MERGE=$(sbatch --parsable --dependency=afterok:"$JID_IMAX" --export=ALL \
  slurm/population/detection/17_merge_peak_first_imax_update.sbatch)
JID_MERGE="${JID_MERGE%%;*}"

echo "Submitted peak-first I_max update: mechanisms=$JID_MOD array=$JID_IMAX merge=$JID_MERGE"
echo "Staged outputs: $PEAK_FIRST_IMAX_OUTPUT_DIR"
echo "No production table will be overwritten."
