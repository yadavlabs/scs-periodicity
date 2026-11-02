#!/usr/bin/env bash
set -euo pipefail

# Phase 1 of the selective I_max migration. This recomputes only index-axon
# I_max thresholds and creates an audited staged table. It does not overwrite
# production data or launch population recruitment.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MAX_PARALLEL_IMAX="${MAX_PARALLEL_IMAX:-41}"
STABLE_IMAX_OUTPUT_DIR="${STABLE_IMAX_OUTPUT_DIR:-$PROJECT/results/population/detection/stable_imax_update}"
STABLE_IMAX_PARTIAL_DIR="${STABLE_IMAX_PARTIAL_DIR:-$STABLE_IMAX_OUTPUT_DIR/partials}"
export PROJECT MAX_PARALLEL_IMAX STABLE_IMAX_OUTPUT_DIR STABLE_IMAX_PARTIAL_DIR

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$STABLE_IMAX_OUTPUT_DIR" "$STABLE_IMAX_PARTIAL_DIR"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_ARGS=()
if [[ -n "${STABLE_IMAX_SOURCE_CSV:-}" ]]; then
  SOURCE_ARGS=(--source-csv "$STABLE_IMAX_SOURCE_CSV")
  export STABLE_IMAX_SOURCE_CSV
fi

python -m population.detection.stable_imax_update prepare \
  --root "$PROJECT" --output-dir "$STABLE_IMAX_OUTPUT_DIR" \
  "${SOURCE_ARGS[@]}"
N_CONDITIONS=$(python -m population.detection.stable_imax_update count | tail -n 1)

UPSTREAM_ARGS=()
if [[ -n "${PIPELINE_DEPENDENCY:-}" ]]; then
  UPSTREAM_ARGS=(--dependency=afterok:"$PIPELINE_DEPENDENCY")
fi
JID_MOD=$(sbatch --parsable "${UPSTREAM_ARGS[@]}" --export=ALL \
  slurm/population/detection/00_compile_mechanisms.sbatch)
JID_MOD="${JID_MOD%%;*}"
JID_IMAX=$(sbatch --parsable --dependency=afterok:"$JID_MOD" \
  --array="0-$((N_CONDITIONS - 1))%${MAX_PARALLEL_IMAX}" --export=ALL \
  slurm/population/detection/12_stable_imax_update_array.sbatch)
JID_IMAX="${JID_IMAX%%;*}"
JID_MERGE=$(sbatch --parsable --dependency=afterok:"$JID_IMAX" --export=ALL \
  slurm/population/detection/13_merge_stable_imax_update.sbatch)
JID_MERGE="${JID_MERGE%%;*}"

echo "Submitted stable-I_max threshold update: mechanisms=$JID_MOD array=$JID_IMAX merge=$JID_MERGE"
echo "Staged outputs: $STABLE_IMAX_OUTPUT_DIR"
echo "No production table will be overwritten."
