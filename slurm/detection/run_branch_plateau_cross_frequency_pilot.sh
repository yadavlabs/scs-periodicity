#!/usr/bin/env bash
set -euo pipefail

# Diagnostic only: sample historical high-amplitude I_max branches and matched
# ordinary controls at every detection frequency. Every tested amplitude,
# spike count, and spike-time array is retained for later example figures.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BRANCH_IMAX_OUTPUT_DIR="${BRANCH_IMAX_OUTPUT_DIR:-$PROJECT/results/detection/branch_plateau_cross_frequency_pilot}"
BRANCH_IMAX_PARTIAL_DIR="${BRANCH_IMAX_PARTIAL_DIR:-$BRANCH_IMAX_OUTPUT_DIR/partials}"
BRANCH_IMAX_MANIFEST="${BRANCH_IMAX_MANIFEST:-$BRANCH_IMAX_OUTPUT_DIR/pilot_conditions.csv}"
CROSS_BRANCH_FREQUENCIES="${CROSS_BRANCH_FREQUENCIES:-10 20 50 100 200}"
CROSS_BRANCH_HIGH_CUTOFF_UA="${CROSS_BRANCH_HIGH_CUTOFF_UA:-800}"
CROSS_BRANCH_MAX_HIGH_PER_FREQUENCY="${CROSS_BRANCH_MAX_HIGH_PER_FREQUENCY:-8}"
CROSS_BRANCH_CONTROLS_PER_FREQUENCY="${CROSS_BRANCH_CONTROLS_PER_FREQUENCY:-4}"
MAX_PARALLEL_BRANCH_IMAX="${MAX_PARALLEL_BRANCH_IMAX:-32}"
export PROJECT BRANCH_IMAX_OUTPUT_DIR BRANCH_IMAX_PARTIAL_DIR BRANCH_IMAX_MANIFEST

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$BRANCH_IMAX_OUTPUT_DIR" "$BRANCH_IMAX_PARTIAL_DIR"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a FREQUENCY_ARGS <<< "$CROSS_BRANCH_FREQUENCIES"
python -m population.detection.branch_plateau_pilot prepare-cross-frequency \
  --root "$PROJECT" \
  --output-dir "$BRANCH_IMAX_OUTPUT_DIR" \
  --frequencies "${FREQUENCY_ARGS[@]}" \
  --high-branch-cutoff-uA "$CROSS_BRANCH_HIGH_CUTOFF_UA" \
  --max-high-per-frequency "$CROSS_BRANCH_MAX_HIGH_PER_FREQUENCY" \
  --controls-per-frequency "$CROSS_BRANCH_CONTROLS_PER_FREQUENCY"
N_CONDITIONS=$(python -m population.detection.branch_plateau_pilot count \
  --manifest "$BRANCH_IMAX_MANIFEST" | tail -n 1)

UPSTREAM_ARGS=()
if [[ -n "${PIPELINE_DEPENDENCY:-}" ]]; then
  UPSTREAM_ARGS=(--dependency=afterok:"$PIPELINE_DEPENDENCY")
fi
JID_MOD=$(sbatch --parsable "${UPSTREAM_ARGS[@]}" --export=ALL \
  slurm/detection/00_compile_mechanisms.sbatch)
JID_MOD="${JID_MOD%%;*}"
JID_PILOT=$(sbatch --parsable --dependency=afterok:"$JID_MOD" \
  --array="0-$((N_CONDITIONS - 1))%${MAX_PARALLEL_BRANCH_IMAX}" --export=ALL \
  slurm/detection/14_branch_plateau_pilot_array.sbatch)
JID_PILOT="${JID_PILOT%%;*}"
JID_MERGE=$(sbatch --parsable --dependency=afterok:"$JID_PILOT" --export=ALL \
  slurm/detection/15_merge_branch_plateau_pilot.sbatch)
JID_MERGE="${JID_MERGE%%;*}"

echo "Submitted cross-frequency I_max pilot: mechanisms=$JID_MOD array=$JID_PILOT merge=$JID_MERGE"
echo "Conditions: $N_CONDITIONS; outputs: $BRANCH_IMAX_OUTPUT_DIR"
echo "No production threshold or response file will be modified."
