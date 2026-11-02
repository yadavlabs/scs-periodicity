#!/usr/bin/env bash
set -euo pipefail

# Diagnostic only: selected 200-Hz high-CV index-axon repetitions are swept
# through the safety domain. Candidate I_max thresholds use only local plateau
# and pre-decline criteria; no production table is modified.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BRANCH_IMAX_OUTPUT_DIR="${BRANCH_IMAX_OUTPUT_DIR:-$PROJECT/results/population/detection/branch_plateau_pilot}"
BRANCH_IMAX_PARTIAL_DIR="${BRANCH_IMAX_PARTIAL_DIR:-$BRANCH_IMAX_OUTPUT_DIR/partials}"
BRANCH_IMAX_MANIFEST="${BRANCH_IMAX_MANIFEST:-$BRANCH_IMAX_OUTPUT_DIR/pilot_conditions.csv}"
BRANCH_IMAX_CVS="${BRANCH_IMAX_CVS:-0.8 0.9 1.0}"
BRANCH_IMAX_REPETITIONS_PER_CV="${BRANCH_IMAX_REPETITIONS_PER_CV:-8}"
MAX_PARALLEL_BRANCH_IMAX="${MAX_PARALLEL_BRANCH_IMAX:-24}"
export PROJECT BRANCH_IMAX_OUTPUT_DIR BRANCH_IMAX_PARTIAL_DIR BRANCH_IMAX_MANIFEST

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$BRANCH_IMAX_OUTPUT_DIR" "$BRANCH_IMAX_PARTIAL_DIR"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a CV_ARGS <<< "$BRANCH_IMAX_CVS"
python -m population.detection.branch_plateau_pilot prepare \
  --root "$PROJECT" \
  --output-dir "$BRANCH_IMAX_OUTPUT_DIR" \
  --cvs "${CV_ARGS[@]}" \
  --repetitions-per-cv "$BRANCH_IMAX_REPETITIONS_PER_CV"
N_CONDITIONS=$(python -m population.detection.branch_plateau_pilot count \
  --manifest "$BRANCH_IMAX_MANIFEST" | tail -n 1)

UPSTREAM_ARGS=()
if [[ -n "${PIPELINE_DEPENDENCY:-}" ]]; then
  UPSTREAM_ARGS=(--dependency=afterok:"$PIPELINE_DEPENDENCY")
fi
JID_MOD=$(sbatch --parsable "${UPSTREAM_ARGS[@]}" --export=ALL \
  slurm/population/detection/00_compile_mechanisms.sbatch)
JID_MOD="${JID_MOD%%;*}"
JID_PILOT=$(sbatch --parsable --dependency=afterok:"$JID_MOD" \
  --array="0-$((N_CONDITIONS - 1))%${MAX_PARALLEL_BRANCH_IMAX}" --export=ALL \
  slurm/population/detection/14_branch_plateau_pilot_array.sbatch)
JID_PILOT="${JID_PILOT%%;*}"
JID_MERGE=$(sbatch --parsable --dependency=afterok:"$JID_PILOT" --export=ALL \
  slurm/population/detection/15_merge_branch_plateau_pilot.sbatch)
JID_MERGE="${JID_MERGE%%;*}"

echo "Submitted branch-aware I_max pilot: mechanisms=$JID_MOD array=$JID_PILOT merge=$JID_MERGE"
echo "Conditions: $N_CONDITIONS; outputs: $BRANCH_IMAX_OUTPUT_DIR"
echo "No production threshold or response file will be modified."
