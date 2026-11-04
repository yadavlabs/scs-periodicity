#!/usr/bin/env bash
set -euo pipefail

# Final diagnostic-only validation of the peak-first / plateau-fallback I_max
# rule across all frequencies and CVs. Production threshold files are read-only.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
BRANCH_IMAX_OUTPUT_DIR="${BRANCH_IMAX_OUTPUT_DIR:-$PROJECT/results/detection/peak_first_imax_validation_pilot}"
BRANCH_IMAX_PARTIAL_DIR="${BRANCH_IMAX_PARTIAL_DIR:-$BRANCH_IMAX_OUTPUT_DIR/partials}"
BRANCH_IMAX_MANIFEST="${BRANCH_IMAX_MANIFEST:-$BRANCH_IMAX_OUTPUT_DIR/pilot_conditions.csv}"
VALIDATION_FREQUENCIES="${VALIDATION_FREQUENCIES:-10 20 50 100 200}"
VALIDATION_CVS="${VALIDATION_CVS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
VALIDATION_REPETITIONS_CV0="${VALIDATION_REPETITIONS_CV0:-1}"
VALIDATION_REPETITIONS_NONZERO="${VALIDATION_REPETITIONS_NONZERO:-3}"
VALIDATION_HIGH_BRANCH_CUTOFF_UA="${VALIDATION_HIGH_BRANCH_CUTOFF_UA:-800}"
MAX_PARALLEL_BRANCH_IMAX="${MAX_PARALLEL_BRANCH_IMAX:-32}"
export PROJECT BRANCH_IMAX_OUTPUT_DIR BRANCH_IMAX_PARTIAL_DIR BRANCH_IMAX_MANIFEST

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$BRANCH_IMAX_OUTPUT_DIR" "$BRANCH_IMAX_PARTIAL_DIR"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

read -r -a FREQUENCY_ARGS <<< "$VALIDATION_FREQUENCIES"
read -r -a CV_ARGS <<< "$VALIDATION_CVS"
python -m detection.branch_plateau_pilot prepare-peak-validation \
  --root "$PROJECT" \
  --output-dir "$BRANCH_IMAX_OUTPUT_DIR" \
  --frequencies "${FREQUENCY_ARGS[@]}" \
  --cvs "${CV_ARGS[@]}" \
  --repetitions-cv0 "$VALIDATION_REPETITIONS_CV0" \
  --repetitions-nonzero "$VALIDATION_REPETITIONS_NONZERO" \
  --high-branch-cutoff-uA "$VALIDATION_HIGH_BRANCH_CUTOFF_UA"
N_CONDITIONS=$(python -m detection.branch_plateau_pilot count \
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

echo "Submitted peak-first I_max validation: mechanisms=$JID_MOD array=$JID_PILOT merge=$JID_MERGE"
echo "Conditions: $N_CONDITIONS; outputs: $BRANCH_IMAX_OUTPUT_DIR"
echo "No production threshold or response file will be modified."
