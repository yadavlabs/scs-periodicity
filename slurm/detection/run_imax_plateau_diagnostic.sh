#!/usr/bin/env bash
set -euo pipefail

# Isolated index-axon diagnostic. Production thresholds and population
# responses are read-only. Defaults select two high-I_max and two target-matched
# ordinary repetitions at CV=1 for 20, 50, and 100 Hz (12 array tasks).

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DIAGNOSTIC_ROOT="${IMAX_DIAGNOSTIC_ROOT:-$PROJECT/results/detection/imax_plateau_diagnostic}"
FREQUENCY_TEXT="${IMAX_DIAGNOSTIC_FREQUENCIES:-20,50,100}"
CV="${IMAX_DIAGNOSTIC_CV:-1.0}"
BRANCH_CUTOFF="${IMAX_BRANCH_CUTOFF_UA:-800}"
REPETITIONS_PER_BRANCH="${IMAX_REPETITIONS_PER_BRANCH:-2}"
MAX_CONCURRENT="${IMAX_MAX_CONCURRENT:-6}"
export PROJECT IMAX_DIAGNOSTIC_ROOT="$DIAGNOSTIC_ROOT"

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$DIAGNOSTIC_ROOT/partials"

module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

IFS=',' read -r -a FREQUENCIES <<< "$FREQUENCY_TEXT"
python -m detection.imax_plateau_diagnostic prepare \
  --root "$PROJECT" \
  --output-dir "$DIAGNOSTIC_ROOT" \
  --frequencies "${FREQUENCIES[@]}" \
  --cv "$CV" \
  --branch-cutoff-uA "$BRANCH_CUTOFF" \
  --repetitions-per-branch "$REPETITIONS_PER_BRANCH"

CONDITIONS="$DIAGNOSTIC_ROOT/diagnostic_conditions.csv"
N_CONDITIONS=$(python -m detection.imax_plateau_diagnostic count --manifest "$CONDITIONS")
if [[ "$N_CONDITIONS" -lt 1 ]]; then
  echo "No diagnostic conditions were selected" >&2
  exit 2
fi
ARRAY_END=$((N_CONDITIONS - 1))

UPSTREAM=()
if [[ -n "${IMAX_DIAGNOSTIC_DEPENDENCY:-}" ]]; then
  UPSTREAM=(--dependency=afterok:"$IMAX_DIAGNOSTIC_DEPENDENCY")
fi

JID_MOD=$(sbatch --parsable "${UPSTREAM[@]}" --export=ALL slurm/detection/00_compile_mechanisms.sbatch)
JID_MOD="${JID_MOD%%;*}"
JID_ARRAY=$(sbatch --parsable --dependency=afterok:"$JID_MOD" \
  --array="0-${ARRAY_END}%${MAX_CONCURRENT}" --export=ALL \
  slurm/detection/10_imax_plateau_diagnostic_array.sbatch)
JID_ARRAY="${JID_ARRAY%%;*}"
JID_MERGE=$(sbatch --parsable --dependency=afterok:"$JID_ARRAY" --export=ALL \
  slurm/detection/11_merge_imax_plateau_diagnostic.sbatch)
JID_MERGE="${JID_MERGE%%;*}"

echo "Submitted I_max diagnostic: mechanisms=$JID_MOD array=$JID_ARRAY merge=$JID_MERGE"
echo "Conditions: $N_CONDITIONS; concurrent array tasks: $MAX_CONCURRENT"
echo "Monitor: squeue -j $JID_MOD,$JID_ARRAY,$JID_MERGE"
echo "Results: $DIAGNOSTIC_ROOT"
