#!/usr/bin/env bash
set -euo pipefail

# This diagnostic does not modify production index thresholds or population
# responses. Eight array tasks cover CV0 rep 0 and one representative regular
# repetition at 20, 50, 100, and 200 Hz.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PILOT_ROOT="${PILOT_ROOT:-$PROJECT/results/detection/index_operating_state_pilot}"
export PROJECT PILOT_ROOT

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$PILOT_ROOT/partials"

module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

python -m detection.index_operating_state_pilot preflight --root "$PROJECT"

UPSTREAM=()
if [[ -n "${PILOT_DEPENDENCY:-}" ]]; then
  UPSTREAM=(--dependency=afterok:"$PILOT_DEPENDENCY")
fi

JID_MOD=$(sbatch --parsable "${UPSTREAM[@]}" --export=ALL slurm/detection/00_compile_mechanisms.sbatch)
JID_MOD="${JID_MOD%%;*}"
JID_PILOT=$(sbatch --parsable --dependency=afterok:"$JID_MOD" --array=0-7%8 --export=ALL slurm/detection/08_index_operating_state_pilot_array.sbatch)
JID_PILOT="${JID_PILOT%%;*}"
JID_MERGE=$(sbatch --parsable --dependency=afterok:"$JID_PILOT" --export=ALL slurm/detection/09_merge_index_operating_state_pilot.sbatch)
JID_MERGE="${JID_MERGE%%;*}"

echo "Submitted index operating-state pilot: mechanisms=$JID_MOD array=$JID_PILOT merge=$JID_MERGE"
echo "Monitor with: squeue -j $JID_MOD,$JID_PILOT,$JID_MERGE"
echo "Results: $PILOT_ROOT"
