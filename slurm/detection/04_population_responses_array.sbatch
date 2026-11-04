#!/bin/bash
#SBATCH --job-name=mrg_pop_resp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/mrg_pop_resp_%A_%a.out
#SBATCH --error=logs/mrg_pop_resp_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROTOCOL="${DETECTION_PROTOCOL_VERSION:-canonical-cv0_tol0p01_peakfirst-imax_v3}"
INDEX_ID="${EXPECTED_INDEX_AXON_ID:-S01_140um_ph1}"
PARTIAL_DIR="$PROJECT/results/detection/partials/$PROTOCOL/$INDEX_ID/population-responses"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$PARTIAL_DIR"

# One task = one cutline x one CV/frequency condition. Diameters run largest
# first in isolated child processes, allowing audited smaller-diameter pruning.
ARGS=(
  --root "$PROJECT" --mode true --trajectory-job-index "$SLURM_ARRAY_TASK_ID"
  --n-jobs "$SLURM_CPUS_PER_TASK" --partial-dir "$PARTIAL_DIR"
  --max-pruning-audit-rate "${MAX_PRUNING_AUDIT_RATE:-0.01}"
  --diameter-pruning-audit-rate "${DIAMETER_PRUNING_AUDIT_RATE:-0.01}"
)
if [[ -n "${POPULATION_MANIFEST:-}" ]]; then
  ARGS+=(--population-manifest "$POPULATION_MANIFEST")
fi
python -m population.detection.detection_CLI run-population-trajectory-condition "${ARGS[@]}"
