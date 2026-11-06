#!/bin/bash
#SBATCH --job-name=mrg_pop_merge_resp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mrg_pop_merge_resp_%j.out
#SBATCH --error=logs/mrg_pop_merge_resp_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROTOCOL="${DETECTION_PROTOCOL_VERSION:-canonical-cv0_tol0p01_peakfirst-imax_v3}"
INDEX_ID="${EXPECTED_INDEX_AXON_ID:-S01_140um_ph1}"
PARTIAL_DIR="$PROJECT/data/detection/partials/$PROTOCOL/$INDEX_ID/population-responses"
OUT="$PROJECT/data/detection/population_responses.csv"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(--root "$PROJECT" --mode true --partial-dir "$PARTIAL_DIR" --output "$OUT")
if [[ -n "${POPULATION_MANIFEST:-}" ]]; then
  ARGS+=(--population-manifest "$POPULATION_MANIFEST")
fi
python -m detection.detection_CLI merge-population-recruitment "${ARGS[@]}"
