#!/bin/bash
#SBATCH --job-name=mrg_pop_merge_regimes
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mrg_pop_merge_regimes_%j.out
#SBATCH --error=logs/mrg_pop_merge_regimes_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROTOCOL="${DETECTION_PROTOCOL_VERSION:-canonical-cv0_tol0p01_peakfirst-imax_v3}"
INDEX_ID="${EXPECTED_INDEX_AXON_ID:-S01_140um_ph1}"
PARTIAL_DIR="$PROJECT/results/detection/partials/$PROTOCOL/$INDEX_ID/index-regimes"
OUT_REGIMES="$PROJECT/data/detection/index_regime_amplitudes.csv"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

python -m detection.detection_CLI merge-index-regimes \
  --root "$PROJECT" --partial-dir "$PARTIAL_DIR" --output "$OUT_REGIMES"
python -m detection.detection_CLI compute-index-cv0 --root "$PROJECT" --force
