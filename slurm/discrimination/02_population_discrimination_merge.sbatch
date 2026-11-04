#!/bin/bash
#SBATCH --job-name=pop_disc_merge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pop_disc_merge_%A.out
#SBATCH --error=logs/pop_disc_merge_%A.err

set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/results/partials/discrimination_v4_manuscript_regularized}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/results/discrimination}"
mkdir -p "$OUT_ROOT"

python -u src/population/discrimination/discrimination_CLI.py merge \
  --partial-root "$PARTIAL_ROOT" \
  --out-root "$OUT_ROOT"

echo "[merge] population observer outputs are in $OUT_ROOT"
