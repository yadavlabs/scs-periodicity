#!/bin/bash
#SBATCH --job-name=mrg_stable_imax
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mrg_stable_imax_%A_%a.out
#SBATCH --error=logs/mrg_stable_imax_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_DIR="${STABLE_IMAX_OUTPUT_DIR:-$PROJECT/results/detection/stable_imax_update}"
PARTIAL_DIR="${STABLE_IMAX_PARTIAL_DIR:-$OUTPUT_DIR/partials}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$PARTIAL_DIR"

SOURCE_ARGS=()
if [[ -n "${STABLE_IMAX_SOURCE_CSV:-}" ]]; then
  SOURCE_ARGS=(--source-csv "$STABLE_IMAX_SOURCE_CSV")
fi

python -m population.detection.stable_imax_update run-condition \
  --root "$PROJECT" \
  --condition-index "$SLURM_ARRAY_TASK_ID" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --partial-dir "$PARTIAL_DIR" \
  "${SOURCE_ARGS[@]}"
