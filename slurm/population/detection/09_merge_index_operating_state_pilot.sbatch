#!/bin/bash
#SBATCH --job-name=merge_index_pilot
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/merge_index_pilot_%j.out
#SBATCH --error=logs/merge_index_pilot_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PILOT_ROOT="${PILOT_ROOT:-$PROJECT/results/population/detection/index_operating_state_pilot}"

cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

python -m population.detection.index_operating_state_pilot merge \
  --root "$PROJECT" \
  --partial-dir "$PILOT_ROOT/partials" \
  --output-dir "$PILOT_ROOT"
