#!/bin/bash
#SBATCH --job-name=mrg_pop_stv
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mrg_pop_stv_%A_%a.out
#SBATCH --error=logs/mrg_pop_stv_%A_%a.err

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

# One task creates/validates all 55 CV/frequency files for one axon.
python -m space_time_v.spacetime_core run-axon \
  --root "$PROJECT" --job-index "$SLURM_ARRAY_TASK_ID"
