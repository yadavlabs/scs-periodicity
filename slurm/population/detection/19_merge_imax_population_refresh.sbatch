#!/bin/bash
#SBATCH --job-name=imax_pop_merge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/imax_pop_merge_%j.out
#SBATCH --error=logs/imax_pop_merge_%j.err
set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
# Stage only: audit before explicit production promotion.
python -m population.detection.refresh_imax_population merge --root "$PROJECT"
