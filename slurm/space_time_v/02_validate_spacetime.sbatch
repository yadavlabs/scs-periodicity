#!/bin/bash
#SBATCH --job-name=mrg_pop_stv_check
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mrg_pop_stv_check_%j.out
#SBATCH --error=logs/mrg_pop_stv_check_%j.err

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m population.space_time_v.spacetime_core validate --root "$PROJECT"
