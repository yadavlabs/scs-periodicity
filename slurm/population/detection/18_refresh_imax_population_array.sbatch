#!/bin/bash
#SBATCH --job-name=imax_pop_refresh
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/imax_pop_refresh_%A_%a.out
#SBATCH --error=logs/imax_pop_refresh_%A_%a.err
set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m population.detection.refresh_imax_population run-axon \
  --root "$PROJECT" --axon-index "$SLURM_ARRAY_TASK_ID" --n-jobs "$SLURM_CPUS_PER_TASK"
