#!/bin/bash
#SBATCH --job-name=pop_locked
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/pop_locked_%A_%a.out
#SBATCH --error=logs/pop_locked_%A_%a.err
set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# One matched MC split; all four controls plus hybrid refit, six starts each.
python -u -m discrimination.locked_mixture_followup fit \
  --root "$PROJECT" --bundle "$LOCKED_BUNDLE" --mc "$SLURM_ARRAY_TASK_ID" \
  --out "$PARTIAL_ROOT/mc_$SLURM_ARRAY_TASK_ID"
