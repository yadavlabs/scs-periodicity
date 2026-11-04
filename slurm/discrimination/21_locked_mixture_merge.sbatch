#!/bin/bash
#SBATCH --job-name=pop_lock_merge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pop_lock_merge_%j.out
#SBATCH --error=logs/pop_lock_merge_%j.err
set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m discrimination.locked_mixture_followup merge \
  --root "$PROJECT" --bundle "$LOCKED_BUNDLE" --partials "$PARTIAL_ROOT" \
  --out "$OUT_ROOT" --expected-mc 200
