#!/bin/bash
#SBATCH --job-name=pop_locked_prepare
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pop_locked_prepare_%j.out
#SBATCH --error=logs/pop_locked_prepare_%j.err

set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
python -u -m population.discrimination.locked_mixture_followup prepare \
  --root "$PROJECT" --bundle "$LOCKED_BUNDLE"
