#!/bin/bash
#SBATCH --job-name=pop_disc_check
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pop_disc_check_%A.out
#SBATCH --error=logs/pop_disc_check_%A.err

set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

OUT_ROOT="${OUT_ROOT:-$PROJECT/results/population/discrimination}"
MC_TOTAL="${MC_TOTAL:-200}"
EXPECTED_CONFIGURATIONS="${EXPECTED_CONFIGURATIONS:-8}"

python -u src/population/discrimination/discrimination_CLI.py validate \
  --out-root "$OUT_ROOT" \
  --expected-mc "$MC_TOTAL" \
  --expected-configurations "$EXPECTED_CONFIGURATIONS"
