#!/bin/bash
#SBATCH --job-name=pop_disc_preflight
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pop_disc_preflight_%j.out
#SBATCH --error=logs/pop_disc_preflight_%j.err

set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
python -u src/population/discrimination/discrimination_core.py --preflight
