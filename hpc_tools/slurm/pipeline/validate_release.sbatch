#!/bin/bash
#SBATCH --job-name=pop_release_check
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pop_release_check_%j.out
#SBATCH --error=logs/pop_release_check_%j.err

set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
python -u hpc_tools/validate_release_pipeline.py
