#!/bin/bash
#SBATCH --job-name=pop_time_patterns
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pop_time_patterns_%j.out
#SBATCH --error=logs/pop_time_patterns_%j.err

set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
python -u hpc_tools/run_notebook_code.py notebooks/01_TimePatterns_Generator.ipynb --through-cell 4
