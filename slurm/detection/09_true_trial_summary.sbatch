#!/bin/bash
#SBATCH --job-name=pop_true_summary
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pop_true_summary_%j.out
#SBATCH --error=logs/pop_true_summary_%j.err

set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
python -u -m population.detection.true_trial_summary --root "$PROJECT"
