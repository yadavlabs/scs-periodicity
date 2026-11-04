#!/bin/bash
#SBATCH --job-name=pop_accepted_fit
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pop_accepted_fit_%j.out
#SBATCH --error=logs/pop_accepted_fit_%j.err

set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
INPUTS="$PROJECT/data/detection/behavior_fit_inputs"
python -u hpc_tools/verify_behavior_fit_inputs.py
python -u -m population.detection.detection_CLI fit-behavior --root "$PROJECT" \
  --index-regimes-csv "$INPUTS/index_regime_amplitudes_pre_imax.csv" \
  --recruitment-outcomes-csv "$INPUTS/population_recruitment_outcomes_pre_imax.csv"
