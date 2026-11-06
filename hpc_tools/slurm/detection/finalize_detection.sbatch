#!/bin/bash
#SBATCH --job-name=pop_det_summary
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pop_det_summary_%j.out
#SBATCH --error=logs/pop_det_summary_%j.err

set -euo pipefail
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u hpc_tools/run_notebook_code.py notebooks/03_DetectionThresholds.ipynb --through-cell 15
