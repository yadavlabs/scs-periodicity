#!/usr/bin/env bash
#SBATCH --job-name=scs_pop_smoke
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/scs_pop_smoke_%j.out
#SBATCH --error=logs/scs_pop_smoke_%j.err

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

python hpc_tools/preflight_population.py --root "$PROJECT"
python hpc_tools/install_mrg_model.py --compile
python hpc_tools/pilot_population.py --root "$PROJECT" --axon-id S01_140um_ph1 --mode impulse
python hpc_tools/pilot_population.py --root "$PROJECT" --axon-id S01_140um_ph1 --mode train
python hpc_tools/pilot_population.py --root "$PROJECT" --axon-id D05_57um_ph0 --mode impulse
