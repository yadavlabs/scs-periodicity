#!/bin/bash
#SBATCH --job-name=mrg_pop_mod
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mrg_pop_mod_%j.out
#SBATCH --error=logs/mrg_pop_mod_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

# Validate the repository's population-aware MRG sources and compile only when
# the Linux mechanism library is absent. The population core discovers the
# normal architecture-specific output directory.
python hpc_tools/install_mrg_model.py --compile
