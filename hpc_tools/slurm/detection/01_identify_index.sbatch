#!/bin/bash
#SBATCH --job-name=mrg_pop_index
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mrg_pop_index_%j.out
#SBATCH --error=logs/mrg_pop_index_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"

# --force deliberately recomputes the local reference result on Longleaf.
ARGS=(--root "$PROJECT" --n-jobs "$SLURM_CPUS_PER_TASK" --force)
if [[ -n "${POPULATION_MANIFEST:-}" ]]; then
  ARGS+=(--population-manifest "$POPULATION_MANIFEST")
fi
python -m detection.detection_CLI identify-index "${ARGS[@]}"

# Reusing old regime/response partials is valid only while the index is stable.
python -m detection.detection_CLI verify-index \
  --root "$PROJECT" --expected-axon-id "${EXPECTED_INDEX_AXON_ID:-S01_140um_ph1}"
