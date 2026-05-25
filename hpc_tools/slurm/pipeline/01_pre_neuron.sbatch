#!/bin/bash
set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

module purge
module load "${PYTHON_MODULE:-python/3.12.4}"
source "${VENV_ACTIVATE:-$PROJECT/venv/bin/activate}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

ARGS=(worker-pre-neuron --n-jobs "${PRE_N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}")
case "${FORCE_PRE_NEURON:-1}" in
  1|true|TRUE|yes|YES|y|Y)
    ARGS+=(--force)
    ;;
esac

python -u hpc_tools/run_pipeline.py "${ARGS[@]}"
