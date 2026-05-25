#!/bin/bash
set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

module purge
module load "${PYTHON_MODULE:-python/3.12.4}"
source "${VENV_ACTIVATE:-$PROJECT/venv/bin/activate}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u src/discrimination/release_metadata.py \
  --target "$PROJECT/data/discrimination" \
  --n-mc "${DISCRIMINATION_MC_TOTAL:-200}" \
  --run-label "${RUN_LABEL:-release}"

echo "[finalize-data] finalized release discrimination metadata in data/discrimination"

PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/data/partials}"
if [[ "${CLEAN_PARTIALS:-1}" == "1" ]]; then
  case "$PARTIAL_ROOT" in
    "$PROJECT"/data/partials)
      rm -rf "$PARTIAL_ROOT"
      echo "[finalize-data] removed transient partials from $PARTIAL_ROOT"
      ;;
    *)
      echo "[finalize-data] leaving unexpected partial path untouched: $PARTIAL_ROOT" >&2
      ;;
  esac
fi

echo "[finalize-data] complete; retrieve generated outputs locally with hpc_tools/download_data.py"
