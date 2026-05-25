#!/bin/bash
set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

if command -v module >/dev/null 2>&1; then
  module purge
  module load "${PYTHON_MODULE:-python/3.12.4}"
fi

if [[ -n "${VENV_ACTIVATE:-}" && -f "${VENV_ACTIVATE}" ]]; then
  source "${VENV_ACTIVATE}"
elif [[ -f "$PROJECT/venv/bin/activate" ]]; then
  source "$PROJECT/venv/bin/activate"
elif [[ -f "$PROJECT/.venv/bin/activate" ]]; then
  source "$PROJECT/.venv/bin/activate"
fi

if [[ -d "$PROJECT/venv/lib" ]]; then
  export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_KIND="${RUN_KIND:?RUN_KIND is required: per_block or per_point}"
PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/data/partials/discrimination/$RUN_KIND}"
case "$RUN_KIND" in
  per_block)
    DEFAULT_OUT_ROOT="$PROJECT/data/discrimination/per-block"
    ;;
  per_point)
    DEFAULT_OUT_ROOT="$PROJECT/data/discrimination/per-point"
    ;;
  *)
    echo "[disc-release-merge] invalid RUN_KIND=$RUN_KIND (expected per_block or per_point)" >&2
    exit 2
    ;;
esac
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT_ROOT}"
mkdir -p "$OUT_ROOT"

python -u src/discrimination/discrimination_CLI.py merge \
  --partial-root "$PARTIAL_ROOT" \
  --out-root "$OUT_ROOT" \
  --release-kind "$RUN_KIND"

echo "[disc-release-merge] wrote merged outputs to $OUT_ROOT"

if [[ "${CLEAN_PARTIALS:-1}" == "1" ]]; then
  case "$PARTIAL_ROOT" in
    "$PROJECT"/data/partials/discrimination/*)
      rm -rf "$PARTIAL_ROOT"
      echo "[disc-release-merge] removed partials from $PARTIAL_ROOT"
      ;;
    *)
      echo "[disc-release-merge] leaving unexpected partial path untouched: $PARTIAL_ROOT" >&2
      ;;
  esac
fi
