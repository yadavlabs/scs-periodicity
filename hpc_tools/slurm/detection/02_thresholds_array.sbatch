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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

FILE_LIST="${FILE_LIST:-$PROJECT/results/npz_files.txt}"
PARTIAL_DIR="${PARTIAL_DIR:-$PROJECT/data/partials/detection/thresholds}"
RESPONSE_PARTIAL_DIR="${RESPONSE_PARTIAL_DIR:-$PROJECT/data/partials/detection/detection_responses}"
FILE_IDX="${FILE_IDX:-${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID or FILE_IDX is required}}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"

mkdir -p "$PARTIAL_DIR" "$RESPONSE_PARTIAL_DIR"

python -u src/detection/detection_CLI.py run-thresholds \
  --file-list "$FILE_LIST" \
  --file-idx "$FILE_IDX" \
  --n-jobs "$N_JOBS" \
  --partial-dir "$PARTIAL_DIR" \
  --response-partial-dir "$RESPONSE_PARTIAL_DIR" \
  --chunk-size "$CHUNK_SIZE"
