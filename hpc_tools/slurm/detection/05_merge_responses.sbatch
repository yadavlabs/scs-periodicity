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

CV0_PARTIAL_DIR="${CV0_PARTIAL_DIR:-${PARTIAL_DIR:-$PROJECT/data/partials/detection/responses}}"
OUT_CV0_RESP="${OUT_CV0_RESP:-${OUT_RESP:-$PROJECT/data/detection/generated/cv0_based_responses.csv}}"

mkdir -p "$(dirname "$OUT_CV0_RESP")"

python -u src/detection/detection_CLI.py merge-responses \
  --partial-dir "$CV0_PARTIAL_DIR" \
  --out "$OUT_CV0_RESP"

if [[ "${CLEAN_PARTIALS:-1}" == "1" ]]; then
  case "$CV0_PARTIAL_DIR" in
    "$PROJECT"/data/partials/detection/*)
      rm -rf "$CV0_PARTIAL_DIR"
      echo "[mrg-merge-cv0-resp] removed partials from $CV0_PARTIAL_DIR"
      ;;
    *)
      echo "[mrg-merge-cv0-resp] leaving unexpected partial path untouched: $CV0_PARTIAL_DIR" >&2
      ;;
  esac
fi
