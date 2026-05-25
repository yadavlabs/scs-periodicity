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

PARTIAL_DIR="${PARTIAL_DIR:-$PROJECT/data/partials/detection/thresholds}"
RESPONSE_PARTIAL_DIR="${RESPONSE_PARTIAL_DIR:-$PROJECT/data/partials/detection/detection_responses}"
OUT_DET="${OUT_DET:-$PROJECT/data/detection/generated/detection_thresholds.csv}"
OUT_DET_RESP="${OUT_DET_RESP:-$PROJECT/data/detection/generated/detection_responses.csv}"
OUT_CV0="${OUT_CV0:-$PROJECT/data/detection/generated/cv0_threshold_max.csv}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.0}"

mkdir -p "$(dirname "$OUT_DET")" "$(dirname "$OUT_DET_RESP")" "$(dirname "$OUT_CV0")"

python -u src/detection/detection_CLI.py merge-thresholds \
  --partial-dir "$PARTIAL_DIR" \
  --out "$OUT_DET"

python -u src/detection/detection_CLI.py merge-detection-responses \
  --partial-dir "$RESPONSE_PARTIAL_DIR" \
  --out "$OUT_DET_RESP"

python -u src/detection/detection_CLI.py compute-cv0 \
  --det-csv "$OUT_DET" \
  --out "$OUT_CV0" \
  --safety-margin "$SAFETY_MARGIN"

if [[ "${CLEAN_PARTIALS:-1}" == "1" ]]; then
  for d in "$PARTIAL_DIR" "$RESPONSE_PARTIAL_DIR"; do
    case "$d" in
      "$PROJECT"/data/partials/detection/*)
        rm -rf "$d"
        echo "[mrg-merge-cv0] removed partials from $d"
        ;;
      *)
        echo "[mrg-merge-cv0] leaving unexpected partial path untouched: $d" >&2
        ;;
    esac
  done
fi
