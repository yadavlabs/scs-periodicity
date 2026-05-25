#!/bin/bash
set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

DATA_DIR="${SPACE_TIME_DIR:-$PROJECT/data/space-time-V/generated}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT/results}"
FILE_LIST="${FILE_LIST:-$RESULTS_DIR/npz_files.txt}"

mkdir -p "$(dirname "$FILE_LIST")"
find "$DATA_DIR" -maxdepth 1 -type f -name 'M*_SpaceTimeV.npz' -print | sort > "$FILE_LIST"

if [[ ! -s "$FILE_LIST" ]]; then
  echo "[filelist] no SpaceTimeV files found in $DATA_DIR" >&2
  exit 2
fi

echo "[filelist] wrote $(wc -l < "$FILE_LIST") files -> $FILE_LIST"
