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
SIM_CSV="${SIM_CSV:-$PROJECT/data/detection/generated/cv0_based_responses.csv}"
SLACK_DIR="${SLACK_DIR:-$PROJECT/data/discrimination/slack}"
PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/data/partials/discrimination/$RUN_KIND}"
MC_TOTAL="${MC_TOTAL:-200}"
MC_SHARDS="${MC_SHARDS:-10}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
SEED="${SEED:-0}"
TASK="${TASK_ID:-${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID or TASK_ID is required}}"

if (( MC_SHARDS <= 0 )); then
  echo "[disc-release] MC_SHARDS must be > 0" >&2
  exit 2
fi
if (( MC_TOTAL <= 0 || MC_TOTAL % MC_SHARDS != 0 )); then
  echo "[disc-release] MC_TOTAL=$MC_TOTAL must be positive and divisible by MC_SHARDS=$MC_SHARDS" >&2
  exit 2
fi

case "$RUN_KIND" in
  per_block)
    SCOPE_KEYS=("block=std0" "block=std1")
    MODEL_BANKS=("rate_combined" "timing_combined" "hybrid_combined")
    ;;
  per_point)
    SCOPE_KEYS=("freq=20|block=std0" "freq=20|block=std1" "freq=50|block=std0" "freq=50|block=std1" "freq=100|block=std0" "freq=100|block=std1" "freq=200|block=std0" "freq=200|block=std1")
    MODEL_BANKS=("hybrid_combined")
    ;;
  *)
    echo "[disc-release] invalid RUN_KIND=$RUN_KIND (expected per_block or per_point)" >&2
    exit 2
    ;;
esac

N_SCOPE=${#SCOPE_KEYS[@]}
N_MODEL=${#MODEL_BANKS[@]}
TOTAL=$(( N_SCOPE * N_MODEL * MC_SHARDS ))
if (( TASK < 0 || TASK >= TOTAL )); then
  echo "[disc-release] invalid TASK=$TASK for TOTAL=$TOTAL" >&2
  exit 2
fi

SHARD_IDX=$(( TASK % MC_SHARDS ))
REM=$(( TASK / MC_SHARDS ))
MODEL_IDX=$(( REM % N_MODEL ))
SCOPE_IDX=$(( REM / N_MODEL ))
MC_PER_SHARD=$(( MC_TOTAL / MC_SHARDS ))
MC_START=$(( SHARD_IDX * MC_PER_SHARD ))
MC_STOP=$(( MC_START + MC_PER_SHARD - 1 ))

SCOPE_KEY="${SCOPE_KEYS[$SCOPE_IDX]}"
MODEL_BANK="${MODEL_BANKS[$MODEL_IDX]}"
SCOPE_SAFE=${SCOPE_KEY//|/_}
SCOPE_SAFE=${SCOPE_SAFE//=/_}
OUT_DIR="$PARTIAL_ROOT/scopekey_${SCOPE_SAFE}/model_${MODEL_BANK}/mc_${MC_START}_${MC_STOP}"
mkdir -p "$OUT_DIR"

echo "[disc-release] task=$TASK/$((TOTAL - 1)) run_kind=$RUN_KIND scope_key=$SCOPE_KEY model=$MODEL_BANK mc=${MC_START}-${MC_STOP} out=$OUT_DIR"

python -u src/discrimination/discrimination_CLI.py fit \
  --run_kind "$RUN_KIND" \
  --sim_csv "$SIM_CSV" \
  --slack_dir "$SLACK_DIR" \
  --out_dir "$OUT_DIR" \
  --scope_key "$SCOPE_KEY" \
  --model_bank "$MODEL_BANK" \
  --n_mc "$MC_TOTAL" \
  --mc_start "$MC_START" \
  --mc_count "$MC_PER_SHARD" \
  --train_frac "$TRAIN_FRAC" \
  --seed "$SEED"
