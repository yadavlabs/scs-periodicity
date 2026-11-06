#!/bin/bash
#SBATCH --job-name=pop_disc
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --array=0-0%1
#SBATCH --output=logs/pop_disc_%A_%a.out
#SBATCH --error=logs/pop_disc_%A_%a.err

set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [[ -n "${FOLLOWUP_MANIFEST:-}" ]]; then
  python -m discrimination.block_cue_followup --root "$PROJECT" \
    --out-root "$(dirname "$FOLLOWUP_MANIFEST")" --verify-only
fi

EVIDENCE_CSV="${EVIDENCE_CSV:-$PROJECT/data/features/population_cv0_template_evidence.csv}"
SLACK_DIR="${SLACK_DIR:-$PROJECT/data/discrimination/slack}"
PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/logs/partials/discrimination_v4_manuscript_regularized}"
FREQUENCIES="${FREQUENCIES:-20,50,100,200}"
READOUT_SCOPES_CSV="${READOUT_SCOPES_CSV:-per_block,per_frequency,per_panel}"
STATE_SPECS_CSV="${STATE_SPECS_CSV:-per_frequency:best_single,per_frequency:mixture,global:fixed_single,per_frequency:fixed_single}"
OBSERVER_SPECS_CSV="${OBSERVER_SPECS_CSV:-}"
MIXTURE_SEMANTICS_CSV="${MIXTURE_SEMANTICS_CSV:-latent_trial}"
MODEL_SPECS_CSV="${MODEL_SPECS_CSV:-rate:none,timing:none,hybrid:none,hybrid:delta_cv}"
MC_TOTAL="${MC_TOTAL:-200}"
MC_SHARDS="${MC_SHARDS:-20}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
SEED_BASE="${SEED_BASE:-0}"
LOSS_REG_L2="${LOSS_REG_L2:-0.001}"
WEIGHT_KL="${WEIGHT_KL:-0.01}"
WEIGHT_ENTROPY_FLOOR_K="${WEIGHT_ENTROPY_FLOOR_K:-3.0}"
WEIGHT_ENTROPY_FLOOR_LAMBDA="${WEIGHT_ENTROPY_FLOOR_LAMBDA:-0.01}"
WEIGHT_MAX_CAP="${WEIGHT_MAX_CAP:-0.65}"
WEIGHT_MAX_CAP_LAMBDA="${WEIGHT_MAX_CAP_LAMBDA:-0.01}"
RATE_GATE_REG="${RATE_GATE_REG:-0.002}"
RATE_GATE_PRIOR_FLOOR="${RATE_GATE_PRIOR_FLOOR:-1.0}"
RATE_GATE_PRIOR_CENTER="${RATE_GATE_PRIOR_CENTER:-0.5}"
RATE_GATE_PRIOR_SLOPE="${RATE_GATE_PRIOR_SLOPE:-8.0}"
N_STARTS_MIXTURE="${N_STARTS_MIXTURE:-6}"
N_STARTS_BEST_SINGLE="${N_STARTS_BEST_SINGLE:-18}"
N_STARTS_FIXED_SINGLE="${N_STARTS_FIXED_SINGLE:-2}"
N_STARTS_FIXED_SELECTION="${N_STARTS_FIXED_SELECTION:-18}"

IFS=',' read -r -a READOUT_SCOPES <<< "$READOUT_SCOPES_CSV"
IFS=',' read -r -a STATE_SPECS <<< "$STATE_SPECS_CSV"
IFS=',' read -r -a OBSERVER_SPECS <<< "$OBSERVER_SPECS_CSV"
IFS=',' read -r -a MIXTURE_SEMANTICS <<< "$MIXTURE_SEMANTICS_CSV"
IFS=',' read -r -a MODEL_SPECS <<< "$MODEL_SPECS_CSV"

if (( MC_TOTAL <= 0 || MC_SHARDS <= 0 || MC_TOTAL % MC_SHARDS != 0 )); then
  echo "[slurm] MC_TOTAL must be positive and divisible by MC_SHARDS" >&2
  exit 2
fi
MC_PER_SHARD=$((MC_TOTAL / MC_SHARDS))
N_READOUT=${#READOUT_SCOPES[@]}
N_STATE_SPEC=${#STATE_SPECS[@]}
N_SEMANTICS=${#MIXTURE_SEMANTICS[@]}
N_MODEL=${#MODEL_SPECS[@]}
N_OBSERVER=0
if [[ -n "$OBSERVER_SPECS_CSV" ]]; then
  N_OBSERVER=${#OBSERVER_SPECS[@]}
fi
if (( N_OBSERVER > 0 )); then
  TOTAL=$((N_OBSERVER * N_SEMANTICS * N_MODEL * MC_SHARDS))
else
  TOTAL=$((N_READOUT * N_STATE_SPEC * N_SEMANTICS * N_MODEL * MC_SHARDS))
fi
TASK=${SLURM_ARRAY_TASK_ID}
if (( TASK < 0 || TASK >= TOTAL )); then
  echo "[slurm] invalid task $TASK for $TOTAL configurations" >&2
  exit 2
fi

REM=$TASK
SHARD_IDX=$((REM % MC_SHARDS)); REM=$((REM / MC_SHARDS))
MODEL_IDX=$((REM % N_MODEL)); REM=$((REM / N_MODEL))
SEMANTICS_IDX=$((REM % N_SEMANTICS)); REM=$((REM / N_SEMANTICS))
if (( N_OBSERVER > 0 )); then
  OBSERVER_IDX=$((REM % N_OBSERVER)); REM=$((REM / N_OBSERVER))
  OBSERVER_SPEC=${OBSERVER_SPECS[$OBSERVER_IDX]}
  IFS=':' read -r WEIGHT_SCOPE READOUT_SCOPE STATE_MODE <<< "$OBSERVER_SPEC"
  if [[ -z "$WEIGHT_SCOPE" || -z "$READOUT_SCOPE" || -z "$STATE_MODE" ]]; then
    echo "[slurm] invalid observer specification: $OBSERVER_SPEC" >&2
    exit 2
  fi
else
  STATE_SPEC_IDX=$((REM % N_STATE_SPEC)); REM=$((REM / N_STATE_SPEC))
  READOUT_IDX=$((REM % N_READOUT)); REM=$((REM / N_READOUT))
  READOUT_SCOPE=${READOUT_SCOPES[$READOUT_IDX]}
  STATE_SPEC=${STATE_SPECS[$STATE_SPEC_IDX]}
  WEIGHT_SCOPE=${STATE_SPEC%%:*}
  STATE_MODE=${STATE_SPEC##*:}
fi
SEMANTICS=${MIXTURE_SEMANTICS[$SEMANTICS_IDX]}
MODEL_SPEC=${MODEL_SPECS[$MODEL_IDX]}
MODEL_FAMILY=${MODEL_SPEC%%:*}
RATE_GATE_MODE=${MODEL_SPEC##*:}
MC_START=$((SHARD_IDX * MC_PER_SHARD))
if [[ "$STATE_MODE" == "best_single" ]]; then
  N_STARTS=$N_STARTS_BEST_SINGLE
elif [[ "$STATE_MODE" == "fixed_single" ]]; then
  N_STARTS=$N_STARTS_FIXED_SINGLE
else
  N_STARTS=$N_STARTS_MIXTURE
fi

OUT_DIR="$PARTIAL_ROOT/weights_${WEIGHT_SCOPE}/readout_${READOUT_SCOPE}/state_${STATE_MODE}/model_${MODEL_FAMILY}_${RATE_GATE_MODE}/semantics_${SEMANTICS}/mc_${MC_START}_$((MC_START + MC_PER_SHARD - 1))"
mkdir -p "$OUT_DIR"

echo "[slurm] task=$TASK/$((TOTAL-1)) weights=$WEIGHT_SCOPE readout=$READOUT_SCOPE state=$STATE_MODE model=$MODEL_FAMILY gate=$RATE_GATE_MODE semantics=$SEMANTICS mc_start=$MC_START mc_count=$MC_PER_SHARD"

python -u src/discrimination/discrimination_core.py \
  --evidence-csv "$EVIDENCE_CSV" \
  --slack-dir "$SLACK_DIR" \
  --out-dir "$OUT_DIR" \
  --frequencies "$FREQUENCIES" \
  --weight-scope "$WEIGHT_SCOPE" \
  --readout-scope "$READOUT_SCOPE" \
  --state-mode "$STATE_MODE" \
  --model-family "$MODEL_FAMILY" \
  --rate-gate-mode "$RATE_GATE_MODE" \
  --mixture-semantics "$SEMANTICS" \
  --n-mc "$MC_TOTAL" \
  --mc-start "$MC_START" \
  --mc-count "$MC_PER_SHARD" \
  --train-fraction "$TRAIN_FRACTION" \
  --seed "$SEED_BASE" \
  --loss-reg-l2 "$LOSS_REG_L2" \
  --weight-kl "$WEIGHT_KL" \
  --weight-entropy-floor-k "$WEIGHT_ENTROPY_FLOOR_K" \
  --weight-entropy-floor-lambda "$WEIGHT_ENTROPY_FLOOR_LAMBDA" \
  --weight-max-cap "$WEIGHT_MAX_CAP" \
  --weight-max-cap-lambda "$WEIGHT_MAX_CAP_LAMBDA" \
  --rate-gate-reg "$RATE_GATE_REG" \
  --rate-gate-prior-floor "$RATE_GATE_PRIOR_FLOOR" \
  --rate-gate-prior-center "$RATE_GATE_PRIOR_CENTER" \
  --rate-gate-prior-slope "$RATE_GATE_PRIOR_SLOPE" \
  --n-starts "$N_STARTS" \
  --fixed-selection-n-starts "$N_STARTS_FIXED_SELECTION"

echo "[slurm] completed $OUT_DIR"
