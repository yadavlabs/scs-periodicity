#!/bin/bash
set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
mkdir -p logs

PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/logs/partials/discrimination_v4_manuscript_regularized}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/data/discrimination/control}"
MAX_PARALLEL="${MAX_PARALLEL:-960}"
MC_TOTAL="${MC_TOTAL:-200}"
MC_SHARDS="${MC_SHARDS:-20}"
READOUT_SCOPES_CSV="${READOUT_SCOPES_CSV:-per_block,per_frequency,per_panel}"
STATE_SPECS_CSV="${STATE_SPECS_CSV:-per_frequency:best_single,per_frequency:mixture,global:fixed_single,per_frequency:fixed_single}"
# Optional paired specifications avoid an unnecessary Cartesian product.
# Format: weight_scope:readout_scope:state_mode,...
OBSERVER_SPECS_CSV="${OBSERVER_SPECS_CSV:-}"
MIXTURE_SEMANTICS_CSV="${MIXTURE_SEMANTICS_CSV:-latent_trial}"
MODEL_SPECS_CSV="${MODEL_SPECS_CSV:-rate:none,timing:none,hybrid:none,hybrid:delta_cv}"
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
DRY_RUN="${DRY_RUN:-0}"

count_csv() {
  local IFS=','
  read -r -a values <<< "$1"
  echo "${#values[@]}"
}

N_READOUT=$(count_csv "$READOUT_SCOPES_CSV")
N_STATE_SPEC=$(count_csv "$STATE_SPECS_CSV")
N_SEMANTICS=$(count_csv "$MIXTURE_SEMANTICS_CSV")
N_MODEL=$(count_csv "$MODEL_SPECS_CSV")
N_OBSERVER=0
if [[ -n "$OBSERVER_SPECS_CSV" ]]; then
  N_OBSERVER=$(count_csv "$OBSERVER_SPECS_CSV")
fi
if (( MC_TOTAL <= 0 || MC_SHARDS <= 0 || MC_TOTAL % MC_SHARDS != 0 )); then
  echo "MC_TOTAL must be positive and divisible by MC_SHARDS" >&2
  exit 2
fi
if (( N_OBSERVER > 0 )); then
  TOTAL=$((N_OBSERVER * N_SEMANTICS * N_MODEL * MC_SHARDS))
  EXPECTED_CONFIGURATIONS=$((N_OBSERVER * N_SEMANTICS * N_MODEL))
else
  TOTAL=$((N_READOUT * N_STATE_SPEC * N_SEMANTICS * N_MODEL * MC_SHARDS))
  EXPECTED_CONFIGURATIONS=$((N_READOUT * N_STATE_SPEC * N_SEMANTICS * N_MODEL))
fi
PARALLEL=$MAX_PARALLEL
if (( PARALLEL > TOTAL )); then PARALLEL=$TOTAL; fi
ARRAY="0-$((TOTAL - 1))%${PARALLEL}"

echo "Population discrimination plan"
echo "  project: $PROJECT"
if (( N_OBSERVER > 0 )); then
  echo "  paired observer specifications: $OBSERVER_SPECS_CSV"
else
  echo "  state specifications: $STATE_SPECS_CSV"
  echo "  readout: $READOUT_SCOPES_CSV"
fi
echo "  models: $MODEL_SPECS_CSV"
echo "  mixture semantics: $MIXTURE_SEMANTICS_CSV"
echo "  mixture regularization: KL=$WEIGHT_KL effective-k>=$WEIGHT_ENTROPY_FLOOR_K (lambda=$WEIGHT_ENTROPY_FLOOR_LAMBDA) max-weight<=$WEIGHT_MAX_CAP (lambda=$WEIGHT_MAX_CAP_LAMBDA)"
echo "  MC: $MC_TOTAL total / $MC_SHARDS shards"
echo "  array: $ARRAY ($TOTAL tasks)"
echo "  partial root: $PARTIAL_ROOT"
echo "  output root: $OUT_ROOT"

export PROJECT PARTIAL_ROOT OUT_ROOT MAX_PARALLEL MC_TOTAL MC_SHARDS
export READOUT_SCOPES_CSV STATE_SPECS_CSV OBSERVER_SPECS_CSV
export MIXTURE_SEMANTICS_CSV MODEL_SPECS_CSV
export EXPECTED_CONFIGURATIONS
export TRAIN_FRACTION SEED_BASE LOSS_REG_L2 WEIGHT_KL
export WEIGHT_ENTROPY_FLOOR_K WEIGHT_ENTROPY_FLOOR_LAMBDA WEIGHT_MAX_CAP WEIGHT_MAX_CAP_LAMBDA
export RATE_GATE_REG RATE_GATE_PRIOR_FLOOR RATE_GATE_PRIOR_CENTER RATE_GATE_PRIOR_SLOPE
export N_STARTS_MIXTURE N_STARTS_BEST_SINGLE N_STARTS_FIXED_SINGLE N_STARTS_FIXED_SELECTION

module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$DRY_RUN" == "1" ]]; then
  if [[ -z "${PIPELINE_DEPENDENCY:-}" ]]; then
    python -u src/discrimination/discrimination_core.py --preflight
  else
    echo "[dry-run] preflight will follow job $PIPELINE_DEPENDENCY"
  fi
  echo "[dry-run] no jobs submitted"
  exit 0
fi
UPSTREAM_DEPENDENCY=()
if [[ -n "${PIPELINE_DEPENDENCY:-}" ]]; then
  PREFLIGHT_JOB=$(sbatch --parsable --dependency="afterok:$PIPELINE_DEPENDENCY" \
    --export=ALL hpc_tools/slurm/discrimination/00_preflight.sbatch)
  PREFLIGHT_JOB="${PREFLIGHT_JOB%%;*}"
  UPSTREAM_DEPENDENCY=(--dependency="afterok:$PREFLIGHT_JOB")
  echo "Submitted population discrimination preflight: $PREFLIGHT_JOB"
else
  python -u src/discrimination/discrimination_core.py --preflight
fi

ARRAY_JOB=$(sbatch --parsable \
  "${UPSTREAM_DEPENDENCY[@]}" \
  --array="$ARRAY" \
  --export=ALL \
  hpc_tools/slurm/discrimination/01_population_discrimination_array.sbatch)
ARRAY_JOB="${ARRAY_JOB%%;*}"
echo "Submitted population discrimination array: $ARRAY_JOB"

MERGE_JOB=$(sbatch --parsable \
  --dependency="afterok:$ARRAY_JOB" \
  --export=ALL \
  hpc_tools/slurm/discrimination/02_population_discrimination_merge.sbatch)
MERGE_JOB="${MERGE_JOB%%;*}"
echo "Submitted population discrimination merge: $MERGE_JOB"

VALIDATE_JOB=$(sbatch --parsable \
  --dependency="afterok:$MERGE_JOB" \
  --export=ALL \
  hpc_tools/slurm/discrimination/03_population_discrimination_validate.sbatch)
VALIDATE_JOB="${VALIDATE_JOB%%;*}"
echo "Submitted population discrimination validation: $VALIDATE_JOB"
echo "Monitor with: squeue -j $ARRAY_JOB,$MERGE_JOB,$VALIDATE_JOB"
if [[ -n "${PIPELINE_JOB_ID_FILE:-}" ]]; then
  printf '%s\n' "$VALIDATE_JOB" > "$PIPELINE_JOB_ID_FILE"
fi
