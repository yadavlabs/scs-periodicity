#!/bin/bash
set -euo pipefail
export PROJECT="${PROJECT:-$PWD}"
cd "$PROJECT"
# Keep this experiment isolated from manuscript/control outputs.
export PARTIAL_ROOT="$PROJECT/results/partials/discrimination_block_cue_followup"
export OUT_ROOT="$PROJECT/results/discrimination_block_cue_followup"
if [[ -e "$OUT_ROOT/submission.lock" ]] || [[ -d "$PARTIAL_ROOT" && -n "$(find "$PARTIAL_ROOT" -mindepth 1 -print -quit)" ]]; then
    echo "Existing follow-up submission/partials found; inspect jobs and results rather than overwrite or resubmit." >&2
    exit 2
fi
export OBSERVER_SPECS_CSV="per_frequency:per_block:mixture"
export MODEL_SPECS_CSV="rate_timing:none,timing_rate:none"
export MIXTURE_SEMANTICS_CSV="latent_trial"
export MC_TOTAL=200 MC_SHARDS=200
export MAX_PARALLEL="${MAX_PARALLEL:-100}"
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL < 1 || MAX_PARALLEL > 400 )); then
    echo "MAX_PARALLEL must be 1..400" >&2
    exit 2
fi
export TRAIN_FRACTION=0.8 SEED_BASE=0 N_STARTS_MIXTURE=6
export LOSS_REG_L2=0.001 WEIGHT_KL=0.01
export WEIGHT_ENTROPY_FLOOR_K=3.0 WEIGHT_ENTROPY_FLOOR_LAMBDA=0.01
export WEIGHT_MAX_CAP=0.65 WEIGHT_MAX_CAP_LAMBDA=0.01
export RATE_GATE_REG=0.002 RATE_GATE_PRIOR_FLOOR=1.0
export RATE_GATE_PRIOR_CENTER=0.5 RATE_GATE_PRIOR_SLOPE=8.0
export FREQUENCIES="20,50,100,200"
export EVIDENCE_CSV="$PROJECT/data/features/population_cv0_template_evidence.csv"
export SLACK_DIR="$PROJECT/data/discrimination/slack"
module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
# Ensure the free-mixture comparison uses the same inputs/core as the audited
# locked-mixture experiment, rather than merely fingerprinting whatever is here.
python -m population.discrimination.locked_mixture_followup verify --root "$PROJECT" \
    --bundle "$PROJECT/results/discrimination_locked_mixture_inputs"
python -m population.discrimination.block_cue_followup --root "$PROJECT" --out-root "$OUT_ROOT"
export FOLLOWUP_MANIFEST="$OUT_ROOT/followup_inputs.json"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    bash slurm/discrimination/run_population_discrimination_pipeline.sh
else
    # Persistent guard also protects against interrupted/partially successful submission.
    mkdir "$OUT_ROOT/submission.lock"
    bash slurm/discrimination/run_population_discrimination_pipeline.sh 2>&1 | tee "$OUT_ROOT/submission.log"
fi
