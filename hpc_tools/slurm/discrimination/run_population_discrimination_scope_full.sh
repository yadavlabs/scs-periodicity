#!/bin/bash
set -eu
set -o pipefail

# Full MC analysis for the four paired observer hypotheses retained after the
# v5 scope pilot. Format: weight_scope:readout_scope:state_mode.
export OBSERVER_SPECS_CSV="${OBSERVER_SPECS_CSV:-per_frequency:per_frequency:mixture,per_block:per_frequency:mixture,per_frequency:per_block:mixture,per_panel:per_panel:mixture}"
export MODEL_SPECS_CSV="${MODEL_SPECS_CSV:-hybrid:none}"
export MIXTURE_SEMANTICS_CSV="${MIXTURE_SEMANTICS_CSV:-latent_trial}"

# One MC split per task yields 4 x 200 = 800 independent one-CPU jobs and uses
# the available Longleaf concurrency without serializing splits inside a task.
export MC_TOTAL="${MC_TOTAL:-200}"
export MC_SHARDS="${MC_SHARDS:-200}"
export MAX_PARALLEL="${MAX_PARALLEL:-800}"
export N_STARTS_MIXTURE="${N_STARTS_MIXTURE:-6}"

# Manuscript-matching soft diversity prior, evaluated after response-equivalent
# regimes are collapsed separately at each frequency.
export LOSS_REG_L2="${LOSS_REG_L2:-0.001}"
export WEIGHT_KL="${WEIGHT_KL:-0.01}"
export WEIGHT_ENTROPY_FLOOR_K="${WEIGHT_ENTROPY_FLOOR_K:-3.0}"
export WEIGHT_ENTROPY_FLOOR_LAMBDA="${WEIGHT_ENTROPY_FLOOR_LAMBDA:-0.01}"
export WEIGHT_MAX_CAP="${WEIGHT_MAX_CAP:-0.65}"
export WEIGHT_MAX_CAP_LAMBDA="${WEIGHT_MAX_CAP_LAMBDA:-0.01}"

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PROJECT
export PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/logs/partials/discrimination_v5_scope_hypothesis_full}"
export OUT_ROOT="${OUT_ROOT:-$PROJECT/data/discrimination/scope}"

bash hpc_tools/slurm/discrimination/run_population_discrimination_pipeline.sh
