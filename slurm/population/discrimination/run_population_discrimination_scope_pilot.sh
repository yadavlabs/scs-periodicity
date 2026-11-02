#!/bin/bash
set -eu
set -o pipefail

# Focused pilot for the crossed mechanistic hypotheses discussed for the
# population observer. Each entry is weight_scope:readout_scope:state_mode.
# This is intentionally paired rather than a Cartesian model sweep.
export OBSERVER_SPECS_CSV="${OBSERVER_SPECS_CSV:-per_frequency:per_frequency:mixture,per_block:per_frequency:mixture,per_frequency:per_block:mixture,per_panel:per_panel:mixture}"
export MODEL_SPECS_CSV="${MODEL_SPECS_CSV:-hybrid:none}"
export MIXTURE_SEMANTICS_CSV="${MIXTURE_SEMANTICS_CSV:-latent_trial}"

# Eight independent MC splits give an initial convergence and held-out
# performance check while remaining small (4 hypotheses x 8 = 32 tasks).
export MC_TOTAL="${MC_TOTAL:-8}"
export MC_SHARDS="${MC_SHARDS:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-32}"
export N_STARTS_MIXTURE="${N_STARTS_MIXTURE:-6}"

# Preserve the manuscript-matching mixture regularization used in v4.
export LOSS_REG_L2="${LOSS_REG_L2:-0.001}"
export WEIGHT_KL="${WEIGHT_KL:-0.01}"
export WEIGHT_ENTROPY_FLOOR_K="${WEIGHT_ENTROPY_FLOOR_K:-3.0}"
export WEIGHT_ENTROPY_FLOOR_LAMBDA="${WEIGHT_ENTROPY_FLOOR_LAMBDA:-0.01}"
export WEIGHT_MAX_CAP="${WEIGHT_MAX_CAP:-0.65}"
export WEIGHT_MAX_CAP_LAMBDA="${WEIGHT_MAX_CAP_LAMBDA:-0.01}"

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export PROJECT
export PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/results/population/partials/discrimination_v5_scope_hypothesis_pilot}"
export OUT_ROOT="${OUT_ROOT:-$PROJECT/results/population/discrimination_v5_scope_hypothesis_pilot}"

bash slurm/population/discrimination/run_population_discrimination_pipeline.sh
