#!/bin/bash
set -eu
set -o pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PILOT_NAME="${PILOT_NAME:-discrimination_v4_regularized_convergence_pilot}"

export PROJECT
export MC_TOTAL="${MC_TOTAL:-2}"
export MC_SHARDS="${MC_SHARDS:-1}"
export MAX_PARALLEL="${MAX_PARALLEL:-48}"
export READOUT_SCOPES_CSV="${READOUT_SCOPES_CSV:-per_block,per_frequency,per_panel}"
export STATE_SPECS_CSV="${STATE_SPECS_CSV:-per_frequency:best_single,per_frequency:mixture,global:fixed_single,per_frequency:fixed_single}"
export MODEL_SPECS_CSV="${MODEL_SPECS_CSV:-rate:none,timing:none,hybrid:none,hybrid:delta_cv}"
export MIXTURE_SEMANTICS_CSV="${MIXTURE_SEMANTICS_CSV:-latent_trial}"
export N_STARTS_BEST_SINGLE="${N_STARTS_BEST_SINGLE:-18}"
export N_STARTS_MIXTURE="${N_STARTS_MIXTURE:-3}"
export N_STARTS_FIXED_SINGLE="${N_STARTS_FIXED_SINGLE:-1}"
export N_STARTS_FIXED_SELECTION="${N_STARTS_FIXED_SELECTION:-18}"
export PARTIAL_ROOT="${PARTIAL_ROOT:-$PROJECT/results/population/partials/$PILOT_NAME}"
export OUT_ROOT="${OUT_ROOT:-$PROJECT/results/population/$PILOT_NAME}"

exec bash "$PROJECT/slurm/population/discrimination/run_population_discrimination_pipeline.sh"
