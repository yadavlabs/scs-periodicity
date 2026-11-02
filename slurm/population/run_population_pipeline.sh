#!/usr/bin/env bash
set -euo pipefail

# Complete Longleaf workflow: generate and validate the phase-aware SpaceTimeV
# cohort, then submit optimized population detection.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MAX_PARALLEL_SPACETIME="${MAX_PARALLEL_SPACETIME:-24}"
EXPECTED_INDEX_AXON_ID="${EXPECTED_INDEX_AXON_ID:-S01_140um_ph1}"
POPULATION_MANIFEST="${POPULATION_MANIFEST:-$PROJECT/data/space-time-V/population/axon_basis_manifest.csv}"

cd "$PROJECT"
mkdir -p logs results/population/detection/partials
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT EXPECTED_INDEX_AXON_ID POPULATION_MANIFEST

N_AXON_JOBS=$(python -m population.space_time_v.spacetime_core job-count --root "$PROJECT" | tail -n 1)
N_PHASE_GROUPS=$((N_AXON_JOBS / 6))
N_CUTLINES=$((N_PHASE_GROUPS / 2))
RECRUITMENT_TASK_COUNT=$((N_PHASE_GROUPS * 55))
export RECRUITMENT_TASK_COUNT

echo "Submitting Longleaf 2.0 population workflow"
echo "  cutlines=$N_CUTLINES phase-groups=$N_PHASE_GROUPS axons=$N_AXON_JOBS SpaceTimeV files=$((N_AXON_JOBS * 55))"
echo "  SpaceTimeV concurrency=$MAX_PARALLEL_SPACETIME"
echo "  expected index=$EXPECTED_INDEX_AXON_ID"

JID_BASES=$(sbatch --parsable --export=ALL slurm/population/space_time_v/00_prepare_bases.sbatch)
JID_BASES="${JID_BASES%%;*}"
JID_STV=$(sbatch --parsable --dependency=afterok:"$JID_BASES" \
  --array="0-$((N_AXON_JOBS - 1))%${MAX_PARALLEL_SPACETIME}" --export=ALL \
  slurm/population/space_time_v/01_generate_spacetime_array.sbatch)
JID_STV="${JID_STV%%;*}"
JID_VALIDATE=$(sbatch --parsable --dependency=afterok:"$JID_STV" --export=ALL \
  slurm/population/space_time_v/02_validate_spacetime.sbatch)
JID_VALIDATE="${JID_VALIDATE%%;*}"
echo "Submitted upstream: bases=$JID_BASES spacetime=$JID_STV validate=$JID_VALIDATE"
PIPELINE_DEPENDENCY="$JID_VALIDATE" bash slurm/population/detection/run_detection_pipeline.sh
