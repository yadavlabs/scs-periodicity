#!/usr/bin/env bash
set -euo pipefail

# Submit the release stages with afterok dependencies, following the v1.0.2
# release coordinator pattern. Notebook 06 runs locally after result download.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MAX_PARALLEL_SPACETIME="${MAX_PARALLEL_SPACETIME:-24}"
EXPECTED_INDEX_AXON_ID="${EXPECTED_INDEX_AXON_ID:-S01_140um_ph1}"
POPULATION_MANIFEST="${POPULATION_MANIFEST:-$PROJECT/data/space-time-V/axon_basis_manifest.csv}"

cd "$PROJECT"
mkdir -p logs results/detection/partials
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

JID_PATTERNS=$(sbatch --parsable --export=ALL slurm/pipeline/00_generate_time_patterns.sbatch)
JID_PATTERNS="${JID_PATTERNS%%;*}"
JID_BASES=$(sbatch --parsable --dependency=afterok:"$JID_PATTERNS" \
  --export=ALL slurm/space_time_v/00_prepare_bases.sbatch)
JID_BASES="${JID_BASES%%;*}"
JID_STV=$(sbatch --parsable --dependency=afterok:"$JID_BASES" \
  --array="0-$((N_AXON_JOBS - 1))%${MAX_PARALLEL_SPACETIME}" --export=ALL \
  slurm/space_time_v/01_generate_spacetime_array.sbatch)
JID_STV="${JID_STV%%;*}"
JID_VALIDATE=$(sbatch --parsable --dependency=afterok:"$JID_STV" --export=ALL \
  slurm/space_time_v/02_validate_spacetime.sbatch)
JID_VALIDATE="${JID_VALIDATE%%;*}"
echo "Submitted upstream: patterns=$JID_PATTERNS bases=$JID_BASES spacetime=$JID_STV validate=$JID_VALIDATE"
PIPELINE_DEPENDENCY="$JID_VALIDATE" bash slurm/detection/run_detection_pipeline.sh
source "$PROJECT/results/pipeline/detection_jobs.env"

# Detection notebook 03 produces reference-phase and recruitment summaries and
# the behavioral fit. The trial summary retains the independent I_max audit.
JID_DETECTION_NOTEBOOK=$(sbatch --parsable \
  --dependency="afterok:$JID_MERGE_TRUE:$JID_MERGE_CV0" --export=ALL \
  slurm/detection/08_finalize_detection.sbatch)
JID_DETECTION_NOTEBOOK="${JID_DETECTION_NOTEBOOK%%;*}"
JID_ACCEPTED_FIT=$(sbatch --parsable --dependency="afterok:$JID_DETECTION_NOTEBOOK" \
  --export=ALL slurm/detection/10_fit_accepted_behavior.sbatch)
JID_ACCEPTED_FIT="${JID_ACCEPTED_FIT%%;*}"
JID_TRUE_SUMMARY=$(sbatch --parsable --dependency="afterok:$JID_MERGE_TRUE" \
  --export=ALL slurm/detection/09_true_trial_summary.sbatch)
JID_TRUE_SUMMARY="${JID_TRUE_SUMMARY%%;*}"
JID_FEATURES=$(sbatch --parsable --dependency="afterok:$JID_MERGE_CV0" \
  --export=ALL slurm/feature/01_build_features.sbatch)
JID_FEATURES="${JID_FEATURES%%;*}"

PIPELINE_DEPENDENCY="$JID_FEATURES" \
PIPELINE_JOB_ID_FILE="$PROJECT/results/pipeline/discrimination_control.id" \
MAX_PARALLEL=480 \
  bash slurm/discrimination/run_population_discrimination_pipeline.sh
JID_CONTROL=$(cat "$PROJECT/results/pipeline/discrimination_control.id")
PIPELINE_DEPENDENCY="$JID_FEATURES" \
PIPELINE_JOB_ID_FILE="$PROJECT/results/pipeline/discrimination_scope.id" \
MAX_PARALLEL=500 \
  bash slurm/discrimination/run_population_discrimination_scope_full.sh
JID_SCOPE=$(cat "$PROJECT/results/pipeline/discrimination_scope.id")
PIPELINE_DEPENDENCY="$JID_SCOPE" \
PIPELINE_JOB_ID_FILE="$PROJECT/results/pipeline/discrimination_locked.id" \
  bash slurm/discrimination/run_population_locked_mixture_followup.sh
JID_LOCKED=$(cat "$PROJECT/results/pipeline/discrimination_locked.id")

JID_RELEASE=$(sbatch --parsable \
  --dependency="afterok:$JID_ACCEPTED_FIT:$JID_TRUE_SUMMARY:$JID_CONTROL:$JID_SCOPE:$JID_LOCKED" \
  --export=ALL slurm/pipeline/99_validate_release.sbatch)
JID_RELEASE="${JID_RELEASE%%;*}"
printf 'release_validation=%s\n' "$JID_RELEASE" > "$PROJECT/results/pipeline/final_job.txt"
echo "Submitted downstream: detection notebook=$JID_DETECTION_NOTEBOOK accepted fit=$JID_ACCEPTED_FIT true summary=$JID_TRUE_SUMMARY features=$JID_FEATURES"
echo "Submitted observers: control=$JID_CONTROL scope=$JID_SCOPE locked=$JID_LOCKED"
echo "Final release validation job: $JID_RELEASE"
