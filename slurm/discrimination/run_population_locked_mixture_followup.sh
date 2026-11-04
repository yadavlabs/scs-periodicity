#!/bin/bash
set -euo pipefail
export PROJECT="${PROJECT:-$PWD}"
cd "$PROJECT"
export LOCKED_BUNDLE="$PROJECT/results/discrimination_locked_mixture_inputs"
export PARTIAL_ROOT="$PROJECT/results/partials/discrimination_locked_mixture_followup"
export OUT_ROOT="$PROJECT/results/discrimination_locked_mixture_followup"
MAX_PARALLEL="${MAX_PARALLEL:-100}"
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL < 1 || MAX_PARALLEL > 200 )); then
  echo "MAX_PARALLEL must be 1..200" >&2
  exit 2
fi
if [[ -e "$OUT_ROOT/submission.lock" ]] || [[ -d "$PARTIAL_ROOT" && -n "$(find "$PARTIAL_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "Existing submission/partials: inspect jobs and results before any retry. No jobs submitted." >&2
  exit 2
fi
module purge
module load python/3.12.4
source venv/bin/activate
export LD_LIBRARY_PATH="$PROJECT/venv/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ -z "${PIPELINE_DEPENDENCY:-}" ]]; then
  python -m population.discrimination.locked_mixture_followup verify --root "$PROJECT" --bundle "$LOCKED_BUNDLE"
fi
echo "Locked primary mixtures: RT, TR, RR, TT + hybrid readout refit"
echo "200 matched splits; 1000 fits; six starts per fit; array 0-199%$MAX_PARALLEL"
echo "Partials: $PARTIAL_ROOT"
echo "Merged/validated: $OUT_ROOT"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run passed; no jobs submitted."
  exit 0
fi
mkdir -p logs "$OUT_ROOT"
# Persistent receipt/lock prevents duplicate submission, including partial failures.
mkdir "$OUT_ROOT/submission.lock"
UPSTREAM_DEPENDENCY=()
if [[ -n "${PIPELINE_DEPENDENCY:-}" ]]; then
  PREPARE_JOB=$(sbatch --parsable --dependency="afterok:$PIPELINE_DEPENDENCY" --export=ALL \
    slurm/discrimination/19_prepare_locked_mixture.sbatch)
  PREPARE_JOB="${PREPARE_JOB%%;*}"
  printf 'prepare=%s\n' "$PREPARE_JOB" >> "$OUT_ROOT/submitted_jobs.txt"
  echo "Submitted locked-mixture preparation: $PREPARE_JOB"
  UPSTREAM_DEPENDENCY=(--dependency="afterok:$PREPARE_JOB")
fi
ARRAY_JOB=$(sbatch --parsable "${UPSTREAM_DEPENDENCY[@]}" --array="0-199%$MAX_PARALLEL" --export=ALL \
  slurm/discrimination/20_locked_mixture_array.sbatch)
ARRAY_JOB="${ARRAY_JOB%%;*}"
printf 'array=%s\n' "$ARRAY_JOB" >> "$OUT_ROOT/submitted_jobs.txt"
echo "Submitted array: $ARRAY_JOB"
MERGE_JOB=$(sbatch --parsable --dependency="afterok:$ARRAY_JOB" --export=ALL \
  slurm/discrimination/21_locked_mixture_merge.sbatch)
MERGE_JOB="${MERGE_JOB%%;*}"
printf 'merge_validate=%s\n' "$MERGE_JOB" >> "$OUT_ROOT/submitted_jobs.txt"
echo "Submitted dependent merge/validation: $MERGE_JOB"
echo "Monitor: squeue -j $ARRAY_JOB,$MERGE_JOB"
if [[ -n "${PIPELINE_JOB_ID_FILE:-}" ]]; then
  printf '%s\n' "$MERGE_JOB" > "$PIPELINE_JOB_ID_FILE"
fi
