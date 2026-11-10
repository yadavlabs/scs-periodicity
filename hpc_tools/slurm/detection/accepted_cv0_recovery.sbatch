#!/bin/bash
#SBATCH --job-name=accepted_cv0
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/accepted_cv0_%j.out
#SBATCH --error=logs/accepted_cv0_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

case "${STAGE:?Set STAGE to pilot, prepare, trajectory, merge, or features}" in
  pilot)
    python -u -m detection.accepted_cv0 pilot --root "$PROJECT"
    ;;
  prepare)
    python -u -m detection.accepted_cv0 prepare --root "$PROJECT"
    ;;
  trajectory)
    python -u -m detection.accepted_cv0 trajectory --root "$PROJECT" \
      --task-index "${SLURM_ARRAY_TASK_ID:?}" --n-jobs "$SLURM_CPUS_PER_TASK"
    ;;
  merge)
    python -u -m detection.accepted_cv0 merge --root "$PROJECT"
    ;;
  features)
    python -u -m feature_engineering.feature_CLI run --root "$PROJECT" \
      --input logs/accepted-cv0-10hz/population_cv0_based_responses.csv \
      --features logs/accepted-cv0-10hz/population_cv0_based_features.csv \
      --evidence logs/accepted-cv0-10hz/population_cv0_template_evidence.csv \
      --collapsed-regimes logs/accepted-cv0-10hz/population_collapsed_regime_diagnostic.csv \
      --regime-equivalence logs/accepted-cv0-10hz/population_regime_response_equivalence.csv \
      --traces logs/accepted-cv0-10hz/population_cv0_time_representations_v5.npz \
      --expected-protocol canonical-cv0_tol0p01_v1 --overwrite
    ;;
  *)
    echo "Unknown accepted-CV0 stage: $STAGE" >&2
    exit 2
    ;;
esac
