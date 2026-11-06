#!/usr/bin/env bash
set -euo pipefail

# Submit the complete restartable, index-anchored population detection stage.
# Existing partial CSVs are reused only when both index-axon identity and the
# production detection protocol match the current code.

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Each simulation task requests 24 CPUs. Forty-one concurrent tasks use 984 of
# the available 1000 CPUs.
MAX_PARALLEL_REGIMES="${MAX_PARALLEL_REGIMES:-41}"
# Total simultaneous response tasks across the true and CV0 branches. Each
# task uses 24 CPUs, so the default requests at most 984 CPUs.
MAX_PARALLEL_RESPONSES="${MAX_PARALLEL_RESPONSES:-41}"
MAX_ARRAY_TASKS="${MAX_ARRAY_TASKS:-1000}"
MAX_PRUNING_AUDIT_RATE="${MAX_PRUNING_AUDIT_RATE:-0.01}"
DIAMETER_PRUNING_AUDIT_RATE="${DIAMETER_PRUNING_AUDIT_RATE:-0.01}"
EXPECTED_INDEX_AXON_ID="${EXPECTED_INDEX_AXON_ID:-S01_140um_ph1}"
export MAX_PRUNING_AUDIT_RATE DIAMETER_PRUNING_AUDIT_RATE EXPECTED_INDEX_AXON_ID

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$PROJECT/data/detection/partials"

module purge
module load python/3.12.4
source venv/bin/activate
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
DETECTION_PROTOCOL_VERSION=$(python -c "from detection.protocol import DETECTION_PROTOCOL_VERSION; print(DETECTION_PROTOCOL_VERSION)" | tail -n 1)
export DETECTION_PROTOCOL_VERSION

# Regime and response caches contain amplitudes defined by a particular index
# axon. Preserve, but remove from active paths, anything anchored elsewhere.
python hpc_tools/prepare_population_index_cache.py \
  --root "$PROJECT" --expected-index-axon-id "$EXPECTED_INDEX_AXON_ID" \
  --expected-protocol-version "$DETECTION_PROTOCOL_VERSION"
python hpc_tools/prepare_population_index_cache.py \
  --root "$PROJECT" --expected-index-axon-id "$EXPECTED_INDEX_AXON_ID" \
  --expected-protocol-version "$DETECTION_PROTOCOL_VERSION" --apply
N_REGIME_CONDITIONS=$(python -c "from detection.detection_core import index_regime_conditions; print(len(index_regime_conditions()))" | tail -n 1)
if [[ -n "${RECRUITMENT_TASK_COUNT:-}" ]]; then
  N_RECRUITMENT_TASKS="$RECRUITMENT_TASK_COUNT"
else
  N_RECRUITMENT_TASKS=$(python -c "import os; from detection.detection_core import population_trajectory_condition_jobs; print(len(population_trajectory_condition_jobs(population_manifest=os.environ.get('POPULATION_MANIFEST'))))" | tail -n 1)
fi

REGIME_ARRAY="0-$((N_REGIME_CONDITIONS - 1))%${MAX_PARALLEL_REGIMES}"
if (( MAX_PARALLEL_RESPONSES < 2 )); then
  echo "MAX_PARALLEL_RESPONSES must be at least 2 (one task per response family)." >&2
  exit 2
fi
TRUE_PARALLEL=$(((MAX_PARALLEL_RESPONSES + 1) / 2))
CV0_PARALLEL=$((MAX_PARALLEL_RESPONSES / 2))

submit_recruitment_batches() {
  local dependency="$1"
  local script="$2"
  local parallel="$3"
  local start=0
  local end
  local jid
  local previous="$dependency"
  while (( start < N_RECRUITMENT_TASKS )); do
    end=$((start + MAX_ARRAY_TASKS - 1))
    if (( end >= N_RECRUITMENT_TASKS )); then
      end=$((N_RECRUITMENT_TASKS - 1))
    fi
    jid=$(sbatch --parsable \
      --dependency=afterok:"$previous" \
      --array="${start}-${end}%${parallel}" \
      --export=ALL \
      "$script")
    jid="${jid%%;*}"
    echo "Submitted $script array ${start}-${end}%${parallel}: $jid (after $previous)" >&2
    previous="$jid"
    start=$((end + 1))
  done
  echo "$previous"
}

echo "Submitting population detection workflow"
echo "  project=$PROJECT"
echo "  index-regime array=$REGIME_ARRAY"
echo "  recruitment tasks=$N_RECRUITMENT_TASKS in batches of at most $MAX_ARRAY_TASKS"
echo "  response concurrency: true=$TRUE_PARALLEL cv0=$CV0_PARALLEL total=$MAX_PARALLEL_RESPONSES"
echo "  pruning audits: maximum=$MAX_PRUNING_AUDIT_RATE diameter=$DIAMETER_PRUNING_AUDIT_RATE"
echo "  expected index axon=$EXPECTED_INDEX_AXON_ID"
echo "  detection protocol=$DETECTION_PROTOCOL_VERSION"
echo "  partial namespace=$DETECTION_PROTOCOL_VERSION/$EXPECTED_INDEX_AXON_ID"

UPSTREAM_DEPENDENCY=()
if [[ -n "${PIPELINE_DEPENDENCY:-}" ]]; then
  UPSTREAM_DEPENDENCY=(--dependency=afterok:"$PIPELINE_DEPENDENCY")
fi
JID_MOD=$(sbatch --parsable "${UPSTREAM_DEPENDENCY[@]}" --export=ALL hpc_tools/slurm/detection/compile_mechanisms.sbatch)
JID_MOD="${JID_MOD%%;*}"
JID_INDEX=$(sbatch --parsable --dependency=afterok:"$JID_MOD" --export=ALL hpc_tools/slurm/detection/identify_index.sbatch)
JID_INDEX="${JID_INDEX%%;*}"
JID_REGIMES=$(sbatch --parsable --dependency=afterok:"$JID_INDEX" --array="$REGIME_ARRAY" --export=ALL hpc_tools/slurm/detection/index_regimes_array.sbatch)
JID_REGIMES="${JID_REGIMES%%;*}"
JID_MERGE_REGIMES=$(sbatch --parsable --dependency=afterok:"$JID_REGIMES" --export=ALL hpc_tools/slurm/detection/merge_index_regimes_cv0.sbatch)
JID_MERGE_REGIMES="${JID_MERGE_REGIMES%%;*}"
JID_TRUE=$(submit_recruitment_batches "$JID_MERGE_REGIMES" hpc_tools/slurm/detection/population_responses_array.sbatch "$TRUE_PARALLEL")
JID_MERGE_TRUE=$(sbatch --parsable --dependency=afterok:"$JID_TRUE" --export=ALL hpc_tools/slurm/detection/merge_population_responses.sbatch)
JID_MERGE_TRUE="${JID_MERGE_TRUE%%;*}"
JID_CV0=$(submit_recruitment_batches "$JID_MERGE_REGIMES" hpc_tools/slurm/detection/population_cv0_responses_array.sbatch "$CV0_PARALLEL")
JID_MERGE_CV0=$(sbatch --parsable --dependency=afterok:"$JID_CV0" --export=ALL hpc_tools/slurm/detection/merge_population_cv0_responses.sbatch)
JID_MERGE_CV0="${JID_MERGE_CV0%%;*}"

echo "Submitted: mechanisms=$JID_MOD index=$JID_INDEX regimes=$JID_REGIMES merge-regimes=$JID_MERGE_REGIMES"
echo "Submitted: true=$JID_TRUE merge-true=$JID_MERGE_TRUE cv0=$JID_CV0 merge-cv0=$JID_MERGE_CV0"
mkdir -p "$PROJECT/logs/pipeline"
printf 'JID_MERGE_REGIMES=%s\nJID_MERGE_TRUE=%s\nJID_MERGE_CV0=%s\n' \
  "$JID_MERGE_REGIMES" "$JID_MERGE_TRUE" "$JID_MERGE_CV0" \
  > "$PROJECT/logs/pipeline/detection_jobs.env"
