#!/usr/bin/env bash
set -euo pipefail

# Recompute only the 10 Hz I_all responses whose accepted CV0 amplitude differs
# from the fresh peak-first run, then rebuild features in an isolated log tree.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
MAX_PARALLEL="${MAX_PARALLEL:-41}"
cd "$PROJECT"
mkdir -p logs/accepted-cv0-10hz

python - <<'PY'
import csv
import json
from pathlib import Path
p = Path("logs/accepted_cv0_pilot.json")
result = json.loads(p.read_text(encoding="utf-8"))
expected = {"protocol": "canonical-cv0_tol0p01_v1", "axon_id": "S01_140um_ph1",
            "cv": 0.1, "freq": 10.0, "rep_index": 15, "thresh_type": "I_all",
            "amp_uA": 256.14013671875, "n_spikes": 14}
if any(result.get(key) != value for key, value in expected.items()):
    raise SystemExit(f"Accepted-CV0 pilot did not match the accepted row: {result}")
expected_times = [
    101.44999395310879, 218.14998699724674, 317.7999810576439,
    520.0999689996243, 619.2499630898237, 804.6499520391226,
    986.5999411940575, 1093.3999348282814, 1197.8499286025763,
    1388.7999172210693, 1491.6999110877514, 1604.3499043732882,
    1776.7498940974474, 1881.5498878508806,
]
observed_times = result.get("spike_times_ms", [])
if len(observed_times) != len(expected_times) or any(
    abs(observed - expected) > 1e-6
    for observed, expected in zip(observed_times, expected_times)
):
    raise SystemExit("Accepted-CV0 pilot spike times differ from the accepted response")
print("Accepted-CV0 index-axon pilot matched all accepted spike times")
baseline = Path("data/detection/population_cv0_based_responses.csv")
with baseline.open(newline="", encoding="utf-8") as stream:
    first = next(csv.DictReader(stream))
if first["detection_protocol_version"] != "canonical-cv0_tol0p01_peakfirst-imax_v3":
    raise SystemExit("The fresh v3 CV0 baseline is missing; recovery cannot proceed")
print("Fresh v3 CV0 baseline is available")
PY

script=hpc_tools/slurm/detection/accepted_cv0_recovery.sbatch
prepare=$(sbatch --parsable --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --export=ALL,STAGE=prepare "$script")
prepare="${prepare%%;*}"
array=$(sbatch --parsable --dependency=afterok:"$prepare" \
  --array="0-769%${MAX_PARALLEL}" --cpus-per-task=24 --mem=16G --time=10:00:00 \
  --export=ALL,STAGE=trajectory "$script")
array="${array%%;*}"
merge=$(sbatch --parsable --dependency=afterok:"$array" \
  --cpus-per-task=1 --mem=32G --time=04:00:00 \
  --export=ALL,STAGE=merge "$script")
merge="${merge%%;*}"
features=$(sbatch --parsable --dependency=afterok:"$merge" \
  --cpus-per-task=4 --mem=64G --time=16:00:00 \
  --export=ALL,STAGE=features "$script")
features="${features%%;*}"

printf 'prepare=%s\narray=%s\nmerge=%s\nfeatures=%s\n' \
  "$prepare" "$array" "$merge" "$features" \
  > logs/accepted-cv0-10hz/jobs.txt
echo "Accepted CV0 recovery submitted: prepare=$prepare array=$array merge=$merge features=$features"
echo "Monitor with: squeue -j $array,$merge,$features"
