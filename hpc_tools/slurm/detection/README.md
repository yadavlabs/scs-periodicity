# Detection SLURM Lane

These scripts rerun the NEURON/MRG detection stage used by the data release.
They are worker scripts: cluster scheduling options such as partition,
account, QOS, CPU count, memory, wall time, array ranges, and log paths are
passed by `hpc_tools/run_pipeline.py submit` rather than encoded in `#SBATCH`
headers.

For full end-to-end regeneration, use the top-level submitter:

```bash
python hpc_tools/query_hpc.py
python hpc_tools/run_pipeline.py submit --project "$PWD"
```

The threshold-array step writes `detection_thresholds.csv` partials and the
matching true-threshold raster partials that merge into
`data/detection/generated/detection_responses.csv`. The later response-array
step writes the CV0-based replay table used downstream:
`data/detection/generated/cv0_based_responses.csv`.
Transient partials live under `data/partials/detection/`, are removed by the
successful merge jobs, and are fully cleared by the finalizer before packaging.
SLURM stdout/stderr paths are supplied by the top-level submitter and normally
land in `logs/`.

The older detection-only submitter and shell wrapper have been removed. Use
`hpc_tools/run_pipeline.py submit` for pilot and full regeneration runs so
resource flags, dependencies, logs, and cleanup stay consistent across stages.
