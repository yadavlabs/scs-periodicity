# Detection jobs

`run_detection_pipeline.sh` submits the production detection stages after an
optional upstream dependency. They compile the MRG mechanism, identify the
index axon, calculate peak-first true-CV index regimes, reconstruct the
accepted CV0 index table, run both population-response branches, and merge the
outputs. The root coordinator then runs the notebook-03 summary, independent
true-CV trial summary, and accepted behavioral fit jobs when their inputs are
ready.

Run the full workflow with `bash hpc_tools/slurm/run_population_pipeline.sh`.
The detection launcher can also run by itself on an existing SpaceTimeV cohort.
Partial simulation files are namespaced by protocol and index axon under
`data/detection/partials/`; incompatible cache files are moved under
`logs/index-cache-archive/` before submission. These working files are excluded
from the public data ZIP.

The accepted behavioral fit and CV0 table use the verified pre-Imax input
bundle under `data/detection/behavior_fit_inputs/`. True-CV population
responses use the final peak-first Imax definition. See `PROVENANCE.md` for the
reason these branches use different index amplitudes.
