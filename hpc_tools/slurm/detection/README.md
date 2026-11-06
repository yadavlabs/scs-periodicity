# Detection jobs

`run_detection_pipeline.sh` submits the production detection stages after an
optional upstream dependency. The jobs compile the MRG mechanism, identify the
index axon, calculate peak-first Imax and other index regimes, run both
population-response branches, and merge the outputs. The root coordinator
then runs `08_finalize_detection.sbatch`, `09_true_trial_summary.sbatch`, and
`10_fit_accepted_behavior.sbatch` after their required inputs complete.

Run the complete release workflow with
`bash hpc_tools/slurm/run_population_pipeline.sh`. The detection launcher can
also be used by itself on an existing SpaceTimeV cohort. Partial simulation
files are namespaced by protocol and index axon under `data/detection/partials/`
and are excluded from the public data ZIP. Incompatible cached outputs are
quarantined under `logs/index-cache-archive/`.

The accepted behavioral fit uses the recovered pre-Imax input bundle under
`data/detection/behavior_fit_inputs/`; the current population response tables
use the final peak-first Imax definition. See `PROVENANCE.md`.
