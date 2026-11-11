# Detection jobs

`run_detection_pipeline.sh` submits the production detection stages after an
optional upstream dependency. The jobs compile the MRG mechanism, identify the
index axon, calculate peak-first Imax and other true-CV index regimes,
reconstruct the accepted v1 CV0 table, run both population-response branches,
and merge the outputs. The root coordinator
then runs `finalize_detection.sbatch`, `true_trial_summary.sbatch`, and
`fit_accepted_behavior.sbatch` after their required inputs complete.

Run the complete release workflow with
`bash hpc_tools/slurm/run_population_pipeline.sh`. The detection launcher can
also be used by itself on an existing SpaceTimeV cohort. Partial simulation
files are namespaced by protocol and index axon under `data/detection/partials/`
and are excluded from the public data ZIP. Incompatible cached outputs are
quarantined under `logs/index-cache-archive/`.

The accepted behavioral fit and CV0 table use the recovered pre-Imax input
bundle under `data/detection/behavior_fit_inputs/`. The true-CV population
responses use the final peak-first Imax definition. See `PROVENANCE.md`.

## Accepted CV0 branch recovery

The submitted observer figures used the earlier v1 CV0 response table, while
the refreshed true-CV response table uses peak-first v3. The archived pre-Imax
input reconstructs the five accepted CV0 index states exactly. Only the 10 Hz
`I_all` CV0 amplitude differs from the fresh v3 run. The recovery workflow
below was used to audit a completed v3 run; a clean full run now generates the
accepted v1 CV0 branch directly.

After uploading the updated code to the completed Longleaf project, run one
index-axon pilot:

```bash
sbatch --parsable --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --export=ALL,STAGE=pilot hpc_tools/slurm/detection/accepted_cv0_recovery.sbatch
```

Check `logs/accepted_cv0_pilot.json` against the accepted index-axon response
(CV 0.1, 10 Hz, repetition 15, `I_all`: 14 spikes at 256.14013671875 uA).
Then submit the dependent preparation, 770-task array, merge, and feature jobs:

```bash
bash hpc_tools/slurm/detection/run_accepted_cv0_recovery.sh
```

The array uses at most 41 tasks with 24 CPUs each (984 CPUs). The final job
writes four feature/evidence tables and a population trace archive. All 770
array tasks and both downstream jobs completed on Longleaf; the downloaded
outputs matched the accepted archive using
`hpc_tools/release_checks/verify_accepted_cv0_recovery.py`. Incorporation into
the full release pipeline and the manuscript-facing table checks remain pending.
