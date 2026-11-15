# Discrimination jobs

The root coordinator calls `run_population_discrimination_pipeline.sh` for the
independent control fits, `run_population_discrimination_scope_full.sh` for the
four manuscript scope hypotheses, and
`run_population_locked_mixture_followup.sh` for conditional diagnostics.
Each launcher submits preflight, array, merge, and validation jobs with
`afterok` dependencies. Production outputs are under
`data/discrimination/{control,scope,locked}/`; temporary partials are under
`logs/partials/` and are excluded from the release archive.

Run the complete workflow with
`bash hpc_tools/slurm/run_population_pipeline.sh`. The control and scope arrays
may run concurrently once population features are complete; the locked followup
waits for the validated scope fit.
