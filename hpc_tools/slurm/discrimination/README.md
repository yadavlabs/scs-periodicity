# Discrimination SLURM Lane

These scripts rerun only the discrimination analyses retained in the release:

- per-block `rate_combined`, `timing_combined`, and `hybrid_combined` mixture fits
- per-point `hybrid_combined` mixture fits

They call `src/discrimination/discrimination_core.py`, which hard-codes the
release fitting settings and exposes only the analysis choices above.

The `.sbatch` files are worker scripts. Cluster scheduling options such as
partition, account, QOS, CPU count, memory, wall time, array ranges, and log
paths are passed by `hpc_tools/run_pipeline.py submit` rather than encoded in
`#SBATCH` headers.

For a full end-to-end regeneration, prefer the top-level submitter:

```bash
python hpc_tools/run_pipeline.py submit --project "$PWD"
```

The merge worker writes compact release outputs directly into
`data/discrimination/per-block` or `data/discrimination/per-point`, then removes
the consumed `data/partials/discrimination/...` shard tree after a successful
merge. The pipeline finalizer removes the top-level `data/partials/` directory
before packaging. For a quick stage-05 smoke test using existing archived
detection data, prefer the top-level pilot mode:

```bash
python hpc_tools/run_pipeline.py submit --pilot --project "$PWD"
```
