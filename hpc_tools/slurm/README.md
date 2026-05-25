# SLURM Scripts

The release regeneration path is:

```text
hpc_tools/slurm/pipeline/
hpc_tools/slurm/detection/
hpc_tools/slurm/discrimination/
```

Those scripts are scheduler-policy neutral worker scripts. They do not encode
`#SBATCH` partition, account, QOS, CPU, memory, wall-time, array, or log-path
directives. `hpc_tools/run_pipeline.py submit` passes those values to `sbatch`
from the command line after the user chooses cluster-appropriate settings.

The active worker scripts live under `hpc_tools/` with the rest of the HPC-only
runtime. The old `manuscript/` nesting used during cleanup has been removed.
