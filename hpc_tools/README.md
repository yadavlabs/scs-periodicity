# Population HPC pilot and regeneration

The population workflow uses 35 COMSOL trajectories, six MRG diameters, and two
nodal phases: 420 axons. `hpc_tools/install_mrg_model.py` downloads ModelDB 3810
at a pinned commit, generates the configurable population HOC locally, and
compiles `AXNODE.mod` on the compute node. The upstream HOC and mechanism source
are not included in this repository or its upload archive.

## Pilot before a full run

From the repository root, install `requirements.txt` in a local
environment and validate the transferred input cohort. The remote setup uses
`hpc_tools/hpc_requirements.txt`, a smaller set for NEURON compute jobs:

```powershell
$env:PYTHONPATH = "$PWD/src"
python hpc_tools/preflight_population.py --root .
```

The preflight must report `status: ready`, 35 cutlines, 420 axons, 55 time
patterns, and index axon `S01_140um_ph1`. Then inspect the proposed transfer and
SLURM command. Replace the host and work directory with your own Longleaf paths:

```powershell
python hpc_tools/repo_upload_hpc.py `
  --host YOUR_ONYEN@longleaf.unc.edu `
  --remote-dir /work/users/Y/O/YOUR_ONYEN `
  --project-name scs-periodicity-population `
  --skip-observer-inputs `
  --setup-env `
  --submit-pilot `
  --dry-run
```

Remove `--dry-run` to upload and submit the pilot. This creates an isolated
Longleaf project, installs the Python environment and pinned MRG model, then
submits `slurm/population/pilot/00_smoke.sbatch`. The pilot compiles the NEURON
mechanism on a compute node and runs three checks: an impulse and 20 Hz train
for the 14 µm index axon, and an impulse for a 5.7 µm axon with the other nodal
phase. It writes JSON results under `results/population/pilot/` and SLURM output
under `logs/`. It does not alter production thresholds or population responses.

Check the job with `squeue -u YOUR_ONYEN`, then inspect the three JSON files and
the matching `.err` log. A successful pilot has three `status: passed` records.
Longleaf pilot job `7866889` completed with exit code `0:0` and three passing
JSON records on 2026-11-02. It confirmed that the login node could download
the pinned ModelDB archive and the compute node could compile `AXNODE.mod`.

## Full population run

After the pilot passes, use the same upload command with `--submit` in place of
`--submit-pilot`. The full launcher at
`slurm/population/run_population_pipeline.sh` generates 23,100 SpaceTimeV
conditions, validates them, and submits index-axon detection and both population
response branches. The feature and discrimination stages are described in
`slurm/population/discrimination/README.md`. Full output tables are large and
are distributed in the release-data archive rather than Git.

The uploader's `--list-files --dry-run` option shows every selected file. Do
not pass `--skip-upstream-inputs` for a new project: the pilot and full run need
the COMSOL cutlines, time patterns, and reference index screen.
