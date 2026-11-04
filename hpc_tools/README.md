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
submits `slurm/pilot/00_smoke.sbatch`. The pilot compiles the NEURON
mechanism on a compute node and runs three checks: an impulse and 20 Hz train
for the 14 µm index axon, and an impulse for a 5.7 µm axon with the other nodal
phase. It writes JSON results under `results/pilot/` and SLURM output
under `logs/`. It does not alter production thresholds or population responses.

Check the job with `squeue -u YOUR_ONYEN`, then inspect the three JSON files and
the matching `.err` log. A successful pilot has three `status: passed` records.
Longleaf pilot job `7866889` completed with exit code `0:0` and three passing
JSON records on 2026-11-02. It confirmed that the login node could download
the pinned ModelDB archive and the compute node could compile `AXNODE.mod`.

## Full population run

Extract the v1.1.0 source ZIP to a clean run directory. Extract the source ZIP
and the data ZIP to a separate reference directory for comparison. The source
ZIP contains a top-level `scs-periodicity-1.1.0/` folder; the data ZIP has
paths relative to that folder. In PowerShell, from the directory holding both
ZIPs:

```powershell
$runParent = Join-Path $PWD 'hpc-run'
$referenceParent = Join-Path $PWD 'accepted-reference'
Expand-Archive scs-periodicity-1.1.0.zip -DestinationPath $runParent
Expand-Archive scs-periodicity-1.1.0.zip -DestinationPath $referenceParent
$run = Join-Path $runParent 'scs-periodicity-1.1.0'
$reference = Join-Path $referenceParent 'scs-periodicity-1.1.0'
Expand-Archive scs-periodicity-population-release-data.zip -DestinationPath $reference
New-Item -ItemType Directory -Force (Join-Path $run 'data/detection/behavior_fit_inputs') | Out-Null
Copy-Item (Join-Path $reference 'data/detection/behavior_fit_inputs/*') `
  (Join-Path $run 'data/detection/behavior_fit_inputs')
Set-Location $run
```

The recovered inputs reproduce the detection fit used in the accepted
manuscript. The uploader transfers them but excludes completed response and
observer outputs. Keep the reference extraction separate while comparing the
newly generated tables and figures.

From the clean run directory, upload and submit to a fresh Longleaf project.
Do not pass `--skip-observer-inputs`: the Slack behavioral inputs are required.

```powershell
python hpc_tools/repo_upload_hpc.py `
  --host tvrdy9@longleaf.unc.edu `
  --remote-dir /work/users/t/v/tvrdy9 `
  --project-name scs-periodicity-population-v110-full `
  --setup-env --submit
```

The coordinator at `slurm/run_population_pipeline.sh` submits
the following dependent stages using Slurm `afterok`, as the v1.0.2 release
coordinator did:

1. Regenerate the 55 time-pattern bundles, then generate and validate 23,100
   SpaceTimeV files.
2. Compile NEURON, identify the index axon, calculate index regimes, and run
   both population response branches.
3. Execute the data-summary cells of detection notebook 03, independently
   summarize the 49,500 true-CV population trials, and refit the accepted
   detection observer from the verified pre-Imax input bundle.
4. Extract the CV0 population features and evidence.
5. Fit and validate the comprehensive observer controls and the four manuscript
   scope hypotheses. Their arrays can run concurrently after feature completion.
6. Prepare the locked primary mixtures from the validated scope fits, run the
   conditional follow-up, and validate all release outputs.

The final job ID is recorded remotely in `results/pipeline/final_job.txt`. After
it completes with Slurm state `COMPLETED` and exit code `0:0`, download the 76
files listed in `hpc_tools/release_output_paths.txt` into the clean local source
directory:

```powershell
python hpc_tools/download_data.py `
  --kind release `
  --host tvrdy9@longleaf.unc.edu `
  --remote-project /work/users/t/v/tvrdy9/scs-periodicity-population-v110-full
```

Install `requirements.txt` in the clean local run directory, execute notebook
06, finalize its exports, and compare the manuscript-facing tables with the
separate accepted-data extraction:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/verify_population_figures.py --python .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe scripts/finalize_population_figures.py
.\.venv\Scripts\python.exe hpc_tools/compare_release_reference.py --reference $reference
```

The comparison checks 26 detection-fit, observer, and figure-statistics tables.
Review the validation reports and any differences before building a new data
archive. The large intermediate SpaceTimeV files and Slurm partials stay on
Longleaf.

The uploader's `--list-files --dry-run` option shows every selected file. Do
not pass `--skip-upstream-inputs` for a new project: the pilot and full run need
the COMSOL cutlines, time patterns, and reference index screen.
