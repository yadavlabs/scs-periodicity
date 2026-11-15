# HPC regeneration

This folder contains the optional full regeneration workflow for the 420-axon
population analysis. Readers using the released data ZIP do not need Longleaf.
The examples use UNC Longleaf; replace the account and work paths for your own
allocation. `slurm/README.md` describes the dependent job stages.

## 1. Prepare a clean source directory

Extract `scs-periodicity-1.1.0.zip` and enter its `scs-periodicity-1.1.0/`
directory. The paired data ZIP contains four small historical inputs needed to
reproduce the accepted detection fit and CV0 amplitudes. Restore only those
files; the remaining generated data can stay in the ZIP:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe hpc_tools/restore_behavior_fit_inputs.py `
  --archive PATH_TO_RELEASE_DATA_ZIP
$env:PYTHONPATH = Join-Path $PWD 'src'
.\.venv\Scripts\python.exe hpc_tools/preflight_population.py --root .
.\.venv\Scripts\python.exe hpc_tools/verify_behavior_fit_inputs.py
```

The preflight should report 35 COMSOL trajectories, 420 axons, 55 time
patterns, and index axon `S01_140um_ph1`. The input verifier checks the
archived pre-Imax tables and their fingerprints. This step also works in a Git
checkout after obtaining the paired data ZIP.

## 2. Upload and submit

From the clean source directory in PowerShell, use your Longleaf username and
assigned work directory. Replace `FIRST/SECOND/YOUR_ONYEN` with its actual
path components:

```powershell
python hpc_tools/repo_upload_hpc.py `
  --host YOUR_ONYEN@longleaf.unc.edu `
  --remote-dir /work/users/FIRST/SECOND/YOUR_ONYEN `
  --project-name scs-periodicity-population-v1.1.0 `
  --setup-env --submit
```

The uploader transfers source inputs and code, installs the smaller
`hpc_requirements.txt` environment, downloads the pinned MRG model from
ModelDB, builds the population HOC template, and submits the full Slurm
dependency chain. The upstream MRG source is not redistributed here.
`--list-files --dry-run` shows the upload selection and remote command without
transferring data. Use a fresh `--project-name` for an independent run.

For a small NEURON check before the full run, use a separate project name and
replace `--submit` with `--submit-pilot`. The pilot runs three impulse/train
conditions and writes its reports under `logs/pilot/` on Longleaf.

## 3. Monitor the run

On Longleaf, enter the uploaded project directory. The coordinator records its
final Slurm job ID in `logs/pipeline/final_job.txt`:

```bash
cat logs/pipeline/final_job.txt
squeue -u "$USER"
sacct -j FINAL_JOB_ID -X -P -n --format=JobID,State,ExitCode,Elapsed
```

The final job should show `COMPLETED` and `0:0`, and
`logs/pipeline/validation.json` should say `"status": "ready"`. If a final
job is cancelled, inspect earlier stage states and the matching file under
`logs/` before downloading. Slurm partials and large intermediate SpaceTimeV
files stay on Longleaf. The simulation arrays use up to 24 CPUs per task; the
observer arrays use one CPU per Monte Carlo shard.

## 4. Download and verify locally

Return to the clean local source directory. Download only the files listed in
`release_output_paths.txt`:

```powershell
python hpc_tools/download_data.py `
  --kind release `
  --host YOUR_ONYEN@longleaf.unc.edu `
  --remote-project /work/users/FIRST/SECOND/YOUR_ONYEN/scs-periodicity-population-v1.1.0
```

The downloader validates archive members and Slurm validation reports before
installing results. It needs space for about 9.5 GiB of extracted results plus
its transfer archive. Use `--archive PATH_ON_LARGE_DRIVE --keep-archive` to keep
the transfer archive elsewhere; `--from-archive --archive PATH` resumes a
completed transfer without SSH.

Use the local environment from step 1 to execute and check notebook 06:

```powershell
.\.venv\Scripts\python.exe hpc_tools/release_checks/verify_population_figures.py --python .\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe hpc_tools/release_checks/finalize_population_figures.py
```

For a strict comparison with the accepted manuscript archive, extract the
paired data ZIP into a *separate* reference directory and run:

```powershell
$reference = 'PATH_TO_ACCEPTED_DATA_EXTRACTION'
.\.venv\Scripts\python.exe hpc_tools/release_checks/verify_accepted_cv0.py --candidate-dir data/features --reference (Join-Path $reference 'data/features')
.\.venv\Scripts\python.exe hpc_tools/release_checks/compare_release_reference.py --reference $reference
```

The comparison covers the accepted CV0 tables and trace archive plus 26
manuscript-facing detection, observer, and figure-statistics tables. The 26-table
check is deliberately strict about full numeric precision; the completed
Longleaf audit and its small optimizer differences are described in
`PROVENANCE.md`. After reviewing those checks, build the paired release data ZIP with
`python src/package_data.py`.
