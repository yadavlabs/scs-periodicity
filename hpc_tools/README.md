# HPC Regeneration

This folder contains the optional full-regeneration workflow: SLURM worker
files and the small helpers used to install the upstream MRG/NEURON model and
move a run to and from an HPC system. Reviewers using the released data archive
do not need this workflow.

The examples below use UNC Longleaf. Replace the username and work directory
for your account.

## 1. Upload and Set Up

From the local repository in PowerShell:

```powershell
python hpc_tools/repo_upload_hpc.py `
  --host YOUR_ONYEN@longleaf.unc.edu `
  --remote-dir /work/users/Y/YOUR_ONYEN `
  --setup-env
```

Add `--pilot-inputs` only when uploading the archived detection response needed
for a stage-05 pilot run. Without that flag the uploaded workspace contains
only inputs needed to regenerate the full data set.

The upload uses one SSH connection, extracts the minimal project at
`REMOTE_DIR/scs-periodicity`, creates `venv`, installs
`hpc_tools/hpc_requirements.txt` plus NEURON, and compiles the MRG mechanism.
The MRG axon files are downloaded during setup from ModelDB accession 3810 at
a pinned revision; they are not redistributed in this repository.

## 2. Open an HPC Shell

```bash
ssh YOUR_ONYEN@longleaf.unc.edu
cd /work/users/Y/YOUR_ONYEN/scs-periodicity
module purge
module load python/3.12.4
source venv/bin/activate
```

Load the Python module before activating `venv` in every new shell because the
environment depends on the module-provided Python shared library.

If setting up the environment manually rather than with `--setup-env`, install
and compile the upstream model after installing NEURON:

```bash
python hpc_tools/install_mrg_model.py --force
cd hpc_tools/MRG_model
nrnivmodl
cd ../..
```

## 3. Query and Dry Run

```bash
python hpc_tools/query_hpc.py

python hpc_tools/run_pipeline.py submit \
  --dry-run \
  --project "$PWD" \
  --python-module python/3.12.4 \
  --venv-activate "$PWD/venv/bin/activate" \
  --partition general \
  --worker-cpus 24 \
  --worker-mem 120G \
  --worker-time 10:00:00
```

On Longleaf, `general` worked for this pipeline. Account and QOS are normally
inferred from the user association; pass them only if required by the current
cluster configuration.

## 4. Pilot

A pilot tests the stage-05 discrimination workflow using the uploaded archived
detection response. It requires the original upload to use `--pilot-inputs`.

```bash
python hpc_tools/run_pipeline.py submit \
  --pilot \
  --project "$PWD" \
  --python-module python/3.12.4 \
  --venv-activate "$PWD/venv/bin/activate" \
  --partition general \
  --worker-cpus 24 \
  --worker-mem 120G \
  --worker-time 10:00:00
```

When it finishes, error logs should be empty:

```bash
find logs -maxdepth 1 -type f -name "*.err" -size +0 -print
```

Pilot model outputs use only two Monte Carlo splits and must not replace the
released 200-split data archive.

## 5. Full Run

```bash
python hpc_tools/run_pipeline.py submit \
  --project "$PWD" \
  --python-module python/3.12.4 \
  --venv-activate "$PWD/venv/bin/activate" \
  --partition general \
  --worker-cpus 24 \
  --worker-mem 120G \
  --worker-time 10:00:00 \
  --discrimination-mc-total 200 \
  --discrimination-mc-shards 10 \
  --discrimination-max-parallel 500
```

SLURM logs are written to `logs/`. Temporary array shards are written to
`data/partials/` during the run and removed by the finalizer. The default
stage-03 detection array reproduces the prior Longleaf resource limit:
55 tasks, 24 CPUs per task, and at most 40 concurrent tasks.

## 6. Retrieve Full Outputs

After the full finalizer job completes, return to the local repository and run:

```powershell
python hpc_tools/download_data.py `
  --host YOUR_ONYEN@longleaf.unc.edu `
  --remote-project /work/users/Y/YOUR_ONYEN/scs-periodicity
```

This downloads only generated pipeline output locations and replaces those
locations under the local `data/` folder after validating the transfer.

Run `notebooks/06_FigureGeneration.ipynb` locally with `USE_NB_CACHE = False`
once to refresh computed caches. Then create the public GitHub/Zenodo asset:

```powershell
python src/package_data.py
```
