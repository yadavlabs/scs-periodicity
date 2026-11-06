# Longleaf jobs

Run `bash hpc_tools/slurm/run_population_pipeline.sh` from the repository root
on Longleaf. It submits the population stages with Slurm `afterok` dependencies:
time patterns, SpaceTimeV, detection, features, discrimination, and release
validation. The Python uploader invokes this coordinator with `--submit`.

The directory has one production launcher for each analysis leg and the batch
files those launchers submit. `pilot/00_smoke.sbatch` is the separate three-case
NEURON check. The population needs more array and merge jobs than v1.0.2
because it simulates 420 axons and fits multiple observer scopes. Historical
exploratory pilots and staged Imax migration jobs are not part of this release
pipeline; the accepted peak-first Imax selector is in `src/detection/` and runs
in the production detection stage.

See `hpc_tools/README.md` for clean extraction, upload, validation, and download.
