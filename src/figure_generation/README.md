# Population figure generation

The accepted figure notebook is `notebooks/06_FigureGeneration.ipynb`.
It reads the population outputs from the release data ZIP and writes figures,
source tables, statistics, and an export manifest under
`results/figures/`.

Figure 1, Figure 2, and Supplementary Figure 1 are accepted, manually assembled
composites. Their PNG/PDF exports are tracked under
`notebooks/cache/static-figures/`, copied to `results/figures/`, and displayed by
the notebook. The PNGs match the accepted manuscript images byte for byte.
Python regenerates the remaining figures and the component assets for Figures
1 and 2, but does not rebuild the manually assembled composites.

From the repository root, install `requirements.txt`, extract the release data
ZIP into the repository, and run:

```powershell
$env:PYTHONPATH = "$PWD/src"
python scripts/verify_population_figures.py
python scripts/finalize_population_figures.py
```

The verifier executes all notebook code cells and saves an executed copy and
report under `results/figure_verification/`. The finalizer checks
that every figure-manifest path exists, makes supplementary provenance paths
repository-relative, and records SHA-256 hashes of the exported figure files.
The source notebook remains free of machine-specific execution outputs.

`figure_generation.figure_model` builds the detection, discrimination, and model
comparison figures. The supplementary modules draw population rasters,
evidence curves, and recruitment maps. `figure_generation.statistics`
reconstructs the saved observer predictions and computes the manuscript
comparisons using 200 Monte Carlo splits and 100 simulated repetitions. These
statistics are conditional on the saved observer fits; they do not treat the
simulated repetitions as independent animals.

The expected production outputs and their validation status are documented in
`PROVENANCE.md`. Figure exports and supporting data are distributed
through the Zenodo release data ZIP, while the code and notebook are in the
source ZIP.
