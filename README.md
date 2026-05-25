# SCS Periodicity Behavioral Prediction

This repository accompanies the manuscript:

**Axonal spike-count regimes link spinal cord stimulation periodicity to
artificial sensory detection and discrimination in rodents**

It contains reviewer-facing notebooks, figure-generation code, and an optional
HPC workflow for regenerating the simulation and behavioral-model outputs.

## Quick Start

The easiest way to inspect the analysis or recreate manuscript figures is:

1. Clone this GitHub repository.
2. Download `scs-periodicity-release-data.zip` from the GitHub Release
   or the matching Zenodo record.
3. Extract the ZIP into the repository root. It populates `data/` and
   `notebooks/cache/`.
4. Install the notebook environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Open `notebooks/06_FigureGeneration.ipynb` and run its cells:

```powershell
jupyter lab notebooks/06_FigureGeneration.ipynb
```

Generated manuscript figures are written to `results/figures/`.

## Contents

```text
notebooks/       Ordered review notebooks for stages 01-06
notebooks/cache/ Cached intermediate tables and static composite figures
src/             Analysis, plotting, and local release-packaging code
data/            Small pipeline inputs in Git; full outputs from release ZIP
hpc_tools/       Longleaf/SLURM regeneration workflow and MRG installer
results/         Local manuscript figure outputs
```

The notebooks follow the analysis in order:

1. `01_TimePatterns_Generator.ipynb`
2. `02_SpaceTimeV_Generator.ipynb`
3. `03_DetectionThresholds.ipynb`
4. `04_FeatureEngineering.ipynb`
5. `05_PeriodicityDiscrimination.ipynb`
6. `06_FigureGeneration.ipynb`

## Data Distribution

Git tracks the code and the small source inputs needed for full regeneration,
including COMSOL exports, source data tables, and the static composite figures.
Generated pipeline outputs and computed notebook caches are distributed in one
release archive:

```text
scs-periodicity-release-data.zip
  data/
  notebooks/cache/
  LICENSE
  THIRD_PARTY_NOTICES.md
  CITATION.cff
```

Attach the ZIP, its SHA-256 checksum, and the generated manifest CSV to each
GitHub Release. Because Zenodo's GitHub integration may archive only the source
ZIP, assemble the matching Zenodo version as a draft containing the GitHub
source ZIP and these three release-data assets, then publish the complete
version once. This gives reviewers one extraction step while preserving an
archived DOI-backed copy of the released data.

The figure notebook uses computed cache files by default for fast redraws. To
recompute figure intermediates from the archived pipeline data, set
`USE_NB_CACHE = False` in its first cell and rerun the notebook.

## Full Regeneration

Full regeneration requires an HPC system with SLURM and NEURON. Instructions
for uploading a minimal HPC workspace, running a pilot, running the full
pipeline, and retrieving the resulting data are in [hpc_tools/README.md](hpc_tools/README.md).

After retrieving new HPC outputs and rebuilding the local notebook cache, a
maintainer creates the release asset locally with:

```powershell
python src/package_data.py
```

This writes:

```text
scs-periodicity-release-data.zip
scs-periodicity-release-data.sha256
scs-periodicity-release-data-manifest.csv
```

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff) and
[`.zenodo.json`](.zenodo.json). Archived releases are available from Zenodo
through the version-independent concept DOI
[10.5281/zenodo.20370976](https://doi.org/10.5281/zenodo.20370976); cite the
specific version used for an analysis.

## License

Original project code is released under the BSD 3-Clause License, and
project-authored research data and derived tables are released under CC BY 4.0.
The upstream MRG model is retrieved from ModelDB only when running optional HPC
regeneration. Figures 1 and 2 include BioRender content and are excluded from
the CC BY data grant. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
