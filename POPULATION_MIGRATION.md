# Population revision provenance

The population code and small source inputs in this release candidate were
transferred from `simulation-behavioral-prediction` commit
`a852a9c6ac82892a5cdb7398f6e267ffb6905ce5`. The previous
`scs-periodicity` baseline was `96ed1bf` (v1.0.2). The public release tree
contains population modules and notebooks; the earlier single-axon workflow
remains available only through historical Git and Zenodo versions.

The 420 production Ve bases were selected by the transferred
`axon_basis_manifest.csv`. Older exploratory Ve bases and incomplete SpaceTimeV
files from the analysis checkout were excluded. Manifest references and time
pattern metadata were made repository-relative.

The original population analysis used a locally modified MRG HOC. Its
executable statements are reproduced by `hpc_tools/build_population_mrg.py`
when applied to the local reference HOC, after ignoring comments and
formatting. The release build starts from ModelDB accession 3810 at commit
`b15daeda6ea368df0198d257f339cf5e840cf31c`. Longleaf pilot job `7866889`
confirmed download, local HOC generation, and `AXNODE.mod` compilation from
that pinned upstream source.

Local validation of this candidate:

- Population preflight: 35 trajectories, 420 axons, 55 time patterns, index
  axon `S01_140um_ph1`.
- NEURON pilot: index-axon impulse (1 spike), index-axon 20 Hz train (40
  spikes), and 5.7 µm phase-0 impulse (1 spike).
- Python tests: 117 passed, with two notebook-kernel tests skipped because
  this checkout has no repository-local `.venv`.
- Pilot upload dry run after cleanup: 713 files, about 0.02 GiB; archive contains no
  single-axon paths or redistributed MRG model files.
- Longleaf pilot job `7866889`: completed with exit code `0:0` in 43 seconds.
  All three JSON results reported `status: passed`; the spike counts (1, 40,
  and 1) matched the local checks.

The production response tables, feature tables, fitted observer results,
locked-mixture follow-up, and manuscript figure exports were copied from the
analysis checkout. All 202 initial copied files (9,782,882,001 bytes) matched
their source byte for byte; the 12 locked-mixture files and five locked-mixture
input files were also checked byte for byte. The true-CV Imax trial summary
and merge manifest were added for Supplementary Figure 5. The large response
tables and derived results are excluded from Git and selected for the data
archive by `src/package_data.py`. Source inputs and the 420 Ve bases are in
the source ZIP only; the data ZIP adds 221 generated files without overlapping
paths.

`scripts/verify_population_figures.py` executed all 20 code cells of the
accepted figure notebook against these copied outputs. All nine regenerated
statistics CSVs match the source exactly, including 3,200 reconstruction-audit
rows and 12,800 panel observations. Of the figure CSVs, 48 match byte for byte;
three regenerated numeric tables differ by at most 1.5e-14, consistent with
floating-point output rounding. `scripts/finalize_population_figures.py`
resolved 62 export-manifest entries, removed source-machine paths from the
supplementary provenance, and rebuilt the 140-file figure checksum manifest.

Before publication, the local analysis requirements were consolidated into
`requirements.txt` (the smaller HPC compute set remains separate), the 35
COMSOL cutline pairs were placed directly under
`data/comsol/` without changing their bytes, and obsolete notebook-assembly
helpers were removed. Historical ignored single-axon outputs were moved to
`.legacy-single-axon/` in this local checkout and remain outside both release
ZIPs. Preflight, the 119-test suite, and all 20 figure-notebook cells passed
again after these changes.

The v1.1.0 population data ZIP contains 221 files (9.16 GiB uncompressed;
859,888,203 bytes compressed). `scripts/verify_population_archive.py` read
every ZIP entry and confirmed its size and SHA-256 against the file manifest,
as well as the archive checksum. The archive SHA-256 is
`c8a710c29a6bdd895b9f60f8e609853f493027ff82e77bdc2912f57f7bb86e50`.
