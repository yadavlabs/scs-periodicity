# Population release v1.1.0

This version replaces the single-axon public workflow with the accepted
420-axon population analysis. The v1.0.2 Git tag and Zenodo record remain the
historical single-axon version.

## Release gate

An initial Longleaf rerun exposed a mismatch between the submitted v1 CV0
protocol and a later peak-first v3 update. The one-command coordinator now
generates v1 CV0 directly while retaining v3 for the true-CV branch. The
direct full-run CV0 features and trace archive match the accepted data. One
control-observer shard reached its original 10-hour Slurm request; its retry,
dependent merge, and final validation are pending. The public release remains
on hold until the final validation, figure notebook, and manuscript-facing
table comparison pass. Keep the accepted data ZIP as the reference until those
checks are complete. See `PROVENANCE.md` for the audit.

## Local release files

Build the source ZIP from the reviewed commit:

```powershell
git archive --format=zip --prefix=scs-periodicity-1.1.0/ `
  --output=scs-periodicity-1.1.0.zip HEAD
```

Build and verify the data files:

```powershell
python src/package_data.py
python hpc_tools/release_checks/verify_population_archive.py
```

The resulting Zenodo upload has four files:

- `scs-periodicity-1.1.0.zip` — population code and source inputs.
- `scs-periodicity-population-release-data.zip` — generated data and result files.
- `scs-periodicity-population-release-data-manifest.csv` — per-file SHA-256 hashes.
- `scs-periodicity-population-release-data.sha256` — data ZIP SHA-256.

The `.sha256` file records the verified data ZIP digest.
No single-axon paths or upstream MRG model files are present in either ZIP.
Extract the source ZIP first and the data ZIP into its top-level directory; the
archives have no overlapping file paths.

## Zenodo version

Use **New version** on the existing v1.0.2 record
(`10.5281/zenodo.20386515`; concept DOI `10.5281/zenodo.20370976`).
Set the version to `v1.1.0` and upload the four files above. Do not import the
old version's files, because they contain the historical single-axon analysis.
Keep the current title and creators; describe the 420-axon population revision.
The source software is BSD 3-Clause, project-authored research data are CC BY
4.0, and third-party terms are detailed in `THIRD_PARTY_NOTICES.md`.

Reserve the new version DOI before publication. Replace the v1.0.2 DOI and
version in the manuscript Data Availability statement and reproducibility
reference with the reserved v1.1.0 DOI. After the deposit and manuscript text
have been checked together, publish the Zenodo version and push the matching
Git tag.
