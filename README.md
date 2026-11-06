# SCS periodicity: population analysis

This repository is being updated for the accepted population-axon version of
**Axonal spike-count regimes link spinal cord stimulation periodicity to
artificial sensory detection and discrimination in rodents**. The current
working tree is a v1.1.0 release candidate; the published v1.0.2 archive remains
the historical single-axon release.

The population cohort has 35 COMSOL trajectories, six MRG diameters, and two
nodal phases per trajectory (420 axons). An index axon defines the stimulation
amplitudes; responses across the full cohort support detection and periodicity
discrimination analyses.

## Repository contents

- `notebooks/`: ordered notebooks for time patterns, Ve bases,
  detection, features, discrimination, and manuscript figures. Accepted static
  artwork exports for Figures 1 and 2 and Supplementary Figure 1 are under
  `notebooks/cache/static-figures/`.
- `src/`: population simulation, detection, discrimination, feature, and figure
  code. There is no single-axon package in this release.
- `data/`: source COMSOL cutlines, Slack inputs, generated time patterns, the
  420 Ve bases, and the reference index screen.
- `hpc_tools/`: Longleaf upload, model installation, Slurm jobs, pilot, and
  full regeneration workflows.

The upstream MRG files are downloaded at a pinned ModelDB commit during HPC
setup. A small generator makes the HOC geometry configurable for the six axon
diameters and two spatial phases; it preserves the upstream biophysics and does
not redistribute the upstream HOC. See [hpc_tools/MRG_model/README.md](hpc_tools/MRG_model/README.md)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The transfer and accepted-fit history are documented in [PROVENANCE.md](PROVENANCE.md).

## HPC test run

The transferred cohort passes the local preflight: 35 cutlines, 55 time
patterns, 420 Ve bases, and reference index axon `S01_140um_ph1`. Longleaf
pilot job `7866889` completed successfully with three passing checks across
two diameters and both nodal phases. See [hpc_tools/README.md](hpc_tools/README.md)
for the pilot and dependent full-run commands.

## Data release

Generated population response tables and fitted observer outputs are too large
for the Git source tree. `src/package_data.py` selects them and the figure
exports for the v1.1.0 data ZIP. The source ZIP contains the COMSOL, Slack,
time-pattern, Ve-basis, and reference inputs. The data ZIP contains generated
outputs and the recovered historical detection-fit inputs, with no duplicate
paths between the archives.
Run `python src/package_data.py --dry-run` to inspect the selection, then
`python src/package_data.py` to build the ZIP, SHA-256 checksum, and file
manifest. The packager rejects missing production outputs and historical
single-axon paths. After depositing the validated archive as a new Zenodo
version, use that version's DOI in the manuscript Data Availability statement.
The exact four-file deposit is described in [RELEASE_V1.1.0.md](RELEASE_V1.1.0.md).

## License

Original project code is BSD 3-Clause; project-authored research data and
derived tables are CC BY 4.0. Upstream MRG files and BioRender material have
separate rights described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
