# Local release checks

These scripts execute the figure notebook and check the release ZIP locally.
They are included in the source archive; the Longleaf uploader does not
transfer them to compute jobs.

- `verify_population_figures.py` executes notebook 06 against downloaded data.
- `finalize_population_figures.py` checks figure paths and hashes its exports.
- `verify_population_archive.py` checks the data ZIP and SHA-256 manifest.

See `hpc_tools/README.md` for the command sequence.
