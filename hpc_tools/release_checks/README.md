# Local release checks

These scripts verify downloaded population results and the paired data ZIP
on a local machine. They are included in the public source
archive so the release checks can be repeated, but the Longleaf uploader does
not transfer them to compute jobs.

- `verify_accepted_cv0.py` compares regenerated CV0 tables and traces with the accepted data.
- `verify_population_figures.py` executes notebook 06 against downloaded data.
- `finalize_population_figures.py` checks figure paths and hashes its exports.
- `compare_release_reference.py` compares 26 tables with the accepted data.
- `verify_population_archive.py` checks the data ZIP and SHA-256 manifest.

See `hpc_tools/README.md` for the command sequence.
