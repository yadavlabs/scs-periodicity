# MRG runtime files

This directory contains no redistributed MRG source. During HPC setup,
`hpc_tools/install_mrg_model.py` downloads ModelDB accession 3810 at the pinned
commit and generates `MRGaxon_population_template.hoc` locally. The generator
makes axon node count and fiber diameter configurable for the population cohort;
the spatial nodal phases are encoded in the population Ve bases. It removes the
upstream interactive IClamp panel because stimulation is provided through the
extracellular Ve playback. The remaining biophysical equations and diameter
table are unchanged.

The generated HOC, upstream files, and compiled mechanism are build artifacts
and are excluded from the repository and release source archive.
