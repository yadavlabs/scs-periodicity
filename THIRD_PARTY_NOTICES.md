# Third-Party Notices

## MRG Axon Model

The optional HPC pipeline uses the MRG axon model from ModelDB accession 3810:

<https://modeldb.science/3810>

The upstream model identifies Cameron C. McIntyre as author and cites:

McIntyre CC, Richardson AG, and Grill WM. Modeling the excitability of
mammalian nerve fibers: influence of afterpotentials on the recovery cycle.
*Journal of Neurophysiology* 87:995-1006, 2002.

No explicit redistribution license was identified for these files. This
repository therefore does not redistribute the upstream model source or the
derived population HOC. During HPC setup, `hpc_tools/install_mrg_model.py`
downloads the upstream ModelDB mirror at a pinned revision and
`hpc_tools/build_population_mrg.py` generates the population HOC locally.
The generator replaces fixed geometry values with the six-diameter,
phase-dependent node configuration and removes the interactive IClamp panel;
the biophysical equations and diameter table remain those of the upstream
model. Users should cite both ModelDB and McIntyre et al. when using the model.

## BioRender Content in Figures 1 and 2

The accepted Figure 1 and Figure 2 composites in
`notebooks/cache/static-figures/` include content created with BioRender.com
under the UNC School of Medicine institutional plan. Their PNG and PDF exports
are included so the manuscript-facing notebook displays the complete figures.
They are excluded from the repository's CC-BY-4.0 data grant and remain subject
to applicable BioRender terms.

Created with BioRender.com.

Publication and attribution for these composites follow the applicable
BioRender and publisher terms.
