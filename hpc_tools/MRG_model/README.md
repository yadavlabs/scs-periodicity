# MRG Model Runtime Directory

The MRG axon source files required by the optional NEURON/HPC regeneration
workflow are not redistributed in this repository.

During HPC setup, `hpc_tools/install_mrg_model.py` downloads ModelDB accession
3810 from ModelDB's official GitHub mirror at the pinned commit recorded in
that script. After installation, this directory contains the upstream model
files and the compiled NEURON mechanisms.

Source:

- ModelDB accession 3810: <https://modeldb.science/3810>
- McIntyre CC, Richardson AG, and Grill WM. Modeling the excitability of
  mammalian nerve fibers: influence of afterpotentials on the recovery cycle.
  *Journal of Neurophysiology*. 2002;87:995-1006.
