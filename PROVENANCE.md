# Population release provenance

The 420-axon analysis and accepted manuscript outputs were transferred from
`simulation-behavioral-prediction` commit
`a852a9c6ac82892a5cdb7398f6e267ffb6905ce5`. The previous public release,
v1.0.2, remains available in Git and Zenodo as the historical single-axon
analysis. The v1.1.0 source archive contains only population analysis code and
source inputs; generated tables and figures are in the paired data archive.

The cohort has 35 COMSOL trajectories, six axon diameters, and two nodal phases
per trajectory. The 420 Ve bases are listed in
`data/space-time-V/axon_basis_manifest.csv`. The configurable HOC is generated
locally from ModelDB 3810 at commit
`b15daeda6ea368df0198d257f339cf5e840cf31c`; the upstream HOC and
mechanism source are not redistributed. See `MRG_model/README.md` and
`THIRD_PARTY_NOTICES.md`.

## Accepted detection fit and Imax update

The submitted detection behavioral fit used the earlier index-axon `I_all`
amplitudes. A later peak-first Imax update changed 38 frequency-by-CV mean
`I_all` amplitudes and refreshed the true-CV population response table. The
accepted figures retain the submitted behavioral fit, while their final
true-CV recruitment panels use the refreshed responses.

The local analysis checkout retained all 5,500 old `I_all` amplitudes and
index-spike counts in `peak_first_imax_updates.csv`. They were used to recover
`data/detection/behavior_fit_inputs/index_regime_amplitudes_pre_imax.csv`.
The earlier 495 true-CV recruitment aggregates were recovered from the saved
submitted `detection_population_states.csv`; the CV0 aggregates were unchanged.
The original pre-update 20.8-million-row response table is not retained
locally. The exact inputs, method, and hashes are recorded in
`data/detection/behavior_fit_inputs/provenance.json` in the data archive.

Refitting with those recovered inputs reproduced all 12 submitted detection
fit CSVs exactly at the dataframe level, including weights, predictions, and
metrics. Refitting directly on the later Imax table gives different weights;
the release pipeline therefore uses the archived pre-update fit inputs for
the submitted behavioral fit. Notebook 03 still regenerates the final
phase/recruitment summaries from the current response tables.

## Verification before the full Longleaf run

- The local preflight passed with 35 trajectories, 420 axons, 55 time patterns,
  and index axon `S01_140um_ph1`.
- Longleaf pilot job `7866889` completed with exit code `0:0` and three passing
  NEURON checks across two diameters and both nodal phases.
- The 114-test Python suite passed after removing notebook-editing helpers.
- The accepted figure notebook executed all 20 code cells. Its nine regenerated
  statistics CSVs matched the analysis checkout exactly, and 62 export records
  across 140 figure files were validated.
- The independent true-CV summary stage reproduced all 49,500 saved trial
  totals exactly at the dataframe level from 20,790,000 population response
  rows.

The complete Longleaf dependency chain and release download are described in
`hpc_tools/README.md`. Its full regeneration has not yet been run on Longleaf.
