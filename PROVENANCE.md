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
mechanism source are not redistributed. See `hpc_tools/MRG_model/README.md` and
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
- The accepted figure notebook executed all 23 code cells. Its nine regenerated
  statistics CSVs matched the analysis checkout exactly, and 68 export records
  across 146 export files were validated. The added Figure 1, Figure 2, and
  Supplementary Figure 1 PNGs match the accepted manuscript images byte for
  byte; their matching PDFs are tracked in the source archive.
- The independent true-CV summary stage reproduced all 49,500 saved trial
  totals exactly at the dataframe level from 20,790,000 population response
  rows.

The complete Longleaf dependency chain and release download are described in
`hpc_tools/README.md`.

## Full Longleaf rerun audit

The full v1.1.0 pipeline completed on Longleaf in November 2026 after a feature
column-assignment bug was fixed and the downstream jobs were resubmitted. The
final pipeline validation reported `ready` for 49,500 trials. The local
download installed all 76 selected outputs; notebook 06 executed all 23 code
cells, and its export check validated 68 figure records. The existing accepted
data ZIP passed its SHA-256 and 228-entry archive check.

That initial fresh run is not an exact reproduction of the accepted archive. A strict
comparison matched 8 of 26 manuscript-facing tables; 18 differed. The
accepted CV0 response table was produced under detection protocol
`canonical-cv0_tol0p01_v1`, while that initial pipeline regenerated the branch
under `canonical-cv0_tol0p01_peakfirst-imax_v3`. This changes a small number of
feature values before the observer fits. Numerical fit results also differ:
the largest detection weight difference was 0.00086, and the largest control
observer mean RMSE difference was 0.00051. The new control summary adds three
parameter-count columns. The accepted data archive remains the reference for
the submitted figures and reported statistics; the fresh-run outputs are kept
separate for audit until the protocol difference is resolved.

The original v1 CV0 source is preserved in the analysis checkout at commit
`e3f6b570d220153a2dddf3f43bb4df0b8e151e4f`. The recovered pre-Imax input
reconstructs the accepted five-row CV0 index table exactly, including its file
hash after accounting for newline convention. The only amplitude difference
between accepted v1 CV0 and fresh v3 CV0 is `I_all` at 10 Hz:
256.14013671875 versus 256.13778084516525 uA. This affects 1,100 trial-level
amplitudes and four population spike counts in the current feature tables.
`src/detection/accepted_cv0.py` and the corresponding Slurm stages independently
resimulate those affected responses before rebuilding the CV0 features.
An independent local NEURON 9.0.2 pilot of the accepted 10 Hz `I_all` index
response (CV 0.1, repetition 15) reproduced all 14 archived spike times
exactly. The Longleaf pilot (job 8792808), all 770 recovery array tasks
(8792953), merge (8792954), and feature extraction (8792955) completed with
zero exit codes. The downloaded recovery matched all four accepted CV0
feature/evidence tables and every array in the accepted 1 ms population trace
archive. These are value-level comparisons; the recovered 20,790,000-row raw
response table remains on Longleaf in the isolated recovery directory. The
accepted data ZIP was preserved during the isolated recovery while the full
pipeline and downstream observer differences were investigated.

The one-command Slurm coordinator now reconstructs the accepted v1 CV0 index
table before its CV0 population array, while the true-CV branch retains the
peak-first v3 protocol. A clean full-run upload submitted the entire dependency
chain. One control-observer Monte Carlo shard completed nine of ten fits before
reaching its requested 10-hour wall time. That shard alone was retried with a
16-hour request, followed by its dependent merge, control validation, and final
release check. The production Slurm request was raised to 16 hours without
changing numerical code. Final validation reported `ready` for 49,500 trials;
the local download installed all 76 selected outputs. The four regenerated CV0
feature/evidence tables and full trace archive match the accepted reference
exactly.

Notebook 06 executed all 23 code cells and validated 68 figure records. A
strict comparison matched 8 of 26 manuscript-facing tables; 18 differ at full
numeric precision. The accepted behavioral fit was rerun locally from the same
archived inputs using the transferred source code and pinned NumPy, pandas,
and SciPy versions. All 12 detection-fit CSVs matched the accepted archive,
including with the single-thread environment variables used on Longleaf. The
Longleaf fit has small numeric differences, consistent with platform-dependent
optimization; this is an inference, not an isolated proof of the library cause.
The transferred observer fitting core differs from the analysis checkout only
in import and output paths; its numerical fitting code is identical. The Slack
parser has the same syntax tree, and all 104 behavioral CSVs are byte-identical.
The largest control-observer mean RMSE difference is 0.000853 in a training
`best_single` row; the largest test-set difference is 0.000451 in the same
configuration. Across mixture test summaries, the largest RMSE difference is
0.000032. The regenerated Table 1 retains the accepted model ranking, all RMSE
values to four decimals, rate shares to three decimals, and AIC/BIC differences
to integer precision. The reliance and RMSE tests have no changes in
significance decisions at p < 0.05 or in confidence-interval signs. One
manuscript p-value changes at its displayed precision: the Holm-adjusted
Wilcoxon comparison of independent hybrid and timing models in the std1 block
is 0.000316 in the accepted analysis
and 0.000336 in the Longleaf rerun. Its corrected-resampling sensitivity p-value
remains 0.565 to three decimals.
The accepted numerical outputs remain the reference for the published figures
and statistics; the full-run outputs remain separate for audit. The release
data ZIP was cleaned of four unused source CSV/HTML files from an optional
locked-mixture preview that notebook 06 does not call. Its figure promotion
manifest now lists the 142 files produced by the clean notebook. Every other
archived file is byte-identical to the pre-cleanup ZIP, except for the updated
promotion manifest. The rebuilt data ZIP passed its SHA-256 and 224-entry
manifest check; the 228-entry predecessor is retained in `.local-archive/`.
