# Population detection jobs

## Branch-aware I_max plateau pilot

`run_branch_plateau_pilot.sh` is a diagnostic-only test of an endpoint-
independent `I_max` definition. By default it selects eight deterministic,
amplitude/count-spanning repetitions at each of CV 0.8, 0.9, and 1.0 for
200 Hz. Each of the 24 array tasks performs an independent 25-uA sweep from
`I_50` through the diagnostic safety bound. The safety-bound response and the
global sampled maximum are recorded but are never used to select a threshold.

The primary candidate is the first 100-uA stable window that remains stable
for another 100 uA with a 1%-of-input-pulse count tolerance. A sustained
decline makes later plateaus ineligible. The merge also compares exact, 1%,
and 2% count tolerances with 100/150-uA candidate and 50/100-uA confirmation
windows. All outputs remain isolated under
`results/detection/branch_plateau_pilot/`; production CSVs are
read-only.

Submit with:

```bash
bash slurm/detection/run_branch_plateau_pilot.sh
```

The merged outputs include the complete diagnostic sweeps, every candidate
classification, the primary per-repetition summary, definition-comparison
tables, and a multipanel audit figure. A condition with a short shoulder
followed by sustained decline is reported as `blocked_before_plateau`; the
algorithm does not jump to a later post-decline branch.

### Final cross-frequency branch pilot

The follow-up pilot samples the historical high-amplitude branch and matched
ordinary controls at 10, 20, 50, 100, and 200 Hz. By default it retains all
high-branch cases when there are at most eight at a frequency, otherwise it
chooses eight cases spanning CV, old threshold amplitude, and spike count. It
adds four count/CV-matched ordinary controls per frequency. The resulting
default manifest contains 52 conditions.

```bash
bash slurm/detection/run_branch_plateau_cross_frequency_pilot.sh
```

This pilot uses the 2%-of-input-pulses, 100-uA candidate plus 100-uA
confirmation rule as its primary definition while retaining all exact, 1%,
and 2% candidate classifications. Outputs are isolated under
`results/detection/branch_plateau_cross_frequency_pilot/`.
`branch_plateau_sweeps.csv` and the per-condition sweep partials retain every
tested amplitude, index-axon spike count, fraction of input pulses, and full
spike-time array, so nonlinear and nonmonotonic example curves can be rebuilt
without rerunning NEURON.

### Peak-first I_max frequency-by-CV validation

The final diagnostic pilot tests the proposed peak-first, plateau-fallback
definition in every frequency-by-CV cell. It selects one repetition at CV0
and three repetitions spanning the low, median, and high historical `I_max`
amplitudes at every nonzero CV. The default manifest therefore contains 155
conditions across five frequencies and eleven CVs.

```bash
bash slurm/detection/run_peak_first_imax_validation_pilot.sh
```

Each condition is sampled every 25 uA from `I_50` through 500 uA above the
rounded grid start. Responses at 800, 1200, 1600, and 2000 uA are retained as
late diagnostic checkpoints when they fall beyond that continuous window.
Only the continuous early-search samples are eligible for selection. A clear
first response peak takes priority; otherwise the first valid pre-decline
plateau is selected using exact, then 1%, then 2% count tolerance.

Outputs remain isolated under
`results/detection/peak_first_imax_validation_pilot/`. The merge
produces per-condition decisions, peak-definition sensitivity across 300/350
uA windows and 2/3/4 confirmation samples, frequency-by-CV summaries, an
automatic acceptance report, and a compact audit figure. Full spike-time
arrays are retained at every sampled amplitude. Production thresholds are
never modified.

## Selective peak-first I_max migration

`run_peak_first_imax_update.sh` recomputes only the index axon's `I_all`
(`I_max`) thresholds. A 25-uA grid characterizes the first 500 uA of the
response branch beginning at `I_50`. A clear initial response-count peak is
selected first; otherwise the strictest stable pre-decline plateau is selected
using exact, then 1%, then 2% count tolerance. The selected local onset is
refined to 0.01 uA. Neither the response at 2000 uA nor a sampled global
maximum participates in selection. Existing fixed-count regime amplitudes and
responses are not simulated again.

The first phase writes a staged index-regime table and an invariance audit to
`results/detection/peak_first_imax_update/`; it never overwrites the
production table. The update audit retains every tested amplitude/count pair.
Population responses at the revised I_max amplitudes are a separate second
phase, run only after the staged thresholds have been audited.

For a later selector-only correction, set `PEAK_FIRST_IMAX_BASE_UPDATES_CSV`
to the prior complete update table, use a fresh partial/output directory, and
optionally set `PEAK_FIRST_IMAX_ARRAY_SPEC` (for example `43,49,53`). Each
selected condition replays the current selector against all retained sweeps,
simulates only repetitions that no longer agree, and still writes a complete
100-repetition partial. The merge replaces those complete cells and validates
all inherited rows under the current rule before assigning the new version.

Download the compact staged table and audit products (not the per-condition
partials) with:

```powershell
python hpc_tools/download_data.py `
  --kind peak-first-imax `
  --host YOUR_ONYEN@longleaf.unc.edu `
  --remote-project /work/users/Y/O/YOUR_ONYEN/scs-periodicity-population
```

## I_max high-amplitude branch diagnostic

The isolated plateau diagnostic can be submitted with:

```bash
bash slurm/detection/run_imax_plateau_diagnostic.sh
```

The default run selects two high-amplitude and two target-spike-count-matched
ordinary `I_all` repetitions at CV=1 for 20, 50, and 100 Hz. It therefore
submits 12 independent array tasks. Selection is recorded before submission in
`diagnostic_conditions.csv`; no production threshold or response file is
modified.

Each condition performs an independent 25-uA sweep from 200 to 2000 uA, then a
0.5-uA sweep around the production threshold, the first near-maximal stable
plateau, and up to four spike-count transitions above 500 uA. In addition to
index-axon spike count, the output records spike times, peak voltage, responses
per input pulse, and whether any pulse produced multiple detected spikes. Full
terminal-node voltage traces are saved only at the diagnostic landmarks.

The dependent merge job produces:

- `imax_amplitude_response_sweeps.png` and `.pdf`;
- `imax_sweep_samples.csv`;
- `imax_plateau_segments.csv`;
- `imax_condition_summary.csv` and `imax_branch_summary.csv`; and
- per-condition compressed voltage traces under `partials/`.

Outputs are isolated under
`results/detection/imax_plateau_diagnostic/`. Defaults may be
overridden without editing code, for example:

```bash
IMAX_DIAGNOSTIC_FREQUENCIES=10,20,50,100,200 \
IMAX_REPETITIONS_PER_BRANCH=1 \
bash slurm/detection/run_imax_plateau_diagnostic.sh
```

The first-stage diagnostic intentionally reruns only the index axon. Once it
identifies whether the late transition is genuine and which plateau definition
is defensible, selected amplitudes can be replayed across the 420-axon
population as a much smaller second-stage sensitivity analysis.

## Index operating-state resolution pilot

The completed pilot can be reproduced with:

```bash
bash slurm/detection/run_index_operating_state_pilot.sh
```

It submits eight independent diagnostic tasks: legacy Slack rep 0 and one
representative regular rep at 20, 50, 100, and 200 Hz. Each task uses 24
processes for its dense local sweep. The adaptive threshold search inside a
task remains sequential. The default threshold tolerance and local sweep
spacing are both 0.01 uA.

Outputs remain isolated under
`results/detection/index_operating_state_pilot/` and do not modify
production thresholds or population responses.

## Production rerun selected after the pilot

The production protocol is `canonical-cv0_tol0p01_peakfirst-imax_v3`:

- every CV0 repetition uses one generated exact-periodic waveform;
- the historical `rep001_slack` CV0 label is preserved only for joins;
- every named population regime is searched to 0.01 uA;
- all CV0 replay amplitudes come from CV0 (there is no all-CV onset safeguard);
- definitionally identical spike-count targets share the adaptive search; and
- identical amplitudes and equivalent repetitions share NEURON executions.

Submit the restartable full detection stage with:

```bash
bash slurm/detection/run_detection_pipeline.sh
```

Before submission, the launcher compares both the index axon and protocol
version on merged detection CSVs. Incompatible derived files are moved to
`results/detection/index-cache-archive/`, not deleted. Production
partials are namespaced by protocol and expected index axon under
`results/detection/partials/`; compatible partials from an
interrupted run remain restartable without a tens-of-thousands-file pre-scan.
