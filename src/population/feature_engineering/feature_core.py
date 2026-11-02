"""Integrated population features derived from CV0-based detection responses.

The observer receives one population representation per trial, never a bank
of axon-specific features.  Axons are handled separately only while forming
the representation:

* the 1 ms trace stores the number of population spikes in each bin;
* the 10 ms trace stores how many axons occupy each variable-duration burst
  bin (contiguous occupied bins define a burst within an axon).

The binary support of each summed trace defines population events and silence
timing.  Trace magnitude is reserved for rate/intensity summaries.  This
preserves burst duration as timing information while preventing the number of
simultaneously active axons from directly entering ``silence_lv``.

Rate evidence uses endpoint-gated hard-JND votes: a feature can vote only when
its matched CV0/CV1 templates are themselves JND-separable.  Averaging the
gated votes makes endpoint reliability the natural evidence scale and prevents
intermediate CVs from exceeding it.  Timing evidence uses a bounded signed
CV0/CV1 distance.  Neither stream is endpoint-affine normalized or multiplied
by a CV-dependent reliability term.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from population.protocol import DETECTION_PROTOCOL_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV = Path("data/detection/population/population_cv0_based_responses.csv")
FEATURE_CSV = Path("data/features/population/population_cv0_based_features.csv")
EVIDENCE_CSV = Path("data/features/population/population_cv0_template_evidence.csv")
TRACE_NPZ = Path("data/features/population/population_cv0_time_representations_v5.npz")
COLLAPSED_REGIME_CSV = Path(
    "data/features/population/population_collapsed_regime_diagnostic.csv"
)
REGIME_EQUIVALENCE_CSV = Path(
    "data/features/population/population_regime_response_equivalence.csv"
)

T0_MS = 0.0
STIMULUS_NOMINAL_MS = 2000.0
# The latest input pulse in the current Slack/generated archive is 2020.65 ms,
# and detection simulates a 100-ms tail after the final input pulse.  A fixed
# 2130-ms window therefore retains every possible delayed response while
# remaining exactly aligned to both 1-ms and 10-ms representations.  The
# extractor explicitly fails if a future dataset exceeds this audited bound.
RESPONSE_PADDING_MS = 130.0
T1_MS = STIMULUS_NOMINAL_MS + RESPONSE_PADDING_MS
MRG_INIT_MS = 100.0
JND_K = 2.5
COLLAPSED_AMPLITUDE_TOLERANCE_UA = 0.01
FEATURE_VERSION = "population-v5-complete-window-response-equivalence"

REGIME_ORDER = (
    "I_first", "I_volley5", "I_volley10", "I_5", "I_10",
    "I_15", "I_25", "I_50", "I_all",
)
RESPONSE_EQUIVALENCE_COLUMNS = (
    "n_spikes_1ms",
    "n_events_10ms",
    "spikes_per_event_10ms",
    "silence_lv_1ms",
    "silence_lv_10ms",
)

TRIAL_KEY_COLUMNS = [
    "detection_protocol_version",
    "index_axon_id",
    "index_cutline_id",
    "index_diameter_um",
    "cv",
    "freq",
    "rep_index",
    "rep_label",
    "n_input_pulses",
    "thresh_type",
    "amp_uA",
]
OPTIONAL_TRIAL_COLUMNS = ["n_index_spikes"]
INPUT_COLUMNS = TRIAL_KEY_COLUMNS + OPTIONAL_TRIAL_COLUMNS + [
    "target_axon_id",
    "n_spikes",
    "spike_times_ms",
]
EVIDENCE_KEY_COLUMNS = [
    "detection_protocol_version", "index_axon_id", "freq", "rep_index", "thresh_type",
]


def _resolve(root: Path, path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _atomic_csv(table: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(output)
    return output


def _parse_spikes(value: object) -> np.ndarray:
    """Fast parser for JSON numeric lists written by the detection stage."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return np.empty(0, dtype=float)
    if isinstance(value, np.ndarray):
        out = np.asarray(value, dtype=float)
        return out[np.isfinite(out)]
    if isinstance(value, (list, tuple)):
        out = np.asarray(value, dtype=float)
        return out[np.isfinite(out)]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return np.empty(0, dtype=float)
    inner = text.strip("[]() ")
    if not inner:
        return np.empty(0, dtype=float)
    out = np.fromstring(inner, dtype=float, sep=",")
    if out.size == 0 and inner:
        try:
            out = np.asarray(json.loads(text), dtype=float)
        except Exception as exc:  # pragma: no cover - defensive malformed input
            raise ValueError(f"Could not parse spike list: {text[:80]!r}") from exc
    return out[np.isfinite(out)]


def _lv_details(sequence: np.ndarray) -> tuple[float, float, int]:
    """Return (LV, sum of LV terms, number of valid adjacent comparisons)."""
    values = np.asarray(sequence, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), 0.0, 0
    left = values[:-1]
    right = values[1:]
    denominator = left + right
    valid = np.isfinite(denominator) & (denominator > 0)
    if not np.any(valid):
        return float("nan"), 0.0, 0
    terms = 3.0 * ((left[valid] - right[valid]) / denominator[valid]) ** 2
    return float(np.mean(terms)), float(np.sum(terms)), int(terms.size)


def _iter_axon_blocks(
    input_csv: Path,
    *,
    read_chunksize: int,
) -> Iterator[pd.DataFrame]:
    """Yield contiguous target-axon blocks without assuming a block row count."""
    header = pd.read_csv(input_csv, nrows=0)
    missing = sorted(set(INPUT_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"Population response CSV is missing columns: {missing}")

    pending: list[pd.DataFrame] = []
    pending_id: str | None = None
    reader = pd.read_csv(input_csv, usecols=INPUT_COLUMNS, chunksize=int(read_chunksize))
    for chunk in reader:
        versions = chunk["detection_protocol_version"].dropna().astype(str).unique().tolist()
        if versions != [DETECTION_PROTOCOL_VERSION]:
            raise RuntimeError(
                "Population response CSV has incompatible detection protocol(s): "
                f"{versions}; expected {DETECTION_PROTOCOL_VERSION!r}"
            )
        ids = chunk["target_axon_id"].astype(str).to_numpy()
        if ids.size == 0:
            continue
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        ends = np.r_[starts[1:], ids.size]
        for start, end in zip(starts, ends):
            block_id = str(ids[start])
            part = chunk.iloc[int(start):int(end)].copy()
            if pending_id is None:
                pending_id = block_id
            if block_id != pending_id:
                yield pd.concat(pending, ignore_index=True)
                pending = []
                pending_id = block_id
            pending.append(part)
    if pending:
        yield pd.concat(pending, ignore_index=True)


def _trial_hash(table: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(table[TRIAL_KEY_COLUMNS], index=False).to_numpy()


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(np.asarray(numerator).shape, np.nan, dtype=float)
    denominator = np.asarray(denominator)
    valid = denominator > 0
    out[valid] = np.asarray(numerator, dtype=float)[valid] / denominator[valid]
    return out


def _binary_support_features(
    counts: np.ndarray,
    *,
    bin_ms: float,
) -> tuple[int, float, int, int, bool]:
    """Return event count, silence LV, LV comparisons, and active-bin count.

    Population events are contiguous runs in the binary support of a summed
    count trace.  Silence durations are measured as inactive bins before,
    between, and after those runs.  This is the binned population analogue of
    the single-axon event/silence representation and preserves burst duration.
    """
    active = np.asarray(counts) > 0
    occupied = np.flatnonzero(active)
    if occupied.size == 0:
        return 0, float("nan"), 0, 0, False

    run_starts_i = np.r_[0, np.flatnonzero(np.diff(occupied) > 1) + 1]
    run_ends_i = np.r_[run_starts_i[1:] - 1, occupied.size - 1]
    starts = occupied[run_starts_i]
    ends = occupied[run_ends_i]

    silence_bins = np.empty(starts.size + 1, dtype=float)
    silence_bins[0] = float(starts[0])
    if starts.size > 1:
        silence_bins[1:-1] = (starts[1:] - ends[:-1] - 1).astype(float)
    silence_bins[-1] = float(active.size - ends[-1] - 1)
    no_silence = bool(np.all(silence_bins == 0))
    if no_silence:
        # Documented limiting convention: a support that fills the complete
        # analysis window has two equal, zero-duration edge silences and hence
        # zero silence variability.  Keep a separate flag so this convention
        # is never confused with an ordinary positive-duration LV estimate.
        silence_lv = 0.0
        n_comparisons = max(0, int(silence_bins.size) - 1)
    else:
        silence_lv, _, n_comparisons = _lv_details(silence_bins * float(bin_ms))
    return int(starts.size), silence_lv, n_comparisons, int(occupied.size), no_silence


def _finalize_population_bins(
    population_spike_bins_1ms: np.memmap,
    population_burst_bins_10ms: np.memmap,
) -> dict[str, np.ndarray]:
    """Summarize the integrated 1 ms spike and 10 ms burst representations."""
    n_trials = int(population_spike_bins_1ms.shape[0])
    outputs = {
        "n_events_1ms": np.zeros(n_trials, dtype=np.int32),
        "silence_lv_1ms": np.full(n_trials, np.nan),
        "n_silence_lv_comparisons_1ms": np.zeros(n_trials, dtype=np.int32),
        "silence_lv_degenerate_no_silence_1ms": np.zeros(n_trials, dtype=bool),
        "n_active_population_bins_1ms": np.zeros(n_trials, dtype=np.int32),
        "population_duty_cycle_1ms": np.zeros(n_trials, dtype=float),
        "n_events_10ms": np.zeros(n_trials, dtype=np.int32),
        "silence_lv_10ms": np.full(n_trials, np.nan),
        "n_silence_lv_comparisons_10ms": np.zeros(n_trials, dtype=np.int32),
        "silence_lv_degenerate_no_silence_10ms": np.zeros(n_trials, dtype=bool),
        "n_active_population_bins_10ms": np.zeros(n_trials, dtype=np.int32),
        "population_duty_cycle_10ms": np.zeros(n_trials, dtype=float),
        "mean_active_axons_per_active_burst_bin_10ms": np.full(n_trials, np.nan),
        "max_active_axons_per_burst_bin_10ms": np.zeros(n_trials, dtype=np.int32),
        # Retained population-volley diagnostics; these are not primary evidence inputs.
        "mean_population_spikes_per_nonzero_bin_1ms": np.full(n_trials, np.nan),
        "sd_population_spikes_per_nonzero_bin_1ms": np.full(n_trials, np.nan),
        "cv_population_spikes_per_nonzero_bin_1ms": np.full(n_trials, np.nan),
        "lv_population_spikes_per_nonzero_bin_1ms": np.full(n_trials, np.nan),
        "lv_population_nonzero_bin_intervals_1ms": np.full(n_trials, np.nan),
    }
    for trial_index in tqdm(range(n_trials), desc="Integrated population bins", unit="trial"):
        spike_counts = np.asarray(population_spike_bins_1ms[trial_index])
        burst_counts = np.asarray(population_burst_bins_10ms[trial_index])

        n_events, silence_lv, n_comparisons, n_active, no_silence = _binary_support_features(
            spike_counts, bin_ms=1.0
        )
        outputs["n_events_1ms"][trial_index] = n_events
        outputs["silence_lv_1ms"][trial_index] = silence_lv
        outputs["n_silence_lv_comparisons_1ms"][trial_index] = n_comparisons
        outputs["silence_lv_degenerate_no_silence_1ms"][trial_index] = no_silence
        outputs["n_active_population_bins_1ms"][trial_index] = n_active
        outputs["population_duty_cycle_1ms"][trial_index] = n_active / float(spike_counts.size)

        n_events, silence_lv, n_comparisons, n_active, no_silence = _binary_support_features(
            burst_counts, bin_ms=10.0
        )
        outputs["n_events_10ms"][trial_index] = n_events
        outputs["silence_lv_10ms"][trial_index] = silence_lv
        outputs["n_silence_lv_comparisons_10ms"][trial_index] = n_comparisons
        outputs["silence_lv_degenerate_no_silence_10ms"][trial_index] = no_silence
        outputs["n_active_population_bins_10ms"][trial_index] = n_active
        outputs["population_duty_cycle_10ms"][trial_index] = n_active / float(burst_counts.size)
        if n_active:
            active_burst_counts = burst_counts[burst_counts > 0].astype(float)
            outputs["mean_active_axons_per_active_burst_bin_10ms"][trial_index] = float(
                np.mean(active_burst_counts)
            )
            outputs["max_active_axons_per_burst_bin_10ms"][trial_index] = int(
                np.max(active_burst_counts)
            )

        occupied = np.flatnonzero(spike_counts)
        if occupied.size == 0:
            continue
        sizes = spike_counts[occupied].astype(float)
        outputs["mean_population_spikes_per_nonzero_bin_1ms"][trial_index] = float(np.mean(sizes))
        if sizes.size >= 2:
            sd = float(np.std(sizes, ddof=1))
            outputs["sd_population_spikes_per_nonzero_bin_1ms"][trial_index] = sd
            mean = float(np.mean(sizes))
            if mean > 0:
                outputs["cv_population_spikes_per_nonzero_bin_1ms"][trial_index] = sd / mean
        size_lv, _, _ = _lv_details(sizes)
        outputs["lv_population_spikes_per_nonzero_bin_1ms"][trial_index] = size_lv
        interval_lv, _, _ = _lv_details(np.diff(occupied.astype(float)))
        outputs["lv_population_nonzero_bin_intervals_1ms"][trial_index] = interval_lv
    return outputs


def extract_population_features(
    *,
    root: Path = PROJECT_ROOT,
    input_csv: Path | str = INPUT_CSV,
    output_csv: Path | str = FEATURE_CSV,
    trace_npz: Path | str | None = TRACE_NPZ,
    read_chunksize: int = 100_000,
    overwrite: bool = False,
) -> Path:
    """Stream the CV0 response table and write one fixed-feature row per trial."""
    root = Path(root).resolve()
    input_path = _resolve(root, input_csv)
    output_path = _resolve(root, output_csv)
    trace_path = None if trace_npz is None else _resolve(root, trace_npz)
    if output_path.is_file() and (trace_path is None or trace_path.is_file()) and not overwrite:
        return output_path
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    blocks = _iter_axon_blocks(input_path, read_chunksize=read_chunksize)
    try:
        first = next(blocks)
    except StopIteration as exc:
        raise ValueError(f"Population response CSV is empty: {input_path}") from exc

    n_trials = len(first)
    if n_trials == 0 or first["target_axon_id"].nunique() != 1:
        raise RuntimeError("Could not identify the first target-axon response block")
    reference_hash = _trial_hash(first)
    metadata_columns = TRIAL_KEY_COLUMNS + [c for c in OPTIONAL_TRIAL_COLUMNS if c in first]
    metadata = first[metadata_columns].copy().reset_index(drop=True)

    total_spikes = np.zeros(n_trials, dtype=np.int64)
    source_total_spikes = np.zeros(n_trials, dtype=np.int64)
    active_axons = np.zeros(n_trials, dtype=np.int32)
    sum_sq_spikes = np.zeros(n_trials, dtype=np.float64)
    max_spikes = np.zeros(n_trials, dtype=np.int32)
    n_axons = 0
    source_count_mismatches = 0
    post_window_spikes = 0
    index_block_count = 0
    index_spikes_in_window: np.ndarray | None = None
    source_index_spikes: np.ndarray | None = None
    index_ids = metadata["index_axon_id"].astype(str).unique().tolist()
    if len(index_ids) != 1:
        raise RuntimeError(f"Expected one index axon id, found {index_ids}")
    index_axon_id = index_ids[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_temporary: Path | None = None
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_temporary = trace_path.with_suffix(trace_path.suffix + ".tmp")
    with tempfile.TemporaryDirectory(prefix="population-feature-bins-", dir=output_path.parent) as temp_dir:
        bin_path = Path(temp_dir) / "population_1ms_bins.dat"
        population_bins = np.memmap(
            bin_path,
            mode="w+",
            dtype=np.uint16,
            shape=(n_trials, int((T1_MS - T0_MS) / 1.0)),
        )
        population_bins[:] = 0
        burst_10ms_path = Path(temp_dir) / "population_burst_occupancy_10ms_bins.dat"
        population_burst_occupancy_10ms = np.memmap(
            burst_10ms_path,
            mode="w+",
            dtype=np.uint16,
            shape=(n_trials, int((T1_MS - T0_MS) / 10.0)),
        )
        population_burst_occupancy_10ms[:] = 0

        def consume(block: pd.DataFrame) -> None:
            nonlocal n_axons, source_count_mismatches, post_window_spikes
            nonlocal index_block_count, index_spikes_in_window, source_index_spikes
            if len(block) != n_trials:
                raise RuntimeError(
                    f"Target {block['target_axon_id'].iloc[0]!r} has {len(block):,} rows; "
                    f"expected {n_trials:,}"
                )
            if not np.array_equal(_trial_hash(block), reference_hash):
                raise RuntimeError(
                    f"Trial ordering differs for target {block['target_axon_id'].iloc[0]!r}"
                )
            n_axons += 1
            source_counts = pd.to_numeric(block["n_spikes"], errors="raise").astype(int).to_numpy()
            spike_values = block["spike_times_ms"].to_numpy()
            filtered_counts = np.zeros(n_trials, dtype=np.int64)
            source_total_spikes[:] += source_counts
            for trial_index, (source_count, value) in enumerate(zip(source_counts, spike_values)):
                if source_count <= 0:
                    continue
                parsed = _parse_spikes(value)
                if parsed.size != int(source_count):
                    source_count_mismatches += 1
                spikes = parsed - MRG_INIT_MS
                post_window_spikes += int(np.count_nonzero(spikes >= T1_MS))
                spikes = spikes[(spikes >= T0_MS) & (spikes < T1_MS)]
                n_filtered = int(spikes.size)
                filtered_counts[trial_index] = n_filtered
                if n_filtered == 0:
                    continue
                total_spikes[trial_index] += n_filtered
                active_axons[trial_index] += 1
                sum_sq_spikes[trial_index] += float(n_filtered * n_filtered)
                max_spikes[trial_index] = max(max_spikes[trial_index], n_filtered)

                indices_1ms = np.floor(spikes).astype(np.int64)
                indices_1ms = np.clip(indices_1ms, 0, population_bins.shape[1] - 1)
                unique_bins, bin_counts = np.unique(indices_1ms, return_counts=True)
                population_bins[trial_index, unique_bins] += bin_counts.astype(np.uint16)
                indices_10ms = np.floor(spikes / 10.0).astype(np.int64)
                indices_10ms = np.clip(
                    indices_10ms, 0, population_burst_occupancy_10ms.shape[1] - 1
                )
                # One unit per axon and occupied 10 ms bin. Consecutive occupied
                # bins form that axon's variable-duration burst envelope.
                population_burst_occupancy_10ms[
                    trial_index, np.unique(indices_10ms)
                ] += 1

            target_axon_id = str(block["target_axon_id"].iloc[0])
            if target_axon_id == index_axon_id:
                index_block_count += 1
                index_spikes_in_window = filtered_counts.copy()
                source_index_spikes = source_counts.astype(np.int64, copy=True)

        consume(first)
        progress = tqdm(blocks, desc="Population axons", unit="axon", initial=1)
        for block in progress:
            consume(block)
            progress.set_postfix(axons=n_axons)
        population_bins.flush()
        population_burst_occupancy_10ms.flush()

        if source_count_mismatches:
            raise RuntimeError(
                f"Parsed spike-list lengths disagreed with n_spikes in "
                f"{source_count_mismatches:,} response rows"
            )
        if post_window_spikes:
            raise RuntimeError(
                f"Found {post_window_spikes:,} response spikes at or after the audited "
                f"{T1_MS:g}-ms analysis endpoint; increase RESPONSE_PADDING_MS rather "
                "than silently truncating delayed responses"
            )
        if index_block_count != 1 or index_spikes_in_window is None or source_index_spikes is None:
            raise RuntimeError(
                f"Expected exactly one response block for index axon {index_axon_id!r}; "
                f"found {index_block_count}"
            )

        population = _finalize_population_bins(
            population_bins, population_burst_occupancy_10ms
        )
        binned_spike_totals = np.asarray(population_bins, dtype=np.uint64).sum(axis=1)
        if not np.array_equal(binned_spike_totals.astype(np.int64), total_spikes):
            raise RuntimeError("Population 1 ms bin counts do not reproduce filtered spike totals")
        if int(np.max(population_burst_occupancy_10ms)) > n_axons:
            raise RuntimeError("A 10 ms bin contains more active axons than the population size")

        features = metadata.copy()
        features.insert(0, "feature_version", FEATURE_VERSION)
        features["response_mode"] = "cv0_based"
        features["analysis_t0_ms"] = T0_MS
        features["analysis_t1_ms"] = T1_MS
        features["analysis_response_padding_ms"] = RESPONSE_PADDING_MS
        features["mrg_init_ms"] = MRG_INIT_MS
        features["n_population_axons"] = n_axons
        features["source_total_population_spikes"] = source_total_spikes
        features["total_population_spikes"] = total_spikes
        # CV0-expanded response jobs cannot know the index response before the
        # population simulation, so their regime metadata historically carried
        # NaN here.  Recover the actual per-trial count from the index axon's own
        # response block instead of copying the blank field from the first axon.
        features["source_n_index_spikes"] = source_index_spikes
        features["n_index_spikes"] = index_spikes_in_window
        features["n_spikes_1ms"] = total_spikes
        features["active_axon_count"] = active_axons
        features["active_axon_fraction"] = active_axons.astype(float) / float(n_axons)
        features["mean_spikes_per_active_axon"] = _safe_divide(total_spikes, active_axons)
        active_variance = np.full(n_trials, np.nan)
        enough = active_axons > 1
        active_variance[enough] = (
            sum_sq_spikes[enough]
            - total_spikes[enough].astype(float) ** 2 / active_axons[enough]
        ) / (active_axons[enough] - 1)
        features["sd_spikes_per_active_axon"] = np.sqrt(np.maximum(active_variance, 0.0))
        features["max_spikes_per_axon"] = max_spikes

        for column, values in population.items():
            features[column] = values
        features["spikes_per_event_1ms"] = _safe_divide(
            total_spikes, features["n_events_1ms"].to_numpy()
        )
        features["spikes_per_event_10ms"] = _safe_divide(
            total_spikes, features["n_events_10ms"].to_numpy()
        )

        if trace_temporary is not None:
            with trace_temporary.open("wb") as archive:
                np.savez_compressed(
                    archive,
                    feature_version=np.asarray(FEATURE_VERSION),
                    detection_protocol_version=np.asarray(DETECTION_PROTOCOL_VERSION),
                    analysis_t0_ms=np.asarray(T0_MS),
                    analysis_t1_ms=np.asarray(T1_MS),
                    analysis_response_padding_ms=np.asarray(RESPONSE_PADDING_MS),
                    mrg_init_ms=np.asarray(MRG_INIT_MS),
                    trial_row=np.arange(n_trials, dtype=np.int32),
                    cv=pd.to_numeric(features["cv"], errors="raise").to_numpy(np.float32),
                    freq=pd.to_numeric(features["freq"], errors="raise").to_numpy(np.float32),
                    rep_index=pd.to_numeric(features["rep_index"], errors="raise").to_numpy(np.int16),
                    thresh_type=features["thresh_type"].astype(str).to_numpy(dtype="U16"),
                    amp_uA=pd.to_numeric(features["amp_uA"], errors="raise").to_numpy(np.float64),
                    n_index_spikes=pd.to_numeric(
                        features["n_index_spikes"], errors="raise"
                    ).to_numpy(np.int32),
                    population_spike_counts_1ms=population_bins,
                    population_burst_occupancy_counts_10ms=population_burst_occupancy_10ms,
                )

        # Windows keeps the temporary file locked until the memmap is closed.
        del population_bins
        del population_burst_occupancy_10ms

    result = _atomic_csv(features, output_path)
    if trace_temporary is not None and trace_path is not None:
        trace_temporary.replace(trace_path)
    return result


def _within_jnd_array(x: np.ndarray, template: np.ndarray, k: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    template = np.asarray(template, dtype=float)
    out = np.zeros(x.shape, dtype=bool)
    finite = np.isfinite(x) & np.isfinite(template)
    positive = finite & (x > 0) & (template > 0)
    out[positive] = (
        np.maximum(x[positive], template[positive])
        / (np.minimum(x[positive], template[positive]) + 1e-9)
    ) <= float(k)
    nonpositive = finite & ~positive
    out[nonpositive] = x[nonpositive] == template[nonpositive]
    return out


def _rate_component_vote(
    table: pd.DataFrame,
    key: str,
    *,
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return endpoint-gated vote, trial validity, endpoint validity, and gate."""
    x = pd.to_numeric(table[key], errors="coerce").to_numpy(float)
    t0 = pd.to_numeric(table[f"{key}__cv0"], errors="coerce").to_numpy(float)
    t1 = pd.to_numeric(table[f"{key}__cv1"], errors="coerce").to_numpy(float)
    endpoint_valid = np.isfinite(t0) & np.isfinite(t1)
    trial_valid = endpoint_valid & np.isfinite(x)
    endpoint_separable = endpoint_valid & ~_within_jnd_array(t0, t1, k)

    within0 = _within_jnd_array(x, t0, k)
    within1 = _within_jnd_array(x, t1, k)
    vote = np.zeros(len(table), dtype=np.int8)
    vote[trial_valid & endpoint_separable & within0 & ~within1] = -1
    vote[trial_valid & endpoint_separable & within1 & ~within0] = 1
    return vote, trial_valid, endpoint_valid, endpoint_separable


def _rate_evidence(
    table: pd.DataFrame,
    keys: Sequence[str],
    *,
    k: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Average endpoint-gated component votes on their reliability scale.

    Endpoint-separability is fixed by each matched CV0/CV1 template pair and
    therefore cannot appear spuriously at an intermediate CV.  A component
    with inseparable endpoints contributes zero at every CV.  Averaging over
    endpoint-valid components (including their zero votes) guarantees that
    the curve is bounded by its endpoint reliability.
    """
    votes = np.zeros((len(table), len(keys)), dtype=np.int8)
    trial_valid = np.zeros((len(table), len(keys)), dtype=bool)
    endpoint_valid = np.zeros((len(table), len(keys)), dtype=bool)
    separable = np.zeros((len(table), len(keys)), dtype=bool)
    components: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key_index, key in enumerate(keys):
        vote, value_valid, templates_valid, gate = _rate_component_vote(table, key, k=k)
        votes[:, key_index] = vote
        trial_valid[:, key_index] = value_valid
        endpoint_valid[:, key_index] = templates_valid
        separable[:, key_index] = gate
        components[key] = (vote, value_valid, gate)

    denominator = endpoint_valid.sum(axis=1)
    evidence = np.zeros(len(table), dtype=float)
    reliability = np.zeros(len(table), dtype=float)
    has_templates = denominator > 0
    evidence[has_templates] = votes.sum(axis=1)[has_templates] / denominator[has_templates]
    reliability[has_templates] = separable.sum(axis=1)[has_templates] / denominator[has_templates]
    return (
        evidence,
        trial_valid.any(axis=1),
        reliability,
        separable.any(axis=1),
        components,
    )


def _bounded_signed_distance_evidence(
    table: pd.DataFrame,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded CV0/CV1 similarity evidence for one timing feature.

    ``(|x-t0| - |x-t1|) / (|x-t0| + |x-t1|)`` is -1 at the
    CV0 template, +1 at the CV1 template, and 0 when the templates are equally
    close.  Identical or unavailable templates provide no defined evidence.
    """
    x = pd.to_numeric(table[key], errors="coerce").to_numpy(float)
    t0 = pd.to_numeric(table[f"{key}__cv0"], errors="coerce").to_numpy(float)
    t1 = pd.to_numeric(table[f"{key}__cv1"], errors="coerce").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(t0) & np.isfinite(t1)
    d0 = np.abs(x - t0)
    d1 = np.abs(x - t1)
    denominator = d0 + d1
    valid = finite & np.isfinite(denominator) & (denominator > 1e-12)
    evidence = np.zeros(len(table), dtype=float)
    evidence[valid] = (d0[valid] - d1[valid]) / denominator[valid]
    evidence = np.clip(evidence, -1.0, 1.0)
    return evidence, valid


def _timing_endpoint_diagnostics(
    table: pd.DataFrame,
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CV0/CV1 templates plus absolute and relative separation."""
    t0 = pd.to_numeric(table[f"{key}__cv0"], errors="coerce").to_numpy(float)
    t1 = pd.to_numeric(table[f"{key}__cv1"], errors="coerce").to_numpy(float)
    valid = np.isfinite(t0) & np.isfinite(t1)
    absolute = np.full(len(table), np.nan)
    relative = np.full(len(table), np.nan)
    absolute[valid] = np.abs(t1[valid] - t0[valid])
    scale = np.abs(t0) + np.abs(t1)
    scaled = valid & (scale > 1e-12)
    relative[scaled] = absolute[scaled] / scale[scaled]
    relative[valid & ~scaled] = 0.0
    return t0, t1, absolute, relative, valid


def _collapsed_regime_diagnostic(
    features: pd.DataFrame,
    *,
    tolerance_uA: float,
) -> pd.DataFrame:
    """List regimes whose amplitudes collapse within the search tolerance."""
    group_columns = [
        "index_axon_id",
        "freq",
        "cv",
        "rep_index",
        "rep_label",
    ]
    rows: list[dict[str, object]] = []
    for group_key, block in features.groupby(group_columns, sort=False, dropna=False):
        candidates = (
            block[["thresh_type", "amp_uA"]]
            .drop_duplicates("thresh_type")
            .sort_values(["amp_uA", "thresh_type"])
        )
        cluster: list[tuple[str, float]] = []

        def emit(items: list[tuple[str, float]]) -> None:
            regimes = sorted({item[0] for item in items})
            if len(regimes) < 2:
                return
            amplitudes = np.asarray([item[1] for item in items], dtype=float)
            row = dict(zip(group_columns, group_key))
            row.update({
                "amplitude_tolerance_uA": float(tolerance_uA),
                "amp_min_uA": float(np.min(amplitudes)),
                "amp_max_uA": float(np.max(amplitudes)),
                "amp_span_uA": float(np.max(amplitudes) - np.min(amplitudes)),
                "n_collapsed_regimes": len(regimes),
                "collapsed_regimes": "|".join(regimes),
            })
            rows.append(row)

        for item in candidates.itertuples(index=False):
            current = (str(item.thresh_type), float(item.amp_uA))
            if cluster and current[1] - cluster[0][1] > float(tolerance_uA) + 1e-12:
                emit(cluster)
                cluster = []
            cluster.append(current)
        emit(cluster)

    columns = group_columns + [
        "amplitude_tolerance_uA",
        "amp_min_uA",
        "amp_max_uA",
        "amp_span_uA",
        "n_collapsed_regimes",
        "collapsed_regimes",
    ]
    return pd.DataFrame(rows, columns=columns)


def _nominal_regime_target(regime: str, n_input_pulses: int) -> int:
    """Return the spike-count target encoded by a regime name."""
    fixed = {"I_first": 1, "I_volley5": 5, "I_volley10": 10}
    if regime in fixed:
        return fixed[regime]
    if regime == "I_all":
        return int(n_input_pulses)
    if regime.startswith("I_"):
        percentage = int(regime.split("_", 1)[1])
        return max(1, int(np.ceil((percentage / 100.0) * int(n_input_pulses))))
    raise ValueError(f"Unknown regime name: {regime!r}")


def _regime_response_equivalence(
    features: pd.DataFrame,
    *,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Build frequency-wide equivalence classes from complete response tensors.

    Two operating states are merged only if all five raw population features
    that generate the observer inputs agree at every CV and simulation
    repetition for that frequency.  This is intentionally stricter than
    comparing bounded/JND-transformed evidence: two distinct responses must
    not collapse merely because a transform maps both to zero or a bound.
    Coincidence at one amplitude or one stimulation pattern is deliberately
    insufficient.

    The class label is selected by the named target closest to the realized
    CV0 index-spike count.  Names are combined only when the closest members
    have the same nominal target; otherwise the stable regime order breaks a
    residual equal-distance tie.
    """
    required = {
        "freq", "cv", "rep_index", "thresh_type", "n_input_pulses",
        "n_index_spikes", *RESPONSE_EQUIVALENCE_COLUMNS,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Cannot derive response-equivalence classes; missing {missing}")

    order_lookup = {name: index for index, name in enumerate(REGIME_ORDER)}
    rows: list[dict[str, object]] = []
    for freq, freq_table in features.groupby("freq", sort=True):
        regimes = [
            name for name in REGIME_ORDER
            if name in set(freq_table["thresh_type"].astype(str))
        ]
        extras = sorted(set(freq_table["thresh_type"].astype(str)).difference(regimes))
        regimes.extend(extras)
        signatures: dict[str, np.ndarray] = {}
        signature_keys: tuple[tuple[float, int], ...] | None = None
        for regime in regimes:
            block = freq_table[freq_table["thresh_type"].astype(str).eq(regime)].copy()
            block = block.sort_values(["cv", "rep_index"])
            keys = tuple(
                zip(
                    pd.to_numeric(block["cv"], errors="raise").astype(float),
                    pd.to_numeric(block["rep_index"], errors="raise").astype(int),
                )
            )
            if signature_keys is None:
                signature_keys = keys
            elif keys != signature_keys:
                raise RuntimeError(
                    f"Regime {regime} at {float(freq):g} Hz lacks complete CV/rep coverage"
                )
            signatures[regime] = block[list(RESPONSE_EQUIVALENCE_COLUMNS)].to_numpy(float)

        unassigned = set(regimes)
        classes: list[list[str]] = []
        for regime in regimes:
            if regime not in unassigned:
                continue
            members = [regime]
            unassigned.remove(regime)
            for candidate in regimes:
                if candidate not in unassigned:
                    continue
                if np.allclose(
                    signatures[regime], signatures[candidate],
                    rtol=0.0, atol=float(atol), equal_nan=True,
                ):
                    members.append(candidate)
                    unassigned.remove(candidate)
            classes.append(members)

        for class_index, members in enumerate(classes):
            class_table = freq_table[freq_table["thresh_type"].astype(str).isin(members)]
            n_input_values = pd.to_numeric(class_table["n_input_pulses"], errors="raise").unique()
            if len(n_input_values) != 1:
                raise RuntimeError(f"Frequency {freq:g} has inconsistent input-pulse counts")
            n_input = int(n_input_values[0])
            cv0 = class_table[np.isclose(pd.to_numeric(class_table["cv"], errors="raise"), 0.0)]
            observed_values = pd.to_numeric(cv0["n_index_spikes"], errors="raise").astype(int).unique()
            if len(observed_values) != 1:
                raise RuntimeError(
                    f"Response-equivalent class {members} at {float(freq):g} Hz has "
                    f"inconsistent CV0 index counts: {observed_values.tolist()}"
                )
            observed = int(observed_values[0])
            targets = {name: _nominal_regime_target(name, n_input) for name in members}
            minimum_error = min(abs(target - observed) for target in targets.values())
            nearest = [
                name for name in members if abs(targets[name] - observed) == minimum_error
            ]
            nearest_targets = {targets[name] for name in nearest}
            if len(nearest_targets) == 1:
                retained = sorted(nearest, key=lambda name: order_lookup.get(name, 10_000))
            else:
                retained = [min(nearest, key=lambda name: order_lookup.get(name, 10_000))]
            representative = retained[0]
            canonical = " + ".join(retained)
            dropped = [name for name in members if name not in retained]
            reference = signatures[members[0]]
            max_difference = max(
                float(np.nanmax(np.abs(reference - signatures[name])))
                if reference.size else 0.0
                for name in members
            )
            class_id = f"f{int(round(float(freq)))}_class{class_index + 1:02d}"
            targets_text = json.dumps(targets, sort_keys=True)
            for name in members:
                rows.append({
                    "freq": float(freq),
                    "thresh_type": name,
                    "equivalence_class_id": class_id,
                    "canonical_regime": canonical,
                    "representative_regime": representative,
                    "regime_active": bool(name == representative),
                    "member_regimes": "|".join(members),
                    "retained_regimes": "|".join(retained),
                    "dropped_regimes": "|".join(dropped),
                    "n_equivalent_regimes": len(members),
                    "nominal_n_input_pulses": n_input,
                    "observed_cv0_index_spikes": observed,
                    "nominal_target_spikes": targets[name],
                    "nominal_targets_by_member": targets_text,
                    "selection_rule": "closest_nominal_target_to_realized_cv0_count",
                    "equivalence_basis": "full_cv_x_rep_engineered_population_features",
                    "response_atol": float(atol),
                    "max_abs_response_difference": max_difference,
                })
    return pd.DataFrame(rows)


def build_population_template_evidence(
    *,
    root: Path = PROJECT_ROOT,
    feature_csv: Path | str = FEATURE_CSV,
    output_csv: Path | str = EVIDENCE_CSV,
    collapsed_regime_csv: Path | str = COLLAPSED_REGIME_CSV,
    regime_equivalence_csv: Path | str = REGIME_EQUIVALENCE_CSV,
    jnd_k: float = JND_K,
    collapsed_amplitude_tolerance_uA: float = COLLAPSED_AMPLITUDE_TOLERANCE_UA,
    overwrite: bool = False,
) -> Path:
    """Compare every trial with its matched CV0/CV1 population templates."""
    root = Path(root).resolve()
    feature_path = _resolve(root, feature_csv)
    output_path = _resolve(root, output_csv)
    collapsed_path = _resolve(root, collapsed_regime_csv)
    equivalence_path = _resolve(root, regime_equivalence_csv)
    if (
        output_path.is_file() and collapsed_path.is_file()
        and equivalence_path.is_file() and not overwrite
    ):
        return output_path
    features = pd.read_csv(feature_path, low_memory=False)
    missing = sorted(set(EVIDENCE_KEY_COLUMNS + ["cv"]) - set(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing evidence keys: {missing}")

    cv = pd.to_numeric(features["cv"], errors="coerce").to_numpy(float)
    endpoint_columns = [
        "n_spikes_1ms",
        "n_events_10ms",
        "spikes_per_event_10ms",
        "silence_lv_1ms",
        "silence_lv_10ms",
        "silence_lv_degenerate_no_silence_1ms",
        "silence_lv_degenerate_no_silence_10ms",
    ]
    for endpoint_cv, suffix in ((0.0, "cv0"), (1.0, "cv1")):
        endpoint = features.loc[np.isclose(cv, endpoint_cv), EVIDENCE_KEY_COLUMNS + endpoint_columns].copy()
        if endpoint.duplicated(EVIDENCE_KEY_COLUMNS).any():
            raise RuntimeError(f"CV{endpoint_cv:g} feature templates contain duplicate trial keys")
        endpoint = endpoint.rename(columns={column: f"{column}__{suffix}" for column in endpoint_columns})
        features = features.merge(endpoint, on=EVIDENCE_KEY_COLUMNS, how="left", validate="many_to_one")

    output_columns = TRIAL_KEY_COLUMNS + [
        "n_index_spikes",
        "feature_version",
        "response_mode",
        "analysis_t0_ms",
        "analysis_t1_ms",
        "analysis_response_padding_ms",
        "mrg_init_ms",
        "n_population_axons",
    ]
    evidence = features[[column for column in output_columns if column in features]].copy()
    evidence["jnd_k"] = float(jnd_k)
    # Silence is a response state, not automatically a missing observation.
    # Zero spike/event counts remain valid rate features. Timing features based
    # on silence variability are undefined for a completely silent population;
    # their evidence functions return zero and expose false availability flags.
    population_spikes = pd.to_numeric(features["n_spikes_1ms"], errors="coerce")
    template_spikes_cv0 = pd.to_numeric(features["n_spikes_1ms__cv0"], errors="coerce")
    template_spikes_cv1 = pd.to_numeric(features["n_spikes_1ms__cv1"], errors="coerce")
    evidence["population_silent"] = population_spikes.eq(0) & population_spikes.notna()
    evidence["population_template_cv0_silent"] = template_spikes_cv0.eq(0) & template_spikes_cv0.notna()
    evidence["population_template_cv1_silent"] = template_spikes_cv1.eq(0) & template_spikes_cv1.notna()
    for suffix, keys in [
        ("1ms", ["n_spikes_1ms"]),
        ("10ms", ["n_events_10ms", "spikes_per_event_10ms"]),
    ]:
        values, valid, reliability, available, components = _rate_evidence(
            features, keys, k=jnd_k
        )
        evidence[f"rate_evidence_{suffix}"] = values
        evidence[f"rate_evidence_{suffix}_defined"] = valid
        evidence[f"rate_endpoint_reliability_{suffix}"] = reliability
        evidence[f"rate_cue_available_{suffix}"] = available
        for key, (vote, component_valid, endpoint_separable) in components.items():
            evidence[f"rate_vote_{key}"] = vote
            evidence[f"rate_vote_{key}_defined"] = component_valid
            evidence[f"rate_endpoint_separable_{key}"] = endpoint_separable

    for suffix, key in [("1ms", "silence_lv_1ms"), ("10ms", "silence_lv_10ms")]:
        values, valid = _bounded_signed_distance_evidence(features, key)
        evidence[f"timing_evidence_{suffix}"] = values
        evidence[f"timing_evidence_{suffix}_defined"] = valid
        evidence[f"timing_cue_available_{suffix}"] = valid
        t0, t1, absolute, relative, separation_defined = _timing_endpoint_diagnostics(
            features, key
        )
        evidence[f"timing_template_cv0_{suffix}"] = t0
        evidence[f"timing_template_cv1_{suffix}"] = t1
        evidence[f"timing_endpoint_abs_delta_{suffix}"] = absolute
        evidence[f"timing_endpoint_relative_delta_{suffix}"] = relative
        evidence[f"timing_endpoint_separation_defined_{suffix}"] = separation_defined
        flag = f"silence_lv_degenerate_no_silence_{suffix}"
        evidence[f"timing_template_cv0_no_silence_{suffix}"] = features[
            f"{flag}__cv0"
        ].astype(bool)
        evidence[f"timing_template_cv1_no_silence_{suffix}"] = features[
            f"{flag}__cv1"
        ].astype(bool)

    diagnostic = _collapsed_regime_diagnostic(
        features,
        tolerance_uA=collapsed_amplitude_tolerance_uA,
    )
    _atomic_csv(diagnostic, collapsed_path)
    equivalence = _regime_response_equivalence(features)
    evidence = evidence.merge(
        equivalence,
        on=["freq", "thresh_type"],
        how="left",
        validate="many_to_one",
    )
    if evidence["equivalence_class_id"].isna().any():
        raise RuntimeError("Some evidence rows lack a response-equivalence assignment")
    _atomic_csv(equivalence, equivalence_path)
    return _atomic_csv(evidence, output_path)


def run_population_feature_pipeline(
    *,
    root: Path = PROJECT_ROOT,
    input_csv: Path | str = INPUT_CSV,
    feature_csv: Path | str = FEATURE_CSV,
    evidence_csv: Path | str = EVIDENCE_CSV,
    collapsed_regime_csv: Path | str = COLLAPSED_REGIME_CSV,
    regime_equivalence_csv: Path | str = REGIME_EQUIVALENCE_CSV,
    trace_npz: Path | str | None = TRACE_NPZ,
    read_chunksize: int = 100_000,
    jnd_k: float = JND_K,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    features = extract_population_features(
        root=root,
        input_csv=input_csv,
        output_csv=feature_csv,
        trace_npz=trace_npz,
        read_chunksize=read_chunksize,
        overwrite=overwrite,
    )
    evidence = build_population_template_evidence(
        root=root,
        feature_csv=features,
        output_csv=evidence_csv,
        collapsed_regime_csv=collapsed_regime_csv,
        regime_equivalence_csv=regime_equivalence_csv,
        jnd_k=jnd_k,
        overwrite=overwrite,
    )
    return features, evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=PROJECT_ROOT)
    common.add_argument("--overwrite", action="store_true")

    extract = sub.add_parser("extract", parents=[common])
    extract.add_argument("--input", type=Path, default=INPUT_CSV)
    extract.add_argument("--output", type=Path, default=FEATURE_CSV)
    extract.add_argument("--traces", type=Path, default=TRACE_NPZ)
    extract.add_argument("--read-chunksize", type=int, default=100_000)

    evidence = sub.add_parser("evidence", parents=[common])
    evidence.add_argument("--features", type=Path, default=FEATURE_CSV)
    evidence.add_argument("--output", type=Path, default=EVIDENCE_CSV)
    evidence.add_argument("--collapsed-regimes", type=Path, default=COLLAPSED_REGIME_CSV)
    evidence.add_argument("--regime-equivalence", type=Path, default=REGIME_EQUIVALENCE_CSV)
    evidence.add_argument("--jnd-k", type=float, default=JND_K)

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--input", type=Path, default=INPUT_CSV)
    run.add_argument("--features", type=Path, default=FEATURE_CSV)
    run.add_argument("--evidence", type=Path, default=EVIDENCE_CSV)
    run.add_argument("--collapsed-regimes", type=Path, default=COLLAPSED_REGIME_CSV)
    run.add_argument("--regime-equivalence", type=Path, default=REGIME_EQUIVALENCE_CSV)
    run.add_argument("--traces", type=Path, default=TRACE_NPZ)
    run.add_argument("--read-chunksize", type=int, default=100_000)
    run.add_argument("--jnd-k", type=float, default=JND_K)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        print(extract_population_features(
            root=args.root,
            input_csv=args.input,
            output_csv=args.output,
            trace_npz=args.traces,
            read_chunksize=args.read_chunksize,
            overwrite=args.overwrite,
        ))
    elif args.command == "evidence":
        print(build_population_template_evidence(
            root=args.root,
            feature_csv=args.features,
            output_csv=args.output,
            collapsed_regime_csv=args.collapsed_regimes,
            regime_equivalence_csv=args.regime_equivalence,
            jnd_k=args.jnd_k,
            overwrite=args.overwrite,
        ))
    else:
        features, evidence = run_population_feature_pipeline(
            root=args.root,
            input_csv=args.input,
            feature_csv=args.features,
            evidence_csv=args.evidence,
            collapsed_regime_csv=args.collapsed_regimes,
            regime_equivalence_csv=args.regime_equivalence,
            trace_npz=args.traces,
            read_chunksize=args.read_chunksize,
            jnd_k=args.jnd_k,
            overwrite=args.overwrite,
        )
        print(features)
        print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
