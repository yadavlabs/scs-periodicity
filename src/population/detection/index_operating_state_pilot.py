"""High-resolution index-axon operating-state diagnostic.

This pilot is deliberately isolated from production threshold and population
response files. It reruns only the CV0 index axon for the Slack repetition and
one representative regular repetition at 20, 50, 100, and 200 Hz. The search
uses a finer tolerance and adds a dense local amplitude sweep around every old
and refined transition so that numerical resolution, skipped spike-count
targets, and nonmonotonic responses can be distinguished before a full rerun.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from population.detection.detection_core import (
    INDEX_REGIMES_CSV,
    PREPAD_MS,
    PROJECT_ROOT,
    PopulationMRGRunner,
    THRESH_TYPES,
    _atomic_dataframe_csv,
    _compute_regimes,
    _config_path,
    _repetition_equivalence_groups,
    identify_index_axon,
    load_population_rep,
    prepend_zeros_ms,
    trim_tail_window_ms,
)
from population.space_time_v.spacetime_core import spacetime_path


PILOT_VERSION = "index_operating_state_resolution_v1"
PILOT_FREQUENCIES = (20.0, 50.0, 100.0, 200.0)
PILOT_REPRESENTATIVES = (0, 1)
DEFAULT_TOL_UA = 0.01
DEFAULT_SWEEP_STEP_UA = 0.01
DEFAULT_SWEEP_HALF_WIDTH_UA = 0.20


def pilot_conditions() -> list[tuple[float, int]]:
    """Return stable (frequency, representative repetition) pilot tasks."""
    return [
        (frequency, rep_index)
        for frequency in PILOT_FREQUENCIES
        for rep_index in PILOT_REPRESENTATIVES
    ]


def target_spike_counts(frequency_hz: float) -> dict[str, int | None]:
    """Return the nominal spike-count target behind each regime label."""
    n_input_pulses = int(round(float(frequency_hz) * 2.0))
    return {
        "I_first": 1,
        "I_volley5": 5,
        "I_volley10": 10,
        "I_5": max(1, int(np.ceil(0.05 * n_input_pulses))),
        "I_10": max(1, int(np.ceil(0.10 * n_input_pulses))),
        "I_15": max(1, int(np.ceil(0.15 * n_input_pulses))),
        "I_25": max(1, int(np.ceil(0.25 * n_input_pulses))),
        "I_50": max(1, int(np.ceil(0.50 * n_input_pulses))),
        "I_all": None,
    }


def target_definition_collisions() -> pd.DataFrame:
    """List labels that are mathematically identical at each frequency."""
    rows: list[dict[str, Any]] = []
    for frequency in PILOT_FREQUENCIES:
        by_target: dict[int, list[str]] = {}
        for regime, target in target_spike_counts(frequency).items():
            if target is not None:
                by_target.setdefault(int(target), []).append(regime)
        for target, regimes in sorted(by_target.items()):
            if len(regimes) > 1:
                rows.append({
                    "freq": frequency,
                    "target_spikes": target,
                    "n_labels": len(regimes),
                    "regimes": "|".join(regimes),
                })
    return pd.DataFrame(rows, columns=["freq", "target_spikes", "n_labels", "regimes"])


def _old_threshold_rows(root: Path, frequency: float, rep_index: int) -> pd.DataFrame:
    path = root / "data" / "detection" / "population" / INDEX_REGIMES_CSV.name
    if not path.is_file():
        raise FileNotFoundError(f"Existing index regime table is required for the pilot: {path}")
    columns = [
        "cv", "freq", "rep_index", "rep_label", "thresh_type", "amp_uA",
        "n_index_spikes", "N_max_spikes",
    ]
    table = pd.read_csv(path, usecols=columns)
    selected = table[
        np.isclose(pd.to_numeric(table["cv"], errors="coerce"), 0.0)
        & np.isclose(pd.to_numeric(table["freq"], errors="coerce"), float(frequency))
        & (pd.to_numeric(table["rep_index"], errors="coerce") == int(rep_index))
    ].copy()
    if selected.empty:
        raise RuntimeError(
            f"No old CV0 thresholds for frequency={frequency:g}, rep={rep_index}"
        )
    return selected.sort_values("thresh_type")


def _represented_repetitions(npz_path: Path, representative: int) -> tuple[int, ...]:
    # The pilot intentionally compares the archived Slack waveform with the
    # generated regular waveform, so it bypasses the selected production CV0
    # canonicalization.
    for group in _repetition_equivalence_groups(npz_path, canonicalize_cv0=False):
        indices = tuple(int(rep_index) for rep_index, _ in group)
        if int(representative) in indices:
            return indices
    raise RuntimeError(f"Repetition {representative} is absent from {npz_path.name}")


def _local_sweep_amplitudes(
    centers: Sequence[float], *, step_uA: float, half_width_uA: float,
) -> np.ndarray:
    if not np.isfinite(step_uA) or step_uA <= 0:
        raise ValueError("sweep step must be finite and positive")
    if not np.isfinite(half_width_uA) or half_width_uA < 0:
        raise ValueError("sweep half-width must be finite and non-negative")
    n_steps = int(np.ceil(float(half_width_uA) / float(step_uA)))
    offsets = np.arange(-n_steps, n_steps + 1, dtype=float) * float(step_uA)
    samples = {
        round(float(center) + float(offset), 9)
        for center in centers
        if np.isfinite(center)
        for offset in offsets
        if float(center) + float(offset) >= 0
    }
    return np.asarray(sorted(samples), dtype=float)


def _near_any(value: float, candidates: Sequence[float], atol: float = 5e-8) -> bool:
    return any(np.isfinite(item) and np.isclose(value, item, atol=atol, rtol=0.0) for item in candidates)


def _plateau_around_threshold(
    local_samples: pd.DataFrame, amplitude_uA: float, spike_count: int, step_uA: float,
) -> tuple[float, float, float, bool, bool]:
    """Estimate the sampled constant-count plateau containing a threshold."""
    if local_samples.empty or not np.isfinite(amplitude_uA):
        return np.nan, np.nan, np.nan, False, False
    ordered = local_samples.sort_values("amp_uA").reset_index(drop=True)
    amplitudes = ordered["amp_uA"].to_numpy(float)
    counts = ordered["n_index_spikes"].to_numpy(int)
    eligible = np.flatnonzero(counts == int(spike_count))
    if eligible.size == 0:
        return np.nan, np.nan, np.nan, False, False
    center = int(eligible[np.argmin(np.abs(amplitudes[eligible] - float(amplitude_uA)))])
    max_gap = float(step_uA) * 1.05 + 1e-12
    left = center
    while left > 0 and counts[left - 1] == counts[center] and amplitudes[left] - amplitudes[left - 1] <= max_gap:
        left -= 1
    right = center
    while right + 1 < len(ordered) and counts[right + 1] == counts[center] and amplitudes[right + 1] - amplitudes[right] <= max_gap:
        right += 1
    return (
        float(amplitudes[left]),
        float(amplitudes[right]),
        float(amplitudes[right] - amplitudes[left]),
        bool(left == 0),
        bool(right == len(ordered) - 1),
    )


def analyze_local_samples(samples: pd.DataFrame, *, step_uA: float) -> dict[str, float | int]:
    """Summarize adjacent fine-grid response decreases for a pilot condition."""
    local = samples[samples["is_local_sweep"].astype(bool)].sort_values("amp_uA")
    if len(local) < 2:
        return {"n_local_samples": len(local), "n_nonmonotonic_steps": 0, "max_spike_drop": 0}
    amplitudes = local["amp_uA"].to_numpy(float)
    counts = local["n_index_spikes"].to_numpy(int)
    gaps = np.diff(amplitudes)
    changes = np.diff(counts)
    adjacent = gaps <= float(step_uA) * 1.05 + 1e-12
    drops = -changes[adjacent & (changes < 0)]
    return {
        "n_local_samples": int(len(local)),
        "n_nonmonotonic_steps": int(drops.size),
        "max_spike_drop": int(np.max(drops)) if drops.size else 0,
    }


def nonmonotonic_step_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Return every adjacent fine-grid decrease in index spike count."""
    columns = [
        "index_axon_id", "freq", "representative_rep_index", "rep_label",
        "amp_low_uA", "amp_high_uA", "n_spikes_low", "n_spikes_high",
        "spike_drop", "amplitude_step_uA",
    ]
    rows: list[dict[str, Any]] = []
    group_columns = ["index_axon_id", "freq", "representative_rep_index", "rep_label"]
    local = samples[samples["is_local_sweep"].astype(bool)].copy()
    for group_key, block in local.groupby(group_columns, sort=True, dropna=False):
        block = block.sort_values("amp_uA")
        if len(block) < 2:
            continue
        amplitudes = block["amp_uA"].to_numpy(float)
        counts = block["n_index_spikes"].to_numpy(int)
        step = float(block["sweep_step_uA"].iloc[0])
        for index in range(len(block) - 1):
            gap = float(amplitudes[index + 1] - amplitudes[index])
            if gap > 1.05 * step + 1e-12 or counts[index + 1] >= counts[index]:
                continue
            row = dict(zip(group_columns, group_key))
            row.update({
                "amp_low_uA": float(amplitudes[index]),
                "amp_high_uA": float(amplitudes[index + 1]),
                "n_spikes_low": int(counts[index]),
                "n_spikes_high": int(counts[index + 1]),
                "spike_drop": int(counts[index] - counts[index + 1]),
                "amplitude_step_uA": gap,
            })
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _partial_paths(partial_dir: Path, frequency: float, rep_index: int) -> tuple[Path, Path, Path]:
    stem = f"f{int(frequency)}_rep{int(rep_index)}"
    return (
        partial_dir / f"search_samples_{stem}.csv",
        partial_dir / f"threshold_comparison_{stem}.csv",
        partial_dir / f"condition_summary_{stem}.csv",
    )


def _run_local_sweep_chunk(
    root: Path,
    index: dict[str, Any],
    frequency: float,
    rep_index: int,
    amplitudes: Sequence[float],
) -> dict[float, tuple[np.ndarray, int]]:
    """Run one amplitude chunk in a fresh process with one reusable runner."""
    npz_path = spacetime_path(
        root / "data" / "space-time-V" / "population",
        str(index["cutline_id"]), float(index["diameter_um"]), 0.0, frequency,
        axon_id=str(index["axon_id"]),
    )
    rep = load_population_rep(npz_path, rep_index, canonicalize_cv0=False)
    config = _config_path(root, rep["diameter_um"], rep["n_valid_nodes"])
    time_trim, voltage_trim = trim_tail_window_ms(
        rep["time_ms"], rep["V_base"], rep["current"],
    )
    time_trim, voltage_trim = prepend_zeros_ms(
        time_trim, voltage_trim, prepad_ms=PREPAD_MS,
    )
    runner = PopulationMRGRunner(
        time_trim, voltage_trim, config_path=config,
        readout_node_number=rep["readout_node_number"],
    )
    return {
        float(amplitude): runner.run_mrg_for_amp(float(amplitude))
        for amplitude in amplitudes
    }


def run_pilot_condition(
    *,
    root: Path = PROJECT_ROOT,
    condition_index: int,
    partial_dir: Path | None = None,
    tol_uA: float = DEFAULT_TOL_UA,
    sweep_step_uA: float = DEFAULT_SWEEP_STEP_UA,
    sweep_half_width_uA: float = DEFAULT_SWEEP_HALF_WIDTH_UA,
    n_jobs: int = 1,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run one frequency/repetition pilot condition and write atomic CSVs."""
    root = Path(root).resolve()
    conditions = pilot_conditions()
    if not 0 <= int(condition_index) < len(conditions):
        raise IndexError(f"condition_index must be in [0, {len(conditions) - 1}]")
    frequency, rep_index = conditions[int(condition_index)]
    partial_dir = Path(partial_dir or root / "results" / "population" / "detection" / "index_operating_state_pilot" / "partials").resolve()
    sample_path, threshold_path, summary_path = _partial_paths(partial_dir, frequency, rep_index)
    if all(path.is_file() for path in (sample_path, threshold_path, summary_path)) and not overwrite:
        return {"samples": sample_path, "thresholds": threshold_path, "summary": summary_path}

    index = identify_index_axon(root=root, n_jobs=1, force=False)
    npz_path = spacetime_path(
        root / "data" / "space-time-V" / "population",
        str(index["cutline_id"]), float(index["diameter_um"]), 0.0, frequency,
        axon_id=str(index["axon_id"]),
    )
    represented_reps = _represented_repetitions(npz_path, rep_index)
    rep = load_population_rep(npz_path, rep_index, canonicalize_cv0=False)
    config = _config_path(root, rep["diameter_um"], rep["n_valid_nodes"])
    old = _old_threshold_rows(root, frequency, rep_index)
    old_amplitudes = pd.to_numeric(old["amp_uA"], errors="coerce").dropna().tolist()
    refined, cache = _compute_regimes(
        rep["time_ms"], rep["V_base"], rep["current"],
        config_path=config, readout_node=rep["readout_node_number"],
        n_input_pulses=int(round(frequency * 2.0)), tol_uA=float(tol_uA),
    )
    refined_amplitudes = [
        float(refined[f"{regime}_uA"])
        for regime in THRESH_TYPES
        if refined.get(f"{regime}_uA") is not None
        and np.isfinite(refined.get(f"{regime}_uA"))
    ]
    local_amplitudes = _local_sweep_amplitudes(
        old_amplitudes + refined_amplitudes,
        step_uA=float(sweep_step_uA), half_width_uA=float(sweep_half_width_uA),
    )

    missing_local = [float(amplitude) for amplitude in local_amplitudes if float(amplitude) not in cache]
    worker_count = min(max(1, int(n_jobs)), max(1, len(missing_local)))
    chunks = [chunk.tolist() for chunk in np.array_split(missing_local, worker_count) if len(chunk)]
    if chunks:
        chunk_results = Parallel(n_jobs=worker_count, backend="loky")(
            delayed(_run_local_sweep_chunk)(root, index, frequency, rep_index, chunk)
            for chunk in chunks
        )
        for chunk_result in chunk_results:
            cache.update(chunk_result)

    sample_rows: list[dict[str, Any]] = []
    for amplitude, (spike_times, n_spikes) in sorted(cache.items()):
        roles = []
        is_local = _near_any(amplitude, local_amplitudes)
        if is_local:
            roles.append("local_sweep")
        if _near_any(amplitude, old_amplitudes):
            roles.append("old_threshold")
        if _near_any(amplitude, refined_amplitudes):
            roles.append("refined_threshold")
        if not roles:
            roles.append("adaptive_search")
        sample_rows.append({
            "pilot_version": PILOT_VERSION,
            "index_axon_id": rep["axon_id"],
            "freq": frequency,
            "cv": 0.0,
            "representative_rep_index": rep_index,
            "rep_label": rep["rep_label"],
            "represented_rep_indices": json.dumps(represented_reps),
            "n_represented_reps": len(represented_reps),
            "amp_uA": float(amplitude),
            "n_index_spikes": int(n_spikes),
            "index_spike_times_ms": json.dumps(np.asarray(spike_times, dtype=float).tolist()),
            "sample_role": "|".join(roles),
            "is_local_sweep": is_local,
            "search_tolerance_uA": float(tol_uA),
            "sweep_step_uA": float(sweep_step_uA),
            "sweep_half_width_uA": float(sweep_half_width_uA),
        })
    samples = pd.DataFrame(sample_rows)
    local = samples[samples["is_local_sweep"]].copy()

    targets = target_spike_counts(frequency)
    old_by_regime = old.set_index("thresh_type")
    threshold_rows: list[dict[str, Any]] = []
    for regime in THRESH_TYPES:
        refined_amp = refined.get(f"{regime}_uA")
        if refined_amp is None or not np.isfinite(refined_amp):
            continue
        refined_amp = float(refined_amp)
        refined_spikes = int(cache[refined_amp][1])
        target = refined["N_max_spikes"] if regime == "I_all" else targets[regime]
        identical_labels = [
            label for label, label_target in targets.items()
            if target is not None and label_target == target
        ]
        old_amp = float(old_by_regime.loc[regime, "amp_uA"]) if regime in old_by_regime.index else np.nan
        old_spikes = int(old_by_regime.loc[regime, "n_index_spikes"]) if regime in old_by_regime.index else 0
        plateau = _plateau_around_threshold(
            local, refined_amp, refined_spikes, float(sweep_step_uA),
        )
        target_observed = bool((local["n_index_spikes"].astype(int) == int(target)).any()) if target is not None else False
        threshold_rows.append({
            "pilot_version": PILOT_VERSION,
            "index_axon_id": rep["axon_id"],
            "freq": frequency,
            "cv": 0.0,
            "representative_rep_index": rep_index,
            "rep_label": rep["rep_label"],
            "represented_rep_indices": json.dumps(represented_reps),
            "n_represented_reps": len(represented_reps),
            "thresh_type": regime,
            "target_spikes": int(target) if target is not None else np.nan,
            "definition_equivalent_regimes": "|".join(identical_labels),
            "definition_duplicate": len(identical_labels) > 1,
            "old_amp_uA": old_amp,
            "old_n_index_spikes": old_spikes,
            "refined_amp_uA": refined_amp,
            "refined_n_index_spikes": refined_spikes,
            "amp_shift_uA": refined_amp - old_amp,
            "target_exactly_observed_in_local_sweep": target_observed,
            "target_skipped_at_refined_crossing": bool(target is not None and refined_spikes > int(target)),
            "plateau_min_sampled_uA": plateau[0],
            "plateau_max_sampled_uA": plateau[1],
            "plateau_sampled_width_uA": plateau[2],
            "plateau_left_censored": plateau[3],
            "plateau_right_censored": plateau[4],
            "search_tolerance_uA": float(tol_uA),
        })
    thresholds = pd.DataFrame(threshold_rows)
    local_summary = analyze_local_samples(samples, step_uA=float(sweep_step_uA))
    summary = pd.DataFrame([{
        "pilot_version": PILOT_VERSION,
        "index_axon_id": rep["axon_id"],
        "freq": frequency,
        "cv": 0.0,
        "representative_rep_index": rep_index,
        "rep_label": rep["rep_label"],
        "represented_rep_indices": json.dumps(represented_reps),
        "n_represented_reps": len(represented_reps),
        "n_cached_samples": len(samples),
        "n_refined_regimes": len(thresholds),
        "n_definition_duplicate_labels": int(thresholds["definition_duplicate"].sum()),
        "n_skipped_target_crossings": int(thresholds["target_skipped_at_refined_crossing"].sum()),
        "n_unique_refined_amplitudes_at_tolerance": int(
            np.unique(np.round(thresholds["refined_amp_uA"].to_numpy(float) / float(tol_uA))).size
        ),
        **local_summary,
        "search_tolerance_uA": float(tol_uA),
        "sweep_step_uA": float(sweep_step_uA),
        "sweep_half_width_uA": float(sweep_half_width_uA),
        "sweep_n_jobs": int(worker_count),
    }])
    _atomic_dataframe_csv(samples, sample_path)
    _atomic_dataframe_csv(thresholds, threshold_path)
    _atomic_dataframe_csv(summary, summary_path)
    return {"samples": sample_path, "thresholds": threshold_path, "summary": summary_path}


def _threshold_column(regime: str) -> str:
    if regime == "I_first":
        return "I_first0_uA"
    if regime == "I_all":
        return "I_all0_uA"
    return f"{regime}_0_uA"


def build_cv0_threshold_proposal(thresholds: pd.DataFrame) -> pd.DataFrame:
    """Build a CV0-only max-across-representative-repetitions proposal."""
    required = {"freq", "thresh_type", "refined_amp_uA", "index_axon_id"}
    missing = sorted(required - set(thresholds.columns))
    if missing:
        raise ValueError(f"Pilot thresholds are missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for frequency, block in thresholds.groupby("freq", sort=True):
        row: dict[str, Any] = {
            "pilot_version": PILOT_VERSION,
            "index_axon_id": str(block["index_axon_id"].iloc[0]),
            "freq": float(frequency),
            "amplitude_definition": "CV0 maximum across Slack and regular representative repetitions",
        }
        for regime in THRESH_TYPES:
            values = pd.to_numeric(
                block.loc[block["thresh_type"] == regime, "refined_amp_uA"], errors="coerce"
            ).dropna()
            row[_threshold_column(regime)] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def merge_pilot_outputs(
    *, root: Path = PROJECT_ROOT, partial_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Validate and merge all eight pilot conditions."""
    root = Path(root).resolve()
    partial_dir = Path(partial_dir or root / "results" / "population" / "detection" / "index_operating_state_pilot" / "partials").resolve()
    output_dir = Path(output_dir or root / "results" / "population" / "detection" / "index_operating_state_pilot").resolve()
    expected = [
        _partial_paths(partial_dir, frequency, rep_index)
        for frequency, rep_index in pilot_conditions()
    ]
    missing = [str(path) for paths in expected for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot merge pilot; {len(missing)} files are missing (first: {missing[0]})")
    samples = pd.concat([pd.read_csv(paths[0]) for paths in expected], ignore_index=True)
    thresholds = pd.concat([pd.read_csv(paths[1]) for paths in expected], ignore_index=True)
    summaries = pd.concat([pd.read_csv(paths[2]) for paths in expected], ignore_index=True)
    collisions = target_definition_collisions()
    proposal = build_cv0_threshold_proposal(thresholds)
    nonmonotonic = nonmonotonic_step_table(samples)
    paths = {
        "samples": output_dir / "index_operating_state_search_samples.csv",
        "thresholds": output_dir / "index_operating_state_threshold_comparison.csv",
        "summary": output_dir / "index_operating_state_condition_summary.csv",
        "collisions": output_dir / "index_operating_state_definition_collisions.csv",
        "nonmonotonic": output_dir / "index_operating_state_nonmonotonic_steps.csv",
        "proposal": output_dir / "index_cv0_threshold_proposal.csv",
        "manifest": output_dir / "pilot_manifest.json",
    }
    _atomic_dataframe_csv(samples, paths["samples"])
    _atomic_dataframe_csv(thresholds, paths["thresholds"])
    _atomic_dataframe_csv(summaries, paths["summary"])
    _atomic_dataframe_csv(collisions, paths["collisions"])
    _atomic_dataframe_csv(nonmonotonic, paths["nonmonotonic"])
    _atomic_dataframe_csv(proposal, paths["proposal"])
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["manifest"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "pilot_version": PILOT_VERSION,
        "status": "complete",
        "frequencies_hz": list(PILOT_FREQUENCIES),
        "representative_repetitions": list(PILOT_REPRESENTATIVES),
        "conditions": len(pilot_conditions()),
        "production_files_modified": False,
        "primary_amplitude_proposal": "CV0-only; no all-CV onset safeguard",
        "silence_policy": (
            "A silent index axon is retained as a zero-spike response. Population rate evidence may still be "
            "defined; unavailable population timing evidence is zero with an explicit false availability flag."
        ),
        "outputs": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }, indent=2), encoding="utf-8")
    temporary.replace(paths["manifest"])
    return paths


def preflight(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    checks = []
    for condition_index, (frequency, rep_index) in enumerate(pilot_conditions()):
        path = spacetime_path(
            root / "data" / "space-time-V" / "population",
            str(index["cutline_id"]), float(index["diameter_um"]), 0.0, frequency,
            axon_id=str(index["axon_id"]),
        )
        checks.append({
            "condition_index": condition_index,
            "freq": frequency,
            "representative_rep_index": rep_index,
            "input": str(path),
            "input_exists": path.is_file(),
        })
    return {
        "status": "ready" if all(item["input_exists"] for item in checks) else "missing_input",
        "pilot_version": PILOT_VERSION,
        "index_axon_id": str(index["axon_id"]),
        "conditions": checks,
        "production_files_modified": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--root", type=Path, default=PROJECT_ROOT)
    run = sub.add_parser("run-condition")
    run.add_argument("--root", type=Path, default=PROJECT_ROOT)
    run.add_argument("--condition-index", type=int, required=True)
    run.add_argument("--partial-dir", type=Path, default=None)
    run.add_argument("--tol-uA", type=float, default=DEFAULT_TOL_UA)
    run.add_argument("--sweep-step-uA", type=float, default=DEFAULT_SWEEP_STEP_UA)
    run.add_argument("--sweep-half-width-uA", type=float, default=DEFAULT_SWEEP_HALF_WIDTH_UA)
    run.add_argument("--n-jobs", type=int, default=1)
    run.add_argument("--overwrite", action="store_true")
    merge = sub.add_parser("merge")
    merge.add_argument("--root", type=Path, default=PROJECT_ROOT)
    merge.add_argument("--partial-dir", type=Path, default=None)
    merge.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.root)
        print(json.dumps(result, indent=2))
        if result["status"] != "ready":
            return 2
    elif args.command == "run-condition":
        result = run_pilot_condition(
            root=args.root, condition_index=args.condition_index,
            partial_dir=args.partial_dir, tol_uA=args.tol_uA,
            sweep_step_uA=args.sweep_step_uA,
            sweep_half_width_uA=args.sweep_half_width_uA,
            n_jobs=args.n_jobs,
            overwrite=args.overwrite,
        )
        print("\n".join(f"{name}: {path}" for name, path in result.items()))
    else:
        result = merge_pilot_outputs(
            root=args.root, partial_dir=args.partial_dir, output_dir=args.output_dir,
        )
        print("\n".join(f"{name}: {path}" for name, path in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
