"""Targeted Longleaf diagnostic for the high-amplitude ``I_all`` branch.

The production definition chooses the lowest amplitude that reaches the largest
raw index-axon spike count observed through 2000 uA.  A one-spike increase late
in that domain can therefore move ``I_all`` from the first saturation plateau
to a second, much higher-amplitude branch.  This module does not change any
production threshold.  It selects matched ordinary/high-branch repetitions,
runs an independent coarse sweep plus fine transition sweeps, and saves voltage
traces at the amplitudes needed to decide whether the late transition is
biophysical, nonmonotonic, or a spike-counting/search artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from population.detection.detection_core import (
    INDEX_REGIMES_CSV,
    PREPAD_MS,
    PROJECT_ROOT,
    PopulationMRGRunner,
    _atomic_dataframe_csv,
    _config_path,
    identify_index_axon,
    load_population_rep,
    prepend_zeros_ms,
    trim_tail_window_ms,
)
from population.space_time_v.spacetime_core import spacetime_path


DIAGNOSTIC_VERSION = "imax_plateau_branch_v1"
DEFAULT_FREQUENCIES = (20.0, 50.0, 100.0)
DEFAULT_CV = 1.0
DEFAULT_BRANCH_CUTOFF_UA = 800.0
DEFAULT_REPETITIONS_PER_BRANCH = 2
DEFAULT_COARSE_MIN_UA = 200.0
DEFAULT_COARSE_MAX_UA = 2000.0
DEFAULT_COARSE_STEP_UA = 25.0
DEFAULT_FINE_STEP_UA = 0.5
DEFAULT_FINE_HALF_WIDTH_UA = 25.0
DEFAULT_LATE_TRANSITION_MIN_UA = 500.0
DEFAULT_MAX_LATE_TRANSITIONS = 4
DEFAULT_PLATEAU_MIN_WIDTH_UA = 100.0
DEFAULT_PLATEAU_MIN_FRACTION_OF_MAX = 0.90


def _spread_rows(frame: pd.DataFrame, count: int, column: str) -> pd.DataFrame:
    """Select deterministic, value-spanning rows without replacement."""
    ordered = frame.sort_values([column, "rep_index"], kind="stable").reset_index(drop=True)
    count = min(max(0, int(count)), len(ordered))
    if count == 0:
        return ordered.iloc[0:0].copy()
    indices = np.unique(np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int))
    if len(indices) < count:
        remaining = [index for index in range(len(ordered)) if index not in set(indices)]
        indices = np.r_[indices, remaining[: count - len(indices)]]
    return ordered.iloc[np.sort(indices[:count])].copy()


def select_conditions(
    table: pd.DataFrame,
    *,
    frequencies: Sequence[float] = DEFAULT_FREQUENCIES,
    cv: float = DEFAULT_CV,
    branch_cutoff_uA: float = DEFAULT_BRANCH_CUTOFF_UA,
    repetitions_per_branch: int = DEFAULT_REPETITIONS_PER_BRANCH,
) -> pd.DataFrame:
    """Select amplitude-spanning high cases and target-matched ordinary cases."""
    required = {
        "freq", "cv", "rep_index", "rep_label", "thresh_type", "amp_uA",
        "N_max_spikes", "n_index_spikes", "index_axon_id", "index_cutline_id",
        "index_diameter_um", "file", "n_input_pulses",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Index-regime table is missing columns: {missing}")
    source = table[
        table["thresh_type"].eq("I_all")
        & np.isclose(pd.to_numeric(table["cv"], errors="coerce"), float(cv))
        & pd.to_numeric(table["freq"], errors="coerce").isin([float(item) for item in frequencies])
    ].copy()
    source["amp_uA"] = pd.to_numeric(source["amp_uA"], errors="coerce")
    source["N_max_spikes"] = pd.to_numeric(source["N_max_spikes"], errors="coerce")
    source["rep_index"] = pd.to_numeric(source["rep_index"], errors="raise").astype(int)

    selected: list[pd.DataFrame] = []
    for frequency in frequencies:
        panel = source[np.isclose(source["freq"].astype(float), float(frequency))].copy()
        high = panel[panel["amp_uA"].gt(float(branch_cutoff_uA))]
        ordinary = panel[panel["amp_uA"].le(float(branch_cutoff_uA))]
        if high.empty or ordinary.empty:
            raise RuntimeError(
                f"Need both I_all branches at {float(frequency):g} Hz, CV={float(cv):g}; "
                f"found ordinary={len(ordinary)}, high={len(high)}"
            )

        high_selected = _spread_rows(high, repetitions_per_branch, "amp_uA")
        high_selected["branch"] = "high"
        high_selected["selection_reason"] = "high-branch amplitude span"
        selected.append(high_selected)

        # Match ordinary controls to the selected high cases by target spike
        # count, with proximity to the ordinary median amplitude as a tie-break.
        remaining = ordinary.copy()
        ordinary_median = float(ordinary["amp_uA"].median())
        ordinary_rows = []
        targets = high_selected["N_max_spikes"].astype(float).tolist()
        while len(ordinary_rows) < int(repetitions_per_branch) and not remaining.empty:
            target = targets[len(ordinary_rows) % len(targets)]
            ranked = remaining.assign(
                target_distance=(remaining["N_max_spikes"] - target).abs(),
                amplitude_distance=(remaining["amp_uA"] - ordinary_median).abs(),
            ).sort_values(
                ["target_distance", "amplitude_distance", "rep_index"], kind="stable"
            )
            chosen = ranked.iloc[0]
            ordinary_rows.append(chosen[ordinary.columns].to_dict())
            remaining = remaining[remaining["rep_index"].ne(int(chosen["rep_index"]))]
        ordinary_selected = pd.DataFrame(ordinary_rows)
        ordinary_selected["branch"] = "ordinary"
        ordinary_selected["selection_reason"] = "N_max-matched ordinary control"
        selected.append(ordinary_selected)

    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["freq", "branch", "amp_uA", "rep_index"], kind="stable").reset_index(drop=True)
    result.insert(0, "condition_index", np.arange(len(result), dtype=int))
    result.insert(0, "diagnostic_version", DIAGNOSTIC_VERSION)
    result = result.rename(columns={
        "amp_uA": "production_Imax_uA",
        "n_index_spikes": "production_Imax_spikes",
        "N_max_spikes": "production_target_spikes",
    })
    return result


def coarse_amplitudes(minimum_uA: float, maximum_uA: float, step_uA: float) -> np.ndarray:
    if not np.isfinite(step_uA) or step_uA <= 0:
        raise ValueError("Coarse step must be positive")
    if maximum_uA <= minimum_uA:
        raise ValueError("Coarse maximum must exceed minimum")
    values = np.arange(float(minimum_uA), float(maximum_uA) + 0.5 * float(step_uA), float(step_uA))
    return np.unique(np.round(np.r_[values, float(minimum_uA), float(maximum_uA)], 9))


def plateau_segments(samples: pd.DataFrame, *, coarse_step_uA: float) -> pd.DataFrame:
    """Return constant-count runs from the independent coarse grid."""
    coarse = samples[samples["sample_role"].astype(str).str.contains("coarse")].copy()
    coarse = coarse.sort_values("amp_uA").drop_duplicates("amp_uA").reset_index(drop=True)
    columns = [
        "segment_index", "start_uA", "end_uA", "width_uA", "n_samples",
        "n_index_spikes", "fraction_of_sampled_max",
    ]
    if coarse.empty:
        return pd.DataFrame(columns=columns)
    amplitudes = coarse["amp_uA"].to_numpy(float)
    counts = coarse["n_index_spikes"].to_numpy(int)
    sampled_max = max(1, int(counts.max()))
    rows = []
    start = 0
    for index in range(1, len(coarse) + 1):
        contiguous = index < len(coarse) and amplitudes[index] - amplitudes[index - 1] <= 1.05 * float(coarse_step_uA)
        same_count = index < len(coarse) and counts[index] == counts[index - 1]
        if contiguous and same_count:
            continue
        rows.append({
            "segment_index": len(rows),
            "start_uA": float(amplitudes[start]),
            "end_uA": float(amplitudes[index - 1]),
            "width_uA": float(amplitudes[index - 1] - amplitudes[start]),
            "n_samples": int(index - start),
            "n_index_spikes": int(counts[start]),
            "fraction_of_sampled_max": float(counts[start] / sampled_max),
        })
        start = index
    return pd.DataFrame(rows, columns=columns)


def first_stable_plateau(
    segments: pd.DataFrame,
    *,
    min_width_uA: float = DEFAULT_PLATEAU_MIN_WIDTH_UA,
    min_fraction_of_max: float = DEFAULT_PLATEAU_MIN_FRACTION_OF_MAX,
) -> dict[str, float | int] | None:
    eligible = segments[
        segments["width_uA"].ge(float(min_width_uA))
        & segments["fraction_of_sampled_max"].ge(float(min_fraction_of_max))
        & segments["n_index_spikes"].gt(0)
    ].sort_values("start_uA")
    return None if eligible.empty else eligible.iloc[0].to_dict()


def transition_brackets(samples: pd.DataFrame, *, late_min_uA: float) -> pd.DataFrame:
    """Return adjacent coarse intervals whose index-axon spike count changes."""
    coarse = samples[samples["sample_role"].astype(str).str.contains("coarse")].copy()
    coarse = coarse.sort_values("amp_uA").drop_duplicates("amp_uA").reset_index(drop=True)
    rows = []
    for index in range(len(coarse) - 1):
        low = coarse.iloc[index]
        high = coarse.iloc[index + 1]
        if int(low["n_index_spikes"]) == int(high["n_index_spikes"]):
            continue
        rows.append({
            "amp_low_uA": float(low["amp_uA"]),
            "amp_high_uA": float(high["amp_uA"]),
            "n_spikes_low": int(low["n_index_spikes"]),
            "n_spikes_high": int(high["n_index_spikes"]),
            "spike_change": int(high["n_index_spikes"] - low["n_index_spikes"]),
            "is_late_transition": bool(float(high["amp_uA"]) >= float(late_min_uA)),
        })
    return pd.DataFrame(rows)


def fine_transition_steps(samples: pd.DataFrame, *, fine_step_uA: float,
                          late_min_uA: float) -> pd.DataFrame:
    """Return adjacent fine-grid response changes without bridging sweep gaps."""
    fine = samples[samples["sample_role"].astype(str).str.contains("fine")].copy()
    fine = fine.sort_values("amp_uA").drop_duplicates("amp_uA").reset_index(drop=True)
    rows = []
    for index in range(len(fine) - 1):
        low = fine.iloc[index]
        high = fine.iloc[index + 1]
        gap = float(high["amp_uA"] - low["amp_uA"])
        change = int(high["n_index_spikes"] - low["n_index_spikes"])
        if gap > 1.05 * float(fine_step_uA) or change == 0:
            continue
        rows.append({
            "amp_low_uA": float(low["amp_uA"]),
            "amp_high_uA": float(high["amp_uA"]),
            "n_spikes_low": int(low["n_index_spikes"]),
            "n_spikes_high": int(high["n_index_spikes"]),
            "spike_change": change,
            "is_late_transition": bool(float(high["amp_uA"]) >= float(late_min_uA)),
        })
    return pd.DataFrame(rows)


def _fine_amplitudes(centers: Iterable[float], *, step_uA: float, half_width_uA: float,
                     minimum_uA: float, maximum_uA: float) -> np.ndarray:
    offsets = np.arange(-float(half_width_uA), float(half_width_uA) + 0.5 * float(step_uA), float(step_uA))
    values = {
        round(float(center) + float(offset), 9)
        for center in centers
        for offset in offsets
        if minimum_uA <= float(center) + float(offset) <= maximum_uA
    }
    return np.asarray(sorted(values), dtype=float)


def _pulse_onsets_ms(time_ms: np.ndarray, current: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    active = current < -1e-6
    starts = np.flatnonzero(active[1:] & ~active[:-1]) + 1
    if active.size and active[0]:
        starts = np.r_[0, starts]
    if len(time_ms) > 1:
        dt_ms = float(time_ms[1] - time_ms[0])
        prepad_exact = float(np.ceil(PREPAD_MS / dt_ms) * dt_ms)
    else:
        prepad_exact = float(PREPAD_MS)
    return np.asarray(time_ms, dtype=float)[starts] + prepad_exact


def pulse_response_summary(spike_times_ms: Sequence[float], pulse_onsets_ms: Sequence[float]) -> dict[str, int | float]:
    spikes = np.asarray(spike_times_ms, dtype=float)
    onsets = np.asarray(pulse_onsets_ms, dtype=float)
    if onsets.size == 0:
        return {
            "n_pulses_with_response": 0, "n_pulses_without_response": 0,
            "n_pulses_with_multiple_spikes": 0, "max_spikes_per_pulse": 0,
            "n_unassigned_spikes": int(spikes.size), "min_interspike_interval_ms": np.nan,
        }
    assignments = np.searchsorted(onsets, spikes, side="right") - 1
    valid = assignments >= 0
    counts = np.bincount(assignments[valid], minlength=len(onsets))
    return {
        "n_pulses_with_response": int(np.sum(counts > 0)),
        "n_pulses_without_response": int(np.sum(counts == 0)),
        "n_pulses_with_multiple_spikes": int(np.sum(counts > 1)),
        "max_spikes_per_pulse": int(counts.max(initial=0)),
        "n_unassigned_spikes": int(np.sum(~valid)),
        "min_interspike_interval_ms": float(np.min(np.diff(spikes))) if spikes.size > 1 else np.nan,
    }


def _condition_paths(partial_dir: Path, row: pd.Series) -> dict[str, Path]:
    stem = f"f{int(float(row['freq']))}_cv{float(row['cv']):.1f}_rep{int(row['rep_index']):03d}"
    return {
        "samples": partial_dir / f"sweep_samples_{stem}.csv",
        "plateaus": partial_dir / f"plateau_segments_{stem}.csv",
        "summary": partial_dir / f"condition_summary_{stem}.csv",
        "traces": partial_dir / f"voltage_traces_{stem}.npz",
    }


def _make_runner(root: Path, index: dict[str, Any], cv: float, frequency: float, rep_index: int):
    path = spacetime_path(
        root / "data" / "space-time-V" / "population",
        str(index["cutline_id"]), float(index["diameter_um"]), float(cv), float(frequency),
        axon_id=str(index["axon_id"]),
    )
    rep = load_population_rep(path, int(rep_index))
    config = _config_path(root, rep["diameter_um"], rep["n_valid_nodes"])
    time_trim, voltage_trim = trim_tail_window_ms(rep["time_ms"], rep["V_base"], rep["current"])
    time_trim, voltage_trim = prepend_zeros_ms(time_trim, voltage_trim, prepad_ms=PREPAD_MS)
    runner = PopulationMRGRunner(
        time_trim, voltage_trim, config_path=config,
        readout_node_number=rep["readout_node_number"],
    )
    return rep, runner


def _run_amplitude_chunk(root: Path, index: dict[str, Any], cv: float, frequency: float,
                         rep_index: int, amplitudes: Sequence[float]) -> dict[float, dict[str, Any]]:
    _, runner = _make_runner(root, index, cv, frequency, rep_index)
    results = {}
    for amplitude in amplitudes:
        spike_times, count = runner.run_mrg_for_amp(float(amplitude))
        voltage = np.asarray(runner.v_rec, dtype=float)
        results[float(amplitude)] = {
            "spike_times_ms": np.asarray(spike_times, dtype=float),
            "n_index_spikes": int(count),
            "peak_voltage_mV": float(np.max(voltage)) if voltage.size else np.nan,
            "minimum_voltage_mV": float(np.min(voltage)) if voltage.size else np.nan,
        }
    return results


def _run_parallel_amplitudes(root: Path, index: dict[str, Any], cv: float, frequency: float,
                             rep_index: int, amplitudes: Sequence[float], n_jobs: int) -> dict[float, dict[str, Any]]:
    values = [float(item) for item in amplitudes]
    workers = min(max(1, int(n_jobs)), max(1, len(values)))
    chunks = [chunk.tolist() for chunk in np.array_split(values, workers) if len(chunk)]
    outputs = Parallel(n_jobs=workers, backend="loky")(
        delayed(_run_amplitude_chunk)(root, index, cv, frequency, rep_index, chunk)
        for chunk in chunks
    )
    merged = {}
    for output in outputs:
        merged.update(output)
    return merged


def _sample_frame(cache: dict[float, dict[str, Any]], roles: dict[float, set[str]],
                  condition: pd.Series, pulse_onsets: np.ndarray) -> pd.DataFrame:
    rows = []
    for amplitude, result in sorted(cache.items()):
        pulse_summary = pulse_response_summary(result["spike_times_ms"], pulse_onsets)
        rows.append({
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "condition_index": int(condition["condition_index"]),
            "freq": float(condition["freq"]), "cv": float(condition["cv"]),
            "rep_index": int(condition["rep_index"]), "rep_label": str(condition["rep_label"]),
            "production_branch": str(condition["branch"]),
            "production_Imax_uA": float(condition["production_Imax_uA"]),
            "production_target_spikes": int(condition["production_target_spikes"]),
            "amp_uA": float(amplitude),
            "n_index_spikes": int(result["n_index_spikes"]),
            "spike_times_ms": json.dumps(result["spike_times_ms"].tolist()),
            "peak_voltage_mV": result["peak_voltage_mV"],
            "minimum_voltage_mV": result["minimum_voltage_mV"],
            "sample_role": "|".join(sorted(roles.get(float(amplitude), {"diagnostic"}))),
            **pulse_summary,
        })
    return pd.DataFrame(rows)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_condition(
    *, root: Path, condition_index: int, manifest: Path, partial_dir: Path,
    coarse_min_uA: float = DEFAULT_COARSE_MIN_UA,
    coarse_max_uA: float = DEFAULT_COARSE_MAX_UA,
    coarse_step_uA: float = DEFAULT_COARSE_STEP_UA,
    fine_step_uA: float = DEFAULT_FINE_STEP_UA,
    fine_half_width_uA: float = DEFAULT_FINE_HALF_WIDTH_UA,
    late_transition_min_uA: float = DEFAULT_LATE_TRANSITION_MIN_UA,
    max_late_transitions: int = DEFAULT_MAX_LATE_TRANSITIONS,
    plateau_min_width_uA: float = DEFAULT_PLATEAU_MIN_WIDTH_UA,
    plateau_min_fraction_of_max: float = DEFAULT_PLATEAU_MIN_FRACTION_OF_MAX,
    n_jobs: int = 1, overwrite: bool = False,
) -> dict[str, Path]:
    root, manifest, partial_dir = Path(root).resolve(), Path(manifest).resolve(), Path(partial_dir).resolve()
    conditions = pd.read_csv(manifest)
    match = conditions[conditions["condition_index"].eq(int(condition_index))]
    if len(match) != 1:
        raise IndexError(f"Condition {condition_index} is absent or duplicated in {manifest}")
    condition = match.iloc[0]
    paths = _condition_paths(partial_dir, condition)
    if all(path.is_file() for path in paths.values()) and not overwrite:
        return paths

    index = identify_index_axon(root=root, n_jobs=1, force=False)
    rep, trace_runner = _make_runner(
        root, index, float(condition["cv"]), float(condition["freq"]), int(condition["rep_index"])
    )
    pulse_onsets = _pulse_onsets_ms(rep["time_ms"], rep["current"])

    coarse = coarse_amplitudes(coarse_min_uA, coarse_max_uA, coarse_step_uA)
    coarse = np.unique(np.round(np.r_[coarse, float(condition["production_Imax_uA"])], 9))
    cache = _run_parallel_amplitudes(
        root, index, float(condition["cv"]), float(condition["freq"]),
        int(condition["rep_index"]), coarse, n_jobs,
    )
    roles: dict[float, set[str]] = {float(amplitude): {"coarse"} for amplitude in coarse}
    roles.setdefault(float(condition["production_Imax_uA"]), set()).add("production_Imax")
    preliminary = _sample_frame(cache, roles, condition, pulse_onsets)
    segments = plateau_segments(preliminary, coarse_step_uA=coarse_step_uA)
    stable = first_stable_plateau(
        segments, min_width_uA=plateau_min_width_uA,
        min_fraction_of_max=plateau_min_fraction_of_max,
    )
    transitions = transition_brackets(preliminary, late_min_uA=late_transition_min_uA)
    late = transitions[transitions.get("is_late_transition", False).astype(bool)].copy() if not transitions.empty else transitions
    if len(late) > int(max_late_transitions):
        late = _spread_rows(
            late.assign(rep_index=np.arange(len(late))), int(max_late_transitions), "amp_high_uA"
        ).drop(columns="rep_index")

    centers = [float(condition["production_Imax_uA"])]
    if stable is not None:
        centers.append(float(stable["start_uA"]))
    if not late.empty:
        centers.extend((0.5 * (late["amp_low_uA"] + late["amp_high_uA"])).tolist())
    fine = _fine_amplitudes(
        centers, step_uA=fine_step_uA, half_width_uA=fine_half_width_uA,
        minimum_uA=coarse_min_uA, maximum_uA=coarse_max_uA,
    )
    missing = [float(amplitude) for amplitude in fine if float(amplitude) not in cache]
    if missing:
        cache.update(_run_parallel_amplitudes(
            root, index, float(condition["cv"]), float(condition["freq"]),
            int(condition["rep_index"]), missing, n_jobs,
        ))
    for amplitude in fine:
        roles.setdefault(float(amplitude), set()).add("fine")
    samples = _sample_frame(cache, roles, condition, pulse_onsets)

    # Recompute segments from the independent coarse grid and estimate the
    # fine-grid threshold of the first stable plateau count.
    segments = plateau_segments(samples, coarse_step_uA=coarse_step_uA)
    stable = first_stable_plateau(
        segments, min_width_uA=plateau_min_width_uA,
        min_fraction_of_max=plateau_min_fraction_of_max,
    )
    stable_count = np.nan if stable is None else int(stable["n_index_spikes"])
    if stable is None:
        stable_threshold = np.nan
    else:
        eligible = samples[samples["n_index_spikes"].ge(int(stable_count))]
        stable_threshold = float(eligible["amp_uA"].min()) if not eligible.empty else np.nan
    transitions = transition_brackets(samples, late_min_uA=late_transition_min_uA)
    late = transitions[transitions["is_late_transition"]].copy() if not transitions.empty else transitions
    fine_changes = fine_transition_steps(
        samples, fine_step_uA=fine_step_uA, late_min_uA=late_transition_min_uA,
    )
    fine_late = fine_changes[fine_changes["is_late_transition"]].copy() if not fine_changes.empty else fine_changes

    for column, value in {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "condition_index": int(condition["condition_index"]),
        "freq": float(condition["freq"]), "cv": float(condition["cv"]),
        "rep_index": int(condition["rep_index"]), "rep_label": str(condition["rep_label"]),
        "production_branch": str(condition["branch"]),
    }.items():
        segments[column] = value

    sampled_max = int(samples["n_index_spikes"].max())
    at_max = samples.iloc[(samples["amp_uA"] - float(coarse_max_uA)).abs().argsort()[:1]].iloc[0]
    summary = pd.DataFrame([{
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "condition_index": int(condition["condition_index"]),
        "freq": float(condition["freq"]), "cv": float(condition["cv"]),
        "rep_index": int(condition["rep_index"]), "rep_label": str(condition["rep_label"]),
        "production_branch": str(condition["branch"]),
        "production_Imax_uA": float(condition["production_Imax_uA"]),
        "production_target_spikes": int(condition["production_target_spikes"]),
        "sampled_max_spikes": sampled_max,
        "spikes_at_max_amplitude": int(at_max["n_index_spikes"]),
        "first_stable_plateau_count": stable_count,
        "first_stable_plateau_threshold_uA": stable_threshold,
        "first_stable_plateau_width_uA": np.nan if stable is None else float(stable["width_uA"]),
        "late_transition_count": int(len(late)),
        "late_upward_transition_count": int((late["spike_change"] > 0).sum()) if not late.empty else 0,
        "late_downward_transition_count": int((late["spike_change"] < 0).sum()) if not late.empty else 0,
        "fine_late_upward_step_count": int((fine_late["spike_change"] > 0).sum()) if not fine_late.empty else 0,
        "fine_late_downward_step_count": int((fine_late["spike_change"] < 0).sum()) if not fine_late.empty else 0,
        "n_sweep_samples": int(len(samples)),
        "n_input_pulses": int(len(pulse_onsets)),
        "coarse_step_uA": float(coarse_step_uA),
        "fine_step_uA": float(fine_step_uA),
        "fine_half_width_uA": float(fine_half_width_uA),
        "plateau_min_width_uA": float(plateau_min_width_uA),
        "plateau_min_fraction_of_max": float(plateau_min_fraction_of_max),
    }])

    # Save complete voltage traces only at diagnostic landmarks, not at every
    # sweep point. This keeps transfer size modest while preserving the evidence
    # needed to inspect peak detection and extra-spike timing.
    trace_amplitudes = {
        float(condition["production_Imax_uA"]), float(coarse_max_uA),
    }
    if np.isfinite(stable_threshold):
        trace_amplitudes.add(float(stable_threshold))
    trace_late = late
    if len(trace_late) > int(max_late_transitions):
        trace_late = _spread_rows(
            trace_late.assign(rep_index=np.arange(len(trace_late))),
            int(max_late_transitions), "amp_high_uA",
        ).drop(columns="rep_index")
    if not trace_late.empty:
        for row in trace_late.itertuples(index=False):
            trace_amplitudes.update([float(row.amp_low_uA), float(row.amp_high_uA)])
    trace_fine = fine_late
    max_fine_trace_steps = 2 * int(max_late_transitions)
    if len(trace_fine) > max_fine_trace_steps:
        trace_fine = _spread_rows(
            trace_fine.assign(rep_index=np.arange(len(trace_fine))),
            max_fine_trace_steps, "amp_high_uA",
        ).drop(columns="rep_index")
    if not trace_fine.empty:
        for row in trace_fine.itertuples(index=False):
            trace_amplitudes.update([float(row.amp_low_uA), float(row.amp_high_uA)])
    trace_amplitudes = sorted(trace_amplitudes)
    voltage_traces, trace_counts, trace_spikes = [], [], []
    trace_time = None
    for amplitude in trace_amplitudes:
        spike_times, count = trace_runner.run_mrg_for_amp(float(amplitude))
        trace_time = np.asarray(trace_runner.t_rec, dtype=np.float32)
        voltage_traces.append(np.asarray(trace_runner.v_rec, dtype=np.float32))
        trace_counts.append(int(count))
        trace_spikes.append(json.dumps(np.asarray(spike_times, dtype=float).tolist()))
    _atomic_npz(
        paths["traces"], diagnostic_version=np.asarray(DIAGNOSTIC_VERSION),
        condition_index=np.asarray(int(condition["condition_index"])),
        freq=np.asarray(float(condition["freq"])), cv=np.asarray(float(condition["cv"])),
        rep_index=np.asarray(int(condition["rep_index"])), rep_label=np.asarray(str(condition["rep_label"])),
        production_branch=np.asarray(str(condition["branch"])),
        amplitude_uA=np.asarray(trace_amplitudes, dtype=float),
        n_index_spikes=np.asarray(trace_counts, dtype=int),
        spike_times_json=np.asarray(trace_spikes, dtype=str),
        time_ms=np.asarray(trace_time, dtype=np.float32),
        voltage_mV=np.stack(voltage_traces).astype(np.float32),
        pulse_onsets_ms=np.asarray(pulse_onsets, dtype=float),
    )
    _atomic_dataframe_csv(samples, paths["samples"])
    _atomic_dataframe_csv(segments, paths["plateaus"])
    _atomic_dataframe_csv(summary, paths["summary"])
    return paths


def prepare(
    *, root: Path = PROJECT_ROOT, output_dir: Path | None = None,
    frequencies: Sequence[float] = DEFAULT_FREQUENCIES, cv: float = DEFAULT_CV,
    branch_cutoff_uA: float = DEFAULT_BRANCH_CUTOFF_UA,
    repetitions_per_branch: int = DEFAULT_REPETITIONS_PER_BRANCH,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = Path(output_dir or root / "results/population/detection/imax_plateau_diagnostic").resolve()
    table_path = root / "data/detection/population" / INDEX_REGIMES_CSV.name
    if not table_path.is_file():
        raise FileNotFoundError(f"Production index-regime table is required: {table_path}")
    conditions = select_conditions(
        pd.read_csv(table_path, low_memory=False), frequencies=frequencies, cv=cv,
        branch_cutoff_uA=branch_cutoff_uA,
        repetitions_per_branch=repetitions_per_branch,
    )
    manifest_path = output_dir / "diagnostic_conditions.csv"
    _atomic_dataframe_csv(conditions, manifest_path)
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    checks = []
    for row in conditions.itertuples(index=False):
        input_path = spacetime_path(
            root / "data/space-time-V/population", str(index["cutline_id"]),
            float(index["diameter_um"]), float(row.cv), float(row.freq),
            axon_id=str(index["axon_id"]),
        )
        checks.append({
            "condition_index": int(row.condition_index), "freq": float(row.freq),
            "cv": float(row.cv), "rep_index": int(row.rep_index),
            "branch": str(row.branch), "input": str(input_path),
            "input_exists": input_path.is_file(),
        })
    result = {
        "status": "ready" if all(item["input_exists"] for item in checks) else "missing_input",
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "conditions": len(conditions),
        "condition_manifest": str(manifest_path),
        "production_files_modified": False,
        "checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = (output_dir / "preflight.json").with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "preflight.json")
    return result


def _plot_merged(samples: pd.DataFrame, summaries: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frequencies = sorted(samples["freq"].unique())
    groups = {
        frequency: summaries[np.isclose(summaries["freq"], frequency)].sort_values(
            ["production_branch", "production_Imax_uA"]
        )
        for frequency in frequencies
    }
    columns = max(len(group) for group in groups.values())
    fig, axes = plt.subplots(len(frequencies), columns, figsize=(4.2 * columns, 3.2 * len(frequencies)), squeeze=False)
    for row_index, frequency in enumerate(frequencies):
        group = groups[frequency]
        for column_index, condition in enumerate(group.itertuples(index=False)):
            ax = axes[row_index, column_index]
            panel = samples[samples["condition_index"].eq(int(condition.condition_index))].sort_values("amp_uA")
            coarse = panel[panel["sample_role"].str.contains("coarse")]
            fine = panel[panel["sample_role"].str.contains("fine")]
            ax.plot(coarse["amp_uA"], coarse["n_index_spikes"], color="0.35", lw=1.2, marker="o", ms=2.8)
            ax.scatter(fine["amp_uA"], fine["n_index_spikes"], color="#2166ac", s=8, alpha=0.55)
            ax.axvline(condition.production_Imax_uA, color="#b2182b", ls="--", lw=1.5)
            if np.isfinite(condition.first_stable_plateau_threshold_uA):
                ax.axvline(condition.first_stable_plateau_threshold_uA, color="#1b9e77", ls=":", lw=1.8)
            ax.set_title(f"{condition.production_branch} | rep {int(condition.rep_index)}")
            ax.set_xlabel("Amplitude (uA)")
            ax.set_ylabel("Index-axon spikes")
            ax.grid(alpha=0.2)
        for column_index in range(len(group), columns):
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].text(-0.27, 0.5, f"{int(frequency)} Hz", transform=axes[row_index, 0].transAxes,
                                rotation=90, va="center", ha="center", fontweight="bold")
    handles = [
        Line2D([0], [0], color="0.35", marker="o", ms=3, label="Independent coarse sweep"),
        Line2D([0], [0], color="#2166ac", marker="o", lw=0, ms=4, label="Fine transition samples"),
        Line2D([0], [0], color="#b2182b", ls="--", label="Production I_max"),
        Line2D([0], [0], color="#1b9e77", ls=":", label="First stable-plateau candidate"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center")
    fig.suptitle("Index-axon I_max amplitude-response diagnostic", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.97))
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"imax_amplitude_response_sweeps.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def merge(*, root: Path = PROJECT_ROOT, manifest: Path, partial_dir: Path,
          output_dir: Path) -> dict[str, Path]:
    root, manifest = Path(root).resolve(), Path(manifest).resolve()
    partial_dir, output_dir = Path(partial_dir).resolve(), Path(output_dir).resolve()
    conditions = pd.read_csv(manifest)
    expected = [_condition_paths(partial_dir, row) for _, row in conditions.iterrows()]
    missing = [str(path) for paths in expected for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot merge diagnostic; {len(missing)} outputs are missing (first: {missing[0]})")
    samples = pd.concat([pd.read_csv(paths["samples"]) for paths in expected], ignore_index=True)
    plateaus = pd.concat([pd.read_csv(paths["plateaus"]) for paths in expected], ignore_index=True)
    summaries = pd.concat([pd.read_csv(paths["summary"]) for paths in expected], ignore_index=True)
    branch_summary = (
        summaries.groupby(["freq", "production_branch"], observed=True)
        .agg(
            repetitions=("rep_index", "size"),
            median_production_Imax_uA=("production_Imax_uA", "median"),
            median_stable_plateau_threshold_uA=("first_stable_plateau_threshold_uA", "median"),
            median_production_target_spikes=("production_target_spikes", "median"),
            median_stable_plateau_spikes=("first_stable_plateau_count", "median"),
            late_upward_transitions=("late_upward_transition_count", "sum"),
            late_downward_transitions=("late_downward_transition_count", "sum"),
        ).reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "conditions": output_dir / "diagnostic_conditions.csv",
        "samples": output_dir / "imax_sweep_samples.csv",
        "plateaus": output_dir / "imax_plateau_segments.csv",
        "condition_summary": output_dir / "imax_condition_summary.csv",
        "branch_summary": output_dir / "imax_branch_summary.csv",
        "manifest": output_dir / "diagnostic_manifest.json",
    }
    _atomic_dataframe_csv(conditions, paths["conditions"])
    _atomic_dataframe_csv(samples, paths["samples"])
    _atomic_dataframe_csv(plateaus, paths["plateaus"])
    _atomic_dataframe_csv(summaries, paths["condition_summary"])
    _atomic_dataframe_csv(branch_summary, paths["branch_summary"])
    figure_paths = _plot_merged(samples, summaries, output_dir)
    temporary = paths["manifest"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "status": "complete",
        "conditions": len(conditions),
        "production_files_modified": False,
        "voltage_trace_partials": [str(item["traces"]) for item in expected],
        "figures": [str(path) for path in figure_paths],
        "outputs": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }, indent=2), encoding="utf-8")
    temporary.replace(paths["manifest"])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    prepare_parser.add_argument("--output-dir", type=Path, default=None)
    prepare_parser.add_argument("--frequencies", type=float, nargs="+", default=list(DEFAULT_FREQUENCIES))
    prepare_parser.add_argument("--cv", type=float, default=DEFAULT_CV)
    prepare_parser.add_argument("--branch-cutoff-uA", type=float, default=DEFAULT_BRANCH_CUTOFF_UA)
    prepare_parser.add_argument("--repetitions-per-branch", type=int, default=DEFAULT_REPETITIONS_PER_BRANCH)

    count_parser = sub.add_parser("count")
    count_parser.add_argument("--manifest", type=Path, required=True)

    run_parser = sub.add_parser("run-condition")
    run_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    run_parser.add_argument("--condition-index", type=int, required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--partial-dir", type=Path, required=True)
    run_parser.add_argument("--coarse-min-uA", type=float, default=DEFAULT_COARSE_MIN_UA)
    run_parser.add_argument("--coarse-max-uA", type=float, default=DEFAULT_COARSE_MAX_UA)
    run_parser.add_argument("--coarse-step-uA", type=float, default=DEFAULT_COARSE_STEP_UA)
    run_parser.add_argument("--fine-step-uA", type=float, default=DEFAULT_FINE_STEP_UA)
    run_parser.add_argument("--fine-half-width-uA", type=float, default=DEFAULT_FINE_HALF_WIDTH_UA)
    run_parser.add_argument("--late-transition-min-uA", type=float, default=DEFAULT_LATE_TRANSITION_MIN_UA)
    run_parser.add_argument("--max-late-transitions", type=int, default=DEFAULT_MAX_LATE_TRANSITIONS)
    run_parser.add_argument("--plateau-min-width-uA", type=float, default=DEFAULT_PLATEAU_MIN_WIDTH_UA)
    run_parser.add_argument("--plateau-min-fraction-of-max", type=float, default=DEFAULT_PLATEAU_MIN_FRACTION_OF_MAX)
    run_parser.add_argument("--n-jobs", type=int, default=1)
    run_parser.add_argument("--overwrite", action="store_true")

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    merge_parser.add_argument("--manifest", type=Path, required=True)
    merge_parser.add_argument("--partial-dir", type=Path, required=True)
    merge_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            root=args.root, output_dir=args.output_dir, frequencies=args.frequencies,
            cv=args.cv, branch_cutoff_uA=args.branch_cutoff_uA,
            repetitions_per_branch=args.repetitions_per_branch,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ready" else 2
    if args.command == "count":
        print(len(pd.read_csv(args.manifest)))
        return 0
    if args.command == "run-condition":
        result = run_condition(
            root=args.root, condition_index=args.condition_index, manifest=args.manifest,
            partial_dir=args.partial_dir, coarse_min_uA=args.coarse_min_uA,
            coarse_max_uA=args.coarse_max_uA, coarse_step_uA=args.coarse_step_uA,
            fine_step_uA=args.fine_step_uA, fine_half_width_uA=args.fine_half_width_uA,
            late_transition_min_uA=args.late_transition_min_uA,
            max_late_transitions=args.max_late_transitions,
            plateau_min_width_uA=args.plateau_min_width_uA,
            plateau_min_fraction_of_max=args.plateau_min_fraction_of_max,
            n_jobs=args.n_jobs, overwrite=args.overwrite,
        )
        print("\n".join(f"{name}: {path}" for name, path in result.items()))
        return 0
    result = merge(
        root=args.root, manifest=args.manifest, partial_dir=args.partial_dir,
        output_dir=args.output_dir,
    )
    print("\n".join(f"{name}: {path}" for name, path in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
