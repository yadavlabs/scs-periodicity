"""Pilot a branch-aware, endpoint-independent population ``I_max`` search.

The production tables are read-only.  Selected 200-Hz repetitions at high CV
are swept on a fixed amplitude grid so several local plateau definitions can
be compared without using the response at 2000 uA or the largest response in
the sweep as a target.  A sustained decline terminates the eligible ascending
branch; later plateaus are recorded diagnostically but cannot be selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from detection.detection_core import (
    ALL_SPIKES_SEARCH,
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
from space_time_v.spacetime_core import spacetime_path


PILOT_VERSION = "branch_aware_imax_pilot_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/detection/branch_plateau_pilot"
CROSS_FREQUENCY_PILOT_VERSION = "branch_aware_imax_cross_frequency_pilot_v2"
DEFAULT_CROSS_FREQUENCY_OUTPUT_DIR = (
    PROJECT_ROOT / "data/detection/branch_plateau_cross_frequency_pilot"
)
PEAK_VALIDATION_PILOT_VERSION = "peak_first_imax_frequency_cv_validation_v1"
DEFAULT_PEAK_VALIDATION_OUTPUT_DIR = (
    PROJECT_ROOT / "data/detection/peak_first_imax_validation_pilot"
)
DEFAULT_FREQUENCY_HZ = 200.0
DEFAULT_CVS = (0.8, 0.9, 1.0)
DEFAULT_REPETITIONS_PER_CV = 8
DEFAULT_CROSS_FREQUENCIES_HZ = (10.0, 20.0, 50.0, 100.0, 200.0)
DEFAULT_HIGH_BRANCH_CUTOFF_UA = 800.0
DEFAULT_MAX_HIGH_PER_FREQUENCY = 8
DEFAULT_CONTROLS_PER_FREQUENCY = 4
DEFAULT_VALIDATION_CVS = tuple(float(value) for value in np.round(np.arange(0.0, 1.01, 0.1), 1))
DEFAULT_VALIDATION_REPETITIONS_CV0 = 1
DEFAULT_VALIDATION_REPETITIONS_NONZERO = 3
DEFAULT_VALIDATION_SWEEP_WIDTH_UA = 500.0
DEFAULT_VALIDATION_SENTINELS_UA = (800.0, 1200.0, 1600.0, 2000.0)
DEFAULT_EARLY_PEAK_SEARCH_WIDTH_UA = 300.0
DEFAULT_EARLY_PEAK_TAIL_SAMPLES = 3
DEFAULT_EARLY_PEAK_MINIMUM_DROP_SPIKES = 1
DEFAULT_EARLY_PEAK_MINIMUM_RISE_FRACTION = 0.05
DEFAULT_STEP_UA = 25.0
DEFAULT_SAFETY_BOUND_UA = float(ALL_SPIKES_SEARCH["amp_max_uA"])
DEFAULT_CANDIDATE_WIDTHS_UA = (100.0, 150.0)
DEFAULT_CONFIRMATION_WIDTHS_UA = (50.0, 100.0)
DEFAULT_TOLERANCE_FRACTIONS = (0.0, 0.01, 0.02)
DEFAULT_DECLINE_WIDTH_UA = 50.0
PRIMARY_CANDIDATE_WIDTH_UA = 100.0
PRIMARY_CONFIRMATION_WIDTH_UA = 100.0
PRIMARY_TOLERANCE_FRACTION = 0.01
CROSS_FREQUENCY_PRIMARY_TOLERANCE_FRACTION = 0.02


def _source_path(root: Path, source_csv: Path | None) -> Path:
    return Path(source_csv or root / "data/detection" / INDEX_REGIMES_CSV.name).resolve()


def _read_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Index-regime source is required: {path}")
    table = pd.read_csv(path, low_memory=False)
    required = {
        "index_axon_id", "index_cutline_id", "index_diameter_um", "file",
        "cv", "freq", "rep_index", "rep_label", "n_input_pulses",
        "thresh_type", "amp_uA", "n_index_spikes", "N_max_spikes",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Index-regime source is missing columns: {missing}")
    return table


def _nearest_unused(panel: pd.DataFrame, column: str, target: float, used: set[int]) -> int | None:
    available = panel[~panel["rep_index"].astype(int).isin(used)].copy()
    if available.empty:
        return None
    available["distance"] = (pd.to_numeric(available[column], errors="coerce") - float(target)).abs()
    chosen = available.sort_values(["distance", "rep_index"], kind="stable").iloc[0]
    return int(chosen["rep_index"])


def select_conditions(
    table: pd.DataFrame,
    *,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    cvs: Sequence[float] = DEFAULT_CVS,
    repetitions_per_cv: int = DEFAULT_REPETITIONS_PER_CV,
) -> pd.DataFrame:
    """Choose deterministic amplitude/count-spanning high-CV repetitions."""
    imax = table[
        table["thresh_type"].eq("I_all")
        & np.isclose(pd.to_numeric(table["freq"], errors="coerce"), float(frequency_hz))
    ].copy()
    i50 = table[
        table["thresh_type"].eq("I_50")
        & np.isclose(pd.to_numeric(table["freq"], errors="coerce"), float(frequency_hz))
    ][["cv", "freq", "rep_index", "amp_uA", "n_index_spikes"]].rename(
        columns={"amp_uA": "I_50_uA", "n_index_spikes": "N_I_50_spikes"}
    )
    first = table[
        table["thresh_type"].eq("I_first")
        & np.isclose(pd.to_numeric(table["freq"], errors="coerce"), float(frequency_hz))
    ][["cv", "freq", "rep_index", "amp_uA"]].rename(columns={"amp_uA": "I_first_uA"})
    imax = imax.merge(i50, on=["cv", "freq", "rep_index"], validate="one_to_one")
    imax = imax.merge(first, on=["cv", "freq", "rep_index"], validate="one_to_one")
    imax["rep_index"] = pd.to_numeric(imax["rep_index"], errors="raise").astype(int)

    selected_rows: list[dict[str, Any]] = []
    for cv in cvs:
        panel = imax[np.isclose(pd.to_numeric(imax["cv"], errors="coerce"), float(cv))].copy()
        if panel.empty:
            raise RuntimeError(f"No 200-Hz I_max rows at CV={float(cv):g}")
        panel = panel.sort_values("rep_index", kind="stable")
        used: set[int] = set()
        reasons: dict[int, list[str]] = {}

        def add(rep_index: int | None, reason: str) -> None:
            if rep_index is None:
                return
            used.add(int(rep_index))
            reasons.setdefault(int(rep_index), []).append(reason)

        amplitude_order = panel.sort_values(["amp_uA", "rep_index"], kind="stable")
        count_order = panel.sort_values(["n_index_spikes", "rep_index"], kind="stable")
        add(int(amplitude_order.iloc[-1]["rep_index"]), "largest old I_max")
        if len(amplitude_order) > 1:
            add(int(amplitude_order.iloc[-2]["rep_index"]), "second-largest old I_max")
        add(int(count_order.iloc[0]["rep_index"]), "minimum old I_max count")
        add(int(count_order.iloc[-1]["rep_index"]), "maximum old I_max count")
        for quantile in (0.25, 0.50, 0.75):
            add(
                _nearest_unused(panel, "amp_uA", float(panel["amp_uA"].quantile(quantile)), used),
                f"old-I_max amplitude q{int(100 * quantile)}",
            )
        add(
            _nearest_unused(panel, "n_index_spikes", float(panel["n_index_spikes"].median()), used),
            "median old I_max count",
        )
        if len(used) < int(repetitions_per_cv):
            ordered = panel.sort_values(["amp_uA", "n_index_spikes", "rep_index"], kind="stable")
            positions = np.unique(np.rint(np.linspace(0, len(ordered) - 1, int(repetitions_per_cv))).astype(int))
            for position in positions:
                add(int(ordered.iloc[int(position)]["rep_index"]), "amplitude-spanning fill")
                if len(used) >= int(repetitions_per_cv):
                    break
        for rep_index in sorted(used)[: int(repetitions_per_cv)]:
            row = panel[panel["rep_index"].eq(rep_index)].iloc[0].to_dict()
            row["selection_reason"] = "; ".join(reasons.get(rep_index, ["deterministic fill"]))
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows).sort_values(["cv", "rep_index"], kind="stable").reset_index(drop=True)
    selected.insert(0, "condition_index", np.arange(len(selected), dtype=int))
    selected.insert(0, "pilot_version", PILOT_VERSION)
    selected = selected.rename(columns={
        "amp_uA": "old_Imax_uA",
        "n_index_spikes": "old_Imax_spikes",
        "N_max_spikes": "old_N_max_spikes",
    })
    return selected


def _spanning_selection(panel: pd.DataFrame, limit: int) -> list[tuple[int, str]]:
    """Select deterministic extrema and quantiles from a candidate panel."""
    if int(limit) <= 0 or panel.empty:
        return []
    if len(panel) <= int(limit):
        return [(int(index), "all eligible high-branch cases") for index in panel.index]

    chosen: list[int] = []
    reasons: dict[int, list[str]] = {}

    def add(index: int, reason: str) -> None:
        index = int(index)
        if index not in chosen:
            chosen.append(index)
        reasons.setdefault(index, []).append(reason)

    stable = panel.sort_values(["cv", "amp_uA", "n_index_spikes", "rep_index"], kind="stable")
    for column, label in (
        ("amp_uA", "amplitude"), ("n_index_spikes", "spike count"), ("cv", "CV"),
    ):
        ordered = panel.sort_values([column, "cv", "rep_index"], kind="stable")
        add(int(ordered.index[0]), f"minimum old-I_max {label}")
        add(int(ordered.index[-1]), f"maximum old-I_max {label}")
    for quantile in (0.25, 0.50, 0.75):
        target = float(panel["amp_uA"].quantile(quantile))
        available = panel.loc[~panel.index.isin(chosen)].copy()
        if not available.empty:
            available["distance"] = (available["amp_uA"].astype(float) - target).abs()
            index = int(available.sort_values(["distance", "cv", "rep_index"], kind="stable").index[0])
            add(index, f"old-I_max amplitude q{int(100 * quantile)}")
        if len(chosen) >= int(limit):
            break
    if len(chosen) < int(limit):
        positions = np.unique(np.rint(np.linspace(0, len(stable) - 1, int(limit))).astype(int))
        for position in positions:
            add(int(stable.index[int(position)]), "joint CV/amplitude/count spanning fill")
            if len(chosen) >= int(limit):
                break
    if len(chosen) < int(limit):
        for index in stable.index:
            add(int(index), "deterministic fill")
            if len(chosen) >= int(limit):
                break
    return [(index, "; ".join(reasons[index])) for index in chosen[: int(limit)]]


def select_cross_frequency_conditions(
    table: pd.DataFrame,
    *,
    frequencies_hz: Sequence[float] = DEFAULT_CROSS_FREQUENCIES_HZ,
    high_branch_cutoff_uA: float = DEFAULT_HIGH_BRANCH_CUTOFF_UA,
    max_high_per_frequency: int = DEFAULT_MAX_HIGH_PER_FREQUENCY,
    controls_per_frequency: int = DEFAULT_CONTROLS_PER_FREQUENCY,
) -> pd.DataFrame:
    """Select high-amplitude ``I_max`` branches and count/CV-matched controls."""
    if not np.isfinite(high_branch_cutoff_uA) or float(high_branch_cutoff_uA) <= 0:
        raise ValueError("high_branch_cutoff_uA must be finite and positive")
    if int(max_high_per_frequency) <= 0:
        raise ValueError("max_high_per_frequency must be positive")
    if int(controls_per_frequency) < 0:
        raise ValueError("controls_per_frequency must be nonnegative")

    frequencies = tuple(float(value) for value in frequencies_hz)
    imax = table[
        table["thresh_type"].eq("I_all")
        & table["freq"].astype(float).isin(frequencies)
    ].copy()
    i50 = table[
        table["thresh_type"].eq("I_50")
        & table["freq"].astype(float).isin(frequencies)
    ][["cv", "freq", "rep_index", "amp_uA", "n_index_spikes"]].rename(
        columns={"amp_uA": "I_50_uA", "n_index_spikes": "N_I_50_spikes"}
    )
    first = table[
        table["thresh_type"].eq("I_first")
        & table["freq"].astype(float).isin(frequencies)
    ][["cv", "freq", "rep_index", "amp_uA"]].rename(columns={"amp_uA": "I_first_uA"})
    imax = imax.merge(i50, on=["cv", "freq", "rep_index"], validate="one_to_one")
    imax = imax.merge(first, on=["cv", "freq", "rep_index"], validate="one_to_one")
    imax["rep_index"] = imax["rep_index"].astype(int)

    selected_rows: list[dict[str, Any]] = []
    for frequency in frequencies:
        panel = imax[np.isclose(imax["freq"].astype(float), frequency)].copy()
        high = panel[panel["amp_uA"].astype(float) > float(high_branch_cutoff_uA)].copy()
        ordinary = panel[panel["amp_uA"].astype(float) <= float(high_branch_cutoff_uA)].copy()
        if high.empty:
            raise RuntimeError(
                f"No historical high-amplitude I_max branch above {high_branch_cutoff_uA:g} uA "
                f"at {frequency:g} Hz"
            )
        high_selection = _spanning_selection(high, int(max_high_per_frequency))
        selected_high = high.loc[[index for index, _ in high_selection]].copy()
        for index, reason in high_selection:
            row = high.loc[index].to_dict()
            row.update({
                "selection_group": "historical_high_branch",
                "matched_high_cv": float(row["cv"]),
                "matched_high_rep_index": int(row["rep_index"]),
                "selection_reason": reason,
            })
            selected_rows.append(row)

        used_controls: set[int] = set()
        targets = selected_high.sort_values(
            ["cv", "n_index_spikes", "amp_uA", "rep_index"], kind="stable"
        )
        target_positions = np.rint(
            np.linspace(0, len(targets) - 1, max(1, int(controls_per_frequency)))
        ).astype(int)
        for position in target_positions[: int(controls_per_frequency)]:
            target = targets.iloc[int(position)]
            available = ordinary.loc[~ordinary.index.isin(used_controls)].copy()
            same_cv = available[np.isclose(available["cv"].astype(float), float(target["cv"]))]
            if not same_cv.empty:
                available = same_cv
            if available.empty:
                raise RuntimeError(f"No unused matched control remains at {frequency:g} Hz")
            count_scale = max(1.0, float(panel["n_index_spikes"].std(ddof=0)))
            available["match_score"] = (
                10.0 * (available["cv"].astype(float) - float(target["cv"])).abs()
                + (available["n_index_spikes"].astype(float) - float(target["n_index_spikes"])).abs()
                / count_scale
            )
            control = available.sort_values(
                ["match_score", "amp_uA", "rep_index"], kind="stable"
            ).iloc[0]
            used_controls.add(int(control.name))
            row = control.to_dict()
            row.update({
                "selection_group": "matched_ordinary_control",
                "matched_high_cv": float(target["cv"]),
                "matched_high_rep_index": int(target["rep_index"]),
                "selection_reason": (
                    f"matched control for high-branch CV={float(target['cv']):g}, "
                    f"rep={int(target['rep_index'])}"
                ),
            })
            selected_rows.append(row)

    selected = pd.DataFrame(selected_rows).sort_values(
        ["freq", "selection_group", "cv", "rep_index"], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "condition_index", np.arange(len(selected), dtype=int))
    selected.insert(0, "pilot_version", CROSS_FREQUENCY_PILOT_VERSION)
    selected = selected.rename(columns={
        "amp_uA": "old_Imax_uA",
        "n_index_spikes": "old_Imax_spikes",
        "N_max_spikes": "old_N_max_spikes",
    })
    selected["historical_high_branch"] = selected["old_Imax_uA"].astype(float).gt(
        float(high_branch_cutoff_uA)
    )
    selected["high_branch_cutoff_uA"] = float(high_branch_cutoff_uA)
    selected["primary_tolerance_fraction"] = CROSS_FREQUENCY_PRIMARY_TOLERANCE_FRACTION
    selected["primary_candidate_width_uA"] = PRIMARY_CANDIDATE_WIDTH_UA
    selected["primary_confirmation_width_uA"] = PRIMARY_CONFIRMATION_WIDTH_UA
    return selected


def select_frequency_cv_validation_conditions(
    table: pd.DataFrame,
    *,
    frequencies_hz: Sequence[float] = DEFAULT_CROSS_FREQUENCIES_HZ,
    cvs: Sequence[float] = DEFAULT_VALIDATION_CVS,
    repetitions_cv0: int = DEFAULT_VALIDATION_REPETITIONS_CV0,
    repetitions_nonzero: int = DEFAULT_VALIDATION_REPETITIONS_NONZERO,
    high_branch_cutoff_uA: float = DEFAULT_HIGH_BRANCH_CUTOFF_UA,
) -> pd.DataFrame:
    """Select low/median/high historical ``I_max`` cases in every frequency-CV cell."""
    if int(repetitions_cv0) <= 0 or int(repetitions_nonzero) <= 0:
        raise ValueError("validation repetition counts must be positive")
    frequencies = tuple(float(value) for value in frequencies_hz)
    cv_values = tuple(float(value) for value in cvs)
    if not frequencies or not cv_values:
        raise ValueError("validation frequencies and CVs must be nonempty")

    imax = table[
        table["thresh_type"].eq("I_all")
        & table["freq"].astype(float).isin(frequencies)
    ].copy()
    i50 = table[
        table["thresh_type"].eq("I_50")
        & table["freq"].astype(float).isin(frequencies)
    ][["cv", "freq", "rep_index", "amp_uA", "n_index_spikes"]].rename(
        columns={"amp_uA": "I_50_uA", "n_index_spikes": "N_I_50_spikes"}
    )
    first = table[
        table["thresh_type"].eq("I_first")
        & table["freq"].astype(float).isin(frequencies)
    ][["cv", "freq", "rep_index", "amp_uA"]].rename(columns={"amp_uA": "I_first_uA"})
    imax = imax.merge(i50, on=["cv", "freq", "rep_index"], validate="one_to_one")
    imax = imax.merge(first, on=["cv", "freq", "rep_index"], validate="one_to_one")
    imax["rep_index"] = pd.to_numeric(imax["rep_index"], errors="raise").astype(int)

    selected_rows: list[dict[str, Any]] = []
    for frequency in frequencies:
        for cv in cv_values:
            panel = imax[
                np.isclose(imax["freq"].astype(float), frequency)
                & np.isclose(imax["cv"].astype(float), cv)
            ].sort_values("rep_index", kind="stable")
            requested = int(repetitions_cv0 if np.isclose(cv, 0.0) else repetitions_nonzero)
            if len(panel) < requested:
                raise RuntimeError(
                    f"Need {requested} I_max repetitions at {frequency:g} Hz, CV={cv:g}; "
                    f"found {len(panel)}"
                )
            used: set[int] = set()
            choices: list[tuple[int, str]] = []
            quantiles = [0.5] if requested == 1 else list(np.linspace(0.0, 1.0, requested))
            for quantile in quantiles:
                target = float(panel["amp_uA"].astype(float).quantile(float(quantile)))
                rep_index = _nearest_unused(panel, "amp_uA", target, used)
                if rep_index is not None:
                    used.add(rep_index)
                    choices.append((rep_index, f"historical I_max amplitude q{int(round(100 * quantile))}"))
            if len(choices) < requested:
                for rep_index in panel["rep_index"].astype(int):
                    if int(rep_index) not in used:
                        choices.append((int(rep_index), "deterministic repetition fill"))
                        used.add(int(rep_index))
                    if len(choices) >= requested:
                        break
            for rank, (rep_index, reason) in enumerate(choices[:requested]):
                row = panel[panel["rep_index"].eq(int(rep_index))].iloc[0].to_dict()
                row.update({
                    "selection_group": "frequency_cv_validation",
                    "selection_rank": int(rank),
                    "selection_reason": reason,
                })
                selected_rows.append(row)

    selected = pd.DataFrame(selected_rows).sort_values(
        ["freq", "cv", "selection_rank", "rep_index"], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "condition_index", np.arange(len(selected), dtype=int))
    selected.insert(0, "pilot_version", PEAK_VALIDATION_PILOT_VERSION)
    selected = selected.rename(columns={
        "amp_uA": "old_Imax_uA",
        "n_index_spikes": "old_Imax_spikes",
        "N_max_spikes": "old_N_max_spikes",
    })
    selected["historical_high_branch"] = selected["old_Imax_uA"].astype(float).gt(
        float(high_branch_cutoff_uA)
    )
    selected["high_branch_cutoff_uA"] = float(high_branch_cutoff_uA)
    selected["primary_tolerance_fraction"] = CROSS_FREQUENCY_PRIMARY_TOLERANCE_FRACTION
    selected["primary_candidate_width_uA"] = PRIMARY_CANDIDATE_WIDTH_UA
    selected["primary_confirmation_width_uA"] = PRIMARY_CONFIRMATION_WIDTH_UA
    selected["validation_sweep_width_uA"] = DEFAULT_VALIDATION_SWEEP_WIDTH_UA
    selected["validation_sentinel_amplitudes_uA"] = json.dumps(
        list(DEFAULT_VALIDATION_SENTINELS_UA)
    )
    selected["early_peak_search_width_uA"] = DEFAULT_EARLY_PEAK_SEARCH_WIDTH_UA
    selected["early_peak_tail_samples"] = DEFAULT_EARLY_PEAK_TAIL_SAMPLES
    return selected


def coarse_grid(start_uA: float, safety_bound_uA: float, step_uA: float) -> np.ndarray:
    if not np.isfinite(step_uA) or step_uA <= 0:
        raise ValueError("step_uA must be positive")
    start = float(np.ceil((float(start_uA) - 1e-12) / float(step_uA)) * float(step_uA))
    if safety_bound_uA <= start:
        raise ValueError("safety bound must exceed the grid start")
    values = np.arange(start, float(safety_bound_uA) + 0.5 * float(step_uA), float(step_uA))
    return np.unique(np.round(np.r_[values[values <= safety_bound_uA], float(safety_bound_uA)], 9))


def validation_amplitude_grid(
    i50_uA: float,
    *,
    step_uA: float = DEFAULT_STEP_UA,
    sweep_width_uA: float = DEFAULT_VALIDATION_SWEEP_WIDTH_UA,
    sentinel_amplitudes_uA: Sequence[float] = DEFAULT_VALIDATION_SENTINELS_UA,
    safety_bound_uA: float = DEFAULT_SAFETY_BOUND_UA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a contiguous early-search grid plus diagnostic-only sentinels."""
    if not np.isfinite(sweep_width_uA) or float(sweep_width_uA) <= 0:
        raise ValueError("sweep_width_uA must be finite and positive")
    start = float(np.ceil((float(i50_uA) - 1e-12) / float(step_uA)) * float(step_uA))
    stop = min(float(safety_bound_uA), start + float(sweep_width_uA))
    early = coarse_grid(float(i50_uA), stop, float(step_uA))
    sentinels = np.asarray(tuple(float(value) for value in sentinel_amplitudes_uA), dtype=float)
    if sentinels.size and (np.any(~np.isfinite(sentinels)) or np.any(sentinels <= 0)):
        raise ValueError("sentinel amplitudes must be finite and positive")
    sentinels = sentinels[(sentinels > early[-1] + 1e-9) & (sentinels <= float(safety_bound_uA))]
    all_amplitudes = np.unique(np.round(np.r_[early, sentinels], 9))
    is_early = np.isin(all_amplitudes, early)
    return all_amplitudes, is_early


def _window_end(amplitudes: np.ndarray, start_index: int, width_uA: float) -> int | None:
    target = float(amplitudes[start_index]) + float(width_uA) - 1e-9
    index = int(np.searchsorted(amplitudes, target, side="left"))
    return index if index < len(amplitudes) else None


def _first_sustained_decline(
    amplitudes: np.ndarray,
    counts: np.ndarray,
    *,
    tolerance_spikes: int,
    decline_width_uA: float,
) -> tuple[int | None, int]:
    if len(amplitudes) < 2:
        return None, 0
    step = float(np.median(np.diff(amplitudes)))
    required = max(2, int(np.ceil(float(decline_width_uA) / step)) + 1)
    running_max = int(counts[0])
    streak = 0
    for index, count in enumerate(counts):
        running_max = max(running_max, int(count))
        if int(count) < running_max - int(tolerance_spikes):
            streak += 1
        else:
            streak = 0
        if streak >= required:
            return int(index - required + 1), required
    return None, required


def _stable_window(
    amplitudes: np.ndarray,
    counts: np.ndarray,
    *,
    start_index: int,
    candidate_width_uA: float,
    confirmation_width_uA: float,
    tolerance_spikes: int,
) -> dict[str, Any] | None:
    candidate_end = _window_end(amplitudes, start_index, candidate_width_uA)
    total_end = _window_end(amplitudes, start_index, candidate_width_uA + confirmation_width_uA)
    if candidate_end is None or total_end is None:
        return None
    candidate = counts[start_index:candidate_end + 1]
    if int(candidate.max()) - int(candidate.min()) > int(tolerance_spikes):
        return None
    level = int(np.rint(np.median(candidate)))
    confirmation = counts[candidate_end + 1:total_end + 1]
    if confirmation.size and np.any(np.abs(confirmation.astype(int) - level) > int(tolerance_spikes)):
        return None
    combined = counts[start_index:total_end + 1]
    return {
        "start_index": int(start_index),
        "candidate_end_index": int(candidate_end),
        "total_end_index": int(total_end),
        "plateau_count": int(level),
        "plateau_min_count": int(combined.min()),
        "plateau_max_count": int(combined.max()),
    }


def classify_branch(
    amplitudes: Sequence[float],
    counts: Sequence[int],
    *,
    n_input_pulses: int,
    tolerance_fraction: float,
    candidate_width_uA: float,
    confirmation_width_uA: float,
    decline_width_uA: float = DEFAULT_DECLINE_WIDTH_UA,
) -> dict[str, Any]:
    """Classify the first ascending branch without consulting a global maximum."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if amplitudes.ndim != 1 or counts.ndim != 1 or len(amplitudes) != len(counts) or len(counts) == 0:
        raise ValueError("amplitudes and counts must be nonempty one-dimensional vectors of equal length")
    if np.any(np.diff(amplitudes) <= 0):
        raise ValueError("amplitudes must be strictly increasing")
    if tolerance_fraction < 0:
        raise ValueError("tolerance_fraction must be nonnegative")
    tolerance = 0 if tolerance_fraction == 0 else max(1, int(np.ceil(tolerance_fraction * n_input_pulses)))
    decline_index, decline_samples = _first_sustained_decline(
        amplitudes, counts, tolerance_spikes=tolerance, decline_width_uA=decline_width_uA,
    )

    accepted: dict[str, Any] | None = None
    for start_index in range(len(amplitudes)):
        candidate = _stable_window(
            amplitudes, counts, start_index=start_index,
            candidate_width_uA=candidate_width_uA,
            confirmation_width_uA=confirmation_width_uA,
            tolerance_spikes=tolerance,
        )
        if candidate is None:
            continue
        if decline_index is not None and decline_index <= candidate["total_end_index"]:
            break
        accepted = candidate
        break

    later: dict[str, Any] | None = None
    if decline_index is not None:
        for start_index in range(int(decline_index) + int(decline_samples), len(amplitudes)):
            later = _stable_window(
                amplitudes, counts, start_index=start_index,
                candidate_width_uA=candidate_width_uA,
                confirmation_width_uA=confirmation_width_uA,
                tolerance_spikes=tolerance,
            )
            if later is not None:
                break

    if accepted is not None:
        status = "stable_predecline_plateau"
    elif decline_index is not None:
        status = "blocked_before_plateau"
    else:
        status = "no_plateau_by_safety_bound"
    result: dict[str, Any] = {
        "classification": status,
        "tolerance_spikes": int(tolerance),
        "plateau_start_uA": np.nan,
        "plateau_candidate_end_uA": np.nan,
        "plateau_confirmed_through_uA": np.nan,
        "plateau_count": np.nan,
        "plateau_min_count": np.nan,
        "plateau_max_count": np.nan,
        "first_sustained_decline_uA": np.nan if decline_index is None else float(amplitudes[decline_index]),
        "later_plateau_start_uA": np.nan,
        "later_plateau_count": np.nan,
    }
    if accepted is not None:
        result.update({
            "plateau_start_uA": float(amplitudes[accepted["start_index"]]),
            "plateau_candidate_end_uA": float(amplitudes[accepted["candidate_end_index"]]),
            "plateau_confirmed_through_uA": float(amplitudes[accepted["total_end_index"]]),
            "plateau_count": int(accepted["plateau_count"]),
            "plateau_min_count": int(accepted["plateau_min_count"]),
            "plateau_max_count": int(accepted["plateau_max_count"]),
        })
    if later is not None:
        result.update({
            "later_plateau_start_uA": float(amplitudes[later["start_index"]]),
            "later_plateau_count": int(later["plateau_count"]),
        })
    return result


def early_recruitment_peak(
    amplitudes: Sequence[float],
    counts: Sequence[int],
    *,
    n_input_pulses: int,
    search_width_uA: float = 300.0,
    tail_samples: int = 3,
    minimum_drop_spikes: int = 1,
    minimum_rise_fraction: float = 0.05,
) -> dict[str, float | int | bool]:
    """Identify a clear first-lobe peak following the initial recruitment rise.

    Only the initial amplitude window is eligible, preventing a later
    suprastimulation maximum from becoming the candidate. A peak must occur
    before the terminal samples and must be followed by a sustained lower
    tail. Flat saturation curves intentionally fail and fall back to the
    plateau definition.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if amplitudes.ndim != 1 or counts.ndim != 1 or amplitudes.size != counts.size:
        raise ValueError("amplitudes and counts must be equal-length one-dimensional arrays")
    if amplitudes.size < int(tail_samples) + 2:
        raise ValueError("peak search requires more samples than the terminal tail")
    if np.any(np.diff(amplitudes) <= 0):
        raise ValueError("amplitudes must be strictly increasing")
    if int(n_input_pulses) <= 0 or int(tail_samples) <= 0:
        raise ValueError("n_input_pulses and tail_samples must be positive")
    limit = float(amplitudes[0]) + float(search_width_uA) + 1e-9
    eligible = np.flatnonzero(amplitudes <= limit)
    if eligible.size < int(tail_samples) + 2:
        raise ValueError("initial peak-search window contains too few samples")
    stop = int(eligible[-1]) + 1
    window_counts = counts[:stop]
    peak_count = int(window_counts.max())
    peak_index = int(np.flatnonzero(window_counts == peak_count)[0])
    tail = window_counts[-int(tail_samples):]
    tail_level = float(np.median(tail))
    rise = int(peak_count - int(window_counts[0]))
    drop = float(peak_count - tail_level)
    minimum_rise = max(1, int(np.ceil(float(minimum_rise_fraction) * int(n_input_pulses))))
    is_clear = bool(
        peak_index < stop - int(tail_samples)
        and rise >= minimum_rise
        and drop >= int(minimum_drop_spikes)
    )
    return {
        "is_clear_peak": is_clear,
        "peak_amplitude_uA": float(amplitudes[peak_index]),
        "peak_count": peak_count,
        "rise_spikes": rise,
        "terminal_drop_spikes": drop,
        "search_end_uA": float(amplitudes[stop - 1]),
    }


def _task_paths(partial_dir: Path, condition_index: int) -> dict[str, Path]:
    stem = f"task_{int(condition_index):03d}"
    return {
        "sweep": partial_dir / f"sweep_{stem}.csv",
        "candidates": partial_dir / f"candidates_{stem}.csv",
        "summary": partial_dir / f"summary_{stem}.csv",
    }


def prepare(
    *,
    root: Path = PROJECT_ROOT,
    source_csv: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    cvs: Sequence[float] = DEFAULT_CVS,
    repetitions_per_cv: int = DEFAULT_REPETITIONS_PER_CV,
) -> dict[str, Any]:
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    source = _read_source(_source_path(root, source_csv))
    conditions = select_conditions(
        source, frequency_hz=frequency_hz, cvs=cvs, repetitions_per_cv=repetitions_per_cv,
    )
    conditions["primary_tolerance_fraction"] = PRIMARY_TOLERANCE_FRACTION
    conditions["primary_candidate_width_uA"] = PRIMARY_CANDIDATE_WIDTH_UA
    conditions["primary_confirmation_width_uA"] = PRIMARY_CONFIRMATION_WIDTH_UA
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _atomic_dataframe_csv(conditions, output_dir / "pilot_conditions.csv")
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    missing_inputs = []
    for row in conditions.itertuples(index=False):
        path = spacetime_path(
            root / "data/space-time-V", str(index["cutline_id"]),
            float(index["diameter_um"]), float(row.cv), float(row.freq), axon_id=str(index["axon_id"]),
        )
        if not path.is_file():
            missing_inputs.append(str(path))
    payload = {
        "status": "ready" if not missing_inputs else "missing_input",
        "pilot_version": PILOT_VERSION,
        "conditions": int(len(conditions)),
        "condition_manifest": str(manifest_path),
        "source_csv": str(_source_path(root, source_csv)),
        "production_files_modified": False,
        "missing_inputs": sorted(set(missing_inputs)),
        "sweep": {
            "start": "ceil(I_50 / step) * step",
            "step_uA": DEFAULT_STEP_UA,
            "diagnostic_safety_bound_uA": DEFAULT_SAFETY_BOUND_UA,
            "endpoint_or_global_max_used_for_selection": False,
        },
        "primary_definition": {
            "candidate_width_uA": PRIMARY_CANDIDATE_WIDTH_UA,
            "confirmation_width_uA": PRIMARY_CONFIRMATION_WIDTH_UA,
            "tolerance_fraction_of_input_pulses": PRIMARY_TOLERANCE_FRACTION,
            "decline_width_uA": DEFAULT_DECLINE_WIDTH_UA,
            "post_decline_plateaus_eligible": False,
        },
    }
    temporary = output_dir / "preflight.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "preflight.json")
    return payload


def prepare_cross_frequency(
    *,
    root: Path = PROJECT_ROOT,
    source_csv: Path | None = None,
    output_dir: Path = DEFAULT_CROSS_FREQUENCY_OUTPUT_DIR,
    frequencies_hz: Sequence[float] = DEFAULT_CROSS_FREQUENCIES_HZ,
    high_branch_cutoff_uA: float = DEFAULT_HIGH_BRANCH_CUTOFF_UA,
    max_high_per_frequency: int = DEFAULT_MAX_HIGH_PER_FREQUENCY,
    controls_per_frequency: int = DEFAULT_CONTROLS_PER_FREQUENCY,
) -> dict[str, Any]:
    """Prepare a cross-frequency high-branch plus matched-control manifest."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    source_path = _source_path(root, source_csv)
    conditions = select_cross_frequency_conditions(
        _read_source(source_path),
        frequencies_hz=frequencies_hz,
        high_branch_cutoff_uA=high_branch_cutoff_uA,
        max_high_per_frequency=max_high_per_frequency,
        controls_per_frequency=controls_per_frequency,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _atomic_dataframe_csv(conditions, output_dir / "pilot_conditions.csv")
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    missing_inputs = []
    for row in conditions.itertuples(index=False):
        path = spacetime_path(
            root / "data/space-time-V", str(index["cutline_id"]),
            float(index["diameter_um"]), float(row.cv), float(row.freq),
            axon_id=str(index["axon_id"]),
        )
        if not path.is_file():
            missing_inputs.append(str(path))
    payload = {
        "status": "ready" if not missing_inputs else "missing_input",
        "pilot_version": CROSS_FREQUENCY_PILOT_VERSION,
        "conditions": int(len(conditions)),
        "condition_manifest": str(manifest_path),
        "source_csv": str(source_path),
        "production_files_modified": False,
        "missing_inputs": sorted(set(missing_inputs)),
        "selection": {
            "frequencies_hz": [float(value) for value in frequencies_hz],
            "high_branch_cutoff_uA": float(high_branch_cutoff_uA),
            "max_high_per_frequency": int(max_high_per_frequency),
            "controls_per_frequency": int(controls_per_frequency),
        },
        "sweep": {
            "start": "ceil(I_50 / step) * step",
            "step_uA": DEFAULT_STEP_UA,
            "diagnostic_safety_bound_uA": DEFAULT_SAFETY_BOUND_UA,
            "endpoint_or_global_max_used_for_selection": False,
            "retained_per_amplitude": [
                "amplitude_uA", "index_axon_spike_count", "fraction_input_spikes",
                "index_axon_spike_times_ms",
            ],
        },
        "primary_definition": {
            "candidate_width_uA": PRIMARY_CANDIDATE_WIDTH_UA,
            "confirmation_width_uA": PRIMARY_CONFIRMATION_WIDTH_UA,
            "tolerance_fraction_of_input_pulses": CROSS_FREQUENCY_PRIMARY_TOLERANCE_FRACTION,
            "decline_width_uA": DEFAULT_DECLINE_WIDTH_UA,
            "post_decline_plateaus_eligible": False,
        },
    }
    temporary = output_dir / "preflight.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "preflight.json")
    return payload


def prepare_peak_validation(
    *,
    root: Path = PROJECT_ROOT,
    source_csv: Path | None = None,
    output_dir: Path = DEFAULT_PEAK_VALIDATION_OUTPUT_DIR,
    frequencies_hz: Sequence[float] = DEFAULT_CROSS_FREQUENCIES_HZ,
    cvs: Sequence[float] = DEFAULT_VALIDATION_CVS,
    repetitions_cv0: int = DEFAULT_VALIDATION_REPETITIONS_CV0,
    repetitions_nonzero: int = DEFAULT_VALIDATION_REPETITIONS_NONZERO,
    high_branch_cutoff_uA: float = DEFAULT_HIGH_BRANCH_CUTOFF_UA,
) -> dict[str, Any]:
    """Prepare the final frequency-by-CV validation of the peak-first rule."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    source_path = _source_path(root, source_csv)
    conditions = select_frequency_cv_validation_conditions(
        _read_source(source_path),
        frequencies_hz=frequencies_hz,
        cvs=cvs,
        repetitions_cv0=repetitions_cv0,
        repetitions_nonzero=repetitions_nonzero,
        high_branch_cutoff_uA=high_branch_cutoff_uA,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _atomic_dataframe_csv(conditions, output_dir / "pilot_conditions.csv")
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    missing_inputs = []
    for row in conditions.itertuples(index=False):
        path = spacetime_path(
            root / "data/space-time-V", str(index["cutline_id"]),
            float(index["diameter_um"]), float(row.cv), float(row.freq),
            axon_id=str(index["axon_id"]),
        )
        if not path.is_file():
            missing_inputs.append(str(path))
    estimated_runs = sum(
        len(validation_amplitude_grid(
            float(i50_uA),
            sweep_width_uA=DEFAULT_VALIDATION_SWEEP_WIDTH_UA,
            sentinel_amplitudes_uA=DEFAULT_VALIDATION_SENTINELS_UA,
        )[0])
        for i50_uA in conditions["I_50_uA"]
    )
    payload = {
        "status": "ready" if not missing_inputs else "missing_input",
        "pilot_version": PEAK_VALIDATION_PILOT_VERSION,
        "conditions": int(len(conditions)),
        "estimated_neuron_runs": int(estimated_runs),
        "condition_manifest": str(manifest_path),
        "source_csv": str(source_path),
        "production_files_modified": False,
        "missing_inputs": sorted(set(missing_inputs)),
        "selection": {
            "frequencies_hz": [float(value) for value in frequencies_hz],
            "cvs": [float(value) for value in cvs],
            "repetitions_cv0": int(repetitions_cv0),
            "repetitions_nonzero": int(repetitions_nonzero),
            "within_cell_sampling": "low/median/high historical I_max amplitude",
        },
        "sweep": {
            "start": "ceil(I_50 / step) * step",
            "step_uA": DEFAULT_STEP_UA,
            "continuous_width_uA": DEFAULT_VALIDATION_SWEEP_WIDTH_UA,
            "diagnostic_only_sentinel_amplitudes_uA": list(DEFAULT_VALIDATION_SENTINELS_UA),
            "diagnostic_safety_bound_uA": DEFAULT_SAFETY_BOUND_UA,
            "endpoint_or_global_max_used_for_selection": False,
            "retained_per_amplitude": [
                "amplitude_uA", "sample_role", "index_axon_spike_count",
                "fraction_input_spikes", "index_axon_spike_times_ms",
            ],
        },
        "selection_rule": {
            "priority": ["clear_early_peak", "strictest_valid_predecline_plateau"],
            "early_peak_search_width_uA": DEFAULT_EARLY_PEAK_SEARCH_WIDTH_UA,
            "early_peak_tail_samples": DEFAULT_EARLY_PEAK_TAIL_SAMPLES,
            "early_peak_minimum_drop_spikes": DEFAULT_EARLY_PEAK_MINIMUM_DROP_SPIKES,
            "early_peak_minimum_rise_fraction": DEFAULT_EARLY_PEAK_MINIMUM_RISE_FRACTION,
            "plateau_tolerance_fallbacks": list(DEFAULT_TOLERANCE_FRACTIONS),
            "post_decline_or_sentinel_responses_eligible": False,
        },
    }
    temporary = output_dir / "preflight.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "preflight.json")
    return payload


def run_condition(
    *,
    root: Path = PROJECT_ROOT,
    condition_index: int,
    manifest: Path,
    partial_dir: Path,
    step_uA: float = DEFAULT_STEP_UA,
    safety_bound_uA: float = DEFAULT_SAFETY_BOUND_UA,
    overwrite: bool = False,
) -> dict[str, Path]:
    root, manifest, partial_dir = Path(root).resolve(), Path(manifest).resolve(), Path(partial_dir).resolve()
    conditions = pd.read_csv(manifest, low_memory=False)
    match = conditions[conditions["condition_index"].astype(int).eq(int(condition_index))]
    if len(match) != 1:
        raise IndexError(f"condition_index {condition_index} is not unique in {manifest}")
    paths = _task_paths(partial_dir, int(condition_index))
    if all(path.is_file() for path in paths.values()) and not overwrite:
        return paths
    row = match.iloc[0]
    pilot_version = str(row.get("pilot_version", PILOT_VERSION))
    primary_tolerance = float(row.get("primary_tolerance_fraction", PRIMARY_TOLERANCE_FRACTION))
    primary_candidate_width = float(row.get("primary_candidate_width_uA", PRIMARY_CANDIDATE_WIDTH_UA))
    primary_confirmation_width = float(
        row.get("primary_confirmation_width_uA", PRIMARY_CONFIRMATION_WIDTH_UA)
    )
    context_columns = (
        "selection_group", "historical_high_branch", "high_branch_cutoff_uA",
        "matched_high_cv", "matched_high_rep_index", "selection_reason", "selection_rank",
    )
    context = {column: row[column] for column in context_columns if column in row.index}
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    input_path = spacetime_path(
        root / "data/space-time-V", str(index["cutline_id"]),
        float(index["diameter_um"]), float(row["cv"]), float(row["freq"]), axon_id=str(index["axon_id"]),
    )
    rep = load_population_rep(input_path, int(row["rep_index"]))
    config = _config_path(root, rep["diameter_um"], rep["n_valid_nodes"])
    time_trim, voltage_trim = trim_tail_window_ms(rep["time_ms"], rep["V_base"], rep["current"])
    time_trim, voltage_trim = prepend_zeros_ms(time_trim, voltage_trim, prepad_ms=PREPAD_MS)
    runner = PopulationMRGRunner(
        time_trim, voltage_trim, config_path=config, readout_node_number=rep["readout_node_number"],
    )
    validation_mode = pilot_version == PEAK_VALIDATION_PILOT_VERSION
    if validation_mode:
        sentinel_values = json.loads(str(row["validation_sentinel_amplitudes_uA"]))
        amplitudes, is_early_search = validation_amplitude_grid(
            float(row["I_50_uA"]),
            step_uA=float(step_uA),
            sweep_width_uA=float(row["validation_sweep_width_uA"]),
            sentinel_amplitudes_uA=sentinel_values,
            safety_bound_uA=float(safety_bound_uA),
        )
    else:
        amplitudes = coarse_grid(float(row["I_50_uA"]), float(safety_bound_uA), float(step_uA))
        is_early_search = np.ones(len(amplitudes), dtype=bool)
    sample_rows = []
    counts = []
    for amplitude, is_early in zip(amplitudes, is_early_search):
        spikes, count = runner.run_mrg_for_amp(float(amplitude))
        counts.append(int(count))
        sample_rows.append({
            "pilot_version": pilot_version,
            "condition_index": int(condition_index),
            "freq": float(row["freq"]), "cv": float(row["cv"]),
            "rep_index": int(row["rep_index"]), "rep_label": str(row["rep_label"]),
            **context,
            "old_Imax_uA": float(row["old_Imax_uA"]),
            "old_Imax_spikes": int(row["old_Imax_spikes"]),
            "amp_uA": float(amplitude), "n_index_spikes": int(count),
            "sample_role": "early_search" if bool(is_early) else "late_sentinel",
            "fraction_input_spikes": float(count / int(row["n_input_pulses"])),
            "spike_times_ms": json.dumps(np.asarray(spikes, dtype=float).tolist()),
        })

    search_amplitudes = amplitudes[is_early_search]
    search_counts = np.asarray(counts, dtype=int)[is_early_search]
    candidate_rows = []
    for tolerance_fraction in DEFAULT_TOLERANCE_FRACTIONS:
        for candidate_width in DEFAULT_CANDIDATE_WIDTHS_UA:
            for confirmation_width in DEFAULT_CONFIRMATION_WIDTHS_UA:
                result = classify_branch(
                    search_amplitudes, search_counts, n_input_pulses=int(row["n_input_pulses"]),
                    tolerance_fraction=float(tolerance_fraction),
                    candidate_width_uA=float(candidate_width),
                    confirmation_width_uA=float(confirmation_width),
                    decline_width_uA=DEFAULT_DECLINE_WIDTH_UA,
                )
                candidate_rows.append({
                    "pilot_version": pilot_version,
                    "condition_index": int(condition_index),
                    "freq": float(row["freq"]), "cv": float(row["cv"]),
                    "rep_index": int(row["rep_index"]), "rep_label": str(row["rep_label"]),
                    **context,
                    "tolerance_fraction": float(tolerance_fraction),
                    "candidate_width_uA": float(candidate_width),
                    "confirmation_width_uA": float(confirmation_width),
                    "decline_width_uA": DEFAULT_DECLINE_WIDTH_UA,
                    "is_primary": bool(
                        np.isclose(tolerance_fraction, primary_tolerance)
                        and np.isclose(candidate_width, primary_candidate_width)
                        and np.isclose(confirmation_width, primary_confirmation_width)
                    ),
                    **result,
                })
    candidates = pd.DataFrame(candidate_rows)
    primary = candidates[candidates["is_primary"]].iloc[0]
    strictest = candidates[
        np.isclose(candidates["candidate_width_uA"], PRIMARY_CANDIDATE_WIDTH_UA)
        & np.isclose(candidates["confirmation_width_uA"], PRIMARY_CONFIRMATION_WIDTH_UA)
        & candidates["classification"].eq("stable_predecline_plateau")
    ].sort_values("tolerance_fraction", kind="stable")
    strictest_row = None if strictest.empty else strictest.iloc[0]
    peak = early_recruitment_peak(
        search_amplitudes,
        search_counts,
        n_input_pulses=int(row["n_input_pulses"]),
        search_width_uA=float(row.get("early_peak_search_width_uA", DEFAULT_EARLY_PEAK_SEARCH_WIDTH_UA)),
        tail_samples=int(row.get("early_peak_tail_samples", DEFAULT_EARLY_PEAK_TAIL_SAMPLES)),
        minimum_drop_spikes=DEFAULT_EARLY_PEAK_MINIMUM_DROP_SPIKES,
        minimum_rise_fraction=DEFAULT_EARLY_PEAK_MINIMUM_RISE_FRACTION,
    )
    if bool(peak["is_clear_peak"]):
        final_method = "early_response_peak"
        final_amplitude_uA = float(peak["peak_amplitude_uA"])
        final_spikes = int(peak["peak_count"])
        final_plateau_tolerance = np.nan
    elif strictest_row is not None:
        final_method = "stable_saturation_plateau"
        final_amplitude_uA = float(strictest_row["plateau_start_uA"])
        final_spikes = int(strictest_row["plateau_count"])
        final_plateau_tolerance = float(strictest_row["tolerance_fraction"])
    else:
        final_method = "unresolved"
        final_amplitude_uA = np.nan
        final_spikes = np.nan
        final_plateau_tolerance = np.nan
    summary = pd.DataFrame([{
        "pilot_version": pilot_version,
        "condition_index": int(condition_index),
        "freq": float(row["freq"]), "cv": float(row["cv"]),
        "rep_index": int(row["rep_index"]), "rep_label": str(row["rep_label"]),
        **context,
        "n_input_pulses": int(row["n_input_pulses"]),
        "I_first_uA": float(row["I_first_uA"]), "I_50_uA": float(row["I_50_uA"]),
        "old_Imax_uA": float(row["old_Imax_uA"]),
        "old_Imax_spikes": int(row["old_Imax_spikes"]),
        "sampled_max_spikes": int(np.max(counts)),
        "endpoint_spikes": int(counts[-1]),
        "neuron_runs": int(len(amplitudes)),
        "early_search_runs": int(np.sum(is_early_search)),
        "late_sentinel_runs": int(np.sum(~is_early_search)),
        "early_peak_is_clear": bool(peak["is_clear_peak"]),
        "early_peak_amplitude_uA": float(peak["peak_amplitude_uA"]),
        "early_peak_count": int(peak["peak_count"]),
        "early_peak_rise_spikes": int(peak["rise_spikes"]),
        "early_peak_terminal_drop_spikes": float(peak["terminal_drop_spikes"]),
        "final_selection_method": final_method,
        "final_selected_amplitude_uA": final_amplitude_uA,
        "final_selected_spikes": final_spikes,
        "final_plateau_tolerance_fraction": final_plateau_tolerance,
        **{column: primary[column] for column in [
            "classification", "tolerance_spikes", "plateau_start_uA",
            "plateau_candidate_end_uA", "plateau_confirmed_through_uA",
            "plateau_count", "plateau_min_count", "plateau_max_count",
            "first_sustained_decline_uA", "later_plateau_start_uA", "later_plateau_count",
        ]},
    }])
    partial_dir.mkdir(parents=True, exist_ok=True)
    _atomic_dataframe_csv(pd.DataFrame(sample_rows), paths["sweep"])
    _atomic_dataframe_csv(candidates, paths["candidates"])
    _atomic_dataframe_csv(summary, paths["summary"])
    return paths


def _plot(sweeps: pd.DataFrame, summaries: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    paths: list[Path] = []
    frequencies = sorted(summaries["freq"].astype(float).unique())
    for frequency in frequencies:
        group = summaries[np.isclose(summaries["freq"].astype(float), frequency)].copy()
        sort_columns = [column for column in ("selection_group", "cv", "old_Imax_uA") if column in group]
        group = group.sort_values(sort_columns, kind="stable")
        columns = min(4, len(group))
        rows = int(np.ceil(len(group) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.7 * columns, 3.0 * rows), squeeze=False)
        for panel_index, summary in enumerate(group.itertuples(index=False)):
            ax = axes.flat[panel_index]
            panel = sweeps[sweeps["condition_index"].eq(int(summary.condition_index))]
            ax.plot(panel["amp_uA"], panel["n_index_spikes"], color="0.25", marker="o", ms=2.2, lw=1)
            ax.axvline(float(summary.old_Imax_uA), color="#6a3d9a", ls="--", lw=1)
            if pd.notna(summary.plateau_start_uA):
                ax.axvspan(float(summary.plateau_start_uA), float(summary.plateau_confirmed_through_uA),
                           color="#1b9e77", alpha=0.18)
                ax.axvline(float(summary.plateau_start_uA), color="#1b9e77", lw=1.8)
            if pd.notna(summary.first_sustained_decline_uA):
                ax.axvline(float(summary.first_sustained_decline_uA), color="#d73027", ls=":", lw=1.5)
            selection_group = str(getattr(summary, "selection_group", "selected")).replace("_", " ")
            ax.set_title(
                f"{selection_group} | CV={float(summary.cv):g}, rep {int(summary.rep_index)}\n"
                f"{summary.classification.replace('_', ' ')}",
                fontsize=8,
            )
            ax.set_xlabel("Amplitude (uA)")
            ax.set_ylabel("Index-axon spikes")
            ax.grid(alpha=0.2)
        for panel_index in range(len(group), rows * columns):
            axes.flat[panel_index].axis("off")
        fig.legend(handles=[
            Line2D([0], [0], color="0.25", marker="o", ms=3, label="25-uA diagnostic sweep"),
            Line2D([0], [0], color="#1b9e77", lw=2, label="Selected pre-decline plateau onset"),
            Line2D([0], [0], color="#d73027", ls=":", label="First sustained decline"),
            Line2D([0], [0], color="#6a3d9a", ls="--", label="Old I_max"),
        ], frameon=False, ncol=4, loc="lower center")
        fig.suptitle(f"{frequency:g}-Hz branch-aware I_max plateau pilot", fontsize=15)
        fig.tight_layout(rect=(0.02, 0.06, 1, 0.96))
        stem = "branch_plateau_pilot" if len(frequencies) == 1 else f"branch_plateau_pilot_{frequency:g}Hz"
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths


def plot_examples(*, output_dir: Path = DEFAULT_CROSS_FREQUENCY_OUTPUT_DIR) -> list[Path]:
    """Plot one informative high-branch/control pair at each frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    output_dir = Path(output_dir).resolve()
    sweeps = pd.read_csv(output_dir / "branch_plateau_sweeps.csv", low_memory=False)
    summaries = pd.read_csv(output_dir / "branch_plateau_primary_summary.csv", low_memory=False)
    candidates = pd.read_csv(output_dir / "branch_plateau_candidates.csv", low_memory=False)
    if "selection_group" not in summaries.columns:
        raise ValueError("Example selection requires cross-frequency selection metadata")

    # Prefer the strictest definition that establishes a complete pre-decline
    # plateau. This prevents the 2% fallback from moving low-frequency onsets
    # earlier when an exact stable plateau is already present.
    eligible = candidates[
        np.isclose(candidates["candidate_width_uA"], PRIMARY_CANDIDATE_WIDTH_UA)
        & np.isclose(candidates["confirmation_width_uA"], PRIMARY_CONFIRMATION_WIDTH_UA)
        & candidates["classification"].eq("stable_predecline_plateau")
    ].sort_values(["condition_index", "tolerance_fraction"], kind="stable")
    chosen = eligible.groupby("condition_index", as_index=False).first()
    if len(chosen) != len(summaries):
        raise RuntimeError(
            f"Strictest-valid definition resolved {len(chosen)} of {len(summaries)} conditions"
        )
    definition_columns = [
        "classification", "tolerance_spikes", "plateau_start_uA",
        "plateau_candidate_end_uA", "plateau_confirmed_through_uA",
        "plateau_count", "plateau_min_count", "plateau_max_count",
        "first_sustained_decline_uA", "later_plateau_start_uA", "later_plateau_count",
    ]
    summaries = summaries.drop(columns=definition_columns).merge(
        chosen[["condition_index", "tolerance_fraction", *definition_columns]].rename(
            columns={"tolerance_fraction": "chosen_tolerance_fraction"}
        ),
        on="condition_index", validate="one_to_one",
    )
    strictest_summary_path = _atomic_dataframe_csv(
        summaries, output_dir / "branch_plateau_strictest_valid_summary.csv"
    )

    metric_rows = []
    for summary in summaries.itertuples(index=False):
        panel = sweeps[sweeps["condition_index"].eq(int(summary.condition_index))].sort_values("amp_uA")
        late = panel[panel["amp_uA"].ge(800.0)]
        peak = early_recruitment_peak(
            panel["amp_uA"], panel["n_index_spikes"],
            n_input_pulses=int(summary.n_input_pulses),
        )
        metric_rows.append({
            "condition_index": int(summary.condition_index),
            "old_extra_spikes": float(summary.old_Imax_spikes - summary.plateau_count),
            "late_range": float(late["n_index_spikes"].max() - late["n_index_spikes"].min()),
            "amplitude_reduction_uA": float(summary.old_Imax_uA - summary.plateau_start_uA),
            "early_peak_is_clear": bool(peak["is_clear_peak"]),
            "early_peak_amplitude_uA": float(peak["peak_amplitude_uA"]),
            "early_peak_count": int(peak["peak_count"]),
            "early_peak_terminal_drop_spikes": float(peak["terminal_drop_spikes"]),
        })
    metrics = pd.DataFrame(metric_rows)
    summaries = summaries.merge(metrics, on="condition_index", validate="one_to_one")

    selected_rows: list[dict[str, Any]] = []
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for frequency in sorted(summaries["freq"].astype(float).unique()):
        high = summaries[
            np.isclose(summaries["freq"].astype(float), frequency)
            & summaries["selection_group"].eq("historical_high_branch")
        ].sort_values(
            ["early_peak_is_clear", "old_extra_spikes", "late_range", "amplitude_reduction_uA"],
            ascending=False, kind="stable",
        )
        chosen_high = high.iloc[0]
        controls = summaries[
            np.isclose(summaries["freq"].astype(float), frequency)
            & summaries["selection_group"].eq("matched_ordinary_control")
        ].copy()
        direct = controls[
            np.isclose(controls["matched_high_cv"].astype(float), float(chosen_high["cv"]))
            & controls["matched_high_rep_index"].astype(int).eq(int(chosen_high["rep_index"]))
        ]
        if direct.empty:
            controls["match_distance"] = (
                10.0 * (controls["cv"].astype(float) - float(chosen_high["cv"])).abs()
                + (controls["old_Imax_spikes"].astype(float) - float(chosen_high["old_Imax_spikes"])).abs()
            )
            chosen_control = controls.sort_values(["match_distance", "rep_index"], kind="stable").iloc[0]
        else:
            chosen_control = direct.sort_values("rep_index", kind="stable").iloc[0]
        pairs.append((chosen_high, chosen_control))
        for role, row in (("historical high branch", chosen_high), ("matched ordinary control", chosen_control)):
            selected_rows.append({
                "role": role,
                "condition_index": int(row["condition_index"]),
                "freq": float(row["freq"]), "cv": float(row["cv"]),
                "rep_index": int(row["rep_index"]),
                "old_Imax_uA": float(row["old_Imax_uA"]),
                "old_Imax_spikes": int(row["old_Imax_spikes"]),
                "plateau_start_uA": float(row["plateau_start_uA"]),
                "plateau_count": int(row["plateau_count"]),
                "chosen_tolerance_fraction": float(row["chosen_tolerance_fraction"]),
                "early_peak_is_clear": bool(row["early_peak_is_clear"]),
                "early_peak_amplitude_uA": float(row["early_peak_amplitude_uA"]),
                "early_peak_count": int(row["early_peak_count"]),
            })

    figure, axes = plt.subplots(2, len(pairs), figsize=(4.0 * len(pairs), 7.3), squeeze=False)
    for column, (high, control) in enumerate(pairs):
        for row_index, summary in enumerate((high, control)):
            ax = axes[row_index, column]
            panel = sweeps[sweeps["condition_index"].eq(int(summary["condition_index"]))].sort_values("amp_uA")
            ax.plot(
                panel["amp_uA"], panel["n_index_spikes"], color="0.18", marker="o",
                markersize=2.8, linewidth=1.3, zorder=2,
            )
            ax.axvspan(
                float(summary["plateau_start_uA"]), float(summary["plateau_confirmed_through_uA"]),
                color="#1b9e77", alpha=0.16, zorder=0,
            )
            ax.axvline(float(summary["plateau_start_uA"]), color="#1b9e77", linewidth=1.8)
            if pd.notna(summary["first_sustained_decline_uA"]):
                ax.axvline(
                    float(summary["first_sustained_decline_uA"]), color="#d73027",
                    linestyle=":", linewidth=1.8,
                )
            ax.axvline(float(summary["old_Imax_uA"]), color="#6a3d9a", linestyle="--", linewidth=1.2)
            ax.scatter(
                [float(summary["old_Imax_uA"])], [float(summary["old_Imax_spikes"])],
                marker="*", s=85, color="#6a3d9a", edgecolor="white", linewidth=0.6, zorder=5,
            )
            onset_sample = panel[
                np.isclose(panel["amp_uA"].astype(float), float(summary["plateau_start_uA"]))
            ]
            onset_count = (
                float(onset_sample.iloc[0]["n_index_spikes"])
                if not onset_sample.empty else float(summary["plateau_count"])
            )
            ax.scatter(
                [float(summary["plateau_start_uA"])], [onset_count],
                marker="o", s=36, color="#1b9e77", edgecolor="white", linewidth=0.6, zorder=5,
            )
            if bool(summary["early_peak_is_clear"]):
                ax.scatter(
                    [float(summary["early_peak_amplitude_uA"])],
                    [float(summary["early_peak_count"])],
                    marker="v", s=72, color="#00a6ca", edgecolor="white", linewidth=0.7, zorder=6,
                )
            ax.text(
                0.03, 0.06,
                f"CV={float(summary['cv']):g}, rep {int(summary['rep_index'])}; "
                f"tol={'exact' if float(summary['chosen_tolerance_fraction']) == 0 else f'{100 * float(summary['chosen_tolerance_fraction']):g}%'}",
                transform=ax.transAxes, fontsize=8.5, ha="left", va="bottom",
            )
            ax.grid(alpha=0.20)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
            if row_index == 0:
                ax.set_title(f"{float(summary['freq']):g} Hz", fontsize=12)
            if column == 0:
                ax.set_ylabel(
                    "Index-axon spikes\nHistorical high branch" if row_index == 0
                    else "Index-axon spikes\nMatched ordinary control"
                )
            if row_index == 1:
                ax.set_xlabel("Amplitude (µA)")
    figure.legend(handles=[
        Line2D([0], [0], color="0.18", marker="o", markersize=3, label="25-µA sweep"),
        Line2D([0], [0], color="#1b9e77", linewidth=2, label="Strictest valid plateau onset"),
        Line2D([0], [0], color="#1b9e77", linewidth=8, alpha=0.16, label="200-µA validation interval"),
        Line2D([0], [0], color="#d73027", linestyle=":", linewidth=2, label="First sustained decline"),
        Line2D([0], [0], color="#6a3d9a", linestyle="--", marker="*", markersize=8,
               label="Historical I_max"),
        Line2D([0], [0], color="#00a6ca", marker="v", linestyle="None", markersize=7,
               label="Clear early-peak candidate"),
    ], loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.005))
    figure.suptitle("Representative index-axon amplitude–spike response curves", fontsize=16)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"branch_plateau_example_amplitude_spike_curves.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    paths.append(_atomic_dataframe_csv(
        pd.DataFrame(selected_rows), output_dir / "branch_plateau_example_conditions.csv"
    ))
    paths.append(strictest_summary_path)
    return paths


def _peak_validation_diagnostics(
    sweeps: pd.DataFrame,
    summaries: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Path], list[Path], dict[str, Any]]:
    """Build sensitivity and cross-CV audit products for the final validation pilot."""
    sensitivity_rows: list[dict[str, Any]] = []
    for summary in summaries.itertuples(index=False):
        panel = sweeps[sweeps["condition_index"].eq(int(summary.condition_index))].copy()
        if "sample_role" in panel.columns:
            panel = panel[panel["sample_role"].eq("early_search")]
        panel = panel.sort_values("amp_uA", kind="stable")
        for search_width in (250.0, 300.0, 350.0, 400.0, 450.0, 500.0):
            for tail_samples in (2, 3, 4):
                result = early_recruitment_peak(
                    panel["amp_uA"], panel["n_index_spikes"],
                    n_input_pulses=int(summary.n_input_pulses),
                    search_width_uA=search_width,
                    tail_samples=tail_samples,
                    minimum_drop_spikes=DEFAULT_EARLY_PEAK_MINIMUM_DROP_SPIKES,
                    minimum_rise_fraction=DEFAULT_EARLY_PEAK_MINIMUM_RISE_FRACTION,
                )
                sensitivity_rows.append({
                    "condition_index": int(summary.condition_index),
                    "freq": float(summary.freq), "cv": float(summary.cv),
                    "rep_index": int(summary.rep_index),
                    "search_width_uA": search_width,
                    "tail_samples": tail_samples,
                    **result,
                })
    sensitivity = pd.DataFrame(sensitivity_rows)
    stability_rows = []
    for condition_index, panel in sensitivity.groupby("condition_index", sort=True):
        primary = panel[
            np.isclose(panel["search_width_uA"], DEFAULT_EARLY_PEAK_SEARCH_WIDTH_UA)
            & panel["tail_samples"].eq(DEFAULT_EARLY_PEAK_TAIL_SAMPLES)
        ].iloc[0]
        robust = panel[
            panel["search_width_uA"].isin([300.0, 350.0])
            & panel["tail_samples"].isin([2, 3, 4])
        ]
        classification_stable = bool(
            robust["is_clear_peak"].astype(bool).eq(bool(primary["is_clear_peak"])).all()
        )
        amplitude_stable = bool(
            not bool(primary["is_clear_peak"])
            or robust.loc[robust["is_clear_peak"].astype(bool), "peak_amplitude_uA"]
            .astype(float).eq(float(primary["peak_amplitude_uA"])).all()
        )
        stability_rows.append({
            "condition_index": int(condition_index),
            "peak_classification_stable_300_350_uA": classification_stable,
            "peak_amplitude_stable_300_350_uA": amplitude_stable,
            "peak_sensitivity_stable": bool(classification_stable and amplitude_stable),
        })
    stability = pd.DataFrame(stability_rows)
    audited = summaries.merge(stability, on="condition_index", validate="one_to_one")

    sentinel_rows = []
    for summary in audited.itertuples(index=False):
        panel = sweeps[sweeps["condition_index"].eq(int(summary.condition_index))]
        early = panel[panel.get("sample_role", pd.Series(index=panel.index, dtype=str)).eq("early_search")]
        selected_sample = early[np.isclose(
            early["amp_uA"].astype(float), float(summary.final_selected_amplitude_uA)
        )]
        actual_selected_spikes = (
            np.nan if selected_sample.empty else int(selected_sample.iloc[0]["n_index_spikes"])
        )
        recommended_amplitude = float(summary.final_selected_amplitude_uA)
        if str(summary.final_selection_method) == "stable_saturation_plateau":
            reaches_level = early[
                early["amp_uA"].astype(float).ge(recommended_amplitude - 1e-9)
                & early["n_index_spikes"].astype(float).ge(float(summary.final_selected_spikes) - 1e-9)
            ].sort_values("amp_uA", kind="stable")
            if not reaches_level.empty:
                recommended_amplitude = float(reaches_level.iloc[0]["amp_uA"])
        recommended_sample = early[np.isclose(
            early["amp_uA"].astype(float), recommended_amplitude
        )]
        recommended_spikes = (
            np.nan if recommended_sample.empty else int(recommended_sample.iloc[0]["n_index_spikes"])
        )
        sentinel = panel[panel.get("sample_role", pd.Series(index=panel.index, dtype=str)).eq("late_sentinel")]
        sentinel_rows.append({
            "condition_index": int(summary.condition_index),
            "actual_spikes_at_reported_selected_amplitude": actual_selected_spikes,
            "recommended_selected_amplitude_uA": recommended_amplitude,
            "recommended_selected_spikes": recommended_spikes,
            "plateau_onset_adjusted_to_reach_representative_count": bool(
                not np.isclose(recommended_amplitude, float(summary.final_selected_amplitude_uA))
            ),
            "late_sentinel_min_spikes": np.nan if sentinel.empty else int(sentinel["n_index_spikes"].min()),
            "late_sentinel_max_spikes": np.nan if sentinel.empty else int(sentinel["n_index_spikes"].max()),
            "late_sentinel_exceeds_selected": bool(
                not sentinel.empty
                and pd.notna(summary.final_selected_spikes)
                and int(sentinel["n_index_spikes"].max()) > float(summary.final_selected_spikes)
            ),
        })
    audited = audited.merge(pd.DataFrame(sentinel_rows), on="condition_index", validate="one_to_one")
    outcome_rows = []
    for summary in audited.itertuples(index=False):
        panel = sensitivity[sensitivity["condition_index"].eq(int(summary.condition_index))]
        amplitudes = np.where(
            panel["is_clear_peak"].astype(bool),
            panel["peak_amplitude_uA"].astype(float),
            float(summary.recommended_selected_amplitude_uA),
        )
        outcome_rows.append({
            "condition_index": int(summary.condition_index),
            "selection_outcome_stable_250_500_uA": bool(np.ptp(amplitudes) <= 1e-9),
            "selection_outcome_amplitude_range_uA": float(np.ptp(amplitudes)),
        })
    audited = audited.merge(pd.DataFrame(outcome_rows), on="condition_index", validate="one_to_one")
    audited["selected_fraction_input_spikes"] = (
        audited["recommended_selected_spikes"].astype(float) / audited["n_input_pulses"].astype(float)
    )
    audited["selected_historical_high_branch"] = (
        audited["recommended_selected_amplitude_uA"].astype(float)
        > audited["high_branch_cutoff_uA"].astype(float)
    )

    by_frequency_cv = (
        audited.groupby(["freq", "cv"], observed=True, dropna=False)
        .agg(
            repetitions=("condition_index", "size"),
            early_peak_repetitions=("early_peak_is_clear", "sum"),
            median_selected_amplitude_uA=("recommended_selected_amplitude_uA", "median"),
            min_selected_amplitude_uA=("recommended_selected_amplitude_uA", "min"),
            max_selected_amplitude_uA=("recommended_selected_amplitude_uA", "max"),
            median_selected_spikes=("recommended_selected_spikes", "median"),
            median_selected_fraction_input_spikes=("selected_fraction_input_spikes", "median"),
            unresolved_repetitions=(
                "final_selection_method", lambda values: int(values.eq("unresolved").sum())
            ),
            sensitivity_unstable_repetitions=(
                "peak_sensitivity_stable", lambda values: int((~values.astype(bool)).sum())
            ),
            outcome_unstable_repetitions=(
                "selection_outcome_stable_250_500_uA",
                lambda values: int((~values.astype(bool)).sum()),
            ),
            historical_high_branch_selections=("selected_historical_high_branch", "sum"),
            late_sentinel_exceeds_selected_repetitions=("late_sentinel_exceeds_selected", "sum"),
        ).reset_index()
    )
    by_frequency_cv["early_peak_fraction"] = (
        by_frequency_cv["early_peak_repetitions"] / by_frequency_cv["repetitions"]
    )
    paths = {
        "validation_conditions": output_dir / "peak_first_validation_condition_summary.csv",
        "peak_sensitivity": output_dir / "peak_first_validation_sensitivity.csv",
        "validation_by_frequency_cv": output_dir / "peak_first_validation_by_frequency_cv.csv",
        "acceptance": output_dir / "peak_first_validation_acceptance.json",
    }
    _atomic_dataframe_csv(audited, paths["validation_conditions"])
    _atomic_dataframe_csv(sensitivity, paths["peak_sensitivity"])
    _atomic_dataframe_csv(by_frequency_cv, paths["validation_by_frequency_cv"])

    total = int(len(audited))
    automatic_checks = {
        "all_conditions_resolved": bool(audited["final_selection_method"].ne("unresolved").all()),
        "all_selection_outcomes_stable_250_500_uA": bool(
            audited["selection_outcome_stable_250_500_uA"].astype(bool).all()
        ),
        "no_selected_amplitude_above_high_branch_cutoff": bool(
            (~audited["selected_historical_high_branch"].astype(bool)).all()
        ),
        "all_frequency_cv_cells_present": bool(
            len(by_frequency_cv)
            == audited["freq"].nunique(dropna=True) * audited["cv"].nunique(dropna=True)
        ),
        "late_sentinels_excluded_from_selection_by_construction": True,
    }
    acceptance = {
        "status": "automatic_checks_passed" if all(automatic_checks.values()) else "review_required",
        "conditions": total,
        "early_peak_conditions": int(audited["early_peak_is_clear"].astype(bool).sum()),
        "plateau_fallback_conditions": int(
            audited["final_selection_method"].eq("stable_saturation_plateau").sum()
        ),
        "unresolved_conditions": int(audited["final_selection_method"].eq("unresolved").sum()),
        "sensitivity_unstable_conditions": int((~audited["peak_sensitivity_stable"].astype(bool)).sum()),
        "selection_outcome_unstable_conditions": int(
            (~audited["selection_outcome_stable_250_500_uA"].astype(bool)).sum()
        ),
        "plateau_onsets_adjusted_to_reach_representative_count": int(
            audited["plateau_onset_adjusted_to_reach_representative_count"].astype(bool).sum()
        ),
        "selected_above_high_branch_cutoff": int(audited["selected_historical_high_branch"].sum()),
        "late_sentinel_exceeds_selected_conditions": int(audited["late_sentinel_exceeds_selected"].sum()),
        "automatic_checks": automatic_checks,
        "manual_review_still_required": [
            "adjacent-CV continuity of selected amplitudes and spike counts",
            "representative raw amplitude-spike curves",
        ],
        "production_files_modified": False,
    }
    temporary = paths["acceptance"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(paths["acceptance"])

    figure_paths: list[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frequencies = sorted(by_frequency_cv["freq"].astype(float).unique())
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(frequencies)))
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
        for color, frequency in zip(colors, frequencies):
            panel = by_frequency_cv[np.isclose(by_frequency_cv["freq"], frequency)].sort_values("cv")
            cv_values = panel["cv"].to_numpy(dtype=float)
            axes[0].plot(cv_values, panel["early_peak_fraction"].to_numpy(dtype=float), marker="o", color=color,
                         label=f"{frequency:g} Hz")
            axes[1].plot(
                cv_values, panel["median_selected_amplitude_uA"].to_numpy(dtype=float),
                marker="o", color=color,
            )
            axes[1].fill_between(
                cv_values, panel["min_selected_amplitude_uA"].to_numpy(dtype=float),
                panel["max_selected_amplitude_uA"].to_numpy(dtype=float), color=color, alpha=0.12,
            )
        axes[0].set(title="Early-peak prevalence", xlabel="CV", ylabel="Fraction of repetitions")
        axes[1].set(title="Peak-first / plateau-fallback selection", xlabel="CV",
                    ylabel="Selected amplitude (uA)")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.set_xlim(-0.02, 1.02)
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].legend(title="Frequency", frameon=False, ncol=2)
        figure.suptitle("Cross-frequency and cross-CV I_max definition validation", fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        for suffix in ("png", "pdf"):
            path = output_dir / f"peak_first_validation_summary.{suffix}"
            figure.savefig(path, dpi=300, bbox_inches="tight")
            figure_paths.append(path)
        plt.close(figure)
    except Exception:
        figure_paths = []
    return paths, figure_paths, acceptance


def merge(*, manifest: Path, partial_dir: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    manifest, partial_dir, output_dir = Path(manifest).resolve(), Path(partial_dir).resolve(), Path(output_dir).resolve()
    conditions = pd.read_csv(manifest)
    expected = [_task_paths(partial_dir, int(index)) for index in conditions["condition_index"]]
    missing = [str(path) for paths in expected for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot merge branch pilot; {len(missing)} outputs are missing (first: {missing[0]})")
    sweeps = pd.concat([pd.read_csv(paths["sweep"]) for paths in expected], ignore_index=True)
    candidates = pd.concat([pd.read_csv(paths["candidates"]) for paths in expected], ignore_index=True)
    summaries = pd.concat([pd.read_csv(paths["summary"]) for paths in expected], ignore_index=True)
    comparison_groups = [
        column for column in ("freq", "selection_group") if column in candidates.columns
    ] + [
        "tolerance_fraction", "candidate_width_uA", "confirmation_width_uA", "classification"
    ]
    comparison = (
        candidates.groupby(
            comparison_groups,
            dropna=False, observed=True,
        ).size().rename("repetitions").reset_index()
    )
    primary_groups = ["freq", "cv"]
    if "selection_group" in summaries.columns:
        primary_groups.append("selection_group")
    primary_groups.append("classification")
    primary_by_cv = (
        summaries.groupby(primary_groups, dropna=False, observed=True)
        .agg(
            repetitions=("rep_index", "size"),
            median_plateau_start_uA=("plateau_start_uA", "median"),
            median_plateau_count=("plateau_count", "median"),
            later_plateau_repetitions=("later_plateau_start_uA", lambda values: int(values.notna().sum())),
        ).reset_index()
    )
    selection_summary = conditions.groupby(
        [column for column in ("freq", "selection_group") if column in conditions.columns] or ["freq"],
        dropna=False, observed=True,
    ).agg(
        repetitions=("rep_index", "size"),
        min_old_Imax_uA=("old_Imax_uA", "min"),
        median_old_Imax_uA=("old_Imax_uA", "median"),
        max_old_Imax_uA=("old_Imax_uA", "max"),
    ).reset_index()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "conditions": output_dir / "pilot_conditions.csv",
        "sweeps": output_dir / "branch_plateau_sweeps.csv",
        "candidates": output_dir / "branch_plateau_candidates.csv",
        "summaries": output_dir / "branch_plateau_primary_summary.csv",
        "comparison": output_dir / "branch_plateau_definition_comparison.csv",
        "primary_by_cv": output_dir / "branch_plateau_primary_by_cv.csv",
        "selection_summary": output_dir / "branch_plateau_selection_summary.csv",
        "manifest": output_dir / "pilot_manifest.json",
    }
    _atomic_dataframe_csv(conditions, paths["conditions"])
    _atomic_dataframe_csv(sweeps, paths["sweeps"])
    _atomic_dataframe_csv(candidates, paths["candidates"])
    _atomic_dataframe_csv(summaries, paths["summaries"])
    _atomic_dataframe_csv(comparison, paths["comparison"])
    _atomic_dataframe_csv(primary_by_cv, paths["primary_by_cv"])
    _atomic_dataframe_csv(selection_summary, paths["selection_summary"])
    figures: list[Path] = []
    figure_error: str | None = None
    validation_paths: dict[str, Path] = {}
    validation_acceptance: dict[str, Any] | None = None
    try:
        versions = sorted(set(conditions.get("pilot_version", pd.Series([PILOT_VERSION])).astype(str)))
        if versions == [PEAK_VALIDATION_PILOT_VERSION]:
            validation_paths, figures, validation_acceptance = _peak_validation_diagnostics(
                sweeps, summaries, output_dir
            )
            paths.update(validation_paths)
        else:
            figures = _plot(sweeps, summaries, output_dir)
    except Exception as error:  # Data products remain auditable if cluster plotting fails.
        figure_error = f"{type(error).__name__}: {error}"
    temporary = paths["manifest"].with_suffix(".json.tmp")
    versions = sorted(set(conditions.get("pilot_version", pd.Series([PILOT_VERSION])).astype(str)))
    temporary.write_text(json.dumps({
        "status": "complete" if figure_error is None else "complete_with_plot_error",
        "pilot_versions": versions,
        "conditions": int(len(conditions)), "production_files_modified": False,
        "endpoint_or_global_max_used_for_selection": False,
        "figures": [str(path) for path in figures],
        "figure_error": figure_error,
        "validation_acceptance": validation_acceptance,
        "sweep_columns": list(sweeps.columns),
        "outputs": {name: str(path) for name, path in paths.items() if name != "manifest"},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(paths["manifest"])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    prepare_parser.add_argument("--source-csv", type=Path, default=None)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare_parser.add_argument("--frequency", type=float, default=DEFAULT_FREQUENCY_HZ)
    prepare_parser.add_argument("--cvs", type=float, nargs="+", default=list(DEFAULT_CVS))
    prepare_parser.add_argument("--repetitions-per-cv", type=int, default=DEFAULT_REPETITIONS_PER_CV)

    cross_parser = sub.add_parser("prepare-cross-frequency")
    cross_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    cross_parser.add_argument("--source-csv", type=Path, default=None)
    cross_parser.add_argument("--output-dir", type=Path, default=DEFAULT_CROSS_FREQUENCY_OUTPUT_DIR)
    cross_parser.add_argument(
        "--frequencies", type=float, nargs="+", default=list(DEFAULT_CROSS_FREQUENCIES_HZ)
    )
    cross_parser.add_argument("--high-branch-cutoff-uA", type=float, default=DEFAULT_HIGH_BRANCH_CUTOFF_UA)
    cross_parser.add_argument("--max-high-per-frequency", type=int, default=DEFAULT_MAX_HIGH_PER_FREQUENCY)
    cross_parser.add_argument("--controls-per-frequency", type=int, default=DEFAULT_CONTROLS_PER_FREQUENCY)

    validation_parser = sub.add_parser("prepare-peak-validation")
    validation_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    validation_parser.add_argument("--source-csv", type=Path, default=None)
    validation_parser.add_argument("--output-dir", type=Path, default=DEFAULT_PEAK_VALIDATION_OUTPUT_DIR)
    validation_parser.add_argument(
        "--frequencies", type=float, nargs="+", default=list(DEFAULT_CROSS_FREQUENCIES_HZ)
    )
    validation_parser.add_argument(
        "--cvs", type=float, nargs="+", default=list(DEFAULT_VALIDATION_CVS)
    )
    validation_parser.add_argument(
        "--repetitions-cv0", type=int, default=DEFAULT_VALIDATION_REPETITIONS_CV0
    )
    validation_parser.add_argument(
        "--repetitions-nonzero", type=int, default=DEFAULT_VALIDATION_REPETITIONS_NONZERO
    )
    validation_parser.add_argument(
        "--high-branch-cutoff-uA", type=float, default=DEFAULT_HIGH_BRANCH_CUTOFF_UA
    )

    count_parser = sub.add_parser("count")
    count_parser.add_argument("--manifest", type=Path, required=True)

    run_parser = sub.add_parser("run-condition")
    run_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    run_parser.add_argument("--condition-index", type=int, required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--partial-dir", type=Path, required=True)
    run_parser.add_argument("--step-uA", type=float, default=DEFAULT_STEP_UA)
    run_parser.add_argument("--safety-bound-uA", type=float, default=DEFAULT_SAFETY_BOUND_UA)
    run_parser.add_argument("--overwrite", action="store_true")

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--manifest", type=Path, required=True)
    merge_parser.add_argument("--partial-dir", type=Path, required=True)
    merge_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    examples_parser = sub.add_parser("plot-examples")
    examples_parser.add_argument("--output-dir", type=Path, default=DEFAULT_CROSS_FREQUENCY_OUTPUT_DIR)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            root=args.root, source_csv=args.source_csv, output_dir=args.output_dir,
            frequency_hz=args.frequency, cvs=args.cvs, repetitions_per_cv=args.repetitions_per_cv,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ready" else 2
    if args.command == "prepare-cross-frequency":
        result = prepare_cross_frequency(
            root=args.root,
            source_csv=args.source_csv,
            output_dir=args.output_dir,
            frequencies_hz=args.frequencies,
            high_branch_cutoff_uA=args.high_branch_cutoff_uA,
            max_high_per_frequency=args.max_high_per_frequency,
            controls_per_frequency=args.controls_per_frequency,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ready" else 2
    if args.command == "prepare-peak-validation":
        result = prepare_peak_validation(
            root=args.root,
            source_csv=args.source_csv,
            output_dir=args.output_dir,
            frequencies_hz=args.frequencies,
            cvs=args.cvs,
            repetitions_cv0=args.repetitions_cv0,
            repetitions_nonzero=args.repetitions_nonzero,
            high_branch_cutoff_uA=args.high_branch_cutoff_uA,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "ready" else 2
    if args.command == "count":
        print(len(pd.read_csv(args.manifest)))
        return 0
    if args.command == "run-condition":
        result = run_condition(
            root=args.root, condition_index=args.condition_index, manifest=args.manifest,
            partial_dir=args.partial_dir, step_uA=args.step_uA,
            safety_bound_uA=args.safety_bound_uA, overwrite=args.overwrite,
        )
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
        return 0
    if args.command == "merge":
        result = merge(manifest=args.manifest, partial_dir=args.partial_dir, output_dir=args.output_dir)
        print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
        return 0
    if args.command == "plot-examples":
        result = plot_examples(output_dir=args.output_dir)
        print(json.dumps([str(path) for path in result], indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
