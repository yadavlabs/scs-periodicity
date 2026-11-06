"""Selective, staged migration to the peak-first population I_max.

Only the index axon's I_all/I_max threshold is simulated again. Existing
I_first, volley, and fixed-percentage regime rows are copied without changing
their amplitudes, realized spike counts, or spike times. The merged output is
written under ``results/`` for audit; this tool never overwrites production
tables. I_max is the refined onset of a clear initial response-count peak, or
the strictest stable pre-decline saturation plateau when no clear peak exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from detection.detection_core import (
    ALL_SPIKES_SEARCH,
    INDEX_REGIMES_CSV,
    PREPAD_MS,
    PROJECT_ROOT,
    REGIME_COLUMNS,
    PopulationMRGRunner,
    _atomic_dataframe_csv,
    _config_path,
    _find_cached_peak_first_threshold,
    _repetition_equivalence_groups,
    _select_peak_first_imax_coarse,
    identify_index_axon,
    index_regime_conditions,
    load_population_rep,
    prepend_zeros_ms,
    trim_tail_window_ms,
)
from detection.protocol import (
    DETECTION_PROTOCOL_VERSION,
    I_MAX_BRANCH_SEARCH_WIDTH_UA,
    I_MAX_COARSE_STEP_UA,
    I_MAX_DECLINE_WIDTH_UA,
    I_MAX_EARLY_PEAK_MINIMUM_DROP_SPIKES,
    I_MAX_EARLY_PEAK_MINIMUM_RISE_FRACTION,
    I_MAX_EARLY_PEAK_SEARCH_WIDTH_UA,
    I_MAX_EARLY_PEAK_TAIL_SAMPLES,
    I_MAX_PLATEAU_CANDIDATE_WIDTH_UA,
    I_MAX_PLATEAU_CONFIRMATION_WIDTH_UA,
    I_MAX_PLATEAU_TOLERANCE_FRACTIONS,
    POPULATION_THRESHOLD_TOL_UA,
)
from space_time_v.spacetime_core import spacetime_path


UPDATE_VERSION = "peak_first_imax_selective_v3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/detection/peak_first_imax_update"
UPDATE_COLUMNS = (
    "update_version", "detection_protocol_version", "index_axon_id",
    "index_cutline_id", "index_diameter_um", "file", "cv", "freq",
    "rep_index", "rep_label", "n_input_pulses", "thresh_type",
    "old_amp_uA", "old_n_index_spikes", "old_N_max_spikes",
    "I_first_uA", "I_50_uA", "amp_uA", "n_index_spikes",
    "index_spike_times_ms", "N_max_spikes", "selection_method", "target_spikes",
    "coarse_selected_amp_uA", "coarse_selected_spikes",
    "early_peak_detected", "early_peak_amp_uA", "early_peak_spikes",
    "early_peak_rise_spikes", "early_peak_terminal_drop_spikes",
    "plateau_tolerance_fraction", "plateau_tolerance_spikes", "plateau_start_uA",
    "plateau_confirmed_through_uA", "first_sustained_decline_uA",
    "grid_start_uA", "grid_end_uA", "coarse_grid_samples",
    "refinement_performed", "sampled_amplitudes_uA", "sampled_spike_counts",
    "neuron_runs",
)


def _partial_path(partial_dir: Path, cv: float, freq: float) -> Path:
    return partial_dir / f"peak_first_imax_cv{float(cv):.1f}_f{int(float(freq))}.csv"


def _source_path(root: Path, source_csv: Path | None) -> Path:
    return Path(source_csv or root / "data/detection" / INDEX_REGIMES_CSV.name).resolve()


def _read_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Existing index-regime table is required: {path}")
    table = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REGIME_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Index-regime table is missing columns: {missing}")
    keys = ["index_axon_id", "cv", "freq", "rep_index", "thresh_type"]
    if table.duplicated(keys).any():
        raise ValueError("Index-regime source contains duplicate regime rows")
    return table


def prepare(
    *, root: Path = PROJECT_ROOT, source_csv: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Validate the old table and write the fixed 55-condition manifest."""
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    source = _read_source(_source_path(root, source_csv))
    required_regimes = {"I_first", "I_50", "I_all"}
    found = set(source["thresh_type"].astype(str))
    if not required_regimes.issubset(found):
        raise ValueError(f"Source lacks required regimes: {sorted(required_regimes - found)}")
    protocols = sorted(source["detection_protocol_version"].astype(str).unique())
    if len(protocols) != 1:
        raise ValueError(f"Source contains multiple protocol versions: {protocols}")

    rows = []
    for condition_index, (cv, freq) in enumerate(index_regime_conditions()):
        panel = source[
            np.isclose(pd.to_numeric(source["cv"], errors="coerce"), cv)
            & np.isclose(pd.to_numeric(source["freq"], errors="coerce"), freq)
        ]
        counts = panel.groupby("thresh_type")["rep_index"].nunique().to_dict()
        if any(int(counts.get(regime, 0)) == 0 for regime in required_regimes):
            raise RuntimeError(f"Incomplete source at CV={cv:g}, frequency={freq:g} Hz")
        rows.append({
            "update_version": UPDATE_VERSION,
            "condition_index": int(condition_index),
            "cv": float(cv),
            "freq": float(freq),
            "source_protocol_version": protocols[0],
            "target_protocol_version": DETECTION_PROTOCOL_VERSION,
            "source_repetitions": int(panel["rep_index"].nunique()),
        })
    manifest = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _atomic_dataframe_csv(manifest, output_dir / "conditions.csv")
    payload = {
        "status": "ready", "update_version": UPDATE_VERSION,
        "source_csv": str(_source_path(root, source_csv)),
        "source_protocol_version": protocols[0],
        "target_protocol_version": DETECTION_PROTOCOL_VERSION,
        "conditions": len(manifest), "production_files_modified": False,
        "condition_manifest": str(manifest_path),
        "definition": {
            "selection_order": ["clear initial response-count peak", "stable pre-decline plateau"],
            "coarse_step_uA": I_MAX_COARSE_STEP_UA,
            "branch_search_width_uA": I_MAX_BRANCH_SEARCH_WIDTH_UA,
            "early_peak_search_width_uA": I_MAX_EARLY_PEAK_SEARCH_WIDTH_UA,
            "early_peak_tail_samples": I_MAX_EARLY_PEAK_TAIL_SAMPLES,
            "early_peak_minimum_drop_spikes": I_MAX_EARLY_PEAK_MINIMUM_DROP_SPIKES,
            "early_peak_minimum_rise_fraction": I_MAX_EARLY_PEAK_MINIMUM_RISE_FRACTION,
            "plateau_candidate_width_uA": I_MAX_PLATEAU_CANDIDATE_WIDTH_UA,
            "plateau_confirmation_width_uA": I_MAX_PLATEAU_CONFIRMATION_WIDTH_UA,
            "plateau_tolerance_fractions": list(I_MAX_PLATEAU_TOLERANCE_FRACTIONS),
            "sustained_decline_width_uA": I_MAX_DECLINE_WIDTH_UA,
            "absolute_safety_bound_uA": float(ALL_SPIKES_SEARCH["amp_max_uA"]),
            "threshold_tolerance_uA": POPULATION_THRESHOLD_TOL_UA,
            "uses_2000_uA_endpoint_or_global_sampled_maximum": False,
            "retains_all_tested_amplitudes_and_counts": True,
        },
    }
    temporary = output_dir / "preflight.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "preflight.json")
    return payload


def _run_one(
    root: Path, index: dict[str, Any], cv: float, freq: float,
    rep_index: int, I_first_uA: float, I_50_uA: float, n_input_pulses: int,
) -> dict[str, Any]:
    path = spacetime_path(
        root / "data/space-time-V", str(index["cutline_id"]),
        float(index["diameter_um"]), cv, freq, axon_id=str(index["axon_id"]),
    )
    rep = load_population_rep(path, rep_index)
    config = _config_path(root, rep["diameter_um"], rep["n_valid_nodes"])
    time_trim, voltage_trim = trim_tail_window_ms(rep["time_ms"], rep["V_base"], rep["current"])
    time_trim, voltage_trim = prepend_zeros_ms(time_trim, voltage_trim, prepad_ms=PREPAD_MS)
    runner = PopulationMRGRunner(
        time_trim, voltage_trim, config_path=config,
        readout_node_number=rep["readout_node_number"],
    )
    cache: dict[float, tuple[np.ndarray, int]] = {}

    def count(amplitude: float) -> int:
        amplitude = float(amplitude)
        if amplitude not in cache:
            spikes, n_spikes = runner.run_mrg_for_amp(amplitude)
            cache[amplitude] = (np.asarray(spikes, dtype=float), int(n_spikes))
        return cache[amplitude][1]

    amplitude, realized_count, metadata = _find_cached_peak_first_threshold(
        count,
        cache,
        scan_start_uA=float(I_50_uA),
        n_input_pulses=int(n_input_pulses),
        tol_uA=float(POPULATION_THRESHOLD_TOL_UA),
        safety_bound_uA=float(ALL_SPIKES_SEARCH["amp_max_uA"]),
    )
    spikes, n_spikes = cache[float(amplitude)]
    if int(n_spikes) != int(realized_count):
        raise RuntimeError("Cached refined I_max response count changed unexpectedly")
    peak = metadata["peak"]
    plateau = metadata["plateau"]
    sampled = sorted((float(amp), int(value[1])) for amp, value in cache.items())
    return {
        "rep_index": int(rep_index), "rep_label": str(rep["rep_label"]),
        "amp_uA": float(amplitude), "n_index_spikes": int(n_spikes),
        "index_spike_times_ms": json.dumps(spikes.tolist()),
        "N_max_spikes": int(n_spikes),
        "selection_method": str(metadata["selection_method"]),
        "target_spikes": int(metadata["target_spikes"]),
        "coarse_selected_amp_uA": float(metadata["coarse_selected_amplitude_uA"]),
        "coarse_selected_spikes": int(metadata["coarse_selected_spikes"]),
        "early_peak_detected": bool(peak["is_clear_peak"]),
        "early_peak_amp_uA": float(peak["peak_amplitude_uA"]),
        "early_peak_spikes": int(peak["peak_count"]),
        "early_peak_rise_spikes": int(peak["rise_spikes"]),
        "early_peak_terminal_drop_spikes": float(peak["terminal_drop_spikes"]),
        "plateau_tolerance_fraction": np.nan if plateau is None else float(plateau["tolerance_fraction"]),
        "plateau_tolerance_spikes": np.nan if plateau is None else int(plateau["tolerance_spikes"]),
        "plateau_start_uA": np.nan if plateau is None else float(plateau["plateau_start_uA"]),
        "plateau_confirmed_through_uA": (
            np.nan if plateau is None else float(plateau["plateau_confirmed_through_uA"])
        ),
        "first_sustained_decline_uA": (
            np.nan if plateau is None else float(plateau["first_sustained_decline_uA"])
        ),
        "grid_start_uA": float(metadata["grid_start_uA"]),
        "grid_end_uA": float(metadata["grid_end_uA"]),
        "coarse_grid_samples": int(metadata["coarse_grid_samples"]),
        "refinement_performed": bool(metadata["refinement_performed"]),
        "sampled_amplitudes_uA": json.dumps([amplitude for amplitude, _ in sampled]),
        "sampled_spike_counts": json.dumps([spike_count for _, spike_count in sampled]),
        "neuron_runs": int(len(cache)),
    }


def _threshold_lookup(panel: pd.DataFrame, regime: str) -> dict[int, float]:
    rows = panel[panel["thresh_type"].eq(regime)]
    return {int(row.rep_index): float(row.amp_uA) for row in rows.itertuples(index=False)}


def _completed_partial_is_compatible(path: Path) -> bool:
    """Reject stale partials from a different I_max definition."""
    try:
        table = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    return bool(
        not table.empty
        and list(table.columns) == list(UPDATE_COLUMNS)
        and table["update_version"].astype(str).eq(UPDATE_VERSION).all()
        and table["detection_protocol_version"].astype(str).eq(DETECTION_PROTOCOL_VERSION).all()
    )


def _retained_selection_failure(row: Any) -> str | None:
    """Return why one retained diagnostic sweep fails the current selector."""
    prefix = f"CV={row.cv:g}, {row.freq:g} Hz, rep={row.rep_index}: "
    sampled_amplitudes = np.asarray(json.loads(row.sampled_amplitudes_uA), dtype=float)
    sampled_counts = np.asarray(json.loads(row.sampled_spike_counts), dtype=int)
    if sampled_amplitudes.size != sampled_counts.size:
        return prefix + "unequal retained sweep lengths"
    grid = np.linspace(
        float(row.grid_start_uA), float(row.grid_end_uA), int(row.coarse_grid_samples),
    )
    grid_counts: list[int] = []
    for amplitude in grid:
        matches = np.flatnonzero(np.isclose(sampled_amplitudes, amplitude, atol=1e-8, rtol=0.0))
        if matches.size != 1:
            return prefix + f"retained sweep lacks unique coarse amplitude {amplitude:g} uA"
        grid_counts.append(int(sampled_counts[int(matches[0])]))
    selected = _select_peak_first_imax_coarse(
        grid, grid_counts, n_input_pulses=int(row.n_input_pulses),
    )
    comparisons = (
        (str(row.selection_method), str(selected["selection_method"]), "selection method"),
        (int(row.target_spikes), int(selected["target_spikes"]), "target count"),
        (
            float(row.coarse_selected_amp_uA),
            float(selected["coarse_selected_amplitude_uA"]),
            "coarse amplitude",
        ),
        (
            int(row.coarse_selected_spikes),
            int(selected["coarse_selected_spikes"]),
            "coarse response count",
        ),
    )
    for retained, current, label in comparisons:
        equal = (
            np.isclose(retained, current, atol=1e-8, rtol=0.0)
            if isinstance(retained, float) else retained == current
        )
        if not equal:
            return prefix + f"retained {label}={retained!r}, current rule={current!r}"
    return None


def _validate_retained_sweep_selections(updates: pd.DataFrame) -> None:
    """Prove inherited update rows still implement the current coarse rule."""
    failures = [
        failure for row in updates.itertuples(index=False)
        if (failure := _retained_selection_failure(row)) is not None
    ]
    if failures:
        preview = "; ".join(failures[:5])
        raise RuntimeError(
            f"{len(failures)} retained I_max update rows do not implement {UPDATE_VERSION}: {preview}"
        )


def run_condition(
    *, root: Path = PROJECT_ROOT, condition_index: int, n_jobs: int = 1,
    source_csv: Path | None = None, partial_dir: Path,
    overwrite: bool = False, base_updates_csv: Path | None = None,
) -> Path:
    """Recompute I_max for every repetition of one CV/frequency condition."""
    root, partial_dir = Path(root).resolve(), Path(partial_dir).resolve()
    conditions = index_regime_conditions()
    if not 0 <= int(condition_index) < len(conditions):
        raise IndexError(f"condition_index must be in [0, {len(conditions) - 1}]")
    cv, freq = conditions[int(condition_index)]
    output = _partial_path(partial_dir, cv, freq)
    if output.is_file() and not overwrite:
        if _completed_partial_is_compatible(output):
            return output
        raise RuntimeError(
            f"Existing partial is incompatible with {UPDATE_VERSION}: {output}. "
            "Use a fresh output directory or pass --overwrite."
        )

    source = _read_source(_source_path(root, source_csv))
    panel = source[
        np.isclose(source["cv"].astype(float), cv)
        & np.isclose(source["freq"].astype(float), freq)
    ].copy()
    old_rows = panel[panel["thresh_type"].eq("I_all")].set_index("rep_index", drop=False)
    first = _threshold_lookup(panel, "I_first")
    fifty = _threshold_lookup(panel, "I_50")
    if set(old_rows.index.astype(int)) != set(first) or set(first) != set(fifty):
        raise RuntimeError(f"I_first/I_50/I_all repetition sets disagree at CV={cv:g}, {freq:g} Hz")

    index = identify_index_axon(root=root, n_jobs=1, force=False)
    path = spacetime_path(
        root / "data/space-time-V", str(index["cutline_id"]),
        float(index["diameter_um"]), cv, freq, axon_id=str(index["axon_id"]),
    )
    retained_rows: list[dict[str, Any]] = []
    invalid_repetitions: set[int] | None = None
    if base_updates_csv is not None:
        base_updates_path = Path(base_updates_csv).resolve()
        if not base_updates_path.is_file():
            raise FileNotFoundError(f"Base I_max updates are required: {base_updates_path}")
        base = pd.read_csv(base_updates_path, low_memory=False)
        missing_columns = sorted(set(UPDATE_COLUMNS) - set(base.columns))
        if missing_columns:
            raise RuntimeError(f"Base I_max updates are missing columns: {missing_columns}")
        base_panel = base[
            np.isclose(base["cv"].astype(float), cv)
            & np.isclose(base["freq"].astype(float), freq)
        ].copy()
        if set(base_panel["rep_index"].astype(int)) != set(old_rows.index.astype(int)):
            raise RuntimeError(f"Base update repetition set is incomplete at CV={cv:g}, {freq:g} Hz")
        invalid_repetitions = {
            int(row.rep_index) for row in base_panel.itertuples(index=False)
            if _retained_selection_failure(row) is not None
        }
        retained = base_panel[~base_panel["rep_index"].astype(int).isin(invalid_repetitions)].copy()
        retained["update_version"] = UPDATE_VERSION
        retained["detection_protocol_version"] = DETECTION_PROTOCOL_VERSION
        retained_rows = retained.loc[:, UPDATE_COLUMNS].to_dict("records")

    work: list[tuple[list[tuple[int, str]], int]] = []
    for waveform_members in _repetition_equivalence_groups(path):
        by_start: dict[tuple[float, float], list[tuple[int, str]]] = {}
        for rep_index, rep_label in waveform_members:
            signature = (first[int(rep_index)], fifty[int(rep_index)])
            by_start.setdefault(signature, []).append((int(rep_index), str(rep_label)))
        for members in by_start.values():
            if invalid_repetitions is None or any(rep in invalid_repetitions for rep, _ in members):
                work.append((members, members[0][0]))

    computed = Parallel(n_jobs=max(1, int(n_jobs)), prefer="processes")(
        delayed(_run_one)(
            root, index, cv, freq, representative, first[representative], fifty[representative],
            int(old_rows.loc[int(representative), "n_input_pulses"]),
        )
        for _, representative in work
    )
    rows = list(retained_rows)
    for (members, representative), result in zip(work, computed):
        for rep_index, rep_label in members:
            old = old_rows.loc[int(rep_index)]
            rows.append({
                "update_version": UPDATE_VERSION,
                "detection_protocol_version": DETECTION_PROTOCOL_VERSION,
                "index_axon_id": str(old["index_axon_id"]),
                "index_cutline_id": str(old["index_cutline_id"]),
                "index_diameter_um": float(old["index_diameter_um"]),
                "file": str(old["file"]), "cv": float(cv), "freq": float(freq),
                "rep_index": int(rep_index), "rep_label": str(rep_label),
                "n_input_pulses": int(old["n_input_pulses"]), "thresh_type": "I_all",
                "old_amp_uA": float(old["amp_uA"]),
                "old_n_index_spikes": int(old["n_index_spikes"]),
                "old_N_max_spikes": int(old["N_max_spikes"]),
                "I_first_uA": float(first[int(rep_index)]), "I_50_uA": float(fifty[int(rep_index)]),
                **{**result, "rep_index": int(rep_index), "rep_label": str(rep_label)},
            })
    rows.sort(key=lambda row: int(row["rep_index"]))
    result = pd.DataFrame(rows, columns=UPDATE_COLUMNS)
    if len(result) != len(old_rows) or set(result["rep_index"].astype(int)) != set(old_rows.index.astype(int)):
        raise RuntimeError(f"Selective condition rerun is incomplete at CV={cv:g}, {freq:g} Hz")
    _validate_retained_sweep_selections(result)
    return _atomic_dataframe_csv(result, output)


def merge(
    *, root: Path = PROJECT_ROOT, source_csv: Path | None = None,
    partial_dir: Path, output_dir: Path = DEFAULT_OUTPUT_DIR,
    base_updates_csv: Path | None = None,
) -> dict[str, Path]:
    """Create a staged table from all partials or targeted replacements of a prior update."""
    root = Path(root).resolve()
    partial_dir, output_dir = Path(partial_dir).resolve(), Path(output_dir).resolve()
    source_path = _source_path(root, source_csv)
    source = _read_source(source_path)
    expected = [_partial_path(partial_dir, cv, freq) for cv, freq in index_regime_conditions()]
    available = [path for path in expected if path.is_file()]
    missing = [str(path) for path in expected if not path.is_file()]
    if base_updates_csv is None and missing:
        raise RuntimeError(
            f"Cannot merge peak-first I_max update; {len(missing)} partials are missing "
            f"(first: {missing[0]})"
        )
    if base_updates_csv is not None and not available:
        raise RuntimeError("Targeted merge requires at least one replacement partial")

    replacements = pd.concat([pd.read_csv(path, low_memory=False) for path in available], ignore_index=True)
    missing_columns = sorted(set(UPDATE_COLUMNS) - set(replacements.columns))
    if missing_columns:
        raise RuntimeError(f"Peak-first I_max partials are missing columns: {missing_columns}")
    if not replacements["update_version"].astype(str).eq(UPDATE_VERSION).all():
        raise RuntimeError("Peak-first I_max partials contain a stale update version")
    if not replacements["detection_protocol_version"].astype(str).eq(DETECTION_PROTOCOL_VERSION).all():
        raise RuntimeError("Peak-first I_max partials contain a stale protocol version")

    keys = ["index_axon_id", "cv", "freq", "rep_index"]
    expected_imax = source[source["thresh_type"].eq("I_all")]
    replaced_cells = replacements[["cv", "freq"]].drop_duplicates().sort_values(["cv", "freq"])
    for cell in replaced_cells.itertuples(index=False):
        expected_reps = set(expected_imax[
            np.isclose(expected_imax["cv"].astype(float), float(cell.cv))
            & np.isclose(expected_imax["freq"].astype(float), float(cell.freq))
        ]["rep_index"].astype(int))
        actual_reps = set(replacements[
            np.isclose(replacements["cv"].astype(float), float(cell.cv))
            & np.isclose(replacements["freq"].astype(float), float(cell.freq))
        ]["rep_index"].astype(int))
        if actual_reps != expected_reps:
            raise RuntimeError(
                f"Replacement partial is incomplete at CV={cell.cv:g}, {cell.freq:g} Hz"
            )

    selector_invalid_base_rows = 0
    if base_updates_csv is None:
        updates = replacements
    else:
        base_updates_path = Path(base_updates_csv).resolve()
        if not base_updates_path.is_file():
            raise FileNotFoundError(f"Base I_max updates are required: {base_updates_path}")
        base = pd.read_csv(base_updates_path, low_memory=False)
        base_missing = sorted(set(UPDATE_COLUMNS) - set(base.columns))
        if base_missing:
            raise RuntimeError(f"Base I_max updates are missing columns: {base_missing}")
        replace_index = pd.MultiIndex.from_frame(replaced_cells[["cv", "freq"]])
        base_cell_index = pd.MultiIndex.from_frame(base[["cv", "freq"]])
        selector_invalid_base_rows = sum(
            _retained_selection_failure(row) is not None
            for row in base.loc[base_cell_index.isin(replace_index)].itertuples(index=False)
        )
        inherited = base.loc[~base_cell_index.isin(replace_index)].copy()
        updates = pd.concat([inherited, replacements], ignore_index=True)
        _validate_retained_sweep_selections(updates)

    updates["update_version"] = UPDATE_VERSION
    updates["detection_protocol_version"] = DETECTION_PROTOCOL_VERSION
    updates = updates.loc[:, UPDATE_COLUMNS]
    if (updates["n_index_spikes"].astype(int) < updates["target_spikes"].astype(int)).any():
        raise RuntimeError("At least one refined I_max response is below its selected target")
    if (updates["amp_uA"].astype(float) > updates["grid_end_uA"].astype(float) + 1e-9).any():
        raise RuntimeError("At least one refined I_max amplitude lies beyond its initial-branch grid")
    if updates.duplicated(keys).any():
        raise RuntimeError("Peak-first I_max partials contain duplicate repetition rows")
    expected_keys = source[source["thresh_type"].eq("I_all")][keys]
    actual_keys = updates[keys]
    if len(actual_keys) != len(expected_keys) or len(actual_keys.merge(expected_keys.drop_duplicates())) != len(expected_keys):
        raise RuntimeError("Peak-first I_max updates do not exactly cover the source I_all rows")

    merged = source.copy()
    before_fixed = source[~source["thresh_type"].eq("I_all")].reset_index(drop=True)
    update_map = updates.set_index(keys)
    row_keys = pd.MultiIndex.from_frame(merged[keys])
    mapped_nmax = update_map["N_max_spikes"].reindex(row_keys).to_numpy()
    if pd.isna(mapped_nmax).any():
        raise RuntimeError("At least one regime row lacks an updated I_max count")
    merged["N_max_spikes"] = mapped_nmax.astype(int)
    merged["detection_protocol_version"] = DETECTION_PROTOCOL_VERSION

    imax_mask = merged["thresh_type"].eq("I_all")
    imax_keys = pd.MultiIndex.from_frame(merged.loc[imax_mask, keys])
    aligned = update_map.reindex(imax_keys)
    for column in ("amp_uA", "n_index_spikes", "index_spike_times_ms"):
        merged.loc[imax_mask, column] = aligned[column].to_numpy()

    after_fixed = merged[~merged["thresh_type"].eq("I_all")].reset_index(drop=True)
    invariant_columns = [
        column for column in REGIME_COLUMNS
        if column not in {"detection_protocol_version", "N_max_spikes"}
    ]
    if not before_fixed[invariant_columns].equals(after_fixed[invariant_columns]):
        raise RuntimeError("Non-I_max regime content changed during the selective merge")

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_path = _atomic_dataframe_csv(merged.loc[:, REGIME_COLUMNS], output_dir / INDEX_REGIMES_CSV.name)
    update_path = _atomic_dataframe_csv(updates.loc[:, UPDATE_COLUMNS], output_dir / "peak_first_imax_updates.csv")
    audit = (
        updates.assign(
            amplitude_change_uA=updates["amp_uA"] - updates["old_amp_uA"],
            count_change=updates["n_index_spikes"] - updates["old_n_index_spikes"],
            changed=~np.isclose(updates["amp_uA"], updates["old_amp_uA"], atol=POPULATION_THRESHOLD_TOL_UA),
        )
        .groupby(["freq", "cv"], as_index=False)
        .agg(
            repetitions=("rep_index", "size"), changed_repetitions=("changed", "sum"),
            early_peak_repetitions=(
                "selection_method", lambda values: int(values.eq("early_response_peak").sum())
            ),
            plateau_repetitions=(
                "selection_method", lambda values: int(values.eq("stable_saturation_plateau").sum())
            ),
            old_median_uA=("old_amp_uA", "median"), new_median_uA=("amp_uA", "median"),
            minimum_new_uA=("amp_uA", "min"), maximum_new_uA=("amp_uA", "max"),
            median_selected_spikes=("n_index_spikes", "median"),
            median_change_uA=("amplitude_change_uA", "median"),
            maximum_neuron_runs=("neuron_runs", "max"),
        )
    )
    audit_path = _atomic_dataframe_csv(audit, output_dir / "peak_first_imax_audit_summary.csv")
    manifest_path = output_dir / "merge_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "status": "complete", "update_version": UPDATE_VERSION,
        "source_csv": str(source_path), "staged_index_regimes_csv": str(staged_path),
        "updates_csv": str(update_path), "audit_summary_csv": str(audit_path),
        "base_updates_csv": None if base_updates_csv is None else str(Path(base_updates_csv).resolve()),
        "replacement_cells": [
            {"cv": float(row.cv), "freq": float(row.freq)}
            for row in replaced_cells.itertuples(index=False)
        ],
        "target_protocol_version": DETECTION_PROTOCOL_VERSION,
        "rows": int(len(merged)),
        "imax_rows_in_update_table": int(len(updates)),
        "replacement_partial_rows": int(len(replacements)),
        "imax_rows_recomputed": (
            int(len(updates)) if base_updates_csv is None else int(selector_invalid_base_rows)
        ),
        "early_peak_repetitions": int(updates["selection_method"].eq("early_response_peak").sum()),
        "plateau_repetitions": int(
            updates["selection_method"].eq("stable_saturation_plateau").sum()
        ),
        "non_imax_rows_resimulated": 0, "non_imax_regime_values_unchanged": True,
        "production_files_modified": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return {
        "staged_index_regimes": staged_path, "updates": update_path,
        "audit_summary": audit_path, "manifest": manifest_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=PROJECT_ROOT)
    common.add_argument("--source-csv", type=Path, default=None)

    prepare_parser = sub.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    count_parser = sub.add_parser("count")

    run_parser = sub.add_parser("run-condition", parents=[common])
    run_parser.add_argument("--condition-index", type=int, required=True)
    run_parser.add_argument("--n-jobs", type=int, default=1)
    run_parser.add_argument("--partial-dir", type=Path, required=True)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--base-updates-csv", type=Path, default=None)

    merge_parser = sub.add_parser("merge", parents=[common])
    merge_parser.add_argument("--partial-dir", type=Path, required=True)
    merge_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    merge_parser.add_argument("--base-updates-csv", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(root=args.root, source_csv=args.source_csv, output_dir=args.output_dir), indent=2))
    elif args.command == "count":
        print(len(index_regime_conditions()))
    elif args.command == "run-condition":
        print(run_condition(
            root=args.root, condition_index=args.condition_index, n_jobs=args.n_jobs,
            source_csv=args.source_csv, partial_dir=args.partial_dir, overwrite=args.overwrite,
            base_updates_csv=args.base_updates_csv,
        ))
    elif args.command == "merge":
        print(json.dumps({key: str(value) for key, value in merge(
            root=args.root, source_csv=args.source_csv, partial_dir=args.partial_dir,
            output_dir=args.output_dir, base_updates_csv=args.base_updates_csv,
        ).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
