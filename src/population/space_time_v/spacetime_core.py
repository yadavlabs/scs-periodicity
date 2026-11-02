"""spacetime_core.py

Core utilities for heterogeneous-population SpaceTimeV generation:
- interpolate manually exported COMSOL cutlines onto MRG nodes;
- generate axon-specific Ve bases and compatible SpaceTimeV bundles;
- prepare one fresh MRG/NEURON geometry worker per fiber diameter.

This module is imported from population notebooks:

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from population.space_time_v.spacetime_core import generate_population_ve_bases
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import warnings
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import interpolate


# ---------------- Configuration ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POPULATION_CUTLINE_IDS = (
    *(f"S{i:02d}" for i in range(1, 14)),
    *(f"M{i:02d}" for i in range(1, 14)),
    *(f"D{i:02d}" for i in range(5, 14)),
)
POPULATION_DIAMETERS_UM = (5.7, 8.7, 10.0, 11.5, 12.8, 14.0)
POPULATION_PHASE_IDS = ("ph0", "ph1")
POPULATION_PHASE_SCHEME = "balanced-antithetic-v1"
POPULATION_PHASE_FRACTIONS = (0.25, 0.75)
POPULATION_CV_VALS = tuple(round(0.1 * i, 1) for i in range(11))
POPULATION_FREQ_VALS_HZ = (10, 20, 50, 100, 200)

# Original discrete MRG ``deltax`` values, expressed as node-to-node spacing.
MRG_NODE_SPACING_MM = {
    5.7: 0.500,
    8.7: 1.000,
    10.0: 1.150,
    11.5: 1.250,
    12.8: 1.350,
    14.0: 1.400,
}


def discover_comsol_cutline_ids(comsol_dir: Path) -> tuple[str, ...]:
    """Discover only cutline IDs having both manual COMSOL polarity exports."""
    negative = {p.name.removesuffix("_1uA_bcs1_neg.txt") for p in comsol_dir.glob("*_1uA_bcs1_neg.txt")}
    positive = {p.name.removesuffix("_1uA_bcs1_pos.txt") for p in comsol_dir.glob("*_1uA_bcs1_pos.txt")}
    complete = tuple(sorted(negative & positive))
    if not complete:
        raise FileNotFoundError(f"No complete COMSOL cutline pairs found in {comsol_dir}")
    unmatched = sorted(negative ^ positive)
    if unmatched:
        warnings.warn(f"Ignoring {len(unmatched)} unpaired COMSOL cutline export(s): {unmatched[:3]}")
    return complete


def production_comsol_cutline_ids(comsol_dir: Path) -> tuple[str, ...]:
    """Validate and return the exact 35-trajectory production cohort."""
    observed = discover_comsol_cutline_ids(comsol_dir)
    missing = sorted(set(POPULATION_CUTLINE_IDS) - set(observed))
    unexpected = sorted(set(observed) - set(POPULATION_CUTLINE_IDS))
    if missing or unexpected:
        raise RuntimeError(
            "Population COMSOL cohort must contain exactly S01-S13, M01-M13, "
            f"and D05-D13; missing={missing}, unexpected={unexpected}"
        )
    return POPULATION_CUTLINE_IDS

SPACETIME_NPZ_FIELDS = frozenset({
    "time_s", "node_numbers", "rep_labels", "current", "V", "axon_id",
    "cutline_id", "diameter_um", "node_arc_length_mm",
    "node_arc_length_requested_mm", "node_spacing_um", "n_valid_nodes",
    "readout_node_number", "phase_id", "phase_fraction", "phase_offset_mm",
    "phase_scheme",
})


def _npz_has_fields(path: Path, required_fields: frozenset[str]) -> bool:
    """Return whether *path* is an openable NPZ with all required fields."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return required_fields.issubset(archive.files)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        return False


def _spacetime_matches_job(path: Path, job: dict[str, object]) -> bool:
    """Return whether an archive belongs to the current phase-aware cohort."""
    if not _npz_has_fields(path, SPACETIME_NPZ_FIELDS):
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                str(archive["axon_id"].item()) == str(job["axon_id"])
                and str(archive["cutline_id"].item()) == str(job["cutline_id"])
                and np.isclose(float(archive["diameter_um"].item()), float(job["diameter_um"]))
                and str(archive["phase_id"].item()) == str(job["phase_id"])
                and np.isclose(float(archive["phase_fraction"].item()), float(job["phase_fraction"]))
                and str(archive["phase_scheme"].item()) == str(job["phase_scheme"])
            )
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        return False


def _atomic_savez_compressed(path: Path, **arrays: object) -> None:
    """Write a compressed NPZ without ever exposing a partial target file.

    The temporary archive is written and flushed in the destination directory,
    checked for a readable ZIP directory and expected fields, and then atomically
    replaces the destination. An interrupted write can therefore leave only an
    ignorable temporary file; an existing valid target remains intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())

        if not _npz_has_fields(temporary_path, frozenset(arrays)):
            raise RuntimeError(f"New NPZ failed integrity validation: {temporary_path}")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# --------------- Utility: identifiers and COMSOL profiles --------------- #

def diameter_token(diameter_um: float) -> str:
    """Return a filename-safe MRG diameter token (for example, 5.7 -> ``57``)."""
    return str(int(round(float(diameter_um) * 10)))


def make_axon_id(cutline_id: str, diameter_um: float, phase_id: str | None = None) -> str:
    """Return the stable population identifier for a trajectory/fiber realization."""
    base = f"{cutline_id}_{diameter_token(diameter_um)}um"
    return base if phase_id is None else f"{base}_{phase_id}"


def population_phase_assignments(cutline_ids: Iterable[str]) -> dict[str, tuple[dict[str, object], ...]]:
    """Return the same two complementary nodal phases at every trajectory.

    Common 0.25/0.75-internode levels form a balanced location x diameter x
    phase design. They avoid placing a node exactly on the artificial COMSOL
    boundary and prevent nodal phase from being confounded with trajectory.
    """
    ids = tuple(sorted({str(cutline_id) for cutline_id in cutline_ids}))
    if not ids:
        raise ValueError("At least one cutline is required for phase assignment")
    return {
        cutline_id: tuple(
            {"phase_id": phase_id, "phase_fraction": phase_fraction}
            for phase_id, phase_fraction in zip(POPULATION_PHASE_IDS, POPULATION_PHASE_FRACTIONS)
        )
        for cutline_id in ids
    }


def load_comsol_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one manual COMSOL line export as arc length (mm) and Ve (mV)."""
    raw = np.loadtxt(path, skiprows=8, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"Expected two numeric columns in {path}; got {raw.shape}")
    arc_length_mm = np.asarray(raw[:, 0], dtype=float)
    ve_mV = np.asarray(raw[:, 1], dtype=float) * 1000.0
    order = np.argsort(arc_length_mm)
    arc_length_mm, ve_mV = arc_length_mm[order], ve_mV[order]
    arc_length_mm, unique_idx = np.unique(arc_length_mm, return_index=True)
    ve_mV = ve_mV[unique_idx]
    if arc_length_mm.size < 4 or not np.all(np.diff(arc_length_mm) > 0):
        raise ValueError(f"COMSOL arc-length samples must be unique and increasing: {path}")
    return arc_length_mm, ve_mV


def _basis_path(
    out_dir: Path, cutline_id: str, diameter_um: float, phase_id: str | None = None,
) -> Path:
    return out_dir / f"{make_axon_id(cutline_id, diameter_um, phase_id)}_1uA_VeBasis.npz"


def _node_arc_lengths_mm(
    diameter_um: float, arc_min_mm: float, arc_max_mm: float,
    phase_fraction: float = 0.0,
) -> np.ndarray:
    """Return all complete MRG node positions inside the COMSOL domain."""
    diameter_um = round(float(diameter_um), 1)
    try:
        spacing_mm = MRG_NODE_SPACING_MM[diameter_um]
    except KeyError as exc:
        raise ValueError(f"Unsupported MRG diameter: {diameter_um} um") from exc
    phase_fraction = float(phase_fraction)
    if not 0.0 <= phase_fraction < 1.0:
        raise ValueError("phase_fraction must be in [0, 1)")
    first = float(arc_max_mm) - phase_fraction * spacing_mm
    n_nodes = int(np.floor((first - float(arc_min_mm)) / spacing_mm + 1e-9)) + 1
    if n_nodes < 1:
        raise ValueError("COMSOL arc-length extent cannot fit an MRG node")
    # Descending arc length retains the established single-axon orientation.
    return first - np.arange(n_nodes, dtype=float) * spacing_mm


# --------------- Ve-basis generation --------------- #

def generate_ve_basis(
    cutline_id: str,
    diameter_um: float,
    *,
    phase_id: str | None = None,
    phase_fraction: float = 0.0,
    phase_scheme: str | None = None,
    comsol_dir: Path,
    out_dir: Path,
) -> dict[str, object]:
    """Interpolate one cutline onto one diameter-specific MRG node basis.

    Every complete MRG interval within the shared positive/negative cutline
    extent is retained. No profile is extrapolated or terminal-value padded.
    """
    diameter_um = round(float(diameter_um), 1)
    neg_path = comsol_dir / f"{cutline_id}_1uA_bcs1_neg.txt"
    pos_path = comsol_dir / f"{cutline_id}_1uA_bcs1_pos.txt"
    if not neg_path.is_file() or not pos_path.is_file():
        raise FileNotFoundError(f"Missing COMSOL export pair for {cutline_id}")
    x_neg, ve_neg = load_comsol_profile(neg_path)
    x_pos, ve_pos = load_comsol_profile(pos_path)
    common_min = max(float(x_neg.min()), float(x_pos.min()))
    common_max = min(float(x_neg.max()), float(x_pos.max()))
    if common_min >= common_max:
        raise ValueError(f"No shared COMSOL arc-length range for {cutline_id}")
    node_arc_mm = _node_arc_lengths_mm(
        diameter_um, common_min, common_max, phase_fraction=phase_fraction,
    )

    # This matches the established single-axon interpolation. The positions
    # above are constrained, so bounds_error cannot cause extrapolation.
    neg_interp = interpolate.interp1d(x_neg, ve_neg, kind="cubic", bounds_error=True)
    pos_interp = interpolate.interp1d(x_pos, ve_pos, kind="cubic", bounds_error=True)
    v_neg_mV = np.asarray(neg_interp(node_arc_mm), dtype=np.float32)
    v_pos_mV = np.asarray(pos_interp(node_arc_mm), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _basis_path(out_dir, cutline_id, diameter_um, phase_id)
    axon_id = make_axon_id(cutline_id, diameter_um, phase_id)
    resolved_phase_scheme = phase_scheme or (
        "legacy-single-phase" if phase_id is None else POPULATION_PHASE_SCHEME
    )
    _atomic_savez_compressed(
        out_path,
        schema_version=np.asarray("population-ve-basis-v5-phase"),
        axon_id=np.asarray(axon_id),
        cutline_id=np.asarray(cutline_id),
        diameter_um=np.asarray(diameter_um, dtype=np.float32),
        phase_id=np.asarray("legacy" if phase_id is None else phase_id),
        phase_fraction=np.asarray(float(phase_fraction), dtype=np.float32),
        phase_offset_mm=np.asarray(
            float(phase_fraction) * MRG_NODE_SPACING_MM[diameter_um], dtype=np.float32,
        ),
        phase_scheme=np.asarray(resolved_phase_scheme),
        node_numbers=np.arange(node_arc_mm.size, dtype=np.int16),
        node_arc_length_mm=node_arc_mm.astype(np.float32),
        node_arc_length_requested_mm=node_arc_mm.astype(np.float32),
        node_spacing_um=np.asarray(MRG_NODE_SPACING_MM[diameter_um] * 1000.0, dtype=np.float32),
        V_neg_mV=v_neg_mV,
        V_pos_mV=v_pos_mV,
        comsol_arc_min_mm=np.asarray(common_min, dtype=np.float32),
        comsol_arc_max_mm=np.asarray(common_max, dtype=np.float32),
        n_valid_nodes=np.asarray(node_arc_mm.size, dtype=np.int16),
        readout_node_number=np.asarray(node_arc_mm.size - 1, dtype=np.int16),
    )
    return {
        "axon_id": axon_id,
        "cutline_id": cutline_id,
        "diameter_um": diameter_um,
        "phase_id": "legacy" if phase_id is None else phase_id,
        "phase_fraction": float(phase_fraction),
        "phase_offset_mm": float(phase_fraction) * MRG_NODE_SPACING_MM[diameter_um],
        "phase_scheme": resolved_phase_scheme,
        "node_spacing_um": MRG_NODE_SPACING_MM[diameter_um] * 1000.0,
        "n_valid_nodes": int(node_arc_mm.size),
        "readout_node_number": int(node_arc_mm.size - 1),
        "basis_npz": out_path,
    }


def generate_population_ve_bases(
    cutline_ids: Iterable[str],
    diameters_um: Iterable[float],
    *,
    comsol_dir: Path,
    out_dir: Path,
) -> list[dict[str, object]]:
    """Generate two phase-stratified bases for every cutline/diameter pair."""
    cutline_ids = tuple(cutline_ids)
    phase_assignments = population_phase_assignments(cutline_ids)
    rows = [
        generate_ve_basis(
            cutline_id,
            diameter_um,
            phase_id=str(phase["phase_id"]),
            phase_fraction=float(phase["phase_fraction"]),
            comsol_dir=comsol_dir,
            out_dir=out_dir,
        )
        for cutline_id in cutline_ids
        for diameter_um in diameters_um
        for phase in phase_assignments[cutline_id]
    ]
    if not rows:
        raise ValueError("At least one cutline and one diameter are required")
    manifest = out_dir / "axon_basis_manifest.csv"
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            basis_path = Path(row["basis_npz"]).resolve()
            try:
                basis_reference = basis_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                basis_reference = str(basis_path)
            writer.writerow({**row, "basis_npz": basis_reference})
    temporary.replace(manifest)
    return rows


# --------------- SpaceTimeV generation --------------- #

def spacetime_path(
    out_dir: Path,
    cutline_id: str,
    diameter_um: float,
    cv: float,
    freq_hz: float,
    phase_id: str | None = None,
    *,
    axon_id: str | None = None,
) -> Path:
    """Return the single-format population SpaceTimeV output path."""
    resolved_axon_id = axon_id or make_axon_id(cutline_id, diameter_um, phase_id)
    return out_dir / (
        f"{resolved_axon_id}_1uA_"
        f"cv{float(cv):.1f}_f{int(freq_hz)}_SpaceTimeV.npz"
    )


def generate_spacetime_from_basis(
    basis_path: Path,
    *,
    cv: float,
    freq_hz: float,
    time_pattern_dir: Path,
    out_dir: Path,
    fs_hz: float = 20_000.0,
) -> Path:
    """Apply a shared waveform to one axon's +/-1 uA Ve basis.

    The primary ``time_s``, ``node_numbers``, ``rep_labels``, ``current``, and
    ``V`` fields match the single-axon SpaceTimeV format; new metadata is
    additive.
    """
    with np.load(basis_path, allow_pickle=False) as basis:
        axon_id = basis["axon_id"].item()
        cutline_id = basis["cutline_id"].item()
        diameter_um = float(basis["diameter_um"])
        phase_id = str(basis["phase_id"].item()) if "phase_id" in basis else "legacy"
        phase_fraction = float(basis["phase_fraction"].item()) if "phase_fraction" in basis else 0.0
        phase_offset_mm = float(basis["phase_offset_mm"].item()) if "phase_offset_mm" in basis else 0.0
        phase_scheme = str(basis["phase_scheme"].item()) if "phase_scheme" in basis else "legacy-single-phase"
        node_numbers = basis["node_numbers"]
        node_arc_length_mm = basis["node_arc_length_mm"]
        requested_arc_length_mm = basis["node_arc_length_requested_mm"]
        node_spacing_um = basis["node_spacing_um"]
        v_neg_mV = basis["V_neg_mV"]
        v_pos_mV = basis["V_pos_mV"]
        n_valid_nodes = basis["n_valid_nodes"]
        readout_node_number = basis["readout_node_number"]

    current_path = time_pattern_dir / f"cv{float(cv):.1f}_f{int(freq_hz)}_waveforms.npz"
    if not current_path.is_file():
        raise FileNotFoundError(f"Population time pattern not found: {current_path}")
    with np.load(current_path, allow_pickle=False) as bundle:
        rep_labels = np.asarray(sorted(bundle.files))
        current = np.stack([np.asarray(bundle[label], dtype=np.float32) for label in rep_labels])
    if np.isclose(float(cv), 0.0):
        regular = [
            index for index, label in enumerate(rep_labels)
            if "regular" in str(label).lower()
        ]
        source_index = regular[0] if regular else 0
        # CV=0 has no random temporal realization. Preserve the 100 historical
        # repetition labels for downstream joins while using one exact-periodic
        # current representation for every repetition.
        current[:] = current[source_index]

    n_reps, n_time = current.shape
    V = np.zeros((n_reps, n_time, int(node_numbers.size)), dtype=np.float32)
    V[current < 0] = v_neg_mV
    V[current > 0] = v_pos_mV
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = spacetime_path(
        out_dir, cutline_id, diameter_um, cv, freq_hz, axon_id=str(axon_id),
    )
    _atomic_savez_compressed(
        out_path,
        time_s=np.arange(n_time, dtype=np.float32) / np.float32(fs_hz),
        node_numbers=node_numbers,
        rep_labels=rep_labels,
        current=current,
        V=V,
        axon_id=np.asarray(axon_id),
        cutline_id=np.asarray(cutline_id),
        diameter_um=np.asarray(diameter_um, dtype=np.float32),
        phase_id=np.asarray(phase_id),
        phase_fraction=np.asarray(phase_fraction, dtype=np.float32),
        phase_offset_mm=np.asarray(phase_offset_mm, dtype=np.float32),
        phase_scheme=np.asarray(phase_scheme),
        node_arc_length_mm=node_arc_length_mm,
        node_arc_length_requested_mm=requested_arc_length_mm,
        node_spacing_um=node_spacing_um,
        n_valid_nodes=n_valid_nodes,
        readout_node_number=readout_node_number,
        cv0_waveform_policy=np.asarray(
            "canonical_exact_periodic" if np.isclose(float(cv), 0.0) else "not_applicable"
        ),
    )
    del V
    return out_path


def generate_population_spacetime(
    basis_rows: Iterable[dict[str, object]],
    cv_vals: Iterable[float],
    freq_vals: Iterable[float],
    *,
    time_pattern_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Generate one compatible SpaceTimeV NPZ per axon/CV/frequency state."""
    rows: list[dict[str, object]] = []
    for basis_row in basis_rows:
        for cv in cv_vals:
            for freq_hz in freq_vals:
                path = spacetime_path(
                    out_dir,
                    str(basis_row["cutline_id"]),
                    float(basis_row["diameter_um"]),
                    float(cv),
                    float(freq_hz),
                    axon_id=str(basis_row["axon_id"]),
                )
                file_exists = path.is_file()
                valid_existing = file_exists
                if file_exists and not overwrite:
                    valid_existing = _npz_has_fields(path, SPACETIME_NPZ_FIELDS)
                if file_exists and not overwrite and not valid_existing:
                    warnings.warn(
                        f"Regenerating invalid population SpaceTimeV archive: {path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if overwrite or not valid_existing:
                    basis_path = Path(basis_row["basis_npz"])
                    if not basis_path.is_absolute():
                        basis_path = PROJECT_ROOT / basis_path
                    path = generate_spacetime_from_basis(
                        basis_path,
                        cv=float(cv),
                        freq_hz=float(freq_hz),
                        time_pattern_dir=time_pattern_dir,
                        out_dir=out_dir,
                    )
                rows.append({
                    "axon_id": basis_row["axon_id"],
                    "cutline_id": basis_row["cutline_id"],
                    "diameter_um": basis_row["diameter_um"],
                    "phase_id": basis_row["phase_id"],
                    "phase_fraction": basis_row["phase_fraction"],
                    "cv": float(cv),
                    "freq_hz": float(freq_hz),
                    "spacetime_npz": str(path),
                })
    if not rows:
        raise ValueError("At least one basis, CV, and frequency are required")
    manifest = out_dir / "population_spacetime_manifest.csv"
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    return rows


def population_axon_jobs(root: Path = PROJECT_ROOT) -> list[dict[str, object]]:
    """Return the stable trajectory/diameter/phase generation-array order."""
    root = root.resolve()
    cutlines = production_comsol_cutline_ids(root / "data" / "comsol")
    phases = population_phase_assignments(cutlines)
    return [
        {
            "axon_id": make_axon_id(cutline_id, diameter_um, str(phase["phase_id"])),
            "cutline_id": cutline_id,
            "diameter_um": float(diameter_um),
            "phase_id": str(phase["phase_id"]),
            "phase_fraction": float(phase["phase_fraction"]),
            "phase_scheme": POPULATION_PHASE_SCHEME,
        }
        for cutline_id in cutlines
        for diameter_um in POPULATION_DIAMETERS_UM
        for phase in phases[cutline_id]
    ]


def prepare_population_bases(root: Path = PROJECT_ROOT) -> Path:
    """Generate the complete anatomy-named Ve-basis cohort and its manifest."""
    root = root.resolve()
    comsol_dir = root / "data" / "comsol"
    out_dir = root / "data" / "space-time-V" / "population"
    generate_population_ve_bases(
        production_comsol_cutline_ids(comsol_dir), POPULATION_DIAMETERS_UM,
        comsol_dir=comsol_dir, out_dir=out_dir,
    )
    return out_dir / "axon_basis_manifest.csv"


def run_population_spacetime_axon_job(
    *, root: Path = PROJECT_ROOT, job_index: int, overwrite: bool = False,
) -> list[Path]:
    """Generate all 55 CV/frequency archives for one axon array task."""
    root = root.resolve()
    jobs = population_axon_jobs(root)
    if not 0 <= job_index < len(jobs):
        raise IndexError(f"job_index must be in [0, {len(jobs) - 1}]")
    job = jobs[job_index]
    out_dir = root / "data" / "space-time-V" / "population"
    basis_path = _basis_path(
        out_dir, str(job["cutline_id"]), float(job["diameter_um"]), str(job["phase_id"]),
    )
    if not basis_path.is_file():
        raise FileNotFoundError(f"Population Ve basis is missing: {basis_path}")
    outputs: list[Path] = []
    for cv in POPULATION_CV_VALS:
        for freq_hz in POPULATION_FREQ_VALS_HZ:
            path = spacetime_path(
                out_dir, str(job["cutline_id"]), float(job["diameter_um"]), cv, freq_hz,
                axon_id=str(job["axon_id"]),
            )
            if overwrite or not _spacetime_matches_job(path, job):
                path = generate_spacetime_from_basis(
                    basis_path, cv=cv, freq_hz=freq_hz,
                    time_pattern_dir=root / "data" / "time-patterns" / "population",
                    out_dir=out_dir,
                )
            outputs.append(path)
    return outputs


def validate_population_spacetime(root: Path = PROJECT_ROOT) -> Path:
    """Validate every v2 basis/archive and atomically write the final manifest."""
    root = root.resolve()
    out_dir = root / "data" / "space-time-V" / "population"
    rows: list[dict[str, object]] = []
    for job in population_axon_jobs(root):
        basis_path = _basis_path(
            out_dir, str(job["cutline_id"]), float(job["diameter_um"]), str(job["phase_id"]),
        )
        if not basis_path.is_file():
            raise FileNotFoundError(f"Missing population Ve basis: {basis_path}")
        with np.load(basis_path, allow_pickle=False) as basis:
            if str(basis["axon_id"].item()) != job["axon_id"]:
                raise RuntimeError(f"Basis axon_id mismatch: {basis_path}")
        for cv in POPULATION_CV_VALS:
            for freq_hz in POPULATION_FREQ_VALS_HZ:
                path = spacetime_path(
                    out_dir, str(job["cutline_id"]), float(job["diameter_um"]), cv, freq_hz,
                    axon_id=str(job["axon_id"]),
                )
                if not _spacetime_matches_job(path, job):
                    raise RuntimeError(f"Missing or invalid population SpaceTimeV archive: {path}")
                rows.append({
                    **job, "cv": float(cv), "freq_hz": float(freq_hz),
                    "spacetime_npz": str(path),
                })
    manifest = out_dir / "population_spacetime_manifest.csv"
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    return manifest


# --------------- MRG fresh-worker preparation --------------- #

@dataclass(frozen=True)
class MRGWorkerSpec:
    """Static geometry and basis jobs for one fresh single-diameter worker."""

    diameter_um: float
    axonnodes: int
    readout_node_number: int
    axon_ids: tuple[str, ...]
    basis_paths: tuple[str, ...]


def _basis_metadata(path: Path) -> tuple[float, int, int, str]:
    with np.load(path, allow_pickle=False) as basis:
        # Basis metadata is float32; normalize 5.6999998 before key lookup.
        diameter_um = round(float(np.asarray(basis["diameter_um"]).item()), 1)
        n_valid_nodes = int(np.asarray(basis["n_valid_nodes"]).item())
        readout_node = int(np.asarray(basis["readout_node_number"]).item())
        axon_id = str(np.asarray(basis["axon_id"]).item())
    return diameter_um, n_valid_nodes, readout_node, axon_id


def discover_mrg_workers(basis_dir: Path) -> list[MRGWorkerSpec]:
    """Make one fresh-worker spec per distinct MRG geometry.

    Nodal phase can change the number of complete nodes retained from a finite
    COMSOL cutline.  Each ``(diameter, node count, readout)`` geometry therefore
    receives its own HOC process; incompatible section arrays are never reused.
    """
    basis_paths = sorted(basis_dir.glob("*_VeBasis.npz"))
    if not basis_paths:
        raise FileNotFoundError(f"No population Ve basis files found in {basis_dir}")
    grouped: dict[float, list[tuple[int, int, str, Path]]] = defaultdict(list)
    for path in basis_paths:
        diameter_um, n_nodes, readout_node, axon_id = _basis_metadata(path)
        grouped[diameter_um].append((n_nodes, readout_node, axon_id, path))

    supported = set(MRG_NODE_SPACING_MM)
    unexpected = set(grouped) - supported
    if unexpected:
        raise ValueError(f"Unsupported MRG diameters in basis files: {sorted(unexpected)}")

    workers: list[MRGWorkerSpec] = []
    for diameter_um in sorted(grouped):
        by_geometry: dict[tuple[int, int], list[tuple[str, Path]]] = defaultdict(list)
        for n_nodes, readout_node, axon_id, path in grouped[diameter_um]:
            by_geometry[(n_nodes, readout_node)].append((axon_id, path))
        for (n_nodes, readout_node), geometry_rows in sorted(by_geometry.items()):
            workers.append(MRGWorkerSpec(
                diameter_um=diameter_um,
                axonnodes=n_nodes,
                readout_node_number=readout_node,
                axon_ids=tuple(row[0] for row in geometry_rows),
                basis_paths=tuple(str(row[1].resolve()) for row in geometry_rows),
            ))
    return workers


def _hoc_string(path: Path) -> str:
    """Return an absolute HOC-safe path, using forward slashes on Windows."""
    return path.resolve().as_posix().replace('"', '\\"')


def write_worker_bundle(worker: MRGWorkerSpec, *, model_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write the HOC configuration and JSON work list for one diameter."""
    if worker.readout_node_number != worker.axonnodes - 1:
        raise ValueError("Population readout must be the terminal valid node")
    out_dir.mkdir(parents=True, exist_ok=True)
    token = diameter_token(worker.diameter_um)
    config_path = out_dir / f"mrg_population_config_{token}um_{worker.axonnodes}nodes.hoc"
    template_path = model_dir / "MRGaxon_population_template.hoc"
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    config_path.write_text(
        "// Generated by spacetime_core; load this in a fresh NEURON process.\n"
        f"mrg_population_fiberD = {worker.diameter_um:.1f}\n"
        f"mrg_population_axonnodes = {worker.axonnodes}\n"
        f"mrg_population_readout_node = {worker.readout_node_number}\n"
        f'xopen("{_hoc_string(template_path)}")\n',
        encoding="utf-8",
    )
    jobs_path = out_dir / f"mrg_population_jobs_{token}um_{worker.axonnodes}nodes.json"
    jobs_path.write_text(json.dumps(asdict(worker), indent=2), encoding="utf-8")
    return config_path, jobs_path


def prepare_mrg_workers(*, root: Path = PROJECT_ROOT, out_dir: Path | None = None) -> list[MRGWorkerSpec]:
    """Discover available diameter groups and create independent worker assets."""
    root = root.resolve()
    workers = discover_mrg_workers(root / "data" / "space-time-V" / "population")
    bundle_dir = out_dir or root / "results" / "population" / "mrg-workers"
    for worker in workers:
        write_worker_bundle(worker, model_dir=root / "MRG_model", out_dir=bundle_dir)
    return workers


# --------------- Command-line entry point --------------- #

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate", help="Resume population basis/SpaceTimeV generation.")
    generate_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    generate_parser.add_argument("--overwrite", action="store_true", help="Regenerate existing SpaceTimeV files.")
    bases_parser = subparsers.add_parser("prepare-bases", help="Generate all v2 population Ve bases.")
    bases_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    axon_parser = subparsers.add_parser("run-axon", help="Generate all SpaceTimeV conditions for one axon array index.")
    axon_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    axon_parser.add_argument("--job-index", type=int, required=True)
    axon_parser.add_argument("--overwrite", action="store_true")
    validate_parser = subparsers.add_parser("validate", help="Validate the complete v2 SpaceTimeV cohort.")
    validate_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    count_parser = subparsers.add_parser("job-count", help="Print the number of axon generation jobs.")
    count_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    worker_parser = subparsers.add_parser("prepare-workers", help="Prepare fresh MRG geometry worker bundles.")
    worker_parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    worker_parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested local population SpaceTimeV command."""
    args = _parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare-bases":
        print(prepare_population_bases(root))
        return 0
    if args.command == "run-axon":
        outputs = run_population_spacetime_axon_job(root=root, job_index=args.job_index, overwrite=args.overwrite)
        print(f"Validated/generated {len(outputs)} SpaceTimeV files for axon job {args.job_index}.")
        return 0
    if args.command == "validate":
        print(validate_population_spacetime(root))
        return 0
    if args.command == "job-count":
        print(len(population_axon_jobs(root)))
        return 0
    if args.command == "generate":
        comsol_dir = root / "data" / "comsol"
        basis_rows = generate_population_ve_bases(
            production_comsol_cutline_ids(comsol_dir),
            POPULATION_DIAMETERS_UM,
            comsol_dir=comsol_dir,
            out_dir=root / "data" / "space-time-V" / "population",
        )
        rows = generate_population_spacetime(
            basis_rows,
            POPULATION_CV_VALS,
            POPULATION_FREQ_VALS_HZ,
            time_pattern_dir=root / "data" / "time-patterns" / "population",
            out_dir=root / "data" / "space-time-V" / "population",
            overwrite=args.overwrite,
        )
        print(f"Population SpaceTimeV manifest contains {len(rows)} files.")
        return 0
    workers = prepare_mrg_workers(root=root, out_dir=args.out_dir)
    for index, worker in enumerate(workers):
        print(f"worker {index}: {worker.diameter_um:.1f} um, {worker.axonnodes} nodes, {len(worker.axon_ids)} axons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
