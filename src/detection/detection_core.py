"""detection_core.py

Index-anchored detection simulations for a heterogeneous MRG axon population.

Workflow:
1. Find the first-recruited (index) axon with a standard bipolar pulse.
2. Apply the established threshold searches to the index axon's
   per-(frequency, CV, repetition) spike-count regimes.
3. Simulate every population axon at each saved index-regime amplitude.
4. Fit constrained operating-state mixtures to the detection behavior without
   rerunning NEURON, and export the primary global mixture for observer controls.

NEURON section arrays are fixed when a HOC file loads.  Work is therefore
partitioned by diameter; every worker process loads exactly one generated MRG
configuration and never exchanges NEURON objects with another process.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.signal import find_peaks
from tqdm import tqdm

from detection.protocol import (
    CV0_PROTOCOL_VERSION,
    DEFAULT_EXPECTED_INDEX_AXON_ID,
    DETECTION_PROTOCOL_VERSION,
    I_MAX_BRANCH_SEARCH_WIDTH_UA,
    I_MAX_COARSE_STEP_UA,
    I_MAX_DECLINE_WIDTH_UA,
    I_MAX_EARLY_PEAK_MINIMUM_DROP_SPIKES,
    I_MAX_EARLY_PEAK_MINIMUM_RISE_FRACTION,
    I_MAX_EARLY_PEAK_SEARCH_WIDTH_UA,
    I_MAX_EARLY_PEAK_TAIL_SAMPLES,
    I_MAX_MIN_SAMPLED_MAX_FRACTION,
    I_MAX_PLATEAU_CANDIDATE_WIDTH_UA,
    I_MAX_PLATEAU_CONFIRMATION_WIDTH_UA,
    I_MAX_PLATEAU_TOLERANCE_FRACTIONS,
    I_MAX_PLATEAU_WIDTH_UA,
    POPULATION_THRESHOLD_TOL_UA,
)
from space_time_v.spacetime_core import (
    MRG_NODE_SPACING_MM,
    diameter_token,
    discover_mrg_workers,
    make_axon_id,
    prepare_mrg_workers,
    spacetime_path,
)
from detection.threshold_search import (
    ALL_SPIKES_SEARCH,
    FIRST_SPIKE_SEARCH,
    INIT_V_MV,
    PREPAD_MS,
    SPIKE_HEIGHT_MV,
    THRESH_TYPES,
    VOLLEY5_MIN_SPIKES,
    VOLLEY10_MIN_SPIKES,
    find_first_spike_threshold,
    prepend_zeros_ms,
    trim_tail_window_ms,
)


# ---------------- Configuration ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (PROJECT_ROOT / "data" / "space-time-V").resolve()
RESULTS_DIR = (PROJECT_ROOT / "data" / "detection").resolve()
MRG_MODEL_DIR = (PROJECT_ROOT / "hpc_tools/MRG_model").resolve()

INDEX_THRESHOLDS_CSV = RESULTS_DIR / "index_axon_thresholds.csv"
INDEX_AXON_CSV = RESULTS_DIR / "index_axon.csv"
INDEX_REGIMES_CSV = RESULTS_DIR / "index_regime_amplitudes.csv"
INDEX_CV0_THRESH_CSV = RESULTS_DIR / "index_cv0_threshold_max.csv"
POPULATION_RESPONSES_CSV = RESULTS_DIR / "population_responses.csv"
POPULATION_CV0_RESPONSES_CSV = RESULTS_DIR / "population_cv0_based_responses.csv"
POPULATION_PARTIAL_DIR = RESULTS_DIR / "partials"
DETECTION_BEHAVIOR_XLSX = PROJECT_ROOT / "data" / "detection" / "slack" / "DetectionThresholds.xlsx"
DETECTION_BEHAVIORAL_MATCHING_DIR = PROJECT_ROOT / "data" / "detection" / "behavioral_matching"

# This is deliberately the same condition grid produced by the time-pattern
# stage.  Keeping it explicit makes SLURM array indices stable and auditable.
POPULATION_CVS = tuple(round(.1 * i, 1) for i in range(11))
POPULATION_FREQS = (10.0, 20.0, 50.0, 100.0, 200.0)

# Population operating-state protocol.  CV=0 is deterministic, so every
# nominal repetition uses the same exact periodic waveform even when legacy
# SpaceTimeV archives retain ``rep001_slack`` as a compatibility label.  The
# finer tolerance is population-specific; the threshold search functions are
# vendored in detection.threshold_search.
REFERENCE_FS_HZ = 20_000.0
REFERENCE_LEAD_MS = 1.0
REFERENCE_PHASE_WIDTH_MS = 0.2
REFERENCE_INTERPHASE_MS = 0.0
REFERENCE_TAIL_MS = 5.0

REGIME_COLUMNS = (
    "detection_protocol_version", "index_axon_id", "index_cutline_id", "index_diameter_um", "file", "cv", "freq",
    "rep_index", "rep_label", "n_input_pulses", "thresh_type", "amp_uA",
    "n_index_spikes", "index_spike_times_ms", "N_max_spikes",
)
POPULATION_RESPONSE_COLUMNS = (
    "detection_protocol_version", "index_axon_id", "index_cutline_id", "index_diameter_um", "target_axon_id",
    "target_cutline_id", "target_diameter_um", "target_n_valid_nodes",
    "target_readout_node_number", "file", "cv", "freq", "rep_index", "rep_label",
    "n_input_pulses", "thresh_type", "amp_uA", "n_index_spikes", "n_spikes",
    "spike_times_ms",
)

os.environ.setdefault("OMP_NUM_THREADS", "1")

DEFAULT_MAX_PRUNING_AUDIT_RATE = 0.01
DEFAULT_DIAMETER_PRUNING_AUDIT_RATE = 0.01


# --------------- Population SpaceTimeV loading --------------- #

def _detection_partial_dir(root: Path, family: str, *, protocol_version: str = DETECTION_PROTOCOL_VERSION) -> Path:
    """Return the cache-isolated partial directory for this protocol."""
    expected_index = os.environ.get(
        "EXPECTED_INDEX_AXON_ID", DEFAULT_EXPECTED_INDEX_AXON_ID,
    )
    return (
        root / "data" / "detection" / "partials"
        / protocol_version / expected_index / family
    )

def load_population_basis(basis_path: Path) -> dict[str, Any]:
    """Load the static, 1 uA Ve basis and geometry for one population axon."""
    with np.load(basis_path, allow_pickle=False) as basis:
        fields = set(basis.files)
        return {
            "basis_path": str(basis_path.resolve()),
            "axon_id": str(basis["axon_id"].item()),
            "cutline_id": str(basis["cutline_id"].item()),
            "diameter_um": round(float(basis["diameter_um"].item()), 1),
            "node_numbers": np.asarray(basis["node_numbers"], dtype=np.int16),
            "V_neg_mV": np.asarray(basis["V_neg_mV"], dtype=np.float32),
            "V_pos_mV": np.asarray(basis["V_pos_mV"], dtype=np.float32),
            "n_valid_nodes": int(basis["n_valid_nodes"].item()),
            "readout_node_number": int(basis["readout_node_number"].item()),
            "phase_id": str(basis["phase_id"].item()) if "phase_id" in fields else "legacy",
            "phase_fraction": float(basis["phase_fraction"].item()) if "phase_fraction" in fields else 0.0,
            "phase_offset_mm": float(basis["phase_offset_mm"].item()) if "phase_offset_mm" in fields else 0.0,
            "phase_scheme": str(basis["phase_scheme"].item()) if "phase_scheme" in fields else "legacy-single-phase",
        }


def _basis_path_for_axon(data_dir: Path, axon_id: str) -> Path:
    path = data_dir / f"{axon_id}_1uA_VeBasis.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Ve basis not found for {axon_id}: {path}")
    return path


def _is_cv0_spacetime_path(npz_path: Path) -> bool:
    """Return whether a population SpaceTimeV filename encodes CV=0."""
    match = re.search(r"_cv([0-9]+(?:\.[0-9]+)?)_f", npz_path.name)
    if match is None:
        return False
    return bool(np.isclose(float(match.group(1)), 0.0))


def _canonical_cv0_source_index(rep_labels: Sequence[str]) -> int:
    """Select the generated exact-periodic waveform in a legacy CV0 bundle."""
    regular = [index for index, label in enumerate(rep_labels) if "regular" in str(label).lower()]
    if regular:
        return int(regular[0])
    if not rep_labels:
        raise ValueError("CV0 bundle contains no repetition labels")
    # Newly regenerated bundles may eventually use only periodic labels.  In
    # that case the first entry is already the canonical waveform.
    return 0


def load_population_rep(
    npz_path: Path, rep_index: int, *, canonicalize_cv0: bool = True,
) -> dict[str, Any]:
    """Load one repetition without expanding the dense ``V`` array from its NPZ.

    Population SpaceTimeV files intentionally retain the single-compatible
    dense ``V`` field.  Detection reconstructs exactly one repetition from the
    compact basis plus its saved current, preventing every worker from loading
    a multi-gigabyte dense bundle unnecessarily.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        rep_labels = [str(label) for label in data["rep_labels"]]
        source_rep_index = (
            _canonical_cv0_source_index(rep_labels)
            if canonicalize_cv0 and _is_cv0_spacetime_path(npz_path)
            else int(rep_index)
        )
        current = np.asarray(data["current"][source_rep_index], dtype=np.float32)
        time_ms = np.asarray(data["time_s"], dtype=np.float32) * 1000.0
        rep_label = rep_labels[rep_index]
        axon_id = str(data["axon_id"].item())
        cutline_id = str(data["cutline_id"].item())
        diameter_um = round(float(data["diameter_um"].item()), 1)
        n_valid_nodes = int(data["n_valid_nodes"].item())
        readout_node_number = int(data["readout_node_number"].item())

    basis = load_population_basis(_basis_path_for_axon(npz_path.parent, axon_id))
    V_base = np.zeros((current.size, n_valid_nodes), dtype=np.float32)
    V_base[current < 0] = basis["V_neg_mV"]
    V_base[current > 0] = basis["V_pos_mV"]
    return {
        "time_ms": time_ms,
        "current": current,
        "V_base": V_base,
        "rep_label": rep_label,
        "waveform_source_rep_index": source_rep_index,
        "axon_id": axon_id,
        "cutline_id": cutline_id,
        "diameter_um": diameter_um,
        "n_valid_nodes": n_valid_nodes,
        "readout_node_number": readout_node_number,
    }


def _repetition_equivalence_groups(
    npz_path: Path, *, canonicalize_cv0: bool = True,
) -> list[list[tuple[int, str]]]:
    """Group repetitions whose saved current arrays are exactly identical.

    CV=0 is a single deterministic representation by protocol; all legacy
    labels therefore form one group using the generated exact-periodic current.
    For CV>0, hashing avoids quadratic comparisons, while the final
    ``array_equal`` check makes hash collisions harmless.  The returned groups
    retain every original repetition index and label for output replication.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        currents = np.asarray(data["current"], dtype=np.float32)
        labels = [str(label) for label in data["rep_labels"]]
    if currents.ndim != 2 or currents.shape[0] != len(labels):
        raise ValueError(f"Invalid current/rep_labels fields in {npz_path}")
    if canonicalize_cv0 and _is_cv0_spacetime_path(npz_path):
        # Validate that a canonical source can be resolved before treating the
        # legacy labels as repeated observations of the same deterministic CV0.
        _canonical_cv0_source_index(labels)
        return [[(rep_index, label) for rep_index, label in enumerate(labels)]]

    groups: list[list[tuple[int, str]]] = []
    representatives: list[np.ndarray] = []
    digest_to_groups: dict[bytes, list[int]] = defaultdict(list)
    for rep_index, (label, current) in enumerate(zip(labels, currents)):
        digest = hashlib.sha256(np.ascontiguousarray(current).view(np.uint8)).digest()
        group_index = next(
            (
                candidate
                for candidate in digest_to_groups[digest]
                if np.array_equal(current, representatives[candidate])
            ),
            None,
        )
        if group_index is None:
            group_index = len(groups)
            groups.append([])
            representatives.append(current.copy())
            digest_to_groups[digest].append(group_index)
        groups[group_index].append((rep_index, label))
    return groups


def _replicate_rows_for_repetitions(
    rows: Sequence[dict[str, Any]], members: Sequence[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Copy representative simulation rows to exactly equivalent repetitions."""
    replicated: list[dict[str, Any]] = []
    for rep_index, rep_label in members:
        for row in rows:
            clone = dict(row)
            clone["rep_index"] = int(rep_index)
            clone["rep_label"] = str(rep_label)
            replicated.append(clone)
    return replicated


# --------------- Diameter-aware NEURON runner --------------- #

_MRG_LOADED = False
_MRG_CONFIG_PATH: Path | None = None
_NEURON_DLL_DIR_HANDLES: list[Any] = []


def _configure_windows_neuron_installer() -> None:
    """Expose an official Windows NEURON install to the active Python process.

    Set ``NEURONHOME=C:\\nrn`` before launching Python.  A normal pip install
    needs none of this; this branch supports the Windows installer and its
    bundled mechanism compiler without requiring a second NEURON installation.
    """
    if platform.system() != "Windows" or _NEURON_DLL_DIR_HANDLES:
        return
    neuron_home = os.environ.get("NEURONHOME", r"C:\nrn")
    if not Path(neuron_home).is_dir():
        return
    home = Path(neuron_home)
    # The Windows installer uses both names to locate lib/nrn.defaults.
    os.environ.setdefault("NEURONHOME", str(home))
    os.environ.setdefault("NRNHOME", str(home))
    python_dir = home / "lib" / "python"
    if python_dir.is_dir() and str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    for dll_dir in (home / "bin", home / "mingw" / "mingw64" / "bin"):
        if dll_dir.is_dir():
            _NEURON_DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))


def _get_h():
    _configure_windows_neuron_installer()
    from neuron import h as _h
    return _h


def _config_path(root: Path, diameter_um: float, axonnodes: int) -> Path:
    token = diameter_token(diameter_um)
    return root / "logs" / "mrg-workers" / f"mrg_population_config_{token}um_{axonnodes}nodes.hoc"


def ensure_population_mrg_loaded(config_path: Path) -> None:
    """Load mechanisms and one fixed population MRG geometry per process."""
    global _MRG_LOADED, _MRG_CONFIG_PATH
    config_path = config_path.resolve()
    if _MRG_LOADED:
        if config_path != _MRG_CONFIG_PATH:
            raise RuntimeError(
                "A NEURON worker cannot switch MRG section-array geometry after loading. "
                "Run each diameter in a separate fresh process."
            )
        return

    dll_name = "nrnmech.dll" if platform.system() == "Windows" else "libnrnmech.so"
    # Windows mknrndll writes beside AXNODE.mod.  On Linux, nrnivmodl writes
    # into an architecture-specific directory (normally x86_64/).
    candidates = [MRG_MODEL_DIR / dll_name]
    if platform.system() != "Windows":
        candidates.extend(MRG_MODEL_DIR.glob(f"*/{dll_name}"))
        candidates.extend(MRG_MODEL_DIR.glob(f"*/.libs/{dll_name}"))
    dll_path = next((path for path in candidates if path.is_file()), MRG_MODEL_DIR / dll_name)
    if not dll_path.is_file():
        raise FileNotFoundError(
            f"{dll_path} is required for the population template. Compile AXNODE.mod first; "
            "mosinit.hoc cannot be used because it loads the fixed single-axon HOC."
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Population MRG configuration not found: {config_path}")

    h = _get_h()
    try:
        h.load_file("stdrun.hoc")
    except Exception:
        pass
    h.nrn_load_dll(str(dll_path).replace("\\", "/"))
    h.load_file(str(config_path).replace("\\", "/"))
    try:
        h.cvode_active(0)
    except Exception:
        pass
    _MRG_CONFIG_PATH = config_path
    _MRG_LOADED = True


class PopulationMRGRunner:
    """MRG runner for one fixed diameter and dynamic terminal readout node."""

    def __init__(self, time_ms: np.ndarray, V_base: np.ndarray, *, config_path: Path, readout_node_number: int):
        ensure_population_mrg_loaded(config_path)
        self.h = _get_h()
        self.time_ms = np.asarray(time_ms, dtype=np.float32)
        self.V_base = np.asarray(V_base, dtype=np.float32)
        if self.V_base.ndim != 2 or self.V_base.shape[0] != self.time_ms.size:
            raise ValueError("V_base must have shape (time, node) and match time_ms")
        self.mrg_nodes = int(self.h.axonnodes)
        if self.V_base.shape[1] != self.mrg_nodes:
            raise ValueError(f"V has {self.V_base.shape[1]} nodes, MRG model has {self.mrg_nodes} nodes")
        self.rec_idx = int(readout_node_number)
        if not 0 <= self.rec_idx < self.mrg_nodes:
            raise ValueError(f"Invalid terminal readout node {self.rec_idx} for {self.mrg_nodes} nodes")
        if self.time_ms.size > 1:
            self.h.dt = float(self.time_ms[1] - self.time_ms[0])
        self.tvec = self.h.Vector(self.time_ms)
        self.base_v, self.play_v = [], []
        for node_idx in range(self.mrg_nodes):
            base = self.h.Vector(self.V_base[:, node_idx])
            playback = self.h.Vector(self.V_base[:, node_idx])
            playback.play(self.h.node[node_idx](0.5)._ref_e_extracellular, self.tvec, 1)
            self.base_v.append(base)
            self.play_v.append(playback)
        self.v_rec, self.t_rec = self.h.Vector(), self.h.Vector()
        self.v_rec.record(self.h.node[self.rec_idx](0.5)._ref_v)
        self.t_rec.record(self.h._ref_t)
        self.tstop = float(self.time_ms[-1])

    def run_mrg_for_amp(self, scale_uA: float) -> tuple[np.ndarray, int]:
        self.v_rec.resize(0)
        self.t_rec.resize(0)
        for base, playback in zip(self.base_v, self.play_v):
            playback.copy(base)
            playback.mul(float(scale_uA))
        self.h.tstop = self.tstop
        self.h.finitialize(float(INIT_V_MV))
        try:
            self.h.frecord_init()
        except Exception:
            pass
        self.h.continuerun(self.tstop)
        peak_idx, _ = find_peaks(np.asarray(self.v_rec), height=float(SPIKE_HEIGHT_MV))
        spike_times_ms = np.asarray(self.t_rec)[peak_idx]
        return spike_times_ms, int(spike_times_ms.size)


# --------------- Index axon selection --------------- #

def _reference_trace(basis: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Create the standard cathodic-first bipolar reference impulse at 1 uA."""
    dt_ms = 1000.0 / REFERENCE_FS_HZ
    n_lead = int(round(REFERENCE_LEAD_MS / dt_ms))
    n_phase = int(round(REFERENCE_PHASE_WIDTH_MS / dt_ms))
    n_gap = int(round(REFERENCE_INTERPHASE_MS / dt_ms))
    n_tail = int(round(REFERENCE_TAIL_MS / dt_ms))
    current = np.zeros(n_lead + 2 * n_phase + n_gap + n_tail, dtype=np.float32)
    current[n_lead:n_lead + n_phase] = -1.0
    pos_start = n_lead + n_phase + n_gap
    current[pos_start:pos_start + n_phase] = 1.0
    V_base = np.zeros((current.size, basis["n_valid_nodes"]), dtype=np.float32)
    V_base[current < 0] = basis["V_neg_mV"]
    V_base[current > 0] = basis["V_pos_mV"]
    time_ms = np.arange(current.size, dtype=np.float32) * np.float32(dt_ms)
    return time_ms, V_base


def _index_threshold_group(root: Path, basis_paths: Sequence[Path], diameter_um: float, axonnodes: int) -> list[dict[str, Any]]:
    config = _config_path(root, diameter_um, axonnodes)
    rows = []
    for basis_path in basis_paths:
        basis = load_population_basis(basis_path)
        time_ms, V_base = _reference_trace(basis)
        time_ms, V_base = prepend_zeros_ms(time_ms, V_base, prepad_ms=PREPAD_MS)
        runner = PopulationMRGRunner(time_ms, V_base, config_path=config, readout_node_number=basis["readout_node_number"])
        cache: dict[float, int] = {}
        def count(amp: float) -> int:
            amp = float(amp)
            if amp not in cache:
                cache[amp] = runner.run_mrg_for_amp(amp)[1]
            return cache[amp]
        search = dict(FIRST_SPIKE_SEARCH)
        search["tol_uA"] = POPULATION_THRESHOLD_TOL_UA
        threshold_uA, n_spikes, *_ = find_first_spike_threshold(count, **search)
        rows.append({
            "detection_protocol_version": DETECTION_PROTOCOL_VERSION,
            "threshold_tolerance_uA": POPULATION_THRESHOLD_TOL_UA,
            "axon_id": basis["axon_id"], "cutline_id": basis["cutline_id"], "diameter_um": basis["diameter_um"],
            "phase_id": basis["phase_id"], "phase_fraction": basis["phase_fraction"],
            "phase_offset_mm": basis["phase_offset_mm"], "phase_scheme": basis["phase_scheme"],
            "n_valid_nodes": basis["n_valid_nodes"], "readout_node_number": basis["readout_node_number"],
            "reference_phase_width_ms": REFERENCE_PHASE_WIDTH_MS,
            "reference_interphase_ms": REFERENCE_INTERPHASE_MS,
            "reference_polarity": "cathodic-first",
            "I_first_uA": threshold_uA, "N_I_first_spikes": int(n_spikes),
        })
    return rows


def identify_index_axon(
    *, root: Path = PROJECT_ROOT, n_jobs: int = 4, force: bool = False,
    population_manifest: Path | None = None, output_dir: Path | None = None,
) -> dict[str, Any]:
    """Find and save the lowest-reference-threshold axon across the population."""
    root = root.resolve()
    output_dir = (output_dir or root / "data" / "detection").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_csv, thresholds_csv = output_dir / INDEX_AXON_CSV.name, output_dir / INDEX_THRESHOLDS_CSV.name
    if not force and index_csv.is_file() and thresholds_csv.is_file():
        return pd.read_csv(index_csv).iloc[0].to_dict()

    workers = discover_mrg_workers(root / "data" / "space-time-V")
    prepare_mrg_workers(root=root)
    selected = {path.resolve() for path in _population_basis_paths(root, population_manifest)}
    tasks = [
        (worker.diameter_um, worker.axonnodes, path)
        for worker in workers
        for path in (Path(item) for item in worker.basis_paths)
        if path.resolve() in selected
    ]
    if not tasks:
        raise RuntimeError("Population manifest selected no MRG threshold tasks")
    # Each process handles exactly one axon and then exits. This keeps all CPU
    # slots load-balanced without ever switching HOC section-array geometry in
    # an already initialized NEURON process.
    with ProcessPoolExecutor(
        max_workers=min(max(1, n_jobs), len(tasks)), max_tasks_per_child=1,
    ) as executor:
        futures = [
            executor.submit(_index_threshold_group, root, [path], diameter, n_nodes)
            for diameter, n_nodes, path in tasks
        ]
        rows_by_group = [future.result() for future in futures]
    table = pd.DataFrame([row for rows in rows_by_group for row in rows])
    table.to_csv(thresholds_csv, index=False)
    eligible = table.dropna(subset=["I_first_uA"]).sort_values(["I_first_uA", "axon_id"], kind="stable")
    if eligible.empty:
        raise RuntimeError("No population axon spiked during the reference-threshold search")
    index_row = eligible.iloc[0].to_dict()
    index_row["index_axon_id"] = index_row["axon_id"]
    pd.DataFrame([index_row]).to_csv(index_csv, index=False)
    return index_row


# --------------- Index-anchored regime thresholds --------------- #

def _first_cached_target_bracket(
    cache: dict[float, tuple[np.ndarray, int]], target: int, *, lower_bound: float,
) -> tuple[float, float] | None:
    """Return the tightest sampled bracket around the first target crossing."""
    observations = sorted(
        (float(amp), int(result[1]))
        for amp, result in cache.items()
        if float(amp) >= float(lower_bound)
    )
    hit_index = next((i for i, (_, count) in enumerate(observations) if count >= target), None)
    if hit_index is None:
        return None
    hi = observations[hit_index][0]
    lo = float(lower_bound) if hit_index == 0 else observations[hit_index - 1][0]
    return float(lo), float(hi)


def _find_cached_plateau_threshold(
    spike_counter: Any,
    cache: dict[float, tuple[np.ndarray, int]],
    I_first: float,
    *,
    scan_start_uA: float | None = None,
    amp_max_uA: float,
    tol_uA: float,
    coarse_step_uA: float = I_MAX_COARSE_STEP_UA,
    plateau_width_uA: float = I_MAX_PLATEAU_WIDTH_UA,
    min_sampled_max_fraction: float = I_MAX_MIN_SAMPLED_MAX_FRACTION,
) -> tuple[float | None, int]:
    """Find the first stable near-maximal response plateau.

    The 2000-uA endpoint remains a safety/reference sample, but a rare final
    spike at that endpoint no longer moves ``I_all`` across a long saturation
    plateau. Starting at the already resolved ``I_50`` threshold, the search
    advances on a fixed coarse grid and accepts the first constant-count run
    spanning ``plateau_width_uA`` whose count is at least
    ``min_sampled_max_fraction`` of the largest count on that grid. The lower
    boundary is then refined to the population threshold tolerance.

    Solving the named fixed-count regimes before calling this function keeps
    their definitions and thresholds independent of the I_max scan.
    """
    values = {
        "I_first": I_first,
        "scan_start_uA": I_first if scan_start_uA is None else scan_start_uA,
        "amp_max_uA": amp_max_uA,
        "tol_uA": tol_uA,
        "coarse_step_uA": coarse_step_uA,
        "plateau_width_uA": plateau_width_uA,
        "min_sampled_max_fraction": min_sampled_max_fraction,
    }
    if any(not np.isfinite(float(value)) for value in values.values()):
        raise ValueError("I_max search parameters must be finite")
    if float(tol_uA) <= 0 or float(coarse_step_uA) <= 0 or float(plateau_width_uA) <= 0:
        raise ValueError("I_max search tolerances and widths must be positive")
    if not 0 < float(min_sampled_max_fraction) <= 1:
        raise ValueError("I_max sampled-maximum fraction must be in (0, 1]")

    lower = max(float(I_first), float(scan_start_uA if scan_start_uA is not None else I_first))
    maximum = float(amp_max_uA)
    if maximum <= lower:
        raise ValueError("I_max upper bound must exceed the scan start")

    # Anchor the diagnostic/production grid to absolute amplitude rather than
    # to the repetition-specific decimal-valued I_50 threshold. This makes the
    # plateau definition reproducible across both workflows.
    grid_start = float(np.ceil((lower - 1e-12) / float(coarse_step_uA)) * float(coarse_step_uA))
    grid = np.arange(grid_start, maximum + 0.5 * float(coarse_step_uA), float(coarse_step_uA))
    grid = np.unique(np.round(np.r_[grid[grid <= maximum], maximum], 9))
    sampled_counts = np.asarray([int(spike_counter(float(amplitude))) for amplitude in grid], dtype=int)
    sampled_maximum = int(sampled_counts.max(initial=0))
    if sampled_maximum <= 0:
        return None, 0
    minimum_count = max(
        1,
        int(np.ceil(float(min_sampled_max_fraction) * sampled_maximum - 1e-12)),
    )

    run_start = float(grid[0])
    run_start_index = 0
    run_count: int | None = None
    previous_amp: float | None = None
    plateau_target: int | None = None
    plateau_start_index: int | None = None

    for index, (amplitude, count) in enumerate(zip(grid, sampled_counts)):
        amplitude = float(amplitude)
        count = int(count)
        contiguous = previous_amp is None or amplitude - previous_amp <= 1.05 * float(coarse_step_uA)
        if run_count is None or count != run_count or not contiguous:
            run_start, run_count = amplitude, count
            run_start_index = int(index)
        previous_amp = amplitude
        if count >= minimum_count and amplitude - run_start >= float(plateau_width_uA):
            plateau_target = count
            plateau_start_index = run_start_index
            break

    if plateau_target is None or plateau_start_index is None:
        raise RuntimeError(
            "No stable near-maximal I_max plateau was found through "
            f"{maximum:g} uA (sampled maximum={sampled_maximum}, "
            f"required count={minimum_count})"
        )

    hi = float(grid[plateau_start_index])
    lo = lower if plateau_start_index == 0 else float(grid[plateau_start_index - 1])
    if int(spike_counter(lo)) >= plateau_target:
        # A preceding transient/nonmonotonic response is not the onset of the
        # selected stable segment. Retain the coarse segment boundary rather
        # than walking backward onto a different response branch.
        return hi, int(plateau_target)
    while hi - lo > float(tol_uA):
        mid = 0.5 * (lo + hi)
        if spike_counter(mid) >= plateau_target:
            hi = mid
        else:
            lo = mid
    return float(hi), int(plateau_target)


def _peak_first_imax_grid(
    scan_start_uA: float,
    *,
    step_uA: float = I_MAX_COARSE_STEP_UA,
    search_width_uA: float = I_MAX_BRANCH_SEARCH_WIDTH_UA,
    safety_bound_uA: float = float(ALL_SPIKES_SEARCH["amp_max_uA"]),
) -> np.ndarray:
    """Return the absolute 25-uA grid used to characterize the initial branch."""
    values = (scan_start_uA, step_uA, search_width_uA, safety_bound_uA)
    if any(not np.isfinite(float(value)) for value in values):
        raise ValueError("I_max branch-search parameters must be finite")
    if float(step_uA) <= 0 or float(search_width_uA) <= 0:
        raise ValueError("I_max branch-search step and width must be positive")
    start = float(np.ceil((float(scan_start_uA) - 1e-12) / float(step_uA)) * float(step_uA))
    stop = min(float(safety_bound_uA), start + float(search_width_uA))
    if stop <= start:
        raise ValueError("I_max branch-search safety bound must exceed its grid start")
    grid = np.arange(start, stop + 0.5 * float(step_uA), float(step_uA))
    return np.unique(np.round(np.r_[grid[grid <= stop], stop], 9))


def _imax_window_end(amplitudes: np.ndarray, start_index: int, width_uA: float) -> int | None:
    target = float(amplitudes[int(start_index)]) + float(width_uA) - 1e-9
    index = int(np.searchsorted(amplitudes, target, side="left"))
    return index if index < len(amplitudes) else None


def _first_sustained_imax_decline(
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


def _stable_imax_window(
    amplitudes: np.ndarray,
    counts: np.ndarray,
    *,
    start_index: int,
    candidate_width_uA: float,
    confirmation_width_uA: float,
    tolerance_spikes: int,
) -> dict[str, int] | None:
    candidate_end = _imax_window_end(amplitudes, start_index, candidate_width_uA)
    total_end = _imax_window_end(
        amplitudes, start_index, candidate_width_uA + confirmation_width_uA,
    )
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


def _classify_peak_first_imax_plateau(
    amplitudes: Sequence[float],
    counts: Sequence[int],
    *,
    n_input_pulses: int,
    tolerance_fraction: float,
    candidate_width_uA: float = I_MAX_PLATEAU_CANDIDATE_WIDTH_UA,
    confirmation_width_uA: float = I_MAX_PLATEAU_CONFIRMATION_WIDTH_UA,
    decline_width_uA: float = I_MAX_DECLINE_WIDTH_UA,
) -> dict[str, Any]:
    """Find the first stable plateau that is complete before sustained decline."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if amplitudes.ndim != 1 or counts.ndim != 1 or len(amplitudes) != len(counts) or not len(counts):
        raise ValueError("amplitudes and counts must be nonempty equal-length vectors")
    if np.any(np.diff(amplitudes) <= 0):
        raise ValueError("amplitudes must be strictly increasing")
    if int(n_input_pulses) <= 0 or float(tolerance_fraction) < 0:
        raise ValueError("n_input_pulses must be positive and tolerance_fraction nonnegative")
    tolerance = (
        0 if float(tolerance_fraction) == 0
        else max(1, int(np.ceil(float(tolerance_fraction) * int(n_input_pulses))))
    )
    decline_index, decline_samples = _first_sustained_imax_decline(
        amplitudes,
        counts,
        tolerance_spikes=tolerance,
        decline_width_uA=float(decline_width_uA),
    )
    accepted = None
    for start_index in range(len(amplitudes)):
        candidate = _stable_imax_window(
            amplitudes,
            counts,
            start_index=start_index,
            candidate_width_uA=float(candidate_width_uA),
            confirmation_width_uA=float(confirmation_width_uA),
            tolerance_spikes=tolerance,
        )
        if candidate is None:
            continue
        if decline_index is not None and decline_index <= candidate["total_end_index"]:
            break
        accepted = candidate
        break
    result: dict[str, Any] = {
        "classification": (
            "stable_predecline_plateau" if accepted is not None
            else "blocked_before_plateau" if decline_index is not None
            else "no_plateau_by_branch_bound"
        ),
        "tolerance_fraction": float(tolerance_fraction),
        "tolerance_spikes": int(tolerance),
        "plateau_start_uA": np.nan,
        "plateau_candidate_end_uA": np.nan,
        "plateau_confirmed_through_uA": np.nan,
        "plateau_count": np.nan,
        "plateau_min_count": np.nan,
        "plateau_max_count": np.nan,
        "first_sustained_decline_uA": (
            np.nan if decline_index is None else float(amplitudes[decline_index])
        ),
        "decline_confirmation_samples": int(decline_samples),
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
    return result


def _early_peak_first_imax_candidate(
    amplitudes: Sequence[float],
    counts: Sequence[int],
    *,
    n_input_pulses: int,
    search_width_uA: float = I_MAX_EARLY_PEAK_SEARCH_WIDTH_UA,
    tail_samples: int = I_MAX_EARLY_PEAK_TAIL_SAMPLES,
    minimum_drop_spikes: int = I_MAX_EARLY_PEAK_MINIMUM_DROP_SPIKES,
    minimum_rise_fraction: float = I_MAX_EARLY_PEAK_MINIMUM_RISE_FRACTION,
) -> dict[str, Any]:
    """Return a clear first-lobe peak, excluding later suprastimulation maxima."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    counts = np.asarray(counts, dtype=int)
    if amplitudes.ndim != 1 or counts.ndim != 1 or amplitudes.size != counts.size:
        raise ValueError("amplitudes and counts must be equal-length one-dimensional arrays")
    if amplitudes.size < int(tail_samples) + 2 or np.any(np.diff(amplitudes) <= 0):
        raise ValueError("peak search requires a strictly increasing grid with enough samples")
    if int(n_input_pulses) <= 0 or int(tail_samples) <= 0:
        raise ValueError("n_input_pulses and tail_samples must be positive")
    eligible = np.flatnonzero(amplitudes <= float(amplitudes[0]) + float(search_width_uA) + 1e-9)
    if eligible.size < int(tail_samples) + 2:
        raise ValueError("initial peak-search window contains too few samples")
    stop = int(eligible[-1]) + 1
    window_counts = counts[:stop]
    peak_count = int(window_counts.max())
    peak_index = int(np.flatnonzero(window_counts == peak_count)[0])
    tail_level = float(np.median(window_counts[-int(tail_samples):]))
    rise = int(peak_count - int(window_counts[0]))
    drop = float(peak_count - tail_level)
    minimum_rise = max(1, int(np.ceil(float(minimum_rise_fraction) * int(n_input_pulses))))
    return {
        "is_clear_peak": bool(
            peak_index < stop - int(tail_samples)
            and rise >= minimum_rise
            and drop >= int(minimum_drop_spikes)
        ),
        "peak_index": int(peak_index),
        "peak_amplitude_uA": float(amplitudes[peak_index]),
        "peak_count": int(peak_count),
        "rise_spikes": int(rise),
        "terminal_drop_spikes": float(drop),
        "search_end_uA": float(amplitudes[stop - 1]),
    }


def _select_peak_first_imax_coarse(
    amplitudes: Sequence[float],
    counts: Sequence[int],
    *,
    n_input_pulses: int,
) -> dict[str, Any]:
    """Apply the validated peak-first rule and strictest-valid plateau fallback."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    counts = np.asarray(counts, dtype=int)
    peak = _early_peak_first_imax_candidate(
        amplitudes, counts, n_input_pulses=int(n_input_pulses),
    )
    plateau = None
    if not bool(peak["is_clear_peak"]):
        for tolerance_fraction in I_MAX_PLATEAU_TOLERANCE_FRACTIONS:
            candidate = _classify_peak_first_imax_plateau(
                amplitudes,
                counts,
                n_input_pulses=int(n_input_pulses),
                tolerance_fraction=float(tolerance_fraction),
            )
            if candidate["classification"] == "stable_predecline_plateau":
                plateau = candidate
                break
    if bool(peak["is_clear_peak"]):
        selected_index = int(peak["peak_index"])
        target_count = int(peak["peak_count"])
        method = "early_response_peak"
    elif plateau is not None:
        target_count = int(plateau["plateau_count"])
        # The stable window establishes which response count defines I_max.
        # Resolve its threshold at the first attainment on this same initial
        # branch, including a brief earlier attainment followed by a
        # non-sustained one-spike dip. ``_classify_peak_first_imax_plateau``
        # already rejects any stable window separated from the rising branch
        # by a sustained decline, so this cannot walk back onto a distinct
        # response branch.
        eligible = np.flatnonzero(counts >= target_count)
        if not eligible.size:
            raise RuntimeError("Stable I_max plateau never attains its representative count")
        selected_index = int(eligible[0])
        method = "stable_saturation_plateau"
    else:
        raise RuntimeError("Peak-first I_max search found neither a clear peak nor a stable plateau")
    return {
        "selection_method": method,
        "coarse_selected_index": int(selected_index),
        "coarse_selected_amplitude_uA": float(amplitudes[selected_index]),
        "coarse_selected_spikes": int(counts[selected_index]),
        "target_spikes": int(target_count),
        "peak": peak,
        "plateau": plateau,
    }


def _find_cached_peak_first_threshold(
    spike_counter: Any,
    cache: dict[float, tuple[np.ndarray, int]],
    *,
    scan_start_uA: float,
    n_input_pulses: int,
    tol_uA: float = POPULATION_THRESHOLD_TOL_UA,
    step_uA: float = I_MAX_COARSE_STEP_UA,
    search_width_uA: float = I_MAX_BRANCH_SEARCH_WIDTH_UA,
    safety_bound_uA: float = float(ALL_SPIKES_SEARCH["amp_max_uA"]),
) -> tuple[float, int, dict[str, Any]]:
    """Select I_max on the first response branch and refine its local onset."""
    if not np.isfinite(float(tol_uA)) or float(tol_uA) <= 0:
        raise ValueError("I_max threshold tolerance must be finite and positive")
    grid = _peak_first_imax_grid(
        float(scan_start_uA),
        step_uA=float(step_uA),
        search_width_uA=float(search_width_uA),
        safety_bound_uA=float(safety_bound_uA),
    )
    counts = np.asarray([int(spike_counter(float(amplitude))) for amplitude in grid], dtype=int)
    selection = _select_peak_first_imax_coarse(
        grid, counts, n_input_pulses=int(n_input_pulses),
    )
    selected_index = int(selection["coarse_selected_index"])
    target = int(selection["target_spikes"])
    hi = float(grid[selected_index])
    lo = float(scan_start_uA) if selected_index == 0 else float(grid[selected_index - 1])
    lower_count = int(spike_counter(lo))
    if selected_index == 0 and lower_count >= target:
        # I_50 is the lower boundary of the eligible branch. If it already
        # realizes the selected plateau count, it is the resolved local onset.
        hi = lo
    refinement_performed = bool(lower_count < target and hi > lo)
    if refinement_performed:
        while hi - lo > float(tol_uA):
            mid = 0.5 * (lo + hi)
            if int(spike_counter(mid)) >= target:
                hi = mid
            else:
                lo = mid
    amplitude = float(hi)
    realized = int(spike_counter(amplitude))
    if realized < target:
        raise RuntimeError("Refined I_max amplitude does not attain the selected response count")
    metadata = {
        **selection,
        "grid_start_uA": float(grid[0]),
        "grid_end_uA": float(grid[-1]),
        "coarse_grid_samples": int(len(grid)),
        "refinement_performed": bool(refinement_performed),
        "refinement_lower_bound_uA": float(lo),
        "refined_amplitude_uA": amplitude,
        "refined_spikes": int(realized),
    }
    return amplitude, int(realized), metadata


def _refine_cached_targets(
    spike_counter: Any,
    cache: dict[float, tuple[np.ndarray, int]],
    targets: Iterable[int],
    *,
    lower_bound: float,
    tol_uA: float,
) -> dict[int, tuple[float | None, int]]:
    """Jointly refine several spike-count thresholds using one shared cache.

    Targets sharing a sampled amplitude interval share its midpoint execution.
    Every iteration rebuilds brackets from all newly cached observations, so a
    simulation performed for one regime can tighten every other regime.
    """
    unique_targets = sorted({int(target) for target in targets})
    resolved: dict[int, tuple[float | None, int]] = {}
    for _ in range(100):
        pending: dict[tuple[float, float], list[int]] = defaultdict(list)
        for target in unique_targets:
            bracket = _first_cached_target_bracket(cache, target, lower_bound=float(lower_bound))
            if bracket is None:
                resolved[target] = (None, 0)
                continue
            lo, hi = bracket
            if hi - lo <= float(tol_uA):
                resolved[target] = (float(hi), int(spike_counter(hi)))
            else:
                pending[(float(lo), float(hi))].append(target)
        if not pending:
            return resolved
        for lo, hi in pending:
            spike_counter(0.5 * (lo + hi))
    raise RuntimeError("Joint regime refinement did not converge after 100 iterations")


def _compute_regimes(
    time_ms: np.ndarray,
    V_base: np.ndarray,
    current: np.ndarray,
    *,
    config_path: Path,
    readout_node: int,
    n_input_pulses: int,
    tol_uA: float | None = POPULATION_THRESHOLD_TOL_UA,
) -> tuple[dict[str, Any], dict[float, tuple[np.ndarray, int]]]:
    """Apply the single-axon search definitions through a population runner.

    Production population searches use a 0.01-uA stopping tolerance.  Passing
    ``None`` explicitly restores the imported single-axon search tolerances;
    the override remains available for diagnostic sensitivity analyses.
    """
    time_trim, V_trim = trim_tail_window_ms(time_ms, V_base, current)
    time_trim, V_trim = prepend_zeros_ms(time_trim, V_trim, prepad_ms=PREPAD_MS)
    runner = PopulationMRGRunner(time_trim, V_trim, config_path=config_path, readout_node_number=readout_node)
    cache: dict[float, tuple[np.ndarray, int]] = {}
    def count(amp: float) -> int:
        amp = float(amp)
        if amp not in cache:
            cache[amp] = runner.run_mrg_for_amp(amp)
        return cache[amp][1]

    first_search = dict(FIRST_SPIKE_SEARCH)
    all_search = dict(ALL_SPIKES_SEARCH)
    if tol_uA is not None:
        if not np.isfinite(tol_uA) or float(tol_uA) <= 0:
            raise ValueError("tol_uA must be finite and positive")
        first_search["tol_uA"] = float(tol_uA)
        all_search["tol_uA"] = float(tol_uA)

    result: dict[str, Any] = {}
    I_first, N_first, *_ = find_first_spike_threshold(count, **first_search)
    if I_first is None:
        for name in THRESH_TYPES:
            result[f"{name}_uA"], result[f"N_{name}_spikes"] = None, 0
        result["N_max_spikes"] = 0
        return result, cache
    result["I_first_uA"], result["N_I_first_spikes"] = float(I_first), int(N_first)
    named_targets = {
        "I_volley5": int(VOLLEY5_MIN_SPIKES),
        "I_volley10": int(VOLLEY10_MIN_SPIKES),
        "I_5": max(1, int(np.ceil(.05 * n_input_pulses))),
        "I_10": max(1, int(np.ceil(.10 * n_input_pulses))),
        "I_15": max(1, int(np.ceil(.15 * n_input_pulses))),
        "I_25": max(1, int(np.ceil(.25 * n_input_pulses))),
        "I_50": max(1, int(np.ceil(.50 * n_input_pulses))),
    }
    # Supply a guaranteed high-amplitude bracket before resolving the fixed
    # count regimes. Their target definitions are unchanged; the later I_max
    # scan is not needed to make them solvable.
    count(float(all_search["amp_max_uA"]))
    refined = _refine_cached_targets(
        count,
        cache,
        named_targets.values(),
        lower_bound=float(I_first),
        tol_uA=float(all_search["tol_uA"]),
    )
    for name, target in named_targets.items():
        amp, n_spikes = refined[target]
        result[f"{name}_uA"], result[f"N_{name}_spikes"] = amp, n_spikes

    scan_start = result.get("I_50_uA")
    if scan_start is None or not np.isfinite(float(scan_start)):
        scan_start = I_first
    I_all, N_max, _ = _find_cached_peak_first_threshold(
        count,
        cache,
        scan_start_uA=scan_start,
        n_input_pulses=int(n_input_pulses),
        tol_uA=float(all_search["tol_uA"]),
        safety_bound_uA=float(all_search["amp_max_uA"]),
    )
    result["I_all_uA"] = float(I_all)
    result["N_I_all_spikes"] = int(count(float(I_all)))
    result["N_max_spikes"] = int(N_max)

    return result, cache


def _index_regime_job(root: Path, index: dict[str, Any], cv: float, freq: float, rep_index: int) -> list[dict[str, Any]]:
    npz_path = spacetime_path(
        root / "data" / "space-time-V", str(index["cutline_id"]),
        float(index["diameter_um"]), cv, freq, axon_id=str(index["axon_id"]),
    )
    rep = load_population_rep(npz_path, rep_index)
    config = _config_path(root, rep["diameter_um"], rep["n_valid_nodes"])
    regimes, cache = _compute_regimes(rep["time_ms"], rep["V_base"], rep["current"], config_path=config, readout_node=rep["readout_node_number"], n_input_pulses=int(round(freq * 2.0)))
    rows = []
    for thresh_type in THRESH_TYPES:
        amp = regimes.get(f"{thresh_type}_uA")
        if amp is None or pd.isna(amp):
            continue
        spikes, n_spikes = cache[float(amp)]
        rows.append({
            "detection_protocol_version": DETECTION_PROTOCOL_VERSION,
            "index_axon_id": rep["axon_id"], "index_cutline_id": rep["cutline_id"], "index_diameter_um": rep["diameter_um"],
            "file": npz_path.name, "cv": cv, "freq": freq, "rep_index": rep_index, "rep_label": rep["rep_label"],
            "n_input_pulses": int(round(freq * 2.0)), "thresh_type": thresh_type, "amp_uA": float(amp),
            "n_index_spikes": int(n_spikes), "index_spike_times_ms": json.dumps(np.asarray(spikes, dtype=float).tolist()),
            "N_max_spikes": int(regimes["N_max_spikes"]),
        })
    return rows


def index_regime_conditions() -> list[tuple[float, float]]:
    """Return the stable CV/frequency order used by the index-regime array."""
    return [(cv, freq) for cv in POPULATION_CVS for freq in POPULATION_FREQS]


def _atomic_dataframe_csv(table: pd.DataFrame, output: Path) -> Path:
    """Write a complete CSV before exposing it to an array-job merge step."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(output)
    return output


def _index_regime_partial_path(partial_dir: Path, cv: float, freq: float) -> Path:
    return partial_dir / f"index_regimes_cv{cv:.1f}_f{int(freq)}.csv"


def run_index_regime_condition(
    *, root: Path = PROJECT_ROOT, condition_index: int, n_jobs: int = 1,
    partial_dir: Path | None = None, overwrite: bool = False,
) -> Path:
    """Compute and persist one CV/frequency condition (all its repetitions).

    This is the atomic unit for the Longleaf index-regime array.  A failed
    task leaves no CSV that can be mistaken for a completed condition.
    """
    root = root.resolve()
    conditions = index_regime_conditions()
    if not 0 <= condition_index < len(conditions):
        raise IndexError(f"condition_index must be in [0, {len(conditions) - 1}]")
    cv, freq = conditions[condition_index]
    partial_dir = (partial_dir or _detection_partial_dir(root, "index-regimes")).resolve()
    output = _index_regime_partial_path(partial_dir, cv, freq)
    if output.is_file() and not overwrite:
        return output

    index = identify_index_axon(root=root, n_jobs=n_jobs, force=False)
    index_path = spacetime_path(
        root / "data" / "space-time-V", str(index["cutline_id"]),
        float(index["diameter_um"]), cv, freq, axon_id=str(index["axon_id"]),
    )
    repetition_groups = _repetition_equivalence_groups(index_path)
    rows_by_job = Parallel(n_jobs=max(1, n_jobs), prefer="processes")(
        delayed(_index_regime_job)(root, index, cv, freq, members[0][0])
        for members in repetition_groups
    )
    rows = [
        row
        for representative_rows, members in zip(rows_by_job, repetition_groups)
        for row in _replicate_rows_for_repetitions(representative_rows, members)
    ]
    rows.sort(key=lambda row: (int(row["rep_index"]), THRESH_TYPES.index(str(row["thresh_type"]))))
    return _atomic_dataframe_csv(pd.DataFrame(rows, columns=REGIME_COLUMNS), output)


def merge_index_regime_partials(
    *, root: Path = PROJECT_ROOT, partial_dir: Path | None = None, output_csv: Path | None = None,
) -> Path:
    """Validate and merge all 55 index-regime array outputs."""
    root = root.resolve()
    partial_dir = (partial_dir or _detection_partial_dir(root, "index-regimes")).resolve()
    output_csv = output_csv or root / "data" / "detection" / INDEX_REGIMES_CSV.name
    expected = [_index_regime_partial_path(partial_dir, cv, freq) for cv, freq in index_regime_conditions()]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot merge index regimes; {len(missing)} condition files are missing (first: {missing[0]})")
    tables = [pd.read_csv(path) for path in expected]
    merged = pd.concat(tables, ignore_index=True)
    duplicate_keys = ["index_axon_id", "cv", "freq", "rep_index", "thresh_type"]
    if merged.duplicated(duplicate_keys).any():
        raise RuntimeError("Index-regime partials contain duplicate condition/repetition/regime rows")
    _atomic_dataframe_csv(merged.loc[:, REGIME_COLUMNS], output_csv)
    return output_csv


def compute_index_regimes(*, root: Path = PROJECT_ROOT, n_jobs: int = 1, force: bool = False) -> pd.DataFrame:
    """Compute one index-anchored regime-amplitude row per freq/CV/repetition.

    Local execution uses the same condition partials as the cluster workflow,
    so it can resume after interruption rather than discarding prior work.
    """
    root = root.resolve(); output = root / "data" / "detection" / INDEX_REGIMES_CSV.name
    if output.is_file() and not force:
        return pd.read_csv(output)
    partial_dir = _detection_partial_dir(root, "index-regimes")
    for condition_index in tqdm(range(len(index_regime_conditions())), desc="Index regime conditions"):
        run_index_regime_condition(
            root=root, condition_index=condition_index, n_jobs=n_jobs,
            partial_dir=partial_dir, overwrite=force,
        )
    merge_index_regime_partials(root=root, partial_dir=partial_dir, output_csv=output)
    return pd.read_csv(output)


# --------------- Index CV0-max amplitudes --------------- #

def compute_index_cv0_threshold_max(*, root: Path = PROJECT_ROOT, force: bool = False) -> pd.DataFrame:
    """Build one canonical CV0 amplitude per frequency and named regime.

    The legacy filename is retained for downstream compatibility.  All nine
    regimes now come from the one deterministic CV0 waveform; no all-CV onset
    safeguard is applied.  The replicated CV0 rows must agree exactly.
    """
    root = root.resolve()
    output = root / "data" / "detection" / INDEX_CV0_THRESH_CSV.name
    amplitude_columns = {
        "I_first": "I_first0_uA", "I_volley5": "I_volley5_0_uA",
        "I_volley10": "I_volley10_0_uA", "I_5": "I_5_0_uA",
        "I_10": "I_10_0_uA", "I_15": "I_15_0_uA",
        "I_25": "I_25_0_uA", "I_50": "I_50_0_uA", "I_all": "I_all0_uA",
    }
    count_columns = {
        regime: f"{column.removesuffix('_uA')}_n_index_spikes"
        for regime, column in amplitude_columns.items()
    }
    required_cached_columns = set(amplitude_columns.values()) | set(count_columns.values()) | {"N_max_spikes"}
    if output.is_file() and not force:
        cached = pd.read_csv(output)
        if required_cached_columns.issubset(cached.columns):
            return cached
    regimes = compute_index_regimes(root=root, n_jobs=1, force=False)
    id_columns = [
        "detection_protocol_version", "index_axon_id", "index_cutline_id",
        "index_diameter_um", "freq",
    ]
    rows: pd.DataFrame | None = None
    for thresh_type in THRESH_TYPES:
        source = regimes[
            (regimes["thresh_type"] == thresh_type)
            & np.isclose(regimes["cv"].astype(float), 0.0)
        ]
        column = amplitude_columns[thresh_type]
        count_column = count_columns[thresh_type]
        grouped = source.groupby(id_columns, as_index=False)
        amplitude_spread = grouped["amp_uA"].agg(
            lambda values: float(np.ptp(np.asarray(values, dtype=float)))
        )
        count_spread = grouped["n_index_spikes"].agg(
            lambda values: float(np.ptp(np.asarray(values, dtype=float)))
        )
        if bool((amplitude_spread["amp_uA"] > 1e-12).any()) or bool(
            (count_spread["n_index_spikes"] > 0).any()
        ):
            raise RuntimeError(
                f"Canonical CV0 repetitions disagree for {thresh_type}; "
                "regenerate or invalidate stale index-regime partials"
            )
        aggregate = grouped.agg(
            **{
                column: ("amp_uA", "first"),
                count_column: ("n_index_spikes", "first"),
            }
        )
        rows = aggregate if rows is None else rows.merge(aggregate, on=id_columns, how="outer")
    if rows is None or rows.empty:
        raise RuntimeError("No index regimes are available to build the CV0 threshold-max table")
    cv0 = regimes[np.isclose(regimes["cv"].astype(float), 0.0)]
    nmax_grouped = cv0.groupby(id_columns, as_index=False)
    nmax_spread = nmax_grouped["N_max_spikes"].agg(
        lambda values: float(np.ptp(np.asarray(values, dtype=float)))
    )
    if bool((nmax_spread["N_max_spikes"] > 0).any()):
        raise RuntimeError("Canonical CV0 N_max_spikes values disagree")
    nmax = nmax_grouped.agg(N_max_spikes=("N_max_spikes", "first"))
    rows = rows.merge(nmax, on=id_columns, how="left", validate="one_to_one")
    rows = rows.sort_values(id_columns).reset_index(drop=True)
    rows.to_csv(output, index=False)
    return rows


def _build_cv0_regime_jobs(root: Path, cv0_table: pd.DataFrame) -> pd.DataFrame:
    """Expand one per-frequency CV0 amplitude table to every CV/repetition."""
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    rows: list[dict[str, Any]] = []
    amp_columns = {
        "I_first": "I_first0_uA", "I_volley5": "I_volley5_0_uA", "I_volley10": "I_volley10_0_uA",
        "I_5": "I_5_0_uA", "I_10": "I_10_0_uA", "I_15": "I_15_0_uA", "I_25": "I_25_0_uA",
        "I_50": "I_50_0_uA", "I_all": "I_all0_uA",
    }
    for table_row in cv0_table.itertuples(index=False):
        freq = float(table_row.freq)
        for cv in tuple(round(.1 * i, 1) for i in range(11)):
            index_path = spacetime_path(
                root / "data" / "space-time-V", str(index["cutline_id"]),
                float(index["diameter_um"]), cv, freq, axon_id=str(index["axon_id"]),
            )
            with np.load(index_path, allow_pickle=False) as data:
                rep_labels = [str(label) for label in data["rep_labels"]]
            for rep_index, rep_label in enumerate(rep_labels):
                for thresh_type, column in amp_columns.items():
                    amp = getattr(table_row, column, np.nan)
                    if pd.isna(amp):
                        continue
                    rows.append({
                        "detection_protocol_version": str(table_row.detection_protocol_version),
                        "index_axon_id": index["axon_id"], "index_cutline_id": index["cutline_id"],
                        "index_diameter_um": index["diameter_um"], "file": index_path.name,
                        "cv": cv, "freq": freq, "rep_index": rep_index, "rep_label": rep_label,
                        "n_input_pulses": int(round(freq * 2.0)), "thresh_type": thresh_type,
                        "amp_uA": float(amp),
                        # Unknown until the index axon is simulated under this
                        # CV/repetition. Its own population-response rows carry
                        # the actual value, which feature extraction propagates.
                        "n_index_spikes": np.nan,
                        "index_spike_times_ms": "[]",
                        "N_max_spikes": int(getattr(table_row, "N_max_spikes")),
                    })
    return pd.DataFrame(rows, columns=REGIME_COLUMNS)


# --------------- Population recruitment at index amplitudes --------------- #

def _population_basis_paths(root: Path, population_manifest: Path | None = None) -> list[Path]:
    """Select population basis files, optionally from an explicit cohort CSV.

    The optional CSV needs an ``axon_id`` column and may contain an ``include``
    column.  This lets a new COMSOL cohort coexist with older basis files while
    keeping explicitly selected cohorts isolated from older files.
    """
    basis_dir = root / "data" / "space-time-V"
    available = {path.name.removesuffix("_1uA_VeBasis.npz"): path for path in basis_dir.glob("*_1uA_VeBasis.npz")}
    if not available:
        raise FileNotFoundError("No population Ve basis files were found")
    if population_manifest is None:
        default_manifest = basis_dir / "axon_basis_manifest.csv"
        if not default_manifest.is_file():
            return [available[key] for key in sorted(available)]
        population_manifest = default_manifest

    manifest = Path(population_manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    table = pd.read_csv(manifest)
    if "axon_id" not in table.columns:
        raise ValueError(f"Population manifest must contain an axon_id column: {manifest}")
    if "include" in table.columns:
        include = table["include"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
        table = table.loc[include]
    ids = table["axon_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Population manifest contains duplicate axon_id values: {manifest}")
    missing = [axon_id for axon_id in ids if axon_id not in available]
    if missing:
        raise FileNotFoundError(f"Population manifest references missing Ve basis {missing[0]!r} ({len(missing)} missing)")
    if not ids:
        raise ValueError(f"Population manifest selects no axons: {manifest}")
    return [available[axon_id] for axon_id in ids]


def population_recruitment_jobs(
    root: Path = PROJECT_ROOT, population_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a stable target-axon/CV/frequency manifest for SLURM arrays."""
    root = root.resolve()
    basis_paths = _population_basis_paths(root, population_manifest)
    jobs: list[dict[str, Any]] = []
    for basis_path in basis_paths:
        basis = load_population_basis(basis_path)
        for cv, freq in index_regime_conditions():
            jobs.append({
                "target_axon_id": basis["axon_id"], "target_cutline_id": basis["cutline_id"],
                "target_diameter_um": basis["diameter_um"], "basis_path": str(basis_path),
                "cv": cv, "freq": freq,
            })
    return jobs


def _cv0_regimes_for_condition(root: Path, cv: float, freq: float) -> pd.DataFrame:
    """Build only one expanded CV0 condition, avoiding a 49,500-row table per task."""
    from detection.accepted_cv0 import INDEX_SHA256, checked_csv

    cv0 = checked_csv(
        root / "data/detection/index_cv0_threshold_max.csv",
        INDEX_SHA256, normalize_newlines=True,
    )
    protocols = cv0["detection_protocol_version"].dropna().astype(str).unique().tolist()
    if protocols != [CV0_PROTOCOL_VERSION]:
        raise RuntimeError(
            f"CV0 recruitment requires {CV0_PROTOCOL_VERSION}; found {protocols}. "
            "Run detection.accepted_cv0 prepare before population CV0 jobs."
        )
    row = cv0[np.isclose(cv0["freq"].astype(float), float(freq))]
    if len(row) != 1:
        raise RuntimeError(f"Expected one CV0 threshold row for frequency {freq}, found {len(row)}")
    table_row = row.iloc[0].to_dict()
    index = identify_index_axon(root=root, n_jobs=1, force=False)
    index_path = spacetime_path(
        root / "data" / "space-time-V", str(index["cutline_id"]),
        float(index["diameter_um"]), cv, freq, axon_id=str(index["axon_id"]),
    )
    with np.load(index_path, allow_pickle=False) as data:
        labels = [str(label) for label in data["rep_labels"]]
    amp_columns = {
        "I_first": "I_first0_uA", "I_volley5": "I_volley5_0_uA", "I_volley10": "I_volley10_0_uA",
        "I_5": "I_5_0_uA", "I_10": "I_10_0_uA", "I_15": "I_15_0_uA", "I_25": "I_25_0_uA",
        "I_50": "I_50_0_uA", "I_all": "I_all0_uA",
    }
    rows: list[dict[str, Any]] = []
    for rep_index, rep_label in enumerate(labels):
        for thresh_type, column in amp_columns.items():
            amp = table_row.get(column, np.nan)
            if pd.isna(amp):
                continue
            rows.append({
                "detection_protocol_version": str(table_row["detection_protocol_version"]),
                "index_axon_id": index["axon_id"], "index_cutline_id": index["cutline_id"],
                "index_diameter_um": index["diameter_um"], "file": index_path.name,
                "cv": cv, "freq": freq, "rep_index": rep_index, "rep_label": rep_label,
                "n_input_pulses": int(round(freq * 2.0)), "thresh_type": thresh_type,
                "amp_uA": float(amp),
                "n_index_spikes": np.nan,
                "index_spike_times_ms": "[]",
                "N_max_spikes": int(table_row["N_max_spikes"]),
            })
    return pd.DataFrame(rows, columns=REGIME_COLUMNS)


def _regimes_for_population_condition(root: Path, mode: str, cv: float, freq: float) -> pd.DataFrame:
    if mode == "true":
        regimes = compute_index_regimes(root=root, n_jobs=1, force=False)
        selected = regimes[np.isclose(regimes["cv"].astype(float), cv) & np.isclose(regimes["freq"].astype(float), freq)]
        if selected.empty:
            raise RuntimeError(f"No true index regimes for CV={cv}, frequency={freq}")
        return selected.loc[:, REGIME_COLUMNS].copy()
    if mode == "cv0":
        return _cv0_regimes_for_condition(root, cv, freq)
    raise ValueError("mode must be 'true' or 'cv0'")


def _canonical_regime_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _regime_repetition_signature(rows: Sequence[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    """Canonicalize every non-identity field before response deduplication."""
    fields = tuple(column for column in REGIME_COLUMNS if column not in {"rep_index", "rep_label"})
    return tuple(
        tuple(_canonical_regime_value(row.get(field)) for field in fields)
        for row in sorted(rows, key=lambda row: str(row["thresh_type"]))
    )


def _equivalent_regime_repetition_groups(
    npz_path: Path,
    regimes_by_rep: dict[int, list[dict[str, Any]]],
    screen_failures: dict[int, bool] | None = None,
) -> list[list[tuple[int, str]]]:
    """Group only repetitions with identical currents and regime metadata."""
    screen_failures = screen_failures or {}
    groups: list[list[tuple[int, str]]] = []
    for waveform_members in _repetition_equivalence_groups(npz_path):
        matching: dict[tuple[Any, ...], list[tuple[int, str]]] = {}
        for rep_index, rep_label in waveform_members:
            if rep_index not in regimes_by_rep:
                raise RuntimeError(f"Regime table lacks repetition {rep_index}")
            signature = (
                _regime_repetition_signature(regimes_by_rep[rep_index]),
                bool(screen_failures.get(rep_index, False)),
            )
            matching.setdefault(signature, []).append((rep_index, rep_label))
        groups.extend(matching.values())
    return groups


def _stable_audit_sample(rate: float, *parts: object) -> bool:
    """Select a deterministic fraction of work for pruning validation."""
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("Pruning audit rates must be between 0 and 1")
    if rate == 0.0:
        return False
    if rate == 1.0:
        return True
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(2**64)
    return draw < rate


def _evaluate_regime_amplitudes(
    runner: Any,
    regime_rows: Sequence[dict[str, Any]],
    *,
    prune_on_max_failure: bool,
    audit_max_failure: bool,
    expected_max_failure: bool,
) -> tuple[dict[float, tuple[np.ndarray, int]], dict[str, int]]:
    """Evaluate unique amplitudes largest-first and optionally prune failures.

    Exact duplicate amplitudes always share one NEURON execution.  When the
    maximum produces no terminal spike, lower amplitudes may be filled with
    zeros.  Deterministically sampled audits execute those lower amplitudes and
    fail loudly if the empirically validated monotonicity assumption is broken.
    ``expected_max_failure`` similarly audits a larger-diameter screen.
    """
    amplitudes = sorted({float(row["amp_uA"]) for row in regime_rows}, reverse=True)
    if not amplitudes:
        raise ValueError("At least one regime amplitude is required")
    results: dict[float, tuple[np.ndarray, int]] = {}
    neuron_runs = 0

    maximum = amplitudes[0]
    spike_times, n_spikes = runner.run_mrg_for_amp(maximum)
    results[maximum] = (np.asarray(spike_times, dtype=float), int(n_spikes))
    neuron_runs += 1
    if expected_max_failure and n_spikes > 0:
        raise RuntimeError(
            "Diameter-pruning audit failed: a smaller-diameter axon responded "
            "after the adjacent larger diameter failed at the same maximum amplitude"
        )

    max_failed = n_spikes == 0
    should_prune = prune_on_max_failure and max_failed
    if should_prune and not audit_max_failure:
        for amplitude in amplitudes[1:]:
            results[amplitude] = (np.empty(0, dtype=float), 0)
    else:
        for amplitude in amplitudes[1:]:
            spike_times, n_spikes = runner.run_mrg_for_amp(amplitude)
            results[amplitude] = (np.asarray(spike_times, dtype=float), int(n_spikes))
            neuron_runs += 1
            if should_prune and n_spikes > 0:
                raise RuntimeError(
                    "Maximum-amplitude pruning audit failed: a lower amplitude "
                    "responded after the maximum amplitude failed"
                )

    return results, {
        "regime_rows": len(regime_rows),
        "unique_amplitudes": len(amplitudes),
        "neuron_runs": neuron_runs,
        "duplicate_runs_saved": len(regime_rows) - len(amplitudes),
        "max_failure_pruned": int(should_prune and not audit_max_failure),
        "max_failure_audited": int(should_prune and audit_max_failure),
        "diameter_failure_audited": int(expected_max_failure),
    }


def _zero_regime_results(regime_rows: Sequence[dict[str, Any]]) -> tuple[dict[float, tuple[np.ndarray, int]], dict[str, int]]:
    """Return schema-compatible zero results for a diameter-pruned repetition."""
    amplitudes = {float(row["amp_uA"]) for row in regime_rows}
    return (
        {amplitude: (np.empty(0, dtype=float), 0) for amplitude in amplitudes},
        {
            "regime_rows": len(regime_rows),
            "unique_amplitudes": len(amplitudes),
            "neuron_runs": 0,
            "duplicate_runs_saved": len(regime_rows) - len(amplitudes),
            "max_failure_pruned": 0,
            "max_failure_audited": 0,
            "diameter_failure_audited": 0,
        },
    )


def _population_response_rows_for_rep(
    root: Path,
    job: dict[str, Any],
    regime_rows: list[dict[str, Any]],
    *,
    prune_on_max_failure: bool = True,
    audit_max_failure: bool = False,
    diameter_screen_failed: bool = False,
    audit_diameter_failure: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Simulate one target axon's one repetition at every defined regime."""
    cv, freq = float(job["cv"]), float(job["freq"])
    basis = load_population_basis(Path(job["basis_path"]))
    npz_path = spacetime_path(
        root / "data" / "space-time-V", basis["cutline_id"], basis["diameter_um"],
        cv, freq, axon_id=str(basis["axon_id"]),
    )
    rep_index = int(regime_rows[0]["rep_index"])
    if diameter_screen_failed and not audit_diameter_failure:
        results, stats = _zero_regime_results(regime_rows)
        stats["diameter_failure_pruned"] = 1
    else:
        rep = load_population_rep(npz_path, rep_index)
        time_trim, V_trim = trim_tail_window_ms(rep["time_ms"], rep["V_base"], rep["current"])
        time_trim, V_trim = prepend_zeros_ms(time_trim, V_trim, prepad_ms=PREPAD_MS)
        config = _config_path(root, rep["diameter_um"], rep["n_valid_nodes"])
        runner = PopulationMRGRunner(time_trim, V_trim, config_path=config, readout_node_number=rep["readout_node_number"])
        results, stats = _evaluate_regime_amplitudes(
            runner,
            regime_rows,
            prune_on_max_failure=prune_on_max_failure,
            # A diameter audit validates the entire zero-filled repetition,
            # including the maximum-to-lower-amplitude implication.
            audit_max_failure=audit_max_failure or audit_diameter_failure,
            expected_max_failure=diameter_screen_failed and audit_diameter_failure,
        )
        stats["diameter_failure_pruned"] = 0
    rows: list[dict[str, Any]] = []
    for regime in regime_rows:
        spike_times, n_spikes = results[float(regime["amp_uA"])]
        n_index = regime.get("n_index_spikes", np.nan)
        if str(basis["axon_id"]) == str(regime["index_axon_id"]):
            # For CV0-expanded amplitudes, the actual index response is not
            # known until this population simulation is run.  Persist it on
            # the index axon's own rows; feature extraction propagates these
            # trial-specific values into the integrated population table.
            n_index = int(n_spikes)
        rows.append({
            "detection_protocol_version": regime["detection_protocol_version"],
            "index_axon_id": regime["index_axon_id"], "index_cutline_id": regime["index_cutline_id"],
            "index_diameter_um": regime["index_diameter_um"], "target_axon_id": basis["axon_id"],
            "target_cutline_id": basis["cutline_id"], "target_diameter_um": basis["diameter_um"],
            "target_n_valid_nodes": basis["n_valid_nodes"], "target_readout_node_number": basis["readout_node_number"],
            "file": npz_path.name, "cv": cv, "freq": freq, "rep_index": rep_index,
            "rep_label": regime["rep_label"], "n_input_pulses": int(regime["n_input_pulses"]),
            "thresh_type": regime["thresh_type"], "amp_uA": float(regime["amp_uA"]),
            "n_index_spikes": None if pd.isna(n_index) else int(n_index), "n_spikes": int(n_spikes),
            "spike_times_ms": json.dumps(np.asarray(spike_times, dtype=float).tolist()),
        })
    return rows, stats


def _population_partial_path(partial_dir: Path, mode: str, job: dict[str, Any]) -> Path:
    prefix = "population_responses" if mode == "true" else "population_cv0_based_responses"
    return partial_dir / f"{prefix}_{job['target_axon_id']}_cv{float(job['cv']):.1f}_f{int(float(job['freq']))}.csv"


def _atomic_json(payload: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def _diameter_screen_failures(path: Path) -> dict[int, bool]:
    """Return whether the adjacent larger diameter failed at its largest amplitude."""
    table = pd.read_csv(path, usecols=["rep_index", "amp_uA", "n_spikes"])
    failures: dict[int, bool] = {}
    for rep_index, rows in table.groupby("rep_index", sort=False):
        max_amp = float(rows["amp_uA"].max())
        at_max = rows[np.isclose(rows["amp_uA"].astype(float), max_amp)]
        failures[int(rep_index)] = not bool((at_max["n_spikes"].astype(int) > 0).any())
    return failures


def _run_population_recruitment_condition(
    *, root: Path, mode: str, job: dict[str, Any], n_jobs: int,
    partial_dir: Path, overwrite: bool, prune_on_max_failure: bool,
    max_pruning_audit_rate: float, diameter_screen_path: Path | None,
    diameter_pruning_audit_rate: float,
    regimes_override: pd.DataFrame | None = None,
) -> Path:
    """Run one target/condition with exact caching and audited monotonic pruning."""
    output = _population_partial_path(partial_dir, mode, job)
    if output.is_file() and not overwrite:
        return output
    regimes = (regimes_override if regimes_override is not None else
               _regimes_for_population_condition(root, mode, float(job["cv"]), float(job["freq"])))
    protocols = regimes["detection_protocol_version"].dropna().astype(str).unique().tolist()
    if len(protocols) != 1:
        raise ValueError(f"Expected one detection protocol for {job}, found {protocols}")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in regimes.to_dict(orient="records"):
        groups[int(record["rep_index"])].append(record)

    screen_failures: dict[int, bool] = {}
    if diameter_screen_path is not None:
        diameter_screen_path = diameter_screen_path.resolve()
        if not diameter_screen_path.is_file():
            raise FileNotFoundError(f"Diameter-screen partial is missing: {diameter_screen_path}")
        screen_failures = _diameter_screen_failures(diameter_screen_path)
        missing_reps = sorted(set(groups) - set(screen_failures))
        if missing_reps:
            raise RuntimeError(f"Diameter-screen partial lacks repetition {missing_reps[0]}")

    target_npz = spacetime_path(
        root / "data" / "space-time-V",
        str(job["target_cutline_id"]),
        float(job["target_diameter_um"]),
        float(job["cv"]),
        float(job["freq"]),
        axon_id=str(job["target_axon_id"]),
    )
    repetition_groups = _equivalent_regime_repetition_groups(target_npz, groups, screen_failures)

    work = []
    for members in repetition_groups:
        rep_index = int(members[0][0])
        regime_rows = groups[rep_index]
        identities = [
            (mode, job["target_axon_id"], job["cv"], job["freq"], member_index)
            for member_index, _ in members
        ]
        diameter_failed = screen_failures.get(rep_index, False)
        work.append((
            members,
            delayed(_population_response_rows_for_rep)(
                root, job, regime_rows,
                prune_on_max_failure=prune_on_max_failure,
                audit_max_failure=any(
                    _stable_audit_sample(max_pruning_audit_rate, "maximum", *identity)
                    for identity in identities
                ),
                diameter_screen_failed=diameter_failed,
                audit_diameter_failure=diameter_failed and any(
                    _stable_audit_sample(diameter_pruning_audit_rate, "diameter", *identity)
                    for identity in identities
                ),
            ),
        ))
    results = Parallel(n_jobs=max(1, n_jobs), prefer="processes")([task for _, task in work])
    rows: list[dict[str, Any]] = []
    stats: dict[str, int] = defaultdict(int)
    for (members, _), (representative_rows, rep_stats) in zip(work, results):
        rows.extend(_replicate_rows_for_repetitions(representative_rows, members))
        for key, value in rep_stats.items():
            stats[key] += int(value)
        stats["logical_repetitions"] += len(members)
        stats["simulated_repetitions"] += 1
        stats["repetition_runs_saved"] += len(members) - 1
        stats["logical_regime_rows"] += len(representative_rows) * len(members)
    rows.sort(key=lambda row: (int(row["rep_index"]), THRESH_TYPES.index(str(row["thresh_type"]))))
    _atomic_dataframe_csv(pd.DataFrame(rows, columns=POPULATION_RESPONSE_COLUMNS), output)
    _atomic_json({
        "detection_protocol_version": protocols[0],
        "mode": mode,
        "target_axon_id": job["target_axon_id"],
        "cv": float(job["cv"]),
        "freq": float(job["freq"]),
        "diameter_screen_path": None if diameter_screen_path is None else str(diameter_screen_path),
        "prune_on_max_failure": bool(prune_on_max_failure),
        "max_pruning_audit_rate": float(max_pruning_audit_rate),
        "diameter_pruning_audit_rate": float(diameter_pruning_audit_rate),
        **dict(stats),
    }, output.with_suffix(".audit.json"))
    return output


def run_population_recruitment_job(
    *, root: Path = PROJECT_ROOT, mode: str, job_index: int, n_jobs: int = 1,
    partial_dir: Path | None = None, overwrite: bool = False,
    population_manifest: Path | None = None, prune_on_max_failure: bool = True,
    max_pruning_audit_rate: float = DEFAULT_MAX_PRUNING_AUDIT_RATE,
) -> Path:
    """Run one target-axon/CV/frequency recruitment task for a SLURM array."""
    root = root.resolve()
    jobs = population_recruitment_jobs(root, population_manifest)
    if not 0 <= job_index < len(jobs):
        raise IndexError(f"job_index must be in [0, {len(jobs) - 1}]")
    job = jobs[job_index]
    if mode not in {"true", "cv0"}:
        raise ValueError("mode must be 'true' or 'cv0'")
    partial_dir = (partial_dir or _detection_partial_dir(
        root, "population-responses" if mode == "true" else "population-cv0-responses",
        protocol_version=CV0_PROTOCOL_VERSION if mode == "cv0" else DETECTION_PROTOCOL_VERSION,
    )).resolve()
    return _run_population_recruitment_condition(
        root=root, mode=mode, job=job, n_jobs=n_jobs, partial_dir=partial_dir,
        overwrite=overwrite, prune_on_max_failure=prune_on_max_failure,
        max_pruning_audit_rate=max_pruning_audit_rate,
        diameter_screen_path=None,
        diameter_pruning_audit_rate=DEFAULT_DIAMETER_PRUNING_AUDIT_RATE,
    )


def population_trajectory_condition_jobs(
    root: Path = PROJECT_ROOT, population_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    """Return one array task per cutline/phase/condition, containing all diameters."""
    root = root.resolve()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for basis_path in _population_basis_paths(root, population_manifest):
        basis = load_population_basis(basis_path)
        key = (str(basis["cutline_id"]), str(basis["phase_id"]))
        grouped[key].append({
            "target_axon_id": basis["axon_id"], "target_cutline_id": basis["cutline_id"],
            "target_diameter_um": basis["diameter_um"], "target_phase_id": basis["phase_id"],
            "basis_path": str(basis_path),
        })
    jobs: list[dict[str, Any]] = []
    for cutline_id, phase_id in sorted(grouped):
        targets = sorted(grouped[(cutline_id, phase_id)], key=lambda row: float(row["target_diameter_um"]), reverse=True)
        for cv, freq in index_regime_conditions():
            jobs.append({"cutline_id": cutline_id, "phase_id": phase_id, "cv": cv, "freq": freq, "targets": targets})
    return jobs


def run_population_target_condition(
    *, root: Path = PROJECT_ROOT, mode: str, target_axon_id: str, cv: float,
    freq: float, n_jobs: int = 1, partial_dir: Path | None = None,
    overwrite: bool = False, prune_on_max_failure: bool = True,
    max_pruning_audit_rate: float = DEFAULT_MAX_PRUNING_AUDIT_RATE,
    diameter_screen_path: Path | None = None,
    diameter_pruning_audit_rate: float = DEFAULT_DIAMETER_PRUNING_AUDIT_RATE,
) -> Path:
    """Run a named target condition; used by isolated per-diameter subprocesses."""
    root = root.resolve()
    basis_path = _basis_path_for_axon(root / "data" / "space-time-V", target_axon_id)
    basis = load_population_basis(basis_path)
    job = {
        "target_axon_id": basis["axon_id"], "target_cutline_id": basis["cutline_id"],
        "target_diameter_um": basis["diameter_um"], "basis_path": str(basis_path),
        "cv": float(cv), "freq": float(freq),
    }
    partial_dir = (partial_dir or _detection_partial_dir(
        root, "population-responses" if mode == "true" else "population-cv0-responses",
        protocol_version=CV0_PROTOCOL_VERSION if mode == "cv0" else DETECTION_PROTOCOL_VERSION,
    )).resolve()
    return _run_population_recruitment_condition(
        root=root, mode=mode, job=job, n_jobs=n_jobs, partial_dir=partial_dir,
        overwrite=overwrite, prune_on_max_failure=prune_on_max_failure,
        max_pruning_audit_rate=max_pruning_audit_rate,
        diameter_screen_path=diameter_screen_path,
        diameter_pruning_audit_rate=diameter_pruning_audit_rate,
    )


def run_population_trajectory_condition_job(
    *, root: Path = PROJECT_ROOT, mode: str, trajectory_job_index: int,
    n_jobs: int = 1, partial_dir: Path | None = None, overwrite: bool = False,
    population_manifest: Path | None = None, prune_on_max_failure: bool = True,
    max_pruning_audit_rate: float = DEFAULT_MAX_PRUNING_AUDIT_RATE,
    prune_by_diameter: bool = True,
    diameter_pruning_audit_rate: float = DEFAULT_DIAMETER_PRUNING_AUDIT_RATE,
) -> list[Path]:
    """Run a trajectory largest-to-smallest, isolating each MRG geometry process."""
    root = root.resolve()
    jobs = population_trajectory_condition_jobs(root, population_manifest)
    if not 0 <= trajectory_job_index < len(jobs):
        raise IndexError(f"trajectory_job_index must be in [0, {len(jobs) - 1}]")
    job = jobs[trajectory_job_index]
    partial_dir = (partial_dir or _detection_partial_dir(
        root, "population-responses" if mode == "true" else "population-cv0-responses",
        protocol_version=CV0_PROTOCOL_VERSION if mode == "cv0" else DETECTION_PROTOCOL_VERSION,
    )).resolve()
    outputs: list[Path] = []
    previous: Path | None = None
    for target in job["targets"]:
        command = [
            sys.executable, "-m", "detection.detection_CLI",
            "run-population-target-condition", "--root", str(root), "--mode", mode,
            "--target-axon-id", str(target["target_axon_id"]), "--cv", str(job["cv"]),
            "--freq", str(job["freq"]), "--n-jobs", str(max(1, n_jobs)),
            "--partial-dir", str(partial_dir),
            "--max-pruning-audit-rate", str(max_pruning_audit_rate),
            "--diameter-pruning-audit-rate", str(diameter_pruning_audit_rate),
        ]
        if not prune_on_max_failure:
            command.append("--no-max-pruning")
        if overwrite:
            command.append("--overwrite")
        if prune_by_diameter and previous is not None:
            command.extend(("--diameter-screen", str(previous)))
        subprocess.run(command, cwd=root, check=True)
        child_job = {**target, "cv": job["cv"], "freq": job["freq"]}
        previous = _population_partial_path(partial_dir, mode, child_job)
        outputs.append(previous)
    return outputs


def merge_population_recruitment_partials(
    *, root: Path = PROJECT_ROOT, mode: str, partial_dir: Path | None = None,
    output_csv: Path | None = None, population_manifest: Path | None = None,
) -> Path:
    """Validate and stream-merge all target-condition recruitment partials."""
    root = root.resolve()
    if mode not in {"true", "cv0"}:
        raise ValueError("mode must be 'true' or 'cv0'")
    partial_dir = (partial_dir or _detection_partial_dir(
        root, "population-responses" if mode == "true" else "population-cv0-responses",
        protocol_version=CV0_PROTOCOL_VERSION if mode == "cv0" else DETECTION_PROTOCOL_VERSION,
    )).resolve()
    output_csv = output_csv or root / "data" / "detection" / (POPULATION_RESPONSES_CSV.name if mode == "true" else POPULATION_CV0_RESPONSES_CSV.name)
    jobs = population_recruitment_jobs(root, population_manifest)
    expected = [_population_partial_path(partial_dir, mode, job) for job in jobs]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot merge {mode} recruitment; {len(missing)} task files are missing (first: {missing[0]})")

    # The accepted v1 CV0 table stores the realized index count on the index
    # axon's rows only. Later CV0 protocols propagate that count to all targets.
    index_count_by_condition: dict[tuple[str, str, str, str], str] | None = None
    cv0_protocol = None
    if mode == "cv0":
        protocols = compute_index_cv0_threshold_max(root=root, force=False)[
            "detection_protocol_version"
        ].dropna().astype(str).unique().tolist()
        if len(protocols) != 1:
            raise ValueError(f"Expected one CV0 protocol, found {protocols}")
        cv0_protocol = protocols[0]
    if mode == "cv0" and cv0_protocol != CV0_PROTOCOL_VERSION:
        index = identify_index_axon(root=root, n_jobs=1, force=False)
        index_id = str(index["axon_id"])
        index_parts = [
            part for job, part in zip(jobs, expected)
            if str(job["target_axon_id"]) == index_id
        ]
        if not index_parts:
            raise RuntimeError(f"No CV0 response partials were found for index axon {index_id!r}")
        index_count_by_condition = {}
        key_columns = ("cv", "freq", "rep_index", "thresh_type")
        for part in index_parts:
            with part.open("r", newline="", encoding="utf-8") as source:
                reader = csv.reader(source)
                header = next(reader, None)
                if header != list(POPULATION_RESPONSE_COLUMNS):
                    raise RuntimeError(f"Unexpected header in index recruitment partial: {part}")
                positions = {name: header.index(name) for name in (*key_columns, "target_axon_id", "n_spikes")}
                for row in reader:
                    if str(row[positions["target_axon_id"]]) != index_id:
                        raise RuntimeError(f"Non-index response found in index partial: {part}")
                    key = tuple(row[positions[name]] for name in key_columns)
                    count = row[positions["n_spikes"]]
                    previous = index_count_by_condition.setdefault(key, count)
                    if previous != count:
                        raise RuntimeError(f"Index response count disagrees for condition {key}")
        if not index_count_by_condition:
            raise RuntimeError("Index-response count lookup is empty")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        for index, part in enumerate(expected):
            with part.open("r", newline="", encoding="utf-8") as source:
                reader = csv.reader(source)
                header = next(reader, None)
                if header != list(POPULATION_RESPONSE_COLUMNS):
                    raise RuntimeError(f"Unexpected header in recruitment partial: {part}")
                if index == 0:
                    writer.writerow(header)
                if index_count_by_condition is None:
                    writer.writerows(reader)
                    continue
                positions = {name: header.index(name) for name in (*key_columns, "n_index_spikes")}
                for row in reader:
                    key = tuple(row[positions[name]] for name in key_columns)
                    try:
                        count = index_count_by_condition[key]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"No realized index response is available for condition {key}"
                        ) from exc
                    existing = row[positions["n_index_spikes"]]
                    if existing not in {"", count}:
                        raise RuntimeError(
                            f"Stored index count {existing!r} disagrees with realized count "
                            f"{count!r} for condition {key}"
                        )
                    row[positions["n_index_spikes"]] = count
                    writer.writerow(row)
    temporary.replace(output_csv)
    return output_csv

def _population_response_group(root: Path, worker, regimes: pd.DataFrame, overwrite: bool, output_stem: str) -> Path:
    """Simulate all axons of one diameter in a single fixed-geometry process."""
    partial_dir = root / "data" / "detection" / "partials"; partial_dir.mkdir(parents=True, exist_ok=True)
    out_path = partial_dir / f"{output_stem}_{diameter_token(worker.diameter_um)}um.csv"
    if out_path.is_file() and not overwrite:
        return out_path
    if out_path.is_file():
        out_path.unlink()
    config = _config_path(root, worker.diameter_um, worker.axonnodes)
    header_written = False
    for basis_path_str in worker.basis_paths:
        basis = load_population_basis(Path(basis_path_str))
        for (cv, freq), condition_rows in regimes.groupby(["cv", "freq"], sort=True):
            npz_path = spacetime_path(
                root / "data" / "space-time-V", basis["cutline_id"],
                basis["diameter_um"], float(cv), float(freq), axon_id=str(basis["axon_id"]),
            )
            for regime in condition_rows.itertuples(index=False):
                rep = load_population_rep(npz_path, int(regime.rep_index))
                time_trim, V_trim = trim_tail_window_ms(rep["time_ms"], rep["V_base"], rep["current"])
                time_trim, V_trim = prepend_zeros_ms(time_trim, V_trim, prepad_ms=PREPAD_MS)
                runner = PopulationMRGRunner(time_trim, V_trim, config_path=config, readout_node_number=rep["readout_node_number"])
                spike_times, n_spikes = runner.run_mrg_for_amp(float(regime.amp_uA))
                n_index = regime.n_index_spikes
                if str(basis["axon_id"]) == str(regime.index_axon_id):
                    n_index = int(n_spikes)
                row = {
                    "detection_protocol_version": DETECTION_PROTOCOL_VERSION,
                    "index_axon_id": regime.index_axon_id, "index_cutline_id": regime.index_cutline_id,
                    "index_diameter_um": regime.index_diameter_um, "target_axon_id": basis["axon_id"],
                    "target_cutline_id": basis["cutline_id"], "target_diameter_um": basis["diameter_um"],
                    "target_n_valid_nodes": basis["n_valid_nodes"], "target_readout_node_number": basis["readout_node_number"],
                    "file": npz_path.name, "cv": cv, "freq": freq, "rep_index": int(regime.rep_index),
                    "rep_label": regime.rep_label, "n_input_pulses": int(regime.n_input_pulses),
                    "thresh_type": regime.thresh_type, "amp_uA": float(regime.amp_uA),
                    "n_index_spikes": None if pd.isna(n_index) else int(n_index),
                    "n_spikes": int(n_spikes),
                    "spike_times_ms": json.dumps(np.asarray(spike_times, dtype=float).tolist()),
                }
                pd.DataFrame([row], columns=POPULATION_RESPONSE_COLUMNS).to_csv(
                    out_path, mode="a", index=False, header=not header_written
                )
                header_written = True
                del rep, V_trim, runner
        gc.collect()
    return out_path


def _simulate_population_from_regimes(root: Path, regimes: pd.DataFrame, *, n_jobs: int, overwrite: bool, output_csv: Path, output_stem: str) -> Path:
    """Run diameter-partitioned population responses for one long-form regime table."""
    results_dir = output_csv.parent; results_dir.mkdir(parents=True, exist_ok=True)
    if regimes.empty:
        raise RuntimeError("No index-regime amplitudes are available for population simulation")
    workers = discover_mrg_workers(root / "data" / "space-time-V")
    prepare_mrg_workers(root=root)
    parts = Parallel(n_jobs=min(max(1, n_jobs), len(workers)), prefer="processes")(
        delayed(_population_response_group)(root, worker, regimes, overwrite, output_stem) for worker in workers
    )
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        wrote_header = False
        for part in parts:
            with part.open("r", newline="", encoding="utf-8") as source:
                reader = csv.reader(source)
                header = next(reader, None)
                if header and not wrote_header:
                    writer.writerow(header); wrote_header = True
                writer.writerows(reader)
    return output_csv


def simulate_population_at_index_regimes(*, root: Path = PROJECT_ROOT, n_jobs: int = 4, overwrite: bool = False) -> Path:
    """Run every axon at each repetition's own index-regime amplitude locally.

    The Longleaf workflow calls :func:`run_population_recruitment_job` through
    an array; this convenience function uses those same restartable files.
    """
    root = root.resolve()
    partial_dir = _detection_partial_dir(root, "population-responses")
    for job_index in tqdm(range(len(population_recruitment_jobs(root))), desc="Population recruitment conditions"):
        run_population_recruitment_job(
            root=root, mode="true", job_index=job_index, n_jobs=n_jobs,
            partial_dir=partial_dir, overwrite=overwrite,
        )
    return merge_population_recruitment_partials(root=root, mode="true", partial_dir=partial_dir)


def simulate_population_at_index_cv0_regimes(*, root: Path = PROJECT_ROOT, n_jobs: int = 4, overwrite: bool = False) -> Path:
    """Run every axon/CV/repetition at index CV0-derived amplitudes locally."""
    root = root.resolve()
    partial_dir = _detection_partial_dir(
        root, "population-cv0-responses", protocol_version=CV0_PROTOCOL_VERSION,
    )
    for job_index in tqdm(range(len(population_recruitment_jobs(root))), desc="Population CV0 recruitment conditions"):
        run_population_recruitment_job(
            root=root, mode="cv0", job_index=job_index, n_jobs=n_jobs,
            partial_dir=partial_dir, overwrite=overwrite,
        )
    return merge_population_recruitment_partials(root=root, mode="cv0", partial_dir=partial_dir)


def verify_index_axon(*, root: Path = PROJECT_ROOT, expected_axon_id: str) -> Path:
    """Fail before reusing cached regimes if the recomputed index axon changed."""
    path = root.resolve() / "data" / "detection" / INDEX_AXON_CSV.name
    if not path.is_file():
        raise FileNotFoundError(f"Index-axon result is missing: {path}")
    table = pd.read_csv(path)
    if len(table) != 1 or "axon_id" not in table:
        raise RuntimeError(f"Expected exactly one index-axon row in {path}")
    observed = str(table.iloc[0]["axon_id"])
    if observed != expected_axon_id:
        raise RuntimeError(
            f"Index axon changed from expected {expected_axon_id!r} to {observed!r}. "
            "Cached index regimes and response partials must not be reused."
        )
    return path


# --------------- Population detection behavioral matching --------------- #

def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Read XLSX shared strings without adding an openpyxl dependency."""
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(archive.read(path))
    return ["".join(node.text or "" for node in item.findall(".//a:t", namespace))
            for item in root.findall("a:si", namespace)]


def _xlsx_sheet_rows(path: Path) -> dict[int, dict[str, str]]:
    """Return the first worksheet as a sparse row/column mapping."""
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared = _xlsx_shared_strings(archive)
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: dict[int, dict[str, str]] = {}
    for row in root.findall(".//a:row", namespace):
        row_number = int(row.attrib.get("r", 0))
        values: dict[str, str] = {}
        for cell in row.findall("a:c", namespace):
            match = re.match(r"([A-Z]+)", cell.attrib.get("r", ""))
            if not match:
                continue
            column = match.group(1)
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", namespace)
            if cell_type == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(".//a:t", namespace))
            elif value_node is None:
                value = ""
            else:
                raw = value_node.text or ""
                if cell_type == "s":
                    try:
                        value = shared[int(raw)]
                    except (IndexError, ValueError):
                        value = raw
                else:
                    value = raw
            values[column] = value
        rows[row_number] = values
    return rows


def load_detection_behavior(path: Path = DETECTION_BEHAVIOR_XLSX) -> pd.DataFrame:
    """Load the Slack et al. detection-threshold workbook in long form."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Detection behavioral workbook not found: {path}")

    rows = _xlsx_sheet_rows(path)
    output: list[dict[str, Any]] = []
    row_numbers = sorted(rows)
    for row_number in row_numbers:
        frequency_match = re.match(r"^(\d+)Hz$", str(rows[row_number].get("C", "")).strip())
        if not frequency_match:
            continue
        frequency_hz = float(frequency_match.group(1))
        cv_row = rows.get(row_number + 1, {})
        cvs: list[float] = []
        for column in ("C", "D", "E", "F", "G", "H"):
            raw = str(cv_row.get(column, "")).strip()
            if raw.lower().startswith("cv"):
                cvs.append(0.0)
            else:
                try:
                    cvs.append(float(raw))
                except ValueError:
                    cvs.append(np.nan)

        current = row_number + 3
        while current in rows:
            values = rows[current]
            if re.match(r"^(\d+)Hz$", str(values.get("C", "")).strip()):
                break
            rat_raw = str(values.get("B", "")).strip()
            if not rat_raw and not str(values.get("C", "")).strip():
                current += 1
                continue
            try:
                rat = int(float(rat_raw))
            except ValueError:
                current += 1
                continue

            for absolute_column, normalized_column, cv in zip(
                ("C", "D", "E", "F", "G", "H"),
                ("I", "J", "K", "L", "M", "N"),
                cvs,
            ):
                try:
                    threshold_uA = float(values.get(absolute_column, ""))
                except ValueError:
                    threshold_uA = np.nan
                try:
                    normalized_threshold = float(values.get(normalized_column, ""))
                except ValueError:
                    normalized_threshold = np.nan
                if np.isfinite(cv) and (np.isfinite(threshold_uA) or np.isfinite(normalized_threshold)):
                    output.append({
                        "freq": frequency_hz,
                        "cv": float(cv),
                        "rat": rat,
                        "threshold_uA": threshold_uA,
                        "norm_threshold": normalized_threshold,
                    })
            current += 1

    table = pd.DataFrame(output)
    if table.empty:
        raise RuntimeError(f"No behavioral detection rows were parsed from {path}")
    for column in ("freq", "cv", "rat", "threshold_uA", "norm_threshold"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table[np.isfinite(table["freq"]) & np.isfinite(table["cv"])].copy()
    table.sort_values(["freq", "cv", "rat"], inplace=True, ignore_index=True)
    return table


def _sem(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return 0.0 if array.size <= 1 else float(np.std(array, ddof=1) / np.sqrt(array.size))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    shifted = values - np.max(values, axis=1, keepdims=True)
    weights = np.exp(np.clip(shifted, -700.0, 700.0))
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights[0] if one_dimensional else weights


def _interpolate_weight_rows(anchor_cvs: np.ndarray, anchor_weights: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    interpolated = np.column_stack([
        np.interp(cvs, anchor_cvs, anchor_weights[:, regime_index])
        for regime_index in range(anchor_weights.shape[1])
    ])
    interpolated = np.maximum(interpolated, 0.0)
    denominator = interpolated.sum(axis=1, keepdims=True)
    return np.divide(interpolated, denominator, out=np.full_like(interpolated, 1.0 / interpolated.shape[1]), where=denominator > 0)


def _normalized_detection_prediction(
    amplitudes: np.ndarray,
    weights: np.ndarray,
    *,
    mode: str,
    sim_cvs: np.ndarray,
    anchor_cvs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute curves, normalized curves, and weights at every CV."""
    n_frequencies, n_cvs, n_regimes = amplitudes.shape
    if mode == "global":
        weights_all = np.broadcast_to(np.asarray(weights, dtype=float), (n_frequencies, n_cvs, n_regimes)).copy()
    elif mode == "per_frequency":
        weights_all = np.broadcast_to(np.asarray(weights, dtype=float)[:, None, :], (n_frequencies, n_cvs, n_regimes)).copy()
    elif mode == "per_point":
        anchor_weights = np.asarray(weights, dtype=float)
        weights_all = np.stack([
            _interpolate_weight_rows(anchor_cvs, anchor_weights[frequency_index], sim_cvs)
            for frequency_index in range(n_frequencies)
        ])
    else:
        raise ValueError(f"Unknown detection mixture mode: {mode}")

    absolute = np.sum(amplitudes * weights_all, axis=2)
    maxima = np.max(absolute, axis=1, keepdims=True)
    normalized = np.divide(absolute, maxima, out=np.full_like(absolute, np.nan), where=np.abs(maxima) > 1e-12)
    return absolute, normalized, weights_all


def _fit_logits(
    objective,
    size: int,
    *,
    seed: int,
    n_starts: int,
    maxiter: int,
) -> tuple[np.ndarray, Any]:
    rng = np.random.default_rng(seed)
    starts = [np.zeros(size, dtype=float)]
    starts.extend(rng.normal(0.0, 0.25, size=size) for _ in range(max(0, n_starts - 1)))
    best_result = None
    for initial in starts:
        result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": int(maxiter), "ftol": 1e-12})
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result
    if best_result is None:
        raise RuntimeError("Detection mixture optimization did not return a result")
    return np.asarray(best_result.x, dtype=float), best_result


def _fit_detection_mixture_mode(
    amplitudes: np.ndarray,
    targets: np.ndarray,
    *,
    sim_cvs: np.ndarray,
    target_cvs: np.ndarray,
    mode: str,
    allow_panel_bias: bool,
    per_point_smooth_lambda: float,
    seed: int,
    n_starts: int,
) -> dict[str, Any]:
    """Fit one convex-mixture flexibility mode to normalized behavior."""
    n_frequencies, _, n_regimes = amplitudes.shape
    target_indices = np.asarray([int(np.flatnonzero(np.isclose(sim_cvs, cv))[0]) for cv in target_cvs], dtype=int)

    def evaluate(weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        absolute, normalized, weights_all = _normalized_detection_prediction(
            amplitudes, weights, mode=mode, sim_cvs=sim_cvs, anchor_cvs=target_cvs,
        )
        sampled = normalized[:, target_indices]
        biases = np.mean(targets - sampled, axis=1) if allow_panel_bias else np.zeros(n_frequencies, dtype=float)
        prediction = normalized + biases[:, None]
        residual = prediction[:, target_indices] - targets
        loss = float(np.mean(np.mean(residual * residual, axis=1)))
        if mode == "per_point" and per_point_smooth_lambda > 0.0:
            anchor_weights = np.asarray(weights, dtype=float)
            loss += float(per_point_smooth_lambda) * float(np.mean(np.diff(anchor_weights, axis=1) ** 2))
        return loss, absolute, prediction, biases, weights_all

    if mode == "global":
        def objective(theta: np.ndarray) -> float:
            return evaluate(_softmax_rows(theta))[0]
        theta, result = _fit_logits(objective, n_regimes, seed=seed, n_starts=n_starts, maxiter=1500)
        fitted_weights = _softmax_rows(theta)
        optimization_results = [result]
    elif mode == "per_frequency":
        rows = []
        optimization_results = []
        for frequency_index in range(n_frequencies):
            def objective(theta: np.ndarray, frequency_index: int = frequency_index) -> float:
                row = _softmax_rows(theta)
                absolute = np.sum(amplitudes[frequency_index] * row[None, :], axis=1)
                normalized = absolute / max(float(np.max(absolute)), 1e-12)
                sampled = normalized[target_indices]
                bias = float(np.mean(targets[frequency_index] - sampled)) if allow_panel_bias else 0.0
                return float(np.mean((sampled + bias - targets[frequency_index]) ** 2))
            theta, result = _fit_logits(
                objective, n_regimes, seed=seed + 101 * (frequency_index + 1),
                n_starts=n_starts, maxiter=1500,
            )
            rows.append(_softmax_rows(theta))
            optimization_results.append(result)
        fitted_weights = np.stack(rows)
    elif mode == "per_point":
        rows = []
        optimization_results = []
        n_anchor = len(target_cvs)
        for frequency_index in range(n_frequencies):
            def objective(theta: np.ndarray, frequency_index: int = frequency_index) -> float:
                anchor_weights = _softmax_rows(theta.reshape(n_anchor, n_regimes))
                weights_all = _interpolate_weight_rows(target_cvs, anchor_weights, sim_cvs)
                absolute = np.sum(amplitudes[frequency_index] * weights_all, axis=1)
                normalized = absolute / max(float(np.max(absolute)), 1e-12)
                sampled = normalized[target_indices]
                bias = float(np.mean(targets[frequency_index] - sampled)) if allow_panel_bias else 0.0
                residual_loss = float(np.mean((sampled + bias - targets[frequency_index]) ** 2))
                smooth = float(np.mean(np.diff(anchor_weights, axis=0) ** 2)) if n_anchor > 1 else 0.0
                return residual_loss + float(per_point_smooth_lambda) * smooth
            theta, result = _fit_logits(
                objective, n_anchor * n_regimes, seed=seed + 1009 * (frequency_index + 1),
                n_starts=max(1, min(2, n_starts)), maxiter=2500,
            )
            rows.append(_softmax_rows(theta.reshape(n_anchor, n_regimes)))
            optimization_results.append(result)
        fitted_weights = np.stack(rows)
    else:
        raise ValueError(f"Unknown detection mixture mode: {mode}")

    loss, absolute, prediction, biases, weights_all = evaluate(fitted_weights)
    return {
        "mode": mode,
        "weights": fitted_weights,
        "weights_all": weights_all,
        "absolute": absolute,
        "prediction": prediction,
        "biases": biases,
        "loss": loss,
        "optimization_success": bool(all(bool(result.success) for result in optimization_results)),
        "optimization_messages": " | ".join(str(result.message) for result in optimization_results),
    }


def _evaluate_best_single_regime(
    amplitudes: np.ndarray,
    targets: np.ndarray,
    *,
    target_indices: np.ndarray,
    allow_panel_bias: bool,
) -> dict[str, Any]:
    best = None
    for regime_index in range(amplitudes.shape[2]):
        absolute = amplitudes[:, :, regime_index]
        maxima = np.max(absolute, axis=1, keepdims=True)
        normalized = np.divide(absolute, maxima, out=np.full_like(absolute, np.nan), where=maxima > 1e-12)
        sampled = normalized[:, target_indices]
        biases = np.mean(targets - sampled, axis=1) if allow_panel_bias else np.zeros(amplitudes.shape[0])
        prediction = normalized + biases[:, None]
        loss = float(np.mean((prediction[:, target_indices] - targets) ** 2))
        candidate = {
            "mode": "best_single", "regime_index": regime_index, "absolute": absolute,
            "prediction": prediction, "biases": biases, "loss": loss,
            "optimization_success": True, "optimization_messages": "enumerated",
        }
        if best is None or candidate["loss"] < best["loss"]:
            best = candidate
    if best is None:
        raise RuntimeError("No detection regimes were available for the single-regime comparison")
    return best


def _evaluate_best_single_regime_per_frequency(
    amplitudes: np.ndarray,
    targets: np.ndarray,
    *,
    target_indices: np.ndarray,
    allow_panel_bias: bool,
) -> dict[str, Any]:
    """Enumerate one fixed, unmixed regime independently per frequency."""
    n_frequencies, n_cvs, n_regimes = amplitudes.shape
    absolute = np.empty((n_frequencies, n_cvs), dtype=float)
    prediction = np.empty_like(absolute)
    biases = np.zeros(n_frequencies, dtype=float)
    regime_indices = np.empty(n_frequencies, dtype=int)
    weights_all = np.zeros_like(amplitudes, dtype=float)
    frequency_losses = np.empty(n_frequencies, dtype=float)

    for frequency_index in range(n_frequencies):
        best = None
        for regime_index in range(n_regimes):
            candidate_absolute = amplitudes[frequency_index, :, regime_index]
            maximum = float(np.max(candidate_absolute))
            candidate_normalized = np.divide(
                candidate_absolute,
                maximum,
                out=np.full_like(candidate_absolute, np.nan),
                where=maximum > 1e-12,
            )
            sampled = candidate_normalized[target_indices]
            bias = (
                float(np.mean(targets[frequency_index] - sampled))
                if allow_panel_bias else 0.0
            )
            candidate_prediction = candidate_normalized + bias
            loss = float(np.mean(
                (candidate_prediction[target_indices] - targets[frequency_index]) ** 2
            ))
            if best is None or loss < best["loss"]:
                best = {
                    "loss": loss,
                    "regime_index": regime_index,
                    "absolute": candidate_absolute,
                    "prediction": candidate_prediction,
                    "bias": bias,
                }
        if best is None:
            raise RuntimeError(
                f"No detection regimes were available for frequency index {frequency_index}"
            )
        regime_index = int(best["regime_index"])
        regime_indices[frequency_index] = regime_index
        absolute[frequency_index] = best["absolute"]
        prediction[frequency_index] = best["prediction"]
        biases[frequency_index] = float(best["bias"])
        frequency_losses[frequency_index] = float(best["loss"])
        weights_all[frequency_index, :, regime_index] = 1.0

    return {
        "mode": "best_single_per_frequency",
        "regime_indices": regime_indices,
        "weights_all": weights_all,
        "absolute": absolute,
        "prediction": prediction,
        "biases": biases,
        "frequency_losses": frequency_losses,
        "loss": float(np.mean(frequency_losses)),
        "optimization_success": True,
        "optimization_messages": "enumerated independently per frequency",
    }


def _detection_fit_metrics(
    fit: dict[str, Any],
    targets: np.ndarray,
    *,
    target_indices: np.ndarray,
    frequencies: np.ndarray,
    n_parameters: int,
    target_kind: str,
) -> pd.DataFrame:
    sampled = np.asarray(fit["prediction"], dtype=float)[:, target_indices]
    rows: list[dict[str, Any]] = []
    for frequency_index, frequency_hz in enumerate(frequencies):
        residual = sampled[frequency_index] - targets[frequency_index]
        rows.append({
            "target_kind": target_kind, "mode": fit["mode"], "freq": frequency_hz,
            "n_observations": len(residual), "nominal_parameters": n_parameters,
            "rmse": float(np.sqrt(np.mean(residual * residual))),
            "mae": float(np.mean(np.abs(residual))),
            "sse": float(np.sum(residual * residual)),
            "optimization_success": fit["optimization_success"],
        })
    residual = sampled - targets
    sse = float(np.sum(residual * residual))
    n_observations = int(residual.size)
    mse = max(sse / max(n_observations, 1), 1e-15)
    rows.append({
        "target_kind": target_kind, "mode": fit["mode"], "freq": "ALL",
        "n_observations": n_observations, "nominal_parameters": n_parameters,
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(np.abs(residual))), "sse": sse,
        "gaussian_aic": float(n_observations * np.log(mse) + 2 * n_parameters),
        "gaussian_bic": float(n_observations * np.log(mse) + np.log(max(n_observations, 1)) * n_parameters),
        "optimization_success": fit["optimization_success"],
    })
    return pd.DataFrame(rows)


def run_population_detection_behavioral_matching(
    *,
    root: Path = PROJECT_ROOT,
    output_dir: Path | None = None,
    target_kinds: Sequence[str] = ("datapoints", "quadratic"),
    allow_panel_bias: bool = True,
    per_point_smooth_lambda: float = 0.001,
    seed: int = 20260828,
    n_starts: int = 4,
    index_regimes_csv: Path | None = None,
    recruitment_outcomes_csv: Path | None = None,
) -> dict[str, Path]:
    """Fit constrained population detection mixtures and save tidy outputs.

    The quadratic global fit is the default frozen observer control. Separate
    target-specific global exports retain the direct-point sensitivity fit.
    Per-frequency fits are intermediate-complexity comparisons, and per-point
    fits are explicitly retained as representational upper bounds.
    """
    root = Path(root).resolve()
    output_dir = Path(output_dir or root / "data" / "detection" / "behavioral_matching").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_kinds = tuple(str(kind).strip().lower() for kind in target_kinds)
    unsupported = sorted(set(target_kinds) - {"datapoints", "quadratic"})
    if unsupported:
        raise ValueError(f"Unsupported detection target kinds: {unsupported}")

    behavior = load_detection_behavior(root / "data" / "detection" / "slack" / "DetectionThresholds.xlsx")
    behavior_summary = (
        behavior.groupby(["freq", "cv"], as_index=False)
        .agg(
            behavior_mean=("norm_threshold", "mean"),
            behavior_sem=("norm_threshold", _sem),
            n_rats=("norm_threshold", "count"),
            threshold_uA_mean=("threshold_uA", "mean"),
            threshold_uA_sem=("threshold_uA", _sem),
        )
        .sort_values(["freq", "cv"], ignore_index=True)
    )

    regimes_path = Path(index_regimes_csv) if index_regimes_csv else root / "data" / "detection" / "index_regime_amplitudes.csv"
    if not regimes_path.is_file():
        raise FileNotFoundError(f"Population index-regime amplitudes not found: {regimes_path}")
    regimes = pd.read_csv(regimes_path)
    required = {"freq", "cv", "rep_index", "thresh_type", "amp_uA", "n_index_spikes"}
    missing = sorted(required - set(regimes.columns))
    if missing:
        raise RuntimeError(f"Population index-regime table is missing columns: {missing}")
    regimes["thresh_type"] = regimes["thresh_type"].astype(str)
    regime_order = [name for name in THRESH_TYPES if name in set(regimes["thresh_type"])]
    if len(regime_order) != len(THRESH_TYPES):
        raise RuntimeError(f"Expected regimes {list(THRESH_TYPES)}, found {regime_order}")
    amplitude_summary = (
        regimes.groupby(["freq", "cv", "thresh_type"], as_index=False)
        .agg(
            amp_uA_mean=("amp_uA", "mean"), amp_uA_sem=("amp_uA", _sem),
            n_repetitions=("rep_index", "nunique"), mean_index_spikes=("n_index_spikes", "mean"),
        )
    )

    frequencies = np.asarray(sorted(set(behavior_summary["freq"]) & set(amplitude_summary["freq"])), dtype=float)
    sim_cvs = np.asarray(sorted(amplitude_summary["cv"].unique()), dtype=float)
    behavior_cvs = np.asarray(sorted(behavior_summary["cv"].unique()), dtype=float)
    if frequencies.size != 5 or sim_cvs.size != 11 or behavior_cvs.size != 6:
        raise RuntimeError(
            f"Unexpected detection grid: {len(frequencies)} frequencies, {len(sim_cvs)} simulation CVs, "
            f"{len(behavior_cvs)} behavioral CVs"
        )
    if any(not np.any(np.isclose(sim_cvs, cv)) for cv in behavior_cvs):
        raise RuntimeError("Behavioral CV grid is not contained in the simulation CV grid")

    amplitudes = np.empty((len(frequencies), len(sim_cvs), len(regime_order)), dtype=float)
    for frequency_index, frequency_hz in enumerate(frequencies):
        for cv_index, cv in enumerate(sim_cvs):
            rows = amplitude_summary[np.isclose(amplitude_summary["freq"], frequency_hz) & np.isclose(amplitude_summary["cv"], cv)]
            values = rows.set_index("thresh_type")["amp_uA_mean"].reindex(regime_order)
            if values.isna().any():
                raise RuntimeError(f"Incomplete regime amplitudes for {frequency_hz:g} Hz, CV={cv:g}")
            amplitudes[frequency_index, cv_index] = values.to_numpy(float)

    observed = np.empty((len(frequencies), len(behavior_cvs)), dtype=float)
    for frequency_index, frequency_hz in enumerate(frequencies):
        rows = behavior_summary[np.isclose(behavior_summary["freq"], frequency_hz)].set_index("cv")
        observed[frequency_index] = rows["behavior_mean"].reindex(behavior_cvs).to_numpy(float)

    curve_tables: list[pd.DataFrame] = []
    metric_tables: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    fit_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    mixture_seed_offsets = {"global": 10_000, "per_frequency": 20_000, "per_point": 30_000}
    for target_kind_index, target_kind in enumerate(target_kinds):
        if target_kind == "datapoints":
            target_cvs = behavior_cvs
            targets = observed
        else:
            target_cvs = sim_cvs
            targets = np.stack([
                np.polyval(np.polyfit(behavior_cvs, observed[frequency_index], 2), sim_cvs)
                for frequency_index in range(len(frequencies))
            ])
        target_indices = np.asarray([int(np.flatnonzero(np.isclose(sim_cvs, cv))[0]) for cv in target_cvs], dtype=int)

        single = _evaluate_best_single_regime(
            amplitudes, targets, target_indices=target_indices, allow_panel_bias=allow_panel_bias,
        )
        fit_lookup[(target_kind, "best_single")] = single
        n_frequency_biases = len(frequencies) if allow_panel_bias else 0
        metric_tables.append(_detection_fit_metrics(
            single, targets, target_indices=target_indices, frequencies=frequencies,
            n_parameters=n_frequency_biases, target_kind=target_kind,
        ))

        single_per_frequency = _evaluate_best_single_regime_per_frequency(
            amplitudes, targets, target_indices=target_indices,
            allow_panel_bias=allow_panel_bias,
        )
        fit_lookup[(target_kind, "best_single_per_frequency")] = single_per_frequency
        metric_tables.append(_detection_fit_metrics(
            single_per_frequency, targets, target_indices=target_indices,
            frequencies=frequencies,
            # Discrete selection does not have a conventional AIC degree of
            # freedom. Count one nominal choice per frequency as a conservative
            # and transparent approximation in addition to panel biases.
            n_parameters=n_frequency_biases + len(frequencies),
            target_kind=target_kind,
        ))
        for frequency_index, frequency_hz in enumerate(frequencies):
            selected_index = int(single_per_frequency["regime_indices"][frequency_index])
            for regime_index, regime in enumerate(regime_order):
                weight_rows.append({
                    "target_kind": target_kind,
                    "mode": "best_single_per_frequency",
                    "weight_group": f"f{frequency_hz:g}",
                    "freq": frequency_hz,
                    "cv": np.nan,
                    "thresh_type": regime,
                    "weight": float(regime_index == selected_index),
                    "effective_regime_count": 1.0,
                    "max_group_weight": 1.0,
                    "role": "parsimonious_comparison",
                })

        fits = [single, single_per_frequency]
        for mode in ("global", "per_frequency", "per_point"):
            fit = _fit_detection_mixture_mode(
                amplitudes, targets, sim_cvs=sim_cvs, target_cvs=target_cvs, mode=mode,
                allow_panel_bias=allow_panel_bias, per_point_smooth_lambda=per_point_smooth_lambda,
                seed=(
                    seed + 40_000 * target_kind_index + mixture_seed_offsets[mode]
                ),
                n_starts=n_starts,
            )
            fit_lookup[(target_kind, mode)] = fit
            fits.append(fit)
            n_groups = 1 if mode == "global" else len(frequencies) if mode == "per_frequency" else len(frequencies) * len(target_cvs)
            n_parameters = n_groups * (len(regime_order) - 1) + n_frequency_biases
            metric_tables.append(_detection_fit_metrics(
                fit, targets, target_indices=target_indices, frequencies=frequencies,
                n_parameters=n_parameters, target_kind=target_kind,
            ))

            if mode == "global":
                groups = [("global", np.nan, np.nan, fit["weights"])]
            elif mode == "per_frequency":
                groups = [(f"f{frequency_hz:g}", frequency_hz, np.nan, fit["weights"][i]) for i, frequency_hz in enumerate(frequencies)]
            else:
                groups = [
                    (f"f{frequency_hz:g}_cv{cv:.1f}", frequency_hz, cv, fit["weights"][frequency_index, cv_index])
                    for frequency_index, frequency_hz in enumerate(frequencies)
                    for cv_index, cv in enumerate(target_cvs)
                ]
            for group_name, frequency_hz, cv, weights in groups:
                safe = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
                effective_k = float(np.exp(-np.sum(safe * np.log(safe))))
                for regime_index, regime in enumerate(regime_order):
                    weight_rows.append({
                        "target_kind": target_kind, "mode": mode, "weight_group": group_name,
                        "freq": frequency_hz, "cv": cv, "thresh_type": regime,
                        "weight": float(weights[regime_index]), "effective_regime_count": effective_k,
                        "max_group_weight": float(np.max(weights)),
                        "role": "frozen_control" if target_kind == "quadratic" and mode == "global" else
                                "representational_upper_bound" if mode == "per_point" else "comparison",
                    })

        target_by_cv = {float(cv): targets[:, index] for index, cv in enumerate(target_cvs)}
        for fit in fits:
            for frequency_index, frequency_hz in enumerate(frequencies):
                for cv_index, cv in enumerate(sim_cvs):
                    behavior_row = behavior_summary[
                        np.isclose(behavior_summary["freq"], frequency_hz)
                        & np.isclose(behavior_summary["cv"], cv)
                    ]
                    curve_tables.append(pd.DataFrame([{
                        "target_kind": target_kind, "mode": fit["mode"], "freq": frequency_hz, "cv": cv,
                        "behavior_target": float(target_by_cv[float(cv)][frequency_index]) if float(cv) in target_by_cv else np.nan,
                        "behavior_mean": float(behavior_row.iloc[0]["behavior_mean"]) if not behavior_row.empty else np.nan,
                        "behavior_sem": float(behavior_row.iloc[0]["behavior_sem"]) if not behavior_row.empty else np.nan,
                        "prediction": float(fit["prediction"][frequency_index, cv_index]),
                        "prediction_absolute_uA": float(fit["absolute"][frequency_index, cv_index]),
                        "panel_bias": float(fit["biases"][frequency_index]),
                        "selected_single_regime": (
                            regime_order[int(fit["regime_index"])]
                            if fit["mode"] == "best_single" else
                            regime_order[int(fit["regime_indices"][frequency_index])]
                            if fit["mode"] == "best_single_per_frequency" else ""
                        ),
                    }]))

    curves = pd.concat(curve_tables, ignore_index=True)
    metrics = pd.concat(metric_tables, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    metrics["selected_single_regime"] = ""
    for target_kind in target_kinds:
        selected = regime_order[int(fit_lookup[(target_kind, "best_single")]["regime_index"])]
        metrics.loc[(metrics["target_kind"] == target_kind) & (metrics["mode"] == "best_single"), "selected_single_regime"] = selected
        per_frequency_selected = fit_lookup[(target_kind, "best_single_per_frequency")]["regime_indices"]
        for frequency_index, frequency_hz in enumerate(frequencies):
            metrics.loc[
                (metrics["target_kind"] == target_kind)
                & (metrics["mode"] == "best_single_per_frequency")
                & np.isclose(pd.to_numeric(metrics["freq"], errors="coerce"), frequency_hz),
                "selected_single_regime",
            ] = regime_order[int(per_frequency_selected[frequency_index])]
        metrics.loc[
            (metrics["target_kind"] == target_kind)
            & (metrics["mode"] == "best_single_per_frequency")
            & metrics["freq"].astype(str).eq("ALL"),
            "selected_single_regime",
        ] = "; ".join(
            f"{frequency_hz:g}Hz={regime_order[int(per_frequency_selected[index])]}"
            for index, frequency_hz in enumerate(frequencies)
        )
    metrics["per_point_smooth_lambda"] = float(per_point_smooth_lambda)
    metrics["allow_panel_bias"] = bool(allow_panel_bias)
    metrics["criterion_note"] = (
        "AIC/BIC use Gaussian residuals and nominal parameter counts; "
        "best_single_per_frequency conservatively counts one discrete choice per frequency, "
        "and per-point is an oracle rather than a primary model."
    )

    outcomes_path = Path(recruitment_outcomes_csv) if recruitment_outcomes_csv else root / "data" / "detection" / "population_recruitment_outcomes.csv"
    population_states = amplitude_summary.rename(columns={"amp_uA_mean": "index_amp_uA_mean", "amp_uA_sem": "index_amp_uA_sem"})
    if outcomes_path.is_file():
        outcomes = pd.read_csv(outcomes_path)
        outcomes = outcomes[outcomes["mode"].astype(str).str.casefold().eq("true")].copy()
        population_states = population_states.merge(
            outcomes, on=["freq", "cv", "thresh_type"], how="left", validate="one_to_one",
        )

    population_mixture_rows: list[dict[str, Any]] = []
    state_value_columns = [
        column for column in (
            "index_amp_uA_mean", "mean_index_spikes", "axons_ever_recruited",
            "axons_always_recruited", "expected_active_axons", "expected_total_spikes",
            "expected_active_fraction",
        ) if column in population_states.columns
    ]
    for target_kind in target_kinds:
        for fit_mode in (
            "best_single", "best_single_per_frequency", "global", "per_frequency", "per_point",
        ):
            fit = fit_lookup[(target_kind, fit_mode)]
            if fit_mode == "best_single":
                weights_all = np.zeros_like(amplitudes)
                weights_all[:, :, int(fit["regime_index"])] = 1.0
            elif fit_mode == "best_single_per_frequency":
                weights_all = np.asarray(fit["weights_all"], dtype=float)
            else:
                weights_all = np.asarray(fit["weights_all"], dtype=float)
            for frequency_index, frequency_hz in enumerate(frequencies):
                for cv_index, cv in enumerate(sim_cvs):
                    rows = population_states[
                        np.isclose(population_states["freq"], frequency_hz)
                        & np.isclose(population_states["cv"], cv)
                    ].set_index("thresh_type").reindex(regime_order)
                    row: dict[str, Any] = {
                        "target_kind": target_kind, "mode": fit_mode,
                        "freq": float(frequency_hz), "cv": float(cv),
                        "panel_bias": float(fit["biases"][frequency_index]),
                        "predicted_normalized_threshold": float(fit["prediction"][frequency_index, cv_index]),
                    }
                    mixture_weights = weights_all[frequency_index, cv_index]
                    for column in state_value_columns:
                        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
                        row[f"mixture_{column}"] = (
                            float(np.sum(mixture_weights * values)) if np.isfinite(values).all() else np.nan
                        )
                    population_mixture_rows.append(row)
    population_mixtures = pd.DataFrame(population_mixture_rows)

    frozen_by_target: dict[str, pd.DataFrame] = {}
    frozen_single_by_target: dict[str, pd.DataFrame] = {}
    for target_kind in target_kinds:
        frozen_target = weights[
            (weights["target_kind"] == target_kind) & (weights["mode"] == "global")
        ].copy()
        if len(frozen_target) != len(regime_order) or not np.isclose(frozen_target["weight"].sum(), 1.0):
            raise RuntimeError(
                f"Global {target_kind} detection mixture did not produce one valid frozen simplex vector"
            )
        frozen_target = frozen_target[
            ["thresh_type", "weight", "effective_regime_count", "max_group_weight"]
        ].copy()
        frozen_target.insert(0, "target_kind", target_kind)
        frozen_target.insert(0, "weights_source", f"population_detection_global_{target_kind}")
        frozen_by_target[target_kind] = frozen_target

        frozen_single = weights[
            (weights["target_kind"] == target_kind)
            & (weights["mode"] == "best_single_per_frequency")
        ].copy()
        expected_rows = len(frequencies) * len(regime_order)
        if len(frozen_single) != expected_rows:
            raise RuntimeError(
                f"Per-frequency best-single {target_kind} export has "
                f"{len(frozen_single)} rows; expected {expected_rows}"
            )
        frequency_sums = frozen_single.groupby("freq", as_index=False)["weight"].sum()
        if not np.allclose(frequency_sums["weight"].to_numpy(float), 1.0):
            raise RuntimeError(
                f"Per-frequency best-single {target_kind} weights are not one-hot simplex vectors"
            )
        frozen_single = frozen_single[
            ["freq", "thresh_type", "weight", "effective_regime_count", "max_group_weight"]
        ].copy()
        frozen_single.insert(0, "target_kind", target_kind)
        frozen_single.insert(
            0, "weights_source", f"population_detection_best_single_per_frequency_{target_kind}",
        )
        frozen_single_by_target[target_kind] = frozen_single

    # The manuscript-facing detection comparison uses smooth quadratic trends.
    # Keep the direct-datapoint vector as a sensitivity export, while the stable
    # generic filename points to the quadratic-derived frozen observer control.
    frozen_control_target = "quadratic" if "quadratic" in frozen_by_target else target_kinds[0]
    frozen = frozen_by_target[frozen_control_target].copy()
    frozen_single = frozen_single_by_target[frozen_control_target].copy()

    paths = {
        "behavior_points": output_dir / "detection_behavior_points.csv",
        "population_states": output_dir / "detection_population_states.csv",
        "curves": output_dir / "detection_curves.csv",
        "metrics": output_dir / "detection_metrics.csv",
        "weights": output_dir / "detection_weights.csv",
        "frozen_global_weights": output_dir / "detection_frozen_global_weights.csv",
        "frozen_per_frequency_best_single": output_dir / "detection_frozen_per_frequency_best_single.csv",
        "mixture_population_summary": output_dir / "detection_mixture_population_summary.csv",
        "manifest": output_dir / "run_manifest.json",
    }
    _atomic_dataframe_csv(behavior, paths["behavior_points"])
    _atomic_dataframe_csv(population_states, paths["population_states"])
    _atomic_dataframe_csv(curves, paths["curves"])
    _atomic_dataframe_csv(metrics, paths["metrics"])
    _atomic_dataframe_csv(weights, paths["weights"])
    _atomic_dataframe_csv(frozen, paths["frozen_global_weights"])
    _atomic_dataframe_csv(frozen_single, paths["frozen_per_frequency_best_single"])
    for target_kind, frozen_target in frozen_by_target.items():
        target_path = output_dir / f"detection_frozen_global_weights_{target_kind}.csv"
        paths[f"frozen_global_weights_{target_kind}"] = target_path
        _atomic_dataframe_csv(frozen_target, target_path)
        single_target_path = output_dir / f"detection_frozen_per_frequency_best_single_{target_kind}.csv"
        paths[f"frozen_per_frequency_best_single_{target_kind}"] = single_target_path
        _atomic_dataframe_csv(frozen_single_by_target[target_kind], single_target_path)
    _atomic_dataframe_csv(population_mixtures, paths["mixture_population_summary"])
    _atomic_json({
        "analysis": "population_detection_behavioral_matching_v3",
        "regime_order": regime_order, "frequencies_hz": frequencies.tolist(),
        "simulation_cvs": sim_cvs.tolist(), "behavioral_cvs": behavior_cvs.tolist(),
        "target_kinds": list(target_kinds), "allow_panel_bias": bool(allow_panel_bias),
        "per_point_smooth_lambda": float(per_point_smooth_lambda), "seed": int(seed),
        "n_starts": int(n_starts),
        "primary_behavior_target": "quadratic" if "quadratic" in target_kinds else target_kinds[0],
        "primary_figure_comparison": "best_single_per_frequency_vs_per_point_upper_bound",
        "constrained_comparisons": [
            "best_single", "best_single_per_frequency", "global", "per_frequency",
        ],
        "per_point_role": "legacy_oracle_upper_bound",
        "frozen_control_model": "global",
        "frozen_control_target": frozen_control_target,
        "frozen_control_source": paths["frozen_global_weights"].name,
        "frozen_per_frequency_best_single_source": paths["frozen_per_frequency_best_single"].name,
    }, paths["manifest"])
    return paths


# --------------- Command-line entry point --------------- #

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common_commands: dict[str, argparse.ArgumentParser] = {}
    for name in ("identify-index", "index-regimes", "compute-index-cv0", "simulate-population", "simulate-population-cv0", "run-all"):
        cmd = sub.add_parser(name)
        common_commands[name] = cmd
        cmd.add_argument("--root", type=Path, default=PROJECT_ROOT)
        cmd.add_argument("--n-jobs", type=int, default=4)
        cmd.add_argument("--force", action="store_true")
    common_commands["identify-index"].add_argument("--population-manifest", type=Path, default=None)
    common_commands["identify-index"].add_argument(
        "--output-dir", type=Path, default=None,
        help="Write index-screen CSVs outside production detection data (useful for preflight).",
    )
    index_condition = sub.add_parser("run-index-regime-condition")
    index_condition.add_argument("--root", type=Path, default=PROJECT_ROOT)
    index_condition.add_argument("--condition-index", type=int, required=True)
    index_condition.add_argument("--n-jobs", type=int, default=4)
    index_condition.add_argument("--partial-dir", type=Path, required=True)
    index_condition.add_argument("--overwrite", action="store_true")
    merge_index = sub.add_parser("merge-index-regimes")
    merge_index.add_argument("--root", type=Path, default=PROJECT_ROOT)
    merge_index.add_argument("--partial-dir", type=Path, required=True)
    merge_index.add_argument("--output", type=Path, default=None)
    recruitment = sub.add_parser("run-population-recruitment")
    recruitment.add_argument("--root", type=Path, default=PROJECT_ROOT)
    recruitment.add_argument("--mode", choices=("true", "cv0"), required=True)
    recruitment.add_argument("--job-index", type=int, required=True)
    recruitment.add_argument("--n-jobs", type=int, default=4)
    recruitment.add_argument("--partial-dir", type=Path, required=True)
    recruitment.add_argument("--overwrite", action="store_true")
    recruitment.add_argument("--population-manifest", type=Path, default=None)
    recruitment.add_argument("--max-pruning-audit-rate", type=float, default=DEFAULT_MAX_PRUNING_AUDIT_RATE)
    recruitment.add_argument("--no-max-pruning", action="store_true")
    target_condition = sub.add_parser("run-population-target-condition")
    target_condition.add_argument("--root", type=Path, default=PROJECT_ROOT)
    target_condition.add_argument("--mode", choices=("true", "cv0"), required=True)
    target_condition.add_argument("--target-axon-id", required=True)
    target_condition.add_argument("--cv", type=float, required=True)
    target_condition.add_argument("--freq", type=float, required=True)
    target_condition.add_argument("--n-jobs", type=int, default=4)
    target_condition.add_argument("--partial-dir", type=Path, required=True)
    target_condition.add_argument("--diameter-screen", type=Path, default=None)
    target_condition.add_argument("--max-pruning-audit-rate", type=float, default=DEFAULT_MAX_PRUNING_AUDIT_RATE)
    target_condition.add_argument("--diameter-pruning-audit-rate", type=float, default=DEFAULT_DIAMETER_PRUNING_AUDIT_RATE)
    target_condition.add_argument("--no-max-pruning", action="store_true")
    target_condition.add_argument("--overwrite", action="store_true")
    trajectory_condition = sub.add_parser("run-population-trajectory-condition")
    trajectory_condition.add_argument("--root", type=Path, default=PROJECT_ROOT)
    trajectory_condition.add_argument("--mode", choices=("true", "cv0"), required=True)
    trajectory_condition.add_argument("--trajectory-job-index", type=int, required=True)
    trajectory_condition.add_argument("--n-jobs", type=int, default=4)
    trajectory_condition.add_argument("--partial-dir", type=Path, required=True)
    trajectory_condition.add_argument("--population-manifest", type=Path, default=None)
    trajectory_condition.add_argument("--max-pruning-audit-rate", type=float, default=DEFAULT_MAX_PRUNING_AUDIT_RATE)
    trajectory_condition.add_argument("--diameter-pruning-audit-rate", type=float, default=DEFAULT_DIAMETER_PRUNING_AUDIT_RATE)
    trajectory_condition.add_argument("--no-max-pruning", action="store_true")
    trajectory_condition.add_argument("--no-diameter-pruning", action="store_true")
    trajectory_condition.add_argument("--overwrite", action="store_true")
    merge_recruitment = sub.add_parser("merge-population-recruitment")
    merge_recruitment.add_argument("--root", type=Path, default=PROJECT_ROOT)
    merge_recruitment.add_argument("--mode", choices=("true", "cv0"), required=True)
    merge_recruitment.add_argument("--partial-dir", type=Path, required=True)
    merge_recruitment.add_argument("--output", type=Path, default=None)
    merge_recruitment.add_argument("--population-manifest", type=Path, default=None)
    verify_index = sub.add_parser("verify-index")
    verify_index.add_argument("--root", type=Path, default=PROJECT_ROOT)
    verify_index.add_argument("--expected-axon-id", required=True)
    fit_behavior = sub.add_parser(
        "fit-behavior",
        help="Fit population detection operating-state mixtures without running NEURON.",
    )
    fit_behavior.add_argument("--root", type=Path, default=PROJECT_ROOT)
    fit_behavior.add_argument("--output-dir", type=Path, default=None)
    fit_behavior.add_argument(
        "--target-kinds", default="datapoints,quadratic",
        help="Comma-separated targets: datapoints, quadratic.",
    )
    fit_behavior.add_argument("--n-starts", type=int, default=4)
    fit_behavior.add_argument("--index-regimes-csv", type=Path, default=None)
    fit_behavior.add_argument("--recruitment-outcomes-csv", type=Path, default=None)
    fit_behavior.add_argument("--seed", type=int, default=20260828)
    fit_behavior.add_argument("--per-point-smooth-lambda", type=float, default=0.001)
    fit_behavior.add_argument("--no-panel-bias", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "identify-index":
        print(identify_index_axon(
            root=args.root, n_jobs=args.n_jobs, force=args.force,
            population_manifest=args.population_manifest, output_dir=args.output_dir,
        ))
    elif args.command == "run-index-regime-condition":
        print(run_index_regime_condition(
            root=args.root, condition_index=args.condition_index, n_jobs=args.n_jobs,
            partial_dir=args.partial_dir, overwrite=args.overwrite,
        ))
    elif args.command == "merge-index-regimes":
        print(merge_index_regime_partials(root=args.root, partial_dir=args.partial_dir, output_csv=args.output))
    elif args.command == "index-regimes":
        print(f"Wrote {len(compute_index_regimes(root=args.root, n_jobs=args.n_jobs, force=args.force))} index-regime rows.")
    elif args.command == "compute-index-cv0":
        print(compute_index_cv0_threshold_max(root=args.root, force=args.force))
    elif args.command == "run-population-recruitment":
        print(run_population_recruitment_job(
            root=args.root, mode=args.mode, job_index=args.job_index, n_jobs=args.n_jobs,
            partial_dir=args.partial_dir, overwrite=args.overwrite,
            population_manifest=args.population_manifest,
            prune_on_max_failure=not args.no_max_pruning,
            max_pruning_audit_rate=args.max_pruning_audit_rate,
        ))
    elif args.command == "run-population-target-condition":
        print(run_population_target_condition(
            root=args.root, mode=args.mode, target_axon_id=args.target_axon_id,
            cv=args.cv, freq=args.freq, n_jobs=args.n_jobs, partial_dir=args.partial_dir,
            overwrite=args.overwrite, prune_on_max_failure=not args.no_max_pruning,
            max_pruning_audit_rate=args.max_pruning_audit_rate,
            diameter_screen_path=args.diameter_screen,
            diameter_pruning_audit_rate=args.diameter_pruning_audit_rate,
        ))
    elif args.command == "run-population-trajectory-condition":
        print(run_population_trajectory_condition_job(
            root=args.root, mode=args.mode, trajectory_job_index=args.trajectory_job_index,
            n_jobs=args.n_jobs, partial_dir=args.partial_dir, overwrite=args.overwrite,
            population_manifest=args.population_manifest,
            prune_on_max_failure=not args.no_max_pruning,
            max_pruning_audit_rate=args.max_pruning_audit_rate,
            prune_by_diameter=not args.no_diameter_pruning,
            diameter_pruning_audit_rate=args.diameter_pruning_audit_rate,
        ))
    elif args.command == "merge-population-recruitment":
        print(merge_population_recruitment_partials(
            root=args.root, mode=args.mode, partial_dir=args.partial_dir, output_csv=args.output,
            population_manifest=args.population_manifest,
        ))
    elif args.command == "verify-index":
        print(verify_index_axon(root=args.root, expected_axon_id=args.expected_axon_id))
    elif args.command == "fit-behavior":
        target_kinds = tuple(kind.strip() for kind in args.target_kinds.split(",") if kind.strip())
        paths = run_population_detection_behavioral_matching(
            root=args.root,
            output_dir=args.output_dir,
            target_kinds=target_kinds,
            allow_panel_bias=not args.no_panel_bias,
            per_point_smooth_lambda=args.per_point_smooth_lambda,
            seed=args.seed,
            n_starts=args.n_starts,
            index_regimes_csv=args.index_regimes_csv,
            recruitment_outcomes_csv=args.recruitment_outcomes_csv,
        )
        print("\n".join(f"{name}: {path}" for name, path in paths.items()))
    elif args.command == "simulate-population":
        print(simulate_population_at_index_regimes(root=args.root, n_jobs=args.n_jobs, overwrite=args.force))
    elif args.command == "simulate-population-cv0":
        print(simulate_population_at_index_cv0_regimes(root=args.root, n_jobs=args.n_jobs, overwrite=args.force))
    else:
        identify_index_axon(root=args.root, n_jobs=args.n_jobs, force=args.force)
        compute_index_regimes(root=args.root, n_jobs=args.n_jobs, force=args.force)
        from detection.accepted_cv0 import prepare_accepted_index
        prepare_accepted_index(
            args.root.resolve(), args.root.resolve() / "data/detection/index_cv0_threshold_max.csv",
        )
        print(simulate_population_at_index_regimes(root=args.root, n_jobs=args.n_jobs, overwrite=args.force))
        print(simulate_population_at_index_cv0_regimes(root=args.root, n_jobs=args.n_jobs, overwrite=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
