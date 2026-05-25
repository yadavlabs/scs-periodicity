"""detection_core.py

Core utilities for:
- Loading space-time extracellular potentials from .npz files
- Running the MRG axon model in NEURON with Ve(t, node) playback
- Efficiently searching for detection thresholds (per repetition)
- Parallelizing across individual repetitions using joblib.Parallel (process-based)

Threshold definitions implemented here (spike-count based):
- I_first     : minimum amplitude that elicits >= 1 spike ("onset")
- I_volley5   : minimum amplitude that elicits >= 5 spikes
- I_volley10  : minimum amplitude that elicits >= 10 spikes
- I_5         : minimum amplitude that elicits >= 5% of input pulses as spikes
- I_10        : minimum amplitude that elicits >= 10% of input pulses as spikes
- I_15        : minimum amplitude that elicits >= 15% of input pulses as spikes
- I_25        : minimum amplitude that elicits >= 25% of input pulses as spikes
- I_50        : minimum amplitude that elicits >= 50% of input pulses as spikes
- I_max       : "saturation" threshold: minimum amplitude that achieves the maximum
                spike count observed (plateau), not necessarily equal to 100% of
                input pulses due to refractory effects

Important downstream behavior (handled in compute_cv0_threshold_max):
- I_first/I_volley5 are *safeguarded* by using the maximum I_first/I_volley5 across ALL CVs for a
  given (cutline, freq), ensuring onset spiking for every CV downstream.
- All other thresholds are computed as the maximum across CV=0 repetitions only
  (Slack-style "CV0-based" amplitudes).

Design choices:
- NEURON objects are NEVER sent between processes.
- Each worker process loads NEURON + mechanisms locally.
"""

import os
import gc
import platform
import csv
import json
import ast
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from tqdm import tqdm

import faulthandler, sys
faulthandler.enable(all_threads=True, file=sys.__stderr__)

# Avoid oversubscription (important on Longleaf)
os.environ.setdefault("OMP_NUM_THREADS", "1")

# ---------------- Configuration ----------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = (PROJECT_ROOT / "data" / "space-time-V" / "generated").resolve()
MRG_MODEL_DIR = (PROJECT_ROOT / "hpc_tools" / "MRG_model").resolve()
RESULTS_DIR = (PROJECT_ROOT / "data" / "detection" / "generated")

OUTPUT_CSV = RESULTS_DIR / "detection_thresholds.csv"
CV0_THRESH_CSV = RESULTS_DIR / "cv0_threshold_max.csv"
DETECTION_RESPONSES_CSV = RESULTS_DIR / "detection_responses.csv"
CV0_RESPONSES_CSV = RESULTS_DIR / "cv0_based_responses.csv"

THRESH_TYPES = (
    "I_first",
    "I_volley5",
    "I_volley10",
    "I_5",
    "I_10",
    "I_15",
    "I_25",
    "I_50",
    "I_max",
)
LEGACY_I_MAX_NAME = "I_" + "all"
LEGACY_THRESHOLD_ALIASES = {LEGACY_I_MAX_NAME: "I_max"}


def canonical_threshold_type(thresh_type: Any) -> str:
    text = str(thresh_type)
    return LEGACY_THRESHOLD_ALIASES.get(text, text)


def normalize_threshold_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy threshold names to the release-facing I_max convention."""

    out = df.copy()
    if "thresh_type" in out.columns:
        out["thresh_type"] = out["thresh_type"].map(canonical_threshold_type)
    rename: dict[str, str] = {}
    for legacy, canonical in LEGACY_THRESHOLD_ALIASES.items():
        for suffix in ("_uA", "0_uA", "_spikes"):
            old = f"{legacy}{suffix}"
            new = f"{canonical}{suffix}"
            if old in out.columns and new not in out.columns:
                rename[old] = new
        old_n = f"N_{legacy}_spikes"
        new_n = f"N_{canonical}_spikes"
        if old_n in out.columns and new_n not in out.columns:
            rename[old_n] = new_n
    if rename:
        out = out.rename(columns=rename)
    return out

RESPONSE_COLUMNS = (
    "file",
    "cutline",
    "cv",
    "freq",
    "rep_index",
    "rep_label",
    "thresh_type",
    "amp_uA",
    "n_spikes",
    "spike_times_ms",
)

DEFAULT_CV_VALUES = tuple(round(i / 10, 1) for i in range(11))
DEFAULT_FREQUENCIES_HZ = (10, 20, 50, 100, 200)
STIM_DURATION_S = 2.0

RECORD_NODE_INDEX = 25
SPIKE_HEIGHT_MV = 0.0
INIT_V_MV = -80.0

PREPAD_MS = 100.0
TAIL_MS = 100.0

# Volley (absolute spike-count) thresholds
VOLLEY5_MIN_SPIKES = 5
VOLLEY10_MIN_SPIKES = 10

# Percent-of-pulses thresholds to compute
PCT_LEVELS = (0.05, 0.10, 0.15, 0.25, 0.50)  # I_5, I_10, I_15, I_25, I_50
PERCENT_SPIKE_THRESH_TYPES = ("I_5", "I_10", "I_15", "I_25", "I_50", "I_max")

FIRST_SPIKE_SEARCH = dict(
    amp_start_uA=200.0,
    amp_max_uA=1000.0,
    bracket_factor=2.0,
    tol_uA=0.1,
    min_spikes=1,
)

ALL_SPIKES_SEARCH = dict(
    amp_max_uA=2000.0,
    growth_factor=2.0,
    plateau_repeats=3,
    tol_uA=0.1,
)

# --------------- Utility: filename parsing --------------- #

def parse_metadata_from_filename(path: Path) -> Tuple[int, float, float]:
    """Parse (cutline, cv, freq) from a file stem like:
    M12_SpaceTimeV_cv0.2_f50_SpaceTimeV.npz
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path.name}")

    cutline_str = parts[0]
    if not cutline_str.startswith("M"):
        raise ValueError(f"Unexpected cutline prefix in {path.name}")
    cutline = int(cutline_str[1:])

    cv_str = parts[2]
    if not cv_str.startswith("cv"):
        raise ValueError(f"Unexpected CV part in {path.name}")
    cv = float(cv_str[2:])

    freq_str = parts[3]
    if not freq_str.startswith("f"):
        raise ValueError(f"Unexpected frequency part in {path.name}")
    freq = float(freq_str[1:])

    return cutline, cv, freq


# --------------- Loading space-time Ve from .npz --------------- #

def load_spacetime_from_npz(npz_path: Path):
    """Load the space-time bundle saved from COMSOL export.

    Expected keys:
      - time_s (T,)
      - node_numbers (N,)
      - rep_labels (R,)
      - current (R, T)
      - V (R, T, N) : Ve in mV for 1 uA boundary current source
    """
    data = np.load(npz_path)

    time_ms = data["time_s"].astype(np.float32) * 1e3
    node_numbers = data["node_numbers"]
    rep_labels = data["rep_labels"]
    current = data["current"]
    V_mV_1uA = data["V"].astype(np.float32)

    return time_ms, node_numbers, rep_labels, current, V_mV_1uA


# ------------- NEURON / MRG Helpers ------------- #

_MRG_LOADED = False
_STDRUN_LOADED = False


def _compiled_mechanism_candidates() -> tuple[Path, ...]:
    dll_name = "nrnmech.dll" if platform.system() == "Windows" else "libnrnmech.so"
    return (
        MRG_MODEL_DIR / "x86_64" / dll_name,
        MRG_MODEL_DIR / dll_name,
    )


def _compiled_mechanism_path() -> Path | None:
    for candidate in _compiled_mechanism_candidates():
        if candidate.exists():
            return candidate
    return None


def _get_h():
    from neuron import h as _h
    return _h


def ensure_mrg_loaded(hoc_file: str = "MRGaxon.hoc") -> None:
    """Load NEURON mechanisms + stdrun + MRGaxon.hoc once per process."""
    global _MRG_LOADED, _STDRUN_LOADED
    if _MRG_LOADED:
        return

    if not MRG_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"MRG model directory not found: {MRG_MODEL_DIR}. "
            "Install the upstream ModelDB model with "
            "`python hpc_tools/install_mrg_model.py --force` before running NEURON."
        )

    h = _get_h()

    # Load stdrun for h.continuerun / h.run behavior
    if not _STDRUN_LOADED:
        try:
            h.load_file("stdrun.hoc")
        except Exception:
            pass
        _STDRUN_LOADED = True

    # Load compiled mechanisms. nrnivmodl writes the shared library under
    # hpc_tools/MRG_model/x86_64/ on Linux; older/local layouts may place it at the root.
    dll_path = _compiled_mechanism_path()
    if dll_path is not None:
        dll_str = str(dll_path).replace("\\", "/")
        print(f"[worker {os.getpid()}] Loading NEURON mechanisms from: {dll_str}")
        h.nrn_load_dll(dll_str)
    else:
        mosinit = MRG_MODEL_DIR / "mosinit.hoc"
        if mosinit.exists():
            mosinit_str = str(mosinit).replace("\\", "/")
            print(f"[worker {os.getpid()}] compiled mechanisms not found, loading mosinit.hoc instead: {mosinit_str}")
            h.load_file(mosinit_str)
        else:
            raise FileNotFoundError(
                "Neither compiled mechanisms nor mosinit.hoc found in "
                f"{MRG_MODEL_DIR}. Install the upstream ModelDB model with "
                "`python hpc_tools/install_mrg_model.py --force`, then compile "
                "the MOD files for this platform."
            )

    hoc_path = MRG_MODEL_DIR / hoc_file
    if not hoc_path.exists():
        raise FileNotFoundError(f"{hoc_file} not found in {MRG_MODEL_DIR}")
    hoc_str = str(hoc_path).replace("\\", "/")
    print(f"[worker {os.getpid()}] Loading {hoc_file} from: {hoc_str}")
    h.load_file(hoc_str)

    # Keep fixed-step integration (avoid surprises)
    try:
        h.cvode_active(0)
    except Exception:
        pass

    _MRG_LOADED = True


def prepend_zeros_ms(
    time_ms: np.ndarray,
    V_mV: np.ndarray,
    prepad_ms: float = PREPAD_MS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepend a short prepad window with Ve=0 to settle initial conditions."""
    time_ms = np.asarray(time_ms, dtype=np.float32)
    V_mV = np.asarray(V_mV, dtype=np.float32)

    if prepad_ms is None or prepad_ms <= 0 or time_ms.size < 2:
        return time_ms, V_mV

    dt = float(time_ms[1] - time_ms[0])
    if dt <= 0:
        return time_ms, V_mV

    n_pad = int(np.ceil(prepad_ms / dt))
    if n_pad <= 0:
        return time_ms, V_mV

    prepad_ms_exact = n_pad * dt
    t_pad = (np.arange(n_pad, dtype=np.float32) * dt)
    t_shift = time_ms + np.float32(prepad_ms_exact)
    t_new = np.concatenate([t_pad, t_shift], axis=0)

    n_nodes = V_mV.shape[1]
    V_pad = np.zeros((n_pad, n_nodes), dtype=np.float32)
    V_new = np.concatenate([V_pad, V_mV], axis=0)

    return t_new, V_new


def trim_tail_window_ms(
    time_ms,
    V_mV_1uA_rep,
    current_rep,
    tail_ms: float = TAIL_MS,
    eps: float = 1e-6,
):
    """Trim simulation to end shortly after stimulation stops."""
    time_ms = np.asarray(time_ms, dtype=np.float32)
    V_rep = np.asarray(V_mV_1uA_rep, dtype=np.float32)
    cur = np.asarray(current_rep, dtype=np.float32)

    T = time_ms.shape[0]
    assert V_rep.shape[0] == T
    assert cur.shape[0] == T

    active = np.abs(cur) >= eps
    if not np.any(active):
        return time_ms, V_rep

    last_idx = np.where(active)[0][-1]
    t_last = time_ms[last_idx] + tail_ms
    cut_idx = np.searchsorted(time_ms, t_last)
    cut_idx = min(cut_idx, T - 1)

    sl = slice(0, cut_idx + 1)
    return time_ms[sl], V_rep[sl, :]


class MRGrunner:
    """NEURON runner with Ve playback."""

    def __init__(self, time_ms, V_mV_1uA_rep, record_node_index: int = RECORD_NODE_INDEX):
        ensure_mrg_loaded()
        self.h = _get_h()

        self.time_ms = np.asarray(time_ms, dtype=np.float32)
        self.V_base = np.asarray(V_mV_1uA_rep, dtype=np.float32)

        num_time_points, num_nodes = self.V_base.shape

        try:
            self.mrg_nodes = int(self.h.axonnodes)
        except Exception:
            self.mrg_nodes = int(self.h.node.count())

        if num_nodes != self.mrg_nodes:
            raise ValueError(f"V has {num_nodes} nodes, MRG model has {self.mrg_nodes} nodes.")

        if num_time_points > 1:
            dt_ms = float(self.time_ms[1] - self.time_ms[0])
            self.h.dt = dt_ms

        self.tvec = self.h.Vector(self.time_ms)

        self.base_v = []
        self.play_v = []
        for i in range(self.mrg_nodes):
            b = self.h.Vector(self.V_base[:, i])
            v = self.h.Vector(self.V_base[:, i])
            v.play(self.h.node[i](0.5)._ref_e_extracellular, self.tvec, 1)
            self.base_v.append(b)
            self.play_v.append(v)

        self.rec_idx = int(record_node_index)
        self.v_rec = self.h.Vector()
        self.t_rec = self.h.Vector()
        self.v_rec.record(self.h.node[self.rec_idx](0.5)._ref_v)
        self.t_rec.record(self.h._ref_t)

        self.tstop = float(self.time_ms[-1])

    def run_mrg_for_amp(
        self,
        scale_uA: float,
        init_v_mV: float = INIT_V_MV,
        spike_height_mV: float = SPIKE_HEIGHT_MV,
    ):
        """Run NEURON for a single current scale and return spike times/count."""
        self.v_rec.resize(0)
        self.t_rec.resize(0)

        h = self.h

        s = float(scale_uA)
        for b, v in zip(self.base_v, self.play_v):
            v.copy(b)
            v.mul(s)

        h.tstop = float(self.tstop)
        h.finitialize(float(init_v_mV))
        try:
            h.frecord_init()
        except Exception:
            pass
        h.continuerun(h.tstop)

        v_arr = np.asarray(self.v_rec)
        t_arr = np.asarray(self.t_rec)

        peak_idx, _ = find_peaks(v_arr, height=float(spike_height_mV))
        spike_times_ms = t_arr[peak_idx]
        return spike_times_ms, int(spike_times_ms.size)


def spike_count_for_amp_runner(runner: MRGrunner, scale_uA: float) -> int:
    """Convenience wrapper to run and return spike count only."""
    _, n_spikes = runner.run_mrg_for_amp(scale_uA)
    return n_spikes


# ------------- Threshold Search Helpers ------------- #

def find_first_spike_threshold(
    spike_counter,
    amp_start_uA: float = 200.0,
    amp_max_uA: float = 1000.0,
    bracket_factor: float = 2.0,
    tol_uA: float = 1.0,
    min_spikes: int = 1,
):
    """Binary search for the minimum amplitude producing >= min_spikes."""
    lo = 0.0
    hi = amp_start_uA

    best_amp = None
    best_n = -1
    second_amp = None
    second_n = -1

    def consider(amp, n):
        nonlocal best_amp, best_n, second_amp, second_n
        if n > best_n:
            second_amp, second_n = best_amp, best_n
            best_amp, best_n = amp, n
        elif n > second_n and amp != best_amp and n < best_n:
            second_amp, second_n = amp, n

    n_hi = spike_counter(hi)
    consider(hi, n_hi)

    while n_hi < min_spikes and hi < amp_max_uA:
        lo = hi
        hi *= bracket_factor
        n_hi = spike_counter(hi)
        consider(hi, n_hi)

    if n_hi < min_spikes:
        return None, 0, best_amp, best_n, second_amp, second_n

    while (hi - lo) > tol_uA:
        mid = 0.5 * (lo + hi)
        n_mid = spike_counter(mid)
        consider(mid, n_mid)
        if n_mid >= min_spikes:
            hi = mid
        else:
            lo = mid

    I_first = hi
    N_first = spike_counter(hi)
    return I_first, N_first, best_amp, best_n, second_amp, second_n


def find_all_spikes_threshold(
    spike_counter,
    I_first: float,
    amp_max_uA: float = 2000.0,
    growth_factor: float = 2.0,
    plateau_repeats: int = 3,
    tol_uA: float = 1.0,
    hint_best: Optional[Tuple[float, int]] = None,
    hint_second: Optional[Tuple[float, int]] = None,
):
    """Find a 'saturation' threshold based on spike-count plateau."""
    amps: List[float] = []
    counts: List[int] = []

    best_amp, best_n = hint_best if hint_best is not None else (None, -1)
    second_amp, second_n = hint_second if hint_second is not None else (None, -1)
    have_best = best_amp is not None and best_n >= 0

    def add_obs(amp: Optional[float], n: Optional[int]) -> None:
        if amp is None or n is None:
            return
        amps.append(float(amp))
        counts.append(int(n))

    def finalize_min_amp_for_observed_max(hi_candidate: Optional[float]):
        """Return the minimum amplitude producing the maximum count observed in this search."""
        if hi_candidate is not None:
            add_obs(float(hi_candidate), spike_counter(float(hi_candidate)))

        last_target_n = -1
        for _ in range(20):
            amps_arr = np.array(amps, dtype=float)
            counts_arr = np.array(counts, dtype=int)
            if counts_arr.size == 0 or counts_arr.max() <= 0:
                return None, amps_arr, counts_arr

            target_n = int(counts_arr.max())
            order = np.argsort(amps_arr)
            amps_sorted = amps_arr[order]
            counts_sorted = counts_arr[order]
            hit_idx = np.where(counts_sorted >= target_n)[0]
            if hit_idx.size == 0:
                return None, amps_arr, counts_arr

            hi_idx = int(hit_idx[0])
            hi = float(amps_sorted[hi_idx])
            lo = float(I_first) if hi_idx == 0 else float(amps_sorted[hi_idx - 1])
            lo = max(float(lo), float(I_first))

            while (hi - lo) > tol_uA:
                mid = 0.5 * (lo + hi)
                n_mid = spike_counter(mid)
                add_obs(mid, n_mid)
                if n_mid >= target_n:
                    hi = mid
                else:
                    lo = mid

            add_obs(hi, spike_counter(hi))
            new_target_n = int(np.array(counts, dtype=int).max())
            if new_target_n == target_n and target_n == last_target_n:
                return float(hi), np.array(amps, dtype=float), np.array(counts, dtype=int)
            if new_target_n == target_n:
                last_target_n = target_n
                continue
            last_target_n = new_target_n

        return float(hi), np.array(amps, dtype=float), np.array(counts, dtype=int)

    n_maxamp = spike_counter(amp_max_uA)
    add_obs(amp_max_uA, n_maxamp)
    add_obs(best_amp, best_n)
    add_obs(second_amp, second_n)

    if not have_best:
        amp = max(float(I_first), float(tol_uA))
        last_counts: List[int] = []

        while amp <= amp_max_uA:
            n = spike_counter(amp)
            add_obs(amp, n)

            last_counts.append(n)
            if len(last_counts) > plateau_repeats:
                last_counts.pop(0)

            if len(last_counts) == plateau_repeats and len(set(last_counts)) == 1:
                break
            amp *= growth_factor

        amps_arr = np.array(amps, dtype=float)
        counts_arr = np.array(counts, dtype=int)
        if counts_arr.size == 0 or counts_arr.max() == 0:
            return None, amps_arr, counts_arr

        N_max = int(counts_arr.max())
        idx_first_max = int(np.where(counts_arr == N_max)[0][0])
        hi = float(amps_arr[idx_first_max])
        lo = float(amps_arr[idx_first_max - 1]) if idx_first_max > 0 else float(I_first)

        while (hi - lo) > tol_uA:
            mid = 0.5 * (lo + hi)
            n_mid = spike_counter(mid)
            amps_arr = np.append(amps_arr, mid)
            counts_arr = np.append(counts_arr, n_mid)
            if n_mid >= N_max:
                hi = mid
            else:
                lo = mid

        return finalize_min_amp_for_observed_max(hi)

    targetN = max(int(best_n), int(n_maxamp))
    if targetN <= 0:
        return None, np.array(amps, dtype=float), np.array(counts, dtype=int)

    if n_maxamp > best_n:
        lo = min(best_amp, amp_max_uA)
        hi = max(best_amp, amp_max_uA)
    else:
        if second_amp is not None and second_n >= 0:
            lo = min(best_amp, second_amp)
            hi = max(best_amp, second_amp)
        else:
            lo = min(I_first, best_amp)
            hi = max(I_first, best_amp)

    lo = max(float(lo), float(I_first))
    hi = min(float(hi), float(amp_max_uA))

    while (hi - lo) > tol_uA:
        mid = 0.5 * (lo + hi)
        n_mid = spike_counter(mid)
        add_obs(mid, n_mid)
        if n_mid >= targetN:
            hi = mid
        else:
            lo = mid

    return finalize_min_amp_for_observed_max(float(hi))


def compute_thresholds_for_rep(
    time_ms,
    V_mV_1uA_rep,
    current_rep,
    n_input_pulses: Optional[int] = None,
    first_spike_params: Dict[str, Any] = None,
    all_spikes_params: Dict[str, Any] = None,
):
    """
    Compute thresholds for a single repetition.

    Returns a dict with:
      I_*_uA (floats or None) and N_*_spikes (ints)
      plus amps_tested_uA / counts_tested from the I_max search.

    Key point: all thresholds share a single amp->spike_count cache within this rep.
    """
    if first_spike_params is None:
        first_spike_params = FIRST_SPIKE_SEARCH
    if all_spikes_params is None:
        all_spikes_params = ALL_SPIKES_SEARCH

    tol_uA = float(all_spikes_params.get("tol_uA", 1.0))

    out: Dict[str, Any] = {}

    try:
        time_trim, V_trim = trim_tail_window_ms(time_ms, V_mV_1uA_rep, current_rep)
        time_trim, V_trim = prepend_zeros_ms(time_trim, V_trim, prepad_ms=PREPAD_MS)

        runner = MRGrunner(time_trim, V_trim, record_node_index=RECORD_NODE_INDEX)

        cache: Dict[float, Tuple[np.ndarray, int]] = {}

        def spike_counter(amp_uA: float) -> int:
            amp = float(amp_uA)
            if amp in cache:
                return int(cache[amp][1])
            spike_times_ms, n = runner.run_mrg_for_amp(
                scale_uA=amp,
                init_v_mV=INIT_V_MV,
                spike_height_mV=SPIKE_HEIGHT_MV,
            )
            cache[amp] = (np.asarray(spike_times_ms, dtype=float), int(n))
            return int(n)

        # ---------------- I_first (>=1) ----------------
        I_first, N_first, best_amp, best_n, second_amp, second_n = find_first_spike_threshold(
            spike_counter, **first_spike_params
        )
        if I_first is None:
            # Fill all as None/0
            for k in ["I_first", "I_volley5", "I_volley10", "I_5", "I_10", "I_15", "I_25", "I_50", "I_max"]:
                out[f"{k}_uA"] = None
                out[f"N_{k}_spikes"] = 0
            out["amps_tested_uA"] = np.array([], dtype=float)
            out["counts_tested"] = np.array([], dtype=int)
            out["N_max_spikes"] = 0
            out["response_rows"] = []
            return out

        if float(I_first) not in cache:
            spike_counter(float(I_first))
        out["I_first_uA"] = float(I_first)
        out["N_I_first_spikes"] = int(N_first)

        # ---------------- I_max (plateau / max) ----------------
        I_max, amps_tested, counts_tested = find_all_spikes_threshold(
            spike_counter,
            I_first,
            hint_best=(best_amp, best_n),
            hint_second=(second_amp, second_n),
            **all_spikes_params,
        )
        out["I_max_uA"] = (None if I_max is None else float(I_max))
        out["amps_tested_uA"] = np.array([] if amps_tested is None else np.atleast_1d(amps_tested), dtype=float)
        out["counts_tested"] = np.array([] if counts_tested is None else np.atleast_1d(counts_tested), dtype=int)

        N_max = int(out["counts_tested"].max()) if (out["counts_tested"].size > 0) else 0
        out["N_max_spikes"] = int(N_max)
        out["N_I_max_spikes"] = 0 if I_max is None else int(spike_counter(float(I_max)))

        # --------------- generic helper (min spikes) ---------------
        def refine_min_spikes(min_spikes: int) -> Tuple[Optional[float], int]:
            """Return min amp achieving >= min_spikes using cache + bisection."""
            if min_spikes <= 0:
                return 0.0, spike_counter(0.0)

            # If already computed some thresholds, cache has a spread of amps.
            # Find first cached amp that satisfies.
            amps_arr = np.array(sorted(cache.keys()), dtype=float)
            counts_arr = np.array([cache[a][1] for a in amps_arr], dtype=int)
            cond = counts_arr >= int(min_spikes)

            if not np.any(cond):
                # Not seen yet: force-evaluate amp_max (from all-spikes search) so we at least
                # know if target is reachable without a full new growth sweep.
                amp_max_uA = float(all_spikes_params.get("amp_max_uA", 2000.0))
                n_maxamp = spike_counter(amp_max_uA)
                # Rebuild arrays
                amps_arr = np.array(sorted(cache.keys()), dtype=float)
                counts_arr = np.array([cache[a][1] for a in amps_arr], dtype=int)
                cond = counts_arr >= int(min_spikes)
                if not np.any(cond):
                    return None, 0

            hi_idx = int(np.where(cond)[0][0])
            hi = float(amps_arr[hi_idx])
            N_hi = int(counts_arr[hi_idx])

            if hi_idx == 0:
                lo = 0.0
                N_lo = spike_counter(lo)
            else:
                lo = float(amps_arr[hi_idx - 1])
                N_lo = int(counts_arr[hi_idx - 1])

            # lo might already satisfy due to non-monotonicity; if so, return lo
            if N_lo >= int(min_spikes):
                return float(lo), int(N_lo)

            while (hi - lo) > tol_uA:
                mid = 0.5 * (lo + hi)
                N_mid = spike_counter(mid)
                if N_mid >= int(min_spikes):
                    hi, N_hi = mid, int(N_mid)
                else:
                    lo = mid
                    N_lo = int(N_mid)

            return float(hi), int(N_hi)

        # ---------------- volley thresholds ----------------
        I_v5, N_v5 = refine_min_spikes(VOLLEY5_MIN_SPIKES)
        out["I_volley5_uA"] = (None if I_v5 is None else float(I_v5))
        out["N_I_volley5_spikes"] = int(N_v5)

        I_v10, N_v10 = refine_min_spikes(VOLLEY10_MIN_SPIKES)
        out["I_volley10_uA"] = (None if I_v10 is None else float(I_v10))
        out["N_I_volley10_spikes"] = int(N_v10)

        # ---------------- percent-of-pulses thresholds ----------------
        # If n_input_pulses not provided, mark them as None
        if n_input_pulses is None or int(n_input_pulses) <= 0:
            for pct, name in [(0.05, "I_5"), (0.10, "I_10"), (0.15, "I_15"), (0.25, "I_25"), (0.50, "I_50")]:
                out[f"{name}_uA"] = None
                out[f"N_{name}_spikes"] = 0
        else:
            nP = int(n_input_pulses)

            # Compute integer spike targets for each pct
            pct_targets: Dict[str, int] = {}
            for pct, name in [(0.05, "I_5"), (0.10, "I_10"), (0.15, "I_15"), (0.25, "I_25"), (0.50, "I_50")]:
                tgt = int(np.ceil(float(pct) * nP))
                tgt = max(1, tgt)
                pct_targets[name] = tgt

            # Build a reuse map: if two names share the same target spike count, compute once
            target_to_names: Dict[int, List[str]] = {}
            for nm, tgt in pct_targets.items():
                target_to_names.setdefault(int(tgt), []).append(nm)

            # Also allow reuse vs I_first/I_volley5/I_volley10 if they coincide
            # (e.g., 5% at 10 Hz => 1 spike)
            known_targets: Dict[int, Tuple[Optional[float], int]] = {}
            known_targets[1] = (out["I_first_uA"], out["N_I_first_spikes"])
            if out["I_volley5_uA"] is not None:
                known_targets[VOLLEY5_MIN_SPIKES] = (out["I_volley5_uA"], out["N_I_volley5_spikes"])
            if out["I_volley10_uA"] is not None:
                known_targets[VOLLEY10_MIN_SPIKES] = (out["I_volley10_uA"], out["N_I_volley10_spikes"])

            for tgt, names in sorted(target_to_names.items(), key=lambda x: x[0]):
                if tgt in known_targets and known_targets[tgt][0] is not None:
                    I_t, N_t = known_targets[tgt]
                else:
                    I_t, N_t = refine_min_spikes(int(tgt))
                    if I_t is not None:
                        known_targets[int(tgt)] = (float(I_t), int(N_t))

                for nm in names:
                    out[f"{nm}_uA"] = (None if I_t is None else float(I_t))
                    out[f"N_{nm}_spikes"] = int(N_t) if I_t is not None else 0

        # Ensure remaining keys exist
        for k in ["I_first", "I_volley5", "I_volley10", "I_5", "I_10", "I_15", "I_25", "I_50", "I_max"]:
            out.setdefault(f"{k}_uA", None)
            out.setdefault(f"N_{k}_spikes", 0)
        out.setdefault("N_max_spikes", 0)

        response_rows: List[Dict[str, Any]] = []
        for thresh_type in THRESH_TYPES:
            amp = out.get(f"{thresh_type}_uA")
            if amp is None or (isinstance(amp, float) and np.isnan(amp)):
                continue
            amp_f = float(amp)
            if amp_f not in cache:
                spike_counter(amp_f)
            spike_times_ms, n_spikes = cache[amp_f]
            response_rows.append(
                dict(
                    thresh_type=str(thresh_type),
                    amp_uA=amp_f,
                    n_spikes=int(n_spikes),
                    spike_times_ms=json.dumps(list(map(float, np.atleast_1d(spike_times_ms)))),
                )
            )
        out["response_rows"] = response_rows

        return out

    except Exception as e:
        print(f"[THRESH-ERROR pid={os.getpid()}] NEURON failed: {e}")
        for k in ["I_first", "I_volley5", "I_volley10", "I_5", "I_10", "I_15", "I_25", "I_50", "I_max"]:
            out[f"{k}_uA"] = None
            out[f"N_{k}_spikes"] = 0
        out["amps_tested_uA"] = np.array([], dtype=float)
        out["counts_tested"] = np.array([], dtype=int)
        out["N_max_spikes"] = 0
        out["response_rows"] = []
        return out


# ------------- Job construction (per rep) ------------- #

def build_rep_jobs(
    data_dir: Path = DATA_DIR,
    npz_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> List[Dict[str, Any]]:
    """Build one job per repetition across one or more .npz files."""
    jobs: List[Dict[str, Any]] = []

    if npz_paths is None:
        npz_files = sorted(data_dir.glob("M*_SpaceTimeV.npz"))
    else:
        npz_files = [Path(p).resolve() for p in npz_paths]
        npz_files = sorted(npz_files)

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found (data_dir={data_dir}, npz_paths={npz_paths})")

    for npz_path in npz_files:
        cutline, cv, freq = parse_metadata_from_filename(npz_path)
        time_ms, node_numbers, rep_labels, current, V_mV_1uA = load_spacetime_from_npz(npz_path)
        num_reps = V_mV_1uA.shape[0]

        for rep_idx in range(num_reps):
            rep_label = str(rep_labels[rep_idx])
            jobs.append(
                dict(
                    file_path=str(npz_path),
                    file_name=npz_path.name,
                    cutline=int(cutline),
                    cv=float(cv),
                    freq=float(freq),
                    rep_index=int(rep_idx),
                    rep_label=rep_label,
                )
            )

        del time_ms, node_numbers, rep_labels, current, V_mV_1uA
        gc.collect()

    return jobs


def process_rep_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint: compute thresholds for one repetition."""
    npz_path = Path(job["file_path"])
    rep_idx = int(job["rep_index"])

    try:
        time_ms, node_numbers, rep_labels, current, V_mV_1uA = load_spacetime_from_npz(npz_path)
        V_rep = V_mV_1uA[rep_idx]
        cur_rep = current[rep_idx]

        freq_hz = float(job["freq"])
        n_input_pulses = int(round(freq_hz * 2.0))  # 2 s train

        thr = compute_thresholds_for_rep(
            time_ms, V_rep, cur_rep, n_input_pulses=n_input_pulses
        )

        row = dict(
            file=job["file_name"],
            cutline=job["cutline"],
            cv=job["cv"],
            freq=job["freq"],
            rep_index=job["rep_index"],
            rep_label=job["rep_label"],
            # Threshold amplitudes
            I_first_uA=thr["I_first_uA"],
            I_volley5_uA=thr["I_volley5_uA"],
            I_volley10_uA=thr["I_volley10_uA"],
            I_5_uA=thr["I_5_uA"],
            I_10_uA=thr["I_10_uA"],
            I_15_uA=thr["I_15_uA"],
            I_25_uA=thr["I_25_uA"],
            I_50_uA=thr["I_50_uA"],
            I_max_uA=thr["I_max_uA"],
            # Spike counts at those thresholds
            N_I_first_spikes=int(thr["N_I_first_spikes"]),
            N_I_volley5_spikes=int(thr["N_I_volley5_spikes"]),
            N_I_volley10_spikes=int(thr["N_I_volley10_spikes"]),
            N_I_5_spikes=int(thr["N_I_5_spikes"]),
            N_I_10_spikes=int(thr["N_I_10_spikes"]),
            N_I_15_spikes=int(thr["N_I_15_spikes"]),
            N_I_25_spikes=int(thr["N_I_25_spikes"]),
            N_I_50_spikes=int(thr["N_I_50_spikes"]),
            N_I_max_spikes=int(thr["N_I_max_spikes"]),
            N_max_spikes=int(thr["N_max_spikes"]),
            amps_tested_uA=json.dumps(list(map(float, np.atleast_1d(thr["amps_tested_uA"])))),
            counts_tested=json.dumps(list(map(int, np.atleast_1d(thr["counts_tested"])))),
        )

        response_rows = []
        for rr in thr.get("response_rows", []):
            response_rows.append(
                dict(
                    file=job["file_name"],
                    cutline=job["cutline"],
                    cv=job["cv"],
                    freq=job["freq"],
                    rep_index=job["rep_index"],
                    rep_label=job["rep_label"],
                    thresh_type=str(rr["thresh_type"]),
                    amp_uA=float(rr["amp_uA"]),
                    n_spikes=int(rr["n_spikes"]),
                    spike_times_ms=str(rr["spike_times_ms"]),
                )
            )

    except Exception as e:
        print(f"[REP-ERROR pid={os.getpid()}] {npz_path.name} rep {rep_idx}: {e}")
        row = dict(
            file=npz_path.name,
            cutline=job["cutline"],
            cv=job["cv"],
            freq=job["freq"],
            rep_index=job["rep_index"],
            rep_label=job["rep_label"],
            I_first_uA=None,
            I_volley5_uA=None,
            I_volley10_uA=None,
            I_5_uA=None,
            I_10_uA=None,
            I_15_uA=None,
            I_25_uA=None,
            I_50_uA=None,
            I_max_uA=None,
            N_I_first_spikes=0,
            N_I_volley5_spikes=0,
            N_I_volley10_spikes=0,
            N_I_5_spikes=0,
            N_I_10_spikes=0,
            N_I_15_spikes=0,
            N_I_25_spikes=0,
            N_I_50_spikes=0,
            N_I_max_spikes=0,
            N_max_spikes=0,
            amps_tested_uA=json.dumps([]),
            counts_tested=json.dumps([]),
        )
        response_rows = []

    try:
        del time_ms, node_numbers, rep_labels, current, V_mV_1uA, V_rep, cur_rep
    except Exception:
        pass
    gc.collect()

    return {"threshold": row, "responses": response_rows}


# ------------- CV0 threshold MAX table ------------- #

def compute_cv0_threshold_max(
    rows: List[Dict[str, Any]],
    output_csv: Path = CV0_THRESH_CSV,
    safety_margin_uA: float = 0.0,
):
    """
    Output is one row per (cutline, freq), with columns:

      I_first0_uA    : SAFEGUARDED max(I_first_uA) across ALL CVs
      I_volley5_0_uA : SAFEGUARDED max(I_volley5_uA) across ALL CVs
      others         : max(threshold_uA) across CV==0 reps only
    """
    if not rows:
        print("[core] No rows passed to compute_cv0_threshold_max.")
        return None

    df = pd.DataFrame(rows)
    if "cv" not in df.columns:
        print("[core] 'cv' column missing; cannot compute threshold table.")
        return None

    # --- I_first safeguard: max across ALL CVs ---
    first_agg = (
        df.groupby(["cutline", "freq"], as_index=False)
          .agg(I_first0_uA=("I_first_uA", "max"))
    )

    # --- I_volley5 safeguard: max across ALL CVs ---
    volley5_agg = (
        df.groupby(["cutline", "freq"], as_index=False)
        .agg(I_volley5_0_uA=("I_volley5_uA", "max"))
    )

    # --- CV0 MAX for everything else ---
    cv0 = df[df["cv"] == 0.0].copy()
    if cv0.empty:
        print("[core] No cv=0 rows found; cannot compute cv0-based thresholds.")
        return None

    cv0_agg = (
        cv0.groupby(["cutline", "freq"], as_index=False)
           .agg(
               I_volley10_0_uA=("I_volley10_uA", "max"),
               I_5_0_uA=("I_5_uA", "max"),
               I_10_0_uA=("I_10_uA", "max"),
               I_15_0_uA=("I_15_uA", "max"),
               I_25_0_uA=("I_25_uA", "max"),
               I_50_0_uA=("I_50_uA", "max"),
               I_max0_uA=("I_max_uA", "max"),
               n_reps=("rep_index", "count"),
           )
    )

    agg = pd.merge(first_agg, volley5_agg, on=["cutline", "freq"], how="inner")
    agg = pd.merge(agg, cv0_agg, on=["cutline", "freq"], how="inner")

    if safety_margin_uA and safety_margin_uA > 0:
        for c in [
            "I_first0_uA",
            "I_volley5_0_uA", "I_volley10_0_uA",
            "I_5_0_uA", "I_10_0_uA", "I_15_0_uA", "I_25_0_uA", "I_50_0_uA",
            "I_max0_uA",
        ]:
            agg[c] = agg[c].astype(float) + float(safety_margin_uA)

    agg.to_csv(output_csv, index=False)
    print(f"[core] Wrote threshold MAX table to {output_csv}")
    return agg


def _load_cv0_threshold_table(thresh_csv: Path = CV0_THRESH_CSV) -> Dict[Tuple[int, float], Dict[str, float]]:
    """Load per-(cutline,freq) threshold table into a dict for fast lookup."""
    if not thresh_csv.exists():
        raise FileNotFoundError(f"cv0 threshold file not found: {thresh_csv}")

    df0 = normalize_threshold_frame(pd.read_csv(thresh_csv))

    key_to_amps: Dict[Tuple[int, float], Dict[str, float]] = {}
    for _, row in df0.iterrows():
        key = (int(row["cutline"]), float(row["freq"]))
        key_to_amps[key] = dict(
            I_first=float(row["I_first0_uA"]),
            I_volley5=float(row["I_volley5_0_uA"]),
            I_volley10=float(row["I_volley10_0_uA"]),
            I_5=float(row["I_5_0_uA"]),
            I_10=float(row["I_10_0_uA"]),
            I_15=float(row["I_15_0_uA"]),
            I_25=float(row["I_25_0_uA"]),
            I_50=float(row["I_50_0_uA"]),
            I_max=float(row["I_max0_uA"]),
        )
    return key_to_amps


# ------------- Responses from CV0 threshold table ------------- #

def build_response_jobs_from_cv0(
    data_dir: Path = DATA_DIR,
    cv0_thresh_csv: Path = CV0_THRESH_CSV,
    npz_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> List[Dict[str, Any]]:
    """Build response jobs: for every repetition, re-run NEURON at amplitudes in cv0 table."""
    key_to_amps = _load_cv0_threshold_table(cv0_thresh_csv)
    base_jobs = build_rep_jobs(data_dir=data_dir, npz_paths=npz_paths)

    jobs: List[Dict[str, Any]] = []
    for job in base_jobs:
        key = (int(job["cutline"]), float(job["freq"]))
        amps = key_to_amps.get(key)
        if amps is None:
            continue
        job2 = dict(job)
        job2["amps"] = amps
        jobs.append(job2)

    print(f"[core] Built {len(jobs)} cv0-response jobs")
    return jobs


def _process_response_job_cv0(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Worker entrypoint: run one repetition at all threshold amplitudes."""
    npz_path = Path(job["file_path"])
    rep_idx  = int(job["rep_index"])
    amps: Dict[str, float] = job["amps"]

    rows_out: List[Dict[str, Any]] = []

    try:
        time_ms, node_numbers, rep_labels, current, V_mV_1uA = load_spacetime_from_npz(npz_path)
        V_rep   = V_mV_1uA[rep_idx]
        cur_rep = current[rep_idx]

        time_trim_ms, V_trim = trim_tail_window_ms(time_ms, V_rep, cur_rep, tail_ms=TAIL_MS)
        time_trim_ms, V_trim = prepend_zeros_ms(time_trim_ms, V_trim, prepad_ms=PREPAD_MS)

        runner = MRGrunner(time_trim_ms, V_trim, record_node_index=RECORD_NODE_INDEX)

        for thresh_name, amp in amps.items():
            if amp is None or (isinstance(amp, float) and np.isnan(amp)):
                continue

            spike_times_ms, n_spikes = runner.run_mrg_for_amp(
                scale_uA=float(amp),
                init_v_mV=INIT_V_MV,
                spike_height_mV=SPIKE_HEIGHT_MV,
            )

            rows_out.append(
                dict(
                    file=npz_path.name,
                    cutline=job["cutline"],
                    cv=job["cv"],
                    freq=job["freq"],
                    rep_index=rep_idx,
                    rep_label=job["rep_label"],
                    thresh_type=str(thresh_name),
                    amp_uA=float(amp),
                    n_spikes=int(n_spikes),
                    spike_times_ms=json.dumps(list(map(float, np.atleast_1d(spike_times_ms)))),
                )
            )

    except Exception as e:
        print(f"[CV0-RESP-ERROR pid={os.getpid()}] {npz_path.name} rep {rep_idx}: {e}")

    try:
        del time_ms, node_numbers, rep_labels, current, V_mV_1uA, V_rep, cur_rep
    except Exception:
        pass
    gc.collect()

    return rows_out


def _threshold_amp_map_from_row(row: pd.Series) -> Dict[str, float]:
    """Extract per-repetition true threshold amplitudes from one threshold row."""
    row = normalize_threshold_frame(row.to_frame().T).iloc[0]
    amps: Dict[str, float] = {}
    for thresh_type in THRESH_TYPES:
        amp_col = f"{thresh_type}_uA"
        if amp_col not in row.index:
            continue
        amp = row.get(amp_col)
        if amp is None or pd.isna(amp):
            continue
        amps[str(thresh_type)] = float(amp)
    return amps


def build_response_jobs_from_thresholds(
    data_dir: Path = DATA_DIR,
    threshold_csv: Path = OUTPUT_CSV,
    npz_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> List[Dict[str, Any]]:
    """Build true-threshold response jobs from detection_thresholds.csv."""
    threshold_csv = Path(threshold_csv).resolve()
    if not threshold_csv.exists():
        raise FileNotFoundError(f"detection threshold file not found: {threshold_csv}")

    df = normalize_threshold_frame(pd.read_csv(threshold_csv))
    path_by_name: Dict[str, Path] = {}
    if npz_paths is not None:
        path_by_name = {Path(path).resolve().name: Path(path).resolve() for path in npz_paths}
        df = df[df["file"].astype(str).isin(set(path_by_name))]

    jobs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        file_name = str(row["file"])
        amps = _threshold_amp_map_from_row(row)
        if not amps:
            continue

        npz_path = path_by_name.get(file_name, (Path(data_dir).resolve() / file_name))
        jobs.append(
            dict(
                file_path=str(npz_path),
                file_name=file_name,
                cutline=int(row["cutline"]),
                cv=float(row["cv"]),
                freq=float(row["freq"]),
                rep_index=int(row["rep_index"]),
                rep_label=str(row["rep_label"]),
                amps=amps,
            )
        )

    print(f"[core] Built {len(jobs)} true-threshold response jobs")
    return jobs


def _process_response_job_thresholds(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Worker entrypoint: rerun one repetition at its own saved thresholds."""
    npz_path = Path(job["file_path"])
    rep_idx = int(job["rep_index"])
    amps: Dict[str, float] = job["amps"]

    rows_out: List[Dict[str, Any]] = []

    try:
        time_ms, node_numbers, rep_labels, current, V_mV_1uA = load_spacetime_from_npz(npz_path)
        rep_label = str(rep_labels[rep_idx])
        if rep_label != str(job["rep_label"]):
            raise ValueError(
                f"Stage-02 label mismatch for {npz_path.name}: rep_index={rep_idx} "
                f"points to {rep_label!r}, but threshold row is labeled {str(job['rep_label'])!r}."
            )

        V_rep = V_mV_1uA[rep_idx]
        cur_rep = current[rep_idx]

        time_trim_ms, V_trim = trim_tail_window_ms(time_ms, V_rep, cur_rep, tail_ms=TAIL_MS)
        time_trim_ms, V_trim = prepend_zeros_ms(time_trim_ms, V_trim, prepad_ms=PREPAD_MS)

        runner = MRGrunner(time_trim_ms, V_trim, record_node_index=RECORD_NODE_INDEX)

        for thresh_name, amp in amps.items():
            if amp is None or (isinstance(amp, float) and np.isnan(amp)):
                continue

            spike_times_ms, n_spikes = runner.run_mrg_for_amp(
                scale_uA=float(amp),
                init_v_mV=INIT_V_MV,
                spike_height_mV=SPIKE_HEIGHT_MV,
            )

            rows_out.append(
                dict(
                    file=job["file_name"],
                    cutline=job["cutline"],
                    cv=job["cv"],
                    freq=job["freq"],
                    rep_index=rep_idx,
                    rep_label=rep_label,
                    thresh_type=str(thresh_name),
                    amp_uA=float(amp),
                    n_spikes=int(n_spikes),
                    spike_times_ms=json.dumps(list(map(float, np.atleast_1d(spike_times_ms)))),
                )
            )

    except Exception as e:
        print(f"[TRUE-RESP-ERROR pid={os.getpid()}] {npz_path.name} rep {rep_idx}: {e}")

    try:
        del time_ms, node_numbers, rep_labels, current, V_mV_1uA, V_rep, cur_rep
    except Exception:
        pass
    gc.collect()

    return rows_out


# ------------- High-level runners ------------- #

def run_all_reps(
    n_jobs: int = -1,
    output_csv: Path = OUTPUT_CSV,
    response_output_csv: Optional[Path] = None,
    data_dir: Path = DATA_DIR,
    npz_paths: Optional[Sequence[Union[str, Path]]] = None,
    chunk_size: int = 200,
    pre_dispatch: str = "2*n_jobs",
    compute_cv0_after: bool = True,
) -> List[Dict[str, Any]]:
    """Compute per-rep thresholds across files and write to a CSV incrementally (resume supported)."""
    jobs = build_rep_jobs(data_dir=data_dir, npz_paths=npz_paths)
    print(f"[core] Built {len(jobs)} rep jobs")

    done_keys = set()
    header_written = False
    response_header_written = bool(
        response_output_csv is not None
        and Path(response_output_csv).exists()
        and Path(response_output_csv).stat().st_size > 0
    )

    if output_csv.exists() and output_csv.stat().st_size > 0:
        try:
            df_done = pd.read_csv(output_csv, usecols=["file", "rep_index"])
            done_keys = set(zip(df_done["file"].astype(str), df_done["rep_index"].astype(int)))
            header_written = True
            print(f"[core] Resuming: found {len(done_keys)} completed reps in {output_csv}")
        except Exception as e:
            print(f"[core] Could not read existing output_csv for resume: {e}")

    pending = [j for j in jobs if (j["file_name"], int(j["rep_index"])) not in done_keys]
    print(f"[core] Pending reps: {len(pending)}")

    if not pending:
        print("[core] Nothing to do.")
        return pd.read_csv(output_csv).to_dict(orient="records")

    def _append_rows(rows: List[Dict[str, Any]]):
        nonlocal header_written
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        mode = "a" if header_written else "w"
        with output_csv.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not header_written:
                writer.writeheader()
                header_written = True
            for r in rows:
                writer.writerow(r)

    def _append_response_rows(rows: List[Dict[str, Any]]):
        nonlocal response_header_written
        if response_output_csv is None or not rows:
            return
        response_path = Path(response_output_csv)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if response_header_written else "w"
        with response_path.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(RESPONSE_COLUMNS))
            if not response_header_written:
                writer.writeheader()
                response_header_written = True
            for r in rows:
                writer.writerow(r)

    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        print(f"[core] Processing chunk {start}..{start+len(chunk)-1} ({len(chunk)} jobs)")

        chunk_rows = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            pre_dispatch=pre_dispatch,
            batch_size=1,
        )(
            delayed(process_rep_job)(job)
            for job in tqdm(chunk, desc="Processing reps", leave=False)
        )

        chunk_results = [r for r in chunk_rows if r is not None]
        threshold_rows = [r["threshold"] for r in chunk_results if r.get("threshold") is not None]
        response_rows = [
            rr
            for r in chunk_results
            for rr in r.get("responses", [])
        ]
        _append_rows(threshold_rows)
        _append_response_rows(response_rows)

        for r in threshold_rows:
            done_keys.add((str(r["file"]), int(r["rep_index"])) )

        gc.collect()

    print(f"[core] Finished. Output at {output_csv}")

    if compute_cv0_after:
        try:
            df_all = pd.read_csv(output_csv)
            compute_cv0_threshold_max(df_all.to_dict(orient="records"), output_csv=CV0_THRESH_CSV)
        except Exception as e:
            print(f"[core] Failed to compute cv0 threshold max: {e}")

    return []


def run_responses_from_cv0(
    n_jobs: int = -1,
    data_dir: Path = DATA_DIR,
    npz_paths: Optional[Sequence[Union[str, Path]]] = None,
    cv0_thresh_csv: Path = CV0_THRESH_CSV,
    output_csv: Path = CV0_RESPONSES_CSV,
) -> List[Dict[str, Any]]:
    """Run NEURON at the CV0-based (and I_first safeguarded) amplitudes."""
    jobs = build_response_jobs_from_cv0(data_dir=data_dir, cv0_thresh_csv=cv0_thresh_csv, npz_paths=npz_paths)
    if not jobs:
        print("[core] No cv0-response jobs to run.")
        return []

    all_rows: List[Dict[str, Any]] = []
    iterator = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_response_job_cv0)(job) for job in tqdm(jobs, desc="CV0 response reps")
    )
    for chunk in iterator:
        if chunk:
            all_rows.extend(chunk)

    if not all_rows:
        print("[core] No cv0-based response rows produced.")
        return []

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESPONSE_COLUMNS))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[core] Wrote {len(all_rows)} cv0-based response rows to {output_csv}")
    return all_rows


def run_responses_from_thresholds(
    n_jobs: int = -1,
    data_dir: Path = DATA_DIR,
    npz_paths: Optional[Sequence[Union[str, Path]]] = None,
    threshold_csv: Path = OUTPUT_CSV,
    output_csv: Path = DETECTION_RESPONSES_CSV,
) -> List[Dict[str, Any]]:
    """Run NEURON at each repetition's own saved true detection thresholds."""
    jobs = build_response_jobs_from_thresholds(
        data_dir=data_dir,
        threshold_csv=threshold_csv,
        npz_paths=npz_paths,
    )
    if not jobs:
        print("[core] No true-threshold response jobs to run.")
        return []

    all_rows: List[Dict[str, Any]] = []
    iterator = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_response_job_thresholds)(job)
        for job in tqdm(jobs, desc="True threshold response reps")
    )
    for chunk in iterator:
        if chunk:
            all_rows.extend(chunk)

    if not all_rows:
        print("[core] No true-threshold response rows produced.")
        return []

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESPONSE_COLUMNS))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[core] Wrote {len(all_rows)} true-threshold response rows to {output_csv}")
    return all_rows


def ensure_cv0_threshold_max_from_file(
    det_csv: Path = OUTPUT_CSV,
    output_csv: Path = CV0_THRESH_CSV,
    safety_margin_uA: float = 0.0,
):
    """Ensure the threshold MAX table exists (compute it if needed)."""
    if output_csv.exists():
        print(f"[core] cv0 threshold MAX already exists: {output_csv}")
        return pd.read_csv(output_csv)

    if not det_csv.exists():
        raise FileNotFoundError(f"[core] detection thresholds CSV not found: {det_csv}")

    print(f"[core] Computing threshold MAX table from {det_csv} ...")
    df = pd.read_csv(det_csv)
    rows = df.to_dict(orient="records")
    return compute_cv0_threshold_max(rows, output_csv=output_csv, safety_margin_uA=safety_margin_uA)


def ensure_responses_from_cv0(
    n_jobs: int = -1,
    data_dir: Path = DATA_DIR,
    cv0_thresh_csv: Path = CV0_THRESH_CSV,
    output_csv: Path = CV0_RESPONSES_CSV,
):
    """Ensure cv0_based_responses.csv exists (compute if needed)."""
    if output_csv.exists():
        print(f"[core] cv0-based responses already exist: {output_csv}")
        return pd.read_csv(output_csv)

    ensure_cv0_threshold_max_from_file(det_csv=OUTPUT_CSV, output_csv=cv0_thresh_csv, safety_margin_uA=0.0)

    print(f"[core] Running NEURON to build cv0-based responses into {output_csv} ...")
    rows = run_responses_from_cv0(
        n_jobs=n_jobs,
        data_dir=data_dir,
        cv0_thresh_csv=cv0_thresh_csv,
        output_csv=output_csv,
    )
    return rows


def ensure_responses_from_thresholds(
    n_jobs: int = -1,
    data_dir: Path = DATA_DIR,
    threshold_csv: Path = OUTPUT_CSV,
    output_csv: Path = DETECTION_RESPONSES_CSV,
):
    """Ensure detection_responses.csv exists (compute if needed)."""
    if output_csv.exists():
        print(f"[core] true-threshold responses already exist: {output_csv}")
        return pd.read_csv(output_csv)

    if not Path(threshold_csv).exists():
        raise FileNotFoundError(f"[core] detection thresholds CSV not found: {threshold_csv}")

    print(f"[core] Running NEURON to build true-threshold responses into {output_csv} ...")
    rows = run_responses_from_thresholds(
        n_jobs=n_jobs,
        data_dir=data_dir,
        threshold_csv=threshold_csv,
        output_csv=output_csv,
    )
    return rows


# ------------- Release-stage helpers ------------- #

@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for stage 03 detection-threshold generation and review."""

    cv_vals: Sequence[float] = DEFAULT_CV_VALUES
    freq_vals: Sequence[int] = DEFAULT_FREQUENCIES_HZ
    cutline: int = 101
    spacetime_dir: Path = DATA_DIR
    time_pattern_dir: Path = PROJECT_ROOT / "data" / "time-patterns" / "generated"
    output_dir: Path = RESULTS_DIR
    detection_csv: Path = OUTPUT_CSV
    cv0_threshold_csv: Path = CV0_THRESH_CSV
    detection_responses_csv: Path = DETECTION_RESPONSES_CSV
    cv0_responses_csv: Path = CV0_RESPONSES_CSV
    n_jobs: int = 1
    chunk_size: int = 200

    def resolved(self) -> "DetectionConfig":
        output_dir = Path(self.output_dir).expanduser().resolve()
        return replace(
            self,
            cv_vals=tuple(float(cv) for cv in self.cv_vals),
            freq_vals=tuple(int(freq) for freq in self.freq_vals),
            spacetime_dir=Path(self.spacetime_dir).expanduser().resolve(),
            time_pattern_dir=Path(self.time_pattern_dir).expanduser().resolve(),
            output_dir=output_dir,
            detection_csv=Path(self.detection_csv).expanduser().resolve(),
            cv0_threshold_csv=Path(self.cv0_threshold_csv).expanduser().resolve(),
            detection_responses_csv=Path(self.detection_responses_csv).expanduser().resolve(),
            cv0_responses_csv=Path(self.cv0_responses_csv).expanduser().resolve(),
        )


@dataclass(frozen=True)
class DetectionStatus:
    """Verification status for stage 03 inputs and cached outputs."""

    input_files: tuple[Path, ...]
    output_files: tuple[Path, ...]
    missing_input_files: tuple[Path, ...]
    missing_output_files: tuple[Path, ...]

    @property
    def inputs_complete(self) -> bool:
        return not self.missing_input_files

    @property
    def outputs_complete(self) -> bool:
        return not self.missing_output_files

    @property
    def complete(self) -> bool:
        return self.inputs_complete and self.outputs_complete

    def summary(self) -> str:
        input_msg = "OK" if self.inputs_complete else f"missing {len(self.missing_input_files)}"
        output_msg = "OK" if self.outputs_complete else f"missing {len(self.missing_output_files)}"
        return f"detection inputs: {input_msg}; outputs: {output_msg}"

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for role, paths in (("input", self.input_files), ("output", self.output_files)):
            for path in paths:
                rows.append(
                    {
                        "role": role,
                        "path": str(_display_path(path)),
                        "exists": path.exists(),
                        "size_bytes": path.stat().st_size if path.exists() else None,
                    }
                )
        return rows


@dataclass(frozen=True)
class DetectionResult:
    """Result returned by the stage-03 runner."""

    ran: bool
    status: DetectionStatus
    output_dir: Path

    def summary(self) -> str:
        action = "generated with NEURON" if self.ran else "reused existing"
        return f"Stage 03 {action} outputs. {self.status.summary()}"


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def iter_detection_pairs(config: DetectionConfig) -> list[tuple[float, int]]:
    return [(float(cv), int(freq)) for freq in config.freq_vals for cv in config.cv_vals]


def spacetime_input_path(config: DetectionConfig, cv: float, freq: float | int) -> Path:
    config = config.resolved()
    return config.spacetime_dir / (
        f"M{config.cutline}_1uA_cv{float(cv):.1f}_f{int(freq)}_SpaceTimeV.npz"
    )


def expected_spacetime_paths(config: DetectionConfig) -> tuple[Path, ...]:
    config = config.resolved()
    return tuple(spacetime_input_path(config, cv, freq) for cv, freq in iter_detection_pairs(config))


def expected_detection_outputs(config: DetectionConfig) -> tuple[Path, ...]:
    config = config.resolved()
    return (
        config.detection_csv,
        config.cv0_threshold_csv,
        config.detection_responses_csv,
        config.cv0_responses_csv,
    )


def verify_detection_outputs(config: DetectionConfig | None = None) -> DetectionStatus:
    config = (config or DetectionConfig()).resolved()
    input_files = expected_spacetime_paths(config)
    output_files = expected_detection_outputs(config)
    return DetectionStatus(
        input_files=input_files,
        output_files=output_files,
        missing_input_files=tuple(path for path in input_files if not path.exists()),
        missing_output_files=tuple(path for path in output_files if not path.exists()),
    )


def check_neuron_ready() -> tuple[bool, str]:
    """Return whether NEURON and compiled MRG mechanisms appear available."""
    try:
        import neuron  # noqa: F401
    except Exception as exc:
        return False, f"Could not import NEURON Python package: {exc}"

    required = ("AXNODE.mod", "MRGaxon.hoc")
    missing = [name for name in required if not (MRG_MODEL_DIR / name).exists()]
    if missing:
        return False, f"Missing MRG model files in {MRG_MODEL_DIR}: {missing}"

    dll_path = _compiled_mechanism_path()
    if dll_path is None:
        candidates = ", ".join(str(path) for path in _compiled_mechanism_candidates())
        return (
            False,
            f"NEURON imported, but compiled mechanisms were not found at any expected path: {candidates}.",
        )
    return True, "NEURON package and compiled MRG mechanisms found."


def require_neuron_ready() -> None:
    ok, message = check_neuron_ready()
    if not ok:
        raise RuntimeError(message)
    ensure_mrg_loaded()


def run_stage(config: DetectionConfig | None = None, force: bool = False) -> DetectionResult:
    """Reuse cached outputs by default; full reruns require NEURON."""
    config = (config or DetectionConfig()).resolved()
    status = verify_detection_outputs(config)
    if status.missing_input_files:
        preview = "\n".join(str(path) for path in status.missing_input_files[:5])
        raise FileNotFoundError("Stage 03 requires stage-02 space-time inputs:\n" + preview)
    if status.outputs_complete and not force:
        return DetectionResult(ran=False, status=status, output_dir=config.output_dir)
    if not force:
        preview = "\n".join(str(path) for path in status.missing_output_files[:5])
        raise FileNotFoundError("Stage 03 cached outputs are missing:\n" + preview)

    require_neuron_ready()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for path in expected_detection_outputs(config):
        if path.exists():
            path.unlink()

    run_all_reps(
        n_jobs=config.n_jobs,
        output_csv=config.detection_csv,
        response_output_csv=config.detection_responses_csv,
        data_dir=config.spacetime_dir,
        chunk_size=config.chunk_size,
        compute_cv0_after=False,
    )
    ensure_cv0_threshold_max_from_file(
        det_csv=config.detection_csv,
        output_csv=config.cv0_threshold_csv,
        safety_margin_uA=0.0,
    )
    ensure_responses_from_cv0(
        n_jobs=config.n_jobs,
        data_dir=config.spacetime_dir,
        cv0_thresh_csv=config.cv0_threshold_csv,
        output_csv=config.cv0_responses_csv,
    )

    status = verify_detection_outputs(config)
    if not status.outputs_complete:
        preview = "\n".join(str(path) for path in status.missing_output_files[:5])
        raise RuntimeError("Stage 03 rerun finished, but outputs are still missing:\n" + preview)
    return DetectionResult(ran=True, status=status, output_dir=config.output_dir)


def describe_npz_arrays(path: str | Path) -> list[dict[str, object]]:
    """Read saved .npz array names, shapes, and dtypes without loading arrays."""
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(Path(path)) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".npy"):
                continue
            array_name = Path(name).stem
            with archive.open(name) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError(f"Unsupported .npy header version {version} in {path}.")
            rows.append({"array": array_name, "shape": shape, "dtype": str(dtype)})
    return rows


def load_detection_tables(
    config: DetectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load stage-03 tables used by downstream manuscript stages."""
    config = (config or DetectionConfig()).resolved()
    return (
        normalize_threshold_frame(pd.read_csv(config.detection_csv)),
        normalize_threshold_frame(pd.read_csv(config.cv0_threshold_csv)),
        normalize_threshold_frame(pd.read_csv(config.cv0_responses_csv)),
    )


def load_detection_response_tables(
    config: DetectionConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load true-threshold and CV0-based response tables with matching schemas."""
    config = (config or DetectionConfig()).resolved()
    return (
        normalize_threshold_frame(pd.read_csv(config.detection_responses_csv)),
        normalize_threshold_frame(pd.read_csv(config.cv0_responses_csv)),
    )


def parse_numeric_list(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    text = str(value).strip()
    if not text:
        return np.array([], dtype=float)
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return np.array([], dtype=float)
    return np.asarray(parsed, dtype=float)


THRESH_SHORT_LABELS = {
    "I_first": "first",
    "I_volley5": "volley5",
    "I_volley10": "volley10",
    "I_5": "5",
    "I_10": "10",
    "I_15": "15",
    "I_25": "25",
    "I_50": "50",
    "I_max": "max",
}

FREQ_COLOR = {
    10.0: "#440154",
    20.0: "#3b528b",
    50.0: "#21918c",
    100.0: "#5ec962",
    200.0: "#fde725",
}

RASTER_COLORS = {
    "I_first": "#0072CE",
    "I_volley5": "#FF6A13",
    "I_volley10": "#FFB000",
    "I_5": "#8E44AD",
    "I_10": "#76B82A",
    "I_15": "#2DB7F5",
    "I_25": "#C41230",
    "I_50": "#0072CE",
    "I_max": "#F05A28",
}


def threshold_display_name(thresh_type: str) -> str:
    return "I_max" if str(thresh_type) == "I_max" else str(thresh_type)


def threshold_math_label(thresh_type: str) -> str:
    labels = {
        "I_first": r"$I_{first}$",
        "I_volley5": r"$I_{volley5}$",
        "I_volley10": r"$I_{volley10}$",
        "I_5": r"$I_5$",
        "I_10": r"$I_{10}$",
        "I_15": r"$I_{15}$",
        "I_25": r"$I_{25}$",
        "I_50": r"$I_{50}$",
        "I_max": r"$I_{max}$",
    }
    return labels.get(str(thresh_type), str(thresh_type))


def threshold_count_math_label(thresh_type: str) -> str:
    labels = {
        "I_first": r"$N_{first}$",
        "I_volley5": r"$N_{volley5}$",
        "I_volley10": r"$N_{volley10}$",
        "I_5": r"$N_5$",
        "I_10": r"$N_{10}$",
        "I_15": r"$N_{15}$",
        "I_25": r"$N_{25}$",
        "I_50": r"$N_{50}$",
        "I_max": r"$N_{max}$",
    }
    return labels.get(str(thresh_type), str(thresh_type))


def _expected_input_pulses(freq: float) -> float:
    return float(freq) * STIM_DURATION_S


def _nice_step(span: float, target_ticks: int = 6) -> float:
    span = float(max(span, 1e-9))
    raw = span / max(int(target_ticks) - 1, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    norm = raw / mag
    if norm <= 1.0:
        return 1.0 * mag
    if norm <= 2.0:
        return 2.0 * mag
    if norm <= 5.0:
        return 5.0 * mag
    return 10.0 * mag


def _set_axis_with_explicit_top_tick(ax, yvals: Sequence[float], target_ticks: int = 6, pad_frac: float = 0.04) -> None:
    arr = np.asarray(yvals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        span = max(abs(hi), 1.0) * 0.08
        lo, hi = lo - span, hi + span
    else:
        pad = (hi - lo) * float(pad_frac)
        lo, hi = lo - pad, hi + pad

    step = _nice_step(hi - lo, target_ticks=target_ticks)
    lo_tick = np.floor(lo / step) * step
    hi_tick = np.ceil(hi / step) * step
    if np.isclose(lo_tick, hi_tick):
        hi_tick = lo_tick + step
    ticks = np.arange(lo_tick, hi_tick + 0.5 * step, step)
    ax.set_ylim(lo_tick, hi_tick)
    ax.set_yticks(ticks)


def select_detection_row(
    df_thresholds: pd.DataFrame,
    cv: float,
    freq: float,
    rep_index: int,
    cutline: int = 101,
) -> pd.Series:
    df_thresholds = normalize_threshold_frame(df_thresholds)
    subset = df_thresholds[
        (df_thresholds["cutline"].astype(int) == int(cutline))
        & np.isclose(df_thresholds["cv"].astype(float), float(cv))
        & np.isclose(df_thresholds["freq"].astype(float), float(freq))
        & (df_thresholds["rep_index"].astype(int) == int(rep_index))
    ]
    if subset.empty:
        raise ValueError(f"No detection row for cutline={cutline}, cv={cv}, freq={freq}, rep_index={rep_index}")
    return subset.iloc[0]


def threshold_summary_for_row(row: pd.Series, thresh_types: Sequence[str] = THRESH_TYPES) -> pd.DataFrame:
    row = normalize_threshold_frame(row.to_frame().T).iloc[0]
    records = []
    for thresh_type in thresh_types:
        amp_col = f"{thresh_type}_uA"
        n_col = f"N_{thresh_type}_spikes"
        if amp_col not in row.index:
            continue
        records.append(
            {
                "thresh_type": thresh_type,
                "display_name": threshold_display_name(thresh_type),
                "amp_uA": row.get(amp_col),
                "n_spikes": row.get(n_col, np.nan),
            }
        )
    return pd.DataFrame.from_records(records)


def cv0_thresholds_long(df_cv0_thresholds: pd.DataFrame) -> pd.DataFrame:
    df_cv0_thresholds = normalize_threshold_frame(df_cv0_thresholds)
    records = []
    for _, row in df_cv0_thresholds.iterrows():
        for thresh_type in THRESH_TYPES:
            col = f"{thresh_type}0_uA"
            if col in row.index:
                records.append(
                    {
                        "cutline": int(row["cutline"]),
                        "freq": float(row["freq"]),
                        "thresh_type": thresh_type,
                        "amp_uA": float(row[col]),
                        "n_reps": int(row.get("n_reps", 0)),
                    }
                )
    return pd.DataFrame.from_records(records)


def detection_threshold_aggregate(df_thresholds: pd.DataFrame, value: str = "amp") -> pd.DataFrame:
    df_thresholds = normalize_threshold_frame(df_thresholds)
    records = []
    for _, row in df_thresholds.iterrows():
        for thresh_type in THRESH_TYPES:
            col = f"{thresh_type}_uA" if value == "amp" else f"N_{thresh_type}_spikes"
            if col in row.index:
                raw_value = float(row[col]) if pd.notna(row[col]) else np.nan
                if value != "amp" and thresh_type in PERCENT_SPIKE_THRESH_TYPES and pd.notna(raw_value):
                    raw_value = 100.0 * raw_value / _expected_input_pulses(float(row["freq"]))
                records.append(
                    {
                        "cutline": int(row["cutline"]),
                        "cv": float(row["cv"]),
                        "freq": float(row["freq"]),
                        "rep_index": int(row["rep_index"]),
                        "thresh_type": thresh_type,
                        "value": raw_value,
                    }
                )
    long = pd.DataFrame.from_records(records)
    if long.empty:
        return long
    return (
        long.groupby(["cutline", "cv", "freq", "thresh_type"], as_index=False)
        .agg(mean=("value", "mean"), sem=("value", "sem"), n=("value", "count"))
        .sort_values(["thresh_type", "freq", "cv"])
    )


def cv0_response_aggregate(df_cv0_responses: pd.DataFrame, error_mode: str = "sem") -> pd.DataFrame:
    df_cv0_responses = normalize_threshold_frame(df_cv0_responses)
    if df_cv0_responses.empty:
        return pd.DataFrame()
    values = df_cv0_responses.copy()
    values["value"] = values["n_spikes"].astype(float)
    is_percent_threshold = values["thresh_type"].isin(PERCENT_SPIKE_THRESH_TYPES)
    values.loc[is_percent_threshold, "value"] = (
        100.0
        * values.loc[is_percent_threshold, "n_spikes"].astype(float)
        / values.loc[is_percent_threshold, "freq"].astype(float).map(_expected_input_pulses)
    )
    agg = (
        values.groupby(["cutline", "cv", "freq", "thresh_type"], as_index=False)
        .agg(mean=("value", "mean"), sem=("value", "sem"), sd=("value", "std"), n=("value", "count"))
        .sort_values(["thresh_type", "freq", "cv"])
    )
    agg["err"] = agg["sd"] if error_mode == "sd" else agg["sem"]
    return agg


def summarize_output_tables(
    df_thresholds: pd.DataFrame,
    df_cv0_thresholds: pd.DataFrame,
    df_cv0_responses: pd.DataFrame,
    df_detection_responses: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = [
        {"table": "detection_thresholds", "rows": len(df_thresholds), "columns": len(df_thresholds.columns)},
        {"table": "cv0_threshold_max", "rows": len(df_cv0_thresholds), "columns": len(df_cv0_thresholds.columns)},
    ]
    if df_detection_responses is not None:
        rows.append(
            {"table": "detection_responses", "rows": len(df_detection_responses), "columns": len(df_detection_responses.columns)}
        )
    rows.append({"table": "cv0_based_responses", "rows": len(df_cv0_responses), "columns": len(df_cv0_responses.columns)})
    return pd.DataFrame(rows)


def summarize_cv0_responses(
    df_cv0_responses: pd.DataFrame,
    freq: float,
    thresh_types: Sequence[str] = ("I_first", "I_50", "I_max"),
    cutline: int = 101,
) -> pd.DataFrame:
    subset = df_cv0_responses[
        (df_cv0_responses["cutline"].astype(int) == int(cutline))
        & np.isclose(df_cv0_responses["freq"].astype(float), float(freq))
        & df_cv0_responses["thresh_type"].isin(thresh_types)
    ]
    return (
        subset.groupby(["cv", "thresh_type"], as_index=False)
        .agg(mean_spikes=("n_spikes", "mean"), median_spikes=("n_spikes", "median"))
        .sort_values(["thresh_type", "cv"])
    )


def _time_pattern_path(config: DetectionConfig, cv: float, freq: float | int) -> Path:
    return config.time_pattern_dir / f"cv{float(cv):.1f}_f{int(freq)}_waveforms.npz"


def _time_pattern_meta_path(config: DetectionConfig, cv: float, freq: float | int) -> Path:
    return config.time_pattern_dir / f"cv{float(cv):.1f}_f{int(freq)}_meta.json"


def load_input_pulse_onsets_s(
    config: DetectionConfig | None = None,
    cv: float = 0.0,
    freq: float = 100.0,
    rep_index: int = 0,
    rep_label: str | None = None,
) -> np.ndarray:
    """Load pulse onsets from a saved stage-01 waveform bundle."""
    config = (config or DetectionConfig()).resolved()
    path = _time_pattern_path(config, cv, freq)
    if not path.exists():
        return np.array([], dtype=float)
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        key = rep_label if rep_label in keys else keys[int(rep_index)]
        waveform = np.asarray(data[key], dtype=float)
    if waveform.size < 2:
        return np.array([], dtype=float)

    meta_path = _time_pattern_meta_path(config, cv, freq)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    fs = float(meta.get("fs", (waveform.size - 1) / 2.0))
    pulse_samples = max(1, int(meta.get("pulse_samples", 1)))

    mask = np.abs(waveform) > 1e-9
    segment_starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    if segment_starts.size == 0:
        return np.array([], dtype=float)

    # Biphasic pulses have two non-zero segments separated by the interphase gap.
    # Keep one onset per full pulse instead of counting both phases.
    pulse_starts = [int(segment_starts[0])]
    last_start = int(segment_starts[0])
    for idx in segment_starts[1:]:
        idx = int(idx)
        if idx - last_start >= pulse_samples:
            pulse_starts.append(idx)
            last_start = idx

    return np.asarray(pulse_starts, dtype=float) / fs


def _response_threshold_label(thresh_type: str) -> str:
    return "I_max" if str(thresh_type) == "I_max" else str(thresh_type)


def _select_cached_response_rows(
    df_responses: pd.DataFrame,
    config: DetectionConfig | None = None,
    cv: float = 0.0,
    freq: float = 100.0,
    rep_index: int = 0,
    cutline: int | None = None,
    thresh_types: Sequence[str] = THRESH_TYPES,
) -> pd.DataFrame:
    """Select and order cached response rows for one stage-03 raster case."""
    config = (config or DetectionConfig()).resolved()
    cutline = config.cutline if cutline is None else int(cutline)
    df_responses = normalize_threshold_frame(df_responses)
    subset = df_responses[
        (df_responses["cutline"].astype(int) == int(cutline))
        & np.isclose(df_responses["freq"].astype(float), float(freq))
        & np.isclose(df_responses["cv"].astype(float), float(cv))
        & (df_responses["rep_index"].astype(int) == int(rep_index))
        & df_responses["thresh_type"].isin(thresh_types)
    ].copy()
    if subset.empty:
        raise ValueError(f"No cached response rows for cutline={cutline}, freq={freq:g}, cv={cv:g}, rep_index={rep_index}.")
    subset["thresh_type"] = pd.Categorical(subset["thresh_type"], categories=list(thresh_types), ordered=True)
    return subset.sort_values("thresh_type").reset_index(drop=True)


def build_detection_response_comparison_table(
    df_detection_responses: pd.DataFrame,
    df_cv0_responses: pd.DataFrame,
    config: DetectionConfig | None = None,
    freq: float = 100.0,
    compare_cv: float = 0.7,
    rep_index: int = 5,
    cutline: int | None = None,
    thresh_types: Sequence[str] = THRESH_TYPES,
) -> pd.DataFrame:
    """Build the compact CV0 / CV example / CV0-based response table used in notebook stage 03."""
    config = (config or DetectionConfig()).resolved()
    cutline = config.cutline if cutline is None else int(cutline)
    case_specs = [
        ("CV0", df_detection_responses, 0.0),
        ("CV Example", df_detection_responses, float(compare_cv)),
        ("CV0-Based Replay", df_cv0_responses, float(compare_cv)),
    ]
    case_tables: list[pd.DataFrame] = []
    for case_label, df, cv in case_specs:
        rows = _select_cached_response_rows(
            df,
            config=config,
            cv=cv,
            freq=freq,
            rep_index=rep_index,
            cutline=cutline,
            thresh_types=thresh_types,
        )
        case_tables.append(
            pd.DataFrame(
                {
                    ("amplitude (uA)", case_label): rows["amp_uA"].astype(float).round(2).to_numpy(),
                    ("spikes", case_label): rows["n_spikes"].astype(int).to_numpy(),
                },
                index=[_response_threshold_label(value) for value in rows["thresh_type"]],
            )
        )

    table = pd.concat(case_tables, axis=1)
    case_labels = [case_label for case_label, _, _ in case_specs]
    table = table[
        [("amplitude (uA)", label) for label in case_labels]
        + [("spikes", label) for label in case_labels]
    ]
    table.index.name = "threshold"
    return table


def rerun_cv0_response_case(
    df_cv0_responses: pd.DataFrame,
    config: DetectionConfig | None = None,
    cv: float = 0.0,
    freq: float = 100.0,
    rep_index: int = 0,
    cutline: int = 101,
    thresh_types: Sequence[str] = THRESH_TYPES,
    init_v_mV: float = INIT_V_MV,
    spike_height_mV: float = SPIKE_HEIGHT_MV,
) -> pd.DataFrame:
    config = (config or DetectionConfig()).resolved()
    subset = df_cv0_responses[
        (df_cv0_responses["cutline"].astype(int) == int(cutline))
        & np.isclose(df_cv0_responses["freq"].astype(float), float(freq))
        & np.isclose(df_cv0_responses["cv"].astype(float), float(cv))
        & (df_cv0_responses["rep_index"].astype(int) == int(rep_index))
        & df_cv0_responses["thresh_type"].isin(thresh_types)
    ].copy()
    if subset.empty:
        raise ValueError(f"No saved CV0 response rows for cutline={cutline}, freq={freq}, cv={cv}, rep_index={rep_index}.")

    npz_path = config.spacetime_dir / str(subset["file"].iloc[0])
    time_ms, node_numbers, rep_labels, current, V_mV_1uA = load_spacetime_from_npz(npz_path)
    rep_label = str(rep_labels[int(rep_index)])
    time_trim_ms, V_trim = trim_tail_window_ms(time_ms, V_mV_1uA[int(rep_index)], current[int(rep_index)], tail_ms=TAIL_MS)
    time_pad_ms, V_pad = prepend_zeros_ms(time_trim_ms, V_trim, prepad_ms=PREPAD_MS)
    runner = MRGrunner(time_pad_ms, V_pad, record_node_index=RECORD_NODE_INDEX)

    records = []
    for _, row in subset.iterrows():
        spike_times_ms, n_spikes = runner.run_mrg_for_amp(
            scale_uA=float(row["amp_uA"]),
            init_v_mV=init_v_mV,
            spike_height_mV=spike_height_mV,
        )
        record = row.to_dict()
        record["rep_label"] = rep_label
        record["n_spikes"] = int(n_spikes)
        record["spike_times_ms"] = json.dumps(list(map(float, np.atleast_1d(spike_times_ms))))
        records.append(record)
    return pd.DataFrame.from_records(records)


def rerun_true_threshold_case(
    df_thresholds: pd.DataFrame,
    config: DetectionConfig | None = None,
    cv: float = 0.0,
    freq: float = 100.0,
    rep_index: int = 0,
    cutline: int = 101,
    thresh_types: Sequence[str] = THRESH_TYPES,
    init_v_mV: float = INIT_V_MV,
    spike_height_mV: float = SPIKE_HEIGHT_MV,
) -> pd.DataFrame:
    config = (config or DetectionConfig()).resolved()
    threshold_row = select_detection_row(df_thresholds, cv=cv, freq=freq, rep_index=rep_index, cutline=cutline)
    npz_path = config.spacetime_dir / str(threshold_row["file"])
    time_ms, node_numbers, rep_labels, current, V_mV_1uA = load_spacetime_from_npz(npz_path)
    rep_label = str(rep_labels[int(rep_index)])
    time_trim_ms, V_trim = trim_tail_window_ms(time_ms, V_mV_1uA[int(rep_index)], current[int(rep_index)], tail_ms=TAIL_MS)
    time_pad_ms, V_pad = prepend_zeros_ms(time_trim_ms, V_trim, prepad_ms=PREPAD_MS)
    runner = MRGrunner(time_pad_ms, V_pad, record_node_index=RECORD_NODE_INDEX)

    records = []
    for thresh_type in thresh_types:
        amp = threshold_row.get(f"{thresh_type}_uA")
        if amp is None or pd.isna(amp):
            continue
        spike_times_ms, n_spikes = runner.run_mrg_for_amp(
            scale_uA=float(amp),
            init_v_mV=init_v_mV,
            spike_height_mV=spike_height_mV,
        )
        records.append(
            {
                "file": str(threshold_row["file"]),
                "cutline": int(threshold_row["cutline"]),
                "cv": float(threshold_row["cv"]),
                "freq": float(threshold_row["freq"]),
                "rep_index": int(rep_index),
                "rep_label": rep_label,
                "thresh_type": str(thresh_type),
                "amp_uA": float(amp),
                "n_spikes": int(n_spikes),
                "spike_times_ms": json.dumps(list(map(float, np.atleast_1d(spike_times_ms)))),
            }
        )
    return pd.DataFrame.from_records(records, columns=list(RESPONSE_COLUMNS))
