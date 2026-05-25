"""Canonical release pipeline and notebook helpers for periodicity discrimination.

This module intentionally contains the release-facing discrimination logic in one
place: feature engineering helpers, model fitting, archived data readers, and
manuscript summary tables. Plotting utilities live in `discrimination_plot.py`.
Command-line orchestration lives in
`discrimination_CLI.py`; older exploratory entrypoints from the original
repository were pruned from the public release surface.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Default Slack freqs to align sim fit/eval scope.
DEFAULT_FREQS = [20.0, 50.0, 100.0, 200.0]
COMBINED_MODEL_BANKS = {"rate_combined", "timing_combined", "hybrid_combined"}
HYBRID_FAMILY_MODEL_BANKS = {"hybrid_combined"}
THRESH_ORDER = (
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


def canonical_threshold_type(thresh_type: object) -> str:
    text = str(thresh_type)
    return LEGACY_THRESHOLD_ALIASES.get(text, text)


def normalize_threshold_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "thresh_type" in out.columns:
        out["thresh_type"] = out["thresh_type"].map(canonical_threshold_type)
    return out

FEATURE_METRIC_CACHE_VERSION = 1
FEATURE_METRIC_CACHE_FILENAME = "feature_metric_summaries.csv"

# Optional SciPy for optimization
try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None


# ============================
# Slack parsing (reused from v1)
# ============================

def parse_rat_name_from_filename(path: str) -> str:
    """Slack files: '23-Dec-2022_Olaf.csv' or '25-Aug-2022_Mars_a.csv'."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    return parts[1]


def infer_block_from_file(df_file: pd.DataFrame) -> Optional[str]:
    """Infer std0 vs std1 for a Slack behavioral file.

    We look at mid CV trials (0<CV<1) and see whether Type=1 or Type=2 dominates.
      - if Type=1 dominates: mid CVs correspond to Type=1 => std0
      - if Type=2 dominates: mid CVs correspond to Type=2 => std1
    """
    if "CV" not in df_file.columns or "Type" not in df_file.columns:
        return None
    mid = df_file[(df_file["CV"] > 0.0) & (df_file["CV"] < 1.0)].copy()
    if mid.empty:
        return None
    vc = mid["Type"].value_counts()
    if vc.empty:
        return None
    top = int(vc.idxmax())
    return "std0" if top == 1 else "std1"


def load_slack_dir(
    slack_dir: str,
    *,
    include_templates: bool = True,
    keep_only_varying_type: bool = True,
    aggregate_sessions_by_rat: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all Slack behavioral files in a directory.

    Returns
    -------
    slack_perrat : per-rat psychometrics
      columns: rat_name,freq,block,cv,k,n,p_correct
    slack_mean : mean psychometrics across rats
      columns: freq,block,cv,k,n,p_mean,p_sem,n_trials

    Parameters
    ----------
    include_templates:
      If True, keep CV==0 and CV==1 template trials in slack_perrat.
      You can later exclude them from evaluation if you want.
    keep_only_varying_type:
      Matches older convention:
        - std0 keeps Type==1 (varying = aperiodic)
        - std1 keeps Type==2 (varying = periodic)
      If False, keeps both.
    aggregate_sessions_by_rat:
      If True (default), first sum k/n across all files belonging to the same
      rat at each (freq, block, cv), then compute p_correct and rat-level means.
      If False, each file contributes separately (session-weighted behavior).
    """
    slack_dir = str(slack_dir)
    files = [os.path.join(slack_dir, f) for f in os.listdir(slack_dir) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No Slack csv files found in slack_dir: {slack_dir}")

    rows = []
    for fp in sorted(files):
        df = pd.read_csv(fp, low_memory=False)
        rat = parse_rat_name_from_filename(fp)

        # Filter to randomized, non-forced trials with valid responses
        if "Randomized" in df.columns:
            df = df[df["Randomized"] == 1]
        if "Forced" in df.columns:
            df = df[df["Forced"] == 0]
        if "Response" in df.columns:
            df = df[df["Response"].isin([1, 2])]

        block = infer_block_from_file(df)
        if block is None:
            continue

        # Standardize columns
        if "CV" in df.columns:
            df["cv"] = pd.to_numeric(df["CV"], errors="coerce")
        elif "cv" in df.columns:
            df["cv"] = pd.to_numeric(df["cv"], errors="coerce")
        else:
            continue

        if "Freq" in df.columns:
            df["freq"] = pd.to_numeric(df["Freq"], errors="coerce")
        elif "Frequency (Hz)" in df.columns:
            df["freq"] = pd.to_numeric(df["Frequency (Hz)"], errors="coerce")
        elif "freq" in df.columns:
            df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
        else:
            continue

        if "Type" in df.columns:
            df["Type"] = pd.to_numeric(df["Type"], errors="coerce")
        else:
            continue

        if "Response" not in df.columns:
            continue
        df["Response"] = pd.to_numeric(df["Response"], errors="coerce")

        df = df.dropna(subset=["cv", "freq", "Type", "Response"])
        if df.empty:
            continue

        # Optional filtering to match the displayed psychometrics
        if not include_templates:
            if block == "std0":
                df = df[df["cv"] > 0.0]
            else:
                df = df[df["cv"] < 1.0]

        if keep_only_varying_type:
            if block == "std0":
                is_template = np.isclose(df["cv"], 0.0)
                df = df[is_template | (df["Type"].astype(int) == 1)]
            else:
                is_template = np.isclose(df["cv"], 1.0)
                df = df[is_template | (df["Type"].astype(int) == 2)]

        if df.empty:
            continue

        # Slack CVs are typically increments of 0.1
        df["cv"] = np.round(df["cv"].to_numpy(float), 1)

        # correctness: Response == Type
        is_correct = (df["Response"].astype(int) == df["Type"].astype(int)).astype(int)

        for (freq, cv), g in df.groupby(["freq", "cv"]):
            n = int(len(g))
            k = int(is_correct.loc[g.index].sum())
            rows.append(
                dict(rat_name=str(rat), freq=float(freq), block=str(block), cv=float(cv), k=k, n=n)
            )

    if not rows:
        raise RuntimeError(f"No usable Slack trials in slack_dir={slack_dir}")

    slack_trials = pd.DataFrame(rows)
    slack_trials["p_correct"] = slack_trials["k"] / slack_trials["n"].clip(lower=1)

    if aggregate_sessions_by_rat:
        slack_perrat = (
            slack_trials.groupby(["rat_name", "freq", "block", "cv"], as_index=False)
            .agg(k=("k", "sum"), n=("n", "sum"))
        )
        slack_perrat["p_correct"] = slack_perrat["k"] / slack_perrat["n"].clip(lower=1)
    else:
        slack_perrat = slack_trials.copy()

    # Mean across rats
    def _sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return 0.0
        return float(np.std(x, ddof=1) / math.sqrt(x.size))

    agg = (
        slack_perrat.groupby(["freq", "block", "cv"], as_index=False)
        .agg(k=("k", "sum"), n=("n", "sum"), p_mean=("p_correct", "mean"), p_sem=("p_correct", _sem), n_trials=("p_correct", "size"))
    )

    return slack_perrat, agg


# ============================
# Simulation IO helpers
# ============================

def _pick_first_present(cols: Sequence[str], df: pd.DataFrame) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def infer_sim_columns(sim: pd.DataFrame) -> Tuple[str, str]:
    """Return (spikes_col, rep_col)."""
    spikes_col = _pick_first_present(
        ["spikes_ms", "spike_times_ms", "spike_times", "spikes"], sim
    )
    rep_col = _pick_first_present(
        ["rep", "rep_id", "rep_index", "rep_idx", "rep_label", "replicate", "trial"], sim
    )
    if spikes_col is None:
        raise ValueError("Simulation dataframe missing spike-train column")
    if rep_col is None:
        raise ValueError("Simulation dataframe missing rep column")
    return spikes_col, rep_col


def parse_spike_list(x) -> np.ndarray:
    """Parse spikes stored as list/np array/space-or-comma string."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return np.array([], dtype=float)
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr[np.isfinite(arr)]
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "[]"}:
        return np.array([], dtype=float)
    # strip brackets
    s = s.strip("[]()")
    # split
    parts = [p for p in s.replace(",", " ").split() if p]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    arr = np.asarray(out, dtype=float)
    return arr[np.isfinite(arr)]


# ============================
# Event detection + features
# ============================

EPS = 1e-9


@dataclass(frozen=True)
class FeatureSettings:
    t0_ms: float = 0.0
    t1_ms: float = 2000.0
    bin_ms: float = 10.0
    min_ct: int = 1
    cap1: bool = True
    gap_close_ms: float = 0.0
    min_event_ms: float = 0.0
    mrg_init_ms: float = 100.0  # subtract from stored spike times


@dataclass(frozen=True)
class ModelSettings:
    jnd_k: float = 2.5
    entmax_alpha: float = 1.1
    scope: str = "per_freq_block"  # per_block|per_freq_block
    train_frac: float = 0.9
    n_mc: int = 25
    seed: int = 0
    do_loto: bool = False
    loto_include_full: bool = True
    all_th_norm: str = "endpoint_pm1"  # currently only endpoint_pm1 used for fitting


class _Event:
    __slots__ = ("start_ms", "end_ms")

    def __init__(self, s: float, e: float):
        self.start_ms = float(s)
        self.end_ms = float(e)


def _lv(seq_ms: np.ndarray) -> float:
    seq = np.asarray(seq_ms, dtype=float)
    seq = seq[np.isfinite(seq)]
    if seq.size < 2:
        return float("nan")
    x = seq[:-1]
    y = seq[1:]
    denom = x + y
    m = np.isfinite(denom) & (denom > 0)
    if not np.any(m):
        return float("nan")
    z = ((x[m] - y[m]) / denom[m]) ** 2
    if z.size == 0 or not np.any(np.isfinite(z)):
        return float("nan")
    return float(3.0 * np.nanmean(z))


def detect_events_binned(spikes_ms: np.ndarray, fs: FeatureSettings) -> Tuple[np.ndarray, np.ndarray, List[_Event], np.ndarray]:
    """Bin-based event detection.

    Returns
    -------
    edges : bin edges
    active : 0/1 bins
    events : list of bin-defined contiguous active segments
    spikes_f : filtered spikes in [t0,t1]
    """
    spikes_ms = np.asarray(spikes_ms, dtype=float)
    spikes_ms = spikes_ms[np.isfinite(spikes_ms)]
    spikes_ms = spikes_ms - float(fs.mrg_init_ms)
    spikes_ms = spikes_ms[(spikes_ms >= fs.t0_ms) & (spikes_ms <= fs.t1_ms)]

    n_bins = int(np.ceil((fs.t1_ms - fs.t0_ms) / fs.bin_ms))
    edges = fs.t0_ms + np.arange(n_bins + 1, dtype=float) * fs.bin_ms

    counts, _ = np.histogram(spikes_ms, bins=edges)
    if fs.cap1:
        counts = np.minimum(counts, 1)
    active = (counts >= int(fs.min_ct)).astype(np.int8)

    # close small gaps
    if fs.gap_close_ms > 0:
        gap_bins = int(np.round(fs.gap_close_ms / fs.bin_ms))
        if gap_bins > 0:
            a = active.copy()
            i = 0
            while i < a.size:
                if a[i] == 0:
                    j = i
                    while j < a.size and a[j] == 0:
                        j += 1
                    left_one = (i - 1 >= 0 and a[i - 1] == 1)
                    right_one = (j < a.size and a[j] == 1)
                    if left_one and right_one and (j - i) <= gap_bins:
                        a[i:j] = 1
                    i = j
                else:
                    i += 1
            active = a

    events: List[_Event] = []
    i = 0
    while i < active.size:
        if active[i] == 1:
            j = i + 1
            while j < active.size and active[j] == 1:
                j += 1
            s = float(edges[i])
            e = float(edges[j])
            if (e - s) >= float(fs.min_event_ms):
                events.append(_Event(s, e))
            i = j
        else:
            i += 1

    return edges, active, events, spikes_ms


def refine_events_to_spikes(events: List[_Event], spikes_ms: np.ndarray) -> List[_Event]:
    """Refine bin-events to first/last spike times within each event."""
    spikes_ms = np.asarray(spikes_ms, dtype=float)
    spikes_ms = spikes_ms[np.isfinite(spikes_ms)]
    out: List[_Event] = []
    for ev in events:
        m = (spikes_ms >= ev.start_ms) & (spikes_ms <= ev.end_ms)
        if not np.any(m):
            continue
        s = float(np.min(spikes_ms[m]))
        e = float(np.max(spikes_ms[m]))
        out.append(_Event(s, e))
    out.sort(key=lambda z: z.start_ms)
    return out


def compute_trial_features(spikes_ms: np.ndarray, fs: FeatureSettings) -> Dict[str, float]:
    """Compute rate + timing features for a single trial."""
    _edges, _active, events_bin, spikes_f = detect_events_binned(spikes_ms, fs)
    events = refine_events_to_spikes(events_bin, spikes_f)

    n_spikes = int(spikes_f.size)
    n_events = int(len(events))
    spikes_per_event = (n_spikes / n_events) if n_events > 0 else float("nan")

    silence_lv = float("nan")
    if n_events > 0:
        sil = []
        sil.append(max(0.0, float(events[0].start_ms) - fs.t0_ms))
        for i in range(n_events - 1):
            sil.append(max(0.0, float(events[i + 1].start_ms) - float(events[i].end_ms)))
        sil.append(max(0.0, fs.t1_ms - float(events[-1].end_ms)))
        sil = np.asarray(sil, dtype=float)
        # quantize silences to bin_ms (so LV respects the bin definition)
        q = float(fs.bin_ms)
        sil_q = np.maximum(np.round(sil / q) * q, 0.0)
        silence_lv = _lv(sil_q)

    return {
        "n_spikes": float(n_spikes),
        "n_events": float(n_events),
        "spikes_per_event": float(spikes_per_event),
        "silence_lv": float(silence_lv),
    }


# ============================
# Evidence
# ============================

def _within_jnd(x: float, t: float, k: float) -> bool:
    if not (np.isfinite(x) and np.isfinite(t)):
        return False
    x = float(x)
    t = float(t)
    if x <= 0 or t <= 0:
        return x == t
    r = max(x, t) / (min(x, t) + EPS)
    return r <= float(k)


def rate_vote_hardjnd(x: Dict[str, float], t0: Dict[str, float], t1: Dict[str, float], k: float, keys: Sequence[str]) -> float:
    """Rate evidence as mean over NONZERO votes in {-1,0,+1}."""
    votes: List[float] = []
    for key in keys:
        xv = x.get(key, float("nan"))
        a = t0.get(key, float("nan"))
        b = t1.get(key, float("nan"))
        if not (np.isfinite(xv) and np.isfinite(a) and np.isfinite(b)):
            continue
        w0 = _within_jnd(xv, a, k)
        w1 = _within_jnd(xv, b, k)
        if w0 and (not w1):
            votes.append(-1.0)
        elif w1 and (not w0):
            votes.append(+1.0)
        else:
            votes.append(0.0)
    nz = [v for v in votes if abs(v) > 1e-12]
    if not nz:
        return 0.0
    return float(np.mean(nz))


def timing_evidence_lv(x_lv: float, t0_lv: float, t1_lv: float) -> float:
    if not (np.isfinite(x_lv) and np.isfinite(t0_lv) and np.isfinite(t1_lv)):
        return 0.0
    d0 = abs(float(x_lv) - float(t0_lv))
    d1 = abs(float(x_lv) - float(t1_lv))
    return float(d0 - d1)  # + => closer to cv1 template


def endpoint_pm1_normalize(y: np.ndarray, y0: float, y1: float) -> np.ndarray:
    """Affine map so that y0 -> -1 and y1 -> +1.

    If y0==y1 or invalid, returns zeros.
    """
    y = np.asarray(y, dtype=float)
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return np.zeros_like(y)
    den = float(y1 - y0)
    if not np.isfinite(den) or abs(den) < 1e-12:
        return np.zeros_like(y)
    # Solve a*y0 + b = -1; a*y1 + b = +1
    a = 2.0 / den
    b = -1.0 - a * float(y0)
    return a * y + b


def endpoint_pm1_trace(values_by_cv: Dict[float, float], lo: float, hi: float) -> Dict[float, float]:
    """Endpoint pm1 transform for a single trace keyed by CV.

    pm1(x; lo, hi) = 2*(x - lo)/(hi - lo) - 1
    If hi ~ lo or endpoints invalid, return zeros for all keys.
    """
    out: Dict[float, float] = {}
    den = float(hi) - float(lo)
    if not (np.isfinite(lo) and np.isfinite(hi)) or abs(den) < 1e-12:
        for cv in values_by_cv.keys():
            out[float(cv)] = 0.0
        return out
    a = 2.0 / den
    b = -1.0 - a * float(lo)
    for cv, x in values_by_cv.items():
        xv = float(x)
        out[float(cv)] = float(a * xv + b) if np.isfinite(xv) else 0.0
    return out


# ============================
# Entmax (numpy) for mixture weights
# ============================

def _entmax_bisect(logits: np.ndarray, alpha: float, n_iter: int = 50) -> np.ndarray:
    """Entmax via bisection (supports 1 < alpha <= 2).

    This is a lightweight numpy implementation inspired by the reference
    implementations (Peters et al.). For alpha->1 it approaches softmax; for
    alpha=2 it becomes sparsemax.
    """
    z = np.asarray(logits, dtype=float)
    if z.ndim != 1:
        raise ValueError("entmax expects 1D logits")
    if z.size == 0:
        return z
    if alpha <= 1.0 + 1e-9:
        # softmax fallback
        z0 = z - np.max(z)
        e = np.exp(z0)
        return e / (e.sum() + EPS)
    if alpha > 2.0:
        raise ValueError("entmax alpha must be <= 2")

    # shift by max for stability
    z = z - np.max(z)
    # power
    pow_ = 1.0 / (alpha - 1.0)

    # Find tau such that sum(p) = 1, where p = [((alpha-1)*(z - tau))_+]^pow_
    # Bracket tau
    tau_lo = np.min(z) - 1.0
    tau_hi = np.max(z)

    def _p(tau: float) -> np.ndarray:
        u = (alpha - 1.0) * (z - tau)
        u = np.maximum(u, 0.0)
        return u ** pow_

    for _ in range(n_iter):
        tau = 0.5 * (tau_lo + tau_hi)
        p = _p(tau)
        s = float(np.sum(p))
        if s > 1.0:
            tau_lo = tau
        else:
            tau_hi = tau

    p = _p(tau_hi)
    s = float(np.sum(p))
    if s <= 0:
        # degenerate
        out = np.zeros_like(p)
        out[np.argmax(z)] = 1.0
        return out
    return p / s


# ============================
# Evidence curves from simulation
# ============================

def build_sim_index(sim: pd.DataFrame) -> Dict[Tuple[float, str, float, int], int]:
    """(freq, thresh, cv, rep) -> row idx."""
    idx: Dict[Tuple[float, str, float, int], int] = {}
    sim_r = sim.reset_index(drop=True)
    for i, r in sim_r.iterrows():
        key = (float(r["freq"]), str(r["thresh_type"]), float(r["cv"]), int(r["rep"]))
        if key not in idx:
            idx[key] = i
    return idx


def get_spikes(sim_r: pd.DataFrame, sim_idx: Dict[Tuple[float, str, float, int], int], spikes_col: str, *, freq: float, th: str, cv: float, rep: int) -> np.ndarray:
    k = (float(freq), str(th), float(cv), int(rep))
    row_i = sim_idx.get(k, None)
    if row_i is None:
        return np.array([], dtype=float)
    return parse_spike_list(sim_r.iloc[row_i][spikes_col])


def estimate_evidence_curves(
    sim: pd.DataFrame,
    *,
    freq: float,
    thresh_types: Sequence[str],
    cvs: Sequence[float],
    reps: Sequence[int],
    fs: FeatureSettings,
    ms: ModelSettings,
    bin_rate_mode: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Estimate mean evidence curves across provided reps.

    Returns
    -------
    mean_e_rate_raw[th] : (len(cvs),)
    mean_e_time_raw[th] : (len(cvs),)
    rate_reliability[th] : (len(cvs),) fraction of nonzero rate evidence among reps
    """
    sim = sim.copy()
    spikes_col, rep_col = infer_sim_columns(sim)
    sim = sim.rename(columns={rep_col: "rep"})
    sim["freq"] = pd.to_numeric(sim["freq"], errors="coerce")
    sim["cv"] = pd.to_numeric(sim["cv"], errors="coerce")
    sim["thresh_type"] = sim["thresh_type"].map(canonical_threshold_type)
    sim["rep"] = pd.to_numeric(sim["rep"], errors="coerce").astype(int)
    sim = sim[np.isfinite(sim["freq"]) & np.isfinite(sim["cv"])].copy()

    sim_r = sim.reset_index(drop=True)
    sim_idx = build_sim_index(sim_r)

    # cache trial features (freq,th,cv,rep) -> dict
    feat_cache: Dict[Tuple[float, str, float, int], Dict[str, float]] = {}

    def _feat(freq_: float, th_: str, cv_: float, rep_: int) -> Dict[str, float]:
        key = (float(freq_), str(th_), float(cv_), int(rep_))
        if key in feat_cache:
            return feat_cache[key]
        spikes = get_spikes(sim_r, sim_idx, spikes_col, freq=freq_, th=th_, cv=cv_, rep=rep_)
        f = compute_trial_features(spikes, fs)
        feat_cache[key] = f
        return f

    # Decide which rate keys to use
    if bin_rate_mode == "bin1":
        rate_keys = ("n_spikes",)
    elif bin_rate_mode == "bin10":
        rate_keys = ("n_events", "spikes_per_event")
    else:
        # fallback: treat 1ms as spikes-only, else events+spk/ev
        rate_keys = ("n_spikes",) if fs.bin_ms <= 1.0 + 1e-9 else ("n_events", "spikes_per_event")

    mean_e_rate_raw: Dict[str, np.ndarray] = {}
    mean_e_time_raw: Dict[str, np.ndarray] = {}
    rate_rel: Dict[str, np.ndarray] = {}

    for th in thresh_types:
        er = np.full(len(cvs), np.nan, dtype=float)
        et = np.full(len(cvs), np.nan, dtype=float)
        rr = np.full(len(cvs), np.nan, dtype=float)

        # Need template features per rep
        for i_cv, cv in enumerate(cvs):
            e_rate_reps: List[float] = []
            e_time_reps: List[float] = []
            nz_rate = 0
            n_rate = 0
            for rep in reps:
                # Require cv0 & cv1 templates for this rep
                f0 = _feat(freq, th, 0.0, rep)
                f1 = _feat(freq, th, 1.0, rep)
                fx = _feat(freq, th, cv, rep)

                x_rate = {k: fx.get(k, np.nan) for k in ("n_spikes", "n_events", "spikes_per_event")}
                t0_rate = {k: f0.get(k, np.nan) for k in ("n_spikes", "n_events", "spikes_per_event")}
                t1_rate = {k: f1.get(k, np.nan) for k in ("n_spikes", "n_events", "spikes_per_event")}

                e_r = rate_vote_hardjnd(x_rate, t0_rate, t1_rate, k=ms.jnd_k, keys=rate_keys)
                e_t = timing_evidence_lv(fx.get("silence_lv", np.nan), f0.get("silence_lv", np.nan), f1.get("silence_lv", np.nan))

                if np.isfinite(e_r):
                    e_rate_reps.append(float(e_r))
                    n_rate += 1
                    if abs(e_r) > 1e-12:
                        nz_rate += 1
                if np.isfinite(e_t):
                    e_time_reps.append(float(e_t))

            if e_rate_reps:
                er[i_cv] = float(np.mean(e_rate_reps))
                rr[i_cv] = float(nz_rate / max(1, n_rate))
            if e_time_reps:
                et[i_cv] = float(np.mean(e_time_reps))

        mean_e_rate_raw[th] = er
        mean_e_time_raw[th] = et
        rate_rel[th] = rr

    return mean_e_rate_raw, mean_e_time_raw, rate_rel


def normalize_curves_endpoint_pm1(
    mean_e_raw: Dict[str, np.ndarray],
    cvs: Sequence[float],
) -> Dict[str, np.ndarray]:
    """Apply endpoint_pm1 per threshold type using cv=0 and cv=1 entries."""
    cvs = np.asarray(cvs, dtype=float)
    i0 = int(np.where(np.isclose(cvs, 0.0))[0][0]) if np.any(np.isclose(cvs, 0.0)) else None
    i1 = int(np.where(np.isclose(cvs, 1.0))[0][0]) if np.any(np.isclose(cvs, 1.0)) else None
    out: Dict[str, np.ndarray] = {}
    for th, y in mean_e_raw.items():
        y = np.asarray(y, dtype=float)
        if i0 is None or i1 is None:
            out[th] = np.zeros_like(y)
            continue
        y0 = float(y[i0])
        y1 = float(y[i1])
        out[th] = endpoint_pm1_normalize(y, y0=y0, y1=y1)
    return out


# ============================
# Notebook-facing feature workflow
# ============================

@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Settings used by the stage-04 notebook walkthrough."""

    sim_csv: Path = Path(__file__).resolve().parents[2] / "data" / "detection" / "generated" / "cv0_based_responses.csv"
    freqs: Tuple[float, ...] = tuple(float(f) for f in DEFAULT_FREQS)
    t0_ms: float = 0.0
    t1_ms: float = 2000.0
    mrg_init_ms: float = 100.0
    min_ct: int = 1
    cap1: bool = True
    gap_close_ms: float = 0.0
    min_event_ms: float = 0.0
    jnd_k: float = 2.5
    bins: Tuple[float, float] = (1.0, 10.0)
    cache_dir: Path = Path(__file__).resolve().parents[2] / "notebooks" / "cache"


@dataclass(frozen=True)
class FeatureSimulation:
    """Normalized stage-03 response table plus the source spike column name."""

    data: pd.DataFrame
    spikes_col: str
    rep_col: str = "rep"


@dataclass(frozen=True)
class FeatureWorkflow:
    """Prepared indexes for interactive feature/evidence inspection."""

    simulation: FeatureSimulation
    config: FeatureEngineeringConfig
    sim_index: Dict[Tuple[float, str, float, int], int]
    cvs_by_freq_thresh: Dict[Tuple[float, str], Tuple[float, ...]]
    reps_by_freq_thresh: Dict[Tuple[float, str], Tuple[int, ...]]
    feature_cache: Dict[Tuple[float, str, float, int, float], Dict[str, object]] = dataclasses.field(default_factory=dict)


def threshold_display_name(thresh_type: str) -> str:
    """Manuscript-facing threshold label while preserving CSV names."""

    labels = {
        "I_first": "I_first",
        "I_volley5": "I_volley5",
        "I_volley10": "I_volley10",
        "I_max": "I_max",
    }
    return labels.get(str(thresh_type), str(thresh_type))


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

FEATURE_RATE_COLORS = {
    "I_first": "#FC9CA2",
    "I_volley5": "#FB747D",
    "I_volley10": "#FA4C58",
    "I_5": "#F92432",
    "I_10": "#E30613",
    "I_15": "#C70512",
    "I_25": "#9F040E",
    "I_50": "#77030B",
    "I_max": "#500207",
}

FEATURE_TIMING_COLORS = {
    "I_first": "#00D4FF",
    "I_volley5": "#01BCEA",
    "I_volley10": "#02A5D5",
    "I_5": "#038DC0",
    "I_10": "#0575AB",
    "I_15": "#065D95",
    "I_25": "#074680",
    "I_50": "#082E6B",
    "I_max": "#091656",
}


def threshold_short_label(thresh_type: str) -> str:
    return THRESH_SHORT_LABELS.get(str(thresh_type), str(thresh_type))


def _normalize_feature_simulation_frame(sim: pd.DataFrame, config: Optional[FeatureEngineeringConfig] = None) -> FeatureSimulation:
    sim = sim.copy()
    spikes_col, rep_col = infer_sim_columns(sim)
    if rep_col != "rep":
        sim = sim.rename(columns={rep_col: "rep"})
    rep_col = "rep"

    required = {"freq", "cv", "thresh_type", rep_col, spikes_col}
    missing = sorted(required - set(sim.columns))
    if missing:
        raise ValueError(f"Simulation dataframe missing required columns: {missing}")

    sim["freq"] = pd.to_numeric(sim["freq"], errors="coerce")
    sim["cv"] = pd.to_numeric(sim["cv"], errors="coerce")
    sim[rep_col] = pd.to_numeric(sim[rep_col], errors="coerce")
    sim["thresh_type"] = sim["thresh_type"].map(canonical_threshold_type)
    if "n_spikes" in sim.columns:
        sim["n_spikes"] = pd.to_numeric(sim["n_spikes"], errors="coerce")
    else:
        sim["n_spikes"] = sim[spikes_col].map(lambda x: int(parse_spike_list(x).size))

    sim = sim[np.isfinite(sim["freq"]) & np.isfinite(sim["cv"]) & np.isfinite(sim[rep_col])].copy()
    if config is not None and config.freqs:
        freq_set = {float(f) for f in config.freqs}
        sim = sim[sim["freq"].isin(freq_set)].copy()
    sim[rep_col] = sim[rep_col].astype(int)
    sim = sim.reset_index(drop=True)
    if sim.empty:
        raise RuntimeError("No usable stage-03 response rows after normalization.")
    return FeatureSimulation(data=sim, spikes_col=spikes_col, rep_col=rep_col)


def _ensure_feature_simulation(
    simulation: Union[FeatureSimulation, pd.DataFrame],
    config: Optional[FeatureEngineeringConfig] = None,
) -> FeatureSimulation:
    if isinstance(simulation, FeatureSimulation):
        return simulation
    return _normalize_feature_simulation_frame(simulation, config=config)


def load_feature_simulation(config: Optional[FeatureEngineeringConfig] = None) -> FeatureSimulation:
    """Load and normalize the stage-03 response table used by stage 04."""

    cfg = config or FeatureEngineeringConfig()
    sim_csv = Path(cfg.sim_csv)
    if not sim_csv.exists():
        raise FileNotFoundError(f"Stage-03 response CSV not found: {sim_csv}")
    return _normalize_feature_simulation_frame(pd.read_csv(sim_csv), config=cfg)


def spike_count_summary(
    simulation: Union[FeatureSimulation, pd.DataFrame],
    freqs: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Summarize mean spike counts by frequency, CV, and threshold type."""

    feature_sim = _ensure_feature_simulation(simulation)
    df = feature_sim.data.copy()
    if freqs is not None:
        freq_set = {float(f) for f in freqs}
        df = df[df["freq"].isin(freq_set)].copy()
    df = df[np.isfinite(df["freq"]) & np.isfinite(df["cv"]) & np.isfinite(df["n_spikes"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["freq", "cv", "thresh_type", "n_mean", "n_sd", "n"])
    return (
        df.groupby(["freq", "cv", "thresh_type"], as_index=False)
        .agg(n_mean=("n_spikes", "mean"), n_sd=("n_spikes", "std"), n=("n_spikes", "size"))
        .sort_values(["freq", "cv", "thresh_type"])
        .reset_index(drop=True)
    )


def spike_count_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the notebook sanity-check pivot table for mean spike counts."""

    if summary.empty:
        return pd.DataFrame()
    pivot = summary.pivot_table(
        index=["freq", "cv"],
        columns="thresh_type",
        values="n_mean",
        aggfunc="first",
    ).sort_index()
    ordered_cols = [col for col in THRESH_ORDER if col in pivot.columns]
    extra_cols = [col for col in pivot.columns if col not in ordered_cols]
    return pivot.loc[:, ordered_cols + extra_cols]


FEATURE_TABLES = {
    "n_spikes": {"bin_ms": 1.0, "feature_key": "n_spikes", "source": "simulation"},
    "n_bursts": {"bin_ms": 10.0, "feature_key": "n_events", "source": "features"},
    "spikes_per_burst": {"bin_ms": 10.0, "feature_key": "spikes_per_event", "source": "features"},
    "LV_ISI": {"bin_ms": 1.0, "feature_key": "silence_lv", "source": "features"},
    "LV_IBI": {"bin_ms": 10.0, "feature_key": "silence_lv", "source": "features"},
}


def feature_metric_summary(
    workflow_or_simulation: Union[FeatureWorkflow, FeatureSimulation, pd.DataFrame],
    metric: str,
    config: Optional[FeatureEngineeringConfig] = None,
    freqs: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Summarize one stage-04 feature metric by frequency, CV, and threshold type."""

    metric = str(metric)
    if metric not in FEATURE_TABLES:
        raise ValueError(f"Unknown feature metric {metric!r}. Options are: {', '.join(FEATURE_TABLES)}")
    workflow = (
        workflow_or_simulation
        if isinstance(workflow_or_simulation, FeatureWorkflow)
        else prepare_feature_workflow(workflow_or_simulation, config=config)
    )
    spec = FEATURE_TABLES[metric]
    df = workflow.simulation.data.copy()
    if freqs is not None:
        freq_set = {float(freq) for freq in freqs}
        df = df[df["freq"].astype(float).isin(freq_set)].copy()
    df = df[np.isfinite(df["freq"]) & np.isfinite(df["cv"]) & df["rep"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["freq", "cv", "thresh_type", "mean", "sd", "n"])

    if spec["source"] == "simulation":
        if "n_spikes" not in df.columns:
            return pd.DataFrame(columns=["freq", "cv", "thresh_type", "mean", "sd", "n"])
        values = df[["freq", "cv", "thresh_type", "n_spikes"]].copy()
        values["value"] = pd.to_numeric(values["n_spikes"], errors="coerce")
    else:
        rows = []
        lookup = df[["freq", "cv", "thresh_type", "rep"]].drop_duplicates()
        for row in lookup.itertuples(index=False):
            details = trial_feature_details(
                workflow,
                freq=float(row.freq),
                thresh_type=str(row.thresh_type),
                cv=float(row.cv),
                rep=int(row.rep),
                bin_ms=float(spec["bin_ms"]),
            )
            rows.append(
                {
                    "freq": float(row.freq),
                    "cv": float(row.cv),
                    "thresh_type": str(row.thresh_type),
                    "value": float(details.get(str(spec["feature_key"]), np.nan)),
                }
            )
        values = pd.DataFrame.from_records(rows)

    values = values[np.isfinite(values["value"])].copy()
    if values.empty:
        return pd.DataFrame(columns=["freq", "cv", "thresh_type", "mean", "sd", "n"])
    return (
        values.groupby(["freq", "cv", "thresh_type"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n=("value", "size"))
        .sort_values(["freq", "cv", "thresh_type"])
        .reset_index(drop=True)
    )


def feature_metric_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    """Return an ordered frequency/CV x threshold table for one feature metric summary."""

    if summary.empty:
        return pd.DataFrame()
    pivot = summary.pivot_table(
        index=["freq", "cv"],
        columns="thresh_type",
        values="mean",
        aggfunc="first",
    ).sort_index()
    ordered_cols = [col for col in THRESH_ORDER if col in pivot.columns]
    extra_cols = [col for col in pivot.columns if col not in ordered_cols]
    return pivot.loc[:, ordered_cols + extra_cols]


def _feature_metric_cache_path(
    config: Optional[FeatureEngineeringConfig] = None,
    cache_csv: Optional[Union[str, Path]] = None,
) -> Path:
    if cache_csv is not None:
        return Path(cache_csv)
    cfg = config or FeatureEngineeringConfig()
    return Path(cfg.cache_dir) / FEATURE_METRIC_CACHE_FILENAME


def _empty_feature_metric_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=["freq", "cv", "thresh_type", "mean", "sd", "n"])


def _filter_feature_metric_cache(
    cache: pd.DataFrame,
    metric: str,
    freqs: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    required = {"cache_version", "metric", "freq", "cv", "thresh_type", "mean", "sd", "n"}
    if cache.empty or not required.issubset(cache.columns):
        return _empty_feature_metric_summary()

    version = pd.to_numeric(cache["cache_version"], errors="coerce")
    mask = version.eq(FEATURE_METRIC_CACHE_VERSION) & cache["metric"].astype(str).eq(str(metric))
    if freqs is not None:
        freq_set = {float(freq) for freq in freqs}
        mask &= pd.to_numeric(cache["freq"], errors="coerce").isin(freq_set)

    out = cache.loc[mask, ["freq", "cv", "thresh_type", "mean", "sd", "n"]].copy()
    if out.empty:
        return _empty_feature_metric_summary()
    out["freq"] = pd.to_numeric(out["freq"], errors="coerce")
    out["cv"] = pd.to_numeric(out["cv"], errors="coerce")
    out["mean"] = pd.to_numeric(out["mean"], errors="coerce")
    out["sd"] = pd.to_numeric(out["sd"], errors="coerce")
    out["n"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")
    out = out[np.isfinite(out["freq"]) & np.isfinite(out["cv"]) & np.isfinite(out["mean"])].copy()
    return out.sort_values(["freq", "cv", "thresh_type"]).reset_index(drop=True)


def build_feature_metric_summary_cache(
    workflow_or_simulation: Union[FeatureWorkflow, FeatureSimulation, pd.DataFrame],
    config: Optional[FeatureEngineeringConfig] = None,
    freqs: Optional[Sequence[float]] = None,
    cache_csv: Optional[Union[str, Path]] = None,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the persistent feature-summary cache used by the stage-04 notebook."""

    cfg = config or (
        workflow_or_simulation.config
        if isinstance(workflow_or_simulation, FeatureWorkflow)
        else FeatureEngineeringConfig()
    )
    path = _feature_metric_cache_path(cfg, cache_csv)
    workflow = (
        workflow_or_simulation
        if isinstance(workflow_or_simulation, FeatureWorkflow)
        else prepare_feature_workflow(workflow_or_simulation, config=cfg)
    )
    metric_names = tuple(FEATURE_TABLES.keys() if metrics is None else metrics)
    cache_freqs = tuple(float(freq) for freq in (freqs if freqs is not None else cfg.freqs))

    frames = []
    for metric in metric_names:
        if metric not in FEATURE_TABLES:
            raise ValueError(f"Unknown feature metric {metric!r}. Options are: {', '.join(FEATURE_TABLES)}")
        summary = feature_metric_summary(workflow, metric, config=cfg, freqs=cache_freqs)
        if summary.empty:
            continue
        frame = summary.copy()
        frame.insert(0, "metric", str(metric))
        frame.insert(0, "cache_version", FEATURE_METRIC_CACHE_VERSION)
        frames.append(frame)

    cache = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["cache_version", "metric", "freq", "cv", "thresh_type", "mean", "sd", "n"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(path, index=False)
    return cache


def feature_metric_summary_cached(
    workflow_or_simulation: Union[FeatureWorkflow, FeatureSimulation, pd.DataFrame],
    metric: str,
    config: Optional[FeatureEngineeringConfig] = None,
    freqs: Optional[Sequence[float]] = None,
    cache_csv: Optional[Union[str, Path]] = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load a feature metric summary from disk, rebuilding it if needed."""

    cfg = config or (
        workflow_or_simulation.config
        if isinstance(workflow_or_simulation, FeatureWorkflow)
        else FeatureEngineeringConfig()
    )
    cache_freqs = tuple(float(freq) for freq in (freqs if freqs is not None else cfg.freqs))
    path = _feature_metric_cache_path(cfg, cache_csv)

    if path.exists() and not rebuild:
        try:
            cached = normalize_threshold_frame(pd.read_csv(path))
            out = _filter_feature_metric_cache(cached, metric=metric, freqs=cache_freqs)
            if not out.empty:
                return out
        except Exception:
            pass

    cached = build_feature_metric_summary_cache(
        workflow_or_simulation,
        config=cfg,
        freqs=cache_freqs,
        cache_csv=path,
    )
    return _filter_feature_metric_cache(cached, metric=metric, freqs=cache_freqs)


def make_feature_settings(config: FeatureEngineeringConfig, bin_ms: float) -> FeatureSettings:
    """Create core feature settings for one bin width."""

    return FeatureSettings(
        t0_ms=float(config.t0_ms),
        t1_ms=float(config.t1_ms),
        bin_ms=float(bin_ms),
        min_ct=int(config.min_ct),
        cap1=bool(config.cap1),
        gap_close_ms=float(config.gap_close_ms),
        min_event_ms=float(config.min_event_ms),
        mrg_init_ms=float(config.mrg_init_ms),
    )


def compute_trial_feature_details(spikes_ms: np.ndarray, fs: FeatureSettings) -> Dict[str, object]:
    """Compute rate/timing features plus plotting details for one trial."""

    edges, active, events_bin, spikes_f = detect_events_binned(spikes_ms, fs)
    events = refine_events_to_spikes(events_bin, spikes_f)

    n_spikes = int(spikes_f.size)
    n_events = int(len(events))
    spikes_per_event = (n_spikes / n_events) if n_events > 0 else float("nan")

    silence_lv = float("nan")
    if n_events > 0:
        sil = []
        sil.append(max(0.0, float(events[0].start_ms) - fs.t0_ms))
        for i in range(n_events - 1):
            sil.append(max(0.0, float(events[i + 1].start_ms) - float(events[i].end_ms)))
        sil.append(max(0.0, fs.t1_ms - float(events[-1].end_ms)))
        sil_arr = np.asarray(sil, dtype=float)
        q = float(fs.bin_ms)
        sil_q = np.maximum(np.round(sil_arr / q) * q, 0.0)
        silence_lv = _lv(sil_q)

    return {
        "edges": edges,
        "active": active,
        "spikes": spikes_f,
        "events": events,
        "n_spikes": float(n_spikes),
        "n_events": float(n_events),
        "spikes_per_event": float(spikes_per_event),
        "silence_lv": float(silence_lv),
    }


def prepare_feature_workflow(
    simulation: Union[FeatureSimulation, pd.DataFrame],
    config: Optional[FeatureEngineeringConfig] = None,
) -> FeatureWorkflow:
    """Build lookup tables used by the stage-04 feature/evidence widget."""

    cfg = config or FeatureEngineeringConfig()
    feature_sim = _ensure_feature_simulation(simulation, config=cfg)
    sim = feature_sim.data.copy().reset_index(drop=True)
    feature_sim = FeatureSimulation(data=sim, spikes_col=feature_sim.spikes_col, rep_col=feature_sim.rep_col)

    cvs_by_ft: Dict[Tuple[float, str], Tuple[float, ...]] = {}
    reps_by_ft: Dict[Tuple[float, str], Tuple[int, ...]] = {}
    for (freq, thresh_type), group in sim.groupby(["freq", "thresh_type"]):
        key = (float(freq), str(thresh_type))
        cvs_by_ft[key] = tuple(sorted(set(float(x) for x in group["cv"].dropna().tolist())))
        reps_by_ft[key] = tuple(sorted(set(int(x) for x in group["rep"].dropna().tolist())))

    if not cvs_by_ft:
        raise RuntimeError("No feature workflow groups were available.")

    return FeatureWorkflow(
        simulation=feature_sim,
        config=cfg,
        sim_index=build_sim_index(sim),
        cvs_by_freq_thresh=cvs_by_ft,
        reps_by_freq_thresh=reps_by_ft,
    )


def feature_frequencies(workflow: FeatureWorkflow) -> List[float]:
    return sorted(set(float(freq) for freq, _thresh in workflow.cvs_by_freq_thresh.keys()))


def feature_threshold_options(workflow: FeatureWorkflow, freq: float) -> List[str]:
    available = {str(thresh) for f, thresh in workflow.cvs_by_freq_thresh.keys() if np.isclose(float(f), float(freq))}
    ordered = [thresh for thresh in THRESH_ORDER if thresh in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def feature_rep_options(workflow: FeatureWorkflow, freq: float, thresh_type: str) -> List[int]:
    return list(workflow.reps_by_freq_thresh.get((float(freq), str(thresh_type)), ()))


def _rate_keys_for_feature_bin(bin_ms: float) -> Tuple[str, ...]:
    return ("n_spikes",) if float(bin_ms) <= 1.0 + 1e-9 else ("n_events", "spikes_per_event")


def trial_feature_details(
    workflow: FeatureWorkflow,
    *,
    freq: float,
    thresh_type: str,
    cv: float,
    rep: int,
    bin_ms: float,
) -> Dict[str, object]:
    """Return cached feature details for one stage-03 response row."""

    key = (float(freq), str(thresh_type), float(cv), int(rep), float(bin_ms))
    cached = workflow.feature_cache.get(key)
    if cached is not None:
        return cached
    spikes = get_spikes(
        workflow.simulation.data,
        workflow.sim_index,
        workflow.simulation.spikes_col,
        freq=float(freq),
        th=str(thresh_type),
        cv=float(cv),
        rep=int(rep),
    )
    details = compute_trial_feature_details(spikes, make_feature_settings(workflow.config, bin_ms))
    workflow.feature_cache[key] = details
    return details


def trial_feature_evidence(
    workflow: FeatureWorkflow,
    *,
    freq: float,
    thresh_type: str,
    cv: float,
    rep: int,
    bin_ms: float,
    jnd_k: Optional[float] = None,
) -> Tuple[float, float, Dict[str, object]]:
    """Compute selected-repetition rate and timing evidence for one CV."""

    k = float(workflow.config.jnd_k if jnd_k is None else jnd_k)
    fx = trial_feature_details(workflow, freq=freq, thresh_type=thresh_type, cv=cv, rep=rep, bin_ms=bin_ms)
    f0 = trial_feature_details(workflow, freq=freq, thresh_type=thresh_type, cv=0.0, rep=rep, bin_ms=bin_ms)
    f1 = trial_feature_details(workflow, freq=freq, thresh_type=thresh_type, cv=1.0, rep=rep, bin_ms=bin_ms)

    rate_keys_all = ("n_spikes", "n_events", "spikes_per_event")
    x_rate = {name: float(fx.get(name, np.nan)) for name in rate_keys_all}
    t0_rate = {name: float(f0.get(name, np.nan)) for name in rate_keys_all}
    t1_rate = {name: float(f1.get(name, np.nan)) for name in rate_keys_all}

    e_rate = rate_vote_hardjnd(x_rate, t0_rate, t1_rate, k=k, keys=_rate_keys_for_feature_bin(bin_ms))
    e_time = timing_evidence_lv(
        float(fx.get("silence_lv", np.nan)),
        float(f0.get("silence_lv", np.nan)),
        float(f1.get("silence_lv", np.nan)),
    )
    return float(e_rate), float(e_time), fx


def collect_feature_evidence_matrices(
    workflow: FeatureWorkflow,
    *,
    freq: float,
    thresh_type: str,
    bin_ms: float,
    reps: Sequence[int],
    cvs: Sequence[float],
    jnd_k: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw rate/timing evidence matrices with shape reps x CVs."""

    reps = [int(rep) for rep in reps]
    cvs = [float(cv) for cv in cvs]
    e_rate = np.full((len(reps), len(cvs)), np.nan, dtype=float)
    e_time = np.full((len(reps), len(cvs)), np.nan, dtype=float)
    for i, rep in enumerate(reps):
        for j, cv in enumerate(cvs):
            er, et, _features = trial_feature_evidence(
                workflow,
                freq=freq,
                thresh_type=thresh_type,
                cv=cv,
                rep=rep,
                bin_ms=bin_ms,
                jnd_k=jnd_k,
            )
            if np.isfinite(er):
                e_rate[i, j] = float(er)
            if np.isfinite(et):
                e_time[i, j] = float(et)
    return e_rate, e_time


def _endpoint_pm1_transform_matrix(values: np.ndarray, y0: float, y1: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return np.zeros_like(values)
    den = float(y1 - y0)
    if not np.isfinite(den) or abs(den) < 1e-12:
        return np.zeros_like(values)
    a = 2.0 / den
    b = -1.0 - a * float(y0)
    return a * values + b


def _sem_1d(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _sem_axis0(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.asarray([], dtype=float)
    return np.asarray([_sem_1d(values[:, j]) for j in range(values.shape[1])], dtype=float)


def normalize_feature_evidence_summary(
    cvs: Sequence[float],
    e_rate_mat: np.ndarray,
    e_time_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Endpoint-normalize mean evidence and summarize SEM across repetitions."""

    cvs_arr = np.asarray(cvs, dtype=float)
    out = {
        "er_mean": np.full(cvs_arr.shape, np.nan, dtype=float),
        "er_sem": np.full(cvs_arr.shape, np.nan, dtype=float),
        "et_mean": np.full(cvs_arr.shape, np.nan, dtype=float),
        "et_sem": np.full(cvs_arr.shape, np.nan, dtype=float),
        "rate_rel": np.full(cvs_arr.shape, np.nan, dtype=float),
    }
    if e_rate_mat.size == 0 or e_time_mat.size == 0:
        return out
    i0 = int(np.where(np.isclose(cvs_arr, 0.0))[0][0]) if np.any(np.isclose(cvs_arr, 0.0)) else None
    i1 = int(np.where(np.isclose(cvs_arr, 1.0))[0][0]) if np.any(np.isclose(cvs_arr, 1.0)) else None
    if i0 is None or i1 is None:
        return out

    er_mean_raw = np.nanmean(e_rate_mat, axis=0)
    et_mean_raw = np.nanmean(e_time_mat, axis=0)
    rate_rel = np.nanmean(np.abs(e_rate_mat) > 1e-12, axis=0)

    er_rep_pm1 = _endpoint_pm1_transform_matrix(e_rate_mat, float(er_mean_raw[i0]), float(er_mean_raw[i1]))
    et_rep_pm1 = _endpoint_pm1_transform_matrix(e_time_mat, float(et_mean_raw[i0]), float(et_mean_raw[i1]))
    er_rep_pm1 = np.clip(er_rep_pm1 * rate_rel[None, :], -1.0, 1.0)
    et_rep_pm1 = np.clip(et_rep_pm1, -1.0, 1.0)

    out["er_mean"] = np.nanmean(er_rep_pm1, axis=0)
    out["er_sem"] = _sem_axis0(er_rep_pm1)
    out["et_mean"] = np.nanmean(et_rep_pm1, axis=0)
    out["et_sem"] = _sem_axis0(et_rep_pm1)
    out["rate_rel"] = rate_rel
    return out


def _square_trace_from_events(
    events: Sequence[_Event],
    t0_ms: float,
    t1_ms: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if events is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    bounds = [float(t0_ms), float(t1_ms)]
    cleaned: List[Tuple[float, float]] = []
    singletons: List[float] = []
    for event in events:
        start = float(getattr(event, "start_ms", np.nan))
        end = float(getattr(event, "end_ms", np.nan))
        if (not np.isfinite(start)) or (not np.isfinite(end)):
            continue
        start = max(float(t0_ms), min(float(t1_ms), start))
        end = max(float(t0_ms), min(float(t1_ms), end))
        if end < start:
            continue
        if end == start:
            singletons.append(start)
            continue
        cleaned.append((start, end))
        bounds.extend([start, end])

    if not cleaned:
        x = np.asarray([float(t0_ms), float(t1_ms)], dtype=float)
        y = np.asarray([0.0, 0.0], dtype=float)
        return x, y, np.asarray(singletons, dtype=float)

    x = np.asarray(sorted(set(bounds)), dtype=float)
    if x.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray(singletons, dtype=float)

    y = np.zeros(x.size, dtype=float)
    for i in range(x.size - 1):
        midpoint = 0.5 * (x[i] + x[i + 1])
        y[i] = 1.0 if any((midpoint >= start) and (midpoint < end) for start, end in cleaned) else 0.0
    y[-1] = y[-2] if y.size >= 2 else 0.0
    return x, y, np.asarray(singletons, dtype=float)



# ============================
# Mixture fit
# ============================

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # stable sigmoid
    out = np.empty_like(x)
    m = x >= 0
    out[m] = 1.0 / (1.0 + np.exp(-x[m]))
    ex = np.exp(x[~m])
    out[~m] = ex / (1.0 + ex)
    return out


def map_pchoose_to_pcorrect(p_choose_cv1: np.ndarray, block: str, cv: np.ndarray) -> np.ndarray:
    """Map latent p(choose CV=1) into observed p(correct) for Slack task.

    Slack scoring depends on which stimulus is the *template* for the block:
      - std0: template is CV=0.0
          * at CV=0.0, correct == choose CV=0 => p_correct = 1 - p_choose_cv1
          * at CV>0,  correct == choose CV=1 => p_correct = p_choose_cv1
      - std1: template is CV=1.0
          * at CV=1.0, correct == choose CV=1 => p_correct = p_choose_cv1
          * at CV<1,  correct == choose CV=0 => p_correct = 1 - p_choose_cv1

    This removes the endpoint discontinuity when you convert Slack p(correct) into a single monotonic
    latent variable p(choose CV=1).
    """
    cv = np.asarray(cv, dtype=float)
    p = np.asarray(p_choose_cv1, dtype=float)

    if block == "std0":
        is_template = np.isclose(cv, 0.0)
        return np.where(is_template, 1.0 - p, p)

    if block == "std1":
        is_template = np.isclose(cv, 1.0)
        return np.where(is_template, p, 1.0 - p)

    raise ValueError(f"Unknown block: {block}")

@dataclass
class FitResult:
    scope_key: str
    model_bank: str
    bin_ms: float
    aux_bin_ms: float
    mc: int
    loto_th: Optional[str]
    drop_thresh: str
    is_loto: bool
    fixed_thresh: Optional[str]
    is_fixed_thresh: bool
    base_run_id: str
    freq: float
    block: str
    n_thresh: int
    params: Dict[str, float]
    weights: Dict[str, float]
    weights_shared: Dict[str, float]
    weights_rate: Dict[str, float]
    weights_timing: Dict[str, float]
    metrics: Dict[str, float]
    loss_reg_l2: float = 1e-3
    weight_kl: float = 0.0
    allow_zero_gains: bool = False
    rate_gate_mode: str = "none"
    rate_gate_reg: float = 0.0
    rate_gate_prior_floor: float = 1.0
    rate_gate_prior_center: float = 0.5
    rate_gate_prior_slope: float = 8.0
    shared_thresh_mode: str = "none"
    weight_entropy_floor_k: float = 0.0
    weight_entropy_floor_lambda: float = 0.0
    weight_max_cap: float = 1.0
    weight_max_cap_lambda: float = 0.0
    thresh_flex_mode: str = "shared"
    weight_smooth_lambda: float = 0.0
    weights_by_cv: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)


def _bin_rate_mode(bin_ms: float) -> str:
    return "bin1" if bin_ms <= 1.0 + 1e-9 else "bin10"


def _is_combined_model_bank(model_bank: str) -> bool:
    return str(model_bank).strip().lower() in COMBINED_MODEL_BANKS


def _uses_rate_gate(model_bank: str, rate_gate_mode: str) -> bool:
    mb = str(model_bank).strip().lower()
    rg = str(rate_gate_mode).strip().lower()
    return bool(mb in HYBRID_FAMILY_MODEL_BANKS and rg == "delta_cv")


def map_pchoose_to_pcorrect_mixed(
    p_choose_cv1: np.ndarray,
    blocks: np.ndarray,
    cvs: np.ndarray,
) -> np.ndarray:
    """Map p(choose CV=1) -> p(correct) for mixed block rows."""
    p = np.asarray(p_choose_cv1, dtype=float)
    b = np.asarray(blocks).astype(str)
    cv = np.asarray(cvs, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    for blk in sorted(set(b.tolist())):
        m = (b == blk)
        out[m] = map_pchoose_to_pcorrect(p[m], blk, cv[m])
    return out


def delta_cv_from_template_mixed(cvs: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Distance from template CV, in [0,1], for mixed std0/std1 rows."""
    cv = np.asarray(cvs, dtype=float)
    b = np.asarray(blocks).astype(str)
    out = np.full(cv.shape, np.nan, dtype=float)
    m0 = (b == "std0")
    m1 = (b == "std1")
    out[m0] = cv[m0]
    out[m1] = 1.0 - cv[m1]
    # Fallback for unexpected blocks: symmetric distance to midpoint.
    mu = ~(m0 | m1)
    out[mu] = np.abs(cv[mu] - 0.5)
    return np.clip(out, 0.0, 1.0)


def rate_gate_from_delta(delta_cv: np.ndarray, floor: float, center: float, slope: float) -> np.ndarray:
    """Monotonic rate gate from deltaCV.

    gate(delta) = floor + (1-floor)*sigmoid(slope*(delta-center))
    """
    d = np.asarray(delta_cv, dtype=float)
    fl = float(np.clip(floor, 0.0, 1.0))
    c = float(np.clip(center, 0.0, 1.0))
    s = float(max(1e-9, slope))
    return fl + (1.0 - fl) * sigmoid(s * (d - c))


def effective_k(weights: np.ndarray) -> float:
    """Effective number of active components: exp(H(w))."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    w = np.clip(w, 1e-12, 1.0)
    h = -float(np.sum(w * np.log(w)))
    return float(np.exp(h))


def _entmax_rows(u_rows: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(u_rows, dtype=float)
    if arr.ndim == 1:
        return _entmax_bisect(arr, alpha=float(alpha))
    return np.vstack([_entmax_bisect(row, alpha=float(alpha)) for row in arr])


def _format_cv_key(cv: float) -> str:
    return f"{float(cv):.12g}"


def _weights_matrix_from_cv_map(
    *,
    cvs: np.ndarray,
    ths: Sequence[str],
    weights_by_cv: Dict[str, Dict[str, float]],
) -> np.ndarray:
    cvs_arr = np.asarray(cvs, dtype=float)
    keys = [_format_cv_key(cv) for cv in cvs_arr]
    missing = [k for k in keys if k not in weights_by_cv]
    if missing:
        raise KeyError(f"Missing per-point weights for anchors: {missing}")
    mat = np.zeros((cvs_arr.size, len(ths)), dtype=float)
    for i, key in enumerate(keys):
        w_row = weights_by_cv[key]
        vals = np.asarray([float(w_row.get(th, 0.0)) for th in ths], dtype=float)
        s = float(vals.sum())
        if not np.isfinite(s) or s <= 0.0:
            raise ValueError(f"Invalid per-point weights for cv={key}")
        mat[i, :] = vals / s
    return mat


def predict_design_outputs(
    *,
    params: Dict[str, float],
    weights: Dict[str, float],
    ths: Sequence[str],
    Er: np.ndarray,
    Et: np.ndarray,
    cvs: np.ndarray,
    blocks: np.ndarray,
    model_bank: str = "hybrid_combined",
    rate_gate_mode: str = "none",
    Er_aux: Optional[np.ndarray] = None,
    Et_aux: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Forward-pass predictions and intermediate drives for a design matrix."""
    w_vec = np.asarray([float(weights[t]) for t in ths], dtype=float)
    w_vec = w_vec / (w_vec.sum() + EPS)

    Er_arr = np.asarray(Er, dtype=float)
    Et_arr = np.asarray(Et, dtype=float)
    if Er_arr.ndim != 2 or Et_arr.ndim != 2 or Er_arr.shape != Et_arr.shape:
        raise ValueError("Er/Et must be 2D matrices with matching shapes")
    n_rows = Er_arr.shape[0]

    if Er_aux is None:
        Er_aux_arr = np.zeros_like(Er_arr)
    else:
        Er_aux_arr = np.asarray(Er_aux, dtype=float)
        if Er_aux_arr.shape != Er_arr.shape:
            raise ValueError("Er_aux must match Er shape")
    if Et_aux is None:
        Et_aux_arr = np.zeros_like(Et_arr)
    else:
        Et_aux_arr = np.asarray(Et_aux, dtype=float)
        if Et_aux_arr.shape != Et_arr.shape:
            raise ValueError("Et_aux must match Et shape")

    rate_drive = Er_arr @ w_vec
    timing_drive = Et_arr @ w_vec
    rate_drive_aux = Er_aux_arr @ w_vec
    timing_drive_aux = Et_aux_arr @ w_vec
    gate_vals = np.ones(n_rows, dtype=float)

    mb = str(model_bank).strip().lower()
    if mb not in {"rate_combined", "timing_combined", "hybrid_combined"}:
        raise ValueError(f"Unknown release model_bank: {model_bank}")

    if _uses_rate_gate(mb, rate_gate_mode):
        delta = delta_cv_from_template_mixed(np.asarray(cvs, dtype=float), np.asarray(blocks))
        gate_vals = rate_gate_from_delta(
            delta,
            float(params.get("rate_gate_floor", 1.0)),
            float(params.get("rate_gate_center", 0.5)),
            float(params.get("rate_gate_slope", 8.0)),
        )
        rate_drive = rate_drive * gate_vals
        rate_drive_aux = rate_drive_aux * gate_vals

    b = float(params.get("b", 0.0))
    temp = float(params.get("temp", 1.0))
    g_rate = float(params.get("g_rate", 0.0))
    g_time = float(params.get("g_time", 0.0))
    g_rate_aux = float(params.get("g_rate_aux", 0.0))
    g_time_aux = float(params.get("g_time_aux", 0.0))

    if mb == "rate_combined":
        drive = g_rate * rate_drive + g_rate_aux * rate_drive_aux
    elif mb == "timing_combined":
        drive = g_time * timing_drive + g_time_aux * timing_drive_aux
    else:
        drive = (
            g_rate * rate_drive
            + g_time * timing_drive
            + g_rate_aux * rate_drive_aux
            + g_time_aux * timing_drive_aux
        )

    z = b + temp * drive
    p_choose = sigmoid(z)
    p_corr = map_pchoose_to_pcorrect_mixed(p_choose, blocks=np.asarray(blocks), cvs=np.asarray(cvs, dtype=float))
    return {
        "z": np.asarray(z, dtype=float),
        "p_choose_cv1": np.asarray(p_choose, dtype=float),
        "p_correct": np.asarray(p_corr, dtype=float),
        "rate_drive": np.asarray(rate_drive, dtype=float),
        "timing_drive": np.asarray(timing_drive, dtype=float),
        "rate_drive_aux": np.asarray(rate_drive_aux, dtype=float),
        "timing_drive_aux": np.asarray(timing_drive_aux, dtype=float),
        "gate": np.asarray(gate_vals, dtype=float),
    }


def predict_design_outputs_rowwise(
    *,
    params: Dict[str, float],
    weights_by_cv: Dict[str, Dict[str, float]],
    ths: Sequence[str],
    Er: np.ndarray,
    Et: np.ndarray,
    cvs: np.ndarray,
    blocks: np.ndarray,
    model_bank: str = "hybrid_combined",
    rate_gate_mode: str = "none",
    Er_aux: Optional[np.ndarray] = None,
    Et_aux: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Forward-pass predictions with rowwise threshold weights keyed by CV."""
    Er_arr = np.asarray(Er, dtype=float)
    Et_arr = np.asarray(Et, dtype=float)
    if Er_arr.ndim != 2 or Et_arr.ndim != 2 or Er_arr.shape != Et_arr.shape:
        raise ValueError("Er/Et must be 2D matrices with matching shapes")
    n_rows = Er_arr.shape[0]

    if Er_aux is None:
        Er_aux_arr = np.zeros_like(Er_arr)
    else:
        Er_aux_arr = np.asarray(Er_aux, dtype=float)
        if Er_aux_arr.shape != Er_arr.shape:
            raise ValueError("Er_aux must match Er shape")
    if Et_aux is None:
        Et_aux_arr = np.zeros_like(Et_arr)
    else:
        Et_aux_arr = np.asarray(Et_aux, dtype=float)
        if Et_aux_arr.shape != Et_arr.shape:
            raise ValueError("Et_aux must match Et shape")

    W = _weights_matrix_from_cv_map(cvs=np.asarray(cvs, dtype=float), ths=ths, weights_by_cv=weights_by_cv)
    rate_drive = np.sum(Er_arr * W, axis=1)
    timing_drive = np.sum(Et_arr * W, axis=1)
    rate_drive_aux = np.sum(Er_aux_arr * W, axis=1)
    timing_drive_aux = np.sum(Et_aux_arr * W, axis=1)
    gate_vals = np.ones(n_rows, dtype=float)

    mb = str(model_bank).strip().lower()
    if mb != "hybrid_combined":
        raise ValueError(f"Per-point release predictions use hybrid_combined only, got: {model_bank}")

    if _uses_rate_gate(mb, rate_gate_mode):
        delta = delta_cv_from_template_mixed(np.asarray(cvs, dtype=float), np.asarray(blocks))
        gate_vals = rate_gate_from_delta(
            delta,
            float(params.get("rate_gate_floor", 1.0)),
            float(params.get("rate_gate_center", 0.5)),
            float(params.get("rate_gate_slope", 8.0)),
        )
        rate_drive = rate_drive * gate_vals
        rate_drive_aux = rate_drive_aux * gate_vals

    b = float(params.get("b", 0.0))
    temp = float(params.get("temp", 1.0))
    g_rate = float(params.get("g_rate", 0.0))
    g_time = float(params.get("g_time", 0.0))
    g_rate_aux = float(params.get("g_rate_aux", 0.0))
    g_time_aux = float(params.get("g_time_aux", 0.0))

    drive = (
        g_rate * rate_drive
        + g_time * timing_drive
        + g_rate_aux * rate_drive_aux
        + g_time_aux * timing_drive_aux
    )

    z = b + temp * drive
    p_choose = sigmoid(z)
    p_corr = map_pchoose_to_pcorrect_mixed(p_choose, blocks=np.asarray(blocks), cvs=np.asarray(cvs, dtype=float))
    return {
        "z": np.asarray(z, dtype=float),
        "p_choose_cv1": np.asarray(p_choose, dtype=float),
        "p_correct": np.asarray(p_corr, dtype=float),
        "rate_drive": np.asarray(rate_drive, dtype=float),
        "timing_drive": np.asarray(timing_drive, dtype=float),
        "rate_drive_aux": np.asarray(rate_drive_aux, dtype=float),
        "timing_drive_aux": np.asarray(timing_drive_aux, dtype=float),
        "gate": np.asarray(gate_vals, dtype=float),
    }


def evaluate_design_metrics(
    *,
    params: Dict[str, float],
    weights: Dict[str, float],
    ths: Sequence[str],
    Er: np.ndarray,
    Et: np.ndarray,
    cvs: np.ndarray,
    blocks: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    y: np.ndarray,
    model_bank: str = "hybrid_combined",
    rate_gate_mode: str = "none",
    Er_aux: Optional[np.ndarray] = None,
    Et_aux: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate fixed params/weights on an already-built design matrix."""
    pred = predict_design_outputs(
        params=params,
        weights=weights,
        ths=ths,
        Er=Er,
        Et=Et,
        Er_aux=Er_aux,
        Et_aux=Et_aux,
        cvs=cvs,
        blocks=blocks,
        model_bank=model_bank,
        rate_gate_mode=rate_gate_mode,
    )
    p_corr = np.asarray(pred["p_correct"], dtype=float)
    p_corr_c = np.clip(p_corr, 1e-6, 1 - 1e-6)
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    ll = k_arr * np.log(p_corr_c) + (n_arr - k_arr) * np.log(1 - p_corr_c)
    rmse = float(np.sqrt(np.nanmean((p_corr - np.asarray(y, dtype=float)) ** 2)))
    mae = float(np.nanmean(np.abs(p_corr - np.asarray(y, dtype=float))))
    # Report trial-normalized NLL (mean negative log-likelihood per trial).
    total_trials = float(np.nansum(n_arr))
    if not np.isfinite(total_trials) or total_trials <= 0.0:
        nll = float(np.nan)
    else:
        nll = float(-np.sum(ll) / total_trials)
    return {
        "nll": nll,
        "rmse": rmse,
        "mae": mae,
    }


def evaluate_design_metrics_rowwise(
    *,
    params: Dict[str, float],
    weights_by_cv: Dict[str, Dict[str, float]],
    ths: Sequence[str],
    Er: np.ndarray,
    Et: np.ndarray,
    cvs: np.ndarray,
    blocks: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    y: np.ndarray,
    model_bank: str = "hybrid_combined",
    rate_gate_mode: str = "none",
    Er_aux: Optional[np.ndarray] = None,
    Et_aux: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    pred = predict_design_outputs_rowwise(
        params=params,
        weights_by_cv=weights_by_cv,
        ths=ths,
        Er=Er,
        Et=Et,
        Er_aux=Er_aux,
        Et_aux=Et_aux,
        cvs=cvs,
        blocks=blocks,
        model_bank=model_bank,
        rate_gate_mode=rate_gate_mode,
    )
    p_corr = np.asarray(pred["p_correct"], dtype=float)
    p_corr_c = np.clip(p_corr, 1e-6, 1 - 1e-6)
    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    ll = k_arr * np.log(p_corr_c) + (n_arr - k_arr) * np.log(1 - p_corr_c)
    rmse = float(np.sqrt(np.nanmean((p_corr - np.asarray(y, dtype=float)) ** 2)))
    mae = float(np.nanmean(np.abs(p_corr - np.asarray(y, dtype=float))))
    total_trials = float(np.nansum(n_arr))
    if not np.isfinite(total_trials) or total_trials <= 0.0:
        nll = float(np.nan)
    else:
        nll = float(-np.sum(ll) / total_trials)
    return {
        "nll": nll,
        "rmse": rmse,
        "mae": mae,
    }


def fit_mixture_from_design(
    *,
    Er: np.ndarray,
    Et: np.ndarray,
    Er_aux: Optional[np.ndarray] = None,
    Et_aux: Optional[np.ndarray] = None,
    cvs: np.ndarray,
    blocks: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    y: np.ndarray,
    ths: Sequence[str],
    ms: ModelSettings,
    loss_reg_l2: float = 1e-3,
    weight_kl: float = 0.0,
    allow_zero_gains: bool = False,
    model_bank: str = "hybrid_combined",
    rate_gate_mode: str = "none",
    rate_gate_reg: float = 0.0,
    rate_gate_prior_floor: float = 1.0,
    rate_gate_prior_center: float = 0.5,
    rate_gate_prior_slope: float = 8.0,
    weight_entropy_floor_k: float = 0.0,
    weight_entropy_floor_lambda: float = 0.0,
    weight_max_cap: float = 1.0,
    weight_max_cap_lambda: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Fit mixture weights + bank gains on a generic design matrix.

    This supports pooled fitting over any combination of frequency/block rows.
    """
    Er = np.asarray(Er, dtype=float)
    Et = np.asarray(Et, dtype=float)
    cvs = np.asarray(cvs, dtype=float)
    blocks = np.asarray(blocks).astype(str)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    if Er.ndim != 2 or Et.ndim != 2 or Er.shape != Et.shape:
        raise ValueError("Er/Et must be 2D matrices with matching shapes")
    if Er.shape[0] == 0:
        raise RuntimeError("No design rows to fit")

    n_th = len(ths)
    if Er.shape[1] != n_th:
        raise ValueError("Er/Et column count must match number of threshold types")
    if Er_aux is None:
        Er_aux_arr = np.zeros_like(Er)
    else:
        Er_aux_arr = np.asarray(Er_aux, dtype=float)
        if Er_aux_arr.shape != Er.shape:
            raise ValueError("Er_aux must match Er shape")
    if Et_aux is None:
        Et_aux_arr = np.zeros_like(Et)
    else:
        Et_aux_arr = np.asarray(Et_aux, dtype=float)
        if Et_aux_arr.shape != Et.shape:
            raise ValueError("Et_aux must match Et shape")

    def softplus(x: float) -> float:
        if x > 30:
            return x
        return float(np.log1p(np.exp(x)))

    model_bank = str(model_bank).strip().lower()
    if model_bank not in {"rate_combined", "timing_combined", "hybrid_combined"}:
        raise ValueError(f"Unknown release model_bank: {model_bank}")
    rate_gate_mode = str(rate_gate_mode).strip().lower()
    if rate_gate_mode not in {"none", "delta_cv"}:
        raise ValueError(f"Unknown rate_gate_mode: {rate_gate_mode}")
    use_rate_gate = _uses_rate_gate(model_bank, rate_gate_mode)

    weight_entropy_floor_k = float(max(0.0, weight_entropy_floor_k))
    weight_entropy_floor_lambda = float(max(0.0, weight_entropy_floor_lambda))
    weight_max_cap = float(np.clip(weight_max_cap, 0.0, 1.0))
    weight_max_cap_lambda = float(max(0.0, weight_max_cap_lambda))
    rate_gate_reg = float(max(0.0, rate_gate_reg))
    rate_gate_prior_floor = float(np.clip(rate_gate_prior_floor, 0.0, 1.0))
    rate_gate_prior_center = float(np.clip(rate_gate_prior_center, 0.0, 1.0))
    rate_gate_prior_slope = float(max(1e-6, rate_gate_prior_slope))

    delta_cv = delta_cv_from_template_mixed(cvs, blocks) if use_rate_gate else None

    def inv_sigmoid(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        return float(np.log(p) - np.log(1.0 - p))

    def inv_softplus(y: float) -> float:
        y = float(max(1e-9, y))
        if y > 30.0:
            return y
        return float(np.log(np.expm1(y)))

    # Parameter layouts:
    # - rate_combined:   [u_th..., b, g_rate_raw, g_rate_aux_raw, temp_raw]
    # - timing_combined: [u_th..., b, g_time_raw, g_time_aux_raw, temp_raw]
    # - hybrid_combined: [u_th..., b, g_r_raw, g_t_raw, g_r_aux_raw, g_t_aux_raw, temp_raw]
    # - optional deltaCV rate gate appends:
    #   [gate_floor_raw, gate_center_raw, gate_slope_raw]
    #   gate(delta) = floor + (1-floor)*sigmoid(slope*(delta-center))
    if model_bank in {"rate_combined", "timing_combined"}:
        n_params = n_th + 4
    else:  # hybrid_combined
        n_params = n_th + 6
    if use_rate_gate:
        # [gate_floor_raw, gate_center_raw, gate_slope_raw]
        n_params += 3

    def _pos(raw: float) -> float:
        return max(0.0, float(raw)) if allow_zero_gains else softplus(float(raw))

    def unpack(theta: np.ndarray):
        u = theta[:n_th]
        i = n_th
        b = float(theta[i]); i += 1
        extras: Dict[str, float] = {}
        gr_aux = 0.0
        gt_aux = 0.0

        if model_bank == "hybrid_combined":
            gr = _pos(theta[i]); i += 1
            gt = _pos(theta[i]); i += 1
            gr_aux = _pos(theta[i]); i += 1
            gt_aux = _pos(theta[i]); i += 1
            temp = _pos(theta[i]); i += 1
        elif model_bank == "rate_combined":
            gr = _pos(theta[i]); i += 1
            gt = 0.0
            gr_aux = _pos(theta[i]); i += 1
            temp = _pos(theta[i]); i += 1
        elif model_bank == "timing_combined":
            gr = 0.0
            gt = _pos(theta[i]); i += 1
            gt_aux = _pos(theta[i]); i += 1
            temp = _pos(theta[i]); i += 1
        else:
            raise ValueError(f"Unknown release model_bank: {model_bank}")
        if use_rate_gate:
            gate_floor = float(sigmoid(np.array([float(theta[i])], dtype=float))[0]); i += 1
            gate_center = float(sigmoid(np.array([float(theta[i])], dtype=float))[0]); i += 1
            gate_slope = float(softplus(float(theta[i]))); i += 1
        else:
            gate_floor = 1.0
            gate_center = 0.5
            gate_slope = 8.0
        extras["rate_gate_floor"] = float(gate_floor)
        extras["rate_gate_center"] = float(gate_center)
        extras["rate_gate_slope"] = float(gate_slope)

        w = _entmax_bisect(u, alpha=ms.entmax_alpha)
        return w, b, gr, gt, gr_aux, gt_aux, temp, extras

    def nll(theta: np.ndarray) -> float:
        w, b, gr, gt, gr_aux, gt_aux, temp, extras = unpack(theta)
        rate_drive = (Er @ w)
        timing_drive = (Et @ w)
        rate_drive_aux = (Er_aux_arr @ w)
        timing_drive_aux = (Et_aux_arr @ w)
        if use_rate_gate and delta_cv is not None:
            gate_vals = rate_gate_from_delta(
                delta_cv,
                extras["rate_gate_floor"],
                extras["rate_gate_center"],
                extras["rate_gate_slope"],
            )
            rate_drive = rate_drive * gate_vals
            if model_bank == "hybrid_combined":
                rate_drive_aux = rate_drive_aux * gate_vals
        if model_bank == "rate_combined":
            drive = gr * rate_drive + gr_aux * rate_drive_aux
        elif model_bank == "timing_combined":
            drive = gt * timing_drive + gt_aux * timing_drive_aux
        elif model_bank == "hybrid_combined":
            drive = gr * rate_drive + gt * timing_drive + gr_aux * rate_drive_aux + gt_aux * timing_drive_aux
        else:
            raise ValueError(f"Unknown release model_bank: {model_bank}")
        z = b + temp * drive
        p_choose = sigmoid(z)
        p_corr = map_pchoose_to_pcorrect_mixed(p_choose, blocks=blocks, cvs=cvs)
        p_corr = np.clip(p_corr, 1e-6, 1 - 1e-6)
        ll = k * np.log(p_corr) + (n - k) * np.log(1 - p_corr)
        total_trials = float(np.nansum(n))
        if not np.isfinite(total_trials) or total_trials <= 0.0:
            total_trials = 1.0
        # Use trial-normalized NLL so regularization scales are tunable across scopes.
        nll_main = float(-(np.sum(ll)) / total_trials)
        reg = loss_reg_l2 * float(np.sum(theta * theta))
        kl_reg = 0.0
        w_safe = np.clip(np.asarray(w, dtype=float), 1e-12, 1.0)
        if float(weight_kl) > 0.0:
            # KL(w || uniform): discourages collapsed one-hot mixtures.
            kl_reg = float(weight_kl) * float(np.sum(w_safe * np.log(w_safe * float(n_th))))
        entropy_floor_reg = 0.0
        if weight_entropy_floor_k > 0.0 and weight_entropy_floor_lambda > 0.0:
            k_eff = effective_k(w_safe)
            entropy_floor_reg = weight_entropy_floor_lambda * float(max(0.0, weight_entropy_floor_k - k_eff) ** 2)
        cap_reg = 0.0
        if weight_max_cap < 1.0 and weight_max_cap_lambda > 0.0:
            over = np.maximum(np.asarray(w, dtype=float) - float(weight_max_cap), 0.0)
            cap_reg = float(weight_max_cap_lambda) * float(np.sum(over * over))
        gate_reg = 0.0
        if use_rate_gate and rate_gate_reg > 0.0:
            gate_reg = rate_gate_reg * float(
                (extras["rate_gate_floor"] - rate_gate_prior_floor) ** 2
                + (extras["rate_gate_center"] - rate_gate_prior_center) ** 2
                + ((extras["rate_gate_slope"] - rate_gate_prior_slope) / max(rate_gate_prior_slope, 1e-6)) ** 2
            )
        return float(nll_main + reg + kl_reg + entropy_floor_reg + cap_reg + gate_reg)

    theta0 = np.zeros(n_params, dtype=float)
    if allow_zero_gains:
        j = n_th + 1  # after bias
        if model_bank == "rate_combined":
            theta0[j] = 0.5
            theta0[j + 1] = 0.5
            theta0[j + 2] = 1.0
        elif model_bank == "timing_combined":
            theta0[j] = 0.5
            theta0[j + 1] = 0.5
            theta0[j + 2] = 1.0
        else:  # hybrid_combined
            theta0[j] = 0.5
            theta0[j + 1] = 0.5
            theta0[j + 2] = 0.5
            theta0[j + 3] = 0.5
            theta0[j + 4] = 1.0
    if use_rate_gate:
        gate_i = n_params - 3
        theta0[gate_i] = inv_sigmoid(rate_gate_prior_floor)
        theta0[gate_i + 1] = inv_sigmoid(rate_gate_prior_center)
        theta0[gate_i + 2] = inv_softplus(rate_gate_prior_slope)

    if minimize is None:
        rng = np.random.default_rng(ms.seed)
        best = (nll(theta0), theta0.copy())
        for _ in range(4000):
            cand = theta0 + rng.normal(scale=0.5, size=theta0.size)
            v = nll(cand)
            if v < best[0]:
                best = (v, cand)
        theta_hat = best[1]
    else:
        bounds = None
        if allow_zero_gains:
            bounds = [(None, None)] * n_params
            j = n_th + 1  # after bias
            if model_bank in {"rate_combined", "timing_combined"}:
                bounds[j] = (0.0, None)
                bounds[j + 1] = (0.0, None)
                bounds[j + 2] = (0.0, None)
            else:  # hybrid_combined
                bounds[j] = (0.0, None)
                bounds[j + 1] = (0.0, None)
                bounds[j + 2] = (0.0, None)
                bounds[j + 3] = (0.0, None)
                bounds[j + 4] = (0.0, None)
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=250))
        theta_hat = res.x

    w, b, gr, gt, gr_aux, gt_aux, temp, extras = unpack(theta_hat)
    params = {"b": float(b), "g_rate": float(gr), "g_time": float(gt), "temp": float(temp)}
    params["g_rate_aux"] = float(gr_aux)
    params["g_time_aux"] = float(gt_aux)
    params["rate_gate_floor"] = float(extras.get("rate_gate_floor", 1.0))
    params["rate_gate_center"] = float(extras.get("rate_gate_center", 0.5))
    params["rate_gate_slope"] = float(extras.get("rate_gate_slope", 8.0))
    weights = {str(ths[i]): float(w[i]) for i in range(n_th)}
    metrics = evaluate_design_metrics(
        params=params,
        weights=weights,
        ths=ths,
        Er=Er,
        Et=Et,
        Er_aux=Er_aux_arr,
        Et_aux=Et_aux_arr,
        cvs=cvs,
        blocks=blocks,
        k=k,
        n=n,
        y=y,
        model_bank=model_bank,
        rate_gate_mode=rate_gate_mode,
    )
    metrics["nll"] = float(nll(theta_hat))
    return params, weights, metrics


def fit_shared_weights_panel_params_from_designs(
    *,
    designs: Sequence[Tuple[Tuple[float, str], Dict[str, np.ndarray]]],
    ths: Sequence[str],
    ms: ModelSettings,
    loss_reg_l2: float = 1e-3,
    weight_kl: float = 0.0,
    allow_zero_gains: bool = False,
    model_bank: str = "hybrid_combined",
    rate_gate_mode: str = "none",
    rate_gate_reg: float = 0.0,
    rate_gate_prior_floor: float = 1.0,
    rate_gate_prior_center: float = 0.5,
    rate_gate_prior_slope: float = 8.0,
    weight_entropy_floor_k: float = 0.0,
    weight_entropy_floor_lambda: float = 0.0,
    weight_max_cap: float = 1.0,
    weight_max_cap_lambda: float = 0.0,
    panel_balance: bool = True,
) -> Tuple[Dict[Tuple[float, str], Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """Jointly fit one shared threshold mixture with panel-specific readout params.

    Each panel in `designs` gets its own (b, gains, temp, gate params), while
    threshold mixture weights are shared across all panels in the group.
    """
    if not designs:
        raise RuntimeError("No panel designs provided")

    # Degenerate case: one panel -> defer to the standard fitter.
    if len(designs) == 1:
        key, d = designs[0]
        params, weights, metrics = fit_mixture_from_design(
            Er=d["Er"],
            Et=d["Et"],
            Er_aux=d.get("Er_aux", None),
            Et_aux=d.get("Et_aux", None),
            cvs=d["cvs"],
            blocks=d["blocks"],
            k=d["k"],
            n=d["n"],
            y=d["y"],
            ths=ths,
            ms=ms,
            loss_reg_l2=loss_reg_l2,
            weight_kl=weight_kl,
            allow_zero_gains=allow_zero_gains,
            model_bank=model_bank,
            rate_gate_mode=rate_gate_mode,
            rate_gate_reg=rate_gate_reg,
            rate_gate_prior_floor=rate_gate_prior_floor,
            rate_gate_prior_center=rate_gate_prior_center,
            rate_gate_prior_slope=rate_gate_prior_slope,
            weight_entropy_floor_k=weight_entropy_floor_k,
            weight_entropy_floor_lambda=weight_entropy_floor_lambda,
            weight_max_cap=weight_max_cap,
            weight_max_cap_lambda=weight_max_cap_lambda,
        )
        return {key: params}, weights, metrics

    n_th = len(ths)
    if n_th <= 0:
        raise ValueError("No threshold types provided")

    def softplus(x: float) -> float:
        if x > 30:
            return x
        return float(np.log1p(np.exp(x)))

    def _pos(raw: float) -> float:
        return max(0.0, float(raw)) if allow_zero_gains else softplus(float(raw))

    model_bank = str(model_bank).strip().lower()
    if model_bank not in {"rate_combined", "timing_combined", "hybrid_combined"}:
        raise ValueError(f"Unknown release model_bank: {model_bank}")
    rate_gate_mode = str(rate_gate_mode).strip().lower()
    if rate_gate_mode not in {"none", "delta_cv"}:
        raise ValueError(f"Unknown rate_gate_mode: {rate_gate_mode}")
    use_rate_gate = _uses_rate_gate(model_bank, rate_gate_mode)

    weight_entropy_floor_k = float(max(0.0, weight_entropy_floor_k))
    weight_entropy_floor_lambda = float(max(0.0, weight_entropy_floor_lambda))
    weight_max_cap = float(np.clip(weight_max_cap, 0.0, 1.0))
    weight_max_cap_lambda = float(max(0.0, weight_max_cap_lambda))
    rate_gate_reg = float(max(0.0, rate_gate_reg))
    rate_gate_prior_floor = float(np.clip(rate_gate_prior_floor, 0.0, 1.0))
    rate_gate_prior_center = float(np.clip(rate_gate_prior_center, 0.0, 1.0))
    rate_gate_prior_slope = float(max(1e-6, rate_gate_prior_slope))

    def inv_sigmoid(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        return float(np.log(p) - np.log(1.0 - p))

    def inv_softplus(y: float) -> float:
        y = float(max(1e-9, y))
        if y > 30.0:
            return y
        return float(np.log(np.expm1(y)))

    # Per-panel parameter layout (excluding shared u_th):
    # - rate_combined:   [b, g_rate_raw, g_rate_aux_raw, temp_raw]
    # - timing_combined: [b, g_time_raw, g_time_aux_raw, temp_raw]
    # - hybrid_combined: [b, g_r_raw, g_t_raw, g_r_aux_raw, g_t_aux_raw, temp_raw]
    # + optional gate [gate_floor_raw, gate_center_raw, gate_slope_raw]
    if model_bank in {"rate_combined", "timing_combined"}:
        panel_dim = 4
    else:  # hybrid_combined
        panel_dim = 6
    if use_rate_gate:
        panel_dim += 3

    panel_rows: List[Dict[str, object]] = []
    for key, d in designs:
        Er = np.asarray(d["Er"], dtype=float)
        Et = np.asarray(d["Et"], dtype=float)
        Er_aux = np.asarray(d.get("Er_aux", np.zeros_like(Er)), dtype=float)
        Et_aux = np.asarray(d.get("Et_aux", np.zeros_like(Et)), dtype=float)
        cvs = np.asarray(d["cvs"], dtype=float)
        blocks = np.asarray(d["blocks"]).astype(str)
        k = np.asarray(d["k"], dtype=float)
        n = np.asarray(d["n"], dtype=float)
        y = np.asarray(d["y"], dtype=float)
        if Er.ndim != 2 or Et.ndim != 2 or Er.shape != Et.shape:
            raise ValueError("Er/Et must be 2D matrices with matching shapes")
        if Er_aux.shape != Er.shape or Et_aux.shape != Et.shape:
            raise ValueError("Er_aux/Et_aux must match Er/Et shapes")
        if Er.shape[1] != n_th:
            raise ValueError("Er/Et column count must match number of threshold types")
        if Er.shape[0] == 0:
            raise RuntimeError("Panel design has zero rows")
        delta_cv = delta_cv_from_template_mixed(cvs, blocks) if use_rate_gate else None
        total_trials = float(np.nansum(n))
        if (not np.isfinite(total_trials)) or total_trials <= 0.0:
            total_trials = 1.0
        panel_rows.append(
            {
                "key": key,
                "Er": Er,
                "Et": Et,
                "Er_aux": Er_aux,
                "Et_aux": Et_aux,
                "cvs": cvs,
                "blocks": blocks,
                "k": k,
                "n": n,
                "y": y,
                "delta_cv": delta_cv,
                "total_trials": total_trials,
            }
        )

    n_panels = len(panel_rows)

    def unpack_panel(theta_p: np.ndarray) -> Tuple[float, float, float, float, float, float, Dict[str, float]]:
        i = 0
        b = float(theta_p[i]); i += 1
        extras: Dict[str, float] = {}
        gr_aux = 0.0
        gt_aux = 0.0

        if model_bank == "hybrid_combined":
            gr = _pos(theta_p[i]); i += 1
            gt = _pos(theta_p[i]); i += 1
            gr_aux = _pos(theta_p[i]); i += 1
            gt_aux = _pos(theta_p[i]); i += 1
            temp = _pos(theta_p[i]); i += 1
        elif model_bank == "rate_combined":
            gr = _pos(theta_p[i]); i += 1
            gt = 0.0
            gr_aux = _pos(theta_p[i]); i += 1
            temp = _pos(theta_p[i]); i += 1
        elif model_bank == "timing_combined":
            gr = 0.0
            gt = _pos(theta_p[i]); i += 1
            gt_aux = _pos(theta_p[i]); i += 1
            temp = _pos(theta_p[i]); i += 1
        else:
            raise ValueError(f"Unknown release model_bank: {model_bank}")

        if use_rate_gate:
            gate_floor = float(sigmoid(np.array([float(theta_p[i])], dtype=float))[0]); i += 1
            gate_center = float(sigmoid(np.array([float(theta_p[i])], dtype=float))[0]); i += 1
            gate_slope = float(softplus(float(theta_p[i]))); i += 1
        else:
            gate_floor = 1.0
            gate_center = 0.5
            gate_slope = 8.0
        extras["rate_gate_floor"] = float(gate_floor)
        extras["rate_gate_center"] = float(gate_center)
        extras["rate_gate_slope"] = float(gate_slope)
        return b, gr, gt, gr_aux, gt_aux, temp, extras

    n_params = n_th + n_panels * panel_dim

    def objective(theta: np.ndarray) -> float:
        u = theta[:n_th]
        w = _entmax_bisect(u, alpha=ms.entmax_alpha)
        w_safe = np.clip(np.asarray(w, dtype=float), 1e-12, 1.0)

        panel_losses: List[float] = []
        trial_counts: List[float] = []
        sq_norm_panels = 0.0

        for p_i, p in enumerate(panel_rows):
            start = n_th + p_i * panel_dim
            stop = start + panel_dim
            theta_p = theta[start:stop]
            sq_norm_panels += float(np.sum(theta_p * theta_p))

            b, gr, gt, gr_aux, gt_aux, temp, extras = unpack_panel(theta_p)
            rate_drive = p["Er"] @ w
            timing_drive = p["Et"] @ w
            rate_drive_aux = p["Er_aux"] @ w
            timing_drive_aux = p["Et_aux"] @ w
            if use_rate_gate and p["delta_cv"] is not None:
                gate_vals = rate_gate_from_delta(
                    np.asarray(p["delta_cv"], dtype=float),
                    extras["rate_gate_floor"],
                    extras["rate_gate_center"],
                    extras["rate_gate_slope"],
                )
                rate_drive = rate_drive * gate_vals
                if model_bank == "hybrid_combined":
                    rate_drive_aux = rate_drive_aux * gate_vals
            if model_bank == "rate_combined":
                drive = gr * rate_drive + gr_aux * rate_drive_aux
            elif model_bank == "timing_combined":
                drive = gt * timing_drive + gt_aux * timing_drive_aux
            elif model_bank == "hybrid_combined":
                drive = gr * rate_drive + gt * timing_drive + gr_aux * rate_drive_aux + gt_aux * timing_drive_aux
            else:
                raise ValueError(f"Unknown release model_bank: {model_bank}")
            z = b + temp * drive
            p_choose = sigmoid(z)
            p_corr = map_pchoose_to_pcorrect_mixed(p_choose, blocks=np.asarray(p["blocks"]), cvs=np.asarray(p["cvs"], dtype=float))
            p_corr = np.clip(p_corr, 1e-6, 1 - 1e-6)
            ll = p["k"] * np.log(p_corr) + (p["n"] - p["k"]) * np.log(1 - p_corr)
            nll_main = float(-(np.sum(ll)) / float(p["total_trials"]))

            gate_reg = 0.0
            if use_rate_gate and rate_gate_reg > 0.0:
                gate_reg = rate_gate_reg * float(
                    (extras["rate_gate_floor"] - rate_gate_prior_floor) ** 2
                    + (extras["rate_gate_center"] - rate_gate_prior_center) ** 2
                    + ((extras["rate_gate_slope"] - rate_gate_prior_slope) / max(rate_gate_prior_slope, 1e-6)) ** 2
                )
            panel_losses.append(float(nll_main + gate_reg))
            trial_counts.append(float(p["total_trials"]))

        if panel_balance:
            data_loss = float(np.mean(panel_losses))
        else:
            tc = np.asarray(trial_counts, dtype=float)
            data_loss = float(np.average(np.asarray(panel_losses, dtype=float), weights=np.maximum(tc, 1e-12)))

        reg_u = float(loss_reg_l2) * float(np.sum(u * u))
        reg_panels = float(loss_reg_l2) * float(sq_norm_panels / max(1, n_panels))

        kl_reg = 0.0
        if float(weight_kl) > 0.0:
            kl_reg = float(weight_kl) * float(np.sum(w_safe * np.log(w_safe * float(n_th))))

        entropy_floor_reg = 0.0
        if weight_entropy_floor_k > 0.0 and weight_entropy_floor_lambda > 0.0:
            k_eff = effective_k(w_safe)
            entropy_floor_reg = weight_entropy_floor_lambda * float(max(0.0, weight_entropy_floor_k - k_eff) ** 2)

        cap_reg = 0.0
        if weight_max_cap < 1.0 and weight_max_cap_lambda > 0.0:
            over = np.maximum(np.asarray(w, dtype=float) - float(weight_max_cap), 0.0)
            cap_reg = float(weight_max_cap_lambda) * float(np.sum(over * over))

        return float(data_loss + reg_u + reg_panels + kl_reg + entropy_floor_reg + cap_reg)

    theta0 = np.zeros(n_params, dtype=float)
    for p_i in range(n_panels):
        start = n_th + p_i * panel_dim
        j = start + 1  # after bias
        if allow_zero_gains:
            if model_bank in {"rate_combined", "timing_combined"}:
                theta0[j] = 0.5
                theta0[j + 1] = 0.5
                theta0[j + 2] = 1.0
            else:  # hybrid_combined
                theta0[j] = 0.5
                theta0[j + 1] = 0.5
                theta0[j + 2] = 0.5
                theta0[j + 3] = 0.5
                theta0[j + 4] = 1.0
        if use_rate_gate:
            gate_i = start + (panel_dim - 3)
            theta0[gate_i] = inv_sigmoid(rate_gate_prior_floor)
            theta0[gate_i + 1] = inv_sigmoid(rate_gate_prior_center)
            theta0[gate_i + 2] = inv_softplus(rate_gate_prior_slope)

    if minimize is None:
        rng = np.random.default_rng(ms.seed)
        best = (objective(theta0), theta0.copy())
        for _ in range(6000):
            cand = theta0 + rng.normal(scale=0.35, size=theta0.size)
            v = objective(cand)
            if v < best[0]:
                best = (v, cand)
        theta_hat = best[1]
    else:
        bounds = None
        if allow_zero_gains:
            bounds = [(None, None)] * n_params
            for p_i in range(n_panels):
                start = n_th + p_i * panel_dim
                j = start + 1  # after bias
                if model_bank in {"rate_combined", "timing_combined"}:
                    bounds[j] = (0.0, None)
                    bounds[j + 1] = (0.0, None)
                    bounds[j + 2] = (0.0, None)
                else:  # hybrid_combined
                    bounds[j] = (0.0, None)
                    bounds[j + 1] = (0.0, None)
                    bounds[j + 2] = (0.0, None)
                    bounds[j + 3] = (0.0, None)
                    bounds[j + 4] = (0.0, None)
        res = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=300))
        theta_hat = res.x

    u_hat = theta_hat[:n_th]
    w_hat = _entmax_bisect(u_hat, alpha=ms.entmax_alpha)
    weights = {str(ths[i]): float(w_hat[i]) for i in range(n_th)}

    params_by_panel: Dict[Tuple[float, str], Dict[str, float]] = {}
    rmse_vals: List[float] = []
    mae_vals: List[float] = []
    for p_i, p in enumerate(panel_rows):
        start = n_th + p_i * panel_dim
        stop = start + panel_dim
        b, gr, gt, gr_aux, gt_aux, temp, extras = unpack_panel(theta_hat[start:stop])
        params = {"b": float(b), "g_rate": float(gr), "g_time": float(gt), "temp": float(temp)}
        params["g_rate_aux"] = float(gr_aux)
        params["g_time_aux"] = float(gt_aux)
        params["rate_gate_floor"] = float(extras.get("rate_gate_floor", 1.0))
        params["rate_gate_center"] = float(extras.get("rate_gate_center", 0.5))
        params["rate_gate_slope"] = float(extras.get("rate_gate_slope", 8.0))
        params_by_panel[p["key"]] = params

        met = evaluate_design_metrics(
            params=params,
            weights=weights,
            ths=ths,
            Er=np.asarray(p["Er"], dtype=float),
            Et=np.asarray(p["Et"], dtype=float),
            Er_aux=np.asarray(p["Er_aux"], dtype=float),
            Et_aux=np.asarray(p["Et_aux"], dtype=float),
            cvs=np.asarray(p["cvs"], dtype=float),
            blocks=np.asarray(p["blocks"]),
            k=np.asarray(p["k"], dtype=float),
            n=np.asarray(p["n"], dtype=float),
            y=np.asarray(p["y"], dtype=float),
            model_bank=model_bank,
            rate_gate_mode=rate_gate_mode,
        )
        rmse_vals.append(float(met["rmse"]))
        mae_vals.append(float(met["mae"]))

    metrics = {
        "nll": float(objective(theta_hat)),
        "rmse": float(np.nanmean(rmse_vals)) if rmse_vals else float("nan"),
        "mae": float(np.nanmean(mae_vals)) if mae_vals else float("nan"),
    }
    return params_by_panel, weights, metrics


def fit_per_point_hybrid_from_design(
    *,
    Er: np.ndarray,
    Et: np.ndarray,
    Er_aux: Optional[np.ndarray] = None,
    Et_aux: Optional[np.ndarray] = None,
    cvs: np.ndarray,
    blocks: np.ndarray,
    k: np.ndarray,
    n: np.ndarray,
    y: np.ndarray,
    ths: Sequence[str],
    ms: ModelSettings,
    loss_reg_l2: float = 1e-3,
    weight_kl: float = 0.0,
    allow_zero_gains: bool = False,
    rate_gate_mode: str = "delta_cv",
    rate_gate_reg: float = 0.0,
    rate_gate_prior_floor: float = 1.0,
    rate_gate_prior_center: float = 0.5,
    rate_gate_prior_slope: float = 8.0,
    weight_entropy_floor_k: float = 0.0,
    weight_entropy_floor_lambda: float = 0.0,
    weight_max_cap: float = 1.0,
    weight_max_cap_lambda: float = 0.0,
    weight_smooth_lambda: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, float]]:
    """Fit hybrid_combined with per-point threshold weights for one panel."""
    Er = np.asarray(Er, dtype=float)
    Et = np.asarray(Et, dtype=float)
    cvs = np.asarray(cvs, dtype=float)
    blocks = np.asarray(blocks).astype(str)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    if Er.ndim != 2 or Et.ndim != 2 or Er.shape != Et.shape:
        raise ValueError("Er/Et must be 2D matrices with matching shapes")
    if Er.shape[0] == 0:
        raise RuntimeError("No design rows to fit")

    n_th = len(ths)
    if Er.shape[1] != n_th:
        raise ValueError("Er/Et column count must match number of threshold types")
    if Er_aux is None:
        Er_aux_arr = np.zeros_like(Er)
    else:
        Er_aux_arr = np.asarray(Er_aux, dtype=float)
        if Er_aux_arr.shape != Er.shape:
            raise ValueError("Er_aux must match Er shape")
    if Et_aux is None:
        Et_aux_arr = np.zeros_like(Et)
    else:
        Et_aux_arr = np.asarray(Et_aux, dtype=float)
        if Et_aux_arr.shape != Et.shape:
            raise ValueError("Et_aux must match Et shape")

    if str(rate_gate_mode).strip().lower() not in {"none", "delta_cv"}:
        raise ValueError(f"Unknown rate_gate_mode: {rate_gate_mode}")

    def softplus(x: float) -> float:
        if x > 30:
            return x
        return float(np.log1p(np.exp(x)))

    def inv_sigmoid(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        return float(np.log(p) - np.log(1.0 - p))

    def inv_softplus(yv: float) -> float:
        yv = float(max(1e-9, yv))
        if yv > 30.0:
            return yv
        return float(np.log(np.expm1(yv)))

    def _pos(raw: float) -> float:
        return max(0.0, float(raw)) if allow_zero_gains else softplus(float(raw))

    sort_idx = np.argsort(cvs, kind="stable")
    inv_idx = np.empty_like(sort_idx)
    inv_idx[sort_idx] = np.arange(sort_idx.size)
    cvs_sorted = cvs[sort_idx]
    if len(np.unique(np.round(cvs_sorted, 9))) != cvs_sorted.size:
        raise ValueError("per-point mode expects one design row per CV anchor within a panel")

    delta_cv = delta_cv_from_template_mixed(cvs, blocks)

    weight_entropy_floor_k = float(max(0.0, weight_entropy_floor_k))
    weight_entropy_floor_lambda = float(max(0.0, weight_entropy_floor_lambda))
    weight_max_cap = float(np.clip(weight_max_cap, 0.0, 1.0))
    weight_max_cap_lambda = float(max(0.0, weight_max_cap_lambda))
    rate_gate_reg = float(max(0.0, rate_gate_reg))
    rate_gate_prior_floor = float(np.clip(rate_gate_prior_floor, 0.0, 1.0))
    rate_gate_prior_center = float(np.clip(rate_gate_prior_center, 0.0, 1.0))
    rate_gate_prior_slope = float(max(1e-6, rate_gate_prior_slope))
    weight_smooth_lambda = float(max(0.0, weight_smooth_lambda))

    n_weight_params = Er.shape[0] * n_th
    n_params = n_weight_params + 9  # b + 4 gains + temp + 3 gate params

    def unpack(theta: np.ndarray):
        U_sorted = theta[:n_weight_params].reshape((Er.shape[0], n_th))
        W_sorted = _entmax_rows(U_sorted, alpha=ms.entmax_alpha)
        W = W_sorted[inv_idx]
        i = n_weight_params
        b = float(theta[i]); i += 1
        gr = _pos(theta[i]); i += 1
        gt = _pos(theta[i]); i += 1
        gr_aux = _pos(theta[i]); i += 1
        gt_aux = _pos(theta[i]); i += 1
        temp = _pos(theta[i]); i += 1
        if str(rate_gate_mode).strip().lower() == "delta_cv":
            gate_floor = float(sigmoid(np.array([float(theta[i])], dtype=float))[0]); i += 1
            gate_center = float(sigmoid(np.array([float(theta[i])], dtype=float))[0]); i += 1
            gate_slope = float(softplus(float(theta[i]))); i += 1
        else:
            gate_floor = 1.0
            gate_center = 0.5
            gate_slope = 8.0
        extras = {
            "rate_gate_floor": float(gate_floor),
            "rate_gate_center": float(gate_center),
            "rate_gate_slope": float(gate_slope),
        }
        return W, W_sorted, b, gr, gt, gr_aux, gt_aux, temp, extras

    def nll(theta: np.ndarray) -> float:
        W, W_sorted, b, gr, gt, gr_aux, gt_aux, temp, extras = unpack(theta)
        rate_drive = np.sum(Er * W, axis=1)
        timing_drive = np.sum(Et * W, axis=1)
        rate_drive_aux = np.sum(Er_aux_arr * W, axis=1)
        timing_drive_aux = np.sum(Et_aux_arr * W, axis=1)
        if str(rate_gate_mode).strip().lower() == "delta_cv":
            gate_vals = rate_gate_from_delta(
                delta_cv,
                extras["rate_gate_floor"],
                extras["rate_gate_center"],
                extras["rate_gate_slope"],
            )
            rate_drive = rate_drive * gate_vals
            rate_drive_aux = rate_drive_aux * gate_vals

        drive = gr * rate_drive + gt * timing_drive + gr_aux * rate_drive_aux + gt_aux * timing_drive_aux
        z = b + temp * drive
        p_choose = sigmoid(z)
        p_corr = map_pchoose_to_pcorrect_mixed(p_choose, blocks=blocks, cvs=cvs)
        p_corr = np.clip(p_corr, 1e-6, 1 - 1e-6)
        ll = k * np.log(p_corr) + (n - k) * np.log(1 - p_corr)
        total_trials = float(np.nansum(n))
        if not np.isfinite(total_trials) or total_trials <= 0.0:
            total_trials = 1.0
        nll_main = float(-(np.sum(ll)) / total_trials)

        reg = float(loss_reg_l2) * float(np.sum(theta * theta))
        kl_reg = 0.0
        entropy_floor_reg = 0.0
        cap_reg = 0.0
        for w_row in np.asarray(W_sorted, dtype=float):
            w_safe = np.clip(np.asarray(w_row, dtype=float), 1e-12, 1.0)
            if float(weight_kl) > 0.0:
                kl_reg += float(weight_kl) * float(np.sum(w_safe * np.log(w_safe * float(n_th))))
            if weight_entropy_floor_k > 0.0 and weight_entropy_floor_lambda > 0.0:
                k_eff = effective_k(w_safe)
                entropy_floor_reg += weight_entropy_floor_lambda * float(max(0.0, weight_entropy_floor_k - k_eff) ** 2)
            if weight_max_cap < 1.0 and weight_max_cap_lambda > 0.0:
                over = np.maximum(np.asarray(w_row, dtype=float) - float(weight_max_cap), 0.0)
                cap_reg += float(weight_max_cap_lambda) * float(np.sum(over * over))
        if W_sorted.shape[0] > 0:
            kl_reg /= float(W_sorted.shape[0])
            entropy_floor_reg /= float(W_sorted.shape[0])
            cap_reg /= float(W_sorted.shape[0])

        smooth_reg = 0.0
        if weight_smooth_lambda > 0.0 and W_sorted.shape[0] > 1:
            diffs = np.diff(np.asarray(W_sorted, dtype=float), axis=0)
            smooth_reg = float(weight_smooth_lambda) * float(np.mean(np.sum(diffs * diffs, axis=1)))

        gate_reg = 0.0
        if str(rate_gate_mode).strip().lower() == "delta_cv" and rate_gate_reg > 0.0:
            gate_reg = rate_gate_reg * float(
                (extras["rate_gate_floor"] - rate_gate_prior_floor) ** 2
                + (extras["rate_gate_center"] - rate_gate_prior_center) ** 2
                + ((extras["rate_gate_slope"] - rate_gate_prior_slope) / max(rate_gate_prior_slope, 1e-6)) ** 2
            )
        return float(nll_main + reg + kl_reg + entropy_floor_reg + cap_reg + smooth_reg + gate_reg)

    theta0 = np.zeros(n_params, dtype=float)
    gate_i = n_weight_params + 6
    if allow_zero_gains:
        theta0[n_weight_params + 1] = 0.5
        theta0[n_weight_params + 2] = 0.5
        theta0[n_weight_params + 3] = 0.5
        theta0[n_weight_params + 4] = 0.5
        theta0[n_weight_params + 5] = 1.0
    if str(rate_gate_mode).strip().lower() == "delta_cv":
        theta0[gate_i] = inv_sigmoid(rate_gate_prior_floor)
        theta0[gate_i + 1] = inv_sigmoid(rate_gate_prior_center)
        theta0[gate_i + 2] = inv_softplus(rate_gate_prior_slope)

    if minimize is None:
        rng = np.random.default_rng(ms.seed)
        best = (nll(theta0), theta0.copy())
        for _ in range(6000):
            cand = theta0 + rng.normal(scale=0.25, size=theta0.size)
            v = nll(cand)
            if v < best[0]:
                best = (v, cand)
        theta_hat = best[1]
    else:
        bounds = None
        if allow_zero_gains:
            bounds = [(None, None)] * n_params
            bounds[n_weight_params + 1] = (0.0, None)
            bounds[n_weight_params + 2] = (0.0, None)
            bounds[n_weight_params + 3] = (0.0, None)
            bounds[n_weight_params + 4] = (0.0, None)
            bounds[n_weight_params + 5] = (0.0, None)
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=400))
        theta_hat = res.x

    W, W_sorted, b, gr, gt, gr_aux, gt_aux, temp, extras = unpack(theta_hat)
    params = {
        "b": float(b),
        "g_rate": float(gr),
        "g_time": float(gt),
        "g_rate_aux": float(gr_aux),
        "g_time_aux": float(gt_aux),
        "temp": float(temp),
        "rate_gate_floor": float(extras.get("rate_gate_floor", 1.0)),
        "rate_gate_center": float(extras.get("rate_gate_center", 0.5)),
        "rate_gate_slope": float(extras.get("rate_gate_slope", 8.0)),
    }
    weights_mean = {str(ths[j]): float(np.mean(W[:, j])) for j in range(n_th)}
    weights_by_cv = {
        _format_cv_key(cvs_sorted[i]): {str(ths[j]): float(W_sorted[i, j]) for j in range(n_th)}
        for i in range(cvs_sorted.size)
    }
    metrics = evaluate_design_metrics_rowwise(
        params=params,
        weights_by_cv=weights_by_cv,
        ths=ths,
        Er=Er,
        Et=Et,
        Er_aux=Er_aux_arr,
        Et_aux=Et_aux_arr,
        cvs=cvs,
        blocks=blocks,
        k=k,
        n=n,
        y=y,
        model_bank="hybrid_combined",
        rate_gate_mode=rate_gate_mode,
    )
    metrics["nll"] = float(nll(theta_hat))
    return params, weights_mean, weights_by_cv, metrics

# ============================
# Orchestration
# ============================

def _split_reps_shared(reps: Sequence[int], train_frac: float, rng: np.random.Generator) -> Tuple[List[int], List[int]]:
    reps = list(sorted(set(int(r) for r in reps)))
    rng.shuffle(reps)
    n_train = int(round(train_frac * len(reps)))
    n_train = max(1, min(len(reps) - 1, n_train)) if len(reps) >= 2 else len(reps)
    train = reps[:n_train]
    test = reps[n_train:]
    return train, test


def _sem_series(x: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
    xv = xv[np.isfinite(xv)]
    if xv.size <= 1:
        return 0.0
    return float(np.std(xv, ddof=1) / np.sqrt(xv.size))


def _parse_weight_dict_cell(x) -> Dict[str, float]:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                out[str(k)] = fv
        return out
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return {}
    s = str(x).strip()
    if not s or s.lower() in {"none", "null", "nan", "{}"}:
        return {}
    try:
        obj = json.loads(s)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out[str(k)] = fv
    return out


def _json_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_mc_summaries(df_res: pd.DataFrame, out_dir: str) -> None:
    if df_res is None or df_res.empty:
        return
    os.makedirs(out_dir, exist_ok=True)

    key_cols = [
        c
        for c in [
            "scope",
            "scope_key",
            "model_bank",
            "freq",
            "block",
            "bin_ms",
            "aux_bin_ms",
            "drop_thresh",
            "is_loto",
            "fixed_thresh",
            "is_fixed_thresh",
            "n_thresh",
            "loss_reg_l2",
            "weight_kl",
            "allow_zero_gains",
            "rate_gate_mode",
            "rate_gate_reg",
            "rate_gate_prior_floor",
            "rate_gate_prior_center",
            "rate_gate_prior_slope",
            "shared_thresh_mode",
            "thresh_flex_mode",
            "weight_entropy_floor_k",
            "weight_entropy_floor_lambda",
            "weight_max_cap",
            "weight_max_cap_lambda",
            "weight_smooth_lambda",
        ]
        if c in df_res.columns
    ]
    if not key_cols:
        return

    work = df_res.copy()
    for c in [
        "rmse_train",
        "mae_train",
        "nll_train",
        "rmse_test",
        "mae_test",
        "gain_rate",
        "gain_timing",
        "bias",
        "temp",
        "gain_rate_aux",
        "gain_timing_aux",
        "rate_gate_floor",
        "rate_gate_center",
        "rate_gate_slope",
        "mc",
    ]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    agg_spec = {}
    for c in [
        "rmse_train",
        "mae_train",
        "nll_train",
        "rmse_test",
        "mae_test",
        "gain_rate",
        "gain_timing",
        "bias",
        "temp",
        "gain_rate_aux",
        "gain_timing_aux",
        "rate_gate_floor",
        "rate_gate_center",
        "rate_gate_slope",
    ]:
        if c in work.columns:
            agg_spec[f"{c}_mean"] = (c, "mean")
            agg_spec[f"{c}_sem"] = (c, _sem_series)

    summary = work.groupby(key_cols, as_index=False, dropna=False).agg(
        n_rows=("mc", "size"),
        n_mc=("mc", lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique()) if len(s) else 0),
        **agg_spec,
    )
    summary.to_csv(os.path.join(out_dir, "results_mc_summary.csv"), index=False)

    weight_records: List[Dict[str, object]] = []
    for _, row in work.iterrows():
        mb = str(row.get("model_bank", "")).strip().lower()
        w = _parse_weight_dict_cell(row.get("weights_json"))
        if not w:
            if mb == "rate_combined":
                w = _parse_weight_dict_cell(row.get("weights_rate_json"))
            elif mb == "timing_combined":
                w = _parse_weight_dict_cell(row.get("weights_timing_json"))
            else:
                w = _parse_weight_dict_cell(row.get("weights_shared_json"))
        if not w:
            continue
        rec_base = {k: row[k] for k in key_cols if k in row.index}
        rec_base["mc"] = row.get("mc", np.nan)
        for th, wt in w.items():
            rec = dict(rec_base)
            rec["thresh_type"] = str(th)
            rec["weight"] = float(wt)
            weight_records.append(rec)

    if weight_records:
        wdf = pd.DataFrame(weight_records)
        wsum = wdf.groupby(key_cols + ["thresh_type"], as_index=False, dropna=False).agg(
            weight_mean=("weight", "mean"),
            weight_sem=("weight", _sem_series),
            n_rows=("weight", "size"),
            n_mc=("mc", lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique()) if len(s) else 0),
        )
        wsum.to_csv(os.path.join(out_dir, "weights_mc_summary.csv"), index=False)

        jrec: List[Dict[str, object]] = []
        for gk, g in wsum.groupby(key_cols, dropna=False):
            if not isinstance(gk, tuple):
                gk = (gk,)
            d = {k: _json_scalar(v) for k, v in zip(key_cols, gk)}
            d["n_mc"] = int(pd.to_numeric(g["n_mc"], errors="coerce").max()) if not g.empty else 0
            d["weights_mean"] = {str(r["thresh_type"]): float(r["weight_mean"]) for _, r in g.iterrows()}
            d["weights_sem"] = {str(r["thresh_type"]): float(r["weight_sem"]) for _, r in g.iterrows()}
            gg = summary
            for k in key_cols:
                if pd.isna(d[k]):
                    gg = gg[gg[k].isna()]
                else:
                    gg = gg[gg[k] == d[k]]
            if not gg.empty:
                row_s = gg.iloc[0]
                param_fields = [
                    "gain_rate",
                    "gain_timing",
                    "bias",
                    "temp",
                    "gain_rate_aux",
                    "gain_timing_aux",
                    "rate_gate_floor",
                    "rate_gate_center",
                    "rate_gate_slope",
                ]
                d["params_mean"] = {
                    k: float(row_s.get(f"{k}_mean", np.nan))
                    for k in param_fields
                    if f"{k}_mean" in row_s.index
                }
                d["params_sem"] = {
                    k: float(row_s.get(f"{k}_sem", np.nan))
                    for k in param_fields
                    if f"{k}_sem" in row_s.index
                }
            jrec.append(d)
        with open(os.path.join(out_dir, "params_mc_summary.jsonl"), "w", encoding="utf-8") as f:
            for r in jrec:
                f.write(json.dumps(r) + "\n")


def _parse_json_float_list(x) -> List[float]:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return [float(v) for v in arr.tolist()]
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "null", "[]"}:
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: List[float] = []
    for v in obj:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(float(fv))
    return out


def write_curve_summaries(df_curves: pd.DataFrame, out_dir: str) -> None:
    if df_curves is None or df_curves.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    p_curves = os.path.join(out_dir, "curves.csv")
    df_curves.to_csv(p_curves, index=False)

    key_cols = [
        c
        for c in [
            "scope",
            "scope_key",
            "model_bank",
            "freq",
            "block",
            "bin_ms",
            "aux_bin_ms",
            "drop_thresh",
            "is_loto",
            "fixed_thresh",
            "is_fixed_thresh",
            "loss_reg_l2",
            "weight_kl",
            "allow_zero_gains",
            "rate_gate_mode",
            "rate_gate_reg",
            "rate_gate_prior_floor",
            "rate_gate_prior_center",
            "rate_gate_prior_slope",
            "shared_thresh_mode",
            "thresh_flex_mode",
            "weight_entropy_floor_k",
            "weight_entropy_floor_lambda",
            "weight_max_cap",
            "weight_max_cap_lambda",
            "weight_smooth_lambda",
        ]
        if c in df_curves.columns
    ]
    if not key_cols:
        return

    metric_fields = [
        ("p_obs_json", "p_obs"),
        ("p_pred_train_json", "p_pred_train"),
        ("p_pred_test_json", "p_pred_test"),
        ("rate_drive_train_json", "rate_drive_train"),
        ("timing_drive_train_json", "timing_drive_train"),
        ("rate_drive_test_json", "rate_drive_test"),
        ("timing_drive_test_json", "timing_drive_test"),
        ("rate_drive_aux_train_json", "rate_drive_aux_train"),
        ("timing_drive_aux_train_json", "timing_drive_aux_train"),
        ("rate_drive_aux_test_json", "rate_drive_aux_test"),
        ("timing_drive_aux_test_json", "timing_drive_aux_test"),
        ("gate_train_json", "gate_train"),
        ("gate_test_json", "gate_test"),
    ]

    long_rows: List[Dict[str, object]] = []
    for _, row in df_curves.iterrows():
        cv_vals = _parse_json_float_list(row.get("cv_json"))
        if not cv_vals:
            continue
        payload: Dict[str, List[float]] = {}
        for col, name in metric_fields:
            vals = _parse_json_float_list(row.get(col))
            if len(vals) == len(cv_vals):
                payload[name] = vals
        if not payload:
            continue
        base = {k: row[k] for k in key_cols if k in row.index}
        for i, cv in enumerate(cv_vals):
            rec = dict(base)
            rec["cv"] = float(cv)
            for name, vals in payload.items():
                rec[name] = float(vals[i])
            long_rows.append(rec)

    if not long_rows:
        return
    ldf = pd.DataFrame(long_rows)

    agg_map = {}
    for c in [
        "p_obs",
        "p_pred_train",
        "p_pred_test",
        "rate_drive_train",
        "timing_drive_train",
        "rate_drive_test",
        "timing_drive_test",
        "rate_drive_aux_train",
        "timing_drive_aux_train",
        "rate_drive_aux_test",
        "timing_drive_aux_test",
        "gate_train",
        "gate_test",
    ]:
        if c in ldf.columns:
            agg_map[f"{c}_mean"] = (c, "mean")
            agg_map[f"{c}_sem"] = (c, _sem_series)

    if not agg_map:
        return
    summary = ldf.groupby(key_cols + ["cv"], as_index=False, dropna=False).agg(
        n_rows=("cv", "size"),
        **agg_map,
    )
    summary.to_csv(os.path.join(out_dir, "curves_mc_summary.csv"), index=False)


def run_fit(
    sim_csv: str,
    slack_dir: str,
    out_dir: str,
    *,
    bins_ms: Sequence[float] = (1.0, 10.0),
    scope: str = "per_freq_block",
    n_mc: int = 25,
    train_frac: float = 0.9,
    seed: int = 0,
    jnd_k: float = 2.5,
    entmax_alpha: float = 1.1,
    loss_reg_l2: float = 1e-3,
    weight_kl: float = 0.0,
    allow_zero_gains: bool = False,
    rate_gate_mode: str = "none",
    rate_gate_reg: float = 0.0,
    rate_gate_prior_floor: float = 1.0,
    rate_gate_prior_center: float = 0.5,
    rate_gate_prior_slope: float = 8.0,
    weight_entropy_floor_k: float = 0.0,
    weight_entropy_floor_lambda: float = 0.0,
    weight_max_cap: float = 1.0,
    weight_max_cap_lambda: float = 0.0,
    do_loto: bool = False,
    loto_include_full: bool = True,
    include_templates_for_slack: bool = True,
    drop_thresh: Optional[str] = None,
    fixed_thresh: Optional[str] = None,
    scope_key: Optional[str] = None,
    mc_start: int = 0,
    mc_count: Optional[int] = None,
    model_banks: Sequence[str] = ("rate_combined", "timing_combined", "hybrid_combined"),
    shared_thresh_mode: str = "none",
    thresh_flex_mode: str = "shared",
    weight_smooth_lambda: float = 0.0,
) -> Tuple[pd.DataFrame, List[FitResult]]:
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    sim = normalize_threshold_frame(pd.read_csv(sim_csv))
    spikes_col, rep_col = infer_sim_columns(sim)
    sim = sim.rename(columns={rep_col: "rep"})
    sim["freq"] = pd.to_numeric(sim["freq"], errors="coerce")
    sim["cv"] = pd.to_numeric(sim["cv"], errors="coerce")
    sim["rep"] = pd.to_numeric(sim["rep"], errors="coerce").astype(int)
    sim["thresh_type"] = sim["thresh_type"].map(canonical_threshold_type)
    sim = sim[np.isfinite(sim["freq"]) & np.isfinite(sim["cv"])].copy()
    if DEFAULT_FREQS:
        sim = sim[sim["freq"].isin([float(f) for f in DEFAULT_FREQS])].copy()

    slack_trials, slack_mean = load_slack_dir(slack_dir, include_templates=include_templates_for_slack)
    if DEFAULT_FREQS:
        slack_trials = slack_trials[slack_trials["freq"].isin([float(f) for f in DEFAULT_FREQS])].copy()
        slack_mean = slack_mean[slack_mean["freq"].isin([float(f) for f in DEFAULT_FREQS])].copy()

    freqs = sorted(sim["freq"].unique().tolist())
    blocks = sorted(slack_mean["block"].unique().tolist())
    drop_thresh_name = None
    if drop_thresh is not None:
        drop_thresh_name = str(drop_thresh).strip()
        if drop_thresh_name.upper() in {"", "NONE", "NULL"}:
            drop_thresh_name = None
    fixed_thresh_name = None
    if fixed_thresh is not None:
        fixed_thresh_name = str(fixed_thresh).strip()
        if fixed_thresh_name.upper() in {"", "NONE", "NULL"}:
            fixed_thresh_name = None
    if fixed_thresh_name is not None and drop_thresh_name is not None:
        raise ValueError("--fixed_thresh and --drop_thresh are mutually exclusive")
    if fixed_thresh_name is not None and bool(do_loto):
        raise ValueError("--fixed_thresh cannot be combined with --do_loto")

    model_banks_use: List[str] = []
    release_model_banks = {"rate_combined", "timing_combined", "hybrid_combined"}
    for mb in model_banks:
        mb_s = str(mb).strip().lower()
        if mb_s in release_model_banks and mb_s not in model_banks_use:
            model_banks_use.append(mb_s)
    if not model_banks_use:
        raise ValueError(
            "model_banks resolved to empty set; expected subset of "
            "{rate_combined,timing_combined,hybrid_combined}"
        )
    shared_thresh_mode = str(shared_thresh_mode).strip().lower()
    if shared_thresh_mode not in {"none", "joint_panel_params"}:
        raise ValueError("shared_thresh_mode must be one of: none, joint_panel_params")
    if shared_thresh_mode == "joint_panel_params" and scope == "per_freq_block":
        print("[warn] shared_thresh_mode=joint_panel_params with scope=per_freq_block has no cross-panel sharing (groups are single-panel).", flush=True)
    thresh_flex_mode = str(thresh_flex_mode).strip().lower()
    if thresh_flex_mode not in {"shared", "per_point"}:
        raise ValueError("thresh_flex_mode must be one of: shared, per_point")
    weight_smooth_lambda = float(max(0.0, weight_smooth_lambda))
    scope = str(scope).strip().lower()
    if scope not in {"per_block", "per_freq_block"}:
        raise ValueError("Release run_fit supports scope='per_block' or scope='per_freq_block'")
    if thresh_flex_mode == "per_point":
        unsupported = [mb for mb in model_banks_use if mb != "hybrid_combined"]
        if unsupported:
            raise ValueError("thresh_flex_mode=per_point currently supports hybrid_combined only")
        if scope != "per_freq_block":
            raise ValueError("thresh_flex_mode=per_point requires scope=per_freq_block")
        if shared_thresh_mode != "none":
            raise ValueError("thresh_flex_mode=per_point requires shared_thresh_mode=none")

    # CV grid from simulation (ensure includes 0 and 1)
    cvs_all = sorted(set(float(x) for x in sim["cv"].unique().tolist()))
    if 0.0 not in cvs_all or 1.0 not in cvs_all:
        raise RuntimeError("Simulation must include cv=0 and cv=1 for endpoint normalization")

    # Slack evaluation CVs: use the union of slack_mean cv per block.
    slack_cvs_by_block: Dict[str, List[float]] = {}
    for b in blocks:
        c = sorted(set(float(x) for x in slack_mean[slack_mean["block"] == b]["cv"].unique().tolist()))
        slack_cvs_by_block[b] = c

    def _iter_scope_keys():
        if scope == "per_block":
            for b in blocks:
                yield (f"block={b}", None, str(b))
        elif scope == "per_freq_block":
            for f in freqs:
                for b in blocks:
                    yield (f"freq={int(f)}|block={b}", float(f), str(b))

    scope_items = list(_iter_scope_keys())
    if scope_key is not None and str(scope_key).strip():
        wanted = {tok.strip() for tok in str(scope_key).split(",") if tok.strip()}
        scope_items = [x for x in scope_items if x[0] in wanted]
        if not scope_items:
            raise RuntimeError(f"No scope keys matched --scope_key={scope_key!r}")

    mc_count_eff = int(n_mc) if mc_count is None else int(mc_count)
    if mc_count_eff <= 0:
        raise ValueError("mc_count must be > 0")
    mc_start_i = int(mc_start)
    mc_indices = list(range(mc_start_i, mc_start_i + mc_count_eff))

    bins_ms_use = [float(b) for b in bins_ms]
    has_bin1 = any(np.isclose(b, 1.0) for b in bins_ms_use)
    has_bin10 = any(np.isclose(b, 10.0) for b in bins_ms_use)
    combined_models_requested = any(_is_combined_model_bank(m) for m in model_banks_use)
    if combined_models_requested and (not has_bin1 or not has_bin10):
        raise ValueError("Combined model banks require bins_ms to include both 1 and 10 ms")
    combined_primary_bin_ms = 10.0 if has_bin10 else (bins_ms_use[-1] if bins_ms_use else 10.0)
    combined_aux_bin_ms = 1.0 if has_bin1 else (bins_ms_use[0] if bins_ms_use else 1.0)

    results: List[FitResult] = []
    rows_results: List[dict] = []
    rows_curves: List[dict] = []
    params_path = os.path.join(out_dir, "params.jsonl")
    with open(params_path, "w", encoding="utf-8") as fp_params:

        for bin_ms in bins_ms:
            bin_ms = float(bin_ms)
            ms = ModelSettings(
                jnd_k=float(jnd_k),
                entmax_alpha=float(entmax_alpha),
                scope=str(scope),
                train_frac=float(train_frac),
                n_mc=int(mc_count_eff),
                seed=int(seed),
                do_loto=bool(do_loto),
                loto_include_full=bool(loto_include_full),
            )
            fs = FeatureSettings(bin_ms=float(bin_ms))
            bin_mode = _bin_rate_mode(float(bin_ms))
            run_combined_this_bin = bool(combined_models_requested and np.isclose(float(bin_ms), float(combined_primary_bin_ms)))
            model_banks_for_bin = [
                mb for mb in model_banks_use
                if (not _is_combined_model_bank(mb)) or run_combined_this_bin
            ]
            if not model_banks_for_bin:
                continue

            for scope_key_i, f_scope, b_scope in scope_items:
                scope_freqs = freqs if f_scope is None else [float(f_scope)]

                # Require thresholds/reps to be available at every frequency in this scope.
                ths_scope_set = None
                reps_scope_set = None
                for f_u in scope_freqs:
                    sim_f = sim[np.isclose(sim["freq"], float(f_u))]
                    if sim_f.empty:
                        ths_scope_set = set()
                        reps_scope_set = set()
                        break
                    th_f = set(sim_f["thresh_type"].astype(str).unique().tolist())
                    rep_f = set(int(x) for x in sim_f["rep"].astype(int).unique().tolist())
                    ths_scope_set = th_f if ths_scope_set is None else (ths_scope_set & th_f)
                    reps_scope_set = rep_f if reps_scope_set is None else (reps_scope_set & rep_f)

                ths_scope = sorted(ths_scope_set) if ths_scope_set else []
                reps_scope = sorted(reps_scope_set) if reps_scope_set else []
                if not ths_scope or len(reps_scope) < 2:
                    print(f"[warn] insufficient threshold/rep coverage for scope_key={scope_key_i}; skipping", flush=True)
                    continue
                if fixed_thresh_name is not None and fixed_thresh_name not in ths_scope:
                    print(
                        f"[warn] fixed_thresh={fixed_thresh_name} not present in scope_key={scope_key_i} "
                        f"(available={ths_scope}); skipping",
                        flush=True,
                    )
                    continue

                # LOTO loop
                loto_list: List[Optional[str]] = [None]
                if fixed_thresh_name is not None:
                    loto_list = [None]
                elif drop_thresh_name is not None:
                    if drop_thresh_name not in ths_scope:
                        print(
                            f"[warn] drop_thresh={drop_thresh_name} not present in scope_key={scope_key_i} "
                            f"(available={ths_scope}); skipping",
                            flush=True,
                        )
                        continue
                    loto_list = [drop_thresh_name]
                elif do_loto:
                    loto_list = ([None] + ths_scope) if loto_include_full else list(ths_scope)

                # scope -> evaluation combinations (freq, block)
                if b_scope is not None and f_scope is not None:
                    eval_combos_all = [(float(f_scope), str(b_scope))]
                elif b_scope is not None:
                    # per_block: pooled fit over all frequencies, restricted to one block
                    eval_combos_all = [(float(f_u), str(b_scope)) for f_u in scope_freqs]
                else:
                    raise RuntimeError(f"Unexpected release scope without block/frequency: {scope_key_i}")

                for loto_th in loto_list:
                    if fixed_thresh_name is not None:
                        ths_use = [str(fixed_thresh_name)]
                    else:
                        ths_use = [t for t in ths_scope if (loto_th is None or t != loto_th)]
                    if not ths_use:
                        continue

                    eval_combos = []
                    for f_fit, b_fit in eval_combos_all:
                        cvs_eval = list(slack_cvs_by_block.get(str(b_fit), []))
                        if not cvs_eval:
                            continue
                        slack_sub = slack_mean[
                            (np.isclose(slack_mean["freq"], float(f_fit)))
                            & (slack_mean["block"] == str(b_fit))
                            & (slack_mean["cv"].isin(cvs_eval))
                        ]
                        if slack_sub.empty:
                            print(f"[warn] no Slack rows for freq={float(f_fit):g}, block={b_fit}; skipping", flush=True)
                            continue
                        eval_combos.append((float(f_fit), str(b_fit)))

                    if not eval_combos:
                        print(f"[warn] no evaluable freq/block combos for scope_key={scope_key_i}; skipping", flush=True)
                        continue

                    i0 = cvs_all.index(0.0)
                    i1 = cvs_all.index(1.0)
                    cv_grid = [float(c) for c in cvs_all]
                    cv_to_i = {float(c): i for i, c in enumerate(cvs_all)}

                    for mc in mc_indices:
                        seed_i = (int(seed) * 1000003 + int(mc) * 9176 + 12345) % (2**32 - 1)
                        rng = np.random.default_rng(seed_i)
                        reps_train, reps_test = _split_reps_shared(reps_scope, train_frac=train_frac, rng=rng)

                        # Build normalized evidence curves once per frequency for this MC split.
                        curves_by_freq: Dict[float, Dict[str, Dict[str, np.ndarray]]] = {}
                        for f_fit in scope_freqs:
                            sim_f = sim[np.isclose(sim["freq"], float(f_fit))]
                            mean_er_tr, mean_et_tr, rel_tr = estimate_evidence_curves(
                                sim_f,
                                freq=float(f_fit),
                                thresh_types=ths_use,
                                cvs=cvs_all,
                                reps=reps_train,
                                fs=fs,
                                ms=ms,
                                bin_rate_mode=bin_mode,
                            )
                            mean_er_te, mean_et_te, rel_te = estimate_evidence_curves(
                                sim_f,
                                freq=float(f_fit),
                                thresh_types=ths_use,
                                cvs=cvs_all,
                                reps=reps_test,
                                fs=fs,
                                ms=ms,
                                bin_rate_mode=bin_mode,
                            )

                            er_tr_pm1: Dict[str, np.ndarray] = {}
                            er_te_pm1: Dict[str, np.ndarray] = {}
                            et_tr_n = normalize_curves_endpoint_pm1(mean_et_tr, cvs_all)
                            et_te_n: Dict[str, np.ndarray] = {}

                            for th in ths_use:
                                # RATE pm1: TRAIN endpoints transform both train and test.
                                lo_r = float(mean_er_tr[th][i0])
                                hi_r = float(mean_er_tr[th][i1])
                                tr_rate_dict = {cv: float(mean_er_tr[th][k]) for k, cv in enumerate(cv_grid)}
                                te_rate_dict = {cv: float(mean_er_te[th][k]) for k, cv in enumerate(cv_grid)}
                                tr_rate_pm1_dict = endpoint_pm1_trace(tr_rate_dict, lo_r, hi_r)
                                te_rate_pm1_dict = endpoint_pm1_trace(te_rate_dict, lo_r, hi_r)
                                er_tr_pm1[th] = np.asarray([tr_rate_pm1_dict[cv] for cv in cv_grid], dtype=float)
                                er_te_pm1[th] = np.asarray([te_rate_pm1_dict[cv] for cv in cv_grid], dtype=float)

                                lo_t = float(mean_et_tr[th][i0])
                                hi_t = float(mean_et_tr[th][i1])
                                et_te_n[th] = endpoint_pm1_normalize(mean_et_te[th], y0=lo_t, y1=hi_t)

                            er_tr_n: Dict[str, np.ndarray] = {}
                            er_te_n: Dict[str, np.ndarray] = {}
                            for th in ths_use:
                                rr_tr = np.asarray(rel_tr.get(th, np.ones(len(cvs_all), dtype=float)), dtype=float)
                                rr_te = np.asarray(rel_te.get(th, np.ones(len(cvs_all), dtype=float)), dtype=float)
                                er_tr_n[th] = er_tr_pm1[th] * rr_tr
                                er_te_n[th] = er_te_pm1[th] * rr_te

                                den = float(mean_er_tr[th][i1]) - float(mean_er_tr[th][i0])
                                if np.isfinite(den) and abs(den) >= 1e-12:
                                    assert np.allclose([er_tr_pm1[th][i0], er_tr_pm1[th][i1]], [-1.0, 1.0], atol=1e-6, rtol=0.0)
                                    if np.isfinite(rr_tr[i0]) and np.isfinite(rr_tr[i1]):
                                        assert np.allclose(
                                            [er_tr_n[th][i0], er_tr_n[th][i1]],
                                            [-float(rr_tr[i0]), float(rr_tr[i1])],
                                            atol=1e-6,
                                            rtol=0.0,
                                        )

                            curves_by_freq[float(f_fit)] = {
                                "er_tr_n": er_tr_n,
                                "er_te_n": er_te_n,
                                "et_tr_n": et_tr_n,
                                "et_te_n": et_te_n,
                            }

                        curves_aux_by_freq: Dict[float, Dict[str, Dict[str, np.ndarray]]] = {}
                        if run_combined_this_bin:
                            fs_aux = FeatureSettings(bin_ms=float(combined_aux_bin_ms))
                            bin_mode_aux = _bin_rate_mode(float(combined_aux_bin_ms))
                            for f_fit in scope_freqs:
                                sim_f = sim[np.isclose(sim["freq"], float(f_fit))]
                                mean_er_tr, mean_et_tr, rel_tr = estimate_evidence_curves(
                                    sim_f,
                                    freq=float(f_fit),
                                    thresh_types=ths_use,
                                    cvs=cvs_all,
                                    reps=reps_train,
                                    fs=fs_aux,
                                    ms=ms,
                                    bin_rate_mode=bin_mode_aux,
                                )
                                mean_er_te, mean_et_te, rel_te = estimate_evidence_curves(
                                    sim_f,
                                    freq=float(f_fit),
                                    thresh_types=ths_use,
                                    cvs=cvs_all,
                                    reps=reps_test,
                                    fs=fs_aux,
                                    ms=ms,
                                    bin_rate_mode=bin_mode_aux,
                                )
                                er_tr_pm1: Dict[str, np.ndarray] = {}
                                er_te_pm1: Dict[str, np.ndarray] = {}
                                et_tr_n = normalize_curves_endpoint_pm1(mean_et_tr, cvs_all)
                                et_te_n: Dict[str, np.ndarray] = {}
                                for th in ths_use:
                                    lo_r = float(mean_er_tr[th][i0])
                                    hi_r = float(mean_er_tr[th][i1])
                                    tr_rate_dict = {cv: float(mean_er_tr[th][k]) for k, cv in enumerate(cv_grid)}
                                    te_rate_dict = {cv: float(mean_er_te[th][k]) for k, cv in enumerate(cv_grid)}
                                    tr_rate_pm1_dict = endpoint_pm1_trace(tr_rate_dict, lo_r, hi_r)
                                    te_rate_pm1_dict = endpoint_pm1_trace(te_rate_dict, lo_r, hi_r)
                                    er_tr_pm1[th] = np.asarray([tr_rate_pm1_dict[cv] for cv in cv_grid], dtype=float)
                                    er_te_pm1[th] = np.asarray([te_rate_pm1_dict[cv] for cv in cv_grid], dtype=float)
                                    lo_t = float(mean_et_tr[th][i0])
                                    hi_t = float(mean_et_tr[th][i1])
                                    et_te_n[th] = endpoint_pm1_normalize(mean_et_te[th], y0=lo_t, y1=hi_t)
                                er_tr_n: Dict[str, np.ndarray] = {}
                                er_te_n: Dict[str, np.ndarray] = {}
                                for th in ths_use:
                                    rr_tr = np.asarray(rel_tr.get(th, np.ones(len(cvs_all), dtype=float)), dtype=float)
                                    rr_te = np.asarray(rel_te.get(th, np.ones(len(cvs_all), dtype=float)), dtype=float)
                                    er_tr_n[th] = er_tr_pm1[th] * rr_tr
                                    er_te_n[th] = er_te_pm1[th] * rr_te
                                curves_aux_by_freq[float(f_fit)] = {
                                    "er_tr_n": er_tr_n,
                                    "er_te_n": er_te_n,
                                    "et_tr_n": et_tr_n,
                                    "et_te_n": et_te_n,
                                }

                        def _build_design_one(freq_block: Tuple[float, str], split: str) -> Optional[Dict[str, np.ndarray]]:
                            f_fit, b_fit = freq_block
                            cvs_eval_fit = list(slack_cvs_by_block.get(str(b_fit), []))
                            if not cvs_eval_fit:
                                return None
                            cvs_eval_arr = np.asarray(cvs_eval_fit, dtype=float)
                            df_t = slack_mean[
                                (np.isclose(slack_mean["freq"], float(f_fit)))
                                & (slack_mean["block"] == str(b_fit))
                                & (slack_mean["cv"].isin(cvs_eval_fit))
                            ].copy()
                            if df_t.empty:
                                return None
                            df_t = df_t.set_index("cv").reindex(cvs_eval_arr)
                            y = df_t["p_mean"].to_numpy(float)
                            k = df_t["k"].to_numpy(float)
                            n = df_t["n"].to_numpy(float)
                            m = np.isfinite(y) & np.isfinite(k) & np.isfinite(n)
                            if not np.any(m):
                                return None
                            cvs_use = cvs_eval_arr[m]
                            y = y[m]
                            k = k[m]
                            n = n[m]

                            idxs = np.asarray([cv_to_i[float(c)] for c in cvs_use], dtype=int)
                            bank = curves_by_freq[float(f_fit)]
                            er_bank = bank["er_tr_n"] if split == "train" else bank["er_te_n"]
                            et_bank = bank["et_tr_n"] if split == "train" else bank["et_te_n"]
                            Er = np.stack([np.asarray(er_bank[t], dtype=float)[idxs] for t in ths_use], axis=1)
                            Et = np.stack([np.asarray(et_bank[t], dtype=float)[idxs] for t in ths_use], axis=1)
                            out = {
                                "Er": Er,
                                "Et": Et,
                                "cvs": cvs_use.astype(float),
                                "blocks": np.asarray([str(b_fit)] * len(cvs_use), dtype=object),
                                "k": k.astype(float),
                                "n": n.astype(float),
                                "y": y.astype(float),
                            }
                            if run_combined_this_bin and float(f_fit) in curves_aux_by_freq:
                                bank_aux = curves_aux_by_freq[float(f_fit)]
                                er_aux_bank = bank_aux["er_tr_n"] if split == "train" else bank_aux["er_te_n"]
                                et_aux_bank = bank_aux["et_tr_n"] if split == "train" else bank_aux["et_te_n"]
                                out["Er_aux"] = np.stack([np.asarray(er_aux_bank[t], dtype=float)[idxs] for t in ths_use], axis=1)
                                out["Et_aux"] = np.stack([np.asarray(et_aux_bank[t], dtype=float)[idxs] for t in ths_use], axis=1)
                            return out

                        def _concat_design(parts: Sequence[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
                            parts = [p for p in parts if p is not None]
                            if not parts:
                                return None
                            out = {
                                "Er": np.vstack([p["Er"] for p in parts]),
                                "Et": np.vstack([p["Et"] for p in parts]),
                                "cvs": np.concatenate([p["cvs"] for p in parts]),
                                "blocks": np.concatenate([p["blocks"] for p in parts]),
                                "k": np.concatenate([p["k"] for p in parts]),
                                "n": np.concatenate([p["n"] for p in parts]),
                                "y": np.concatenate([p["y"] for p in parts]),
                            }
                            if all(("Er_aux" in p and "Et_aux" in p) for p in parts):
                                out["Er_aux"] = np.vstack([p["Er_aux"] for p in parts])
                                out["Et_aux"] = np.vstack([p["Et_aux"] for p in parts])
                            return out

                        design_train_map: Dict[Tuple[float, str], Dict[str, np.ndarray]] = {}
                        design_test_map: Dict[Tuple[float, str], Dict[str, np.ndarray]] = {}
                        for fb in eval_combos:
                            dtr = _build_design_one(fb, split="train")
                            dte = _build_design_one(fb, split="test")
                            if dtr is None or dte is None:
                                continue
                            design_train_map[fb] = dtr
                            design_test_map[fb] = dte

                        combos_ready = sorted(design_train_map.keys())
                        if not combos_ready:
                            print(f"[warn] no design rows ready for scope_key={scope_key_i}, mc={mc}", flush=True)
                            continue

                        # Release scope semantics:
                        #  - per_freq_block: fit separately per (freq, block)
                        #  - per_block: pooled across frequencies within one block
                        if scope == "per_freq_block":
                            fit_groups = [[fb] for fb in combos_ready]
                        else:
                            fit_groups = [combos_ready]

                        drop_name = str(loto_th) if loto_th is not None else "NONE"
                        is_loto = (drop_name != "NONE")
                        fixed_name = str(fixed_thresh_name) if fixed_thresh_name is not None else "NONE"
                        is_fixed_thresh = (fixed_name != "NONE")
                        base_run_id = f"{scope_key_i}|bin={float(bin_ms):g}|mc={int(mc)}"
                        if is_fixed_thresh:
                            base_run_id = f"{base_run_id}|fixed={fixed_name}"

                        for group in fit_groups:
                            design_pool_train = _concat_design([design_train_map[fb] for fb in group])
                            if design_pool_train is None:
                                continue

                            for model_bank in model_banks_for_bin:
                                if _is_combined_model_bank(model_bank) and not run_combined_this_bin:
                                    continue
                                base_run_id_model = f"{base_run_id}|model={str(model_bank)}"
                                gate_enabled = str(model_bank) in HYBRID_FAMILY_MODEL_BANKS
                                rate_gate_mode_eff = str(rate_gate_mode) if gate_enabled else "none"
                                rate_gate_reg_eff = float(rate_gate_reg) if gate_enabled else 0.0
                                use_rowwise_weights = bool(thresh_flex_mode == "per_point" and str(model_bank) == "hybrid_combined")
                                weights_by_cv: Dict[str, Dict[str, float]] = {}
                                if use_rowwise_weights:
                                    if len(group) != 1:
                                        raise RuntimeError("per-point hybrid mode expects single-panel fit groups")
                                    panel_key = group[0]
                                    dtr_panel = design_train_map[panel_key]
                                    params, weights, weights_by_cv, _ = fit_per_point_hybrid_from_design(
                                        Er=dtr_panel["Er"],
                                        Et=dtr_panel["Et"],
                                        Er_aux=dtr_panel.get("Er_aux", None),
                                        Et_aux=dtr_panel.get("Et_aux", None),
                                        cvs=dtr_panel["cvs"],
                                        blocks=dtr_panel["blocks"],
                                        k=dtr_panel["k"],
                                        n=dtr_panel["n"],
                                        y=dtr_panel["y"],
                                        ths=ths_use,
                                        ms=ms,
                                        loss_reg_l2=float(loss_reg_l2),
                                        weight_kl=float(weight_kl),
                                        allow_zero_gains=bool(allow_zero_gains),
                                        rate_gate_mode=str(rate_gate_mode_eff),
                                        rate_gate_reg=float(rate_gate_reg_eff),
                                        rate_gate_prior_floor=float(rate_gate_prior_floor),
                                        rate_gate_prior_center=float(rate_gate_prior_center),
                                        rate_gate_prior_slope=float(rate_gate_prior_slope),
                                        weight_entropy_floor_k=float(weight_entropy_floor_k),
                                        weight_entropy_floor_lambda=float(weight_entropy_floor_lambda),
                                        weight_max_cap=float(weight_max_cap),
                                        weight_max_cap_lambda=float(weight_max_cap_lambda),
                                        weight_smooth_lambda=float(weight_smooth_lambda),
                                    )
                                    params_by_panel = {panel_key: params}
                                elif shared_thresh_mode == "joint_panel_params":
                                    designs_group = [((float(f_fit), str(b_fit)), design_train_map[(f_fit, b_fit)]) for f_fit, b_fit in group]
                                    params_by_panel, weights, _ = fit_shared_weights_panel_params_from_designs(
                                        designs=designs_group,
                                        ths=ths_use,
                                        ms=ms,
                                        loss_reg_l2=float(loss_reg_l2),
                                        weight_kl=float(weight_kl),
                                        allow_zero_gains=bool(allow_zero_gains),
                                        model_bank=str(model_bank),
                                        rate_gate_mode=str(rate_gate_mode_eff),
                                        rate_gate_reg=float(rate_gate_reg_eff),
                                        rate_gate_prior_floor=float(rate_gate_prior_floor),
                                        rate_gate_prior_center=float(rate_gate_prior_center),
                                        rate_gate_prior_slope=float(rate_gate_prior_slope),
                                        weight_entropy_floor_k=float(weight_entropy_floor_k),
                                        weight_entropy_floor_lambda=float(weight_entropy_floor_lambda),
                                        weight_max_cap=float(weight_max_cap),
                                        weight_max_cap_lambda=float(weight_max_cap_lambda),
                                        panel_balance=True,
                                    )
                                else:
                                    params, weights, _ = fit_mixture_from_design(
                                        Er=design_pool_train["Er"],
                                        Et=design_pool_train["Et"],
                                        Er_aux=design_pool_train.get("Er_aux", None),
                                        Et_aux=design_pool_train.get("Et_aux", None),
                                        cvs=design_pool_train["cvs"],
                                        blocks=design_pool_train["blocks"],
                                        k=design_pool_train["k"],
                                        n=design_pool_train["n"],
                                        y=design_pool_train["y"],
                                        ths=ths_use,
                                        ms=ms,
                                        loss_reg_l2=float(loss_reg_l2),
                                        weight_kl=float(weight_kl),
                                        allow_zero_gains=bool(allow_zero_gains),
                                        model_bank=str(model_bank),
                                        rate_gate_mode=str(rate_gate_mode_eff),
                                        rate_gate_reg=float(rate_gate_reg_eff),
                                        rate_gate_prior_floor=float(rate_gate_prior_floor),
                                        rate_gate_prior_center=float(rate_gate_prior_center),
                                        rate_gate_prior_slope=float(rate_gate_prior_slope),
                                        weight_entropy_floor_k=float(weight_entropy_floor_k),
                                        weight_entropy_floor_lambda=float(weight_entropy_floor_lambda),
                                        weight_max_cap=float(weight_max_cap),
                                        weight_max_cap_lambda=float(weight_max_cap_lambda),
                                    )
                                    params_by_panel = {(float(f_fit), str(b_fit)): params for f_fit, b_fit in group}

                                for f_fit, b_fit in group:
                                    params = params_by_panel[(float(f_fit), str(b_fit))]
                                    dtr = design_train_map[(f_fit, b_fit)]
                                    dte = design_test_map[(f_fit, b_fit)]
                                    if use_rowwise_weights:
                                        met_tr = evaluate_design_metrics_rowwise(
                                            params=params,
                                            weights_by_cv=weights_by_cv,
                                            ths=ths_use,
                                            Er=dtr["Er"],
                                            Et=dtr["Et"],
                                            Er_aux=dtr.get("Er_aux", None),
                                            Et_aux=dtr.get("Et_aux", None),
                                            cvs=dtr["cvs"],
                                            blocks=dtr["blocks"],
                                            k=dtr["k"],
                                            n=dtr["n"],
                                            y=dtr["y"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                        met_te = evaluate_design_metrics_rowwise(
                                            params=params,
                                            weights_by_cv=weights_by_cv,
                                            ths=ths_use,
                                            Er=dte["Er"],
                                            Et=dte["Et"],
                                            Er_aux=dte.get("Er_aux", None),
                                            Et_aux=dte.get("Et_aux", None),
                                            cvs=dte["cvs"],
                                            blocks=dte["blocks"],
                                            k=dte["k"],
                                            n=dte["n"],
                                            y=dte["y"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                    else:
                                        met_tr = evaluate_design_metrics(
                                            params=params,
                                            weights=weights,
                                            ths=ths_use,
                                            Er=dtr["Er"],
                                            Et=dtr["Et"],
                                            Er_aux=dtr.get("Er_aux", None),
                                            Et_aux=dtr.get("Et_aux", None),
                                            cvs=dtr["cvs"],
                                            blocks=dtr["blocks"],
                                            k=dtr["k"],
                                            n=dtr["n"],
                                            y=dtr["y"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                        met_te = evaluate_design_metrics(
                                            params=params,
                                            weights=weights,
                                            ths=ths_use,
                                            Er=dte["Er"],
                                            Et=dte["Et"],
                                            Er_aux=dte.get("Er_aux", None),
                                            Et_aux=dte.get("Et_aux", None),
                                            cvs=dte["cvs"],
                                            blocks=dte["blocks"],
                                            k=dte["k"],
                                            n=dte["n"],
                                            y=dte["y"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                    met = {
                                        "rmse_train": float(met_tr["rmse"]),
                                        "mae_train": float(met_tr["mae"]),
                                        "nll_train": float(met_tr["nll"]),
                                        "rmse_test": float(met_te["rmse"]),
                                        "mae_test": float(met_te["mae"]),
                                    }

                                    weights_main = {str(k): float(v) for k, v in weights.items()}
                                    if model_bank == "hybrid_combined":
                                        weights_shared = dict(weights_main)
                                        weights_rate = {}
                                        weights_timing = {}
                                    elif model_bank == "rate_combined":
                                        weights_shared = {}
                                        weights_rate = dict(weights_main)
                                        weights_timing = {}
                                    elif model_bank == "timing_combined":
                                        weights_shared = {}
                                        weights_rate = {}
                                        weights_timing = dict(weights_main)
                                    else:
                                        raise ValueError(f"Unknown release model_bank: {model_bank}")

                                    fr = FitResult(
                                        scope_key=str(scope_key_i),
                                        model_bank=str(model_bank),
                                        bin_ms=float(bin_ms),
                                        aux_bin_ms=float(combined_aux_bin_ms) if _is_combined_model_bank(model_bank) else float("nan"),
                                        mc=int(mc),
                                        loto_th=str(loto_th) if loto_th is not None else None,
                                        drop_thresh=drop_name,
                                        is_loto=is_loto,
                                        fixed_thresh=str(fixed_thresh_name) if fixed_thresh_name is not None else None,
                                        is_fixed_thresh=bool(is_fixed_thresh),
                                        base_run_id=base_run_id_model,
                                        freq=float(f_fit),
                                        block=str(b_fit),
                                        n_thresh=int(len(ths_use)),
                                        params={**params, "freq": float(f_fit), "block": str(b_fit), "model_bank": str(model_bank)},
                                        weights=weights_main,
                                        weights_shared=weights_shared,
                                        weights_rate=weights_rate,
                                        weights_timing=weights_timing,
                                        metrics=met,
                                        loss_reg_l2=float(loss_reg_l2),
                                        weight_kl=float(weight_kl),
                                        allow_zero_gains=bool(allow_zero_gains),
                                        rate_gate_mode=str(rate_gate_mode_eff),
                                        rate_gate_reg=float(rate_gate_reg_eff),
                                        rate_gate_prior_floor=float(rate_gate_prior_floor),
                                        rate_gate_prior_center=float(rate_gate_prior_center),
                                        rate_gate_prior_slope=float(rate_gate_prior_slope),
                                        shared_thresh_mode=str(shared_thresh_mode),
                                        weight_entropy_floor_k=float(weight_entropy_floor_k),
                                        weight_entropy_floor_lambda=float(weight_entropy_floor_lambda),
                                        weight_max_cap=float(weight_max_cap),
                                        weight_max_cap_lambda=float(weight_max_cap_lambda),
                                        thresh_flex_mode=str(thresh_flex_mode),
                                        weight_smooth_lambda=float(weight_smooth_lambda),
                                        weights_by_cv={str(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in weights_by_cv.items()},
                                    )
                                    results.append(fr)
                                    fp_params.write(json.dumps(dataclasses.asdict(fr)) + "\n")

                                    rows_results.append(
                                        dict(
                                            scope=str(scope),
                                            scope_key=str(scope_key_i),
                                            model_bank=str(model_bank),
                                            freq=float(f_fit),
                                            block=str(b_fit),
                                            bin_ms=float(bin_ms),
                                            aux_bin_ms=float(combined_aux_bin_ms) if _is_combined_model_bank(model_bank) else float("nan"),
                                            mc=int(mc),
                                            loto_th=str(loto_th) if loto_th is not None else "",
                                            drop_thresh=drop_name,
                                            is_loto=bool(is_loto),
                                            fixed_thresh=str(fixed_thresh_name) if fixed_thresh_name is not None else "",
                                            is_fixed_thresh=bool(is_fixed_thresh),
                                            base_run_id=str(base_run_id_model),
                                            n_thresh=int(len(ths_use)),
                                            rmse_train=float(met["rmse_train"]),
                                            mae_train=float(met["mae_train"]),
                                            nll_train=float(met["nll_train"]),
                                            rmse_test=float(met["rmse_test"]),
                                            mae_test=float(met["mae_test"]),
                                            weights_json=json.dumps(weights_main, sort_keys=True),
                                            weights_shared_json=json.dumps(weights_shared, sort_keys=True),
                                            weights_rate_json=json.dumps(weights_rate, sort_keys=True),
                                            weights_timing_json=json.dumps(weights_timing, sort_keys=True),
                                            weights_by_cv_json=json.dumps(weights_by_cv, sort_keys=True),
                                            gain_rate=float(params["g_rate"]),
                                            gain_timing=float(params["g_time"]),
                                            gain_rate_aux=float(params.get("g_rate_aux", np.nan)),
                                            gain_timing_aux=float(params.get("g_time_aux", np.nan)),
                                            bias=float(params["b"]),
                                            temp=float(params["temp"]),
                                            loss_reg_l2=float(loss_reg_l2),
                                            weight_kl=float(weight_kl),
                                            allow_zero_gains=bool(allow_zero_gains),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                            rate_gate_reg=float(rate_gate_reg_eff),
                                            rate_gate_prior_floor=float(rate_gate_prior_floor),
                                            rate_gate_prior_center=float(rate_gate_prior_center),
                                            rate_gate_prior_slope=float(rate_gate_prior_slope),
                                            shared_thresh_mode=str(shared_thresh_mode),
                                            rate_gate_floor=float(params.get("rate_gate_floor", 1.0)),
                                            rate_gate_center=float(params.get("rate_gate_center", 0.5)),
                                            rate_gate_slope=float(params.get("rate_gate_slope", 8.0)),
                                            weight_entropy_floor_k=float(weight_entropy_floor_k),
                                            weight_entropy_floor_lambda=float(weight_entropy_floor_lambda),
                                            weight_max_cap=float(weight_max_cap),
                                            weight_max_cap_lambda=float(weight_max_cap_lambda),
                                            thresh_flex_mode=str(thresh_flex_mode),
                                            weight_smooth_lambda=float(weight_smooth_lambda),
                                        )
                                    )

                                    if use_rowwise_weights:
                                        pred_tr = predict_design_outputs_rowwise(
                                            params=params,
                                            weights_by_cv=weights_by_cv,
                                            ths=ths_use,
                                            Er=dtr["Er"],
                                            Et=dtr["Et"],
                                            Er_aux=dtr.get("Er_aux", None),
                                            Et_aux=dtr.get("Et_aux", None),
                                            cvs=dtr["cvs"],
                                            blocks=dtr["blocks"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                        pred_te = predict_design_outputs_rowwise(
                                            params=params,
                                            weights_by_cv=weights_by_cv,
                                            ths=ths_use,
                                            Er=dte["Er"],
                                            Et=dte["Et"],
                                            Er_aux=dte.get("Er_aux", None),
                                            Et_aux=dte.get("Et_aux", None),
                                            cvs=dte["cvs"],
                                            blocks=dte["blocks"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                    else:
                                        pred_tr = predict_design_outputs(
                                            params=params,
                                            weights=weights_main,
                                            ths=ths_use,
                                            Er=dtr["Er"],
                                            Et=dtr["Et"],
                                            Er_aux=dtr.get("Er_aux", None),
                                            Et_aux=dtr.get("Et_aux", None),
                                            cvs=dtr["cvs"],
                                            blocks=dtr["blocks"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                        pred_te = predict_design_outputs(
                                            params=params,
                                            weights=weights_main,
                                            ths=ths_use,
                                            Er=dte["Er"],
                                            Et=dte["Et"],
                                            Er_aux=dte.get("Er_aux", None),
                                            Et_aux=dte.get("Et_aux", None),
                                            cvs=dte["cvs"],
                                            blocks=dte["blocks"],
                                            model_bank=str(model_bank),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                        )
                                    rows_curves.append(
                                        dict(
                                            scope=str(scope),
                                            scope_key=str(scope_key_i),
                                            model_bank=str(model_bank),
                                            freq=float(f_fit),
                                            block=str(b_fit),
                                            bin_ms=float(bin_ms),
                                            aux_bin_ms=float(combined_aux_bin_ms) if _is_combined_model_bank(model_bank) else float("nan"),
                                            mc=int(mc),
                                            drop_thresh=drop_name,
                                            is_loto=bool(is_loto),
                                            fixed_thresh=str(fixed_thresh_name) if fixed_thresh_name is not None else "",
                                            is_fixed_thresh=bool(is_fixed_thresh),
                                            loss_reg_l2=float(loss_reg_l2),
                                            weight_kl=float(weight_kl),
                                            allow_zero_gains=bool(allow_zero_gains),
                                            rate_gate_mode=str(rate_gate_mode_eff),
                                            rate_gate_reg=float(rate_gate_reg_eff),
                                            rate_gate_prior_floor=float(rate_gate_prior_floor),
                                            rate_gate_prior_center=float(rate_gate_prior_center),
                                            rate_gate_prior_slope=float(rate_gate_prior_slope),
                                            shared_thresh_mode=str(shared_thresh_mode),
                                            weight_entropy_floor_k=float(weight_entropy_floor_k),
                                            weight_entropy_floor_lambda=float(weight_entropy_floor_lambda),
                                            weight_max_cap=float(weight_max_cap),
                                            weight_max_cap_lambda=float(weight_max_cap_lambda),
                                            thresh_flex_mode=str(thresh_flex_mode),
                                            weight_smooth_lambda=float(weight_smooth_lambda),
                                            weights_by_cv_json=json.dumps(weights_by_cv, sort_keys=True),
                                            cv_json=json.dumps(np.asarray(dte["cvs"], dtype=float).tolist()),
                                            p_obs_json=json.dumps(np.asarray(dte["y"], dtype=float).tolist()),
                                            p_pred_train_json=json.dumps(np.asarray(pred_tr["p_correct"], dtype=float).tolist()),
                                            p_pred_test_json=json.dumps(np.asarray(pred_te["p_correct"], dtype=float).tolist()),
                                            rate_drive_train_json=json.dumps(np.asarray(pred_tr["rate_drive"], dtype=float).tolist()),
                                            timing_drive_train_json=json.dumps(np.asarray(pred_tr["timing_drive"], dtype=float).tolist()),
                                            rate_drive_test_json=json.dumps(np.asarray(pred_te["rate_drive"], dtype=float).tolist()),
                                            timing_drive_test_json=json.dumps(np.asarray(pred_te["timing_drive"], dtype=float).tolist()),
                                            rate_drive_aux_train_json=json.dumps(np.asarray(pred_tr["rate_drive_aux"], dtype=float).tolist()),
                                            timing_drive_aux_train_json=json.dumps(np.asarray(pred_tr["timing_drive_aux"], dtype=float).tolist()),
                                            rate_drive_aux_test_json=json.dumps(np.asarray(pred_te["rate_drive_aux"], dtype=float).tolist()),
                                            timing_drive_aux_test_json=json.dumps(np.asarray(pred_te["timing_drive_aux"], dtype=float).tolist()),
                                            gate_train_json=json.dumps(np.asarray(pred_tr["gate"], dtype=float).tolist()),
                                            gate_test_json=json.dumps(np.asarray(pred_te["gate"], dtype=float).tolist()),
                                        )
                                    )

    df_res = pd.DataFrame(rows_results)
    df_res.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    df_curves = pd.DataFrame(rows_curves)
    if not df_curves.empty:
        write_curve_summaries(df_curves, out_dir)
    write_mc_summaries(df_res, out_dir)
    return df_res, results




# ============================
# Release data readers, plotting, and CLI
# ============================

FREQUENCIES: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0)
BLOCKS: tuple[str, ...] = ("std0", "std1")
BLOCK_LABELS: Mapping[str, str] = {
    "std0": "Standard CV=0",
    "std1": "Standard CV=1",
}

RELEASE_MODEL_BANKS: tuple[str, ...] = (
    "rate_combined",
    "timing_combined",
    "hybrid_combined",
)
MODEL_LABELS: Mapping[str, str] = {
    "rate_combined": "Rate",
    "timing_combined": "Timing",
    "hybrid_combined": "Hybrid",
    "per_point_hybrid": "Per-point hybrid",
}
THRESH_REGIME_TICKS: Mapping[str, str] = {
    "I_first": "first",
    "I_volley5": "volley5",
    "I_volley10": "volley10",
    "I_5": "5%",
    "I_10": "10%",
    "I_15": "15%",
    "I_25": "25%",
    "I_50": "50%",
    "I_max": "max",
}

RELEASE_PIPELINE_SETTINGS: Mapping[str, Any] = {
    "bins_ms": (1.0, 10.0),
    "primary_bin_ms": 10.0,
    "aux_bin_ms": 1.0,
    "n_mc": 200,
    "train_frac": 0.8,
    "seed": 0,
    "jnd_k": 2.5,
    "entmax_alpha": 1.1,
    "loss_reg_l2": 0.0001,
    "weight_kl": 0.01,
    "allow_zero_gains": False,
    "rate_gate_mode_hybrid": "delta_cv",
    "rate_gate_reg_hybrid": 0.002,
    "rate_gate_mode_rate_timing": "none",
    "rate_gate_reg_rate_timing": 0.0,
    "rate_gate_prior_floor": 1.0,
    "rate_gate_prior_center": 0.5,
    "rate_gate_prior_slope": 8.0,
    "weight_entropy_floor_k": 3.0,
    "weight_entropy_floor_lambda": 0.01,
    "weight_max_cap": 0.65,
    "weight_max_cap_lambda": 0.01,
    "weight_smooth_lambda_shared": 0.0,
    "weight_smooth_lambda_per_point": 0.001,
}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DiscriminationDataConfig:
    """Filesystem and stage settings for the cleaned stage-05 notebook."""

    repo_root: Path = field(default_factory=_default_repo_root)
    data_dir: Optional[Path] = None
    sim_csv: Optional[Path] = None

    @property
    def data_root(self) -> Path:
        if self.data_dir is not None:
            return Path(self.data_dir)
        return self.repo_root / "data" / "discrimination"

    @property
    def per_block_dir(self) -> Path:
        return self.data_root / "per-block"

    @property
    def per_point_dir(self) -> Path:
        return self.data_root / "per-point"

    @property
    def slack_dir(self) -> Path:
        return self.data_root / "slack"

    @property
    def run_config_path(self) -> Path:
        return self.data_root / "run_config.json"

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "data_manifest.json"

    @property
    def detection_response_csv(self) -> Path:
        if self.sim_csv is not None:
            return Path(self.sim_csv)
        return self.repo_root / "data" / "detection" / "generated" / "cv0_based_responses.csv"


@dataclass(frozen=True)
class DiscriminationData:
    """Loaded release discrimination tables."""

    config: DiscriminationDataConfig
    run_config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    slack_per_rat: pd.DataFrame
    slack_mean: pd.DataFrame
    per_block_results: pd.DataFrame
    per_block_summary: pd.DataFrame
    per_block_curves: pd.DataFrame
    per_block_curve_summary: pd.DataFrame
    per_block_weights: pd.DataFrame
    per_point_results: pd.DataFrame
    per_point_summary: pd.DataFrame
    per_point_curves: pd.DataFrame
    per_point_curve_summary: pd.DataFrame
    per_point_weights: pd.DataFrame


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing release discrimination file: {path}")
    return normalize_threshold_frame(pd.read_csv(path))


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return json.loads(value)
    return value


def _sem(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _sem0(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _as_float(value: Any) -> float:
    return float(pd.to_numeric(value, errors="coerce"))


def _block_label(block: str) -> str:
    return BLOCK_LABELS.get(str(block), str(block))


def _parse_num_dict(value: Any) -> dict[str, float]:
    try:
        raw = _json_value(value)
    except Exception:
        return {}
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, item in raw.items():
        try:
            number = float(item)
        except Exception:
            continue
        if np.isfinite(number):
            out[canonical_threshold_type(key)] = number
    return out


def _to_choice_display(cv_values: Sequence[float], p_correct_values: Sequence[float], block: str) -> np.ndarray:
    """Match the manuscript display convention used by Figure 4."""

    cv = np.asarray(cv_values, dtype=float)
    p = np.asarray(p_correct_values, dtype=float)
    if str(block) == "std0":
        return np.where(np.isclose(cv, 0.0), 1.0 - p, p)
    if str(block) == "std1":
        return np.where(np.isclose(cv, 1.0), 1.0 - p, p)
    return p


def threshold_label(thresh_type: str) -> str:
    """Return the manuscript display label for a threshold type."""

    return threshold_display_name(str(thresh_type))


def load_discrimination_data(
    config: Optional[DiscriminationDataConfig] = None,
) -> DiscriminationData:
    """Load the compact release stage-05 data products."""

    cfg = config or DiscriminationDataConfig()
    run_config = _read_json(cfg.run_config_path) if cfg.run_config_path.exists() else {}
    manifest = _read_json(cfg.manifest_path) if cfg.manifest_path.exists() else {}
    slack_per_rat, slack_mean = load_slack_dir(str(cfg.slack_dir), include_templates=True)

    return DiscriminationData(
        config=cfg,
        run_config=run_config,
        manifest=manifest,
        slack_per_rat=slack_per_rat,
        slack_mean=slack_mean,
        per_block_results=_read_csv(cfg.per_block_dir / "merged_results.csv"),
        per_block_summary=_read_csv(cfg.per_block_dir / "merged_results_mc_summary.csv"),
        per_block_curves=_read_csv(cfg.per_block_dir / "merged_curves.csv"),
        per_block_curve_summary=_read_csv(cfg.per_block_dir / "merged_curves_mc_summary.csv"),
        per_block_weights=_read_csv(cfg.per_block_dir / "merged_weights_mc_summary.csv"),
        per_point_results=_read_csv(cfg.per_point_dir / "merged_results.csv"),
        per_point_summary=_read_csv(cfg.per_point_dir / "merged_results_mc_summary.csv"),
        per_point_curves=_read_csv(cfg.per_point_dir / "merged_curves.csv"),
        per_point_curve_summary=_read_csv(cfg.per_point_dir / "merged_curves_mc_summary.csv"),
        per_point_weights=_read_csv(cfg.per_point_dir / "merged_weights_mc_summary.csv"),
    )


def data_inventory_table(data: DiscriminationData) -> pd.DataFrame:
    """Summarize the copied data products used by the notebook."""

    rows = []
    for label, df in (
        ("per-block results", data.per_block_results),
        ("per-block MC summary", data.per_block_summary),
        ("per-block curves", data.per_block_curves),
        ("per-block curve summary", data.per_block_curve_summary),
        ("per-block weights", data.per_block_weights),
        ("per-point results", data.per_point_results),
        ("per-point MC summary", data.per_point_summary),
        ("per-point curves", data.per_point_curves),
        ("per-point curve summary", data.per_point_curve_summary),
        ("per-point weights", data.per_point_weights),
    ):
        rows.append({"table": label, "rows": len(df), "columns": len(df.columns)})
    rows.append({"table": "Slack behavioral files", "rows": len(data.slack_per_rat), "columns": len(data.slack_per_rat.columns)})
    return pd.DataFrame(rows)


def run_table(data: DiscriminationData) -> pd.DataFrame:
    """Show the fixed manuscript analyses without exploratory hyperparameters."""

    return pd.DataFrame(
        [
            {
                "analysis": "per-block mixtures",
                "scope": "block-specific fit",
                "models": "Rate, Timing, Hybrid",
                "threshold weights": "one shared mixture per block",
                "raw runs": len(data.per_block_results),
                "summary rows": len(data.per_block_summary),
            },
            {
                "analysis": "per-point hybrid",
                "scope": "frequency x block fit",
                "models": "Hybrid",
                "threshold weights": "mixture can vary with CV",
                "raw runs": len(data.per_point_results),
                "summary rows": len(data.per_point_summary),
            },
        ]
    )


def fixed_settings_table(data: DiscriminationData) -> pd.DataFrame:
    """Return the hard-coded fitting settings used by the manuscript lane."""

    settings = dict(RELEASE_PIPELINE_SETTINGS)
    settings.update(data.run_config.get("fixed_pipeline_settings", {}))
    display_keys = [
        "bins_ms",
        "n_mc",
        "train_frac",
        "jnd_k",
        "entmax_alpha",
        "loss_reg_l2",
        "weight_kl",
        "rate_gate_mode_hybrid",
        "rate_gate_reg_hybrid",
        "weight_entropy_floor_k",
        "weight_entropy_floor_lambda",
        "weight_max_cap",
        "weight_max_cap_lambda",
        "weight_smooth_lambda_per_point",
    ]
    return pd.DataFrame(
        [{"setting": key, "value": settings.get(key)} for key in display_keys]
    )


def slack_coverage_table(data: DiscriminationData) -> pd.DataFrame:
    """Summarize behavioral data coverage by frequency and standard block."""

    per_rat = data.slack_per_rat.copy()
    per_rat["freq"] = pd.to_numeric(per_rat["freq"], errors="coerce")
    per_rat["cv"] = pd.to_numeric(per_rat["cv"], errors="coerce")
    rows = []
    for (freq, block), sub in per_rat.groupby(["freq", "block"], dropna=False):
        rows.append(
            {
                "freq": float(freq),
                "block": str(block),
                "rats": int(sub["rat_name"].nunique()) if "rat_name" in sub else np.nan,
                "cv levels": int(sub["cv"].nunique()),
                "trials": int(pd.to_numeric(sub.get("n", 0), errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["block", "freq"]).reset_index(drop=True)


def model_comparison_table(data: DiscriminationData) -> pd.DataFrame:
    """Return manuscript model performance by block and frequency."""

    pb = data.per_block_summary.copy()
    pb["analysis"] = "per-block"
    pb["model"] = pb["model_bank"].map(MODEL_LABELS).fillna(pb["model_bank"])

    pp = data.per_point_summary.copy()
    pp["analysis"] = "per-point"
    pp["model"] = MODEL_LABELS["per_point_hybrid"]

    keep = [
        "analysis",
        "model",
        "model_bank",
        "freq",
        "block",
        "rmse_test_mean",
        "rmse_test_sem",
        "mae_test_mean",
        "mae_test_sem",
        "n_mc",
    ]
    out = pd.concat([pb[keep], pp[keep]], ignore_index=True)
    return out.sort_values(["block", "freq", "analysis", "model"]).reset_index(drop=True)


def model_comparison_pivot(data: DiscriminationData, *, include_block_means: bool = True) -> pd.DataFrame:
    """Return RMSE comparison table with optional block-mean summary rows."""

    comparison = model_comparison_table(data)
    pivot = comparison.pivot_table(
        index=["block", "freq"],
        columns=["analysis", "model"],
        values="rmse_test_mean",
    )
    if not include_block_means or pivot.empty:
        return pivot

    mean_pivot = comparison.pivot_table(
        index="block",
        columns=["analysis", "model"],
        values="rmse_test_mean",
        aggfunc="mean",
    )
    blocks = list(pivot.index.get_level_values("block").unique())
    ordered_blocks = [block for block in BLOCKS if block in blocks]
    ordered_blocks.extend(block for block in blocks if block not in ordered_blocks)

    frames = []
    for block in ordered_blocks:
        frames.append(pivot.loc[[block]])
        if block in mean_pivot.index:
            mean_row = mean_pivot.loc[[block]].copy()
            mean_row.index = pd.MultiIndex.from_tuples(
                [(block, f"{block} mean")],
                names=["block", "freq"],
            )
            frames.append(mean_row)
    return pd.concat(frames).loc[:, pivot.columns]


def best_per_block_models(data: DiscriminationData) -> pd.DataFrame:
    """Identify the lowest-RMSE per-block model in each panel."""

    df = data.per_block_summary.copy()
    idx = df.groupby(["freq", "block"], dropna=False)["rmse_test_mean"].idxmin()
    out = df.loc[idx, ["freq", "block", "model_bank", "rmse_test_mean", "rmse_test_sem"]].copy()
    out["model"] = out["model_bank"].map(MODEL_LABELS).fillna(out["model_bank"])
    return out.sort_values(["block", "freq"]).reset_index(drop=True)


def per_block_model_winners(data: DiscriminationData) -> dict[str, str]:
    """Return the winning per-block model for each standard block."""

    df = data.per_block_results.copy()
    df = df[df["model_bank"].isin(RELEASE_MODEL_BANKS)].copy()
    score = (
        df.groupby(["block", "model_bank"], as_index=False)
        .agg(rmse_mean=("rmse_test", "mean"), rmse_sem=("rmse_test", _sem0), n_rows=("rmse_test", "size"))
    )
    return {
        str(block): str(group.sort_values("rmse_mean").iloc[0]["model_bank"])
        for block, group in score.groupby("block")
    }


def _ordered_threshold_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["thresh_type"] = pd.Categorical(out["thresh_type"], categories=list(THRESH_ORDER), ordered=True)
    return out.sort_values("thresh_type")


def per_block_weight_summary(data: DiscriminationData) -> pd.DataFrame:
    """Average per-block manuscript weights across frequencies for compact display."""

    w = data.per_block_weights.copy()
    w = w[w["model_bank"].isin(RELEASE_MODEL_BANKS)].copy()
    rows = []
    for keys, sub in w.groupby(["model_bank", "block", "thresh_type"], dropna=False):
        model_bank, block, thresh_type = keys
        rows.append(
            {
                "model_bank": model_bank,
                "model": MODEL_LABELS.get(str(model_bank), str(model_bank)),
                "block": block,
                "thresh_type": thresh_type,
                "threshold": threshold_label(str(thresh_type)),
                "weight_mean": float(pd.to_numeric(sub["weight_mean"], errors="coerce").mean()),
                "weight_sem": _sem(sub["weight_mean"]),
                "n_freq": int(sub["freq"].nunique()),
            }
        )
    return _ordered_threshold_frame(pd.DataFrame(rows))


def per_point_weight_summary(data: DiscriminationData) -> pd.DataFrame:
    """Decode and summarize CV-specific threshold weights for the per-point model."""

    rows: list[dict[str, Any]] = []
    curves = data.per_point_curves.copy()
    for row in curves.itertuples(index=False):
        weights_by_cv = _json_value(getattr(row, "weights_by_cv_json", None)) or {}
        for cv_key, weights in weights_by_cv.items():
            cv = _as_float(cv_key)
            for thresh_type in THRESH_ORDER:
                rows.append(
                    {
                        "freq": float(getattr(row, "freq")),
                        "block": str(getattr(row, "block")),
                        "mc": int(getattr(row, "mc")),
                        "cv": cv,
                        "thresh_type": thresh_type,
                        "threshold": threshold_label(thresh_type),
                        "regime_label": THRESH_REGIME_TICKS[thresh_type],
                        "weight": float(weights.get(thresh_type, 0.0)),
                    }
                )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    out = (
        raw.groupby(["freq", "block", "cv", "thresh_type", "threshold", "regime_label"], as_index=False)
        .agg(
            weight_mean=("weight", "mean"),
            weight_sem=("weight", _sem),
            n_mc=("mc", "nunique"),
        )
    )
    return _ordered_threshold_frame(out)


def per_block_regime_weight_summary(data: DiscriminationData) -> pd.DataFrame:
    """Decode per-block raw mixture weights from the copied manuscript results."""

    rows: list[dict[str, Any]] = []
    base = data.per_block_results.copy()
    base = base[base["model_bank"].isin(RELEASE_MODEL_BANKS)].copy()
    for (block, model_bank), sub in base.groupby(["block", "model_bank"], dropna=False):
        dedup_cols = [col for col in ["mc", "weights_json"] if col in sub.columns]
        if dedup_cols:
            sub = sub.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols)
        for row in sub.itertuples(index=False):
            weights = _parse_num_dict(getattr(row, "weights_json", {}))
            for thresh_type in THRESH_ORDER:
                rows.append(
                    {
                        "block": str(block),
                        "model_bank": str(model_bank),
                        "model": MODEL_LABELS.get(str(model_bank), str(model_bank)),
                        "mc": int(getattr(row, "mc", -1)),
                        "thresh_type": thresh_type,
                        "threshold": threshold_label(thresh_type),
                        "regime_label": THRESH_REGIME_TICKS[thresh_type],
                        "weight": float(weights.get(thresh_type, np.nan)),
                    }
                )
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    out = (
        raw.groupby(["block", "model_bank", "model", "thresh_type", "threshold", "regime_label"], as_index=False)
        .agg(
            weight_mean=("weight", "mean"),
            weight_sem=("weight", _sem0),
            n_mc=("mc", "nunique"),
        )
    )
    return _ordered_threshold_frame(out)


def rerun_commands() -> pd.DataFrame:
    """Return the explicit rerun commands for the copied manuscript lane."""

    return pd.DataFrame(
        [
            {
                "step": "Full release pipeline",
                "command": "python hpc_tools/run_pipeline.py submit --project $PWD",
            },
            {
                "step": "Stage-05 pilot using archived detection data",
                "command": (
                    "python hpc_tools/run_pipeline.py submit --pilot --project $PWD"
                ),
            },
        ]
    )


def _scope_keys_for_run_kind(run_kind: str) -> tuple[str, ...]:
    if run_kind == "per_block":
        return tuple(f"block={block}" for block in BLOCKS)
    if run_kind == "per_point":
        return tuple(f"freq={int(freq)}|block={block}" for freq in FREQUENCIES for block in BLOCKS)
    raise ValueError(f"Unknown run_kind: {run_kind}")


def validate_fit_request(run_kind: str, scope_key: str, model_bank: str) -> None:
    """Validate the intentionally small release rerun surface."""

    run_kind = str(run_kind)
    model_bank = str(model_bank)
    if scope_key not in _scope_keys_for_run_kind(run_kind):
        valid = ", ".join(_scope_keys_for_run_kind(run_kind))
        raise ValueError(f"Invalid scope_key={scope_key!r} for {run_kind}. Valid values: {valid}")
    if run_kind == "per_block" and model_bank not in RELEASE_MODEL_BANKS:
        raise ValueError(f"per_block release fits use: {', '.join(RELEASE_MODEL_BANKS)}")
    if run_kind == "per_point" and model_bank != "hybrid_combined":
        raise ValueError("per_point release fits use hybrid_combined only")


def run_release_fit(
    *,
    run_kind: str,
    sim_csv: str | Path,
    slack_dir: str | Path,
    out_dir: str | Path,
    scope_key: str,
    model_bank: str,
    n_mc: int = 200,
    mc_start: int = 0,
    mc_count: Optional[int] = None,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[Any]]:
    """Run one release discrimination fit with fixed release settings."""

    validate_fit_request(run_kind, scope_key, model_bank)
    settings = dict(RELEASE_PIPELINE_SETTINGS)
    is_hybrid = str(model_bank) == "hybrid_combined"
    is_per_point = str(run_kind) == "per_point"
    return run_fit(
        sim_csv=str(sim_csv),
        slack_dir=str(slack_dir),
        out_dir=str(out_dir),
        bins_ms=settings["bins_ms"],
        scope="per_freq_block" if is_per_point else "per_block",
        n_mc=int(n_mc),
        train_frac=float(train_frac),
        seed=int(seed),
        jnd_k=float(settings["jnd_k"]),
        entmax_alpha=float(settings["entmax_alpha"]),
        loss_reg_l2=float(settings["loss_reg_l2"]),
        weight_kl=float(settings["weight_kl"]),
        allow_zero_gains=bool(settings["allow_zero_gains"]),
        rate_gate_mode=settings["rate_gate_mode_hybrid"] if is_hybrid else settings["rate_gate_mode_rate_timing"],
        rate_gate_reg=float(settings["rate_gate_reg_hybrid"] if is_hybrid else settings["rate_gate_reg_rate_timing"]),
        rate_gate_prior_floor=float(settings["rate_gate_prior_floor"]),
        rate_gate_prior_center=float(settings["rate_gate_prior_center"]),
        rate_gate_prior_slope=float(settings["rate_gate_prior_slope"]),
        weight_entropy_floor_k=float(settings["weight_entropy_floor_k"]),
        weight_entropy_floor_lambda=float(settings["weight_entropy_floor_lambda"]),
        weight_max_cap=float(settings["weight_max_cap"]),
        weight_max_cap_lambda=float(settings["weight_max_cap_lambda"]),
        include_templates_for_slack=True,
        scope_key=scope_key,
        mc_start=int(mc_start),
        mc_count=mc_count,
        model_banks=(model_bank,),
        shared_thresh_mode="none" if is_per_point else "joint_panel_params",
        thresh_flex_mode="per_point" if is_per_point else "shared",
        weight_smooth_lambda=float(
            settings["weight_smooth_lambda_per_point"] if is_per_point else settings["weight_smooth_lambda_shared"]
        ),
    )




__all__ = [
    "BLOCKS",
    "FREQUENCIES",
    "RELEASE_MODEL_BANKS",
    "MODEL_LABELS",
    "THRESH_ORDER",
    "THRESH_REGIME_TICKS",
    "RELEASE_PIPELINE_SETTINGS",
    "DiscriminationDataConfig",
    "DiscriminationData",
    "best_per_block_models",
    "data_inventory_table",
    "fixed_settings_table",
    "load_discrimination_data",
    "rerun_commands",
    "run_table",
    "model_comparison_pivot",
    "model_comparison_table",
    "per_block_model_winners",
    "per_block_regime_weight_summary",
    "per_block_weight_summary",
    "per_point_weight_summary",
    "run_release_fit",
    "slack_coverage_table",
    "threshold_label",
    "validate_fit_request",
]
