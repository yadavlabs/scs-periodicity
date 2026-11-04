"""Threshold helpers retained from the original analysis for population use."""
from __future__ import annotations
from typing import Tuple
import numpy as np

THRESH_TYPES = ("I_first", "I_volley5", "I_volley10", "I_5", "I_10", "I_15", "I_25", "I_50", "I_all")
SPIKE_HEIGHT_MV = 0.0
INIT_V_MV = -80.0
PREPAD_MS = 100.0
TAIL_MS = 100.0
VOLLEY5_MIN_SPIKES = 5
VOLLEY10_MIN_SPIKES = 10
FIRST_SPIKE_SEARCH = dict(amp_start_uA=200.0, amp_max_uA=1000.0, bracket_factor=2.0, tol_uA=0.1, min_spikes=1)
ALL_SPIKES_SEARCH = dict(amp_max_uA=2000.0, growth_factor=2.0, plateau_repeats=3, tol_uA=0.1)

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
