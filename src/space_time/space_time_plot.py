"""Plotting helpers for stage 02 space-time outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .space_time_core import ComsolProfiles, SpaceTimeData, load_spacetime_npz


def plot_comsol_profiles(profiles: ComsolProfiles):
    """Plot interpolated COMSOL profiles used for current-to-voltage mapping."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(profiles.node_numbers, profiles.v_neg_mv, "o-", label="-1 uA phase")
    ax.plot(profiles.node_numbers, profiles.v_pos_mv, "o-", label="+1 uA phase")
    ax.set_title(f"Interpolated COMSOL profile: M{profiles.cutline}")
    ax.set_xlabel("Node number")
    ax.set_ylabel("Extracellular potential (mV)")
    ax.grid(True)
    ax.legend(framealpha=0)
    fig.tight_layout()
    return fig, ax

def _select_repetition(rep_labels: np.ndarray, rep_label_substr: str) -> int:
    matches = [i for i, label in enumerate(rep_labels) if rep_label_substr in str(label)]
    if not matches:
        raise ValueError(f"No repetition label contains {rep_label_substr!r}.")
    return matches[0]


def _first_nonzero_window(
    time_s: np.ndarray,
    current: np.ndarray,
    window_ms: float | tuple[float, float],
    fs: float,
) -> slice:
    if isinstance(window_ms, tuple):
        if len(window_ms) != 2:
            raise ValueError("window_ms tuple must be (start_ms, end_ms).")
        start_ms, end_ms = (float(window_ms[0]), float(window_ms[1]))
        if end_ms <= start_ms:
            raise ValueError("window_ms end must be greater than start.")
        time_ms = np.asarray(time_s, dtype=float) * 1000.0
        start = int(np.searchsorted(time_ms, start_ms, side="left"))
        end = int(np.searchsorted(time_ms, end_ms, side="right"))
        start = max(0, min(start, len(time_s) - 1))
        end = max(start + 1, min(end, len(time_s)))
        return slice(start, end)

    nonzero = np.where(current != 0)[0]
    if len(nonzero) == 0:
        raise ValueError("Selected repetition has no non-zero current values.")
    pulse_idx = int(nonzero[0])
    window_len_samples = int((window_ms / 1000.0) * fs)
    pre_samples = int((5.0 / 1000.0) * fs)
    start = max(0, pulse_idx - pre_samples)
    end = min(len(time_s), pulse_idx + window_len_samples)
    return slice(start, end)


def plot_spacetime_construction(
    time_s: np.ndarray,
    node_numbers: np.ndarray,
    current: np.ndarray,
    v_ext_mv: np.ndarray,
    label: str,
    window_ms: float | tuple[float, float] = 30.0,
    fs: float = 20_000.0,
    cmap: str = "seismic",
    dpi: int = 200,
):
    """Plot one current waveform and its mapped space-time voltage window."""
    import matplotlib.pyplot as plt

    window = _first_nonzero_window(time_s, current, window_ms=window_ms, fs=fs)
    t_win = time_s[window]
    current_win = current[window]
    v_win = v_ext_mv[window, :]
    vmax = float(np.nanmax(np.abs(v_win)))
    dt_ms = 1000.0 / fs
    extent = [
        t_win[0] * 1000.0 - 0.5 * dt_ms,
        t_win[-1] * 1000.0 + 0.5 * dt_ms,
        node_numbers[0] - 0.5,
        node_numbers[-1] + 0.5,
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.5, 4.8),
        dpi=dpi,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    axes[0].plot(t_win * 1000.0, current_win, linewidth=1.2)
    axes[0].set_ylabel("Current (uA)")
    axes[0].set_title(f"{label}: biphasic pulse snippet", fontsize=11)
    axes[0].grid(alpha=0.3)

    image = axes[1].imshow(
        v_win.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="none",
        resample=False,
    )
    cbar = fig.colorbar(image, ax=axes[1], pad=0.02)
    cbar.set_label("Extracellular potential (mV)", fontsize=9)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Node number")

    for ax in axes:
        ax.tick_params(direction="out", length=4, width=0.8)

    return fig, axes


def plot_spacetime_pub(
    npz_path: str | Path,
    rep_label_substr: str = "rep002_gamma",
    window_ms: float = 30.0,
    fs: float = 20_000.0,
    cmap: str = "seismic",
    dpi: int = 200,
):
    """Load a saved stage-02 bundle and make the manuscript-style space-time plot."""
    data = load_spacetime_npz(npz_path)
    rep_idx = _select_repetition(data.rep_labels, rep_label_substr)
    return plot_spacetime_construction(
        time_s=data.time_s,
        node_numbers=data.node_numbers,
        current=data.current[rep_idx],
        v_ext_mv=data.v_ext_mv[rep_idx],
        label=str(data.rep_labels[rep_idx]),
        window_ms=window_ms,
        fs=fs,
        cmap=cmap,
        dpi=dpi,
    )
