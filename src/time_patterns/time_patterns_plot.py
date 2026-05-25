"""Plotting helpers for stage 01 time-pattern outputs."""

from __future__ import annotations

import numpy as np

from .time_patterns_core import StimSpec, TimePatternConfig, load_time_pattern_bundle


def plot_time_pattern_example(
    config: TimePatternConfig | None = None,
    cv: float = 0.5,
    freq: int = 50,
    max_generated: int = 3,
):
    """Plot the Slack repetition and a few generated repetitions for one pair."""
    import matplotlib.pyplot as plt

    meta, waveforms = load_time_pattern_bundle(config=config, cv=cv, freq=freq)
    slack = waveforms["rep001_slack"]
    generated_keys = sorted(key for key in waveforms if "gamma" in key)
    if not generated_keys:
        generated_keys = sorted(key for key in waveforms if "regular" in key)
    generated_keys = generated_keys[:max_generated]

    fs = float(meta["fs"])
    time = np.arange(slack.size) / fs
    zoom_mask = time <= 0.1

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(time, slack, label="Slack (rep001)", linewidth=1.0)
    for key in generated_keys:
        axes[0].plot(time, waveforms[key], alpha=0.3, label=key)
    axes[0].set_title(f"Full waveform (3 s), cv={cv}, freq={freq} Hz")
    axes[0].set_ylabel("Current (uA)")
    axes[0].set_xlim(0, time[-1])
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time[zoom_mask], slack[zoom_mask], label="Slack (rep001)", linewidth=1.0)
    for key in generated_keys:
        axes[1].plot(time[zoom_mask], waveforms[key][zoom_mask], alpha=0.5, label=key)
    axes[1].set_title("Zoomed waveform (0-0.1 s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Current (uA)")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def plot_waveform_construction(
    generated_waveform: np.ndarray,
    pulse_times_s: np.ndarray,
    spec: StimSpec,
    cv: float,
    freq: float | int,
    zoom_end_s: float = 0.1,
    reference_waveform: np.ndarray | None = None,
):
    """Plot one worked example from pulse times to sampled waveform."""
    import matplotlib.pyplot as plt

    time = np.arange(generated_waveform.size) / spec.fs
    stim_mask = time <= spec.stim_dur_s
    zoom_mask = time <= zoom_end_s
    stim_pad_s = 0.05 * float(spec.stim_dur_s)
    stim_xlim = (-stim_pad_s, float(spec.stim_dur_s) + stim_pad_s)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), height_ratios=[1, 2, 2])
    axes[0].eventplot(pulse_times_s, lineoffsets=0, linelengths=0.8, colors="black")
    axes[0].set_title(f"Drawn pulse times, cv={cv}, freq={int(freq)} Hz")
    axes[0].set_xlim(*stim_xlim)
    axes[0].set_yticks([])

    if reference_waveform is not None:
        axes[1].plot(
            time[stim_mask],
            reference_waveform[stim_mask],
            label="Slack waveform",
            linewidth=1.0,
        )
    axes[1].plot(
        time[stim_mask],
        generated_waveform[stim_mask],
        label="Generated waveform",
        alpha=0.75,
    )
    axes[1].set_title("Full 2 s stimulation window")
    axes[1].set_ylabel("Current (uA)")
    axes[1].set_xlim(*stim_xlim)
    axes[1].grid(True)
    axes[1].legend()

    if reference_waveform is not None:
        axes[2].plot(
            time[zoom_mask],
            reference_waveform[zoom_mask],
            label="Slack waveform",
            linewidth=1.0,
        )
    axes[2].plot(
        time[zoom_mask],
        generated_waveform[zoom_mask],
        label="Generated waveform",
        alpha=0.75,
    )
    axes[2].set_title(f"Zoomed pulse structure, 0-{zoom_end_s:g} s")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current (uA)")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    return fig, axes
