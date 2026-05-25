"""Plotting and notebook widget helpers for stage 03 detection outputs."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .detection_core import (
    DetectionConfig,
    PREPAD_MS,
    RASTER_COLORS,
    THRESH_TYPES,
    FREQ_COLOR,
    build_detection_response_comparison_table,
    cv0_response_aggregate,
    detection_threshold_aggregate,
    load_input_pulse_onsets_s,
    normalize_threshold_frame,
    parse_numeric_list,
    summarize_cv0_responses,
    threshold_count_math_label,
    threshold_display_name,
    threshold_math_label,
    _set_axis_with_explicit_top_tick,
)


def _plot_threshold_grid(
    summary: pd.DataFrame,
    title: str,
    ylabel: str,
    value_col: str = "mean",
    cutline: int = 101,
    error_col: str = "sem",
    ylabel_by_row: Sequence[str] | None = None,
    label_fn=threshold_math_label,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    for ax, thresh_type in zip(axes, THRESH_TYPES):
        y_plot: list[np.ndarray] = []
        subset = summary[
            (summary["cutline"].astype(int) == int(cutline))
            & (summary["thresh_type"] == thresh_type)
        ]
        for freq in sorted(subset["freq"].astype(float).unique()):
            s = subset[np.isclose(subset["freq"].astype(float), float(freq))].sort_values("cv")
            x = s["cv"].astype(float).to_numpy()
            y = s[value_col].astype(float).to_numpy()
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                label=f"{int(freq)} Hz",
                color=FREQ_COLOR.get(float(freq)),
            )
            if error_col and error_col in s.columns:
                err = s[error_col].fillna(0).astype(float).to_numpy()
                y_plot.extend([y - err, y + err])
                ax.fill_between(x, y - err, y + err, color=FREQ_COLOR.get(float(freq)), alpha=0.12, linewidth=0)
            else:
                y_plot.append(y)
        if y_plot:
            _set_axis_with_explicit_top_tick(ax, np.concatenate(y_plot), target_ticks=6)
        ax.set_title(label_fn(thresh_type))
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.05, 1.05)
    for ax in axes[6:]:
        ax.set_xlabel("Coefficient of Variation (CV)")
    if ylabel_by_row is None:
        for ax in axes[::3]:
            ax.set_ylabel(ylabel)
    else:
        for row_idx, row_ylabel in enumerate(ylabel_by_row):
            axes[row_idx * 3].set_ylabel(row_ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Frequency", loc="lower center", ncol=5, frameon=False)
    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig, axes


def plot_spike_count_threshold_amplitudes(df_thresholds: pd.DataFrame, cutline: int = 101):
    summary = detection_threshold_aggregate(df_thresholds, value="amp")
    return _plot_threshold_grid(summary, "Spike-Count Threshold Amplitudes", "Threshold Amplitude (uA)", cutline=cutline)


def plot_threshold_spike_counts(df_thresholds: pd.DataFrame, cutline: int = 101):
    summary = detection_threshold_aggregate(df_thresholds, value="count")
    return _plot_threshold_grid(
        summary,
        "Threshold Spike Counts",
        "Threshold Spikes",
        cutline=cutline,
        error_col="sem",
        ylabel_by_row=("Threshold Spikes (#)", "Threshold Spikes (%)", "Threshold Spikes (%)"),
        label_fn=threshold_count_math_label,
    )


def plot_thresholds_by_cv(df_thresholds: pd.DataFrame, cutline: int = 101):
    return plot_spike_count_threshold_amplitudes(df_thresholds, cutline=cutline)


def plot_cv0_response_counts_grid(
    df_cv0_responses: pd.DataFrame,
    cutline: int = 101,
    error_mode: str = "sem",
):
    summary = cv0_response_aggregate(df_cv0_responses, error_mode=error_mode)
    error_col = "sd" if error_mode == "sd" else "sem"
    return _plot_threshold_grid(
        summary,
        "CV0-Based Threshold Spike Counts",
        "Threshold Spikes",
        cutline=cutline,
        error_col=error_col,
        ylabel_by_row=("Threshold Spikes (#)", "Threshold Spikes (%)", "Threshold Spikes (%)"),
        label_fn=threshold_count_math_label,
    )


def plot_cv0_response_counts(
    df_cv0_responses: pd.DataFrame,
    cutline: int = 101,
    freq: float | None = None,
    **kwargs,
):
    if freq is None:
        return plot_cv0_response_counts_grid(df_cv0_responses, cutline=cutline, **kwargs)
    import matplotlib.pyplot as plt

    summary = summarize_cv0_responses(df_cv0_responses, freq=freq, cutline=cutline)
    fig, ax = plt.subplots(figsize=(7, 4))
    for thresh_type, subset in summary.groupby("thresh_type"):
        ax.plot(subset["cv"], subset["mean_spikes"], marker="o", label=threshold_display_name(thresh_type))
    ax.set_title(f"CV0-based responses at {int(freq)} Hz")
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel("Mean spikes")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_threshold_search_trace(row: pd.Series, title: str | None = None):
    import matplotlib.pyplot as plt

    amps = parse_numeric_list(row.get("amps_tested_uA"))
    counts = parse_numeric_list(row.get("counts_tested"))
    fig, ax = plt.subplots(figsize=(7, 4))
    if amps.size and counts.size:
        order = np.argsort(amps)
        ax.plot(amps[order], counts[order], marker="o", linewidth=1.4)
    for thresh_type in THRESH_TYPES:
        amp = row.get(f"{thresh_type}_uA")
        if amp is not None and pd.notna(amp):
            ax.axvline(float(amp), color=RASTER_COLORS.get(thresh_type, "0.4"), alpha=0.25, linewidth=1.0)
    ax.set_xlabel("Amplitude (uA)")
    ax.set_ylabel("Spike count")
    ax.set_title(title or "Threshold search trace")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_threshold_response_raster(
    df_cv0_responses: pd.DataFrame,
    config: DetectionConfig | None = None,
    cv: float = 0.0,
    freq: float = 100.0,
    rep_index: int = 0,
    cutline: int = 101,
    thresh_types: Sequence[str] = THRESH_TYPES,
    xlim_s: tuple[float, float] = (0.0, 2.0),
    title: str | None = None,
):
    import matplotlib.pyplot as plt

    config = (config or DetectionConfig()).resolved()
    subset = df_cv0_responses[
        (df_cv0_responses["cutline"].astype(int) == int(cutline))
        & np.isclose(df_cv0_responses["cv"].astype(float), float(cv))
        & np.isclose(df_cv0_responses["freq"].astype(float), float(freq))
        & (df_cv0_responses["rep_index"].astype(int) == int(rep_index))
        & df_cv0_responses["thresh_type"].isin(thresh_types)
    ].copy()
    if subset.empty:
        raise ValueError(f"No response rows for cutline={cutline}, cv={cv}, freq={freq}, rep_index={rep_index}")

    subset["thresh_type"] = pd.Categorical(subset["thresh_type"], categories=list(thresh_types), ordered=True)
    subset = subset.sort_values("thresh_type").reset_index(drop=True)
    rep_label = str(subset["rep_label"].iloc[0]) if "rep_label" in subset.columns else None
    pulse_s = load_input_pulse_onsets_s(config=config, cv=cv, freq=freq, rep_index=rep_index, rep_label=rep_label)
    pulse_s = pulse_s[(pulse_s >= xlim_s[0]) & (pulse_s <= xlim_s[1])]

    n_rows = len(subset) + 1
    fig, ax = plt.subplots(figsize=(12, 6))
    y_input = n_rows
    if pulse_s.size:
        ax.eventplot(pulse_s, lineoffsets=y_input, linelengths=0.72, colors="black", linewidths=0.8)

    yticks = [y_input]
    ylabels = ["Input Pulses"]
    right_labels = [f"{pulse_s.size} pulses"]
    for idx, (_, row) in enumerate(subset.iterrows()):
        y = n_rows - idx - 1
        spike_s = (parse_numeric_list(row.get("spike_times_ms")) - PREPAD_MS) / 1000.0
        shown = spike_s[(spike_s >= xlim_s[0]) & (spike_s <= xlim_s[1])]
        color = RASTER_COLORS.get(str(row["thresh_type"]), "C0")
        if shown.size:
            ax.eventplot(shown, lineoffsets=y, linelengths=0.72, colors=color, linewidths=0.9)
        yticks.append(y)
        ylabels.append(f"{threshold_math_label(str(row['thresh_type']))}\n({float(row['amp_uA']):.2f} uA)")
        right_labels.append(f"{shown.size} spike" + ("" if shown.size == 1 else "s"))

    ax.set_xlim(*xlim_s)
    ax.set_ylim(0.35, n_rows + 0.9)
    ax.set_xlabel("Time (sec)")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    for tick_label in ax.get_yticklabels():
        tick_label.set_multialignment("center")
        tick_label.set_horizontalalignment("right")
    ax.grid(True, axis="x", alpha=0.22)
    title_text = title or f"{float(freq):g} Hz | CV={float(cv):g}"
    if rep_label and rep_label not in title_text:
        title_text += f" | {rep_label}"
    ax.set_title(title_text, fontsize=18)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(yticks)
    ax_right.set_yticklabels(right_labels)
    ax_right.tick_params(axis="y", length=0)
    ax_right.grid(False)
    fig.tight_layout()
    return fig, ax

def make_detection_response_comparison_widget(
    df_detection_responses: pd.DataFrame,
    df_cv0_responses: pd.DataFrame,
    config: DetectionConfig | None = None,
    default_freq: float = 100.0,
    default_cv: float = 0.7,
    default_rep_index: int = 5,
    cutline: int | None = None,
    thresh_types: Sequence[str] = THRESH_TYPES,
):
    """Create the stage-03 notebook widget for cached threshold-response inspection."""
    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, clear_output, display
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("ipywidgets and IPython are required for the stage-03 comparison widget.") from exc

    config = (config or DetectionConfig()).resolved()
    cutline = config.cutline if cutline is None else int(cutline)

    def available_frequencies() -> list[float]:
        rows = df_detection_responses[df_detection_responses["cutline"].astype(int) == int(cutline)]
        return sorted(rows["freq"].astype(float).unique().tolist())

    def available_cvs(freq: float) -> list[float]:
        true_rows = df_detection_responses[
            (df_detection_responses["cutline"].astype(int) == int(cutline))
            & np.isclose(df_detection_responses["freq"].astype(float), float(freq))
        ]
        replay_rows = df_cv0_responses[
            (df_cv0_responses["cutline"].astype(int) == int(cutline))
            & np.isclose(df_cv0_responses["freq"].astype(float), float(freq))
        ]
        cvs = sorted(set(true_rows["cv"].astype(float)).intersection(set(replay_rows["cv"].astype(float))))
        return [cv for cv in cvs if not np.isclose(cv, 0.0)]

    def available_reps(freq: float, compare_cv: float) -> list[int]:
        rep_sets: list[set[int]] = []
        for df, cv in ((df_detection_responses, 0.0), (df_detection_responses, compare_cv), (df_cv0_responses, compare_cv)):
            rows = df[
                (df["cutline"].astype(int) == int(cutline))
                & np.isclose(df["freq"].astype(float), float(freq))
                & np.isclose(df["cv"].astype(float), float(cv))
            ]
            rep_sets.append(set(rows["rep_index"].astype(int).unique().tolist()))
        return sorted(set.intersection(*rep_sets)) if rep_sets else []

    def set_dropdown_options(dropdown, options, preferred=None) -> None:
        options = list(options)
        dropdown.options = options
        if not options:
            dropdown.value = None
        elif preferred in options:
            dropdown.value = preferred
        elif dropdown.value not in options:
            dropdown.value = options[0]

    def display_table(table: pd.DataFrame) -> pd.DataFrame:
        shown = table.copy()
        for column in shown.columns:
            if column[0] == "amplitude (uA)":
                shown[column] = shown[column].map(lambda value: f"{float(value):.2f}")
            elif column[0] == "spikes":
                shown[column] = shown[column].astype(int)
        return shown

    frequency_options = available_frequencies()
    if not frequency_options:
        raise ValueError("No cached detection responses are available for the selected cutline.")

    freq_dropdown = widgets.Dropdown(
        options=frequency_options,
        value=float(default_freq) if float(default_freq) in frequency_options else frequency_options[0],
        description="freq (Hz)",
        layout=widgets.Layout(width="210px"),
    )
    cv_dropdown = widgets.Dropdown(description="CV", layout=widgets.Layout(width="180px"))
    rep_dropdown = widgets.Dropdown(description="rep", layout=widgets.Layout(width="180px"))
    output = widgets.Output()
    controls_ready = False

    def refresh_cv_and_rep_options() -> None:
        freq = float(freq_dropdown.value)
        cv_options = available_cvs(freq)
        set_dropdown_options(cv_dropdown, cv_options, preferred=float(default_cv) if float(default_cv) in cv_options else None)
        if cv_dropdown.value is not None:
            rep_options = available_reps(freq, float(cv_dropdown.value))
            set_dropdown_options(
                rep_dropdown,
                rep_options,
                preferred=int(default_rep_index) if int(default_rep_index) in rep_options else None,
            )

    def refresh_rep_options() -> None:
        if cv_dropdown.value is None:
            return
        rep_options = available_reps(float(freq_dropdown.value), float(cv_dropdown.value))
        set_dropdown_options(rep_dropdown, rep_options, preferred=rep_dropdown.value)

    def render(_=None) -> None:
        if not controls_ready or cv_dropdown.value is None or rep_dropdown.value is None:
            return
        freq = float(freq_dropdown.value)
        compare_cv = float(cv_dropdown.value)
        rep_index = int(rep_dropdown.value)
        with output:
            clear_output(wait=False)
            try:
                table = build_detection_response_comparison_table(
                    df_detection_responses,
                    df_cv0_responses,
                    config=config,
                    freq=freq,
                    compare_cv=compare_cv,
                    rep_index=rep_index,
                    cutline=cutline,
                    thresh_types=thresh_types,
                )
                display(Markdown(f"**Cached response comparison: {freq:g} Hz, comparison CV={compare_cv:g}, rep_index={rep_index}**"))
                display(display_table(table))

                plot_specs = [
                    ("CV0 true-threshold response", df_detection_responses, 0.0, f"CV0 true-threshold response: {freq:g} Hz | CV=0"),
                    ("CV example true-threshold response", df_detection_responses, compare_cv, f"CV example true-threshold response: {freq:g} Hz | CV={compare_cv:g}"),
                    ("CV0-based replay", df_cv0_responses, compare_cv, f"CV0-based replay: {freq:g} Hz | CV={compare_cv:g}"),
                ]
                for label, df, cv, title in plot_specs:
                    display(Markdown(f"**{label}**"))
                    fig, _ = plot_threshold_response_raster(
                        df,
                        config=config,
                        cv=cv,
                        freq=freq,
                        rep_index=rep_index,
                        cutline=cutline,
                        xlim_s=(0.0, 2.0),
                        title=title,
                    )
                    display(fig)
                    plt.close(fig)
            except Exception as exc:
                display(Markdown(f"**Could not render this selection:** `{exc}`"))

    def on_frequency_change(change) -> None:
        nonlocal controls_ready
        controls_ready = False
        refresh_cv_and_rep_options()
        controls_ready = True
        render()

    def on_cv_change(change) -> None:
        nonlocal controls_ready
        if not controls_ready:
            return
        controls_ready = False
        refresh_rep_options()
        controls_ready = True
        render()

    refresh_cv_and_rep_options()
    controls_ready = True
    freq_dropdown.observe(on_frequency_change, names="value")
    cv_dropdown.observe(on_cv_change, names="value")
    rep_dropdown.observe(render, names="value")
    render()
    return widgets.VBox([widgets.HBox([freq_dropdown, cv_dropdown, rep_dropdown]), output])

