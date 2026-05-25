"""Plotting and notebook widget helpers for stage 04/05 discrimination outputs."""

from __future__ import annotations

from typing import Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .discrimination_core import (
    BLOCKS,
    FEATURE_RATE_COLORS,
    FEATURE_TIMING_COLORS,
    FREQUENCIES,
    MODEL_LABELS,
    RELEASE_MODEL_BANKS,
    THRESH_ORDER,
    THRESH_REGIME_TICKS,
    DiscriminationData,
    FeatureEngineeringConfig,
    FeatureSimulation,
    FeatureWorkflow,
    collect_feature_evidence_matrices,
    feature_frequencies,
    feature_rep_options,
    feature_threshold_options,
    normalize_feature_evidence_summary,
    per_block_model_winners,
    per_block_weight_summary,
    per_point_weight_summary,
    prepare_feature_workflow,
    threshold_display_name,
    threshold_label,
    trial_feature_evidence,
    _as_float,
    _block_label,
    _json_value,
    _ordered_threshold_frame,
    _parse_num_dict,
    _sem0,
    _square_trace_from_events,
    _to_choice_display,
)

MODEL_COLORS: Mapping[str, str] = {
    "rate_combined": "#d62728",
    "timing_combined": "#1f77b4",
    "hybrid_combined": "#7b2cbf",
    "per_point_hybrid": "#f4a261",
}

THRESH_COLORS: Mapping[str, tuple[float, float, float, float]] = {
    thresh: plt.cm.viridis(x)
    for thresh, x in zip(THRESH_ORDER, np.linspace(0.05, 0.98, len(THRESH_ORDER)))
}


def build_feature_evidence_figure(
    workflow: FeatureWorkflow,
    *,
    freq: Optional[float] = None,
    thresh_type: Optional[str] = None,
    rep: Optional[int] = None,
    jnd_k: Optional[float] = None,
):
    """Build the stage-04 interactive feature/evidence figure."""

    
    freqs = feature_frequencies(workflow)
    if not freqs:
        raise RuntimeError("Feature workflow has no frequency options.")
    freq = float(freqs[-1] if freq is None else freq)

    thresh_options = feature_threshold_options(workflow, freq)
    if not thresh_options:
        raise RuntimeError(f"Feature workflow has no threshold options for freq={freq:g}.")
    thresh_type = str(thresh_options[0] if thresh_type is None else thresh_type)
    thresh_label = threshold_display_name(thresh_type)

    rep_options = feature_rep_options(workflow, freq, thresh_type)
    if not rep_options:
        raise RuntimeError(f"Feature workflow has no repetitions for freq={freq:g}, thresh_type={thresh_type}.")
    rep = int(rep_options[0] if rep is None else rep)
    jnd_k = float(workflow.config.jnd_k if jnd_k is None else jnd_k)

    cvs = sorted(float(cv) for cv in workflow.cvs_by_freq_thresh.get((float(freq), str(thresh_type)), ()))
    if (0.0 not in cvs) or (1.0 not in cvs):
        raise RuntimeError(f"Selection freq={freq:g}, thresh_type={thresh_type} is missing CV templates 0 and 1.")

    bins = tuple(float(b) for b in workflow.config.bins)
    if 1.0 not in bins or 10.0 not in bins:
        raise ValueError("The stage-04 figure expects config.bins to include 1.0 and 10.0 ms.")

    features = {1.0: {}, 10.0: {}}
    e_rate = {1.0: [], 10.0: []}
    e_time = {1.0: [], 10.0: []}
    rate_features = {"n_spikes_1": [], "n_events_10": [], "spe_10": []}
    timing_features = {"lv_1": [], "lv_10": []}

    for cv in cvs:
        er1, et1, f1 = trial_feature_evidence(
            workflow, freq=freq, thresh_type=thresh_type, cv=cv, rep=rep, bin_ms=1.0, jnd_k=jnd_k
        )
        er10, et10, f10 = trial_feature_evidence(
            workflow, freq=freq, thresh_type=thresh_type, cv=cv, rep=rep, bin_ms=10.0, jnd_k=jnd_k
        )
        features[1.0][cv] = f1
        features[10.0][cv] = f10
        e_rate[1.0].append(er1)
        e_time[1.0].append(et1)
        e_rate[10.0].append(er10)
        e_time[10.0].append(et10)
        rate_features["n_spikes_1"].append(float(f1.get("n_spikes", np.nan)))
        rate_features["n_events_10"].append(float(f10.get("n_events", np.nan)))
        rate_features["spe_10"].append(float(f10.get("spikes_per_event", np.nan)))
        timing_features["lv_1"].append(float(f1.get("silence_lv", np.nan)))
        timing_features["lv_10"].append(float(f10.get("silence_lv", np.nan)))

    reps_all = feature_rep_options(workflow, freq, thresh_type)
    er1_mat, et1_mat = collect_feature_evidence_matrices(
        workflow, freq=freq, thresh_type=thresh_type, bin_ms=1.0, reps=reps_all, cvs=cvs, jnd_k=jnd_k
    )
    er10_mat, et10_mat = collect_feature_evidence_matrices(
        workflow, freq=freq, thresh_type=thresh_type, bin_ms=10.0, reps=reps_all, cvs=cvs, jnd_k=jnd_k
    )
    norm1 = normalize_feature_evidence_summary(cvs, er1_mat, et1_mat)
    norm10 = normalize_feature_evidence_summary(cvs, er10_mat, et10_mat)

    cv_arr = np.asarray(cvs, dtype=float)
    rate_color = FEATURE_RATE_COLORS.get(thresh_type, "#E30613")
    timing_color = FEATURE_TIMING_COLORS.get(thresh_type, "#0575AB")
    cv_ticks_dense = np.round(np.arange(0.0, 1.01, 0.1), 1)
    cv_ticklabels_sparse = [f"{x:.1f}" if int(round(x * 10)) % 2 == 0 else "" for x in cv_ticks_dense]

    fig = plt.figure(figsize=(17.5, 13.4), constrained_layout=True)
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.10, 1.10, 0.95, 0.95, 1.00],
        width_ratios=[1.0, 1.0],
    )

    ax_raster = fig.add_subplot(gs[0, :])
    ax_env = fig.add_subplot(gs[1, :], sharex=ax_raster, sharey=ax_raster)
    ax_f_rate = fig.add_subplot(gs[2, 0])
    ax_f_time = fig.add_subplot(gs[2, 1])
    ax_e_rate = fig.add_subplot(gs[3, 0], sharex=ax_f_rate)
    ax_e_time = fig.add_subplot(gs[3, 1], sharex=ax_f_time)
    ax_m_rate = fig.add_subplot(gs[4, 0], sharex=ax_f_rate)
    ax_m_time = fig.add_subplot(gs[4, 1], sharex=ax_f_time)

    cmap_cv = plt.get_cmap("viridis")
    cv_levels = [0.10] if len(cvs) == 1 else np.linspace(0.10, 0.95, len(cvs))
    cv_colors = {cv: cmap_cv(float(cv_levels[i])) for i, cv in enumerate(cvs)}
    offsets = np.arange(len(cvs))[::-1]
    cv_to_offset = {cv: offsets[i] for i, cv in enumerate(cvs)}
    row_half = 0.40
    spike_half = row_half
    ymin = float(np.min(offsets) - row_half - 0.05)
    ymax = float(np.max(offsets) + row_half + 0.05)
    time_ticks = np.arange(float(workflow.config.t0_ms), float(workflow.config.t1_ms) + 1e-9, 250.0)
    time_xlim = (float(workflow.config.t0_ms) - 5.0, float(workflow.config.t1_ms) + 5.0)

    for cv in cvs:
        offset = float(cv_to_offset[cv])
        spikes = np.asarray(features[1.0].get(cv, {}).get("spikes", []), dtype=float)
        if spikes.size:
            ax_raster.vlines(spikes, offset - spike_half, offset + spike_half, linewidth=0.9, alpha=0.95, color=cv_colors[cv])

    ax_raster.set_title("Spikes (1 ms bins)")
    ax_raster.set_yticks(list(cv_to_offset.values()))
    ax_raster.set_yticklabels([f"{float(cv):.1f}" for cv in cvs])
    ax_raster.set_ylabel("Coefficient of Variation (CV)")
    ax_raster.set_ylim(ymin, ymax)
    ax_raster.set_xlim(*time_xlim)
    ax_raster.set_xticks(time_ticks)
    ax_raster.grid(True, axis="x", alpha=0.25)
    ax_raster.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

    for cv in cvs:
        offset = float(cv_to_offset[cv])
        events10 = features[10.0].get(cv, {}).get("events", [])
        x_sq, y_sq, x_single = _square_trace_from_events(events10, workflow.config.t0_ms, workflow.config.t1_ms)
        if x_sq.size >= 2:
            y_centered = offset + (2.0 * y_sq - 1.0) * row_half
            ax_env.step(x_sq, y_centered, where="post", linewidth=1.6, color=cv_colors[cv])
        if x_single.size:
            ax_env.vlines(x_single, offset - row_half, offset + row_half, linewidth=1.4, alpha=0.95, color=cv_colors[cv])

    ax_env.set_title("Bursts (10 ms bins)")
    ax_env.set_yticks(list(cv_to_offset.values()))
    ax_env.set_yticklabels([f"{float(cv):.1f}" for cv in cvs])
    ax_env.set_ylabel("Coefficient of Variation (CV)")
    ax_env.set_xticks(time_ticks)
    ax_env.set_xlabel("Time (ms)")
    ax_env.grid(True, axis="x", alpha=0.25)

    ax_f_rate.plot(
        cv_arr,
        np.asarray(rate_features["n_spikes_1"], float),
        marker="o",
        linewidth=2.0,
        linestyle=":",
        color=rate_color,
        label="n_spikes (1 ms)",
    )
    ax_f_rate_b = ax_f_rate.twinx()
    ax_f_rate_b.plot(
        cv_arr,
        np.asarray(rate_features["n_events_10"], float),
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color=rate_color,
        label="n_bursts (10 ms)",
    )
    ax_f_rate_b.plot(
        cv_arr,
        np.asarray(rate_features["spe_10"], float),
        marker="o",
        linewidth=2.2,
        linestyle=(0, (3, 1, 1, 1)),
        color=rate_color,
        label="spikes_per_burst (10 ms)",
    )

    spike_vals = np.asarray(rate_features["n_spikes_1"], float)
    spike_vals = spike_vals[np.isfinite(spike_vals)]
    if spike_vals.size:
        s_min = float(np.min(spike_vals))
        s_max = float(np.max(spike_vals))
        s_span = max(1e-6, s_max - s_min)
        ax_f_rate.set_ylim(s_min - 0.08 * s_span, s_max + 0.36 * s_span)

    burst_vals = np.concatenate([
        np.asarray(rate_features["n_events_10"], float),
        np.asarray(rate_features["spe_10"], float),
    ])
    burst_vals = burst_vals[np.isfinite(burst_vals)]
    if burst_vals.size:
        b_min = float(np.min(burst_vals))
        b_max = float(np.max(burst_vals))
        b_span = max(1e-6, b_max - b_min)
        b_lo = max(0.0, b_min - 0.15 * b_span)
        b_hi = b_max + 1.05 * b_span
        if b_hi <= b_lo:
            b_hi = b_lo + 1.0
        ax_f_rate_b.set_ylim(b_lo, b_hi)

    ax_f_rate.set_title(r"Rate Features ($F_{rate}$)")
    ax_f_rate.set_ylabel("Spike Metric")
    ax_f_rate_b.set_ylabel("Burst Metric")
    ax_f_rate_b.spines["right"].set_visible(True)
    ax_f_rate_b.spines["right"].set_linewidth(1.0)
    ax_f_rate_b.spines["right"].set_color("0.35")
    ax_f_rate_b.tick_params(axis="y", which="both", right=True, labelright=True, direction="out", length=4, pad=4)
    ax_f_rate.grid(False)
    ax_f_rate.set_xticks(cv_ticks_dense)
    ax_f_rate.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    rate_feat_left_handles = [
        plt.Line2D([0], [0], color=rate_color, linestyle=":", linewidth=2.4, marker="None", label="n_spikes"),
    ]
    rate_feat_right_handles = [
        plt.Line2D([0], [0], color=rate_color, linestyle="--", linewidth=2.4, marker="None", label="n_bursts"),
        plt.Line2D([0], [0], color=rate_color, linestyle=(0, (3, 1, 1, 1)), linewidth=2.4, marker="None", label="spikes_per_burst"),
    ]
    leg_primary = ax_f_rate.legend(
        handles=rate_feat_left_handles,
        title="Primary Axis",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.06),
        borderaxespad=0.0,
        handlelength=4.6,
        handletextpad=0.6,
    )
    leg_secondary = ax_f_rate.legend(
        handles=rate_feat_right_handles,
        title="Secondary Axis",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.06),
        borderaxespad=0.0,
        handlelength=4.6,
        handletextpad=0.6,
    )
    ax_f_rate.add_artist(leg_primary)

    ax_f_time.plot(
        cv_arr,
        np.asarray(timing_features["lv_1"], float),
        marker="o",
        linewidth=2.0,
        linestyle=":",
        color=timing_color,
        label="LV_ISI",
    )
    ax_f_time_b = ax_f_time.twinx()
    ax_f_time_b.plot(
        cv_arr,
        np.asarray(timing_features["lv_10"], float),
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color=timing_color,
        label="LV_IBI",
    )

    lv_isi_vals = np.asarray(timing_features["lv_1"], float)
    lv_isi_vals = lv_isi_vals[np.isfinite(lv_isi_vals)]
    if lv_isi_vals.size:
        li_min = float(np.min(lv_isi_vals))
        li_max = float(np.max(lv_isi_vals))
        li_span = max(1e-6, li_max - li_min)
        ax_f_time.set_ylim(li_min - 0.08 * li_span, li_max + 0.30 * li_span)

    lv_ibi_vals = np.asarray(timing_features["lv_10"], float)
    lv_ibi_vals = lv_ibi_vals[np.isfinite(lv_ibi_vals)]
    if lv_ibi_vals.size:
        lb_min = float(np.min(lv_ibi_vals))
        lb_max = float(np.max(lv_ibi_vals))
        lb_span = max(1e-6, lb_max - lb_min)
        ax_f_time_b.set_ylim(lb_min - 0.15 * lb_span, lb_max + 1.05 * lb_span)

    ax_f_time.set_title(r"Timing Features ($F_{timing}$)")
    ax_f_time.set_ylabel("Spike Metric")
    ax_f_time_b.set_ylabel("Burst Metric")
    ax_f_time_b.spines["right"].set_visible(True)
    ax_f_time_b.spines["right"].set_linewidth(1.0)
    ax_f_time_b.spines["right"].set_color("0.35")
    ax_f_time_b.tick_params(axis="y", which="both", right=True, labelright=True, direction="out", length=4, pad=4)
    ax_f_time.grid(False)
    ax_f_time.set_xticks(cv_ticks_dense)
    ax_f_time.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    time_feat_left_handles = [
        plt.Line2D([0], [0], color=timing_color, linestyle=":", linewidth=2.4, marker="None", label="LV_ISI"),
    ]
    time_feat_right_handles = [
        plt.Line2D([0], [0], color=timing_color, linestyle="--", linewidth=2.4, marker="None", label="LV_IBI"),
    ]
    leg_time_primary = ax_f_time.legend(
        handles=time_feat_left_handles,
        title="Primary Axis",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.06),
        borderaxespad=0.0,
        handlelength=4.6,
        handletextpad=0.6,
    )
    leg_time_secondary = ax_f_time.legend(
        handles=time_feat_right_handles,
        title="Secondary Axis",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.06),
        borderaxespad=0.0,
        handlelength=4.6,
        handletextpad=0.6,
    )
    ax_f_time.add_artist(leg_time_primary)

    i0 = int(np.where(np.isclose(cv_arr, 0.0))[0][0])
    i1 = int(np.where(np.isclose(cv_arr, 1.0))[0][0])
    e_time_plot = {}
    for bin_ms in (1.0, 10.0):
        arr = np.asarray(e_time[bin_ms], dtype=float)
        den = float(arr[i1] - arr[i0]) if arr.size else np.nan
        if (not np.isfinite(den)) or abs(den) < 1e-12:
            e_time_plot[bin_ms] = np.zeros_like(arr)
        else:
            e_time_plot[bin_ms] = np.clip(2.0 * (arr - float(arr[i0])) / den - 1.0, -1.0, 1.0)

    ax_e_rate.plot(
        cv_arr,
        np.asarray(e_rate[1.0], float),
        marker="o",
        linewidth=2.0,
        linestyle=":",
        color=rate_color,
        label="Spikes (1 ms)",
    )
    ax_e_rate.plot(
        cv_arr,
        np.asarray(e_rate[10.0], float),
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color=rate_color,
        label="Bursts (10 ms)",
    )
    ax_e_rate.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax_e_rate.set_ylim(-1.05, 1.05)
    ax_e_rate.set_title(r"Rate Evidence ($E_{rate}$)")
    ax_e_rate.set_xlabel("Coefficient of Variation (CV)")
    ax_e_rate.set_ylabel("Normalized Evidence", labelpad=0)
    ax_e_rate.set_xticks(cv_ticks_dense)
    ax_e_rate.set_xticklabels(cv_ticklabels_sparse)
    ax_e_rate.grid(True, alpha=0.25)
    rate_ev_handles = [
        plt.Line2D([0], [0], color=rate_color, linestyle=":", linewidth=2.0, marker="None", label="Spikes (1 ms)"),
        plt.Line2D([0], [0], color=rate_color, linestyle="--", linewidth=2.0, marker="None", label="Bursts (10 ms)"),
    ]
    ax_e_rate.legend(handles=rate_ev_handles, frameon=False, fontsize=8, handlelength=4.6, handletextpad=0.6)

    ax_e_time.plot(
        cv_arr,
        np.asarray(e_time_plot[1.0], float),
        marker="o",
        linewidth=2.0,
        linestyle=":",
        color=timing_color,
        label="Spikes (1 ms)",
    )
    ax_e_time.plot(
        cv_arr,
        np.asarray(e_time_plot[10.0], float),
        marker="o",
        linewidth=2.0,
        linestyle="--",
        color=timing_color,
        label="Bursts (10 ms)",
    )
    ax_e_time.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax_e_time.set_ylim(-1.05, 1.05)
    ax_e_time.set_title(r"Timing Evidence ($E_{timing}$)")
    ax_e_time.set_xlabel("Coefficient of Variation (CV)")
    ax_e_time.set_ylabel("Normalized Evidence", labelpad=0)
    ax_e_time.set_xticks(cv_ticks_dense)
    ax_e_time.set_xticklabels(cv_ticklabels_sparse)
    ax_e_time.grid(True, alpha=0.25)
    time_ev_handles = [
        plt.Line2D([0], [0], color=timing_color, linestyle=":", linewidth=2.0, marker="None", label="Spikes (1 ms)"),
        plt.Line2D([0], [0], color=timing_color, linestyle="--", linewidth=2.0, marker="None", label="Bursts (10 ms)"),
    ]
    ax_e_time.legend(handles=time_ev_handles, frameon=False, fontsize=8, handlelength=4.6, handletextpad=0.6)

    y1 = np.asarray(norm1["er_mean"], float)
    s1 = np.asarray(norm1["er_sem"], float)
    y10 = np.asarray(norm10["er_mean"], float)
    s10 = np.asarray(norm10["er_sem"], float)
    ax_m_rate.fill_between(cv_arr, np.clip(y1 - s1, -1.0, 1.0), np.clip(y1 + s1, -1.0, 1.0), color=rate_color, alpha=0.10, linewidth=0)
    ax_m_rate.fill_between(cv_arr, np.clip(y10 - s10, -1.0, 1.0), np.clip(y10 + s10, -1.0, 1.0), color=rate_color, alpha=0.16, linewidth=0)
    ax_m_rate.plot(cv_arr, np.clip(y1, -1.0, 1.0), marker="o", linewidth=2.0, linestyle=":", color=rate_color, label="Spikes (1 ms)")
    ax_m_rate.plot(cv_arr, np.clip(y10, -1.0, 1.0), marker="o", linewidth=2.0, linestyle="--", color=rate_color, label="Bursts (10 ms)")
    ax_m_rate.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax_m_rate.set_ylim(-1.05, 1.05)
    ax_m_rate.set_title(r"Mean Rate Evidence ($\hat{E}_{rate}$)")
    ax_m_rate.set_xlabel("Coefficient of Variation (CV)")
    ax_m_rate.set_ylabel("Reliability Scaled\nNormalized Evidence")
    ax_m_rate.set_xticks(cv_ticks_dense)
    ax_m_rate.set_xticklabels(cv_ticklabels_sparse)
    ax_m_rate.grid(True, alpha=0.25)
    ax_m_rate.legend(handles=rate_ev_handles, frameon=False, fontsize=8, handlelength=4.6, handletextpad=0.6)

    y1t = np.asarray(norm1["et_mean"], float)
    s1t = np.asarray(norm1["et_sem"], float)
    y10t = np.asarray(norm10["et_mean"], float)
    s10t = np.asarray(norm10["et_sem"], float)
    ax_m_time.fill_between(cv_arr, np.clip(y1t - s1t, -1.0, 1.0), np.clip(y1t + s1t, -1.0, 1.0), color=timing_color, alpha=0.10, linewidth=0)
    ax_m_time.fill_between(cv_arr, np.clip(y10t - s10t, -1.0, 1.0), np.clip(y10t + s10t, -1.0, 1.0), color=timing_color, alpha=0.16, linewidth=0)
    ax_m_time.plot(cv_arr, np.clip(y1t, -1.0, 1.0), marker="o", linewidth=2.0, linestyle=":", color=timing_color, label="Spikes (1 ms)")
    ax_m_time.plot(cv_arr, np.clip(y10t, -1.0, 1.0), marker="o", linewidth=2.0, linestyle="--", color=timing_color, label="Bursts (10 ms)")
    ax_m_time.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax_m_time.set_ylim(-1.05, 1.05)
    ax_m_time.set_title(r"Mean Timing Evidence ($\hat{E}_{timing}$)")
    ax_m_time.set_xlabel("Coefficient of Variation (CV)")
    ax_m_time.set_ylabel("Normalized Evidence")
    ax_m_time.set_xticks(cv_ticks_dense)
    ax_m_time.set_xticklabels(cv_ticklabels_sparse)
    ax_m_time.grid(True, alpha=0.25)
    ax_m_time.legend(handles=time_ev_handles, frameon=False, fontsize=8, handlelength=4.6, handletextpad=0.6)

    fig.suptitle(
        f"{thresh_label} Single Repetition Feature Engineering ({int(round(freq))} Hz, rep {rep}, JND k={jnd_k:g})",
        y=1.03,
    )
    return fig


def make_feature_evidence_widget(
    workflow_or_simulation: Union[FeatureWorkflow, FeatureSimulation, pd.DataFrame],
    config: Optional[FeatureEngineeringConfig] = None,
    default_freq: float = 200.0,
    default_thresh_type: str = "I_25",
    default_rep: int = 0,
    default_jnd_k: Optional[float] = None,
):
    """Create the interactive stage-04 dropdown widget."""

    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display
    from ipywidgets import HBox, Layout, VBox

    workflow = workflow_or_simulation if isinstance(workflow_or_simulation, FeatureWorkflow) else prepare_feature_workflow(workflow_or_simulation, config=config)
    out = widgets.Output()
    state = {"suppress_render": False}

    freq_options = feature_frequencies(workflow)
    if not freq_options:
        raise RuntimeError("No frequency options are available for the feature widget.")

    init_freq = float(default_freq) if any(np.isclose(float(default_freq), f) for f in freq_options) else float(freq_options[-1])
    init_thresh_options = feature_threshold_options(workflow, init_freq)
    init_thresh = str(default_thresh_type) if str(default_thresh_type) in init_thresh_options else (init_thresh_options[0] if init_thresh_options else None)
    init_rep_options = feature_rep_options(workflow, init_freq, init_thresh) if init_thresh is not None else []
    init_rep = int(default_rep) if int(default_rep) in init_rep_options else (init_rep_options[0] if init_rep_options else None)
    init_jnd = float(workflow.config.jnd_k if default_jnd_k is None else default_jnd_k)

    w_freq = widgets.Dropdown(options=freq_options, value=init_freq, description="freq (Hz)", layout=Layout(width="170px"))
    w_thresh = widgets.Dropdown(
        options=[(threshold_display_name(option), option) for option in init_thresh_options],
        value=init_thresh,
        description="thresh",
        layout=Layout(width="220px"),
    )
    w_rep = widgets.Dropdown(options=init_rep_options, value=init_rep, description="rep", layout=Layout(width="170px"))

    def _loading_message():
        return HTML(
            "<div style=\"padding:16px 0;color:#555;font-size:14px;\">"
            "Updating feature/evidence plot..."
            "</div>"
        )

    def _update_thresh_options():
        options = feature_threshold_options(workflow, float(w_freq.value))
        w_thresh.options = [(threshold_display_name(option), option) for option in options]
        w_thresh.value = options[0] if options else None
        _update_rep_options()

    def _update_rep_options():
        if w_thresh.value is None:
            w_rep.options = []
            w_rep.value = None
            return
        options = feature_rep_options(workflow, float(w_freq.value), str(w_thresh.value))
        w_rep.options = options
        w_rep.value = options[0] if options else None

    def _on_freq_change(*_args):
        if state["suppress_render"]:
            return
        state["suppress_render"] = True
        try:
            _update_thresh_options()
        finally:
            state["suppress_render"] = False
        _render()

    def _on_thresh_change(*_args):
        if state["suppress_render"]:
            return
        state["suppress_render"] = True
        try:
            _update_rep_options()
        finally:
            state["suppress_render"] = False
        _render()

    def _render(*_args):
        if state["suppress_render"]:
            return
        with out:
            clear_output(wait=False)
            display(_loading_message())
        if w_thresh.value is None or w_rep.value is None:
            with out:
                clear_output(wait=False)
                print("No valid frequency, threshold, and repetition selection.")
            return
        try:
            fig = build_feature_evidence_figure(
                workflow,
                freq=float(w_freq.value),
                thresh_type=str(w_thresh.value),
                rep=int(w_rep.value),
                jnd_k=init_jnd,
            )
        except Exception as exc:
            with out:
                clear_output(wait=False)
                print(f"Could not render feature evidence plot: {exc}")
            return
        with out:
            clear_output(wait=False)
            display(fig)
        plt.close(fig)

    w_freq.observe(_on_freq_change, names="value")
    w_thresh.observe(_on_thresh_change, names="value")
    w_rep.observe(_render, names="value")

    box = VBox([HBox([w_freq, w_thresh, w_rep], layout=Layout(flex_flow="row wrap")), out])
    _render()
    return box
def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_model_comparison(data: DiscriminationData) -> plt.Figure:
    """Plot release discrimination RMSE for per-block and per-point fits."""

    pb = data.per_block_summary.copy()
    pp = data.per_point_summary.copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, dpi=150)
    for ax, block in zip(axes, BLOCKS):
        sub = pb[pb["block"] == block]
        for model_bank in RELEASE_MODEL_BANKS:
            m = sub[sub["model_bank"] == model_bank].sort_values("freq")
            ax.errorbar(
                m["freq"],
                m["rmse_test_mean"],
                yerr=m["rmse_test_sem"],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                capsize=2.5,
                color=MODEL_COLORS[model_bank],
                label=MODEL_LABELS[model_bank],
            )
        pps = pp[pp["block"] == block].sort_values("freq")
        ax.errorbar(
            pps["freq"],
            pps["rmse_test_mean"],
            yerr=pps["rmse_test_sem"],
            marker="s",
            linewidth=1.9,
            linestyle="--",
            markersize=4.5,
            capsize=2.5,
            color=MODEL_COLORS["per_point_hybrid"],
            label=MODEL_LABELS["per_point_hybrid"],
        )
        ax.set_title(_block_label(block))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xticks(list(FREQUENCIES))
        _style_axes(ax)
    axes[0].set_ylabel("Test RMSE")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Manuscript Discrimination Model Comparison", y=1.02)
    fig.tight_layout()
    return fig


def plot_psychometric_fits(
    data: DiscriminationData,
    *,
    include_per_point: bool = True,
    winner_override: Optional[Mapping[str, str]] = None,
) -> plt.Figure:
    """Reproduce the manuscript psychometric curve-matching figure."""

    pb = data.per_block_curve_summary.copy()
    pp = data.per_point_curve_summary.copy()
    slack = data.slack_mean.copy()
    winners = dict(winner_override or per_block_model_winners(data))
    xticks_dense = np.round(np.arange(0.0, 1.01, 0.1), 1)
    xticklabels_sparse = [
        f"{x:.1f}" if (int(round(x * 10)) % 2 == 0) else ""
        for x in xticks_dense
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for row_idx, block in enumerate(BLOCKS):
        for col_idx, freq in enumerate(FREQUENCIES):
            ax = axes[row_idx, col_idx]

            if not slack.empty:
                obs = slack[(slack["block"] == block) & np.isclose(slack["freq"], freq)].copy().sort_values("cv")
                if not obs.empty and {"p_mean", "p_sem"}.issubset(obs.columns):
                    x_obs = obs["cv"].to_numpy(float)
                    y_obs = _to_choice_display(x_obs, obs["p_mean"].to_numpy(float), block)
                    yerr = np.nan_to_num(obs["p_sem"].to_numpy(float), nan=0.0)
                    ax.errorbar(
                        x_obs,
                        y_obs,
                        yerr=yerr,
                        fmt="o",
                        color="black",
                        markersize=5,
                        capsize=2,
                        linewidth=1.2,
                        zorder=6,
                    )

            panel = pb[(pb["block"] == block) & np.isclose(pb["freq"], freq)].copy()
            for model_bank in RELEASE_MODEL_BANKS:
                sub = panel[panel["model_bank"] == model_bank].sort_values("cv")
                if sub.empty:
                    continue
                x = sub["cv"].to_numpy(float)
                y = _to_choice_display(x, sub["p_pred_test_mean"].to_numpy(float), block)
                sem = np.nan_to_num(sub["p_pred_test_sem"].to_numpy(float), nan=0.0)
                is_winner = model_bank == winners.get(block)
                linewidth = 2.6 if is_winner else 1.3
                alpha = 0.95 if is_winner else 0.35
                ax.plot(
                    x,
                    y,
                    color=MODEL_COLORS[model_bank],
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=3,
                )
                if is_winner:
                    ax.fill_between(
                        x,
                        np.clip(y - sem, 0, 1),
                        np.clip(y + sem, 0, 1),
                        color=MODEL_COLORS[model_bank],
                        alpha=0.16,
                        zorder=2,
                    )

            if include_per_point:
                pps = pp[(pp["block"] == block) & np.isclose(pp["freq"], freq)].copy().sort_values("cv")
                if not pps.empty:
                    x_pcv = pps["cv"].to_numpy(float)
                    y_pcv = _to_choice_display(x_pcv, pps["p_pred_test_mean"].to_numpy(float), block)
                    se_pcv = np.nan_to_num(pps["p_pred_test_sem"].to_numpy(float), nan=0.0)
                    ax.errorbar(
                        x_pcv,
                        y_pcv,
                        yerr=se_pcv,
                        fmt="s",
                        linestyle="none",
                        color=MODEL_COLORS["per_point_hybrid"],
                        ecolor=MODEL_COLORS["per_point_hybrid"],
                        markersize=7.0,
                        markeredgewidth=0.0,
                        elinewidth=1.0,
                        capsize=2,
                        alpha=0.95,
                        zorder=4,
                    )

            ax.set_title(f"{block} | {int(freq)} Hz")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.25)
            ax.set_xticks(xticks_dense)
            ax.tick_params(axis="x", labelbottom=(row_idx == 1))
            if row_idx == 1:
                ax.set_xticklabels(xticklabels_sparse)
                ax.set_xlabel("Coefficient of Variation (CV)")
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel("P(CV1)" if block == "std0" else "P(CV0)")

    handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS["rate_combined"], lw=2, label="Rate"),
        plt.Line2D([0], [0], color=MODEL_COLORS["timing_combined"], lw=2, label="Timing"),
        plt.Line2D([0], [0], color=MODEL_COLORS["hybrid_combined"], lw=2, label="Hybrid"),
        plt.Line2D(
            [0],
            [0],
            color=MODEL_COLORS["per_point_hybrid"],
            marker="s",
            linestyle="none",
            markersize=7.5,
            label="Per-Point Mixture",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False)
    fig.suptitle("Behavioral Psychometric Curve Matching (Per-Block Curves + Per-Point Mixture Points)", y=1.02)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    return fig


def plot_per_block_weights(data: DiscriminationData) -> plt.Figure:
    """Plot the threshold mixtures retained for per-block release fits."""

    weights = per_block_weight_summary(data)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.3), sharey=True, dpi=150)
    for r, block in enumerate(BLOCKS):
        for c, model_bank in enumerate(RELEASE_MODEL_BANKS):
            ax = axes[r, c]
            sub = weights[(weights["block"] == block) & (weights["model_bank"] == model_bank)].copy()
            sub = _ordered_threshold_frame(sub)
            colors = [THRESH_COLORS[str(t)] for t in sub["thresh_type"]]
            x = np.arange(len(sub))
            ax.bar(
                x,
                sub["weight_mean"],
                yerr=sub["weight_sem"],
                color=colors,
                edgecolor="white",
                linewidth=0.7,
                capsize=2.0,
            )
            ax.set_title(f"{_block_label(block)} | {MODEL_LABELS[model_bank]}")
            ax.set_xticks(x)
            ax.set_xticklabels([threshold_label(str(t)) for t in sub["thresh_type"]], rotation=45, ha="right")
            ax.set_ylim(0, max(0.4, float(weights["weight_mean"].max()) * 1.2))
            _style_axes(ax)
            if c == 0:
                ax.set_ylabel("Mean mixture weight")
    fig.suptitle("Per-Block Threshold Mixture Weights", y=1.02)
    fig.tight_layout()
    return fig


def plot_per_point_weights(
    data: DiscriminationData,
    *,
    freq: float = 100.0,
    block: str = "std0",
) -> plt.Figure:
    """Plot CV-varying threshold weights for one per-point hybrid panel."""

    weights = per_point_weight_summary(data)
    sub = weights[(np.isclose(weights["freq"], float(freq))) & (weights["block"] == block)].copy()
    if sub.empty:
        raise ValueError(f"No per-point weights for freq={freq}, block={block}")
    cvs = np.array(sorted(sub["cv"].unique()), dtype=float)
    y_stack = []
    for thresh_type in THRESH_ORDER:
        t = sub[sub["thresh_type"].astype(str) == thresh_type].sort_values("cv")
        y_stack.append(t["weight_mean"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)
    ax.stackplot(
        cvs,
        *y_stack,
        colors=[THRESH_COLORS[t] for t in THRESH_ORDER],
        alpha=0.92,
        linewidth=0.0,
    )
    ax.set_xlim(float(cvs.min()), float(cvs.max()))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Coefficient of variation (CV)")
    ax.set_ylabel("Mixture weight")
    ax.set_title(f"Per-Point Hybrid Threshold Weights ({int(freq)} Hz, {_block_label(block)})")
    _style_axes(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=THRESH_COLORS[t], edgecolor="none", label=threshold_label(t))
        for t in THRESH_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    return fig


def plot_regime_mixture_comparison(
    data: DiscriminationData,
    *,
    winner_override: Optional[Mapping[str, str]] = None,
) -> plt.Figure:
    """Reproduce the manuscript hybrid regime-weight comparison figure."""

    winners = dict(winner_override or per_block_model_winners(data))
    base = data.per_block_results.copy()
    base = base[base["model_bank"].isin(RELEASE_MODEL_BANKS)].copy()
    point_weights = per_point_weight_summary(data)
    if point_weights.empty:
        raise RuntimeError("No per-point weights found in the release discrimination data.")
    panel_weight_summary = (
        point_weights.pivot_table(
            index=["block", "freq", "cv"],
            columns="thresh_type",
            values="weight_mean",
            aggfunc="mean",
            observed=False,
        )
        .reset_index()
        .sort_values(["block", "freq", "cv"])
    )
    for thresh_type in THRESH_ORDER:
        if thresh_type not in panel_weight_summary.columns:
            panel_weight_summary[thresh_type] = 0.0

    fig = plt.figure(figsize=(20, 7.35), constrained_layout=False)
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[0.84, 3.66], wspace=0.065)
    gs_left = outer_gs[0, 0].subgridspec(2, 1, hspace=0.20)
    gs_right = outer_gs[0, 1].subgridspec(2, 4, hspace=0.22, wspace=0.10)

    left_axes = [fig.add_subplot(gs_left[0, 0]), fig.add_subplot(gs_left[1, 0])]
    left_axes[1].sharex(left_axes[0])
    left_axes[1].sharey(left_axes[0])

    right_axes: list[list[plt.Axes]] = []
    sharey_right = None
    for row_idx in range(2):
        row_axes = []
        for col_idx in range(4):
            ax = fig.add_subplot(gs_right[row_idx, col_idx], sharey=sharey_right)
            if sharey_right is None:
                sharey_right = ax
            row_axes.append(ax)
        right_axes.append(row_axes)

    x_reg = np.arange(len(THRESH_ORDER), dtype=float)
    xticks_cv = np.round(np.arange(0.0, 1.01, 0.1), 1)
    xticklabels_cv_sparse = [
        f"{value:.1f}" if np.isclose((value * 10) % 2, 0) else ""
        for value in xticks_cv
    ]

    left_ymax = 0.0
    for block in BLOCKS:
        for model_bank in RELEASE_MODEL_BANKS:
            sub = base[(base["block"] == block) & (base["model_bank"] == model_bank)].copy()
            dedup_cols = [col for col in ["mc", "weights_json"] if col in sub.columns]
            if dedup_cols:
                sub = sub.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols)
            vals = []
            for row in sub.itertuples(index=False):
                weights = _parse_num_dict(getattr(row, "weights_json", {}))
                vals.append([float(weights.get(thresh_type, np.nan)) for thresh_type in THRESH_ORDER])
            if vals:
                left_ymax = max(left_ymax, float(np.nanmax(np.asarray(vals, dtype=float))))
    left_ylim_top = max(0.36, np.ceil((left_ymax + 0.01) / 0.02) * 0.02)

    for row_idx, block in enumerate(BLOCKS):
        ax = left_axes[row_idx]
        for model_bank in RELEASE_MODEL_BANKS:
            sub = base[(base["block"] == block) & (base["model_bank"] == model_bank)].copy()
            dedup_cols = [col for col in ["mc", "weights_json"] if col in sub.columns]
            if dedup_cols:
                sub = sub.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols)
            vals = []
            for row in sub.itertuples(index=False):
                weights = _parse_num_dict(getattr(row, "weights_json", {}))
                vals.append([float(weights.get(thresh_type, np.nan)) for thresh_type in THRESH_ORDER])
            if not vals:
                continue
            y_all = np.asarray(vals, dtype=float)
            y = np.nanmean(y_all, axis=0)
            se = np.nanstd(y_all, axis=0, ddof=1) / np.sqrt(y_all.shape[0]) if y_all.shape[0] > 1 else np.zeros(y_all.shape[1])
            is_winner = model_bank == winners.get(block)
            linewidth = 2.7 if is_winner else 1.35
            alpha = 0.98 if is_winner else 0.38
            zorder = 4 if is_winner else 2
            ax.plot(
                x_reg,
                y,
                color=MODEL_COLORS[model_bank],
                linewidth=linewidth,
                alpha=alpha,
                marker="o",
                markersize=4,
                zorder=zorder,
            )
            if is_winner:
                ax.fill_between(
                    x_reg,
                    y - se,
                    y + se,
                    color=MODEL_COLORS[model_bank],
                    alpha=0.12,
                    zorder=zorder - 1,
                )

        ax.set_title(f"{block} | Per-Block")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.2, len(THRESH_ORDER) - 0.8)
        ax.set_ylim(0.0, left_ylim_top)
        ax.set_xticks(x_reg)
        if row_idx == 1:
            ax.set_xticklabels([THRESH_REGIME_TICKS[t] for t in THRESH_ORDER], rotation=30, ha="right")
            ax.set_xlabel("Spike-Count Regime")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel("Weight")

    for row_idx, block in enumerate(BLOCKS):
        for col_idx, freq in enumerate(FREQUENCIES):
            ax = right_axes[row_idx][col_idx]
            sub = panel_weight_summary[
                (panel_weight_summary["block"] == block)
                & np.isclose(panel_weight_summary["freq"], freq)
            ].copy()
            sub = sub.sort_values("cv")
            if sub.empty:
                ax.set_title(f"{block} | {int(freq)} Hz (no data)")
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.25)
                continue

            x = sub["cv"].to_numpy(float)
            y_stack = [sub[thresh_type].to_numpy(float) for thresh_type in THRESH_ORDER]
            ax.stackplot(
                x,
                *y_stack,
                colors=[THRESH_COLORS[t] for t in THRESH_ORDER],
                alpha=0.92,
                linewidth=0.0,
            )
            ax.set_title(f"{block} | {int(freq)} Hz")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0, 1)
            ax.set_xticks(xticks_cv)
            ax.grid(True, alpha=0.20)
            if row_idx == 1:
                ax.set_xticklabels(xticklabels_cv_sparse)
                ax.set_xlabel("Coefficient of Variation (CV)")
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel("Regime Mass", labelpad=6)
            else:
                ax.tick_params(axis="y", labelleft=False)

    model_handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS["rate_combined"], lw=2.0, label="Rate"),
        plt.Line2D([0], [0], color=MODEL_COLORS["timing_combined"], lw=2.0, label="Timing"),
        plt.Line2D([0], [0], color=MODEL_COLORS["hybrid_combined"], lw=2.0, label="Hybrid"),
    ]
    regime_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=THRESH_COLORS[t], edgecolor="none", label=THRESH_REGIME_TICKS[t])
        for t in THRESH_ORDER
    ]

    fig.suptitle("Hybrid Regime Weights: Per-Block vs Per-Point Mixture", y=1.01)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.14)

    left_bbox_top = left_axes[0].get_position()
    right_bbox_tl = right_axes[0][0].get_position()
    right_bbox_tr = right_axes[0][-1].get_position()
    right_group_center = (right_bbox_tl.x0 + right_bbox_tr.x1) / 2

    fig.text(left_bbox_top.x0 - 0.028, left_bbox_top.y1 + 0.040, "a", fontsize=16, fontweight="bold")
    fig.text(right_bbox_tl.x0 - 0.028, right_bbox_tl.y1 + 0.040, "b", fontsize=16, fontweight="bold")

    fig.legend(
        handles=regime_handles,
        loc="lower center",
        bbox_to_anchor=(right_group_center, -0.005),
        ncol=9,
        frameon=False,
    )
    left_axes[0].legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
    )
    return fig
