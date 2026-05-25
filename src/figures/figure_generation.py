"""High-level plumbing for ``06_FigureGeneration.ipynb``.

This module intentionally coordinates paths, constants, and save helpers.  The
scientific plotting functions stay in their stage packages where the data
contracts are defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import io
import json
import os
import shutil
import sys
import zipfile
from typing import Any, Iterable, Literal, Mapping

import matplotlib

if "ipykernel" not in sys.modules and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import ensure_src_on_path, find_repo_root, notebook_cache_dir


THRESH_ORDER = [
    "I_first",
    "I_volley5",
    "I_volley10",
    "I_5",
    "I_10",
    "I_15",
    "I_25",
    "I_50",
    "I_max",
]
FREQ_ORDER = [10.0, 20.0, 50.0, 100.0, 200.0]
CV_ORDER = [round(0.1 * i, 1) for i in range(11)]

FREQ_COLOR = {
    10.0: "#440154",
    20.0: "#3b528b",
    50.0: "#21918c",
    100.0: "#5ec962",
    200.0: "#fde725",
}
FREQ_MARKER = {
    10.0: "o",
    20.0: "s",
    50.0: "D",
    100.0: "^",
    200.0: "v",
}


@dataclass(frozen=True)
class ManuscriptFigureSpec:
    """One manuscript figure plus its display caption and expected output stem."""

    key: str
    label: str
    title: str
    caption: str
    stems: tuple[str, ...]
    static_media: str | None = None
    static_stems: tuple[str, ...] = ()

    @property
    def static_source_stems(self) -> tuple[str, ...]:
        return self.static_stems or self.stems


MANUSCRIPT_FIGURE_SPECS: tuple[ManuscriptFigureSpec, ...] = (
    ManuscriptFigureSpec(
        key="figure_1",
        label="Figure 1",
        title="Computational Modeling Pipeline Overview",
        stems=("fig01_computational_modeling_pipeline_overview",),
        static_media="word/media/image1.png",
        static_stems=("Figure1_PipelineOverview",),
        caption="Figure 1: Computational Modeling Pipeline Overview. a) COMSOL biophysically accurate finite-element model of rodent spinal cord used to simulate spinal cord stimulation. The heatmap represents the cross-sectional current density at the rostral-caudal midline of the electrode in response to +/-1 uA stimulation. A single axon trajectory (red) was placed in a dorsal column region of maximal current density, and electric potentials along its length were exported. b) Inter-pulse intervals were randomly sampled from a gamma distribution to produce pulse trains with the desired CV (0, 0.1, ..., 1) and number of pulses (2*frequency) in a 2-s window. c) COMSOL-derived potentials in response to +/-1 uA bipolar, biphasic stimulation were combined with generated time patterns and used as extracellular potentials to drive an MRG axon model. The +/-1 uA input current amplitude was recursively scaled to satisfy the corresponding spike-count criteria, and this panel shows example spike-count threshold amplitudes and simulated axonal spike trains. d) Finally, mixtures of spike-count regimes were used to describe experimental observations of detection threshold and periodicity discrimination. Created with BioRender.com.",
    ),
    ManuscriptFigureSpec(
        key="figure_2",
        label="Figure 2",
        title="Periodicity Discrimination Model Overview",
        stems=("fig02_periodicity_discrimination_model_overview",),
        static_media="word/media/image2.png",
        static_stems=("Figure2_ObserverModel",),
        caption="Figure 2: Periodicity Discrimination Model Overview. Simulated CV0-based spike trains were converted into rate, timing, and hybrid feature matrices for each frequency. For each model family, signed evidence relative to the CV = 0 and CV = 1 template endpoints was computed and normalized separately for each Monte Carlo training set, then combined using per-block (same mixture for each CV and frequency in the same block) or per-point (hybrid only; separate mixtures per-CV, per-frequency) regime mixture weights. The 10-ms and 1-ms evidence channels were retained separately and integrated with fitted gains before logistic readout; in the hybrid model, the rate pathway included an additional fitted rate-gain term to allow different relative timing and rate contribution across CV values. The resulting drive signals were passed through a logistic decision stage to predict the psychometric curves. Parameters were optimized to minimize mismatch between predicted and observed psychometric curves. The lower panels show example behavioral datapoint (black) predictions for the per-block rate (red), timing (blue), and hybrid (purple) models, as well as the per-point hybrid model (orange). Created with BioRender.com.",
    ),
    ManuscriptFigureSpec(
        key="figure_3",
        label="Figure 3",
        title="Detection Threshold Behavioral Matching",
        stems=("fig03_detection_threshold_behavioral_matching",),
        caption="Figure 3: Detection Threshold Behavioral Matching. This figure shows simulated spike-count threshold amplitudes and compares them to the detection threshold trends observed experimentally. Colored traces represent the different spike-count thresholds marked in the bottom legend. a) Absolute threshold type amplitudes (uA) as a function of CV. b) Normalized threshold type amplitudes as a function of CV. Values normalized by the maximum value across all CVs for each threshold type individually to illustrate their directional trends. c) Normalized behavioral threshold datapoints (mean +/- SEM) were fit with quadratic trends, and simulated spike-count regimes were used to match the trends. The best-fit single regime could not accurately describe the observed behavior, but per-point regime mixtures could describe behavioral trends and individual datapoints.",
    ),
    ManuscriptFigureSpec(
        key="figure_4",
        label="Figure 4",
        title="Periodicity Discrimination Behavioral Matching",
        stems=("fig04_behavioral_psychometric_curve_matching",),
        caption="Figure 4: Periodicity Discrimination Behavioral Matching. This figure shows the resulting model fits to the experimentally observed psychometric curves. Individual behavioral datapoints (black) show the mean +/- SEM and the colors represent the per-block mixture rate (red), timing (blue), and hybrid (purple) models, and the per-point mixture hybrid model (orange), as described in the bottom legend. Per-block models were fit across all CVs and frequencies in each block (std0 and std1) with a shared mixture of CV0-based spike-count regimes, while the per-point model had separate mixtures for each CV and frequency. The best-fit per-block models are shown in bold.",
    ),
    ManuscriptFigureSpec(
        key="figure_5",
        label="Figure 5",
        title="Rate Reliance of Hybrid Discrimination Models",
        stems=("fig05_rate_reliance",),
        caption="Figure 5: Rate Reliance of Hybrid Discrimination Models. This figure shows the effective rate share for the per-block and per-point hybrid models as described by the bottom legend. a) Bold traces show effective rate share, calculated as the proportion of total model drive attributable to rate evidence after learned rate/timing scaling. Shaded areas under the curves illustrate the AUC used as a summary metric of average rate reliance across CV. b) Mean effective rate share AUC values are shown for each frequency. Circle datapoints denote std0 blocks and square datapoints denote std1 blocks. Asterisk significance markers indicate Bonferroni-corrected p < 0.05 for comparisons against the corresponding 20 Hz AUC value. Error bars show [2.5, 97.5] percentile intervals, indicating cross-repetition spread.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_1",
        label="Supplementary Figure 1",
        title="Computational Model Domains",
        stems=("suppfig01_computational_model_domains",),
        static_media="word/media/image6.png",
        static_stems=("SuppFigure1_SpinalCordAnatomy",),
        caption="Supplementary Figure 1: Computational Model Domains. The biophysically accurate layers of the rodent spinal cord were comprised of materials representing epidural fat, platinum electrode faces, dura mater, arachnoid mater, cerebrospinal fluid, pia mater, white matter, and grey matter.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_2",
        label="Supplementary Figure 2",
        title="Spike-Count Threshold Amplitudes",
        stems=("suppfig02_spike_count_threshold_amplitudes",),
        caption="Supplementary Figure 2: Spike-Count Threshold Amplitudes. Plots show the simulated input current (uA) required to satisfy the different spike-count criteria as a function of CV. Colors correspond to frequencies of 10, 20, 50, 100, and 200 Hz as described by the bottom legend. Datapoints at each CV value represent the mean +/- SEM across 100 simulated repetitions.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_3",
        label="Supplementary Figure 3",
        title="Regime Spike Counts",
        stems=("suppfig03_threshold_spike_counts",),
        caption="Supplementary Figure 3: Regime Spike Counts. Plots show the number (top row) or percentage (bottom two rows) of simulated criteria-satisfying spikes produced as a function of CV. Colors correspond to frequencies of 10, 20, 50, 100, and 200 Hz as described by the bottom legend. Datapoints at each CV value represent the mean +/- SEM across 100 simulated repetitions.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_4",
        label="Supplementary Figure 4",
        title="CV0-Based Regime Spike Counts",
        stems=("suppfig04_cv0_based_threshold_spike_counts",),
        caption="Supplementary Figure 4: CV0-Based Regime Spike Counts. Plots show the number (top row) or percentage (bottom two rows) of simulated spikes produced as a function of CV using the spike-count threshold amplitudes determined at CV=0 for all CV values. Colors correspond to frequencies of 10, 20, 50, 100, and 200 Hz as described by the bottom legend. Datapoints at each CV value represent the mean +/- SEM across 100 simulated repetitions.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_5",
        label="Supplementary Figure 5",
        title="Feature Engineering",
        stems=("suppfig05_single_repetition_feature_engineering",),
        caption="Supplementary Figure 5: Feature Engineering. This figure shows an illustrative example of the process used to convert spike trains into signed template evidence relative to the CV = 0 and CV = 1 endpoint templates for a single 200 Hz repetition and the CV0-based spike-count threshold amplitude. a) Spikes simulated for all CVs using the CV0-based input current amplitude. b) Bursts for all CVs computed by collapsing contiguous active 10-ms bins into single events. c) Rate features for spikes and bursts plotted as functions of CV. d) Timing features for spikes and bursts plotted as functions of CV. e) Rate evidence for spikes and bursts computed for each CV by comparing feature differences between CV = 0 and CV = 1 endpoint templates and satisfying the JND ratio > 2.5. f) Timing evidence for spikes and bursts computed for each CV by comparing feature differences between CV = 0 and CV = 1 endpoint templates.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_6",
        label="Supplementary Figure 6",
        title="Normalized Evidence Curves",
        stems=("suppfig06_combined_normalized_evidence_curves",),
        caption="Supplementary Figure 6. Normalized Evidence Curves. This figure shows the signed rate and timing template evidence for the spike and burst features for each CV0-based spike-count regime. Datapoints are mean +/- SEM across the 200 Monte Carlo training sets, and the negative and positive evidence values correspond to CV = 0 and CV = 1, respectively. a) Reliability scaled spike rate evidence as a function of CV. b) Reliability scaled burst rate evidence as a function of CV. c) Spike timing evidence as a function of CV. d) Burst timing evidence as a function of CV.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_7",
        label="Supplementary Figure 7",
        title="Sub-Refractory Inter-Pulse Intervals",
        stems=("suppfig07_sub_refractory_inter_pulse_intervals",),
        caption="Supplementary Figure 7: Sub-Refractory Inter-Pulse Intervals. This figure shows the fraction of expected spikes at the plateau input current as a function of the fraction of inter-pulse intervals <= 1.10 ms. Colors and markers described in the bottom legend represent frequencies 10, 20, 50, 100, and 200 Hz. Each datapoint represents a single simulated repetition.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_8",
        label="Supplementary Figure 8",
        title="Detection Threshold Spike-Count Regime Mixtures",
        stems=("suppfig08_detection_mixture_weights",),
        caption="Supplementary Figure 8: Detection Threshold Spike-Count Regime Mixtures. This figure shows stacked area plots for the per-point mixtures across frequencies and CVs identified for the behavioral quadratic fit using all repetitions (top row) and the behavioral datapoint fit using the Slack-matched repetition (bottom row). Colors represent the different spike-count thresholds, as described by the bottom legend.",
    ),
    ManuscriptFigureSpec(
        key="supplementary_figure_9",
        label="Supplementary Figure 9",
        title="Periodicity Discrimination Spike-Count Regime Mixtures",
        stems=("suppfig09_periodicity_discrimination_mixture_weights",),
        caption="Supplementary Figure 9: Periodicity Discrimination Spike-Count Regime Mixtures. This figure shows the learned CV0-based spike-count regime mixture weights for the per-block and per-point models. The top and bottom rows show std0 and std1 blocks, respectively. a) Block-shared regime weights for the rate (red), timing (blue), and hybrid (purple) per-block models; the best-fit model is shown in bold. b) Stacked area plots for the per-point mixtures across frequencies and CVs; colors represent the different CV0-based spike-count regimes, as described by the bottom legend.",
    ),
)


def set_manuscript_style(dpi: int = 120) -> None:
    """Apply the light matplotlib style used by the review notebooks."""

    plt.rcParams.update(
        {
            "figure.dpi": int(dpi),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(
    fig,
    out_dir: str | Path,
    stem: str,
    *,
    formats: Iterable[Literal["png", "pdf", "svg"]] = ("png", "pdf"),
    bbox_inches: str = "tight",
    dpi: int | None = None,
) -> list[Path]:
    """Save a matplotlib figure in one or more formats and return paths."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = out_path / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": bbox_inches}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        written.append(path)
    return written


@dataclass(frozen=True)
class FigureGenerationPaths:
    """Repository-relative paths used by the manuscript figure notebook."""

    repo_root: Path

    @classmethod
    def from_cwd(cls, start: str | Path | None = None) -> "FigureGenerationPaths":
        return cls(find_repo_root(start))

    @classmethod
    def from_repo_root(cls, repo_root: str | Path) -> "FigureGenerationPaths":
        return cls(Path(repo_root).resolve())

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def output_dir(self) -> Path:
        return self.repo_root / "results" / "figures"

    @property
    def legacy_figure_dir(self) -> Path:
        return self.repo_root / "results" / "figures"

    @property
    def notebook_cache_dir(self) -> Path:
        return notebook_cache_dir(self.repo_root)

    @property
    def detection_dir(self) -> Path:
        return self.data_dir / "detection" / "generated"

    @property
    def discrimination_per_block_dir(self) -> Path:
        return self.data_dir / "discrimination" / "per-block"

    @property
    def time_patterns_dir(self) -> Path:
        return self.data_dir / "time-patterns" / "generated"

    @property
    def detection_threshold_csv(self) -> Path:
        return self.detection_dir / "detection_thresholds.csv"

    @property
    def cv0_responses_csv(self) -> Path:
        return self.detection_dir / "cv0_based_responses.csv"

    @property
    def detection_behavior_xlsx(self) -> Path:
        return self.data_dir / "detection" / "slack" / "DetectionThresholds.xlsx"

    @property
    def static_figure_dir(self) -> Path:
        return self.notebook_cache_dir / "static-figures"

    @property
    def discrimination_per_point_dir(self) -> Path:
        return self.data_dir / "discrimination" / "per-point"

    @property
    def discrimination_slack_dir(self) -> Path:
        return self.data_dir / "discrimination" / "slack"

    @property
    def detection_responses_csv(self) -> Path:
        return self.detection_dir / "detection_responses.csv"


def setup_figure_generation(
    start: str | Path | None = None,
    *,
    dpi: int = 120,
    add_src: bool = True,
) -> FigureGenerationPaths:
    """Resolve paths, create output/cache folders, and apply plot style."""

    paths = FigureGenerationPaths.from_cwd(start)
    if add_src:
        ensure_src_on_path(paths.repo_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.notebook_cache_dir.mkdir(parents=True, exist_ok=True)
    paths.static_figure_dir.mkdir(parents=True, exist_ok=True)
    set_manuscript_style(dpi=dpi)
    return paths


def save_manuscript_figure(fig, stem: str, paths: FigureGenerationPaths, **kwargs) -> list[Path]:
    """Save a manuscript figure to the canonical results folder."""

    return save_figure(fig, paths.output_dir, stem, **kwargs)

_LEGACY_CELL_ORDER: tuple[int, ...] = (2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)
_LEGACY_CELL_SOURCES_JSON = '{"2": "# Shared imports + paths for manuscript figure generation\\n\\nimport ast\\nimport json\\nimport math\\nimport re\\nimport sys\\nimport zipfile\\nimport xml.etree.ElementTree as ET\\nfrom pathlib import Path\\nimport numpy as np\\nimport pandas as pd\\nimport matplotlib.pyplot as plt\\nfrom IPython.display import display\\n\\n# Bootstrap src/ imports before using repository helpers.\\ndef _bootstrap_repo_root(start: Path) -> Path:\\n    for candidate in (start.resolve(), *start.resolve().parents):\\n        if (candidate / \'src\').exists() and (candidate / \'data\').exists():\\n            src_dir = candidate / \'src\'\\n            if str(src_dir) not in sys.path:\\n                sys.path.insert(0, str(src_dir))\\n            return candidate\\n    raise RuntimeError(\'Could not locate repository root.\')\\n\\nREPO_ROOT = _bootstrap_repo_root(Path.cwd())\\n\\nfrom utils import (\\n    get_spikes_col as _get_spikes_col,\\n    parse_detection_behavioral_xlsx,\\n    parse_spikes as _parse_spikes,\\n    sem,\\n    to_num,\\n)\\nfrom figures.figure_generation import (\\n    CV_ORDER,\\n    FREQ_COLOR,\\n    FREQ_MARKER,\\n    FREQ_ORDER,\\n    THRESH_ORDER,\\n    setup_figure_generation,\\n)\\nfrom detection import (\\n    normalize_threshold_frame,\\n    plot_cv0_response_counts,\\n    plot_spike_count_threshold_amplitudes,\\n    plot_threshold_spike_counts,\\n)\\n\\nFIGURE_PATHS = setup_figure_generation(REPO_ROOT)\\nDATA_DIR = FIGURE_PATHS.data_dir\\nRESULTS_DIR_MANUSCRIPT = FIGURE_PATHS.discrimination_per_block_dir\\nMANUSCRIPT_OUT_DIR = FIGURE_PATHS.output_dir\\nSRC_DIR = REPO_ROOT / \'src\'\\n\\nDT_CSV = FIGURE_PATHS.detection_threshold_csv\\nCV0_RESP_CSV = FIGURE_PATHS.cv0_responses_csv\\nDETECTION_XLSX = FIGURE_PATHS.detection_behavior_xlsx\\nTIME_PATTERNS_DIR = FIGURE_PATHS.time_patterns_dir\\nFEAT_SIM_CSV = CV0_RESP_CSV\\n\\nprint(\'REPO_ROOT:\', REPO_ROOT)\\nprint(\'MANUSCRIPT_OUT_DIR:\', MANUSCRIPT_OUT_DIR)\\nprint(\'DT_CSV exists:\', DT_CSV.exists())\\nprint(\'CV0_RESP_CSV exists:\', CV0_RESP_CSV.exists())\\nprint(\'DETECTION_XLSX exists:\', DETECTION_XLSX.exists())", "4": "# Detection manuscript figures\\n\\nif not DT_CSV.exists() or not CV0_RESP_CSV.exists():\\n    raise FileNotFoundError(\'Detection inputs missing. Expected data/detection/generated/{detection_thresholds.csv,cv0_based_responses.csv}\')\\n\\n# --- Load + prep tables ---\\ndf_dt = normalize_threshold_frame(pd.read_csv(DT_CSV))\\ndf_cv0 = normalize_threshold_frame(pd.read_csv(CV0_RESP_CSV))\\n\\nfor c in [\'cutline\', \'cv\', \'freq\', \'rep_index\', \'N_max_spikes\']:\\n    if c in df_dt.columns:\\n        df_dt[c] = to_num(df_dt[c])\\nfor c in [\'cutline\', \'cv\', \'freq\', \'rep_index\', \'n_spikes\', \'amp_uA\']:\\n    if c in df_cv0.columns:\\n        df_cv0[c] = to_num(df_cv0[c])\\n\\ndf_dt[\'rep_label\'] = df_dt[\'rep_label\'].astype(str)\\ndf_cv0[\'rep_label\'] = df_cv0[\'rep_label\'].astype(str)\\ndf_cv0[\'thresh_type\'] = df_cv0[\'thresh_type\'].astype(str)\\n\\ncutline = 101\\nfreq_vals = sorted(set(float(x) for x in df_dt[\'freq\'].dropna().unique().tolist()))\\ncv_vals = sorted(set(float(x) for x in df_dt[\'cv\'].dropna().unique().tolist()))\\n\\nERROR_MODE = \'sem\'  # one-line toggle: \'sd\' or \'sem\'\\n\\nTHRESH_SHORT_LABELS = {\\n    \'I_first\': \'first\',\\n    \'I_volley5\': \'volley5\',\\n    \'I_volley10\': \'volley10\',\\n    \'I_5\': \'5\',\\n    \'I_10\': \'10\',\\n    \'I_15\': \'15\',\\n    \'I_25\': \'25\',\\n    \'I_50\': \'50\',\\n    \'I_max\': \'max\',\\n}\\n\\ndef _th_suffix(th):\\n    return THRESH_SHORT_LABELS.get(str(th), str(th))\\n\\ndef _math_thresh_label(symbol: str, th: str) -> str:\\n    suf = _th_suffix(th)\\n    return rf\'${symbol}_{{{suf}}}$\'\\n\\ndef _math_tick_label(symbol: str, th: str) -> str:\\n    return _math_thresh_label(symbol, th)\\n\\ndef _display_thresh_name(symbol: str, th: str) -> str:\\n    return _math_thresh_label(symbol, th)\\n\\ndef _error_stat(x):\\n    a = np.asarray(x, dtype=float)\\n    a = a[np.isfinite(a)]\\n    if a.size <= 1:\\n        return 0.0\\n    sd = float(np.std(a, ddof=1))\\n    if str(ERROR_MODE).strip().lower() == \'sem\':\\n        return float(sd / np.sqrt(a.size))\\n    return sd\\n\\n# Aggregates used by multiple panels\\nagg_all = (\\n    df_dt.groupby([\'cutline\', \'cv\', \'freq\'], as_index=False)\\n    .agg(\\n        **{f\'{t}_mean\': (f\'{t}_uA\', \'mean\') for t in THRESH_ORDER},\\n        **{f\'{t}_sem\': (f\'{t}_uA\', _error_stat) for t in THRESH_ORDER},\\n        **{f\'N_{t}_mean\': (f\'N_{t}_spikes\', \'mean\') for t in THRESH_ORDER},\\n        **{f\'N_{t}_sem\': (f\'N_{t}_spikes\', _error_stat) for t in THRESH_ORDER},\\n        N_max_mean=(\'N_max_spikes\', \'mean\'),\\n        n_reps=(\'I_first_uA\', \'count\'),\\n    )\\n)\\nagg_all[\'expected_spikes\'] = agg_all[\'freq\'] * 2.0\\nfor t in THRESH_ORDER:\\n    agg_all[f\'N_{t}_norm\'] = agg_all[f\'N_{t}_mean\'] / agg_all[\'expected_spikes\']\\n    agg_all[f\'N_{t}_norm_sem\'] = agg_all[f\'N_{t}_sem\'] / agg_all[\'expected_spikes\']\\n\\nslack_mask = df_dt[\'rep_label\'].str.contains(\'slack\', case=False, na=False)\\nif \'rep_index\' in df_dt.columns:\\n    slack_mask = slack_mask | np.isclose(df_dt[\'rep_index\'], 0.0, equal_nan=False)\\ndf_dt_slack = df_dt[slack_mask].copy()\\nagg_slack = (\\n    df_dt_slack.groupby([\'cutline\', \'cv\', \'freq\'], as_index=False)\\n    .agg(\\n        **{f\'{t}_mean\': (f\'{t}_uA\', \'mean\') for t in THRESH_ORDER},\\n        **{f\'{t}_sem\': (f\'{t}_uA\', _error_stat) for t in THRESH_ORDER},\\n        **{f\'N_{t}_mean\': (f\'N_{t}_spikes\', \'mean\') for t in THRESH_ORDER},\\n        **{f\'N_{t}_sem\': (f\'N_{t}_spikes\', _error_stat) for t in THRESH_ORDER},\\n        N_max_mean=(\'N_max_spikes\', \'mean\'),\\n        n_reps=(\'I_first_uA\', \'count\'),\\n    )\\n)\\nagg_slack[\'expected_spikes\'] = agg_slack[\'freq\'] * 2.0\\nfor t in THRESH_ORDER:\\n    agg_slack[f\'N_{t}_norm\'] = agg_slack[f\'N_{t}_mean\'] / agg_slack[\'expected_spikes\']\\n    agg_slack[f\'N_{t}_norm_sem\'] = agg_slack[f\'N_{t}_sem\'] / agg_slack[\'expected_spikes\']\\n\\n# cv0-response aggregates (discrimination-at-cv0-amplitudes panel)\\nagg_cv0_raw = (\\n    df_cv0.groupby([\'cutline\', \'cv\', \'freq\', \'thresh_type\'], as_index=False)\\n    .agg(N_mean=(\'n_spikes\', \'mean\'), N_sem=(\'n_spikes\', _error_stat))\\n)\\npivot_cv0_mean = (\\n    agg_cv0_raw\\n    .pivot_table(index=[\'cutline\', \'cv\', \'freq\'], columns=\'thresh_type\', values=\'N_mean\')\\n    .reset_index()\\n)\\npivot_cv0_sem = (\\n    agg_cv0_raw\\n    .pivot_table(index=[\'cutline\', \'cv\', \'freq\'], columns=\'thresh_type\', values=\'N_sem\')\\n    .reset_index()\\n)\\npivot_cv0_mean.columns.name = None\\npivot_cv0_sem.columns.name = None\\nrename_mean = {t: f\'N_cv0_{t}_mean\' for t in THRESH_ORDER if t in pivot_cv0_mean.columns}\\nrename_sem = {t: f\'N_cv0_{t}_sem\' for t in THRESH_ORDER if t in pivot_cv0_sem.columns}\\npivot_cv0 = pivot_cv0_mean.rename(columns=rename_mean).merge(\\n    pivot_cv0_sem.rename(columns=rename_sem),\\n    on=[\'cutline\', \'cv\', \'freq\'],\\n    how=\'outer\',\\n)\\npivot_cv0[\'expected_spikes\'] = pivot_cv0[\'freq\'] * 2.0\\nfor t in THRESH_ORDER:\\n    c_mean = f\'N_cv0_{t}_mean\'\\n    c_sem = f\'N_cv0_{t}_sem\'\\n    if c_mean in pivot_cv0.columns:\\n        pivot_cv0[f\'N_cv0_{t}_norm\'] = pivot_cv0[c_mean] / pivot_cv0[\'expected_spikes\']\\n    if c_sem in pivot_cv0.columns:\\n        pivot_cv0[f\'N_cv0_{t}_norm_sem\'] = pivot_cv0[c_sem] / pivot_cv0[\'expected_spikes\']\\n\\ndef _nice_step(span, target_ticks=6):\\n    span = float(max(span, 1e-9))\\n    raw = span / max(int(target_ticks) - 1, 1)\\n    mag = 10.0 ** np.floor(np.log10(raw))\\n    norm = raw / mag\\n    if norm <= 1.0:\\n        return 1.0 * mag\\n    if norm <= 2.0:\\n        return 2.0 * mag\\n    if norm <= 5.0:\\n        return 5.0 * mag\\n    return 10.0 * mag\\n\\ndef _set_axis_with_explicit_top_tick(ax, yvals, target_ticks=6, pad_frac=0.04):\\n    arr = np.asarray(yvals, dtype=float)\\n    arr = arr[np.isfinite(arr)]\\n    if arr.size == 0:\\n        return\\n    lo = float(np.min(arr))\\n    hi = float(np.max(arr))\\n    if np.isclose(lo, hi):\\n        span = max(abs(hi), 1.0) * 0.08\\n        lo, hi = lo - span, hi + span\\n    else:\\n        pad = (hi - lo) * float(pad_frac)\\n        lo, hi = lo - pad, hi + pad\\n\\n    step = _nice_step(hi - lo, target_ticks=target_ticks)\\n    lo_tick = np.floor(lo / step) * step\\n    hi_tick = np.ceil(hi / step) * step\\n    if np.isclose(lo_tick, hi_tick):\\n        hi_tick = lo_tick + step\\n    ticks = np.arange(lo_tick, hi_tick + 0.5 * step, step)\\n    ax.set_ylim(lo_tick, hi_tick)\\n    ax.set_yticks(ticks)\\n\\n\\ndef _set_axis_with_anchor_tick(ax, yvals, anchor=500.0, anchor_frac=0.25, target_ticks=6, pad_frac=0.04):\\n    arr = np.asarray(yvals, dtype=float)\\n    arr = arr[np.isfinite(arr)]\\n    if arr.size == 0:\\n        return\\n\\n    lo = float(np.min(arr))\\n    hi = float(np.max(arr))\\n    if np.isclose(lo, hi):\\n        span = max(abs(hi), abs(anchor), 1.0) * 0.08\\n        lo, hi = lo - span, hi + span\\n    else:\\n        pad = (hi - lo) * float(pad_frac)\\n        lo, hi = lo - pad, hi + pad\\n\\n    p = float(np.clip(anchor_frac, 0.05, 0.95))\\n    low_dist = max(float(anchor) - lo, 0.0)\\n    high_dist = max(hi - float(anchor), 0.0)\\n    span = max(\\n        low_dist / p if p > 0 else 0.0,\\n        high_dist / (1.0 - p) if p < 1.0 else 0.0,\\n        1e-9,\\n    )\\n\\n    ylo = float(anchor) - p * span\\n    yhi = float(anchor) + (1.0 - p) * span\\n\\n    # Build 5 ticks with anchor at index 1 (25% height), using a step that\'s a multiple of 5 uA.\\n    req_step = max(float(anchor) - lo, (hi - float(anchor)) / 3.0, 1e-9)\\n    step = 5.0 * np.ceil(req_step / 5.0)\\n    ticks = float(anchor) + step * (np.arange(5, dtype=float) - 1.0)\\n    ax.set_ylim(float(ticks[0]), float(ticks[-1]))\\n    ax.set_yticks(ticks)\\n\\n# --- Figure A: Stacked threshold rasters (two examples) ---\\nRASTER_CASES = [\\n    {\'freq_hz\': 100.0, \'cv\': 0.0, \'rep_index\': 1},\\n    {\'freq_hz\': 100.0, \'cv\': 1.0, \'rep_index\': 95},  # rep096_gamma: near-monotonic threshold spikes and clear separation\\n]\\n\\nRASTER_PREPAD_MS = 100.0\\nRASTER_XMIN_MS = 0.0\\nRASTER_XMAX_MS = 2000.0\\n\\n# Include a 100 ms lead-in before t=0 so first spikes are easy to see.\\nRASTER_LEADIN_MS = 10.0\\n\\n# Time-pattern traces are sampled at 20 kHz in this pipeline.\\nRASTER_FS_HZ = 20000.0\\n\\n# Match MATLAB default axes color order (cycles after 7 colors).\\nMATLAB_COLOR_ORDER = [\\n    (0.0000, 0.4470, 0.7410),\\n    (0.8500, 0.3250, 0.0980),\\n    (0.9290, 0.6940, 0.1250),\\n    (0.4940, 0.1840, 0.5560),\\n    (0.4660, 0.6740, 0.1880),\\n    (0.3010, 0.7450, 0.9330),\\n    (0.6350, 0.0780, 0.1840),\\n]\\n\\n# Geometry knobs to visually match Slack-style stacked rasters.\\nRASTER_ROW_STEP = 1.05\\nRASTER_SPIKE_HALF_HEIGHT = 0.44\\nRASTER_SPIKE_WIDTH = 1.2\\n\\n\\ndef _count_label(n, singular, plural=None):\\n    n_int = int(n)\\n    if plural is None:\\n        plural = singular + \'s\'\\n    noun = singular if abs(n_int) == 1 else plural\\n    return f\'{n_int} {noun}\'\\n\\n\\ndef _parse_spike_times_ms_local(raw):\\n    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):\\n        return np.array([], dtype=float)\\n    if isinstance(raw, (list, tuple, np.ndarray)):\\n        arr = np.asarray(raw, dtype=float)\\n        return arr[np.isfinite(arr)]\\n    s_local = str(raw).strip()\\n    if not s_local or s_local.lower() in {\'nan\', \'none\', \'[]\'}:\\n        return np.array([], dtype=float)\\n    try:\\n        parsed = ast.literal_eval(s_local)\\n        arr = np.asarray(parsed, dtype=float)\\n        return arr[np.isfinite(arr)]\\n    except Exception:\\n        parts = [p for p in s_local.replace(\',\', \' \').split() if p]\\n        out = []\\n        for p in parts:\\n            try:\\n                out.append(float(p))\\n            except Exception:\\n                pass\\n        arr = np.asarray(out, dtype=float)\\n        return arr[np.isfinite(arr)]\\n\\n\\ndef _regular_train_times_local(freq_hz: float, duration_s: float) -> np.ndarray:\\n    \\"\\"\\"Deterministic regular train with fixed event count N=round(freq*duration).\\"\\"\\"\\n    if float(freq_hz) <= 0.0:\\n        return np.array([], dtype=float)\\n    N = int(round(float(freq_hz) * float(duration_s)))\\n    if N <= 0:\\n        return np.array([], dtype=float)\\n    period = 1.0 / float(freq_hz)\\n    t = np.arange(N, dtype=float) * period\\n    return t[t < float(duration_s)]\\n\\n\\ndef _draw_gamma_times_fixed_count_local(\\n    *,\\n    freq_hz: float,\\n    cv: float,\\n    duration_s: float,\\n    rng: np.random.Generator,\\n    min_ipi_s: float,\\n    max_tries: int = 50,\\n) -> np.ndarray:\\n    \\"\\"\\"Gamma-IPI train used by 01_TimePatterns_Generator.ipynb (fixed-N construction).\\"\\"\\"\\n    if float(freq_hz) <= 0.0:\\n        return np.array([], dtype=float)\\n\\n    N = int(round(float(freq_hz) * float(duration_s)))\\n    if N <= 0:\\n        return np.array([], dtype=float)\\n\\n    target_mean_ipi = 1.0 / float(freq_hz)\\n    k = 1.0 / (float(cv) ** 2) if float(cv) > 0.0 else 1e12\\n    theta = target_mean_ipi / k\\n\\n    ipis = None\\n    for _ in range(int(max_tries)):\\n        ip = rng.gamma(shape=k, scale=theta, size=N)\\n        ip = np.maximum(ip, float(min_ipi_s) * 1.05)\\n        ip *= (target_mean_ipi / float(np.mean(ip)))\\n        ipis = ip\\n        if float(np.min(ip)) >= float(min_ipi_s):\\n            break\\n\\n    if ipis is None:\\n        return np.array([], dtype=float)\\n\\n    times = np.cumsum(ipis)\\n    total_span = float(times[-1])\\n    if total_span >= float(duration_s):\\n        factor = (float(duration_s) - float(min_ipi_s)) / max(total_span, 1e-12)\\n        times = times * factor\\n\\n    return times[times < float(duration_s)]\\n\\n\\ndef _pair_seed_default_local(cv: float, freq_hz: float, base_seed: int = 123) -> int:\\n    \\"\\"\\"Rebuild per-pair seeds exactly like 01_TimePatterns_Generator.ipynb.\\"\\"\\"\\n    cv_all = [round(0.1 * i, 1) for i in range(11)]\\n    freq_all = [10.0, 20.0, 50.0, 100.0, 200.0]\\n    pairs = [(float(c), float(f)) for c in cv_all for f in freq_all]\\n    seeds = np.random.default_rng(int(base_seed)).integers(\\n        0, 2**32 - 1, size=len(pairs), dtype=np.uint64\\n    )\\n    mapping = {pair: int(seed) for pair, seed in zip(pairs, seeds)}\\n    key = (round(float(cv), 1), float(freq_hz))\\n    return int(mapping.get(key, int(base_seed)))\\n\\n\\ndef _synth_input_pulse_onsets_s_local(*, freq_hz, cv, rep_index, rep_label, meta):\\n    \\"\\"\\"\\n    Reconstruct generated pulse-event times from the fixed-N generator logic.\\n\\n    This guarantees N=round(freq*stim_duration) for generated reps (rep002+),\\n    while rep001_slack continues to use the waveform-as-recorded fallback.\\n    \\"\\"\\"\\n    rep_label_s = str(rep_label).lower()\\n    if int(rep_index) <= 0 or (\'slack\' in rep_label_s):\\n        return None\\n\\n    stim_dur_s = float(meta.get(\'stim_dur_s\', 2.0)) if isinstance(meta, dict) else 2.0\\n    fs_hz = float(meta.get(\'fs\', RASTER_FS_HZ)) if isinstance(meta, dict) else float(RASTER_FS_HZ)\\n    neg_n = int(meta.get(\'neg_samples\', 4)) if isinstance(meta, dict) else 4\\n    iph_n = int(meta.get(\'interphase_samples\', 1)) if isinstance(meta, dict) else 1\\n    pos_n = int(meta.get(\'pos_samples\', 4)) if isinstance(meta, dict) else 4\\n    min_ipi_s = float(neg_n + iph_n + pos_n) / max(fs_hz, 1e-12)\\n\\n    if np.isclose(float(cv), 0.0) or (\'regular\' in rep_label_s):\\n        return _regular_train_times_local(float(freq_hz), stim_dur_s)\\n\\n    # rep_index is 0-based in detection outputs: rep1 -> rep002_gamma (first generated draw)\\n    n_draws = int(rep_index)\\n    pair_seed = _pair_seed_default_local(float(cv), float(freq_hz), base_seed=123)\\n    rng = np.random.default_rng(int(pair_seed))\\n\\n    times = np.array([], dtype=float)\\n    for _ in range(max(n_draws, 1)):\\n        times = _draw_gamma_times_fixed_count_local(\\n            freq_hz=float(freq_hz),\\n            cv=float(cv),\\n            duration_s=stim_dur_s,\\n            rng=rng,\\n            min_ipi_s=min_ipi_s,\\n        )\\n    return np.asarray(times, dtype=float)\\n\\n\\ndef _extract_input_pulse_onsets_s(*, freq_hz, cv, rep_index, rep_label):\\n    npz_path = TIME_PATTERNS_DIR / f\'cv{float(cv):.1f}_f{int(float(freq_hz))}_waveforms.npz\'\\n    meta_path = TIME_PATTERNS_DIR / f\'cv{float(cv):.1f}_f{int(float(freq_hz))}_meta.json\'\\n\\n    meta = {}\\n    if meta_path.exists():\\n        try:\\n            with open(meta_path, \'r\', encoding=\'utf-8\') as f:\\n                meta = json.load(f)\\n        except Exception:\\n            meta = {}\\n\\n    # Read the stored Stage-01 waveform directly for both Slack and generated reps.\\n    # This avoids duplicating generator logic and keeps rasters tied to the data archive.\\n    if not npz_path.exists():\\n        return np.array([], dtype=float)\\n\\n    bundle = np.load(npz_path)\\n\\n    # Prefer matching by rep_label; fallback to index if labels differ by CV family.\\n    if str(rep_label) in bundle.files:\\n        key = str(rep_label)\\n    else:\\n        keys_sorted = sorted(bundle.files)\\n        if int(rep_index) < 0 or int(rep_index) >= len(keys_sorted):\\n            return np.array([], dtype=float)\\n        key = keys_sorted[int(rep_index)]\\n\\n    wave = np.asarray(bundle[key], dtype=float)\\n    active = wave < 0.0\\n    rising = np.where(active[1:] & ~active[:-1])[0] + 1\\n    if active.size and active[0]:\\n        rising = np.r_[0, rising]\\n\\n    onset_ms = rising.astype(float) * (1000.0 / float(RASTER_FS_HZ))\\n    onset_ms = onset_ms[(onset_ms >= float(RASTER_XMIN_MS)) & (onset_ms <= float(RASTER_XMAX_MS))]\\n    return onset_ms / 1000.0\\n\\n\\ndef _plot_one_stacked_raster(case):\\n    freq_hz = float(case[\'freq_hz\'])\\n    cv = float(case[\'cv\'])\\n    rep_index = int(case[\'rep_index\'])\\n\\n    raster_sub = df_cv0[\\n        (df_cv0[\'cutline\'] == cutline)\\n        & np.isclose(df_cv0[\'freq\'], freq_hz)\\n        & np.isclose(df_cv0[\'cv\'], cv)\\n        & (df_cv0[\'rep_index\'] == rep_index)\\n    ].copy()\\n\\n    if raster_sub.empty:\\n        raise RuntimeError(\\n            f\'No raster rows for cutline={cutline}, freq={freq_hz}, cv={cv}, rep_index={rep_index}\'\\n        )\\n\\n    raster_sub = raster_sub[raster_sub[\'thresh_type\'].isin(THRESH_ORDER)].copy()\\n    raster_sub[\'thresh_type\'] = pd.Categorical(raster_sub[\'thresh_type\'], categories=THRESH_ORDER, ordered=True)\\n    raster_sub = raster_sub.sort_values(\'thresh_type\').reset_index(drop=True)\\n\\n    rep_label = str(raster_sub[\'rep_label\'].iloc[0])\\n    input_pulses_s = _extract_input_pulse_onsets_s(\\n        freq_hz=freq_hz,\\n        cv=cv,\\n        rep_index=rep_index,\\n        rep_label=rep_label,\\n    )\\n\\n    row_records = [\\n        {\\n            \'label_left\': \'Input Pulses\',\\n            \'label_right\': _count_label(input_pulses_s.size, \'pulse\'),\\n            \'events_s\': input_pulses_s,\\n            \'color\': (0.10, 0.10, 0.10),\\n        }\\n    ]\\n\\n    for i, r in raster_sub.iterrows():\\n        st_ms = _parse_spike_times_ms_local(r.get(\'spike_times_ms\', \'[]\'))\\n        st_ms = st_ms - float(RASTER_PREPAD_MS)\\n        st_ms = st_ms[(st_ms >= float(RASTER_XMIN_MS)) & (st_ms <= float(RASTER_XMAX_MS))]\\n        st_s = st_ms / 1000.0\\n\\n        amp_uA = float(r.get(\'amp_uA\', np.nan))\\n        left = (\\n            _display_thresh_name(\'I\', str(r[\'thresh_type\'])) + \\"\\\\n\\" + f\\"({amp_uA:.2f} \\" + r\\"$\\\\mu$A)\\"\\n            if np.isfinite(amp_uA)\\n            else _display_thresh_name(\'I\', str(r[\'thresh_type\'])) + \\"\\\\n\\" + r\\"(nan $\\\\mu$A)\\"\\n        )\\n\\n        row_records.append(\\n            {\\n                \'label_left\': left,\\n                \'label_right\': _count_label(st_s.size, \'spike\'),\\n                \'events_s\': st_s,\\n                \'color\': MATLAB_COLOR_ORDER[i % len(MATLAB_COLOR_ORDER)],\\n            }\\n        )\\n\\n    fig, ax = plt.subplots(figsize=(16, 7.8))\\n    y_positions = np.arange(len(row_records), dtype=float)[::-1] * float(RASTER_ROW_STEP)\\n\\n    for y, rec in zip(y_positions, row_records):\\n        events = np.asarray(rec[\'events_s\'], dtype=float)\\n        if events.size:\\n            ax.vlines(\\n                events,\\n                y - float(RASTER_SPIKE_HALF_HEIGHT),\\n                y + float(RASTER_SPIKE_HALF_HEIGHT),\\n                color=rec[\'color\'],\\n                linewidth=float(RASTER_SPIKE_WIDTH),\\n                zorder=3,\\n            )\\n\\n    x_min_s = -float(RASTER_LEADIN_MS) / 1000.0\\n    x_max_s = float(RASTER_XMAX_MS) / 1000.0\\n    ax.set_xlim(x_min_s, x_max_s)\\n    ax.set_xticks(np.arange(0.0, x_max_s + 1e-9, 0.25))\\n\\n    ax.set_ylim(-0.95, float(y_positions.max()) + 0.95)\\n    ax.set_yticks(y_positions)\\n    ax.set_yticklabels([r[\'label_left\'] for r in row_records], fontsize=15)\\n    for _lbl in ax.get_yticklabels():\\n        _lbl.set_multialignment(\'center\')\\n        _lbl.set_horizontalalignment(\'right\')\\n    ax.set_xlabel(\'Time (sec)\')\\n    fig.suptitle(f\'{freq_hz:g} Hz | CV={cv:g}\', fontsize=24, y=0.995)\\n\\n    # Vertical guide lines only, like MATLAB-style panel view.\\n    ax.grid(True, axis=\'x\', alpha=0.22, zorder=1)\\n\\n    # Box around the plot (MATLAB \\"box on\\" style).\\n    for spine in [\'top\', \'right\', \'left\', \'bottom\']:\\n        ax.spines[spine].set_visible(True)\\n        ax.spines[spine].set_linewidth(1.0)\\n    ax.tick_params(axis=\'both\', which=\'both\', direction=\'in\', top=True)\\n\\n    ax_r = ax.twinx()\\n    ax_r.set_ylim(ax.get_ylim())\\n    ax_r.set_yticks(y_positions)\\n    ax_r.set_yticklabels([r[\'label_right\'] for r in row_records], fontsize=15)\\n    ax_r.tick_params(axis=\'y\', length=0, pad=6)\\n    ax_r.grid(False)\\n    ax_r.spines[\'right\'].set_visible(True)\\n    ax_r.spines[\'right\'].set_linewidth(1.0)\\n    ax_r.spines[\'top\'].set_visible(True)\\n    ax_r.spines[\'top\'].set_linewidth(1.0)\\n\\n    plt.tight_layout()\\n    plt.show()\\n\\n    raster_tag = f\\"{int(freq_hz)}Hz_cv{str(cv).replace(\'.\', \'p\')}_rep{rep_index}\\"\\n    if np.isclose(float(freq_hz), 100.0) and np.isclose(float(cv), 0.0):\\n        out_stem = \'fig01_threshold_raster_100Hz_cv0\'\\n    elif np.isclose(float(freq_hz), 100.0) and np.isclose(float(cv), 1.0):\\n        out_stem = \'fig01_threshold_raster_100Hz_cv1_reference\'\\n    else:\\n        out_stem = f\'fig_det_thresh_raster_{raster_tag}\'\\n    fig.savefig(MANUSCRIPT_OUT_DIR / f\'{out_stem}.png\', bbox_inches=\'tight\')\\n    fig.savefig(MANUSCRIPT_OUT_DIR / f\'{out_stem}.pdf\', bbox_inches=\'tight\')\\n\\n\\nfor _case in RASTER_CASES:\\n    _plot_one_stacked_raster(_case)\\n\\n# --- Figure A: 3x3 threshold amplitudes ---\\nfig, axes = plot_spike_count_threshold_amplitudes(df_dt, cutline=cutline)\\nplt.show()\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig03_spike_count_threshold_amplitudes.png\', bbox_inches=\'tight\')\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig03_spike_count_threshold_amplitudes.pdf\', bbox_inches=\'tight\')\\n\\n# --- Figure B: 3x3 threshold spike numbers ---\\nfig, axes = plot_threshold_spike_counts(df_dt, cutline=cutline)\\nplt.show()\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig04_threshold_spike_counts.png\', bbox_inches=\'tight\')\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig04_threshold_spike_counts.pdf\', bbox_inches=\'tight\')\\n\\n# --- Figure C: sub-refractory ISIs scatter ---\\nABS_REF_MS = 1.4\\n\\ndef _extract_spike_times_from_trace(t, I, eps=0.0):\\n    I = np.asarray(I)\\n    t = np.asarray(t)\\n    active_neg = I < -eps\\n    rising = np.where(active_neg[1:] & ~active_neg[:-1])[0] + 1\\n    return t[rising]\\n\\ndef _compute_isi_stats_all_reps(cv_vals, freq_vals, fs=20000.0, abs_ref_ms=1.0):\\n    records = []\\n    for cv in cv_vals:\\n        for freq in freq_vals:\\n            npz_path = TIME_PATTERNS_DIR / f\'cv{cv}_f{int(freq)}_waveforms.npz\'\\n            if not npz_path.exists():\\n                continue\\n            bundle = np.load(npz_path)\\n            rep_labels = np.array(sorted(bundle.files))\\n            waves = [np.array(bundle[k], dtype=np.float32) for k in rep_labels]\\n            current = np.stack(waves, axis=0)\\n            t = np.arange(current.shape[1], dtype=np.float32) / float(fs)\\n            for i_rep in range(current.shape[0]):\\n                st = _extract_spike_times_from_trace(t, current[i_rep], eps=0.0)\\n                if st.size < 2:\\n                    continue\\n                isi_ms = np.diff(st) * 1000.0\\n                n_isi = int(len(isi_ms))\\n                n_short = int(np.sum(isi_ms < abs_ref_ms))\\n                records.append({\\n                    \'cv\': float(cv),\\n                    \'freq\': float(freq),\\n                    \'rep_label\': str(rep_labels[i_rep]),\\n                    \'frac_short\': float(n_short / n_isi) if n_isi > 0 else np.nan,\\n                })\\n    return pd.DataFrame(records)\\n\\nDATA_CACHE_DIR = REPO_ROOT / \'data\' / \'notebook-cache\'\\nDATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)\\nisi_cache = DATA_CACHE_DIR / \'cache_detection_isi_rep_stats.csv\'\\nif isi_cache.exists():\\n    isi_rep_stats = pd.read_csv(isi_cache)\\nelse:\\n    isi_rep_stats = _compute_isi_stats_all_reps(cv_vals, freq_vals, fs=20000.0, abs_ref_ms=ABS_REF_MS)\\n    try:\\n        isi_rep_stats.to_csv(isi_cache, index=False)\\n    except Exception:\\n        pass\\n\\ndf_scatter = df_dt.merge(isi_rep_stats, on=[\'cv\', \'freq\', \'rep_label\'], how=\'left\')\\ndf_scatter = df_scatter[df_scatter[\'cutline\'] == cutline].copy()\\n\\ndf_scatter[\'N_max_norm\'] = df_scatter[\'N_max_spikes\'] / (df_scatter[\'freq\'] * 2.0)\\n\\nfig, ax = plt.subplots(figsize=(8, 6))\\nfor f in sorted(set(df_scatter[\'freq\'].dropna().tolist()), reverse=True):\\n    sf = df_scatter[np.isclose(df_scatter[\'freq\'], f)]\\n    if sf.empty:\\n        continue\\n    ax.scatter(\\n        sf[\'frac_short\'].to_numpy(float),\\n        sf[\'N_max_norm\'].to_numpy(float),\\n        s=45,\\n        alpha=0.8,\\n        marker=FREQ_MARKER.get(f, \'o\'),\\n        color=FREQ_COLOR.get(f, \'k\'),\\n        label=f\'{int(f)} Hz\',\\n    )\\nax.set_xlabel(\'Fraction of Intervals < 1.4 ms\')\\nax.set_ylabel(\'Fraction of Expected Spikes\')\\nax.set_ylim(0.55, 1.05)\\nax.grid(True, alpha=0.25)\\nax.legend(title=\'Frequency\', frameon=True, loc=\'lower left\')\\nfig.suptitle(\'Sub-Refractory Inter-Pulse Intervals\', y=0.98, fontsize=20)\\nplt.tight_layout(rect=[0, 0, 1, 0.96])\\nplt.show()\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'suppfig02_sub_refractory_inter_pulse_intervals.png\', bbox_inches=\'tight\')\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'suppfig02_sub_refractory_inter_pulse_intervals.pdf\', bbox_inches=\'tight\')\\n\\n# --- Figure D: detection behavioral matching (3x5) ---\\nfrom scipy.optimize import minimize\\n\\nbeh = parse_detection_behavioral_xlsx(DETECTION_XLSX)\\nplot_freqs = [10.0, 20.0, 50.0, 100.0, 200.0]\\ncurve_grid = np.linspace(0.0, 1.0, 200)\\nbottom_ylim = {\\n    10.0: (0.80, 1.00),\\n    20.0: (0.80, 1.00),\\n    50.0: (0.80, 1.00),\\n    100.0: (0.80, 1.00),\\n    200.0: (0.80, 1.00),\\n}\\nbottom_ylim_panel_c = {\\n    10.0: (0.80, 1.00),\\n    20.0: (0.85, 1.00),\\n    50.0: (0.80, 1.00),\\n    100.0: (0.85, 1.00),\\n    200.0: (0.85, 1.00),\\n}\\n\\nviridis_colors = plt.cm.viridis(np.linspace(0, 1, len(THRESH_ORDER)))\\nthr_colors = {t: viridis_colors[i] for i, t in enumerate(THRESH_ORDER)}\\nDETECTION_DATAPOINT_CVS = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)\\nDETECTION_EXPLORATORY_REG_USE_ENTMAX = True\\nDETECTION_EXPLORATORY_REG_ENTMAX_ALPHA = 1.15\\nDETECTION_EXPLORATORY_REG_WEIGHT_KL = 0.0\\nDETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K = 3.0\\nDETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA = 0.0\\nDETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP = 1.0\\nDETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA = 0.0\\nDETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA = 0.001\\n\\ndef _rmse(a, b):\\n    a = np.asarray(a, dtype=float)\\n    b = np.asarray(b, dtype=float)\\n    return float(np.sqrt(np.mean((a - b) ** 2)))\\n\\ndef _detection_model_label(th: str) -> str:\\n    return f\\"I_{_th_suffix(th)}\\"\\n\\ndef _build_behavior_target(grp: pd.DataFrame, x_eval: np.ndarray):\\n    xb = grp[\'cv\'].to_numpy(float)\\n    yb = grp[\'mean_norm\'].to_numpy(float)\\n    eb = grp[\'sem_norm\'].to_numpy(float)\\n    if xb.size < 3:\\n        yy = np.interp(x_eval, xb, yb)\\n        coefs = np.asarray([], dtype=float)\\n    else:\\n        coefs = np.polyfit(xb, yb, 2)\\n        yy = np.polyval(coefs, x_eval)\\n    return {\'xb\': xb, \'yb\': yb, \'eb\': eb, \'yy\': yy, \'coefs\': coefs, \'fit_x\': np.asarray(x_eval, dtype=float), \'fit_y\': np.asarray(yy, dtype=float), \'fit_label\': \'quadratic\'}\\n\\ndef _build_behavior_target_datapoints(grp: pd.DataFrame, x_eval: np.ndarray):\\n    xb = grp[\'cv\'].to_numpy(float)\\n    yb = grp[\'mean_norm\'].to_numpy(float)\\n    eb = grp[\'sem_norm\'].to_numpy(float)\\n    yy = np.interp(x_eval, xb, yb)\\n    return {\'xb\': xb, \'yb\': yb, \'eb\': eb, \'yy\': yy, \'coefs\': np.asarray([], dtype=float), \'fit_x\': np.asarray(xb, dtype=float), \'fit_y\': np.asarray(yb, dtype=float), \'fit_label\': \'datapoints\'}\\n\\ndef _normalize_curve(y: np.ndarray) -> np.ndarray:\\n    y = np.asarray(y, dtype=float)\\n    y = np.where(np.isfinite(y), y, np.nan)\\n    y_max = np.nanmax(y) if np.any(np.isfinite(y)) else np.nan\\n    if not np.isfinite(y_max) or abs(y_max) < 1e-12:\\n        return np.full_like(y, np.nan, dtype=float)\\n    return y / y_max\\n\\ndef _sample_curve_to_x(x_full: np.ndarray, y_full: np.ndarray, x_target: np.ndarray) -> np.ndarray:\\n    x_full = np.asarray(x_full, dtype=float)\\n    y_full = np.asarray(y_full, dtype=float)\\n    x_target = np.asarray(x_target, dtype=float)\\n    return np.interp(x_target, x_full, y_full)\\n\\ndef _build_regime_abs_curves(sub_f: pd.DataFrame, thresh_order, x_eval: np.ndarray):\\n    x = sub_f[\'cv\'].to_numpy(float)\\n    out = {}\\n    for t in thresh_order:\\n        y = np.asarray(sub_f[f\'{t}_mean\'].to_numpy(float), dtype=float)\\n        if not np.any(np.isfinite(y)):\\n            out[t] = np.full_like(x_eval, np.nan, dtype=float)\\n        else:\\n            out[t] = np.interp(x_eval, x, y)\\n    return x, out\\n\\ndef _fit_detection_bias(target_y: np.ndarray, pred_y: np.ndarray) -> float:\\n    target_y = np.asarray(target_y, dtype=float)\\n    pred_y = np.asarray(pred_y, dtype=float)\\n    return float(np.mean(target_y - pred_y))\\n\\ndef _project_simplex_rows(W: np.ndarray) -> np.ndarray:\\n    W = np.asarray(W, dtype=float)\\n    W = np.clip(W, 0.0, None)\\n    row_sums = np.sum(W, axis=1, keepdims=True)\\n    zero_rows = row_sums[:, 0] <= 0.0\\n    if np.any(zero_rows):\\n        W[zero_rows, :] = 1.0 / max(1, W.shape[1])\\n        row_sums = np.sum(W, axis=1, keepdims=True)\\n    return W / row_sums\\n\\ndef _entmax_bisect_1d(logits: np.ndarray, alpha: float = 1.1, n_iter: int = 60) -> np.ndarray:\\n    z = np.asarray(logits, dtype=float).reshape(-1)\\n    if z.size == 0:\\n        return z\\n    if alpha <= 1.0:\\n        z0 = z - np.max(z)\\n        e = np.exp(z0)\\n        return e / np.sum(e)\\n    if alpha > 2.0:\\n        alpha = 2.0\\n    d = alpha - 1.0\\n    z_shift = z - np.max(z)\\n    lo = np.min(z_shift) - 1.0\\n    hi = np.max(z_shift)\\n    for _ in range(int(n_iter)):\\n        tau = 0.5 * (lo + hi)\\n        p = np.maximum(z_shift - tau, 0.0) ** (1.0 / d)\\n        s = np.sum(p)\\n        if s == 0.0:\\n            hi = tau\\n        elif s > 1.0:\\n            lo = tau\\n        else:\\n            hi = tau\\n    p = np.maximum(z_shift - hi, 0.0) ** (1.0 / d)\\n    s = np.sum(p)\\n    if s <= 0.0:\\n        return np.full_like(z, 1.0 / max(1, z.size), dtype=float)\\n    return p / s\\n\\ndef _effective_k(weights: np.ndarray) -> float:\\n    w = np.asarray(weights, dtype=float)\\n    w = np.clip(w, 1e-12, 1.0)\\n    H = -float(np.sum(w * np.log(w)))\\n    return float(np.exp(H))\\n\\ndef _weight_penalty_rows(W: np.ndarray, weight_kl: float = 0.0, weight_entropy_floor_k: float = 0.0, weight_entropy_floor_lambda: float = 0.0, weight_max_cap: float = 1.0, weight_max_cap_lambda: float = 0.0, weight_smooth_lambda: float = 0.0) -> float:\\n    W = np.asarray(W, dtype=float)\\n    if W.ndim == 1:\\n        W = W[None, :]\\n    if W.size == 0:\\n        return 0.0\\n    n_th = int(W.shape[1])\\n    W_safe = np.clip(W, 1e-12, 1.0)\\n    reg = 0.0\\n    if weight_kl > 0.0:\\n        kl_rows = np.sum(W_safe * np.log(W_safe * float(n_th)), axis=1)\\n        reg += float(weight_kl) * float(np.mean(kl_rows))\\n    if weight_entropy_floor_k > 0.0 and weight_entropy_floor_lambda > 0.0:\\n        keffs = np.asarray([_effective_k(row) for row in W_safe], dtype=float)\\n        reg += float(weight_entropy_floor_lambda) * float(np.mean(np.maximum(weight_entropy_floor_k - keffs, 0.0) ** 2))\\n    if weight_max_cap < 1.0 and weight_max_cap_lambda > 0.0:\\n        over = np.maximum(W - float(weight_max_cap), 0.0)\\n        reg += float(weight_max_cap_lambda) * float(np.mean(np.sum(over * over, axis=1)))\\n    if weight_smooth_lambda > 0.0 and W.shape[0] > 1:\\n        dW = np.diff(W, axis=0)\\n        reg += float(weight_smooth_lambda) * float(np.mean(np.sum(dW * dW, axis=1)))\\n    return float(reg)\\n\\ndef _fit_detection_mixture(target_by_freq, regime_by_freq, thresh_order, x_anchor_by_freq, mode=\'per_cv\', use_entmax: bool = False, entmax_alpha: float = 1.1, weight_kl: float = 0.0, weight_entropy_floor_k: float = 0.0, weight_entropy_floor_lambda: float = 0.0, weight_max_cap: float = 1.0, weight_max_cap_lambda: float = 0.0, weight_smooth_lambda: float = 0.0):\\n    if mode != \'per_cv\':\\n        raise ValueError(f\'Unsupported detection mixture mode: {mode}\')\\n\\n    freqs = list(target_by_freq.keys())\\n    curves = {}\\n    biases = {}\\n    per_freq_rmse = {}\\n    weights_by_freq = {}\\n    weights_by_freq_cv = {}\\n    n_th = len(thresh_order)\\n    w0 = np.full(n_th, 1.0 / n_th, dtype=float)\\n    u0 = np.zeros(n_th, dtype=float)\\n\\n    def _rows_from_params(theta, n_rows: int) -> np.ndarray:\\n        arr = np.asarray(theta, dtype=float).reshape(n_rows, n_th)\\n        if use_entmax:\\n            return np.vstack([_entmax_bisect_1d(row, alpha=entmax_alpha) for row in arr])\\n        return _project_simplex_rows(arr)\\n\\n    def _penalty(W: np.ndarray) -> float:\\n        return _weight_penalty_rows(\\n            W,\\n            weight_kl=weight_kl,\\n            weight_entropy_floor_k=weight_entropy_floor_k,\\n            weight_entropy_floor_lambda=weight_entropy_floor_lambda,\\n            weight_max_cap=weight_max_cap,\\n            weight_max_cap_lambda=weight_max_cap_lambda,\\n            weight_smooth_lambda=weight_smooth_lambda,\\n        )\\n\\n    for f in freqs:\\n        target_y = np.asarray(target_by_freq[f][\'fit_y\'], dtype=float)\\n        fit_x = np.asarray(target_by_freq[f][\'fit_x\'], dtype=float)\\n        x_anchor = np.asarray(x_anchor_by_freq[f], dtype=float)\\n        T_anchor = np.stack([\\n            np.interp(x_anchor, curve_grid, np.asarray(regime_by_freq[f][t], dtype=float))\\n            for t in thresh_order\\n        ], axis=1)\\n        n_anchor = int(len(x_anchor))\\n\\n        if use_entmax:\\n            theta0 = np.tile(u0, n_anchor)\\n\\n            def objective(theta):\\n                W = _rows_from_params(theta, n_anchor)\\n                pred_abs_anchor = np.sum(W * T_anchor, axis=1)\\n                pred_abs = np.interp(curve_grid, x_anchor, pred_abs_anchor)\\n                pred = _normalize_curve(pred_abs)\\n                pred_fit = _sample_curve_to_x(curve_grid, pred, fit_x)\\n                bias = _fit_detection_bias(target_y, pred_fit)\\n                resid = target_y - (pred_fit + bias)\\n                return float(np.mean(resid ** 2)) + _penalty(W)\\n\\n            res = minimize(objective, theta0, method=\'L-BFGS-B\', options={\'maxiter\': 4000, \'ftol\': 1e-12})\\n            if not res.success:\\n                print(f\'[fig09] regularized detection per-point mixture fit warning @ {int(f)} Hz: {res.message}\')\\n            W = _rows_from_params(res.x, n_anchor)\\n        else:\\n            theta0 = np.tile(w0, n_anchor)\\n\\n            def unpack(theta):\\n                W = np.asarray(theta, dtype=float).reshape(n_anchor, n_th)\\n                return _project_simplex_rows(W)\\n\\n            def objective(theta):\\n                W = unpack(theta)\\n                pred_abs_anchor = np.sum(W * T_anchor, axis=1)\\n                pred_abs = np.interp(curve_grid, x_anchor, pred_abs_anchor)\\n                pred = _normalize_curve(pred_abs)\\n                pred_fit = _sample_curve_to_x(curve_grid, pred, fit_x)\\n                bias = _fit_detection_bias(target_y, pred_fit)\\n                resid = target_y - (pred_fit + bias)\\n                return float(np.mean(resid ** 2))\\n\\n            res = minimize(\\n                objective,\\n                theta0,\\n                method=\'L-BFGS-B\',\\n                bounds=[(0.0, 1.0)] * int(theta0.size),\\n                options={\'maxiter\': 4000, \'ftol\': 1e-12},\\n            )\\n            if not res.success:\\n                print(f\'[fig09] detection per-point mixture fit warning @ {int(f)} Hz: {res.message}\')\\n            W = unpack(res.x)\\n\\n        pred_abs_anchor = np.sum(W * T_anchor, axis=1)\\n        pred_abs = np.interp(curve_grid, x_anchor, pred_abs_anchor)\\n        weights_by_freq[f] = {t: float(np.mean(W[:, i])) for i, t in enumerate(thresh_order)}\\n        weights_by_freq_cv[f] = {\\n            float(x_anchor[k]): {t: float(W[k, i]) for i, t in enumerate(thresh_order)}\\n            for k in range(n_anchor)\\n        }\\n        pred = _normalize_curve(pred_abs)\\n        pred_fit = _sample_curve_to_x(curve_grid, pred, fit_x)\\n        bias = _fit_detection_bias(target_y, pred_fit)\\n        curves[f] = np.asarray(pred + bias, dtype=float)\\n        biases[f] = float(bias)\\n        per_freq_rmse[f] = _rmse(target_y, pred_fit + bias)\\n\\n    return {\\n        \'mode\': str(mode),\\n        \'weights_by_freq\': weights_by_freq,\\n        \'weights_by_freq_cv\': weights_by_freq_cv,\\n        \'biases\': biases,\\n        \'curves\': curves,\\n        \'per_freq_rmse\': per_freq_rmse,\\n        \'mean_rmse\': float(np.mean(list(per_freq_rmse.values()))),\\n    }\\n\\ndef _fit_single_regime_detection_models(target_by_freq, regime_by_freq, thresh_order):\\n    freqs = list(target_by_freq.keys())\\n    out = {}\\n    for t in thresh_order:\\n        curves = {}\\n        biases = {}\\n        per_freq_rmse = {}\\n        for f in freqs:\\n            target_y = np.asarray(target_by_freq[f][\'fit_y\'], dtype=float)\\n            fit_x = np.asarray(target_by_freq[f][\'fit_x\'], dtype=float)\\n            pred_y_abs = np.asarray(regime_by_freq[f][t], dtype=float)\\n            pred = _normalize_curve(pred_y_abs)\\n            pred_fit = _sample_curve_to_x(curve_grid, pred, fit_x)\\n            bias = _fit_detection_bias(target_y, pred_fit)\\n            curve = pred + bias\\n            curves[f] = curve\\n            biases[f] = float(bias)\\n            per_freq_rmse[f] = _rmse(target_y, pred_fit + bias)\\n        out[t] = {\\n            \'biases\': biases,\\n            \'curves\': curves,\\n            \'per_freq_rmse\': per_freq_rmse,\\n            \'mean_rmse\': float(np.mean(list(per_freq_rmse.values()))),\\n        }\\n    return out\\n\\nbehavior_targets = {}\\nbehavior_targets_points = {}\\nregime_targets = {}\\nregime_targets_points = {}\\nx_by_freq = {}\\nx_by_freq_points = {}\\nfor f in plot_freqs:\\n    sb = beh[np.isclose(beh[\'freq\'], f)].copy()\\n    grp = (\\n        sb.groupby(\'cv\', as_index=False)\\n        .agg(mean_norm=(\'norm_threshold\', \'mean\'), sem_norm=(\'norm_threshold\', sem))\\n        .sort_values(\'cv\')\\n    )\\n    grp_points = grp[np.isclose(grp[\'cv\'].to_numpy(float)[:, None], DETECTION_DATAPOINT_CVS[None, :], atol=1e-9).any(axis=1)].copy()\\n    behavior_targets[f] = _build_behavior_target(grp, curve_grid)\\n    behavior_targets_points[f] = _build_behavior_target_datapoints(grp_points, curve_grid)\\n\\n    sub_f = agg_all[(agg_all[\'cutline\'] == cutline) & np.isclose(agg_all[\'freq\'], f)].sort_values(\'cv\')\\n    x_vals, curves = _build_regime_abs_curves(sub_f, THRESH_ORDER, curve_grid)\\n    x_by_freq[f] = np.asarray(x_vals, dtype=float)\\n    x_by_freq_points[f] = np.asarray([x for x in x_vals if np.any(np.isclose(float(x), DETECTION_DATAPOINT_CVS, atol=1e-9))], dtype=float)\\n    regime_targets[f] = curves\\n    sub_f_points = agg_slack[(agg_slack[\'cutline\'] == cutline) & np.isclose(agg_slack[\'freq\'], f)].sort_values(\'cv\')\\n    _x_vals_pts, curves_points = _build_regime_abs_curves(sub_f_points, THRESH_ORDER, curve_grid)\\n    regime_targets_points[f] = curves_points\\n\\nsingle_regime_fits = _fit_single_regime_detection_models(behavior_targets, regime_targets, THRESH_ORDER)\\nbest_single_thresh = min(THRESH_ORDER, key=lambda t: single_regime_fits[t][\'mean_rmse\'])\\nbest_single_fit = single_regime_fits[best_single_thresh]\\n\\nmixture_fit_per_cv_reg = _fit_detection_mixture(\\n    behavior_targets,\\n    regime_targets,\\n    THRESH_ORDER,\\n    x_by_freq,\\n    mode=\'per_cv\',\\n    use_entmax=DETECTION_EXPLORATORY_REG_USE_ENTMAX,\\n    entmax_alpha=DETECTION_EXPLORATORY_REG_ENTMAX_ALPHA,\\n    weight_kl=DETECTION_EXPLORATORY_REG_WEIGHT_KL,\\n    weight_entropy_floor_k=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K,\\n    weight_entropy_floor_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA,\\n    weight_max_cap=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP,\\n    weight_max_cap_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA,\\n    weight_smooth_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA,\\n)\\n\\nmixture_fit_points_per_cv_reg = _fit_detection_mixture(\\n    behavior_targets_points,\\n    regime_targets_points,\\n    THRESH_ORDER,\\n    x_by_freq_points,\\n    mode=\'per_cv\',\\n    use_entmax=DETECTION_EXPLORATORY_REG_USE_ENTMAX,\\n    entmax_alpha=DETECTION_EXPLORATORY_REG_ENTMAX_ALPHA,\\n    weight_kl=DETECTION_EXPLORATORY_REG_WEIGHT_KL,\\n    weight_entropy_floor_k=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K,\\n    weight_entropy_floor_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA,\\n    weight_max_cap=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP,\\n    weight_max_cap_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA,\\n    weight_smooth_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA,\\n)\\n\\ndef _detection_rmse_summary_row(fit_target: str, model: str, fit: dict) -> dict:\\n    return {\\n        \'Fit Target\': fit_target,\\n        \'Model\': model,\\n        **{f\'{int(f)} Hz RMSE\': float(fit[\'per_freq_rmse\'][f]) for f in plot_freqs},\\n        \'Mean RMSE\': float(fit[\'mean_rmse\']),\\n    }\\n\\n# Compact output matching the detection-threshold manuscript wording.\\ndetection_matching_summary_df = pd.DataFrame([\\n    _detection_rmse_summary_row(\\n        \'Behavioral quadratic trends\',\\n        f\'Best single regime ({_detection_model_label(best_single_thresh)})\',\\n        best_single_fit,\\n    ),\\n    _detection_rmse_summary_row(\\n        \'Behavioral quadratic trends\',\\n        \'Per-point mixture (all repetitions)\',\\n        mixture_fit_per_cv_reg,\\n    ),\\n    _detection_rmse_summary_row(\\n        \'Behavioral datapoints\',\\n        \'Per-point mixture (Slack-matched repetition)\',\\n        mixture_fit_points_per_cv_reg,\\n    ),\\n])\\ndetection_matching_summary_df.to_csv(MANUSCRIPT_OUT_DIR / \'table_fig09_detection_threshold_matching_rmse.csv\', index=False)\\n\\ndetection_matching_summary_display = detection_matching_summary_df[[\'Fit Target\', \'Model\', \'Mean RMSE\']].copy()\\ndetection_matching_summary_display[\'Mean RMSE\'] = detection_matching_summary_display[\'Mean RMSE\'].map(lambda x: f\'{float(x):.4f}\')\\ndisplay(detection_matching_summary_display)\\n\\n# Final Slack-rep detection matching plot.\\nfig_pts, axes_pts = plt.subplots(3, 5, figsize=(18, 11), sharex=\'col\')\\nfor j, f in enumerate(plot_freqs):\\n    sub_f = agg_all[(agg_all[\'cutline\'] == cutline) & np.isclose(agg_all[\'freq\'], f)].sort_values(\'cv\')\\n    if sub_f.empty:\\n        continue\\n\\n    ax_top = axes_pts[0, j]\\n    for t in THRESH_ORDER:\\n        ax_top.plot(sub_f[\'cv\'], sub_f[f\'{t}_mean\'], marker=\'o\', markersize=2.8, linewidth=1.4, color=thr_colors[t])\\n    ax_top.set_title(f\'{int(f)} Hz\')\\n    ax_top.set_xlim(-0.03, 1.03)\\n    ax_top.grid(True, alpha=0.25)\\n\\n    ax_mid = axes_pts[1, j]\\n    for t in THRESH_ORDER:\\n        y = np.asarray(sub_f[f\'{t}_mean\'].to_numpy(float), dtype=float)\\n        y = np.where(np.isfinite(y), y, np.nan)\\n        y_max = np.nanmax(y) if np.any(np.isfinite(y)) else np.nan\\n        if not np.isfinite(y_max) or abs(y_max) < 1e-12:\\n            continue\\n        y_norm = y / y_max\\n        ax_mid.plot(sub_f[\'cv\'], y_norm, marker=\'o\', markersize=2.8, linewidth=1.4, color=thr_colors[t])\\n    if f in bottom_ylim:\\n        ylo, yhi = bottom_ylim[f]\\n        ax_mid.set_ylim(ylo + 0.05, yhi + 0.025)\\n        ax_mid.set_yticks(np.linspace(ylo + 0.05, yhi, 4))\\n    ax_mid.set_xlim(-0.03, 1.03)\\n    ax_mid.grid(True, alpha=0.25)\\n\\n    ax_bot = axes_pts[2, j]\\n    stats = behavior_targets[f]\\n    ax_bot.plot(curve_grid, mixture_fit_per_cv_reg[\'curves\'][f], linestyle=\'-\', linewidth=2.2, color=\'#f4a261\', alpha=1.0, zorder=1)\\n    ax_bot.plot(curve_grid, best_single_fit[\'curves\'][f], linestyle=\'-\', linewidth=1.6, color=thr_colors[best_single_thresh], zorder=2)\\n    ax_bot.plot(curve_grid, stats[\'yy\'], linestyle=(0, (2.0, 3.4)), linewidth=1.6, color=\'k\', alpha=0.95, zorder=3)\\n    ax_bot.errorbar(stats[\'xb\'], stats[\'yb\'], yerr=stats[\'eb\'], fmt=\'o\', color=\'k\', ecolor=\'k\', markersize=3, capsize=2, linewidth=1.0, zorder=5)\\n    stats_slack = behavior_targets_points[f]\\n    x_fit = np.asarray(stats_slack[\'fit_x\'], dtype=float)\\n    y_mix_fit = _sample_curve_to_x(curve_grid, mixture_fit_points_per_cv_reg[\'curves\'][f], x_fit)\\n    ax_bot.scatter(x_fit, y_mix_fit, s=52, marker=\'s\', color=\'#f4a261\', edgecolor=\'white\', linewidth=0.4, zorder=4)\\n    if f in bottom_ylim_panel_c:\\n        ylo, yhi = bottom_ylim_panel_c[f]\\n        ax_bot.set_ylim(ylo, yhi)\\n        _yticks = np.linspace(ylo, yhi, 6)\\n        ax_bot.set_yticks(_yticks)\\n        ax_bot.set_yticklabels([f\'{ylo:.2f}\', \'\', \'\', \'\', \'\', f\'{yhi:.2f}\'])\\n    ax_bot.grid(True, alpha=0.25)\\n    ax_bot.set_xlabel(\'Coefficient of Variation (CV)\')\\n\\n    ax_bot.set_xlim(-0.03, 1.03)\\n    ax_bot.set_xticks(np.round(np.arange(0.0, 1.01, 0.1), 1))\\n    ax_bot.set_xticklabels([\'0\' if np.isclose(v, 0.0) else f\'{v:.1f}\' if int(round(v * 10)) % 2 == 0 else \'\' for v in np.round(np.arange(0.0, 1.01, 0.1), 1)])\\n\\naxes_pts[0, 0].set_ylabel(r\'Threshold Amplitude ($\\\\mu$A)\')\\naxes_pts[1, 0].set_ylabel(\'Normalized Threshold\')\\naxes_pts[2, 0].set_ylabel(\'Normalized Threshold\')\\nthreshold_handles_pts = [plt.Line2D([0], [0], color=thr_colors[t], marker=\'o\', markersize=3.5, linewidth=1.4, label=_math_tick_label(\'I\', t)) for t in THRESH_ORDER]\\nfig_pts.legend(handles=threshold_handles_pts, title=\'Spike-Count Threshold\', loc=\'lower center\', ncol=9, frameon=False, fontsize=12, title_fontsize=12)\\noverlay_handles_pts = [\\n    plt.Line2D([0], [0], color=\'k\', linestyle=(0, (2.0, 3.4)), linewidth=1.6, label=\'Behavior (Quadratic)\'),\\n    plt.Line2D([0], [0], color=thr_colors[best_single_thresh], linestyle=\'-\', linewidth=1.6, label=f\'Best Single ({_math_tick_label(\\"I\\", best_single_thresh)})\'),\\n    plt.Line2D([0], [0], color=\'#f4a261\', linestyle=\'-\', linewidth=2.2, label=\'Per-Point Mixture\'),\\n    plt.Line2D([0], [0], linestyle=\'None\', marker=\'s\', markersize=7.5, markerfacecolor=\'#f4a261\', markeredgecolor=\'white\', markeredgewidth=0.5, label=\'Per-Point Mixture (Slack Rep)\'),\\n]\\naxes_pts[2, 0].legend(handles=overlay_handles_pts, loc=\'lower left\', frameon=False, fontsize=8, handlelength=3.0)\\nfig_pts.suptitle(\'Detection Threshold Behavioral Matching\', y=0.995, fontsize=20)\\nplt.tight_layout(rect=[0, 0.06, 1, 0.95])\\nfor row_idx, panel_lab in enumerate([\'a\', \'b\', \'c\']):\\n    pos = axes_pts[row_idx, 0].get_position()\\n    fig_pts.text(pos.x0 - 0.055, pos.y1, panel_lab, ha=\'left\', va=\'top\', fontsize=13, fontweight=\'bold\')\\nplt.show()\\n\\nfig_pts.savefig(MANUSCRIPT_OUT_DIR / \'fig09_detection_threshold_behavioral_matching.png\', bbox_inches=\'tight\')\\nfig_pts.savefig(MANUSCRIPT_OUT_DIR / \'fig09_detection_threshold_behavioral_matching.pdf\', bbox_inches=\'tight\')\\n\\ncategory_fit_points = mixture_fit_points_per_cv_reg\\ncv_mass_order_points = sorted(float(cv) for cv in category_fit_points[\'weights_by_freq_cv\'][plot_freqs[0]].keys())\\n\\nexploratory_weight_rows_points = []\\nfor f in plot_freqs:\\n    for cv in cv_mass_order_points:\\n        row = {\'Frequency (Hz)\': int(f), \'CV\': float(cv), \'Mixture Mode\': str(category_fit_points[\'mode\'])}\\n        row.update({_detection_model_label(t): float(category_fit_points[\'weights_by_freq_cv\'][f][float(cv)][t]) for t in THRESH_ORDER})\\n        exploratory_weight_rows_points.append(row)\\nexploratory_weight_points_df = pd.DataFrame(exploratory_weight_rows_points)\\ncv_mass_order_all = sorted(float(cv) for cv in mixture_fit_per_cv_reg[\'weights_by_freq_cv\'][plot_freqs[0]].keys())\\nexploratory_weight_rows_all = []\\nfor f in plot_freqs:\\n    for cv in cv_mass_order_all:\\n        row = {\'Frequency (Hz)\': int(f), \'CV\': float(cv), \'Mixture Mode\': str(mixture_fit_per_cv_reg[\'mode\'])}\\n        row.update({_detection_model_label(t): float(mixture_fit_per_cv_reg[\'weights_by_freq_cv\'][f][float(cv)][t]) for t in THRESH_ORDER})\\n        exploratory_weight_rows_all.append(row)\\nexploratory_weight_df_all = pd.DataFrame(exploratory_weight_rows_all)\\n\\nfig_ex_pts, axes_ex_pts = plt.subplots(2, len(plot_freqs), figsize=(19.5, 8.0), sharey=True)\\nif len(plot_freqs) == 1:\\n    axes_ex_pts = np.asarray(axes_ex_pts).reshape(2, 1)\\nregime_colors_pts = [thr_colors[t] for t in THRESH_ORDER]\\nfor j, f in enumerate(plot_freqs):\\n    ax_top = axes_ex_pts[0, j]\\n    Y_top = np.asarray([\\n        exploratory_weight_df_all.loc[exploratory_weight_df_all[\'Frequency (Hz)\'] == int(f), _detection_model_label(t)].to_numpy(float)\\n        for t in THRESH_ORDER\\n    ], dtype=float)\\n    ax_top.stackplot(cv_mass_order_all, *Y_top, colors=regime_colors_pts, alpha=0.9, linewidth=0.0)\\n    ax_top.set_title(f\'{int(f)} Hz\', fontsize=12)\\n    ax_top.set_xlim(min(cv_mass_order_all), max(cv_mass_order_all))\\n    ax_top.set_ylim(0.0, 1.0)\\n    ax_top.set_xticks(np.round(np.arange(0.0, 1.01, 0.1), 1))\\n    ax_top.tick_params(axis=\'x\', labelbottom=False)\\n    ax_top.grid(True, alpha=0.22)\\n\\n    ax_bot = axes_ex_pts[1, j]\\n    Y_bot = np.asarray([\\n        exploratory_weight_points_df.loc[exploratory_weight_points_df[\'Frequency (Hz)\'] == int(f), _detection_model_label(t)].to_numpy(float)\\n        for t in THRESH_ORDER\\n    ], dtype=float)\\n    ax_bot.stackplot(cv_mass_order_points, *Y_bot, colors=regime_colors_pts, alpha=0.9, linewidth=0.0)\\n    ax_bot.set_xlabel(\'Coefficient of Variation (CV)\')\\n    ax_bot.set_xlim(min(cv_mass_order_points), max(cv_mass_order_points))\\n    ax_bot.set_ylim(0.0, 1.0)\\n    ax_bot.set_xticks(cv_mass_order_points)\\n    ax_bot.grid(True, alpha=0.22)\\naxes_ex_pts[0, 0].set_ylabel(\'Regime Mass\\\\nAll Reps\')\\naxes_ex_pts[1, 0].set_ylabel(\'Regime Mass\\\\nSlack Rep\')\\nhandles_ex_pts = [plt.Rectangle((0, 0), 1, 1, facecolor=thr_colors[t], edgecolor=\'none\', alpha=0.9, label=_math_tick_label(\'I\', t)) for t in THRESH_ORDER]\\nfig_ex_pts.legend(handles=handles_ex_pts, loc=\'lower center\', bbox_to_anchor=(0.5, 0.01), ncol=9, frameon=False, fontsize=10)\\nfig_ex_pts.suptitle(\'Detection Regime Mass by Threshold (All-Reps vs Slack-Rep)\', y=1.01, fontsize=18)\\nplt.tight_layout(rect=[0, 0.07, 1, 0.95])\\nplt.show()\\n\\n# Export the two-row all-reps vs slack-rep regime-mass data and figure.\\nexploratory_weight_df_all.to_csv(MANUSCRIPT_OUT_DIR / \'suppfig05_detection_mixture_weights_individual_regimes.csv\', index=False)\\nexploratory_weight_all_vs_slack_df = pd.concat([\\n    exploratory_weight_df_all.assign(Source=\'All-Reps\'),\\n    exploratory_weight_points_df.assign(Source=\'Slack-Rep\')\\n], ignore_index=True)\\nexploratory_weight_all_vs_slack_df.to_csv(MANUSCRIPT_OUT_DIR / \'suppfig05_detection_mixture_weights.csv\', index=False)\\nfig_ex_pts.savefig(MANUSCRIPT_OUT_DIR / \'suppfig05_detection_mixture_weights.png\', bbox_inches=\'tight\')\\nfig_ex_pts.savefig(MANUSCRIPT_OUT_DIR / \'suppfig05_detection_mixture_weights.pdf\', bbox_inches=\'tight\')\\n\\n# --- Figure E: cv0-based discrimination spike numbers (3x3) ---\\nfig, axes = plot_cv0_response_counts(df_cv0, cutline=cutline)\\nplt.show()\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig06_cv0_based_threshold_spike_counts.png\', bbox_inches=\'tight\')\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig06_cv0_based_threshold_spike_counts.pdf\', bbox_inches=\'tight\')\\n\\nprint(\'Wrote detection/discrimination-threshold figures to:\', MANUSCRIPT_OUT_DIR)", "6": "# Feature-engineering manuscript figures\\n\\nif not FEAT_SIM_CSV.exists():\\n    raise FileNotFoundError(f\'Missing feature-engineering input: {FEAT_SIM_CSV}\')\\n\\nif str((REPO_ROOT / \'src\' / \'discrimination\').resolve()) not in sys.path:\\n    sys.path.insert(0, str((REPO_ROOT / \'src\' / \'discrimination\').resolve()))\\n\\nfrom discrimination_core import (\\n    FeatureSettings,\\n    parse_spike_list,\\n    detect_events_binned,\\n    refine_events_to_spikes,\\n    compute_trial_features,\\n    rate_vote_hardjnd,\\n    timing_evidence_lv,\\n)\\n\\nPIPE_T0_MS = 0.0\\nPIPE_T1_MS = 2000.0\\nPIPE_MRG_INIT_MS = 100.0\\nPIPE_MIN_CT = 1\\nPIPE_CAP1 = True\\nPIPE_GAP_CLOSE_MS = 0.0\\nPIPE_MIN_EVENT_MS = 0.0\\nPIPE_JND_K = 2.5\\nPIPE_BINS = (1.0, 10.0)\\n\\nTHRESH_SHORT_LABELS = {\\n    \'I_first\': \'first\',\\n    \'I_volley5\': \'volley5\',\\n    \'I_volley10\': \'volley10\',\\n    \'I_5\': \'5\',\\n    \'I_10\': \'10\',\\n    \'I_15\': \'15\',\\n    \'I_25\': \'25\',\\n    \'I_50\': \'50\',\\n    \'I_max\': \'max\',\\n}\\n\\ndef _th_suffix(th):\\n    return THRESH_SHORT_LABELS.get(str(th), str(th))\\n\\ndef _math_regime(symbol: str, key: str, hat: bool = False) -> str:\\n    base = rf\'\\\\hat{{{symbol}}}\' if hat else symbol\\n    return rf\'${base}_{{{key}}}$\'\\n\\ndef _math_thresh(symbol: str, th: str, hat: bool = False) -> str:\\n    return _math_regime(symbol, _th_suffix(th), hat=hat)\\n\\nsim = pd.read_csv(FEAT_SIM_CSV)\\nsim[\'freq\'] = to_num(sim[\'freq\'])\\nsim[\'cv\'] = to_num(sim[\'cv\'])\\nrep_col = \'rep_index\' if \'rep_index\' in sim.columns else (\'rep\' if \'rep\' in sim.columns else None)\\nif rep_col is None:\\n    raise RuntimeError(\'Could not find rep column in cv0_based_responses.csv\')\\nsim[\'rep\'] = to_num(sim[rep_col]).astype(\'Int64\')\\nspikes_col = _get_spikes_col(sim)\\nif spikes_col is None:\\n    raise RuntimeError(\'Could not find spikes column in cv0_based_responses.csv\')\\n\\nsim[\'thresh_type\'] = sim[\'thresh_type\'].astype(str)\\nsim = sim[np.isfinite(sim[\'freq\']) & np.isfinite(sim[\'cv\']) & sim[\'rep\'].notna()].copy()\\nsim[\'rep\'] = sim[\'rep\'].astype(int)\\nsim = sim.reset_index(drop=True)\\n\\nSIM_IDX = {}\\nfor i, r in sim.iterrows():\\n    key = (float(r[\'freq\']), str(r[\'thresh_type\']), float(r[\'cv\']), int(r[\'rep\']))\\n    if key not in SIM_IDX:\\n        SIM_IDX[key] = int(i)\\n\\nCVS_BY_FT = {}\\nREPS_BY_FT = {}\\nfor (f, th), g in sim.groupby([\'freq\', \'thresh_type\']):\\n    CVS_BY_FT[(float(f), str(th))] = sorted(set(float(x) for x in g[\'cv\'].dropna().tolist()))\\n    REPS_BY_FT[(float(f), str(th))] = sorted(set(int(x) for x in g[\'rep\'].dropna().tolist()))\\n\\n\\ndef get_spikes(freq: float, th: str, cv: float, rep: int) -> np.ndarray:\\n    idx = SIM_IDX.get((float(freq), str(th), float(cv), int(rep)), None)\\n    if idx is None:\\n        return np.asarray([], dtype=float)\\n    return parse_spike_list(sim.iloc[idx][spikes_col])\\n\\n\\ndef rate_keys_for_bin(bin_ms: float):\\n    return (\'n_spikes\',) if float(bin_ms) <= 1.0 + 1e-9 else (\'n_events\', \'spikes_per_event\')\\n\\n\\ndef trial_features(freq: float, th: str, cv: float, rep: int, bin_ms: float):\\n    fs = FeatureSettings(\\n        t0_ms=PIPE_T0_MS,\\n        t1_ms=PIPE_T1_MS,\\n        bin_ms=float(bin_ms),\\n        min_ct=PIPE_MIN_CT,\\n        cap1=PIPE_CAP1,\\n        gap_close_ms=PIPE_GAP_CLOSE_MS,\\n        min_event_ms=PIPE_MIN_EVENT_MS,\\n        mrg_init_ms=PIPE_MRG_INIT_MS,\\n    )\\n    spikes = get_spikes(freq, th, cv, rep)\\n    feat = compute_trial_features(spikes, fs)\\n    _edges, _active, events_bin, spikes_f = detect_events_binned(spikes, fs)\\n    events = refine_events_to_spikes(events_bin, spikes_f)\\n    feat[\'spikes_filtered\'] = np.asarray(spikes_f, dtype=float)\\n    feat[\'events\'] = events\\n    return feat\\n\\n\\ndef trial_evidence(freq: float, th: str, cv: float, rep: int, bin_ms: float, jnd_k: float):\\n    fx = trial_features(freq, th, cv, rep, bin_ms)\\n    f0 = trial_features(freq, th, 0.0, rep, bin_ms)\\n    f1 = trial_features(freq, th, 1.0, rep, bin_ms)\\n\\n    x_rate = {k: fx.get(k, np.nan) for k in (\'n_spikes\', \'n_events\', \'spikes_per_event\')}\\n    t0_rate = {k: f0.get(k, np.nan) for k in (\'n_spikes\', \'n_events\', \'spikes_per_event\')}\\n    t1_rate = {k: f1.get(k, np.nan) for k in (\'n_spikes\', \'n_events\', \'spikes_per_event\')}\\n\\n    e_rate = rate_vote_hardjnd(x_rate, t0_rate, t1_rate, float(jnd_k), rate_keys_for_bin(bin_ms))\\n    e_time = timing_evidence_lv(fx.get(\'silence_lv\', np.nan), f0.get(\'silence_lv\', np.nan), f1.get(\'silence_lv\', np.nan))\\n    return float(e_rate), float(e_time), fx\\n\\n\\ndef endpoint_pm1_matrix(y_mat: np.ndarray, y0: float, y1: float):\\n    arr = np.asarray(y_mat, dtype=float)\\n    den = float(y1 - y0)\\n    if (not np.isfinite(den)) or abs(den) < 1e-12:\\n        return np.zeros_like(arr)\\n    a = 2.0 / den\\n    b = -1.0 - a * float(y0)\\n    return a * arr + b\\n\\n\\ndef sem_axis0(a: np.ndarray):\\n    out = np.zeros(a.shape[1], dtype=float)\\n    for j in range(a.shape[1]):\\n        col = np.asarray(a[:, j], dtype=float)\\n        col = col[np.isfinite(col)]\\n        if col.size <= 1:\\n            out[j] = 0.0\\n        else:\\n            out[j] = np.std(col, ddof=1) / np.sqrt(col.size)\\n    return out\\n\\n\\ndef collect_rep_mats(freq, th, bin_ms, reps, cvs, jnd_k):\\n    er = np.full((len(reps), len(cvs)), np.nan, dtype=float)\\n    et = np.full((len(reps), len(cvs)), np.nan, dtype=float)\\n    for i, rep in enumerate(reps):\\n        for j, cv in enumerate(cvs):\\n            e_r, e_t, _ = trial_evidence(freq, th, cv, rep, bin_ms, jnd_k)\\n            er[i, j] = e_r\\n            et[i, j] = e_t\\n    return er, et\\n\\n\\ndef normalize_mean_sem(cvs, er_mat, et_mat):\\n    cvs = np.asarray(cvs, dtype=float)\\n    i0 = int(np.where(np.isclose(cvs, 0.0))[0][0])\\n    i1 = int(np.where(np.isclose(cvs, 1.0))[0][0])\\n\\n    er_mean_raw = np.nanmean(er_mat, axis=0)\\n    et_mean_raw = np.nanmean(et_mat, axis=0)\\n    rr = np.nanmean(np.abs(er_mat) > 1e-12, axis=0)\\n\\n    er_pm1 = endpoint_pm1_matrix(er_mat, float(er_mean_raw[i0]), float(er_mean_raw[i1])) * rr[None, :]\\n    et_pm1 = endpoint_pm1_matrix(et_mat, float(et_mean_raw[i0]), float(et_mean_raw[i1]))\\n\\n    return {\\n        \'er_mean\': np.nanmean(er_pm1, axis=0),\\n        \'er_sem\': sem_axis0(er_pm1),\\n        \'et_mean\': np.nanmean(et_pm1, axis=0),\\n        \'et_sem\': sem_axis0(et_pm1),\\n        \'rate_rel\': rr,\\n    }\\n\\n\\ndef square_trace(events, t0, t1):\\n    bounds = [float(t0), float(t1)]\\n    clean = []\\n    singles = []\\n    for ev in events:\\n        s = float(getattr(ev, \'start_ms\', np.nan))\\n        e = float(getattr(ev, \'end_ms\', np.nan))\\n        if not (np.isfinite(s) and np.isfinite(e)):\\n            continue\\n        s = max(float(t0), min(float(t1), s))\\n        e = max(float(t0), min(float(t1), e))\\n        if e < s:\\n            continue\\n        if e == s:\\n            singles.append(s)\\n            continue\\n        clean.append((s, e))\\n        bounds.extend([s, e])\\n\\n    if not clean:\\n        return np.asarray([t0, t1], dtype=float), np.asarray([0.0, 0.0], dtype=float), np.asarray(singles, dtype=float)\\n\\n    x = np.asarray(sorted(set(bounds)), dtype=float)\\n    y = np.zeros(x.size, dtype=float)\\n    for i in range(x.size - 1):\\n        m = 0.5 * (x[i] + x[i + 1])\\n        on = any((m >= s) and (m < e) for s, e in clean)\\n        y[i] = 1.0 if on else 0.0\\n    y[-1] = y[-2] if y.size >= 2 else 0.0\\n    return x, y, np.asarray(singles, dtype=float)\\n\\n\\n# Shared feature-engineering caches\\nCACHE_DIR = REPO_ROOT / \'data\' / \'discrimination\' / \'per-block\'\\nCACHE_DIR.mkdir(parents=True, exist_ok=True)\\nCACHE_SINGLE_JSON = CACHE_DIR / \'feature_cache_single.json\'\\nCACHE_SINGLE_META_JSON = CACHE_DIR / \'feature_cache_single_meta.json\'\\nCACHE_ALL_CSV = CACHE_DIR / \'feature_cache_all.csv\'\\nCACHE_ALL_META_JSON = CACHE_DIR / \'feature_cache_all_meta.json\'\\nCACHE_ALL_MC_TRAIN_CSV = CACHE_DIR / \'feature_cache_all_mc_train.csv\'\\nCACHE_ALL_MC_TRAIN_META_JSON = CACHE_DIR / \'feature_cache_all_mc_train_meta.json\'\\n\\n\\ndef _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:\\n    import hashlib\\n    h = hashlib.sha256()\\n    with open(path, \'rb\') as f:\\n        while True:\\n            b = f.read(chunk_size)\\n            if not b:\\n                break\\n            h.update(b)\\n    return h.hexdigest()\\n\\n\\ndef _portable_relpath(path: Path, root: Path) -> str:\\n    try:\\n        rel = path.resolve().relative_to(root.resolve())\\n        return str(rel).replace(\'\\\\\\\\\', \'/\')\\n    except Exception:\\n        return path.name\\n\\n\\nSOURCE_REL = _portable_relpath(FEAT_SIM_CSV, REPO_ROOT)\\nSOURCE_SHA256 = _file_sha256(FEAT_SIM_CSV)\\nSOURCE_SIZE = int(FEAT_SIM_CSV.stat().st_size)\\nPIPELINE_META = {\\n    \'t0_ms\': float(PIPE_T0_MS),\\n    \'t1_ms\': float(PIPE_T1_MS),\\n    \'mrg_init_ms\': float(PIPE_MRG_INIT_MS),\\n    \'min_ct\': int(PIPE_MIN_CT),\\n    \'cap1\': bool(PIPE_CAP1),\\n    \'gap_close_ms\': float(PIPE_GAP_CLOSE_MS),\\n    \'min_event_ms\': float(PIPE_MIN_EVENT_MS),\\n}\\n\\n\\n# Fixed manuscript selection\\nfreq = 200.0 if any(np.isclose(sim[\'freq\'], 200.0)) else float(sorted(sim[\'freq\'].unique())[-1])\\nths = sorted(set(str(t) for (f, t) in CVS_BY_FT.keys() if np.isclose(f, freq)))\\nth = \'I_10\' if \'I_10\' in ths else (ths[0] if ths else None)\\nif th is None:\\n    raise RuntimeError(\'No threshold types available for feature-engineering figure.\')\\n\\nreps_all = REPS_BY_FT.get((freq, th), [])\\nif not reps_all:\\n    raise RuntimeError(f\'No reps for freq={freq}, thresh={th}\')\\nrep = 1 if 1 in reps_all else int(reps_all[0])\\n\\ncvs = sorted(float(c) for c in CVS_BY_FT.get((freq, th), []))\\nif (0.0 not in cvs) or (1.0 not in cvs):\\n    raise RuntimeError(\'Selected freq/thresh is missing template CVs 0/1.\')\\n\\nsingle_meta = {\\n    \'cache_version\': 3,\\n    \'cache_kind\': \'feature_cache_single\',\\n    \'source_relpath\': str(SOURCE_REL),\\n    \'source_sha256\': str(SOURCE_SHA256),\\n    \'source_size\': int(SOURCE_SIZE),\\n    \'selection\': {\\n        \'freq\': float(freq),\\n        \'thresh_type\': str(th),\\n        \'rep\': int(rep),\\n        \'cvs\': [float(c) for c in cvs],\\n        \'bins_ms\': [1.0, 10.0],\\n        \'jnd_k\': float(PIPE_JND_K),\\n    },\\n    \'pipeline\': dict(PIPELINE_META),\\n}\\n\\nuse_single_cache = False\\nif CACHE_SINGLE_JSON.exists() and CACHE_SINGLE_META_JSON.exists():\\n    try:\\n        prev_single_meta = json.loads(CACHE_SINGLE_META_JSON.read_text(encoding=\'utf-8\'))\\n        use_single_cache = (prev_single_meta == single_meta)\\n    except Exception:\\n        use_single_cache = False\\n\\nif use_single_cache:\\n    single_payload = json.loads(CACHE_SINGLE_JSON.read_text(encoding=\'utf-8\'))\\n\\n    cvs = [float(c) for c in single_payload[\'cvs\']]\\n    cv_arr = np.asarray(cvs, dtype=float)\\n\\n    spikes_by_cv = {float(k): np.asarray(v, dtype=float) for k, v in single_payload[\'spikes_1\'].items()}\\n    square_by_cv = {\\n        float(k): {\\n            \'x\': np.asarray(v.get(\'x\', []), dtype=float),\\n            \'y\': np.asarray(v.get(\'y\', []), dtype=float),\\n            \'single\': np.asarray(v.get(\'single\', []), dtype=float),\\n        }\\n        for k, v in single_payload[\'square_10\'].items()\\n    }\\n\\n    ev_rate = {\\n        1.0: [float(x) for x in single_payload[\'ev_rate_1\']],\\n        10.0: [float(x) for x in single_payload[\'ev_rate_10\']],\\n    }\\n    ev_time = {\\n        1.0: [float(x) for x in single_payload[\'ev_time_1\']],\\n        10.0: [float(x) for x in single_payload[\'ev_time_10\']],\\n    }\\n    rate_feats = {\\n        \'n_spikes_1\': [float(x) for x in single_payload[\'rate_feats\'][\'n_spikes_1\']],\\n        \'n_events_10\': [float(x) for x in single_payload[\'rate_feats\'][\'n_events_10\']],\\n        \'spe_10\': [float(x) for x in single_payload[\'rate_feats\'][\'spe_10\']],\\n    }\\n    timing_feats = {\\n        \'lv_1\': [float(x) for x in single_payload[\'timing_feats\'][\'lv_1\']],\\n        \'lv_10\': [float(x) for x in single_payload[\'timing_feats\'][\'lv_10\']],\\n    }\\n    norm1 = {k: np.asarray(v, dtype=float) for k, v in single_payload[\'norm1\'].items()}\\n    norm10 = {k: np.asarray(v, dtype=float) for k, v in single_payload[\'norm10\'].items()}\\n\\nelse:\\n    # Selected-rep traces + plot-ready burst envelopes\\n    spikes_by_cv = {}\\n    square_by_cv = {}\\n    ev_rate = {1.0: [], 10.0: []}\\n    ev_time = {1.0: [], 10.0: []}\\n\\n    rate_feats = {\'n_spikes_1\': [], \'n_events_10\': [], \'spe_10\': []}\\n    timing_feats = {\'lv_1\': [], \'lv_10\': []}\\n\\n    for cv in cvs:\\n        e_r1, e_t1, f1 = trial_evidence(freq, th, cv, rep, 1.0, PIPE_JND_K)\\n        e_r10, e_t10, f10 = trial_evidence(freq, th, cv, rep, 10.0, PIPE_JND_K)\\n\\n        ev_rate[1.0].append(e_r1)\\n        ev_rate[10.0].append(e_r10)\\n        ev_time[1.0].append(e_t1)\\n        ev_time[10.0].append(e_t10)\\n\\n        spikes = np.asarray(f1.get(\'spikes_filtered\', []), dtype=float)\\n        spikes_by_cv[float(cv)] = spikes\\n\\n        x_sq, y_sq, x_single = square_trace(f10.get(\'events\', []), PIPE_T0_MS, PIPE_T1_MS)\\n        square_by_cv[float(cv)] = {\\n            \'x\': np.asarray(x_sq, dtype=float),\\n            \'y\': np.asarray(y_sq, dtype=float),\\n            \'single\': np.asarray(x_single, dtype=float),\\n        }\\n\\n        rate_feats[\'n_spikes_1\'].append(f1.get(\'n_spikes\', np.nan))\\n        rate_feats[\'n_events_10\'].append(f10.get(\'n_events\', np.nan))\\n        rate_feats[\'spe_10\'].append(f10.get(\'spikes_per_event\', np.nan))\\n        timing_feats[\'lv_1\'].append(f1.get(\'silence_lv\', np.nan))\\n        timing_feats[\'lv_10\'].append(f10.get(\'silence_lv\', np.nan))\\n\\n    # Mean/SEM evidence across reps\\n    er1_mat, et1_mat = collect_rep_mats(freq, th, 1.0, reps_all, cvs, PIPE_JND_K)\\n    er10_mat, et10_mat = collect_rep_mats(freq, th, 10.0, reps_all, cvs, PIPE_JND_K)\\n    norm1 = normalize_mean_sem(cvs, er1_mat, et1_mat)\\n    norm10 = normalize_mean_sem(cvs, er10_mat, et10_mat)\\n\\n    cv_arr = np.asarray(cvs, dtype=float)\\n\\n    single_payload = {\\n        \'cvs\': [float(c) for c in cvs],\\n        \'spikes_1\': {f\'{float(k):g}\': np.asarray(v, dtype=float).tolist() for k, v in spikes_by_cv.items()},\\n        \'square_10\': {\\n            f\'{float(k):g}\': {\\n                \'x\': np.asarray(v[\'x\'], dtype=float).tolist(),\\n                \'y\': np.asarray(v[\'y\'], dtype=float).tolist(),\\n                \'single\': np.asarray(v[\'single\'], dtype=float).tolist(),\\n            }\\n            for k, v in square_by_cv.items()\\n        },\\n        \'ev_rate_1\': [float(x) for x in ev_rate[1.0]],\\n        \'ev_rate_10\': [float(x) for x in ev_rate[10.0]],\\n        \'ev_time_1\': [float(x) for x in ev_time[1.0]],\\n        \'ev_time_10\': [float(x) for x in ev_time[10.0]],\\n        \'rate_feats\': {k: [float(x) for x in vals] for k, vals in rate_feats.items()},\\n        \'timing_feats\': {k: [float(x) for x in vals] for k, vals in timing_feats.items()},\\n        \'norm1\': {k: np.asarray(v, dtype=float).tolist() for k, v in norm1.items()},\\n        \'norm10\': {k: np.asarray(v, dtype=float).tolist() for k, v in norm10.items()},\\n    }\\n    try:\\n        CACHE_SINGLE_JSON.write_text(json.dumps(single_payload), encoding=\'utf-8\')\\n        CACHE_SINGLE_META_JSON.write_text(json.dumps(single_meta, indent=2), encoding=\'utf-8\')\\n    except Exception:\\n        pass\\n\\n# --- Figure F: release-core-consistent workflow ---\\nfig = plt.figure(figsize=(17.5, 10.8), constrained_layout=True)\\ngs = fig.add_gridspec(4, 2, height_ratios=[1.10, 1.10, 0.95, 0.95], width_ratios=[1.0, 1.0])\\n\\nax_raster = fig.add_subplot(gs[0, :])\\nax_env = fig.add_subplot(gs[1, :], sharex=ax_raster, sharey=ax_raster)\\nax_f_rate = fig.add_subplot(gs[2, 0])\\nax_f_time = fig.add_subplot(gs[2, 1])\\nax_e_rate = fig.add_subplot(gs[3, 0], sharex=ax_f_rate)\\nax_e_time = fig.add_subplot(gs[3, 1], sharex=ax_f_time)\\n\\ncmap_cv = plt.get_cmap(\'viridis\')\\ncv_levels = [0.10] if len(cvs) == 1 else np.linspace(0.10, 0.95, len(cvs))\\ncv_colors = {cv: cmap_cv(float(cv_levels[i])) for i, cv in enumerate(cvs)}\\noffsets = np.arange(len(cvs))[::-1]\\ncv_to_offset = {cv: offsets[i] for i, cv in enumerate(cvs)}\\nrow_half = 0.40\\nspike_half = row_half\\n\\nymin = float(np.min(offsets) - row_half - 0.05)\\nymax = float(np.max(offsets) + row_half + 0.05)\\n\\n# panel 1: raster (1 ms)\\nfor cv in cvs:\\n    off = float(cv_to_offset[cv])\\n    spikes = np.asarray(spikes_by_cv.get(float(cv), []), dtype=float)\\n    if spikes.size:\\n        ax_raster.vlines(spikes, off - spike_half, off + spike_half, linewidth=0.9, alpha=0.95, color=cv_colors[cv])\\nax_raster.set_title(\'Spikes (1 ms bins)\')\\nax_raster.set_yticks(list(cv_to_offset.values()))\\nax_raster.set_yticklabels([f\'{float(cv):.1f}\' for cv in cvs])\\nax_raster.set_ylabel(\'Coefficient of Variation (CV)\')\\nax_raster.set_ylim(ymin, ymax)\\ntime_ticks = np.arange(float(PIPE_T0_MS), float(PIPE_T1_MS) + 1e-9, 250.0)\\ntime_xlim = (float(PIPE_T0_MS) - 5.0, float(PIPE_T1_MS) + 5.0)\\nax_raster.set_xticks(time_ticks)\\nax_raster.set_xlim(*time_xlim)\\nax_raster.grid(True, axis=\'x\', alpha=0.25)\\nax_raster.tick_params(axis=\'x\', which=\'both\', bottom=True, labelbottom=False)\\n\\n# panel 2: timing envelope (10 ms bursts)\\nfor cv in cvs:\\n    off = float(cv_to_offset[cv])\\n    sq = square_by_cv.get(float(cv), None)\\n    if sq is None:\\n        x_sq = np.asarray([PIPE_T0_MS, PIPE_T1_MS], dtype=float)\\n        y_sq = np.asarray([0.0, 0.0], dtype=float)\\n        x_single = np.asarray([], dtype=float)\\n    else:\\n        x_sq = np.asarray(sq.get(\'x\', []), dtype=float)\\n        y_sq = np.asarray(sq.get(\'y\', []), dtype=float)\\n        x_single = np.asarray(sq.get(\'single\', []), dtype=float)\\n    if x_sq.size >= 2:\\n        y_centered = off + (2.0 * y_sq - 1.0) * row_half\\n        ax_env.step(x_sq, y_centered, where=\'post\', linewidth=1.6, color=cv_colors[cv])\\n    if x_single.size:\\n        ax_env.vlines(x_single, off - row_half, off + row_half, linewidth=1.4, alpha=0.95, color=cv_colors[cv])\\nax_env.set_title(\'Bursts (10 ms bins)\')\\nax_env.set_yticks(list(cv_to_offset.values()))\\nax_env.set_yticklabels([f\'{float(cv):.1f}\' for cv in cvs])\\nax_env.set_ylabel(\'Coefficient of Variation (CV)\')\\nax_env.set_ylim(ymin, ymax)\\nax_env.set_xticks(time_ticks)\\nax_env.set_xlabel(\'Time (ms)\')\\nax_env.grid(True, axis=\'x\', alpha=0.25)\\nax_env.set_ylim(ax_raster.get_ylim())\\n\\nRATE_COLOR = \'#E30613\'  # I_10 rate color\\nTIMING_COLOR = \'#0575AB\'  # I_10 timing color\\ncv_ticks_dense = np.round(np.arange(0.0, 1.01, 0.1), 1)\\ncv_ticklabels_sparse = [f\'{x:.1f}\' if int(round(x * 10)) % 2 == 0 else \'\' for x in cv_ticks_dense]\\n\\n# panel 3: selected-rep features\\nh_ns = ax_f_rate.plot(\\n    cv_arr,\\n    np.asarray(rate_feats[\'n_spikes_1\'], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\':\',\\n    color=RATE_COLOR,\\n    label=\'n_spikes (1 ms)\',\\n)[0]\\nax_f_rate_b = ax_f_rate.twinx()\\nh_ne = ax_f_rate_b.plot(\\n    cv_arr,\\n    np.asarray(rate_feats[\'n_events_10\'], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\'--\',\\n    color=RATE_COLOR,\\n    label=\'n_bursts (10 ms)\',\\n)[0]\\nh_spe = ax_f_rate_b.plot(\\n    cv_arr,\\n    np.asarray(rate_feats[\'spe_10\'], float),\\n    marker=\'o\',\\n    linewidth=2.2,\\n    linestyle=(0, (3, 1, 1, 1)),\\n    color=RATE_COLOR,\\n    label=\'spikes_per_burst (10 ms)\',\\n)[0]\\n\\n# Add headroom so upper-left legend does not overlap traces too much.\\nspike_vals = np.asarray(rate_feats[\'n_spikes_1\'], float)\\nspike_vals = spike_vals[np.isfinite(spike_vals)]\\nif spike_vals.size:\\n    s_min = float(np.min(spike_vals))\\n    s_max = float(np.max(spike_vals))\\n    s_span = max(1e-6, s_max - s_min)\\n    ax_f_rate.set_ylim(s_min - 0.06 * s_span, s_max + 0.10 * s_span)\\n\\nburst_vals = np.concatenate([\\n    np.asarray(rate_feats[\'n_events_10\'], float),\\n    np.asarray(rate_feats[\'spe_10\'], float),\\n])\\nburst_vals = burst_vals[np.isfinite(burst_vals)]\\nif burst_vals.size:\\n    b_min = float(np.min(burst_vals))\\n    b_max = float(np.max(burst_vals))\\n    b_span = max(1e-6, b_max - b_min)\\n    # Keep right axis non-negative to avoid visual confusion in ticks.\\n    b_lo = max(0.0, b_min - 0.15 * b_span)\\n    b_hi = b_max + 0.75 * b_span\\n    if b_hi <= b_lo:\\n        b_hi = b_lo + 1.0\\n    ax_f_rate_b.set_ylim(b_lo, b_hi)\\n\\nax_f_rate.set_title(r\'Rate Features ($F_{rate}$)\')\\nax_f_rate.set_ylabel(\'Spike Metric\')\\nax_f_rate_b.set_ylabel(\'Burst Metric\')\\nax_f_rate_b.spines[\'right\'].set_visible(True)\\nax_f_rate_b.spines[\'right\'].set_linewidth(1.0)\\nax_f_rate_b.spines[\'right\'].set_color(\'0.35\')\\nax_f_rate_b.tick_params(axis=\'y\', which=\'both\', right=True, labelright=True, direction=\'out\', length=4, pad=4)\\nax_f_rate.grid(False)\\nax_f_rate.set_xticks(cv_ticks_dense)\\nax_f_rate.tick_params(axis=\'x\', which=\'both\', bottom=True, labelbottom=False)\\nrate_feat_left_handles = [\\n    plt.Line2D([0], [0], color=RATE_COLOR, linestyle=\':\', linewidth=2.4, marker=\'None\', label=\'n_spikes\'),\\n]\\nrate_feat_right_handles = [\\n    plt.Line2D([0], [0], color=RATE_COLOR, linestyle=\'--\', linewidth=2.4, marker=\'None\', label=\'n_bursts\'),\\n    plt.Line2D([0], [0], color=RATE_COLOR, linestyle=(0, (3, 1, 1, 1)), linewidth=2.4, marker=\'None\', label=\'spikes_per_burst\'),\\n]\\nleg_primary = ax_f_rate.legend(\\n    handles=rate_feat_left_handles,\\n    title=\'Primary Axis\',\\n    frameon=False,\\n    fontsize=8,\\n    title_fontsize=8,\\n    loc=\'upper left\',\\n    handlelength=4.6,\\n    handletextpad=0.6,\\n)\\nleg_secondary = ax_f_rate.legend(\\n    handles=rate_feat_right_handles,\\n    title=\'Secondary Axis\',\\n    frameon=False,\\n    fontsize=8,\\n    title_fontsize=8,\\n    loc=\'upper right\',\\n    handlelength=4.6,\\n    handletextpad=0.6,\\n)\\nax_f_rate.add_artist(leg_primary)\\n\\nh_lv_isi = ax_f_time.plot(\\n    cv_arr,\\n    np.asarray(timing_feats[\'lv_1\'], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\':\',\\n    color=TIMING_COLOR,\\n    label=\'LV_ISI\',\\n)[0]\\nax_f_time_b = ax_f_time.twinx()\\nh_lv_ibi = ax_f_time_b.plot(\\n    cv_arr,\\n    np.asarray(timing_feats[\'lv_10\'], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\'--\',\\n    color=TIMING_COLOR,\\n    label=\'LV_IBI\',\\n)[0]\\n\\nlv_isi_vals = np.asarray(timing_feats[\'lv_1\'], float)\\nlv_isi_vals = lv_isi_vals[np.isfinite(lv_isi_vals)]\\nif lv_isi_vals.size:\\n    li_min = float(np.min(lv_isi_vals))\\n    li_max = float(np.max(lv_isi_vals))\\n    li_span = max(1e-6, li_max - li_min)\\n    ax_f_time.set_ylim(li_min - 0.06 * li_span, li_max + 0.06 * li_span)\\n\\nlv_ibi_vals = np.asarray(timing_feats[\'lv_10\'], float)\\nlv_ibi_vals = lv_ibi_vals[np.isfinite(lv_ibi_vals)]\\nif lv_ibi_vals.size:\\n    lb_min = float(np.min(lv_ibi_vals))\\n    lb_max = float(np.max(lv_ibi_vals))\\n    lb_span = max(1e-6, lb_max - lb_min)\\n    ax_f_time_b.set_ylim(lb_min - 0.15 * lb_span, lb_max + 0.75 * lb_span)\\n\\nax_f_time.set_title(r\'Timing Features ($F_{timing}$)\')\\nax_f_time.set_ylabel(\'Spike Metric\')\\nax_f_time_b.set_ylabel(\'Burst Metric\')\\nax_f_time_b.spines[\'right\'].set_visible(True)\\nax_f_time_b.spines[\'right\'].set_linewidth(1.0)\\nax_f_time_b.spines[\'right\'].set_color(\'0.35\')\\nax_f_time_b.tick_params(axis=\'y\', which=\'both\', right=True, labelright=True, direction=\'out\', length=4, pad=4)\\nax_f_time.grid(False)\\nax_f_time.set_xticks(cv_ticks_dense)\\nax_f_time.tick_params(axis=\'x\', which=\'both\', bottom=True, labelbottom=False)\\n\\ntime_feat_left_handles = [\\n    plt.Line2D([0], [0], color=TIMING_COLOR, linestyle=\':\', linewidth=2.4, marker=\'None\', label=\'LV_ISI\'),\\n]\\ntime_feat_right_handles = [\\n    plt.Line2D([0], [0], color=TIMING_COLOR, linestyle=\'--\', linewidth=2.4, marker=\'None\', label=\'LV_IBI\'),\\n]\\nleg_time_primary = ax_f_time.legend(\\n    handles=time_feat_left_handles,\\n    title=\'Primary Axis\',\\n    frameon=False,\\n    fontsize=8,\\n    title_fontsize=8,\\n    loc=\'upper left\',\\n    handlelength=4.6,\\n    handletextpad=0.6,\\n)\\nleg_time_secondary = ax_f_time.legend(\\n    handles=time_feat_right_handles,\\n    title=\'Secondary Axis\',\\n    frameon=False,\\n    fontsize=8,\\n    title_fontsize=8,\\n    loc=\'upper right\',\\n    handlelength=4.6,\\n    handletextpad=0.6,\\n)\\nax_f_time.add_artist(leg_time_primary)\\n\\n# panel 4: selected-rep evidence (raw)\\n# For visual comparability with rate hard-JND evidence, show timing selected-rep evidence\\n# in endpoint_pm1 space as used by the model pipeline.\\ni0 = int(np.where(np.isclose(cv_arr, 0.0))[0][0])\\ni1 = int(np.where(np.isclose(cv_arr, 1.0))[0][0])\\nev_time_plot = {}\\nfor b in (1.0, 10.0):\\n    arr = np.asarray(ev_time[b], dtype=float)\\n    den = float(arr[i1] - arr[i0]) if arr.size else np.nan\\n    if (not np.isfinite(den)) or abs(den) < 1e-12:\\n        ev_time_plot[b] = np.zeros_like(arr)\\n    else:\\n        ev_time_plot[b] = 2.0 * (arr - float(arr[i0])) / den - 1.0\\nax_e_rate.plot(\\n    cv_arr,\\n    np.asarray(ev_rate[1.0], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\':\',\\n    color=RATE_COLOR,\\n    label=\'Spikes (1 ms)\',\\n)\\nax_e_rate.plot(\\n    cv_arr,\\n    np.asarray(ev_rate[10.0], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\'--\',\\n    color=RATE_COLOR,\\n    label=\'Bursts (10 ms)\',\\n)\\nax_e_rate.axhline(0.0, linewidth=1.0, alpha=0.5)\\nax_e_rate.set_title(r\'Rate Evidence ($E_{rate}$)\')\\nax_e_rate.set_xlabel(\'Coefficient of Variation (CV)\')\\nax_e_rate.set_ylabel(\'Normalized Evidence\', labelpad=0)\\nax_e_rate.set_xticks(cv_ticks_dense)\\nax_e_rate.set_xticklabels(cv_ticklabels_sparse)\\nax_e_rate.grid(True, alpha=0.25)\\nrate_ev_handles = [\\n    plt.Line2D([0], [0], color=RATE_COLOR, linestyle=\':\', linewidth=2.0, marker=\'None\', label=\'Spikes (1 ms)\'),\\n    plt.Line2D([0], [0], color=RATE_COLOR, linestyle=\'--\', linewidth=2.0, marker=\'None\', label=\'Bursts (10 ms)\'),\\n]\\nax_e_rate.legend(handles=rate_ev_handles, frameon=False, fontsize=8, handlelength=4.6, handletextpad=0.6)\\n\\nax_e_time.plot(\\n    cv_arr,\\n    np.asarray(ev_time_plot[1.0], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\':\',\\n    color=TIMING_COLOR,\\n    label=\'Spikes (1 ms)\',\\n)\\nax_e_time.plot(\\n    cv_arr,\\n    np.asarray(ev_time_plot[10.0], float),\\n    marker=\'o\',\\n    linewidth=2.0,\\n    linestyle=\'--\',\\n    color=TIMING_COLOR,\\n    label=\'Bursts (10 ms)\',\\n)\\nax_e_time.axhline(0.0, linewidth=1.0, alpha=0.5)\\nax_e_time.set_title(r\'Timing Evidence ($E_{timing}$)\')\\nax_e_time.set_xlabel(\'Coefficient of Variation (CV)\')\\nax_e_time.set_ylabel(\'Normalized Evidence\', labelpad=0)\\nax_e_time.set_xticks(cv_ticks_dense)\\nax_e_time.set_xticklabels(cv_ticklabels_sparse)\\nax_e_time.grid(True, alpha=0.25)\\ntime_ev_handles = [\\n    plt.Line2D([0], [0], color=TIMING_COLOR, linestyle=\':\', linewidth=2.0, marker=\'None\', label=\'Spikes (1 ms)\'),\\n    plt.Line2D([0], [0], color=TIMING_COLOR, linestyle=\'--\', linewidth=2.0, marker=\'None\', label=\'Bursts (10 ms)\'),\\n]\\nax_e_time.legend(handles=time_ev_handles, frameon=False, fontsize=8, handlelength=4.6, handletextpad=0.6)\\n\\nfig.canvas.draw()\\npos_raster = ax_raster.get_position()\\npos_env = ax_env.get_position()\\npos_f_rate = ax_f_rate.get_position()\\npos_f_time = ax_f_time.get_position()\\npos_e_rate = ax_e_rate.get_position()\\npos_e_time = ax_e_time.get_position()\\n\\n# Left-column labels share one vertical alignment; right-column labels share another.\\nx_left_labels = min(pos_raster.x0, pos_env.x0, pos_f_rate.x0, pos_e_rate.x0) - 0.04\\nx_right_labels = min(pos_f_time.x0, pos_e_time.x0) - 0.04\\ny_pad = 0.006\\n\\nleft_panel_positions = {\'a\': pos_raster, \'b\': pos_env, \'c\': pos_f_rate, \'e\': pos_e_rate}\\nright_panel_positions = {\'d\': pos_f_time, \'f\': pos_e_time}\\nleft_panel_y_offsets = {\'a\': y_pad, \'b\': y_pad, \'c\': y_pad - 0.008, \'e\': y_pad}\\nright_panel_y_offsets = {\'d\': y_pad - 0.008, \'f\': y_pad}\\nfor panel_lab in [\'a\', \'b\', \'c\', \'e\']:\\n    pos = left_panel_positions[panel_lab]\\n    fig.text(x_left_labels, pos.y1 + left_panel_y_offsets[panel_lab], panel_lab, ha=\'left\', va=\'bottom\', fontsize=12, fontweight=\'bold\')\\nfor panel_lab in [\'d\', \'f\']:\\n    pos = right_panel_positions[panel_lab]\\n    fig.text(x_right_labels, pos.y1 + right_panel_y_offsets[panel_lab], panel_lab, ha=\'left\', va=\'bottom\', fontsize=12, fontweight=\'bold\')\\n\\nfig.suptitle(\'I_10 Single Repetition Feature Engineering\', y=1.04)\\nplt.show()\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig07_single_repetition_feature_engineering.png\', bbox_inches=\'tight\')\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'fig07_single_repetition_feature_engineering.pdf\', bbox_inches=\'tight\')\\n\\n# --- Figure G: all-threshold evidence curves (all reps + MC training companion) ---\\nFREQ_GRID = [20.0, 50.0, 100.0, 200.0]\\nMC_TRAIN_DRAWS = 200\\nMC_TRAIN_FRACTION = 0.80\\nMC_TRAIN_SEED = 20260314\\n\\nall_cache_meta = {\\n    \'cache_version\': 3,\\n    \'cache_kind\': \'feature_cache_all\',\\n    \'source_relpath\': str(SOURCE_REL),\\n    \'source_sha256\': str(SOURCE_SHA256),\\n    \'source_size\': int(SOURCE_SIZE),\\n    \'freq_grid\': [float(f) for f in FREQ_GRID],\\n    \'bins_ms\': [1.0, 10.0],\\n    \'jnd_k\': float(PIPE_JND_K),\\n    \'pipeline\': dict(PIPELINE_META),\\n}\\nall_mc_train_cache_meta = {\\n    \'cache_version\': 1,\\n    \'cache_kind\': \'feature_cache_all_mc_train\',\\n    \'source_relpath\': str(SOURCE_REL),\\n    \'source_sha256\': str(SOURCE_SHA256),\\n    \'source_size\': int(SOURCE_SIZE),\\n    \'freq_grid\': [float(f) for f in FREQ_GRID],\\n    \'bins_ms\': [1.0, 10.0],\\n    \'jnd_k\': float(PIPE_JND_K),\\n    \'pipeline\': dict(PIPELINE_META),\\n    \'mc_train_draws\': int(MC_TRAIN_DRAWS),\\n    \'mc_train_fraction\': float(MC_TRAIN_FRACTION),\\n    \'mc_train_seed\': int(MC_TRAIN_SEED),\\n}\\n\\nREQUIRED_CURVE_COLS = {\\n    \'freq\', \'thresh_type\', \'bin_ms\', \'cv\', \'er_mean\', \'et_mean\', \'er_sem\', \'et_sem\', \'n_reps\'\\n}\\n\\n\\ndef _load_curve_cache(cache_csv: Path, cache_meta_json: Path, expected_meta: dict):\\n    use_cache = False\\n    df = pd.DataFrame()\\n    if cache_csv.exists() and cache_meta_json.exists():\\n        try:\\n            prev_meta = json.loads(cache_meta_json.read_text(encoding=\'utf-8\'))\\n            use_cache = (prev_meta == expected_meta)\\n        except Exception:\\n            use_cache = False\\n    if use_cache:\\n        df = pd.read_csv(cache_csv)\\n        if not REQUIRED_CURVE_COLS.issubset(set(df.columns)):\\n            use_cache = False\\n            df = pd.DataFrame()\\n    return use_cache, df\\n\\n\\ndef _coerce_curve_frame(df: pd.DataFrame) -> pd.DataFrame:\\n    if df.empty:\\n        return df\\n    out = df.copy()\\n    for col in (\'freq\', \'bin_ms\', \'cv\', \'er_mean\', \'et_mean\', \'er_sem\', \'et_sem\'):\\n        out[col] = pd.to_numeric(out[col], errors=\'coerce\')\\n    if \'n_reps\' in out.columns:\\n        out[\'n_reps\'] = pd.to_numeric(out[\'n_reps\'], errors=\'coerce\').fillna(0).astype(int)\\n    if \'n_train\' in out.columns:\\n        out[\'n_train\'] = pd.to_numeric(out[\'n_train\'], errors=\'coerce\').fillna(0).astype(int)\\n    if \'n_draws\' in out.columns:\\n        out[\'n_draws\'] = pd.to_numeric(out[\'n_draws\'], errors=\'coerce\').fillna(0).astype(int)\\n    out = out[np.isfinite(out[\'freq\']) & np.isfinite(out[\'bin_ms\']) & np.isfinite(out[\'cv\'])].copy()\\n    out[\'thresh_type\'] = out[\'thresh_type\'].astype(str)\\n    return out\\n\\n\\ndef _sem_from_rows(arr: np.ndarray) -> np.ndarray:\\n    arr = np.asarray(arr, dtype=float)\\n    if arr.ndim == 1:\\n        arr = arr[None, :]\\n    n = int(arr.shape[0])\\n    if n <= 1:\\n        return np.zeros(arr.shape[1], dtype=float)\\n    return np.nanstd(arr, axis=0, ddof=1) / np.sqrt(float(n))\\n\\n\\ndef _curve_rows_from_vectors(freq_i, th_i, b, cvs_i, er_vec, et_vec, er_sem_vec, et_sem_vec, n_reps, n_train=None, n_draws=None):\\n    rows = []\\n    for cv_i, er_i, et_i, er_se_i, et_se_i in zip(cvs_i, er_vec, et_vec, er_sem_vec, et_sem_vec):\\n        row = {\\n            \'freq\': float(freq_i),\\n            \'thresh_type\': str(th_i),\\n            \'bin_ms\': float(b),\\n            \'cv\': float(cv_i),\\n            \'er_mean\': float(er_i),\\n            \'et_mean\': float(et_i),\\n            \'er_sem\': float(er_se_i),\\n            \'et_sem\': float(et_se_i),\\n            \'n_reps\': int(n_reps),\\n        }\\n        if n_train is not None:\\n            row[\'n_train\'] = int(n_train)\\n        if n_draws is not None:\\n            row[\'n_draws\'] = int(n_draws)\\n        rows.append(row)\\n    return rows\\n\\n\\ndef _seed_for_freq_thresh(freq_i: float, th_i: str) -> int:\\n    key_sum = sum((j + 1) * ord(ch) for j, ch in enumerate(str(th_i)))\\n    return int(MC_TRAIN_SEED + round(float(freq_i) * 100.0) + key_sum)\\n\\n\\n_REP_MAT_CACHE = {}\\n\\n\\ndef _get_rep_mats(freq_i: float, th_i: str, b: float, reps_i, cvs_i):\\n    key = (float(freq_i), str(th_i), float(b), tuple(reps_i), tuple(float(c) for c in cvs_i))\\n    mats = _REP_MAT_CACHE.get(key)\\n    if mats is None:\\n        mats = collect_rep_mats(float(freq_i), th_i, float(b), reps_i, cvs_i, PIPE_JND_K)\\n        _REP_MAT_CACHE[key] = mats\\n    return mats\\n\\n\\ndef _build_all_curve_frame() -> pd.DataFrame:\\n    rows = []\\n    for freq_i in FREQ_GRID:\\n        ths_i = sorted(set(str(t) for (f, t) in CVS_BY_FT.keys() if np.isclose(float(f), float(freq_i))))\\n        for th_i in ths_i:\\n            cvs_i = sorted(float(c) for c in CVS_BY_FT.get((float(freq_i), th_i), []))\\n            if (0.0 not in cvs_i) or (1.0 not in cvs_i):\\n                continue\\n            reps_i = list(REPS_BY_FT.get((float(freq_i), th_i), []))\\n            if not reps_i:\\n                continue\\n            for b in (1.0, 10.0):\\n                er_mat, et_mat = _get_rep_mats(float(freq_i), th_i, float(b), reps_i, cvs_i)\\n                norm = normalize_mean_sem(cvs_i, er_mat, et_mat)\\n                rows.extend(\\n                    _curve_rows_from_vectors(\\n                        freq_i,\\n                        th_i,\\n                        b,\\n                        cvs_i,\\n                        np.asarray(norm[\'er_mean\'], dtype=float),\\n                        np.asarray(norm[\'et_mean\'], dtype=float),\\n                        np.asarray(norm[\'er_sem\'], dtype=float),\\n                        np.asarray(norm[\'et_sem\'], dtype=float),\\n                        n_reps=len(reps_i),\\n                    )\\n                )\\n    return pd.DataFrame(rows)\\n\\n\\ndef _build_mc_train_curve_frame() -> pd.DataFrame:\\n    rows = []\\n    for freq_i in FREQ_GRID:\\n        ths_i = sorted(set(str(t) for (f, t) in CVS_BY_FT.keys() if np.isclose(float(f), float(freq_i))))\\n        for th_i in ths_i:\\n            cvs_i = sorted(float(c) for c in CVS_BY_FT.get((float(freq_i), th_i), []))\\n            if (0.0 not in cvs_i) or (1.0 not in cvs_i):\\n                continue\\n            reps_i = list(REPS_BY_FT.get((float(freq_i), th_i), []))\\n            n_reps = len(reps_i)\\n            if n_reps < 2:\\n                continue\\n            n_train = max(1, min(n_reps - 1, int(np.floor(MC_TRAIN_FRACTION * n_reps))))\\n            rng = np.random.default_rng(_seed_for_freq_thresh(float(freq_i), th_i))\\n            split_indices = [np.sort(rng.choice(n_reps, size=n_train, replace=False)) for _draw in range(int(MC_TRAIN_DRAWS))]\\n            for b in (1.0, 10.0):\\n                er_all, et_all = _get_rep_mats(float(freq_i), th_i, float(b), reps_i, cvs_i)\\n                er_draws = []\\n                et_draws = []\\n                for train_idx in split_indices:\\n                    norm = normalize_mean_sem(cvs_i, er_all[train_idx, :], et_all[train_idx, :])\\n                    er_draws.append(np.asarray(norm[\'er_mean\'], dtype=float))\\n                    et_draws.append(np.asarray(norm[\'et_mean\'], dtype=float))\\n                er_draws = np.asarray(er_draws, dtype=float)\\n                et_draws = np.asarray(et_draws, dtype=float)\\n                rows.extend(\\n                    _curve_rows_from_vectors(\\n                        freq_i,\\n                        th_i,\\n                        b,\\n                        cvs_i,\\n                        np.nanmean(er_draws, axis=0),\\n                        np.nanmean(et_draws, axis=0),\\n                        _sem_from_rows(er_draws),\\n                        _sem_from_rows(et_draws),\\n                        n_reps=n_reps,\\n                        n_train=n_train,\\n                        n_draws=int(MC_TRAIN_DRAWS),\\n                    )\\n                )\\n    return pd.DataFrame(rows)\\n\\n\\nuse_all_cache, curves_df = _load_curve_cache(CACHE_ALL_CSV, CACHE_ALL_META_JSON, all_cache_meta)\\nif not use_all_cache:\\n    curves_df = _build_all_curve_frame()\\n    curves_df.to_csv(CACHE_ALL_CSV, index=False)\\n    CACHE_ALL_META_JSON.write_text(json.dumps(all_cache_meta, indent=2), encoding=\'utf-8\')\\ncurves_df = _coerce_curve_frame(curves_df)\\nif curves_df.empty:\\n    raise RuntimeError(\'Could not build all-repetition all-threshold curves for requested frequency grid.\')\\n\\nuse_all_mc_train_cache, curves_train_df = _load_curve_cache(\\n    CACHE_ALL_MC_TRAIN_CSV,\\n    CACHE_ALL_MC_TRAIN_META_JSON,\\n    all_mc_train_cache_meta,\\n)\\nif not use_all_mc_train_cache:\\n    curves_train_df = _build_mc_train_curve_frame()\\n    curves_train_df.to_csv(CACHE_ALL_MC_TRAIN_CSV, index=False)\\n    CACHE_ALL_MC_TRAIN_META_JSON.write_text(json.dumps(all_mc_train_cache_meta, indent=2), encoding=\'utf-8\')\\ncurves_train_df = _coerce_curve_frame(curves_train_df)\\nif curves_train_df.empty:\\n    raise RuntimeError(\'Could not build MC training all-threshold curves for requested frequency grid.\')\\n\\nth_present = sorted(set(curves_df[\'thresh_type\'].tolist()) | set(curves_train_df[\'thresh_type\'].tolist()))\\nth_order = [t for t in THRESH_ORDER if t in th_present] + [t for t in th_present if t not in THRESH_ORDER]\\n\\nth_colors_rate = {\\n    \'I_first\': \'#FC9CA2\',\\n    \'I_volley5\': \'#FB747D\',\\n    \'I_volley10\': \'#FA4C58\',\\n    \'I_5\': \'#F92432\',\\n    \'I_10\': \'#E30613\',\\n    \'I_15\': \'#C70512\',\\n    \'I_25\': \'#9F040E\',\\n    \'I_50\': \'#77030B\',\\n    \'I_max\': \'#500207\',\\n}\\nth_colors_timing = {\\n    \'I_first\': \'#00D4FF\',\\n    \'I_volley5\': \'#01BCEA\',\\n    \'I_volley10\': \'#02A5D5\',\\n    \'I_5\': \'#038DC0\',\\n    \'I_10\': \'#0575AB\',\\n    \'I_15\': \'#065D95\',\\n    \'I_25\': \'#074680\',\\n    \'I_50\': \'#082E6B\',\\n    \'I_max\': \'#091656\',\\n}\\n\\n\\ndef _add_legend_swatch(fig, x0: float, x1: float, y: float, color: str):\\n    fig.add_artist(\\n        plt.Line2D(\\n            [x0, x1],\\n            [y, y],\\n            transform=fig.transFigure,\\n            color=color,\\n            alpha=0.16,\\n            linewidth=6.4,\\n            solid_capstyle=\'round\',\\n            clip_on=False,\\n        )\\n    )\\n    fig.add_artist(\\n        plt.Line2D(\\n            [x0, x1],\\n            [y, y],\\n            transform=fig.transFigure,\\n            color=color,\\n            linewidth=2.4,\\n            solid_capstyle=\'round\',\\n            clip_on=False,\\n        )\\n    )\\n\\n\\ndef _add_regime_legend(fig):\\n    y_rate = 0.119\\n    y_timing = 0.105\\n    y_label = 0.112\\n    x_left = 0.10\\n    x_anchor_start = 0.14\\n    x_anchor_end = 0.89\\n    swatch_dx0 = 0.034\\n    swatch_dx1 = 0.012\\n\\n    fig.text(x_left, y_rate, \'Rate\', ha=\'right\', va=\'center\', fontsize=10, fontweight=\'semibold\')\\n    fig.text(x_left, y_timing, \'Timing\', ha=\'right\', va=\'center\', fontsize=10, fontweight=\'semibold\')\\n\\n    n_lab = max(1, len(th_order))\\n    xs = [0.5 * (x_anchor_start + x_anchor_end)] if n_lab == 1 else np.linspace(x_anchor_start, x_anchor_end, n_lab)\\n    for x, th_i in zip(xs, th_order):\\n        if th_i not in th_colors_rate or th_i not in th_colors_timing:\\n            continue\\n        _add_legend_swatch(fig, x - swatch_dx0, x - swatch_dx1, y_rate, th_colors_rate[th_i])\\n        _add_legend_swatch(fig, x - swatch_dx0, x - swatch_dx1, y_timing, th_colors_timing[th_i])\\n        fig.text(x, y_label, _math_thresh(\'E\', th_i, hat=True), ha=\'left\', va=\'center\', fontsize=12)\\n\\n\\ndef plot_all_thresh_combined_4x4(curve_frame: pd.DataFrame, out_stem: str, title: str):\\n    fig, axes = plt.subplots(4, 4, figsize=(18.8, 14.0), sharex=True, sharey=True, gridspec_kw={\'hspace\': 0.14})\\n    row_specs = [\\n        (\'rate\', 1.0),\\n        (\'rate\', 10.0),\\n        (\'timing\', 1.0),\\n        (\'timing\', 10.0),\\n    ]\\n\\n    y_ticks_dense = np.linspace(-1.0, 1.0, 9)\\n    y_major_vals = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)\\n\\n    for r, (bank_key, b) in enumerate(row_specs):\\n        ycol = \'er_mean\' if bank_key == \'rate\' else \'et_mean\'\\n        sem_col = \'er_sem\' if bank_key == \'rate\' else \'et_sem\'\\n        th_colors = th_colors_rate if bank_key == \'rate\' else th_colors_timing\\n        line_style = \':\' if float(b) <= 1.0 + 1e-9 else \'--\'\\n\\n        for c, freq_i in enumerate(FREQ_GRID):\\n            ax = axes[r, c]\\n            sub = curve_frame[np.isclose(curve_frame[\'freq\'], float(freq_i)) & np.isclose(curve_frame[\'bin_ms\'], float(b))]\\n\\n            for th_i in th_order:\\n                d = sub[sub[\'thresh_type\'] == th_i].sort_values(\'cv\')\\n                if d.empty:\\n                    continue\\n                x = np.asarray(d[\'cv\'], dtype=float)\\n                y = np.asarray(d[ycol], dtype=float)\\n                se = np.asarray(d[sem_col], dtype=float) if sem_col in d.columns else np.zeros_like(y)\\n                ax.fill_between(\\n                    x,\\n                    np.clip(y - se, -1.0, 1.0),\\n                    np.clip(y + se, -1.0, 1.0),\\n                    color=th_colors[th_i],\\n                    alpha=0.12,\\n                    linewidth=0,\\n                    zorder=1,\\n                )\\n                ax.plot(\\n                    x,\\n                    y,\\n                    marker=\'o\',\\n                    linewidth=1.8,\\n                    linestyle=line_style,\\n                    color=th_colors[th_i],\\n                    alpha=0.95,\\n                    zorder=2,\\n                    label=th_i,\\n                )\\n\\n            ax.set_ylim(-1.05, 1.05)\\n            ax.set_yticks(y_ticks_dense)\\n            y_tick_labels = []\\n            for _yt in y_ticks_dense:\\n                if np.any(np.isclose(float(_yt), y_major_vals, atol=1e-9)):\\n                    y_tick_labels.append(f\'{float(_yt):g}\')\\n                else:\\n                    y_tick_labels.append(\'\')\\n            ax.set_yticklabels(y_tick_labels)\\n            ax.grid(True, alpha=0.25)\\n            if r == 0:\\n                ax.set_title(f\'{int(round(freq_i))} Hz\')\\n            if r == 3:\\n                ax.set_xlabel(\'Coefficient of Variation (CV)\')\\n            if c == 0:\\n                if r in (0, 1):\\n                    ax.set_ylabel(\'Reliability Scaled\\\\nNormalized Evidence\')\\n                else:\\n                    ax.set_ylabel(\'Normalized Evidence\')\\n\\n    style_handles = [\\n        plt.Line2D([0], [0], color=\'black\', linestyle=\':\', linewidth=2.3, marker=\'None\', label=\'Spikes (1 ms)\'),\\n        plt.Line2D([0], [0], color=\'black\', linestyle=\'--\', linewidth=2.3, marker=\'None\', label=\'Bursts (10 ms)\'),\\n    ]\\n    lg_style = fig.legend(\\n        style_handles,\\n        [h.get_label() for h in style_handles],\\n        loc=\'lower center\',\\n        bbox_to_anchor=(0.5, 0.136),\\n        ncol=2,\\n        frameon=False,\\n        handlelength=3.0,\\n        columnspacing=1.8,\\n    )\\n    fig.add_artist(lg_style)\\n    _add_regime_legend(fig)\\n\\n    fig.suptitle(title, y=1.01)\\n    fig.subplots_adjust(left=0.06, right=0.985, top=0.915, bottom=0.215, wspace=0.04, hspace=0.14)\\n\\n    for r in range(len(row_specs)):\\n        x_mid_gap = 0.5 * (axes[r, 1].get_position().x1 + axes[r, 2].get_position().x0)\\n        row_top = max(axes[r, c].get_position().y1 for c in range(len(FREQ_GRID)))\\n        if r == 0:\\n            y_text = row_top + 0.006\\n        else:\\n            prev_bottom = min(axes[r - 1, c].get_position().y0 for c in range(len(FREQ_GRID)))\\n            y_text = 0.5 * (prev_bottom + row_top)\\n        fig.text(0.012, y_text, [\'a\', \'b\', \'c\', \'d\'][r], ha=\'left\', va=\'center\', fontsize=12, fontweight=\'bold\')\\n\\n    plt.show()\\n    fig.savefig(MANUSCRIPT_OUT_DIR / f\'{out_stem}.png\', bbox_inches=\'tight\')\\n    fig.savefig(MANUSCRIPT_OUT_DIR / f\'{out_stem}.pdf\', bbox_inches=\'tight\')\\n\\n\\ndef plot_combined_training_evidence_2x4(curve_frame: pd.DataFrame, out_stem: str, title: str):\\n    bins_use = (1.0, 10.0)\\n    freq_list = [f for f in FREQ_GRID if any(np.isclose(curve_frame[\'freq\'], float(f)))]\\n    if not freq_list:\\n        raise RuntimeError(\'No frequencies available for combined training evidence figure.\')\\n\\n    def _combine_bank(bank_key: str) -> pd.DataFrame:\\n        ycol = \'er_mean\' if bank_key == \'rate\' else \'et_mean\'\\n        sem_col = \'er_sem\' if bank_key == \'rate\' else \'et_sem\'\\n        rows = []\\n        sub = curve_frame.copy()\\n        cvs_local = sorted(float(x) for x in sub[\'cv\'].dropna().unique())\\n        for freq_i in freq_list:\\n            sub_f = sub[np.isclose(sub[\'freq\'], float(freq_i))]\\n            for th_i in th_order:\\n                for cv_i in cvs_local:\\n                    vals = []\\n                    ses = []\\n                    for b in bins_use:\\n                        d = sub_f[\\n                            (sub_f[\'thresh_type\'] == th_i)\\n                            & np.isclose(sub_f[\'cv\'], float(cv_i))\\n                            & np.isclose(sub_f[\'bin_ms\'], float(b))\\n                        ]\\n                        if d.empty:\\n                            continue\\n                        vals.append(float(d[ycol].iloc[0]))\\n                        ses.append(float(d[sem_col].iloc[0]) if sem_col in d.columns and np.isfinite(d[sem_col].iloc[0]) else 0.0)\\n                    if not vals:\\n                        continue\\n                    rows.append({\\n                        \'freq\': float(freq_i),\\n                        \'thresh_type\': str(th_i),\\n                        \'cv\': float(cv_i),\\n                        \'mean\': float(np.mean(vals)),\\n                        \'sem\': float(np.sqrt(np.sum(np.square(ses))) / max(1, len(vals))),\\n                    })\\n        return pd.DataFrame(rows)\\n\\n    rate_df = _combine_bank(\'rate\')\\n    timing_df = _combine_bank(\'timing\')\\n\\n    fig, axes = plt.subplots(2, len(freq_list), figsize=(18.5, 8.8), sharex=True, sharey=True)\\n    if len(freq_list) == 1:\\n        axes = np.asarray(axes).reshape(2, 1)\\n\\n    row_specs = [\\n        (\'rate\', rate_df, th_colors_rate, \'Combined Rate Evidence\'),\\n        (\'timing\', timing_df, th_colors_timing, \'Combined Timing Evidence\'),\\n    ]\\n\\n    y_ticks_dense = np.linspace(-1.0, 1.0, 9)\\n    y_major_vals = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)\\n\\n    for r_idx, (_bank_key, df_bank, th_colors, row_ylabel) in enumerate(row_specs):\\n        for c_idx, freq_i in enumerate(freq_list):\\n            ax = axes[r_idx, c_idx]\\n            d_freq = df_bank[np.isclose(df_bank[\'freq\'], float(freq_i))]\\n            for th_i in th_order:\\n                d = d_freq[d_freq[\'thresh_type\'] == th_i].sort_values(\'cv\')\\n                if d.empty:\\n                    continue\\n                x = np.asarray(d[\'cv\'], dtype=float)\\n                y = np.asarray(d[\'mean\'], dtype=float)\\n                se = np.asarray(d[\'sem\'], dtype=float)\\n                ax.fill_between(\\n                    x,\\n                    np.clip(y - se, -1.0, 1.0),\\n                    np.clip(y + se, -1.0, 1.0),\\n                    color=th_colors[th_i],\\n                    alpha=0.12,\\n                    linewidth=0,\\n                    zorder=1,\\n                )\\n                ax.plot(\\n                    x,\\n                    y,\\n                    marker=\'o\',\\n                    linewidth=2.0,\\n                    color=th_colors[th_i],\\n                    alpha=0.96,\\n                    zorder=2,\\n                )\\n            ax.set_ylim(-1.05, 1.05)\\n            ax.set_yticks(y_ticks_dense)\\n            y_tick_labels = []\\n            for _yt in y_ticks_dense:\\n                if np.any(np.isclose(float(_yt), y_major_vals, atol=1e-9)):\\n                    y_tick_labels.append(f\'{float(_yt):g}\')\\n                else:\\n                    y_tick_labels.append(\'\')\\n            ax.set_yticklabels(y_tick_labels)\\n            ax.grid(True, alpha=0.25)\\n            if r_idx == 0:\\n                ax.set_title(f\'{int(freq_i)} Hz\', fontsize=12)\\n            if c_idx == 0:\\n                ax.set_ylabel(row_ylabel)\\n            if r_idx == len(row_specs) - 1:\\n                ax.set_xlabel(\'Coefficient of Variation (CV)\')\\n\\n    fig.suptitle(title, y=0.985)\\n    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.22, wspace=0.06, hspace=0.14)\\n    _add_regime_legend(fig)\\n    plt.show()\\n    fig.savefig(MANUSCRIPT_OUT_DIR / f\'{out_stem}.png\', bbox_inches=\'tight\')\\n    fig.savefig(MANUSCRIPT_OUT_DIR / f\'{out_stem}.pdf\', bbox_inches=\'tight\')\\n\\n\\nplot_all_thresh_combined_4x4(\\n    curves_train_df,\\n    \'fig08_combined_normalized_evidence_curves\',\\n    \'Combined Normalized Evidence Curves\',\\n)\\n\\nprint(\'Single feature cache:\', \'loaded\' if use_single_cache else \'rebuilt\', \'->\', CACHE_SINGLE_JSON)\\nprint(\'All-threshold curve cache:\', \'loaded\' if use_all_cache else \'rebuilt\', \'->\', CACHE_ALL_CSV)\\nprint(\'MC training curve cache:\', \'loaded\' if use_all_mc_train_cache else \'rebuilt\', \'->\', CACHE_ALL_MC_TRAIN_CSV)\\nprint(\'Wrote feature-engineering figures to:\', MANUSCRIPT_OUT_DIR)\\n", "8": "import json\\nimport sys\\nfrom pathlib import Path\\nimport numpy as np\\nimport pandas as pd\\nimport matplotlib.pyplot as plt\\nfrom IPython.display import display\\n\\n# Resolve project-root-relative paths robustly whether cwd is repo root or notebooks/\\nif Path(\'results\').exists():\\n    RESULTS_DIR = Path(\'data/discrimination/per-block\')\\nelse:\\n    RESULTS_DIR = Path(\'../data/discrimination/per-block\')\\n\\nif Path(\'data\').exists():\\n    DEFAULT_SLACK_DIR = Path(\'data/discrimination/slack\')\\nelse:\\n    DEFAULT_SLACK_DIR = Path(\'../data/discrimination/slack\')\\n\\nif Path(\'src/discrimination\').exists():\\n    sys.path.append(str(Path(\'src/discrimination\').resolve()))\\nelif Path(\'../src/discrimination\').exists():\\n    sys.path.append(str(Path(\'../src/discrimination\').resolve()))\\n\\nfrom discrimination_core import load_slack_dir, map_pchoose_to_pcorrect_mixed, sigmoid\\nfrom scipy.stats import wilcoxon\\n\\nplt.rcParams[\'figure.dpi\'] = 120\\n\\nif Path(\'results\').exists():\\n    OUT_DIR = Path(\'results/figures\')\\nelse:\\n    OUT_DIR = Path(\'../results/figures\')\\nOUT_DIR.mkdir(parents=True, exist_ok=True)\\n\\nMODELS = [\'rate_combined\', \'timing_combined\', \'hybrid_combined\']\\nBLOCKS = [\'std0\', \'std1\']\\nFREQS = [20.0, 50.0, 100.0, 200.0]\\nTOPK_ORACLE = 25  # retained for provenance; this locked setup has one config/model per block\\n\\n# Requested manuscript colors (reuse old non-combined palette):\\n# rate_combined=red, timing_combined=blue, hybrid_combined=purple\\nCOLOR_MAP = {\\n    \'rate_combined\': \'#d62728\',\\n    \'timing_combined\': \'#1f77b4\',\\n    \'hybrid_combined\': \'#7b2cbf\',\\n}\\n\\nTHRESH_ORDER = [\'I_first\', \'I_volley5\', \'I_volley10\', \'I_5\', \'I_10\', \'I_15\', \'I_25\', \'I_50\', \'I_max\']\\n\\ndef _sem(x):\\n    a = np.asarray(x, dtype=float)\\n    a = a[np.isfinite(a)]\\n    if a.size <= 1:\\n        return 0.0\\n    return float(np.std(a, ddof=1) / np.sqrt(a.size))\\n\\ndef _parse_float_list(x):\\n    if isinstance(x, (list, tuple, np.ndarray)):\\n        a = np.asarray(x, dtype=float)\\n        return a[np.isfinite(a)]\\n    if x is None or (isinstance(x, float) and not np.isfinite(x)):\\n        return np.array([], dtype=float)\\n    s = str(x).strip()\\n    if not s:\\n        return np.array([], dtype=float)\\n    try:\\n        obj = json.loads(s)\\n        a = np.asarray(obj, dtype=float)\\n        return a[np.isfinite(a)]\\n    except Exception:\\n        return np.array([], dtype=float)\\n\\ndef _parse_num_dict(x):\\n    if isinstance(x, dict):\\n        d = x\\n    else:\\n        try:\\n            d = json.loads(str(x))\\n        except Exception:\\n            return {}\\n    out = {}\\n    for k, v in d.items():\\n        try:\\n            fv = float(v)\\n            if np.isfinite(fv):\\n                out[str(k)] = fv\\n        except Exception:\\n            continue\\n    return out\\n\\ndef _to_choice_display(cv_vals, p_correct_vals, block):\\n    # Matches the manuscript display convention used in 05 notebook\\n    # std0 -> display p(cv1); std1 -> display p(cv0)\\n    cv = np.asarray(cv_vals, dtype=float)\\n    p = np.asarray(p_correct_vals, dtype=float)\\n    if str(block) == \'std0\':\\n        return np.where(np.isclose(cv, 0.0), 1.0 - p, p)\\n    if str(block) == \'std1\':\\n        return np.where(np.isclose(cv, 1.0), 1.0 - p, p)\\n    return p\\n\\ndef _aggregate_curve_rows(sub_df, pred_col=\'p_pred_test_json\', obs_col=\'p_obs_json\'):\\n    curves_pred, curves_obs = [], []\\n    base_cv = None\\n    for _, r in sub_df.iterrows():\\n        cv = _parse_float_list(r.get(\'cv_json\', None))\\n        yp = _parse_float_list(r.get(pred_col, None))\\n        yo = _parse_float_list(r.get(obs_col, None))\\n        if cv.size == 0 or yp.size != cv.size:\\n            continue\\n        if base_cv is None:\\n            base_cv = cv\\n        if base_cv.size != cv.size or not np.allclose(base_cv, cv):\\n            continue\\n        curves_pred.append(yp)\\n        if yo.size == cv.size:\\n            curves_obs.append(yo)\\n    if base_cv is None or not curves_pred:\\n        return pd.DataFrame()\\n    Yp = np.vstack(curves_pred)\\n    out = pd.DataFrame({\\n        \'cv\': base_cv,\\n        \'pred_mean\': np.nanmean(Yp, axis=0),\\n        \'pred_sem\': np.nanstd(Yp, axis=0, ddof=1) / np.sqrt(Yp.shape[0]) if Yp.shape[0] > 1 else np.zeros(base_cv.size),\\n    })\\n    if curves_obs:\\n        Yo = np.vstack(curves_obs)\\n        out[\'obs_mean\'] = np.nanmean(Yo, axis=0)\\n        out[\'obs_sem\'] = np.nanstd(Yo, axis=0, ddof=1) / np.sqrt(Yo.shape[0]) if Yo.shape[0] > 1 else np.zeros(base_cv.size)\\n    return out\\n\\ndef _effective_rate_share_row(row):\\n    cv = _parse_float_list(row.get(\'cv_json\', None))\\n    rr = _parse_float_list(row.get(\'rate_drive_test_json\', None))\\n    tt = _parse_float_list(row.get(\'timing_drive_test_json\', None))\\n    rra = _parse_float_list(row.get(\'rate_drive_aux_test_json\', None))\\n    tta = _parse_float_list(row.get(\'timing_drive_aux_test_json\', None))\\n    if cv.size == 0:\\n        return pd.DataFrame()\\n    if rr.size != cv.size:\\n        rr = np.zeros(cv.size, dtype=float)\\n    if tt.size != cv.size:\\n        tt = np.zeros(cv.size, dtype=float)\\n    if rra.size != cv.size:\\n        rra = np.zeros(cv.size, dtype=float)\\n    if tta.size != cv.size:\\n        tta = np.zeros(cv.size, dtype=float)\\n\\n    gr = float(row.get(\'gain_rate\', 0.0)) if np.isfinite(row.get(\'gain_rate\', np.nan)) else 0.0\\n    gt = float(row.get(\'gain_timing\', 0.0)) if np.isfinite(row.get(\'gain_timing\', np.nan)) else 0.0\\n    gra = float(row.get(\'gain_rate_aux\', 0.0)) if np.isfinite(row.get(\'gain_rate_aux\', np.nan)) else 0.0\\n    gta = float(row.get(\'gain_timing_aux\', 0.0)) if np.isfinite(row.get(\'gain_timing_aux\', np.nan)) else 0.0\\n\\n    rate_sig = gr * rr + gra * rra\\n    time_sig = gt * tt + gta * tta\\n    rate_abs = np.abs(rate_sig)\\n    time_abs = np.abs(time_sig)\\n    den = rate_abs + time_abs\\n    with np.errstate(divide=\'ignore\', invalid=\'ignore\'):\\n        share = np.where(den > 1e-12, rate_abs / den, np.nan)\\n    return pd.DataFrame({\'cv\': cv, \'rate_share\': share})\\n\\ndef _aggregate_effective_share(sub_df):\\n    rows = []\\n    base_cv = None\\n    for _, r in sub_df.iterrows():\\n        c = _effective_rate_share_row(r)\\n        if c.empty:\\n            continue\\n        cv = c[\'cv\'].to_numpy(float)\\n        sh = c[\'rate_share\'].to_numpy(float)\\n        if base_cv is None:\\n            base_cv = cv\\n        if base_cv.size != cv.size or not np.allclose(base_cv, cv):\\n            continue\\n        rows.append(sh)\\n    if base_cv is None or not rows:\\n        return pd.DataFrame()\\n    Y = np.vstack(rows)\\n    return pd.DataFrame({\\n        \'cv\': base_cv,\\n        \'share_mean\': np.nanmean(Y, axis=0),\\n        \'share_sem\': np.nanstd(Y, axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros(base_cv.size),\\n    })\\n\\ndef _rmse_mean_sd(values):\\n    vals = np.asarray(values, dtype=float)\\n    vals = vals[np.isfinite(vals)]\\n    if vals.size == 0:\\n        return np.nan, np.nan\\n    mean = float(np.mean(vals))\\n    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0\\n    return mean, sd\\n\\ndef _holm_adjust(pvals):\\n    p = np.asarray(pvals, dtype=float)\\n    out = np.full(p.shape, np.nan, dtype=float)\\n    mask = np.isfinite(p)\\n    if not np.any(mask):\\n        return out\\n    pv = p[mask]\\n    order = np.argsort(pv)\\n    ranked = pv[order]\\n    m = ranked.size\\n    adj_sorted = np.empty(m, dtype=float)\\n    running = 0.0\\n    for i, val in enumerate(ranked):\\n        scaled = (m - i) * val\\n        running = max(running, scaled)\\n        adj_sorted[i] = min(running, 1.0)\\n    adj = np.empty(m, dtype=float)\\n    adj[order] = adj_sorted\\n    out[np.where(mask)[0]] = adj\\n    return out\\n\\ndef _p_to_stars(p):\\n    if not np.isfinite(p):\\n        return \'\'\\n    if p < 1e-3:\\n        return \'***\'\\n    if p < 1e-2:\\n        return \'**\'\\n    if p < 5e-2:\\n        return \'*\'\\n    return \'\'\\n\\ndef _format_p(p):\\n    if not np.isfinite(p):\\n        return \'NA\'\\n    if p < 1e-3:\\n        return f\'{p:.3e}\'\\n    return f\'{p:.6f}\'\\n\\ndef _paired_wilcoxon_greater(x, y):\\n    xa = np.asarray(x, dtype=float)\\n    ya = np.asarray(y, dtype=float)\\n    mask = np.isfinite(xa) & np.isfinite(ya)\\n    xa = xa[mask]\\n    ya = ya[mask]\\n    if xa.size == 0:\\n        return np.nan\\n    if np.allclose(xa, ya):\\n        return 1.0\\n    try:\\n        return float(wilcoxon(xa, ya, alternative=\'greater\', zero_method=\'wilcox\').pvalue)\\n    except Exception:\\n        return np.nan\\n\\ndef _add_sig_bracket(ax, x1, x2, y, text, h=0.004, lw=1.0):\\n    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=\'black\', linewidth=lw, clip_on=False)\\n    ax.text((x1 + x2) / 2.0, y + h + 0.0015, text, ha=\'center\', va=\'bottom\', fontsize=10)\\n\\ndef _build_block_rmse_summary(pb_df, unit=\'mc_block_mean\'):\\n    unit = str(unit).strip().lower()\\n    if unit == \'mc_block_mean\':\\n        unit_df = (\\n            pb_df.groupby([\'block\', \'mc\', \'model_bank\'], as_index=False)\\n            .agg(rmse_value=(\'rmse_test\', \'mean\'))\\n        )\\n        pair_index = [\'mc\']\\n    elif unit == \'raw_row\':\\n        unit_df = pb_df[[\'block\', \'freq\', \'mc\', \'model_bank\', \'rmse_test\']].copy()\\n        unit_df = unit_df.rename(columns={\'rmse_test\': \'rmse_value\'})\\n        pair_index = [\'freq\', \'mc\']\\n    else:\\n        raise ValueError(\\"unit must be one of {\'mc_block_mean\', \'raw_row\'}\\")\\n\\n    summary = (\\n        unit_df.groupby([\'block\', \'model_bank\'], as_index=False)\\n        .agg(\\n            per_block_rmse_mean=(\'rmse_value\', \'mean\'),\\n            per_block_rmse_sd=(\'rmse_value\', _sd),\\n            n_samples=(\'rmse_value\', \'size\'),\\n        )\\n    )\\n    summary[\'rank_per_block\'] = summary.groupby(\'block\')[\'per_block_rmse_mean\'].rank(method=\'dense\')\\n    _best = summary.groupby(\'block\')[\'per_block_rmse_mean\'].transform(\'min\')\\n    summary[\'pct_vs_best_per_block\'] = np.where(\\n        _best > 1e-12,\\n        100.0 * (summary[\'per_block_rmse_mean\'] - _best) / _best,\\n        np.nan,\\n    )\\n\\n    stat_rows = []\\n    for blk in BLOCKS:\\n        block_means = summary[summary[\'block\'] == blk].sort_values(\'per_block_rmse_mean\')\\n        best_model = str(block_means.iloc[0][\'model_bank\'])\\n        piv = unit_df[unit_df[\'block\'] == blk].pivot(index=pair_index, columns=\'model_bank\', values=\'rmse_value\').dropna()\\n        other_models = [m for m in block_means[\'model_bank\'].tolist() if str(m) != best_model]\\n        p_raw = [_paired_wilcoxon_greater(piv[str(m)], piv[best_model]) for m in other_models]\\n        p_adj = _holm_adjust(p_raw)\\n        for mdl, p0, p1 in zip(other_models, p_raw, p_adj):\\n            stat_rows.append({\\n                \'block\': blk,\\n                \'model_bank\': str(mdl),\\n                \'best_model\': best_model,\\n                \'n_pairs\': int(len(piv)),\\n                \'delta_rmse_mean\': float(np.nanmean(np.asarray(piv[str(mdl)], dtype=float) - np.asarray(piv[best_model], dtype=float))),\\n                \'p_raw\': p0,\\n                \'p_holm\': p1,\\n                \'stars\': _p_to_stars(p1),\\n                \'sig_mark\': \'*\' if np.isfinite(p1) and p1 < 5e-2 else \'\',\\n                \'unit\': unit,\\n            })\\n    stats = pd.DataFrame(stat_rows)\\n\\n    summary = summary.merge(stats[[\'block\', \'model_bank\', \'p_raw\', \'p_holm\', \'sig_mark\']], on=[\'block\', \'model_bank\'], how=\'left\')\\n    summary[\'sig_mark\'] = summary[\'sig_mark\'].fillna(\'\')\\n    summary[\'model_display\'] = summary[\'model_bank\'].astype(str).str.replace(\'_combined\', \'\', regex=False)\\n    summary[\'rmse_display\'] = [\\n        f\'{m:.3f} +/- {s:.3f}{st}\'\\n        for m, s, st in zip(summary[\'per_block_rmse_mean\'], summary[\'per_block_rmse_sd\'], summary[\'sig_mark\'])\\n    ]\\n\\n    display_table = summary.sort_values([\'block\', \'rank_per_block\', \'per_block_rmse_mean\']).reset_index(drop=True)[[\\n        \'block\', \'model_display\', \'rmse_display\', \'pct_vs_best_per_block\'\\n    ]].rename(columns={\\n        \'block\': \'Block\',\\n        \'model_display\': \'Model\',\\n        \'rmse_display\': \'Block RMSE Mean +/- SD\',\\n        \'pct_vs_best_per_block\': \'Difference vs. Best (%)\',\\n    })\\n    display_table[\'Difference vs. Best (%)\'] = display_table[\'Difference vs. Best (%)\'].map(lambda v: \'0\' if np.isfinite(v) and abs(v) < 5e-3 else (f\'{v:.2f}\' if np.isfinite(v) else \'NA\'))\\n    return summary, stats, display_table, unit_df\\n\\ndef _plot_block_rmse_bars(summary_df, stats_df, title, out_prefix):\\n    plot_order = [\'rate_combined\', \'timing_combined\', \'hybrid_combined\']\\n    plot_labels = [\'std0\', \'std1\']\\n    model_labels = {\'rate_combined\': \'Rate\', \'timing_combined\': \'Timing\', \'hybrid_combined\': \'Hybrid\'}\\n    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8))\\n    x = np.arange(len(BLOCKS), dtype=float)\\n    bar_width = 0.22\\n\\n    for bi, blk in enumerate(BLOCKS):\\n        sub = summary_df[summary_df[\'block\'] == blk].set_index(\'model_bank\').reindex(plot_order)\\n        best_model = str(sub[\'per_block_rmse_mean\'].idxmin())\\n        for j, mdl in enumerate(plot_order):\\n            xpos = x[bi] + (j - 1) * bar_width\\n            mean = float(sub.loc[mdl, \'per_block_rmse_mean\'])\\n            sd = float(sub.loc[mdl, \'per_block_rmse_sd\'])\\n            ax.bar(\\n                xpos,\\n                mean,\\n                width=bar_width,\\n                color=COLOR_MAP[mdl],\\n                edgecolor=\'white\',\\n                linewidth=0.8,\\n                yerr=sd,\\n                capsize=3,\\n                label=model_labels[mdl] if bi == 0 else None,\\n            )\\n\\n    panel_top = 0.0\\n    for bi, blk in enumerate(BLOCKS):\\n        sub = summary_df[summary_df[\'block\'] == blk].set_index(\'model_bank\').reindex(plot_order)\\n        means = sub[\'per_block_rmse_mean\'].to_numpy(float)\\n        sds = sub[\'per_block_rmse_sd\'].to_numpy(float)\\n        best_model = str(sub[\'per_block_rmse_mean\'].idxmin())\\n        best_idx = plot_order.index(best_model)\\n        panel_top = max(panel_top, float(np.nanmax(means + sds)))\\n        for mdl in plot_order:\\n            if mdl == best_model:\\n                continue\\n            st = stats_df[(stats_df[\'block\'] == blk) & (stats_df[\'model_bank\'] == mdl)]\\n            if st.empty:\\n                continue\\n            stars = str(st.iloc[0][\'stars\'])\\n            if not stars:\\n                continue\\n            i = plot_order.index(mdl)\\n            x_best = x[bi] + (best_idx - 1) * bar_width\\n            x_i = x[bi] + (i - 1) * bar_width\\n            y = float(max(means[best_idx] + sds[best_idx], means[i] + sds[i])) + 0.008 + 0.018 * abs(i - best_idx)\\n            _add_sig_bracket(ax, min(x_best, x_i), max(x_best, x_i), y, stars)\\n            panel_top = max(panel_top, y + 0.02)\\n\\n    ax.set_xticks(x)\\n    ax.set_xticklabels(plot_labels)\\n    ax.set_ylabel(\'Block Test RMSE\')\\n    ax.grid(True, axis=\'y\', alpha=0.25)\\n    ax.set_axisbelow(True)\\n    ax.set_ylim(0.0, panel_top + 0.01)\\n    fig.legend(loc=\'lower center\', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)\\n    fig.suptitle(title, y=0.98)\\n    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.92])\\n    plt.show()\\n    fig.savefig(OUT_DIR / f\'{out_prefix}.png\', bbox_inches=\'tight\')\\n    fig.savefig(OUT_DIR / f\'{out_prefix}.pdf\', bbox_inches=\'tight\')\\n    return fig\\n\\ndef _frozen_combined_bin_ablation_row(row):\\n    cv = _parse_float_list(row.get(\'cv_json\', None))\\n    y = _parse_float_list(row.get(\'p_obs_json\', None))\\n    if cv.size == 0 or y.size != cv.size:\\n        return None\\n\\n    def _vec(key, default_zero=False):\\n        arr = _parse_float_list(row.get(key, None))\\n        if arr.size == cv.size:\\n            return arr.astype(float)\\n        if default_zero:\\n            return np.zeros(cv.size, dtype=float)\\n        return None\\n\\n    rate = _vec(\'rate_drive_test_json\', default_zero=True)\\n    timing = _vec(\'timing_drive_test_json\', default_zero=True)\\n    rate_aux = _vec(\'rate_drive_aux_test_json\', default_zero=True)\\n    timing_aux = _vec(\'timing_drive_aux_test_json\', default_zero=True)\\n\\n    bias = float(row.get(\'bias\', 0.0)) if np.isfinite(row.get(\'bias\', np.nan)) else 0.0\\n    temp = float(row.get(\'temp\', 1.0)) if np.isfinite(row.get(\'temp\', np.nan)) else 1.0\\n    gr = float(row.get(\'gain_rate\', 0.0)) if np.isfinite(row.get(\'gain_rate\', np.nan)) else 0.0\\n    gt = float(row.get(\'gain_timing\', 0.0)) if np.isfinite(row.get(\'gain_timing\', np.nan)) else 0.0\\n    gra = float(row.get(\'gain_rate_aux\', 0.0)) if np.isfinite(row.get(\'gain_rate_aux\', np.nan)) else 0.0\\n    gta = float(row.get(\'gain_timing_aux\', 0.0)) if np.isfinite(row.get(\'gain_timing_aux\', np.nan)) else 0.0\\n\\n    mb = str(row.get(\'model_bank\', \'\')).strip().lower()\\n    if mb == \'rate_combined\':\\n        drive_full = gr * rate + gra * rate_aux\\n        drive_10 = gr * rate\\n        drive_1 = gra * rate_aux\\n    elif mb == \'timing_combined\':\\n        drive_full = gt * timing + gta * timing_aux\\n        drive_10 = gt * timing\\n        drive_1 = gta * timing_aux\\n    elif mb == \'hybrid_combined\':\\n        drive_full = gr * rate + gt * timing + gra * rate_aux + gta * timing_aux\\n        drive_10 = gr * rate + gt * timing\\n        drive_1 = gra * rate_aux + gta * timing_aux\\n    else:\\n        return None\\n\\n    blocks = np.asarray([str(row.get(\'block\', \'\'))] * cv.size, dtype=object)\\n\\n    def _predict_rmse(drive):\\n        z = bias + temp * np.asarray(drive, dtype=float)\\n        p_choose = sigmoid(z)\\n        p_corr = np.asarray(map_pchoose_to_pcorrect_mixed(p_choose, blocks=blocks, cvs=cv), dtype=float)\\n        return float(np.sqrt(np.nanmean((p_corr - y) ** 2)))\\n\\n    return {\\n        \'rmse_full\': _predict_rmse(drive_full),\\n        \'rmse_10ms_only\': _predict_rmse(drive_10),\\n        \'rmse_1ms_only\': _predict_rmse(drive_1),\\n    }\\n\\ndef _build_frozen_bin_ablation_long(base_df):\\n    rows = []\\n    model_labels = {\\n        \'rate_combined\': \'Rate\',\\n        \'timing_combined\': \'Timing\',\\n        \'hybrid_combined\': \'Hybrid\',\\n    }\\n    scoped = base_df[(base_df[\'scope\'] == \'per_block\') & (base_df[\'model_bank\'].isin(MODELS)) & (base_df[\'drop_thresh\'] == \'NONE\') & (~base_df[\'is_loto\'])].copy()\\n    scoped = scoped[scoped[\'block\'].isin(BLOCKS) & scoped[\'freq\'].isin(FREQS)].copy()\\n    for _, r in scoped.iterrows():\\n        rec = _frozen_combined_bin_ablation_row(r)\\n        if rec is None:\\n            continue\\n        rows.append({\\n            \'block\': str(r[\'block\']),\\n            \'model\': model_labels[str(r[\'model_bank\'])],\\n            \'freq\': float(r[\'freq\']),\\n            \'mc\': int(r[\'mc\']),\\n            **rec,\\n        })\\n    return pd.DataFrame(rows)\\n\\ndef _build_frozen_bin_ablation_summary(ablation_long_df):\\n    rows = []\\n    for blk in BLOCKS:\\n        for mdl in [\'Rate\', \'Timing\', \'Hybrid\']:\\n            sub = ablation_long_df[(ablation_long_df[\'block\'] == blk) & (ablation_long_df[\'model\'] == mdl)]\\n            mean_full, sd_full = _rmse_mean_sd(sub[\'rmse_full\'])\\n            mean_10, sd_10 = _rmse_mean_sd(sub[\'rmse_10ms_only\'])\\n            mean_1, sd_1 = _rmse_mean_sd(sub[\'rmse_1ms_only\'])\\n            rows.append({\\n                \'block\': blk,\\n                \'model\': mdl,\\n                \'rmse_full_mean\': mean_full,\\n                \'rmse_full_sd\': sd_full,\\n                \'rmse_10ms_only_mean\': mean_10,\\n                \'rmse_10ms_only_sd\': sd_10,\\n                \'rmse_1ms_only_mean\': mean_1,\\n                \'rmse_1ms_only_sd\': sd_1,\\n                \'n_rows\': int(len(sub)),\\n            })\\n    return pd.DataFrame(rows)\\n\\ndef _build_frozen_bin_ablation_stats(ablation_long_df):\\n    mc_level = (\\n        ablation_long_df\\n        .groupby([\'block\', \'model\', \'mc\'], as_index=False)\\n        .agg(\\n            rmse_full=(\'rmse_full\', \'mean\'),\\n            rmse_10ms_only=(\'rmse_10ms_only\', \'mean\'),\\n            rmse_1ms_only=(\'rmse_1ms_only\', \'mean\'),\\n        )\\n    )\\n    rows = []\\n    for blk in BLOCKS:\\n        for mdl in [\'Rate\', \'Timing\', \'Hybrid\']:\\n            sub = mc_level[(mc_level[\'block\'] == blk) & (mc_level[\'model\'] == mdl)].sort_values(\'mc\')\\n            p_10 = _paired_wilcoxon_greater(sub[\'rmse_10ms_only\'], sub[\'rmse_full\'])\\n            p_1 = _paired_wilcoxon_greater(sub[\'rmse_1ms_only\'], sub[\'rmse_full\'])\\n            rows.append({\\n                \'block\': blk,\\n                \'model\': mdl,\\n                \'contrast\': \'10 ms only vs full\',\\n                \'n_mc\': int(len(sub)),\\n                \'delta_rmse_mean\': float(np.nanmean(np.asarray(sub[\'rmse_10ms_only\'], dtype=float) - np.asarray(sub[\'rmse_full\'], dtype=float))),\\n                \'p_raw\': p_10,\\n            })\\n            rows.append({\\n                \'block\': blk,\\n                \'model\': mdl,\\n                \'contrast\': \'1 ms only vs full\',\\n                \'n_mc\': int(len(sub)),\\n                \'delta_rmse_mean\': float(np.nanmean(np.asarray(sub[\'rmse_1ms_only\'], dtype=float) - np.asarray(sub[\'rmse_full\'], dtype=float))),\\n                \'p_raw\': p_1,\\n            })\\n    out = pd.DataFrame(rows)\\n    out[\'p_holm\'] = _holm_adjust(out[\'p_raw\'].to_numpy(float))\\n    out[\'stars\'] = [_p_to_stars(p) for p in out[\'p_holm\']]\\n    return out\\n\\n", "9": "# Load raw merged outputs (full-mode source)\\nruns = pd.read_csv(RESULTS_DIR / \'merged_results.csv\')\\ncurves = pd.read_csv(RESULTS_DIR / \'merged_curves.csv\')\\n\\n# Manuscript-pruned tables omit exploratory run-index columns; add locked defaults.\\nfor _df in (runs, curves):\\n    if \'scope\' not in _df.columns:\\n        _df[\'scope\'] = \'per_block\'\\n    if \'drop_thresh\' not in _df.columns:\\n        _df[\'drop_thresh\'] = \'NONE\'\\n    if \'is_loto\' not in _df.columns:\\n        _df[\'is_loto\'] = False\\n\\nruns[\'model_bank\'] = runs[\'model_bank\'].astype(str).str.lower()\\ncurves[\'model_bank\'] = curves[\'model_bank\'].astype(str).str.lower()\\nruns[\'drop_thresh\'] = runs[\'drop_thresh\'].astype(str)\\ncurves[\'drop_thresh\'] = curves[\'drop_thresh\'].astype(str)\\nruns[\'block\'] = runs[\'block\'].astype(str)\\ncurves[\'block\'] = curves[\'block\'].astype(str)\\nruns[\'scope\'] = runs[\'scope\'].astype(str)\\n\\nfor c in [\'freq\', \'bin_ms\', \'aux_bin_ms\', \'mc\', \'rmse_test\', \'gain_rate\', \'gain_timing\', \'gain_rate_aux\', \'gain_timing_aux\']:\\n    if c in runs.columns:\\n        runs[c] = pd.to_numeric(runs[c], errors=\'coerce\')\\nfor c in [\'freq\', \'bin_ms\', \'aux_bin_ms\', \'mc\']:\\n    if c in curves.columns:\\n        curves[c] = pd.to_numeric(curves[c], errors=\'coerce\')\\n\\nif \'is_loto\' in runs.columns:\\n    runs[\'is_loto\'] = runs[\'is_loto\'].astype(bool)\\nif \'is_loto\' in curves.columns:\\n    curves[\'is_loto\'] = curves[\'is_loto\'].astype(bool)\\n\\n# Load Slack summary directly (for real p_sem error bars)\\nslack_mean = None\\nif DEFAULT_SLACK_DIR.exists():\\n    try:\\n        _slack_perrat, slack_mean = load_slack_dir(str(DEFAULT_SLACK_DIR), include_templates=True, keep_only_varying_type=True)\\n        slack_mean[\'freq\'] = pd.to_numeric(slack_mean[\'freq\'], errors=\'coerce\')\\n        slack_mean[\'cv\'] = pd.to_numeric(slack_mean[\'cv\'], errors=\'coerce\')\\n        slack_mean[\'block\'] = slack_mean[\'block\'].astype(str)\\n        print(f\'Loaded Slack SEM from: {DEFAULT_SLACK_DIR}\')\\n    except Exception as e:\\n        print(f\'[warn] Failed loading Slack SEM from {DEFAULT_SLACK_DIR}: {e}\')\\n        slack_mean = None\\nelse:\\n    print(f\'[warn] Slack directory not found: {DEFAULT_SLACK_DIR}; falling back to p_obs_json without SEM.\')\\n\\n# Merge curve JSON payloads onto run rows by unique identifiers\\nmerge_keys = [c for c in [\'partial_path\', \'model_bank\', \'freq\', \'block\', \'bin_ms\', \'mc\', \'drop_thresh\', \'is_loto\'] if c in runs.columns and c in curves.columns]\\ncurve_cols = [\\n    \'cv_json\', \'p_obs_json\', \'p_pred_test_json\',\\n    \'rate_drive_test_json\', \'timing_drive_test_json\',\\n    \'rate_drive_aux_test_json\', \'timing_drive_aux_test_json\',\\n]\\ndf = runs.merge(curves[merge_keys + curve_cols], on=merge_keys, how=\'left\', validate=\'one_to_one\')\\n\\n# Locked manuscript filter\\nbase = df[(df[\'scope\'] == \'per_block\') & (df[\'model_bank\'].isin(MODELS)) & (df[\'drop_thresh\'] == \'NONE\') & (~df[\'is_loto\'])].copy()\\nbase = base[base[\'freq\'].isin(FREQS) & base[\'block\'].isin(BLOCKS)].copy()\\n\\n# Per-block oracle winners (lower rmse is better)\\nper_block_score = (\\n    base.groupby([\'block\', \'model_bank\'], as_index=False)\\n    .agg(per_block_rmse_mean=(\'rmse_test\', \'mean\'), per_block_rmse_sem=(\'rmse_test\', _sem), n_rows=(\'rmse_test\', \'size\'))\\n)\\nper_block_score[\'rank_per_block\'] = per_block_score.groupby(\'block\')[\'per_block_rmse_mean\'].rank(method=\'dense\')\\nwinner_by_block = {b: g.sort_values(\'per_block_rmse_mean\').iloc[0][\'model_bank\'] for b, g in per_block_score.groupby(\'block\')}\\n\\ndisplay(per_block_score.sort_values([\'block\', \'per_block_rmse_mean\']).reset_index(drop=True))\\nprint(\'winner_by_block:\', winner_by_block)\\n\\n", "10": "THRESH_REGIME_TICKS = {\\n    \'I_first\': \'first\',\\n    \'I_volley5\': \'volley5\',\\n    \'I_volley10\': \'volley10\',\\n    \'I_5\': \'5%\',\\n    \'I_10\': \'10%\',\\n    \'I_15\': \'15%\',\\n    \'I_25\': \'25%\',\\n    \'I_50\': \'50%\',\\n    \'I_max\': \'max\',\\n}\\n\\n# 2x4 block-shared regime-weight diagnostic removed; keep the single pipeline-overview example below.\\nx = np.arange(len(THRESH_ORDER), dtype=float)\\n\\n# Figure 11 helper: single example panel for pipeline overview\\nEXAMPLE_BLOCK = \'std1\'\\nEXAMPLE_FREQ = 100.0\\nEXAMPLE_MODEL = \'hybrid_combined\'\\n\\nfig_ex, ax_ex = plt.subplots(1, 1, figsize=(6.2, 4.2))\\nsub_ex = base[(base[\'block\'] == EXAMPLE_BLOCK) & np.isclose(base[\'freq\'], EXAMPLE_FREQ) & (base[\'model_bank\'] == EXAMPLE_MODEL)]\\nif sub_ex.empty:\\n    raise RuntimeError(f\'No rows found for {EXAMPLE_BLOCK=} {EXAMPLE_FREQ=} {EXAMPLE_MODEL=}\')\\nvals_ex = []\\nfor _, r in sub_ex.iterrows():\\n    wd = _parse_num_dict(r.get(\'weights_json\', {}))\\n    vals_ex.append([float(wd.get(t, np.nan)) for t in THRESH_ORDER])\\nY_ex = np.asarray(vals_ex, dtype=float)\\ny_ex = np.nanmean(Y_ex, axis=0)\\nse_ex = np.nanstd(Y_ex, axis=0, ddof=1) / np.sqrt(Y_ex.shape[0]) if Y_ex.shape[0] > 1 else np.zeros(Y_ex.shape[1])\\nax_ex.plot(x, y_ex, color=COLOR_MAP[EXAMPLE_MODEL], linewidth=2.6, alpha=0.95, marker=\'o\', markersize=4)\\nax_ex.fill_between(x, y_ex - se_ex, y_ex + se_ex, color=COLOR_MAP[EXAMPLE_MODEL], alpha=0.12)\\nax_ex.set_title(f\'{EXAMPLE_BLOCK} | {int(EXAMPLE_FREQ)} Hz\')\\nax_ex.grid(True, alpha=0.25)\\nax_ex.set_xticks(x)\\nax_ex.set_xticklabels([THRESH_REGIME_TICKS[t] for t in THRESH_ORDER], rotation=30, ha=\'right\')\\nax_ex.set_xlabel(\'Spike-Count Regime\')\\nax_ex.set_ylabel(\'Weight\')\\nfig_ex.suptitle(\'Block-Shared Regime Weights\', y=1.02)\\nplt.tight_layout(rect=[0, 0, 1, 0.96])\\nplt.show()\\nfig_ex.savefig(OUT_DIR / \'fig01_block_shared_regime_weights_std1_100Hz_hybrid.png\', bbox_inches=\'tight\')\\nfig_ex.savefig(OUT_DIR / \'fig01_block_shared_regime_weights_std1_100Hz_hybrid.pdf\', bbox_inches=\'tight\')", "11": "# Supplementary Figure 1: Behavioral psychometric curves\\nimport os, sys\\nfrom pathlib import Path\\nimport numpy as np\\nimport pandas as pd\\nimport matplotlib.pyplot as plt\\n\\nSRC_DISC = (REPO_ROOT / \'src\' / \'discrimination\').resolve()\\nif SRC_DISC.exists() and str(SRC_DISC) not in sys.path:\\n    sys.path.insert(0, str(SRC_DISC))\\n\\nfrom discrimination_core import infer_block_from_file, parse_rat_name_from_filename\\n\\nSLACK_DIR = REPO_ROOT / \'data\' / \'discrimination\' / \'slack\'\\nif not SLACK_DIR.exists():\\n    raise FileNotFoundError(f\'Slack directory not found: {SLACK_DIR}\')\\n\\nrows = []\\nfor fp in sorted(SLACK_DIR.glob(\'*.csv\')):\\n    df = pd.read_csv(fp, low_memory=False)\\n\\n    if \'Randomized\' in df.columns:\\n        df = df[df[\'Randomized\'] == 1]\\n    if \'Forced\' in df.columns:\\n        df = df[df[\'Forced\'] == 0]\\n\\n    if \'Response\' not in df.columns:\\n        continue\\n    df[\'Response\'] = pd.to_numeric(df[\'Response\'], errors=\'coerce\')\\n    df = df[df[\'Response\'].isin([1, 2])]\\n\\n    if \'CV\' in df.columns:\\n        df[\'cv\'] = pd.to_numeric(df[\'CV\'], errors=\'coerce\')\\n    elif \'cv\' in df.columns:\\n        df[\'cv\'] = pd.to_numeric(df[\'cv\'], errors=\'coerce\')\\n    else:\\n        continue\\n\\n    if \'Freq\' in df.columns:\\n        df[\'freq\'] = pd.to_numeric(df[\'Freq\'], errors=\'coerce\')\\n    elif \'Frequency (Hz)\' in df.columns:\\n        df[\'freq\'] = pd.to_numeric(df[\'Frequency (Hz)\'], errors=\'coerce\')\\n    elif \'freq\' in df.columns:\\n        df[\'freq\'] = pd.to_numeric(df[\'freq\'], errors=\'coerce\')\\n    else:\\n        continue\\n\\n    if \'Type\' in df.columns:\\n        df[\'Type\'] = pd.to_numeric(df[\'Type\'], errors=\'coerce\')\\n\\n    df = df.dropna(subset=[\'cv\', \'freq\', \'Response\'])\\n    if df.empty:\\n        continue\\n\\n    block = infer_block_from_file(df)\\n    if block is None:\\n        continue\\n\\n    rat_name = parse_rat_name_from_filename(str(fp))\\n    df[\'cv\'] = np.round(df[\'cv\'].astype(float), 1)\\n\\n    if block == \'std0\':\\n        is_correct = np.where(np.isclose(df[\'cv\'], 0.0), (df[\'Response\'] == 2), (df[\'Response\'] == 1))\\n    else:\\n        is_correct = np.where(np.isclose(df[\'cv\'], 1.0), (df[\'Response\'] == 1), (df[\'Response\'] == 2))\\n\\n    tmp = pd.DataFrame({\\n        \'rat_name\': rat_name,\\n        \'freq\': df[\'freq\'].astype(float),\\n        \'block\': block,\\n        \'cv\': df[\'cv\'].astype(float),\\n        \'correct\': is_correct.astype(int),\\n    })\\n    rows.append(tmp)\\n\\nif not rows:\\n    raise ValueError(\'No usable Slack trials found after filters.\')\\n\\ntrials = pd.concat(rows, ignore_index=True)\\nperrat = (trials.groupby([\'rat_name\', \'freq\', \'block\', \'cv\'], as_index=False)\\n          .agg(k=(\'correct\', \'sum\'), n=(\'correct\', \'size\')))\\nperrat[\'p_correct\'] = perrat[\'k\'] / perrat[\'n\']\\nperrat[\'p_trend\'] = perrat[\'p_correct\']\\nm0 = (perrat[\'block\'] == \'std0\') & np.isclose(perrat[\'cv\'], 0.0)\\nm1 = (perrat[\'block\'] == \'std1\') & np.isclose(perrat[\'cv\'], 1.0)\\nperrat.loc[m0 | m1, \'p_trend\'] = 1.0 - perrat.loc[m0 | m1, \'p_correct\']\\n\\nsummary = (perrat.groupby([\'freq\', \'block\', \'cv\'], as_index=False)\\n           .agg(p_mean=(\'p_trend\', \'mean\'),\\n                p_sem=(\'p_trend\', lambda x: np.nanstd(np.asarray(x, float), ddof=1) / np.sqrt(max(len(x), 1)) if len(x) > 1 else 0.0),\\n                n_rats=(\'rat_name\', \'nunique\'),\\n                n_points=(\'p_trend\', \'size\'))\\n           .sort_values([\'block\', \'freq\', \'cv\']))\\n\\nfreqs = sorted(summary[\'freq\'].unique().tolist())\\nblocks = [\'std0\', \'std1\']\\nfig, axes = plt.subplots(len(blocks), len(freqs), figsize=(4.0 * len(freqs), 3.4 * len(blocks)), sharey=True)\\nif len(freqs) == 1:\\n    axes = np.array(axes).reshape(len(blocks), 1)\\n\\nfor ci, f in enumerate(freqs):\\n    for ri, b in enumerate(blocks):\\n        ax = axes[ri, ci]\\n        pr = perrat[(perrat[\'freq\'] == f) & (perrat[\'block\'] == b)].copy()\\n        sm = summary[(summary[\'freq\'] == f) & (summary[\'block\'] == b)].copy()\\n\\n        if pr.empty:\\n            ax.set_title(f\'{b} | {int(round(float(f)))} Hz (no data)\')\\n            ax.set_xlabel(\'Coefficient of Variation (CV)\')\\n            if ci == 0:\\n                ax.set_ylabel(\'P(CV1)\' if b == \'std0\' else \'P(CV0)\')\\n            ax.grid(True, alpha=0.25)\\n            continue\\n\\n        for _, g in pr.groupby(\'rat_name\'):\\n            g = g.sort_values(\'cv\')\\n            ax.plot(g[\'cv\'], g[\'p_trend\'], color=\'0.75\', linewidth=1.0, alpha=0.7)\\n\\n        sm = sm.sort_values(\'cv\')\\n        ax.errorbar(sm[\'cv\'], sm[\'p_mean\'], yerr=sm[\'p_sem\'], marker=\'o\', linewidth=2.0, color=\'black\', capsize=2)\\n\\n        ax.set_title(f\'{b} | {int(round(float(f)))} Hz\')\\n        ax.set_xlim(-0.02, 1.02)\\n        ax.set_ylim(-0.02, 1.02)\\n        ax.set_xlabel(\'Coefficient of Variation (CV)\')\\n        if ci == 0:\\n            ax.set_ylabel(\'P(CV1)\' if b == \'std0\' else \'P(CV0)\')\\n        ax.grid(True, alpha=0.25)\\n\\nfig.suptitle(\'Behavioral Psychometric Curves\', y=1.02)\\nfig.tight_layout()\\nplt.show()\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'suppfig01_behavioral_psychometric_curves.png\', bbox_inches=\'tight\')\\nfig.savefig(MANUSCRIPT_OUT_DIR / \'suppfig01_behavioral_psychometric_curves.pdf\', bbox_inches=\'tight\')\\n", "12": "# Table: per_block-only RMSE summary (clean manuscript version)\\n\\nruns_tbl = runs[(runs[\'model_bank\'].isin(MODELS)) & (runs[\'drop_thresh\'] == \'NONE\') & (~runs[\'is_loto\'])].copy()\\nruns_tbl = runs_tbl[runs_tbl[\'freq\'].isin(FREQS) & runs_tbl[\'block\'].isin(BLOCKS)].copy()\\n\\ndef _sd(x):\\n    a = np.asarray(x, dtype=float)\\n    a = a[np.isfinite(a)]\\n    if a.size <= 1:\\n        return 0.0\\n    return float(np.std(a, ddof=1))\\n\\n# Lock strictly to per_block scope\\npb = runs_tbl[runs_tbl[\'scope\'] == \'per_block\'].copy()\\n\\n# Build the aligned model-performance summary using 200 MC block means.\\npb_block_mcmean, model_block_stats_mcmean, display_table_mcmean, pb_unit_mcmean = _build_block_rmse_summary(pb, unit=\'mc_block_mean\')\\n\\nprint(\'Model performance table using 200 MC block means:\')\\ndisplay(display_table_mcmean)\\nmodel_block_stats_mcmean_display = model_block_stats_mcmean.copy()\\nmodel_block_stats_mcmean_display[\'p_raw\'] = model_block_stats_mcmean_display[\'p_raw\'].map(_format_p)\\nmodel_block_stats_mcmean_display[\'p_holm\'] = model_block_stats_mcmean_display[\'p_holm\'].map(_format_p)\\ndisplay(model_block_stats_mcmean_display)\\n\\n# Frequency/block-level performance outputs for reference.\\npb_fb = (\\n    pb.groupby([\'block\', \'freq\', \'model_bank\'], as_index=False)\\n    .agg(\\n        fb_rmse_mean=(\'rmse_test\', \'mean\'),\\n        fb_rmse_sd=(\'rmse_test\', _sd),\\n        n_mc_per_freq_block=(\'rmse_test\', \'size\'),\\n    )\\n)\\ntable_mcmean = pb_fb.merge(pb_block_mcmean, on=[\'block\', \'model_bank\'], how=\'left\')\\ntable_mcmean = table_mcmean.sort_values([\'block\', \'freq\', \'rank_per_block\', \'fb_rmse_mean\']).reset_index(drop=True)\\n\\n# Outputs\\ndisplay_table_mcmean.to_csv(OUT_DIR / \'table_per_block_rmse_summary_display_mc_block_mean.csv\', index=False)\\nmodel_block_stats_mcmean.to_csv(OUT_DIR / \'table_per_block_rmse_summary_stats_mc_block_mean.csv\', index=False)\\ntable_mcmean.to_csv(OUT_DIR / \'table_per_block_only_rmse_by_freq_block_mc_block_mean.csv\', index=False)\\nprint(\'Wrote:\', OUT_DIR / \'table_per_block_rmse_summary_display_mc_block_mean.csv\')\\nprint(\'Wrote:\', OUT_DIR / \'table_per_block_rmse_summary_stats_mc_block_mean.csv\')\\nprint(\'Wrote:\', OUT_DIR / \'table_per_block_only_rmse_by_freq_block_mc_block_mean.csv\')\\nprint(\'All table metrics are computed from scope=per_block only.\')\\nprint(\'Dispersion columns report SD (not SEM). This version uses 200 paired block means (one per MC run), averaged across the four manuscript frequencies within each block.\')\\n\\n", "13": "# Exploratory Figure 10: per-block model curves + per-point hybrid test predictions\\nDISC_PER_BLOCK_DIR = REPO_ROOT / \'data\' / \'discrimination\' / \'manuscript\' / \'per-block\'\\nDISC_PERCV_DIR = REPO_ROOT / \'data\' / \'discrimination\' / \'manuscript\' / \'per-point\'\\n\\nruns_pb = pd.read_csv(DISC_PER_BLOCK_DIR / \'merged_results.csv\')\\ncurves_pb = pd.read_csv(DISC_PER_BLOCK_DIR / \'merged_curves.csv\')\\nruns_pcv = pd.read_csv(DISC_PERCV_DIR / \'merged_results.csv\')\\ncurves_pcv = pd.read_csv(DISC_PERCV_DIR / \'merged_curves.csv\')\\n\\n# Manuscript-pruned tables are already locked to the retained analyses.\\nfor _df in (runs_pb, curves_pb):\\n    if \'scope\' not in _df.columns:\\n        _df[\'scope\'] = \'per_block\'\\n    if \'drop_thresh\' not in _df.columns:\\n        _df[\'drop_thresh\'] = \'NONE\'\\n    if \'is_loto\' not in _df.columns:\\n        _df[\'is_loto\'] = False\\nfor _df in (runs_pcv, curves_pcv):\\n    if \'scope\' not in _df.columns:\\n        _df[\'scope\'] = \'per_freq_block\'\\n    if \'drop_thresh\' not in _df.columns:\\n        _df[\'drop_thresh\'] = \'NONE\'\\n    if \'is_loto\' not in _df.columns:\\n        _df[\'is_loto\'] = False\\n    if \'thresh_flex_mode\' not in _df.columns:\\n        _df[\'thresh_flex_mode\'] = \'per_point\'\\n\\nfor _df in [runs_pb, curves_pb, runs_pcv, curves_pcv]:\\n    if \'model_bank\' in _df.columns:\\n        _df[\'model_bank\'] = _df[\'model_bank\'].astype(str).str.lower()\\n    if \'drop_thresh\' in _df.columns:\\n        _df[\'drop_thresh\'] = _df[\'drop_thresh\'].astype(str)\\n    if \'block\' in _df.columns:\\n        _df[\'block\'] = _df[\'block\'].astype(str)\\n    if \'scope\' in _df.columns:\\n        _df[\'scope\'] = _df[\'scope\'].astype(str)\\n    if \'shared_thresh_mode\' in _df.columns:\\n        _df[\'shared_thresh_mode\'] = _df[\'shared_thresh_mode\'].astype(str)\\n    if \'thresh_flex_mode\' in _df.columns:\\n        _df[\'thresh_flex_mode\'] = _df[\'thresh_flex_mode\'].astype(str)\\n    for _c in [\'freq\', \'bin_ms\', \'aux_bin_ms\', \'mc\', \'rmse_test\', \'weight_smooth_lambda\']:\\n        if _c in _df.columns:\\n            _df[_c] = pd.to_numeric(_df[_c], errors=\'coerce\')\\n    for _c in [\'is_loto\', \'is_fixed_thresh\']:\\n        if _c in _df.columns:\\n            _df[_c] = _df[_c].astype(bool)\\n\\nmerge_keys_pb = [c for c in [\'partial_path\', \'model_bank\', \'freq\', \'block\', \'bin_ms\', \'mc\', \'drop_thresh\', \'is_loto\'] if c in runs_pb.columns and c in curves_pb.columns]\\ncurve_cols_pb = [\\n    \'cv_json\', \'p_obs_json\', \'p_pred_test_json\',\\n    \'rate_drive_test_json\', \'timing_drive_test_json\',\\n    \'rate_drive_aux_test_json\', \'timing_drive_aux_test_json\',\\n]\\ndf_pb = runs_pb.merge(curves_pb[merge_keys_pb + curve_cols_pb], on=merge_keys_pb, how=\'left\', validate=\'one_to_one\')\\n\\npb_base = df_pb[\\n    (df_pb[\'scope\'] == \'per_block\')\\n    & (df_pb[\'model_bank\'].isin(MODELS))\\n    & (df_pb[\'drop_thresh\'] == \'NONE\')\\n    & (~df_pb[\'is_loto\'])\\n    & (df_pb[\'freq\'].isin(FREQS))\\n    & (df_pb[\'block\'].isin(BLOCKS))\\n].copy()\\n\\npb_scores_exp = (\\n    pb_base.groupby([\'block\', \'model_bank\'], as_index=False)\\n    .agg(per_block_rmse_mean=(\'rmse_test\', \'mean\'), per_block_rmse_sem=(\'rmse_test\', _sem), n_rows=(\'rmse_test\', \'size\'))\\n)\\nwinner_by_block_exp = {\\n    blk: grp.sort_values(\'per_block_rmse_mean\').iloc[0][\'model_bank\']\\n    for blk, grp in pb_scores_exp.groupby(\'block\')\\n}\\n\\npcv_hybrid_curves = curves_pcv[\\n    (curves_pcv[\'scope\'] == \'per_freq_block\')\\n    & (curves_pcv[\'model_bank\'] == \'hybrid_combined\')\\n    & (curves_pcv[\'drop_thresh\'] == \'NONE\')\\n    & (~curves_pcv[\'is_loto\'])\\n    & (np.isclose(curves_pcv[\'bin_ms\'], 10.0))\\n    & (curves_pcv[\'freq\'].isin(FREQS))\\n    & (curves_pcv[\'block\'].isin(BLOCKS))\\n    & (curves_pcv[\'thresh_flex_mode\'].astype(str).str.lower() == \'per_point\')\\n].copy()\\n\\npcv_hybrid_runs = runs_pcv[\\n    (runs_pcv[\'scope\'] == \'per_freq_block\')\\n    & (runs_pcv[\'model_bank\'] == \'hybrid_combined\')\\n    & (runs_pcv[\'drop_thresh\'] == \'NONE\')\\n    & (~runs_pcv[\'is_loto\'])\\n    & (np.isclose(runs_pcv[\'bin_ms\'], 10.0))\\n    & (runs_pcv[\'freq\'].isin(FREQS))\\n    & (runs_pcv[\'block\'].isin(BLOCKS))\\n    & (runs_pcv[\'thresh_flex_mode\'].astype(str).str.lower() == \'per_point\')\\n].copy()\\n\\npcv_panel_rmse = (\\n    pcv_hybrid_runs.groupby([\'block\', \'freq\'], as_index=False)\\n    .agg(rmse_mean=(\'rmse_test\', \'mean\'), rmse_sem=(\'rmse_test\', _sem), n_mc=(\'rmse_test\', \'size\'))\\n    .sort_values([\'block\', \'freq\'])\\n    .reset_index(drop=True)\\n)\\npcv_block_rmse = (\\n    pcv_hybrid_runs.groupby([\'block\'], as_index=False)\\n    .agg(rmse_mean=(\'rmse_test\', \'mean\'), rmse_sem=(\'rmse_test\', _sem), n_mc=(\'rmse_test\', \'size\'))\\n    .sort_values([\'block\'])\\n    .reset_index(drop=True)\\n)\\n\\nprint(\'Exploratory per-point hybrid RMSE by freq/block:\')\\ndisplay(pcv_panel_rmse)\\nprint(\'Exploratory per-point hybrid mean RMSE by block:\')\\ndisplay(pcv_block_rmse)\\n\\nfig_exp10, axes_exp10 = plt.subplots(2, 4, figsize=(16, 7), sharey=True)\\norange = \'#f4a261\'\\nxticks_dense = np.round(np.arange(0.0, 1.01, 0.1), 1)\\nxticklabels_sparse = [f\'{x:.1f}\' if (int(round(x * 10)) % 2 == 0) else \'\' for x in xticks_dense]\\n\\nfor ri, blk in enumerate(BLOCKS):\\n    for ci, f in enumerate(FREQS):\\n        ax = axes_exp10[ri, ci]\\n\\n        obs_done = False\\n        if isinstance(slack_mean, pd.DataFrame) and (not slack_mean.empty):\\n            so = slack_mean[(slack_mean[\'block\'] == blk) & np.isclose(slack_mean[\'freq\'], f)].copy().sort_values(\'cv\')\\n            if not so.empty and {\'p_mean\', \'p_sem\'}.issubset(so.columns):\\n                x_obs = so[\'cv\'].to_numpy(float)\\n                y_obs = _to_choice_display(x_obs, so[\'p_mean\'].to_numpy(float), blk)\\n                se_obs = np.asarray(so[\'p_sem\'].to_numpy(float), dtype=float)\\n                se_obs = np.nan_to_num(se_obs, nan=0.0)\\n                ax.errorbar(\\n                    x_obs, y_obs, yerr=se_obs,\\n                    fmt=\'o\', color=\'black\', markersize=5, capsize=2, linewidth=1.2, zorder=6\\n                )\\n                obs_done = True\\n\\n        if not obs_done:\\n            wmdl = winner_by_block_exp.get(blk, MODELS[0])\\n            s_obs = pb_base[(pb_base[\'block\'] == blk) & np.isclose(pb_base[\'freq\'], f) & (pb_base[\'model_bank\'] == wmdl)]\\n            obs = _aggregate_curve_rows(s_obs)\\n            if not obs.empty and {\'obs_mean\', \'obs_sem\'}.issubset(obs.columns):\\n                y_obs = _to_choice_display(obs[\'cv\'].to_numpy(float), obs[\'obs_mean\'].to_numpy(float), blk)\\n                ax.errorbar(\\n                    obs[\'cv\'].to_numpy(float), y_obs, yerr=obs[\'obs_sem\'].to_numpy(float),\\n                    fmt=\'o\', color=\'black\', markersize=5, capsize=2, linewidth=1.2, zorder=6\\n                )\\n\\n        for mdl in MODELS:\\n            sub = pb_base[(pb_base[\'block\'] == blk) & np.isclose(pb_base[\'freq\'], f) & (pb_base[\'model_bank\'] == mdl)]\\n            cs = _aggregate_curve_rows(sub)\\n            if cs.empty:\\n                continue\\n            x = cs[\'cv\'].to_numpy(float)\\n            y = _to_choice_display(x, cs[\'pred_mean\'].to_numpy(float), blk)\\n            se = cs[\'pred_sem\'].to_numpy(float)\\n            is_winner = (mdl == winner_by_block_exp.get(blk))\\n            lw = 2.6 if is_winner else 1.3\\n            alpha = 0.95 if is_winner else 0.35\\n            ax.plot(x, y, color=COLOR_MAP[mdl], linewidth=lw, alpha=alpha, zorder=3)\\n            if is_winner:\\n                ax.fill_between(x, np.clip(y - se, 0, 1), np.clip(y + se, 0, 1), color=COLOR_MAP[mdl], alpha=0.16, zorder=2)\\n\\n        sub_pcv = pcv_hybrid_curves[(pcv_hybrid_curves[\'block\'] == blk) & np.isclose(pcv_hybrid_curves[\'freq\'], f)]\\n        cs_pcv = _aggregate_curve_rows(sub_pcv)\\n        if not cs_pcv.empty:\\n            x_pcv = cs_pcv[\'cv\'].to_numpy(float)\\n            y_pcv = _to_choice_display(x_pcv, cs_pcv[\'pred_mean\'].to_numpy(float), blk)\\n            se_pcv = cs_pcv[\'pred_sem\'].to_numpy(float)\\n            ax.errorbar(\\n                x_pcv, y_pcv, yerr=se_pcv,\\n                fmt=\'s\', linestyle=\'none\',\\n                color=orange, ecolor=orange,\\n                markersize=7.0, markeredgewidth=0.0,\\n                elinewidth=1.0, capsize=2,\\n                alpha=0.95, zorder=4\\n            )\\n\\n        ax.set_title(f\'{blk} | {int(f)} Hz\')\\n        ax.set_xlim(-0.03, 1.03)\\n        ax.set_ylim(-0.02, 1.02)\\n        ax.grid(True, alpha=0.25)\\n        ax.set_xticks(xticks_dense)\\n        ax.tick_params(axis=\'x\', labelbottom=(ri == 1))\\n        if ri == 1:\\n            ax.set_xticklabels(xticklabels_sparse)\\n            ax.set_xlabel(\'Coefficient of Variation (CV)\')\\n        else:\\n            ax.set_xticklabels([])\\n        if ci == 0:\\n            ax.set_ylabel(\'P(CV1)\' if blk == \'std0\' else \'P(CV0)\')\\n\\nhandles_exp10 = [\\n    plt.Line2D([0], [0], color=COLOR_MAP[\'rate_combined\'], lw=2, label=\'Rate\'),\\n    plt.Line2D([0], [0], color=COLOR_MAP[\'timing_combined\'], lw=2, label=\'Timing\'),\\n    plt.Line2D([0], [0], color=COLOR_MAP[\'hybrid_combined\'], lw=2, label=\'Hybrid\'),\\n    plt.Line2D([0], [0], color=orange, marker=\'s\', linestyle=\'none\', markersize=7.5, label=\'Per-Point Mixture\'),\\n]\\nfig_exp10.legend(handles=handles_exp10, loc=\'lower center\', bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False)\\nfig_exp10.suptitle(\'Behavioral Psychometric Curve Matching (Per-Block Curves + Per-Point Mixture Points)\', y=1.02)\\nplt.tight_layout(rect=[0, 0.06, 1, 0.98])\\nplt.show()\\n\\nfig_exp10.savefig(OUT_DIR / \'fig10_behavioral_psychometric_curve_matching.png\', bbox_inches=\'tight\')\\nfig_exp10.savefig(OUT_DIR / \'fig10_behavioral_psychometric_curve_matching.pdf\', bbox_inches=\'tight\')\\n", "14": "# Exploratory supplementary figure: per-point hybrid threshold mass by panel\\nregime_colors_panel = [plt.cm.viridis(v) for v in np.linspace(0.05, 0.98, len(THRESH_ORDER))]\\n\\npanel_weight_rows = []\\nfor _, r in pcv_hybrid_curves.iterrows():\\n    try:\\n        weight_map_by_cv = json.loads(str(r.get(\'weights_by_cv_json\', \'{}\')))\\n    except Exception:\\n        weight_map_by_cv = {}\\n    if not isinstance(weight_map_by_cv, dict):\\n        continue\\n    for cv_key, wd in weight_map_by_cv.items():\\n        try:\\n            cv_val = float(cv_key)\\n        except Exception:\\n            continue\\n        if not isinstance(wd, dict):\\n            continue\\n        row = {\\n            \'block\': str(r[\'block\']),\\n            \'freq\': float(r[\'freq\']),\\n            \'mc\': float(r[\'mc\']) if np.isfinite(r.get(\'mc\', np.nan)) else np.nan,\\n            \'cv\': float(cv_val),\\n        }\\n        for t in THRESH_ORDER:\\n            row[t] = float(_parse_num_dict(wd).get(t, 0.0))\\n        panel_weight_rows.append(row)\\n\\npanel_weight_df = pd.DataFrame(panel_weight_rows)\\nif panel_weight_df.empty:\\n    raise RuntimeError(\'No per-point weights found in data/discrimination/per-point/merged_curves.csv\')\\n\\npanel_weight_summary = (\\n    panel_weight_df.groupby([\'block\', \'freq\', \'cv\'], as_index=False)[THRESH_ORDER]\\n    .mean()\\n    .sort_values([\'block\', \'freq\', \'cv\'])\\n    .reset_index(drop=True)\\n)\\n\\n# Reuse the averaged per-point weights below in the combined figure only.\\n# Combined supplementary figure: block-shared + per-point regime weights\\nfig_combo = plt.figure(figsize=(20, 7.35), constrained_layout=False)\\nouter_gs = fig_combo.add_gridspec(\\n    1, 2,\\n    width_ratios=[0.84, 3.66],\\n    wspace=0.065,\\n)\\ngs_left = outer_gs[0, 0].subgridspec(2, 1, hspace=0.20)\\ngs_right = outer_gs[0, 1].subgridspec(2, 4, hspace=0.22, wspace=0.10)\\n\\ncombo_axes_left = [\\n    fig_combo.add_subplot(gs_left[0, 0]),\\n    fig_combo.add_subplot(gs_left[1, 0]),\\n]\\ncombo_axes_left[1].sharex(combo_axes_left[0])\\ncombo_axes_left[1].sharey(combo_axes_left[0])\\n\\ncombo_axes_right = []\\nsharey_right = None\\nfor ri in range(2):\\n    row_axes = []\\n    for ci in range(4):\\n        ax = fig_combo.add_subplot(gs_right[ri, ci], sharey=sharey_right)\\n        if sharey_right is None:\\n            sharey_right = ax\\n        row_axes.append(ax)\\n    combo_axes_right.append(row_axes)\\n\\nx_reg = np.arange(len(THRESH_ORDER), dtype=float)\\nmodel_order_combo = [\'rate_combined\', \'timing_combined\', \'hybrid_combined\']\\nxticks_cv = np.round(np.arange(0.0, 1.01, 0.1), 1)\\nxticklabels_cv_sparse = [f\'{v:.1f}\' if np.isclose((v * 10) % 2, 0) else \'\' for v in xticks_cv]\\n\\nleft_ymax = 0.0\\nfor blk in BLOCKS:\\n    for mdl in model_order_combo:\\n        sub = base[(base[\'block\'] == blk) & (base[\'model_bank\'] == mdl)].copy()\\n        dedup_cols = [c for c in [\'mc\', \'weights_json\'] if c in sub.columns]\\n        if dedup_cols:\\n            sub = sub.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols)\\n        vals = []\\n        for _, r in sub.iterrows():\\n            wd = _parse_num_dict(r.get(\'weights_json\', {}))\\n            vals.append([float(wd.get(t, np.nan)) for t in THRESH_ORDER])\\n        if vals:\\n            Y = np.asarray(vals, dtype=float)\\n            left_ymax = max(left_ymax, float(np.nanmax(Y)))\\nleft_ylim_top = max(0.36, np.ceil((left_ymax + 0.01) / 0.02) * 0.02)\\n\\nfor ri, blk in enumerate(BLOCKS):\\n    ax = combo_axes_left[ri]\\n    for mdl in model_order_combo:\\n        sub = base[(base[\'block\'] == blk) & (base[\'model_bank\'] == mdl)].copy()\\n        dedup_cols = [c for c in [\'mc\', \'weights_json\'] if c in sub.columns]\\n        if dedup_cols:\\n            sub = sub.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols)\\n        vals = []\\n        for _, r in sub.iterrows():\\n            wd = _parse_num_dict(r.get(\'weights_json\', {}))\\n            vals.append([float(wd.get(t, np.nan)) for t in THRESH_ORDER])\\n        if not vals:\\n            continue\\n        Y = np.asarray(vals, dtype=float)\\n        y = np.nanmean(Y, axis=0)\\n        se = np.nanstd(Y, axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros(Y.shape[1])\\n        is_winner = (mdl == winner_by_block.get(blk))\\n        lw = 2.7 if is_winner else 1.35\\n        alpha = 0.98 if is_winner else 0.38\\n        z = 4 if is_winner else 2\\n        ax.plot(\\n            x_reg, y,\\n            color=COLOR_MAP[mdl],\\n            linewidth=lw,\\n            alpha=alpha,\\n            marker=\'o\',\\n            markersize=4,\\n            zorder=z,\\n        )\\n        if is_winner:\\n            ax.fill_between(x_reg, y - se, y + se, color=COLOR_MAP[mdl], alpha=0.12, zorder=z - 1)\\n\\n    ax.set_title(f\'{blk} | Per-Block\')\\n    ax.grid(True, alpha=0.25)\\n    ax.set_xlim(-0.2, len(THRESH_ORDER) - 0.8)\\n    ax.set_ylim(0.0, left_ylim_top)\\n    ax.set_xticks(x_reg)\\n    if ri == 1:\\n        ax.set_xticklabels([THRESH_REGIME_TICKS[t] for t in THRESH_ORDER], rotation=30, ha=\'right\')\\n        ax.set_xlabel(\'Spike-Count Regime\')\\n    else:\\n        ax.tick_params(axis=\'x\', labelbottom=False)\\n    ax.set_ylabel(\'Weight\')\\n\\nfor ri, blk in enumerate(BLOCKS):\\n    for ci, f in enumerate(FREQS):\\n        ax = combo_axes_right[ri][ci]\\n        sub = panel_weight_summary[(panel_weight_summary[\'block\'] == blk) & np.isclose(panel_weight_summary[\'freq\'], f)].copy()\\n        sub = sub.sort_values(\'cv\')\\n        if sub.empty:\\n            ax.set_title(f\'{blk} | {int(f)} Hz (no data)\')\\n            ax.set_xlim(-0.02, 1.02)\\n            ax.set_ylim(0, 1)\\n            ax.grid(True, alpha=0.25)\\n            continue\\n\\n        x = sub[\'cv\'].to_numpy(float)\\n        y_stack = [sub[t].to_numpy(float) for t in THRESH_ORDER]\\n        ax.stackplot(x, *y_stack, colors=regime_colors_panel, alpha=0.92, linewidth=0.0)\\n        ax.set_title(f\'{blk} | {int(f)} Hz\')\\n        ax.set_xlim(0.0, 1.0)\\n        ax.set_ylim(0, 1)\\n        ax.set_xticks(xticks_cv)\\n        ax.grid(True, alpha=0.20)\\n        if ri == 1:\\n            ax.set_xticklabels(xticklabels_cv_sparse)\\n            ax.set_xlabel(\'Coefficient of Variation (CV)\')\\n        else:\\n            ax.set_xticklabels([])\\n        if ci == 0:\\n            ax.set_ylabel(\'Regime Mass\', labelpad=6)\\n        else:\\n            ax.tick_params(axis=\'y\', labelleft=False)\\n\\nmodel_handles_combo = [\\n    plt.Line2D([0], [0], color=COLOR_MAP[\'rate_combined\'], lw=2.0, label=\'Rate\'),\\n    plt.Line2D([0], [0], color=COLOR_MAP[\'timing_combined\'], lw=2.0, label=\'Timing\'),\\n    plt.Line2D([0], [0], color=COLOR_MAP[\'hybrid_combined\'], lw=2.0, label=\'Hybrid\'),\\n]\\nregime_handles_combo = [\\n    plt.Rectangle((0, 0), 1, 1, facecolor=regime_colors_panel[i], edgecolor=\'none\', label=THRESH_REGIME_TICKS[t])\\n    for i, t in enumerate(THRESH_ORDER)\\n]\\n\\nfig_combo.suptitle(\'Hybrid Regime Weights: Per-Block vs Per-Point Mixture\', y=1.01)\\nfig_combo.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.14)\\n\\nleft_bbox_top = combo_axes_left[0].get_position()\\nleft_bbox_bot = combo_axes_left[1].get_position()\\nright_bbox_tl = combo_axes_right[0][0].get_position()\\nright_bbox_tr = combo_axes_right[0][-1].get_position()\\nleft_group_center = (left_bbox_top.x0 + left_bbox_top.x1) / 2\\nright_group_center = (right_bbox_tl.x0 + right_bbox_tr.x1) / 2\\n\\nfig_combo.text(left_bbox_top.x0 - 0.028, left_bbox_top.y1 + 0.040, \'a\', fontsize=16, fontweight=\'bold\')\\nfig_combo.text(right_bbox_tl.x0 - 0.028, right_bbox_tl.y1 + 0.040, \'b\', fontsize=16, fontweight=\'bold\')\\n\\nfig_combo.legend(\\n    handles=regime_handles_combo,\\n    loc=\'lower center\',\\n    bbox_to_anchor=(right_group_center, -0.005),\\n    ncol=9,\\n    frameon=False,\\n)\\ncombo_axes_left[0].legend(\\n    handles=model_handles_combo,\\n    loc=\'upper center\',\\n    bbox_to_anchor=(0.50, 0.995),\\n    ncol=3,\\n    frameon=False,\\n    handlelength=1.8,\\n    columnspacing=1.4,\\n)\\nplt.show()\\n\\nfig_combo.savefig(OUT_DIR / \'fig11_block_shared_regime_weights.png\', bbox_inches=\'tight\')\\nfig_combo.savefig(OUT_DIR / \'fig11_block_shared_regime_weights.pdf\', bbox_inches=\'tight\')\\n", "15": "# Effective-rate-share inputs used by the final combined rate-reliance figure\\nmerge_keys_pcv_eff = [\\n    \'partial_path\', \'scope\', \'model_bank\', \'freq\', \'block\', \'bin_ms\', \'mc\', \'drop_thresh\', \'is_loto\'\\n]\\ncurve_cols_pcv_eff = [\\n    \'cv_json\',\\n    \'rate_drive_test_json\', \'timing_drive_test_json\',\\n    \'rate_drive_aux_test_json\', \'timing_drive_aux_test_json\',\\n]\\ndf_pcv_eff = runs_pcv.merge(\\n    curves_pcv[merge_keys_pcv_eff + curve_cols_pcv_eff],\\n    on=merge_keys_pcv_eff,\\n    how=\'left\',\\n    validate=\'one_to_one\'\\n)\\n\\npcv_effective_base = df_pcv_eff[\\n    (df_pcv_eff[\'scope\'] == \'per_freq_block\')\\n    & (df_pcv_eff[\'model_bank\'] == \'hybrid_combined\')\\n    & (df_pcv_eff[\'drop_thresh\'] == \'NONE\')\\n    & (~df_pcv_eff[\'is_loto\'])\\n    & (np.isclose(df_pcv_eff[\'bin_ms\'], 10.0))\\n    & (df_pcv_eff[\'freq\'].isin(FREQS))\\n    & (df_pcv_eff[\'block\'].isin(BLOCKS))\\n    & (df_pcv_eff[\'thresh_flex_mode\'].astype(str).str.lower() == \'per_point\')\\n].copy()\\n\\n# Old signed-rate-contribution and standalone effective-rate-share plots removed; final combined rate-reliance figure uses pcv_effective_base.\\n", "16": "# Rate-reliance AUC and targeted statistics for the final Supp. Fig. 03 table\\n#\\n# Repetition timing realizations are the inferential unit. MC fits are used only to\\n# create cross-fitted held-out estimates for each rep, then each rep is averaged over\\n# its test-split appearances before the manuscript\'s targeted statistical tests.\\n\\nRR_REBUILD_REP_RATE_RELIANCE = False\\nRR_BOOT_N = 5000\\nRR_PERM_N = 10000\\nRR_RNG_SEED = 20260420\\nRR_TRAIN_FRAC = 0.8\\nRR_BASE_SEED = 0\\n\\nrr_rep_auc_path = OUT_DIR / \'exploratory_rate_reliance_rep_auc.csv\'\\nrr_rep_split_auc_path = OUT_DIR / \'exploratory_rate_reliance_rep_split_auc.csv\'\\nrr_rep_cv_share_path = OUT_DIR / \'exploratory_rate_reliance_rep_cv_share.csv\'\\nrr_rep_cv_share_summary_path = OUT_DIR / \'exploratory_rate_reliance_rep_cv_share_summary.csv\'\\nrr_rep_summary_path = OUT_DIR / \'exploratory_rate_reliance_rep_auc_summary.csv\'\\nrr_rep_freq_posthoc_path = OUT_DIR / \'exploratory_rate_reliance_rep_freq_posthoc.csv\'\\nrr_rep_pooled_block_path = OUT_DIR / \'exploratory_rate_reliance_naive_all_frequency_pooled_block_contrast.csv\'\\nrr_rep_block_by_freq_path = OUT_DIR / \'exploratory_rate_reliance_rep_block_by_freq_posthoc.csv\'\\nrr_rep_evidence_cache_path = OUT_DIR / \'exploratory_rate_reliance_rep_evidence_cache.pkl\'\\n\\n\\ndef _rr_json_dict(x):\\n    if isinstance(x, dict):\\n        return x\\n    if x is None or (isinstance(x, float) and not np.isfinite(x)):\\n        return {}\\n    s = str(x).strip()\\n    if not s or s.lower() == \'nan\':\\n        return {}\\n    try:\\n        return json.loads(s)\\n    except Exception:\\n        return {}\\n\\n\\ndef _rr_scalar_pm1(x, lo, hi):\\n    den = float(hi) - float(lo)\\n    if not (np.isfinite(x) and np.isfinite(lo) and np.isfinite(hi)) or abs(den) < 1e-12:\\n        return 0.0\\n    return float(2.0 * (float(x) - float(lo)) / den - 1.0)\\n\\n\\ndef _rr_auc_from_cv_share(cv, share):\\n    cv = np.asarray(cv, dtype=float)\\n    share = np.asarray(share, dtype=float)\\n    mask = np.isfinite(cv) & np.isfinite(share)\\n    if mask.sum() < 2:\\n        return np.nan\\n    cv = cv[mask]\\n    share = share[mask]\\n    order = np.argsort(cv)\\n    cv = cv[order]\\n    share = share[order]\\n    span = float(cv[-1] - cv[0])\\n    if span <= 0.0:\\n        return float(np.nanmean(share))\\n    return float(np.trapezoid(share, cv) / span)\\n\\n\\ndef _rr_params_from_row(r):\\n    return {\\n        \'g_rate\': float(r.get(\'gain_rate\', 0.0)),\\n        \'g_time\': float(r.get(\'gain_timing\', 0.0)),\\n        \'g_rate_aux\': float(r.get(\'gain_rate_aux\', 0.0)),\\n        \'g_time_aux\': float(r.get(\'gain_timing_aux\', 0.0)),\\n        \'b\': float(r.get(\'bias\', 0.0)),\\n        \'temp\': float(r.get(\'temp\', 1.0)),\\n        \'rate_gate_floor\': float(r.get(\'rate_gate_floor\', 1.0)),\\n        \'rate_gate_center\': float(r.get(\'rate_gate_center\', 0.5)),\\n        \'rate_gate_slope\': float(r.get(\'rate_gate_slope\', 8.0)),\\n    }\\n\\n\\ndef _rr_sem(x):\\n    a = np.asarray(x, dtype=float)\\n    a = a[np.isfinite(a)]\\n    return float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0\\n\\n\\ndef _rr_q025(x):\\n    a = np.asarray(x, dtype=float)\\n    a = a[np.isfinite(a)]\\n    return float(np.nanpercentile(a, 2.5)) if a.size else np.nan\\n\\n\\ndef _rr_q975(x):\\n    a = np.asarray(x, dtype=float)\\n    a = a[np.isfinite(a)]\\n    return float(np.nanpercentile(a, 97.5)) if a.size else np.nan\\n\\n\\ndef _rr_holm(pvals):\\n    p = np.asarray(pvals, dtype=float)\\n    out = np.full(p.shape, np.nan, dtype=float)\\n    good = np.where(np.isfinite(p))[0]\\n    if good.size == 0:\\n        return out\\n    order = good[np.argsort(p[good])]\\n    m = len(order)\\n    prev = 0.0\\n    for rank, idx_p in enumerate(order, start=1):\\n        val = min(1.0, (m - rank + 1) * float(p[idx_p]))\\n        prev = max(prev, val)\\n        out[idx_p] = prev\\n    return out\\n\\n\\ndef _rr_stars(p):\\n    if not np.isfinite(p):\\n        return \'NA\'\\n    if p < 0.001:\\n        return \'***\'\\n    if p < 0.01:\\n        return \'**\'\\n    if p < 0.05:\\n        return \'*\'\\n    return \'ns\'\\n\\n\\ndef _rr_bonferroni(pvals):\\n    p = np.asarray(pvals, dtype=float)\\n    out = np.full(p.shape, np.nan, dtype=float)\\n    good = np.isfinite(p)\\n    if np.any(good):\\n        out[good] = np.minimum(1.0, p[good] * int(np.sum(good)))\\n    return out\\n\\n\\ndef _rr_bootstrap_ci_unpaired(a, b, rng, n_boot=RR_BOOT_N):\\n    a = np.asarray(a, dtype=float)\\n    b = np.asarray(b, dtype=float)\\n    a = a[np.isfinite(a)]\\n    b = b[np.isfinite(b)]\\n    if a.size == 0 or b.size == 0:\\n        return np.nan, np.nan\\n    vals = np.empty(int(n_boot), dtype=float)\\n    for i in range(int(n_boot)):\\n        vals[i] = float(np.mean(rng.choice(b, size=b.size, replace=True)) - np.mean(rng.choice(a, size=a.size, replace=True)))\\n    return tuple(np.nanpercentile(vals, [2.5, 97.5]))\\n\\n\\ndef _rr_perm_p_unpaired(a, b, rng, n_perm=RR_PERM_N):\\n    a = np.asarray(a, dtype=float)\\n    b = np.asarray(b, dtype=float)\\n    a = a[np.isfinite(a)]\\n    b = b[np.isfinite(b)]\\n    if a.size == 0 or b.size == 0:\\n        return np.nan\\n    obs = abs(float(np.mean(b) - np.mean(a)))\\n    joined = np.concatenate([a, b])\\n    n_a = a.size\\n    hits = 0\\n    for _ in range(int(n_perm)):\\n        perm = rng.permutation(joined)\\n        diff = float(np.mean(perm[n_a:]) - np.mean(perm[:n_a]))\\n        hits += abs(diff) >= obs - 1e-12\\n    return float((hits + 1) / (int(n_perm) + 1))\\n\\n\\ndef _rr_perm_p_unpaired_directional(a, b, rng, n_perm=RR_PERM_N):\\n    a = np.asarray(a, dtype=float)\\n    b = np.asarray(b, dtype=float)\\n    a = a[np.isfinite(a)]\\n    b = b[np.isfinite(b)]\\n    if a.size == 0 or b.size == 0:\\n        return np.nan, np.nan\\n    obs = float(np.mean(a) - np.mean(b))\\n    joined = np.concatenate([a, b])\\n    n_a = a.size\\n    hits_a_gt_b = 0\\n    hits_b_gt_a = 0\\n    for _ in range(int(n_perm)):\\n        perm = rng.permutation(joined)\\n        diff = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))\\n        hits_a_gt_b += diff >= obs - 1e-12\\n        hits_b_gt_a += diff <= obs + 1e-12\\n    denom = float(int(n_perm) + 1)\\n    return float((hits_a_gt_b + 1) / denom), float((hits_b_gt_a + 1) / denom)\\n\\n\\ndef _rr_build_rep_level_auc():\\n    import pickle\\n\\n    if str((REPO_ROOT / \'src\' / \'discrimination\').resolve()) not in sys.path:\\n        sys.path.insert(0, str((REPO_ROOT / \'src\' / \'discrimination\').resolve()))\\n\\n    from discrimination_core import (\\n        FeatureSettings,\\n        parse_spike_list,\\n        compute_trial_features,\\n        rate_vote_hardjnd,\\n        timing_evidence_lv,\\n        predict_design_outputs,\\n        predict_design_outputs_rowwise,\\n        _split_reps_shared,\\n    )\\n\\n    pb_results = pd.read_csv(DATA_DIR / \'discrimination\' / \'per-block\' / \'merged_results.csv\')\\n    pcv_results = pd.read_csv(DATA_DIR / \'discrimination\' / \'per-point\' / \'merged_results.csv\')\\n    if \'is_fixed_thresh\' not in pb_results.columns:\\n        pb_results[\'is_fixed_thresh\'] = False\\n    if \'is_fixed_thresh\' not in pcv_results.columns:\\n        pcv_results[\'is_fixed_thresh\'] = False\\n\\n    pb_rows = pb_results[\\n        (pb_results[\'scope\'].astype(str) == \'per_block\')\\n        & (pb_results[\'model_bank\'].astype(str) == \'hybrid_combined\')\\n        & (pb_results[\'drop_thresh\'].astype(str) == \'NONE\')\\n        & (~pb_results[\'is_loto\'].astype(bool))\\n        & (~pb_results[\'is_fixed_thresh\'].astype(bool))\\n        & (np.isclose(pd.to_numeric(pb_results[\'bin_ms\'], errors=\'coerce\'), 10.0))\\n        & (np.isclose(pd.to_numeric(pb_results[\'aux_bin_ms\'], errors=\'coerce\'), 1.0))\\n        & (pd.to_numeric(pb_results[\'freq\'], errors=\'coerce\').isin(FREQS))\\n        & (pb_results[\'block\'].astype(str).isin(BLOCKS))\\n    ].copy()\\n    pcv_rows = pcv_results[\\n        (pcv_results[\'scope\'].astype(str) == \'per_freq_block\')\\n        & (pcv_results[\'model_bank\'].astype(str) == \'hybrid_combined\')\\n        & (pcv_results[\'drop_thresh\'].astype(str) == \'NONE\')\\n        & (~pcv_results[\'is_loto\'].astype(bool))\\n        & (~pcv_results[\'is_fixed_thresh\'].astype(bool))\\n        & (np.isclose(pd.to_numeric(pcv_results[\'bin_ms\'], errors=\'coerce\'), 10.0))\\n        & (np.isclose(pd.to_numeric(pcv_results[\'aux_bin_ms\'], errors=\'coerce\'), 1.0))\\n        & (pcv_results[\'thresh_flex_mode\'].astype(str) == \'per_point\')\\n        & (pd.to_numeric(pcv_results[\'freq\'], errors=\'coerce\').isin(FREQS))\\n        & (pcv_results[\'block\'].astype(str).isin(BLOCKS))\\n    ].copy()\\n\\n    sim_rr = pd.read_csv(CV0_RESP_CSV)\\n    sim_rr[\'freq\'] = pd.to_numeric(sim_rr[\'freq\'], errors=\'coerce\')\\n    sim_rr[\'cv\'] = pd.to_numeric(sim_rr[\'cv\'], errors=\'coerce\').round(1)\\n    rep_source_col = \'rep_index\' if \'rep_index\' in sim_rr.columns else \'rep\'\\n    sim_rr[\'rep\'] = pd.to_numeric(sim_rr[rep_source_col], errors=\'coerce\').astype(int)\\n    sim_rr = sim_rr[\\n        sim_rr[\'freq\'].isin(FREQS)\\n        & sim_rr[\'cv\'].isin(CV_ORDER)\\n        & sim_rr[\'thresh_type\'].isin(THRESH_ORDER)\\n    ].copy()\\n    reps_all = sorted(sim_rr[\'rep\'].dropna().astype(int).unique().tolist())\\n\\n    if rr_rep_evidence_cache_path.exists() and not RR_REBUILD_REP_RATE_RELIANCE:\\n        with open(rr_rep_evidence_cache_path, \'rb\') as f:\\n            evidence_cache = pickle.load(f)\\n    else:\\n        feat_cache = {}\\n        fs_by_bin = {10.0: FeatureSettings(bin_ms=10.0), 1.0: FeatureSettings(bin_ms=1.0)}\\n        for bin_ms, fs in fs_by_bin.items():\\n            for _, r in sim_rr.iterrows():\\n                key = (float(bin_ms), float(r[\'freq\']), str(r[\'thresh_type\']), float(r[\'cv\']), int(r[\'rep\']))\\n                feat_cache[key] = compute_trial_features(parse_spike_list(r[\'spike_times_ms\']), fs)\\n\\n        evidence_cache = {}\\n        for bin_ms in (10.0, 1.0):\\n            rate_keys = (\'n_spikes\',) if np.isclose(bin_ms, 1.0) else (\'n_events\', \'spikes_per_event\')\\n            for freq in FREQS:\\n                for th in THRESH_ORDER:\\n                    for rep in reps_all:\\n                        f0 = feat_cache[(float(bin_ms), float(freq), th, 0.0, int(rep))]\\n                        f1 = feat_cache[(float(bin_ms), float(freq), th, 1.0, int(rep))]\\n                        t0_rate = {k: f0.get(k, np.nan) for k in (\'n_spikes\', \'n_events\', \'spikes_per_event\')}\\n                        t1_rate = {k: f1.get(k, np.nan) for k in (\'n_spikes\', \'n_events\', \'spikes_per_event\')}\\n                        for cv in CV_ORDER:\\n                            fx = feat_cache[(float(bin_ms), float(freq), th, float(cv), int(rep))]\\n                            x_rate = {k: fx.get(k, np.nan) for k in (\'n_spikes\', \'n_events\', \'spikes_per_event\')}\\n                            er = rate_vote_hardjnd(x_rate, t0_rate, t1_rate, k=2.5, keys=rate_keys)\\n                            et = timing_evidence_lv(fx.get(\'silence_lv\', np.nan), f0.get(\'silence_lv\', np.nan), f1.get(\'silence_lv\', np.nan))\\n                            evidence_cache[(float(bin_ms), float(freq), th, float(cv), int(rep))] = {\\n                                \'rate\': float(er),\\n                                \'timing\': float(et),\\n                                \'rate_nonzero\': float(abs(er) > 1e-12),\\n                            }\\n        with open(rr_rep_evidence_cache_path, \'wb\') as f:\\n            pickle.dump(evidence_cache, f)\\n\\n    def _row_auc_rows(fit_rows, model_label, rowwise_weights=False):\\n        split_rows = []\\n        cv_share_rows = []\\n        for _, r in fit_rows.iterrows():\\n            freq = float(r[\'freq\'])\\n            block = str(r[\'block\'])\\n            mc = int(r[\'mc\'])\\n            seed_i = (int(RR_BASE_SEED) * 1000003 + int(mc) * 9176 + 12345) % (2**32 - 1)\\n            train_reps, test_reps = _split_reps_shared(reps_all, train_frac=RR_TRAIN_FRAC, rng=np.random.default_rng(seed_i))\\n            params = _rr_params_from_row(r)\\n            weights = _rr_json_dict(r.get(\'weights_json\', \'{}\'))\\n            weights_by_cv = _rr_json_dict(r.get(\'weights_by_cv_json\', \'{}\')) if rowwise_weights else None\\n            rate_gate_mode = str(r.get(\'rate_gate_mode\', \'none\'))\\n\\n            endpoints = {}\\n            for bin_ms in (10.0, 1.0):\\n                endpoints[bin_ms] = {}\\n                for th in THRESH_ORDER:\\n                    vals = {}\\n                    for cv in (0.0, 1.0):\\n                        rr = [evidence_cache[(float(bin_ms), freq, th, cv, int(rep))][\'rate\'] for rep in train_reps]\\n                        tt = [evidence_cache[(float(bin_ms), freq, th, cv, int(rep))][\'timing\'] for rep in train_reps]\\n                        vals[cv] = {\'rate\': float(np.nanmean(rr)), \'timing\': float(np.nanmean(tt))}\\n                    endpoints[bin_ms][th] = vals\\n\\n            for rep in test_reps:\\n                er = np.zeros((len(CV_ORDER), len(THRESH_ORDER)), dtype=float)\\n                et = np.zeros_like(er)\\n                er_aux = np.zeros_like(er)\\n                et_aux = np.zeros_like(er)\\n                for i_cv, cv in enumerate(CV_ORDER):\\n                    for j_th, th in enumerate(THRESH_ORDER):\\n                        for bin_ms, mat_r, mat_t in ((10.0, er, et), (1.0, er_aux, et_aux)):\\n                            raw = evidence_cache[(float(bin_ms), freq, th, float(cv), int(rep))]\\n                            e0 = endpoints[bin_ms][th][0.0]\\n                            e1 = endpoints[bin_ms][th][1.0]\\n                            mat_r[i_cv, j_th] = _rr_scalar_pm1(raw[\'rate\'], e0[\'rate\'], e1[\'rate\']) * float(raw[\'rate_nonzero\'])\\n                            mat_t[i_cv, j_th] = _rr_scalar_pm1(raw[\'timing\'], e0[\'timing\'], e1[\'timing\'])\\n\\n                cvs_arr = np.asarray(CV_ORDER, dtype=float)\\n                blocks_arr = np.asarray([block] * len(CV_ORDER), dtype=object)\\n                if rowwise_weights:\\n                    pred = predict_design_outputs_rowwise(\\n                        params=params,\\n                        weights_by_cv=weights_by_cv,\\n                        ths=THRESH_ORDER,\\n                        Er=er,\\n                        Et=et,\\n                        Er_aux=er_aux,\\n                        Et_aux=et_aux,\\n                        cvs=cvs_arr,\\n                        blocks=blocks_arr,\\n                        model_bank=\'hybrid_combined\',\\n                        rate_gate_mode=rate_gate_mode,\\n                    )\\n                else:\\n                    pred = predict_design_outputs(\\n                        params=params,\\n                        weights=weights,\\n                        ths=THRESH_ORDER,\\n                        Er=er,\\n                        Et=et,\\n                        Er_aux=er_aux,\\n                        Et_aux=et_aux,\\n                        cvs=cvs_arr,\\n                        blocks=blocks_arr,\\n                        model_bank=\'hybrid_combined\',\\n                        rate_gate_mode=rate_gate_mode,\\n                    )\\n                rate_sig = params[\'g_rate\'] * pred[\'rate_drive\'] + params[\'g_rate_aux\'] * pred[\'rate_drive_aux\']\\n                time_sig = params[\'g_time\'] * pred[\'timing_drive\'] + params[\'g_time_aux\'] * pred[\'timing_drive_aux\']\\n                denom = np.abs(rate_sig) + np.abs(time_sig)\\n                share = np.divide(np.abs(rate_sig), denom, out=np.zeros_like(denom, dtype=float), where=denom > 1e-12)\\n                for cv_i, share_i in zip(cvs_arr, share):\\n                    cv_share_rows.append({\\n                        \'model_label\': model_label,\\n                        \'block\': block,\\n                        \'freq\': freq,\\n                        \'mc\': mc,\\n                        \'rep\': int(rep),\\n                        \'cv\': float(cv_i),\\n                        \'rate_share\': float(share_i),\\n                    })\\n                split_rows.append({\\n                    \'model_label\': model_label,\\n                    \'block\': block,\\n                    \'freq\': freq,\\n                    \'mc\': mc,\\n                    \'rep\': int(rep),\\n                    \'auc\': _rr_auc_from_cv_share(cvs_arr, share),\\n                })\\n        return split_rows, cv_share_rows\\n\\n    pb_split_rows, pb_cv_share_rows = _row_auc_rows(pb_rows, \'Per-Block Hybrid\', rowwise_weights=False)\\n    pcv_split_rows, pcv_cv_share_rows = _row_auc_rows(pcv_rows, \'Per-Point Hybrid\', rowwise_weights=True)\\n    split_auc = pd.DataFrame(pb_split_rows + pcv_split_rows)\\n    split_auc.to_csv(rr_rep_split_auc_path, index=False)\\n\\n    cv_share = pd.DataFrame(pb_cv_share_rows + pcv_cv_share_rows)\\n    cv_share.to_csv(rr_rep_cv_share_path, index=False)\\n    cv_share_summary = (\\n        cv_share.groupby([\'model_label\', \'block\', \'freq\', \'rep\', \'cv\'], as_index=False)\\n        .agg(rate_share=(\'rate_share\', \'mean\'), n_test_splits=(\'rate_share\', \'size\'))\\n    )\\n    cv_share_summary.to_csv(rr_rep_cv_share_summary_path, index=False)\\n\\n    rep_auc = (\\n        split_auc.groupby([\'model_label\', \'block\', \'freq\', \'rep\'], as_index=False)\\n        .agg(auc=(\'auc\', \'mean\'), n_test_splits=(\'auc\', \'size\'))\\n    )\\n    rep_auc.to_csv(rr_rep_auc_path, index=False)\\n    return rep_auc\\n\\n\\nif RR_REBUILD_REP_RATE_RELIANCE or not rr_rep_auc_path.exists() or not rr_rep_cv_share_summary_path.exists():\\n    rep_auc_long = _rr_build_rep_level_auc()\\nelse:\\n    rep_auc_long = pd.read_csv(rr_rep_auc_path)\\n\\nrep_auc_long[\'freq\'] = pd.to_numeric(rep_auc_long[\'freq\'], errors=\'coerce\')\\nrep_auc_long = rep_auc_long[rep_auc_long[\'block\'].isin(BLOCKS) & rep_auc_long[\'freq\'].isin(FREQS)].copy()\\n\\nrep_auc_summary = (\\n    rep_auc_long.groupby([\'model_label\', \'block\', \'freq\'], as_index=False)\\n    .agg(\\n        auc_mean=(\'auc\', \'mean\'),\\n        auc_sem=(\'auc\', _rr_sem),\\n        auc_q025=(\'auc\', _rr_q025),\\n        auc_q975=(\'auc\', _rr_q975),\\n        n_rep=(\'auc\', \'size\'),\\n        mean_test_splits=(\'n_test_splits\', \'mean\'),\\n    )\\n)\\nrep_auc_summary.to_csv(rr_rep_summary_path, index=False)\\n\\nrr_rng = np.random.default_rng(RR_RNG_SEED)\\n\\n# Frequency family used by panel-b markers and the Supp. Fig. 03 table.\\n# All pairwise frequency rows are kept in the CSV so the Bonferroni family is explicit;\\n# the final table filters this to comparisons against 20 Hz.\\nfreq_rows = []\\nfor model_label in [\'Per-Block Hybrid\', \'Per-Point Hybrid\']:\\n    for block in BLOCKS:\\n        for i, freq_a in enumerate(FREQS):\\n            for freq_b in FREQS[i + 1:]:\\n                a = rep_auc_long[\\n                    (rep_auc_long[\'model_label\'] == model_label)\\n                    & (rep_auc_long[\'block\'] == block)\\n                    & np.isclose(rep_auc_long[\'freq\'], float(freq_a))\\n                ][\'auc\'].to_numpy(float)\\n                b = rep_auc_long[\\n                    (rep_auc_long[\'model_label\'] == model_label)\\n                    & (rep_auc_long[\'block\'] == block)\\n                    & np.isclose(rep_auc_long[\'freq\'], float(freq_b))\\n                ][\'auc\'].to_numpy(float)\\n                ci_low, ci_high = _rr_bootstrap_ci_unpaired(a, b, rr_rng)\\n                freq_rows.append({\\n                    \'contrast_family\': \'freq_b_minus_freq_a_unpaired_rep\',\\n                    \'model_label\': model_label,\\n                    \'block\': block,\\n                    \'freq_a\': float(freq_a),\\n                    \'freq_b\': float(freq_b),\\n                    \'contrast\': f\'{int(freq_b)} - {int(freq_a)} Hz\',\\n                    \'mean_diff\': float(np.nanmean(b) - np.nanmean(a)),\\n                    \'ci_low\': ci_low,\\n                    \'ci_high\': ci_high,\\n                    \'p_perm\': _rr_perm_p_unpaired(a, b, rr_rng),\\n                    \'n_a\': int(np.isfinite(a).sum()),\\n                    \'n_b\': int(np.isfinite(b).sum()),\\n                })\\nrep_freq_posthoc = pd.DataFrame(freq_rows)\\nrep_freq_posthoc[\'p_holm\'] = _rr_holm(rep_freq_posthoc[\'p_perm\'].to_numpy(float))\\nrep_freq_posthoc[\'stars\'] = rep_freq_posthoc[\'p_holm\'].map(_rr_stars)\\nrep_freq_posthoc[\'p_bonferroni_family\'] = _rr_bonferroni(rep_freq_posthoc[\'p_perm\'].to_numpy(float))\\nrep_freq_posthoc[\'stars_bonferroni_family\'] = rep_freq_posthoc[\'p_bonferroni_family\'].map(_rr_stars)\\nrep_freq_posthoc.to_csv(rr_rep_freq_posthoc_path, index=False)\\n\\n# Direction-aware std0/std1 contrasts at each frequency.\\n# These are intentionally unpaired by repetition, matching the frequency-family tests\\n# above and asking whether the observed direction is larger than expected under\\n# exchangeability of the block labels at that frequency.\\nblock_by_freq_rows = []\\nfor model_label in [\'Per-Block Hybrid\', \'Per-Point Hybrid\']:\\n    for freq in FREQS:\\n        std0_vals = rep_auc_long[\\n            (rep_auc_long[\'model_label\'] == model_label)\\n            & (rep_auc_long[\'block\'] == \'std0\')\\n            & np.isclose(rep_auc_long[\'freq\'], float(freq))\\n        ][\'auc\'].to_numpy(float)\\n        std1_vals = rep_auc_long[\\n            (rep_auc_long[\'model_label\'] == model_label)\\n            & (rep_auc_long[\'block\'] == \'std1\')\\n            & np.isclose(rep_auc_long[\'freq\'], float(freq))\\n        ][\'auc\'].to_numpy(float)\\n        ci_low, ci_high = _rr_bootstrap_ci_unpaired(std1_vals, std0_vals, rr_rng)\\n        p_std0_gt_std1, p_std1_gt_std0 = _rr_perm_p_unpaired_directional(std0_vals, std1_vals, rr_rng)\\n        mean_diff = float(np.nanmean(std0_vals) - np.nanmean(std1_vals))\\n        block_by_freq_rows.append({\\n            \'contrast_family\': \'block_std0_minus_std1_by_frequency_unmatched_rep_directional\',\\n            \'model_label\': model_label,\\n            \'freq\': float(freq),\\n            \'contrast\': \'std0 - std1\',\\n            \'mean_diff\': mean_diff,\\n            \'ci_low\': ci_low,\\n            \'ci_high\': ci_high,\\n            \'direction\': \'std0 > std1\' if mean_diff > 0 else \'std1 > std0\' if mean_diff < 0 else \'tie\',\\n            \'p_perm_std0_gt_std1\': p_std0_gt_std1,\\n            \'p_perm_std1_gt_std0\': p_std1_gt_std0,\\n            \'n_std0\': int(np.isfinite(std0_vals).sum()),\\n            \'n_std1\': int(np.isfinite(std1_vals).sum()),\\n        })\\nrep_block_by_freq_posthoc = pd.DataFrame(block_by_freq_rows)\\nrep_block_by_freq_posthoc[\'p_bonferroni_std0_gt_std1\'] = _rr_bonferroni(rep_block_by_freq_posthoc[\'p_perm_std0_gt_std1\'].to_numpy(float))\\nrep_block_by_freq_posthoc[\'p_bonferroni_std1_gt_std0\'] = _rr_bonferroni(rep_block_by_freq_posthoc[\'p_perm_std1_gt_std0\'].to_numpy(float))\\nrep_block_by_freq_posthoc[\'stars_std0_gt_std1\'] = rep_block_by_freq_posthoc[\'p_bonferroni_std0_gt_std1\'].map(_rr_stars)\\nrep_block_by_freq_posthoc[\'stars_std1_gt_std0\'] = rep_block_by_freq_posthoc[\'p_bonferroni_std1_gt_std0\'].map(_rr_stars)\\nrep_block_by_freq_posthoc[\'p_observed_direction\'] = np.where(\\n    rep_block_by_freq_posthoc[\'mean_diff\'] >= 0,\\n    rep_block_by_freq_posthoc[\'p_perm_std0_gt_std1\'],\\n    rep_block_by_freq_posthoc[\'p_perm_std1_gt_std0\'],\\n)\\nrep_block_by_freq_posthoc[\'p_bonferroni_observed_direction\'] = np.where(\\n    rep_block_by_freq_posthoc[\'mean_diff\'] >= 0,\\n    rep_block_by_freq_posthoc[\'p_bonferroni_std0_gt_std1\'],\\n    rep_block_by_freq_posthoc[\'p_bonferroni_std1_gt_std0\'],\\n)\\nrep_block_by_freq_posthoc[\'stars_observed_direction\'] = rep_block_by_freq_posthoc[\'p_bonferroni_observed_direction\'].map(_rr_stars)\\nrep_block_by_freq_posthoc.to_csv(rr_rep_block_by_freq_path, index=False)\\n\\n# Pooled std0/std1 block contrast used in the manuscript text and final table.\\npooled_block_rows = []\\nfor model_label in [\'Per-Block Hybrid\', \'Per-Point Hybrid\']:\\n    std0_vals = rep_auc_long[\\n        (rep_auc_long[\'model_label\'] == model_label)\\n        & (rep_auc_long[\'block\'] == \'std0\')\\n        & (rep_auc_long[\'freq\'].isin(FREQS))\\n    ][\'auc\'].to_numpy(float)\\n    std1_vals = rep_auc_long[\\n        (rep_auc_long[\'model_label\'] == model_label)\\n        & (rep_auc_long[\'block\'] == \'std1\')\\n        & (rep_auc_long[\'freq\'].isin(FREQS))\\n    ][\'auc\'].to_numpy(float)\\n    ci_low, ci_high = _rr_bootstrap_ci_unpaired(std1_vals, std0_vals, rr_rng)\\n    p_perm = _rr_perm_p_unpaired(std1_vals, std0_vals, rr_rng)\\n    pooled_block_rows.append({\\n        \'contrast_family\': \'block_std0_minus_std1_all_frequency_unmatched_rep\',\\n        \'model_label\': model_label,\\n        \'freq_group\': \'all_20_200\',\\n        \'freqs\': \',\'.join(str(int(f)) for f in FREQS),\\n        \'contrast\': \'std0 - std1\',\\n        \'mean_diff\': float(np.nanmean(std0_vals) - np.nanmean(std1_vals)),\\n        \'ci_low\': ci_low,\\n        \'ci_high\': ci_high,\\n        \'p_perm\': p_perm,\\n        \'p_perm_unstratified\': p_perm,\\n        \'n_std0\': int(np.isfinite(std0_vals).sum()),\\n        \'n_std1\': int(np.isfinite(std1_vals).sum()),\\n    })\\nrep_pooled_block_posthoc = pd.DataFrame(pooled_block_rows)\\nrep_pooled_block_posthoc[\'p_bonferroni_two_models\'] = _rr_bonferroni(rep_pooled_block_posthoc[\'p_perm\'].to_numpy(float))\\nrep_pooled_block_posthoc[\'stars_bonferroni_two_models\'] = rep_pooled_block_posthoc[\'p_bonferroni_two_models\'].map(_rr_stars)\\nrep_pooled_block_posthoc.to_csv(rr_rep_pooled_block_path, index=False)\\n", "17": "# Combined rate-reliance figure and final Supp. Fig. 03 statistics table\\n# Panel a: effective rate-share curves over CV.\\n# Panel b: combined rep-level AUC summary. Symbols floating above each point mark\\n# significant increases relative to 20 Hz for that model/block trace.\\n\\nif \'rep_auc_summary\' not in globals() or rep_auc_summary.empty:\\n    rep_auc_summary = pd.read_csv(OUT_DIR / \'exploratory_rate_reliance_rep_auc_summary.csv\')\\nif \'rep_freq_posthoc\' not in globals() or rep_freq_posthoc.empty:\\n    rep_freq_posthoc = pd.read_csv(OUT_DIR / \'exploratory_rate_reliance_rep_freq_posthoc.csv\')\\nrr_rep_cv_share_summary_path = OUT_DIR / \'exploratory_rate_reliance_rep_cv_share_summary.csv\'\\nif rr_rep_cv_share_summary_path.exists():\\n    rep_cv_share_summary = pd.read_csv(rr_rep_cv_share_summary_path)\\nelse:\\n    raise FileNotFoundError(f\'Missing {rr_rep_cv_share_summary_path}; run the preceding rate-reliance AUC/statistics cell first.\')\\nrr_rep_block_by_freq_path = OUT_DIR / \'exploratory_rate_reliance_rep_block_by_freq_posthoc.csv\'\\nif rr_rep_block_by_freq_path.exists():\\n    rep_block_by_freq_posthoc = pd.read_csv(rr_rep_block_by_freq_path)\\nelse:\\n    rep_block_by_freq_posthoc = pd.DataFrame()\\nrep_auc_summary[\'freq\'] = pd.to_numeric(rep_auc_summary[\'freq\'], errors=\'coerce\')\\nfor _c in [\'freq\', \'cv\', \'rate_share\']:\\n    if _c in rep_cv_share_summary.columns:\\n        rep_cv_share_summary[_c] = pd.to_numeric(rep_cv_share_summary[_c], errors=\'coerce\')\\nfor _c in [\'freq_a\', \'freq_b\', \'mean_diff\', \'p_holm\', \'p_bonferroni_family\']:\\n    if _c in rep_freq_posthoc.columns:\\n        rep_freq_posthoc[_c] = pd.to_numeric(rep_freq_posthoc[_c], errors=\'coerce\')\\nfor _c in [\'freq\', \'mean_diff\', \'ci_low\', \'ci_high\', \'p_perm_std0_gt_std1\', \'p_perm_std1_gt_std0\', \'p_bonferroni_std0_gt_std1\', \'p_bonferroni_std1_gt_std0\']:\\n    if _c in rep_block_by_freq_posthoc.columns:\\n        rep_block_by_freq_posthoc[_c] = pd.to_numeric(rep_block_by_freq_posthoc[_c], errors=\'coerce\')\\n\\npurple_rr = COLOR_MAP[\'hybrid_combined\']\\norange_rr = \'#f4a261\'\\nrate_ref_color_rr = COLOR_MAP[\'rate_combined\']\\ntiming_ref_color_rr = COLOR_MAP[\'timing_combined\']\\n\\n\\ndef _rr_panel_label(ax, letter, dx=-0.14, dy=1.05):\\n    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=14, fontweight=\'bold\', ha=\'left\', va=\'top\')\\n\\n\\ndef _rr_p_to_symbol(p):\\n    return np.isfinite(p) and (p < 0.05)\\n\\n\\ndef _rr_aggregate_rep_cv_share(model_label, block, freq):\\n    sub = rep_cv_share_summary[\\n        (rep_cv_share_summary[\'model_label\'] == model_label)\\n        & (rep_cv_share_summary[\'block\'] == block)\\n        & np.isclose(rep_cv_share_summary[\'freq\'], float(freq))\\n    ].copy()\\n    if sub.empty:\\n        return pd.DataFrame(columns=[\'cv\', \'share_mean\', \'share_sem\', \'share_q025\', \'share_q975\'])\\n    return (\\n        sub.groupby(\'cv\', as_index=False)\\n        .agg(\\n            share_mean=(\'rate_share\', \'mean\'),\\n            share_sem=(\'rate_share\', _rr_sem),\\n            share_q025=(\'rate_share\', _rr_q025),\\n            share_q975=(\'rate_share\', _rr_q975),\\n        )\\n        .sort_values(\'cv\')\\n    )\\n\\n\\nfig_rr_combo = plt.figure(figsize=(16.5, 8.8), constrained_layout=False)\\ngs_rr_combo = fig_rr_combo.add_gridspec(nrows=2, ncols=1, height_ratios=[2.0, 1.12], hspace=0.34)\\ngs_rr_curves = gs_rr_combo[0].subgridspec(nrows=2, ncols=4, hspace=0.24, wspace=0.10)\\naxes_rr_curve = np.empty((2, 4), dtype=object)\\nfor ri in range(2):\\n    for ci in range(4):\\n        axes_rr_curve[ri, ci] = fig_rr_combo.add_subplot(gs_rr_curves[ri, ci])\\nax_rr_sum = fig_rr_combo.add_subplot(gs_rr_combo[1, 0])\\n\\nxticks_cv_major_rr = np.round(np.arange(0.0, 1.01, 0.2), 1)\\nxticks_cv_minor_rr = np.round(np.arange(0.0, 1.01, 0.1), 1)\\nfor ri, blk in enumerate(BLOCKS):\\n    for ci, f in enumerate(FREQS):\\n        ax = axes_rr_curve[ri, ci]\\n        ss_pb = _rr_aggregate_rep_cv_share(\'Per-Block Hybrid\', blk, f)\\n        ss_pp = _rr_aggregate_rep_cv_share(\'Per-Point Hybrid\', blk, f)\\n\\n        if ss_pb.empty or ss_pp.empty:\\n            raise ValueError(f\'Missing rep-level rate-share curve for {blk} | {int(f)} Hz; rerun the preceding rate-reliance AUC/statistics cell.\')\\n\\n        # Draw per-point first, then per-block, so overlap reads more purple than red/orange.\\n        if not ss_pp.empty:\\n            ax.fill_between(\\n                ss_pp[\'cv\'].to_numpy(float), 0.0, ss_pp[\'share_mean\'].to_numpy(float),\\n                color=orange_rr, alpha=0.10, linewidth=0, zorder=2,\\n            )\\n        if not ss_pb.empty:\\n            ax.fill_between(\\n                ss_pb[\'cv\'].to_numpy(float), 0.0, ss_pb[\'share_mean\'].to_numpy(float),\\n                color=purple_rr, alpha=0.22, linewidth=0, zorder=2.1,\\n            )\\n        if not ss_pp.empty:\\n            ax.plot(ss_pp[\'cv\'], ss_pp[\'share_mean\'], color=orange_rr, linewidth=2.4, alpha=0.78, zorder=3)\\n        if not ss_pb.empty:\\n            ax.plot(ss_pb[\'cv\'], ss_pb[\'share_mean\'], color=purple_rr, linewidth=2.4, alpha=0.86, zorder=4)\\n\\n        panel_a_sig_markers_rr = []\\n        if blk == \'std0\' and not rep_block_by_freq_posthoc.empty:\\n            for model_label, color_rr, x_ax in [\\n                (\'Per-Block Hybrid\', purple_rr, 0.47),\\n                (\'Per-Point Hybrid\', orange_rr, 0.53),\\n            ]:\\n                stat = rep_block_by_freq_posthoc[\\n                    (rep_block_by_freq_posthoc[\'model_label\'] == model_label)\\n                    & np.isclose(rep_block_by_freq_posthoc[\'freq\'], float(f))\\n                ]\\n                if stat.empty:\\n                    continue\\n                p = float(stat.iloc[0].get(\'p_bonferroni_std0_gt_std1\', np.nan))\\n                mean_diff = float(stat.iloc[0].get(\'mean_diff\', np.nan))\\n                if mean_diff > 0 and _rr_p_to_symbol(p):\\n                    panel_a_sig_markers_rr.append((x_ax, color_rr))\\n\\n        ax.set_title(f\'{blk} | {int(f)} Hz\', fontsize=11, pad=4)\\n        ax.set_xlim(0.0, 1.0)\\n        ax.set_ylim(0.0, 1.05)\\n        ax.spines[\'top\'].set_visible(False)\\n        ax.spines[\'right\'].set_visible(False)\\n        ax.spines[\'left\'].set_visible(True)\\n        ax.spines[\'bottom\'].set_visible(True)\\n        ax.set_xticks(xticks_cv_major_rr)\\n        ax.set_xticks(xticks_cv_minor_rr, minor=True)\\n        ax.tick_params(axis=\'x\', which=\'minor\', length=2.5, width=0.8)\\n        ax.grid(True, alpha=0.22, linewidth=0.8)\\n        if ri == 1:\\n            ax.set_xlabel(\'Coefficient of Variation (CV)\', fontsize=10)\\n            ax.set_xticklabels([f\'{x:.1f}\' for x in xticks_cv_major_rr], fontsize=9)\\n        else:\\n            ax.set_xticklabels([])\\n        if ci == 0:\\n            ax.set_ylabel(\'Effective Rate Share\', fontsize=10)\\n        else:\\n            ax.set_yticklabels([])\\n        if len(panel_a_sig_markers_rr) == 1:\\n            panel_a_sig_markers_rr = [(0.50, panel_a_sig_markers_rr[0][1])]\\n        for x_ax, color_rr in panel_a_sig_markers_rr:\\n            ax.text(\\n                x_ax, 0.975, \'*\', transform=ax.get_yaxis_transform(),\\n                ha=\'center\', va=\'center\', color=color_rr,\\n                fontsize=14, fontweight=\'bold\', clip_on=False, zorder=8,\\n            )\\n\\nfig_rr_combo.text(0.5, 0.975, \'Rate Reliance\', ha=\'center\', va=\'top\', fontsize=20)\\n\\nfreq_order_rr_combo = [20.0, 50.0, 100.0, 200.0]\\nx_pos_rr_combo = {f: i for i, f in enumerate(freq_order_rr_combo)}\\npoint_offsets_rr_combo = {\\n    (\'Per-Point Hybrid\', \'std0\'): -0.15,\\n    (\'Per-Point Hybrid\', \'std1\'): -0.05,\\n    (\'Per-Block Hybrid\', \'std0\'): 0.05,\\n    (\'Per-Block Hybrid\', \'std1\'): 0.15,\\n}\\nmodel_colors_rr_combo = {\'Per-Block Hybrid\': purple_rr, \'Per-Point Hybrid\': orange_rr}\\nblock_markers_rr_combo = {\'std0\': \'o\', \'std1\': \'s\'}\\nsig_symbols_rr_combo = {\'std0\': \'*\', \'std1\': \'*\'}\\n\\nfor model_label in [\'Per-Point Hybrid\', \'Per-Block Hybrid\']:\\n    for blk in BLOCKS:\\n        sub = rep_auc_summary[\\n            (rep_auc_summary[\'model_label\'] == model_label)\\n            & (rep_auc_summary[\'block\'] == blk)\\n        ].sort_values(\'freq\')\\n        x = np.asarray([x_pos_rr_combo[float(f)] + point_offsets_rr_combo[(model_label, blk)] for f in sub[\'freq\']], dtype=float)\\n        y = sub[\'auc_mean\'].to_numpy(float)\\n        yerr = np.vstack([\\n            y - sub[\'auc_q025\'].to_numpy(float),\\n            sub[\'auc_q975\'].to_numpy(float) - y,\\n        ])\\n        ax_rr_sum.plot(x, y, color=model_colors_rr_combo[model_label], linewidth=2.5, alpha=0.95, zorder=2)\\n        ax_rr_sum.errorbar(\\n            x, y, yerr=yerr, fmt=block_markers_rr_combo[blk], color=model_colors_rr_combo[model_label],\\n            markerfacecolor=model_colors_rr_combo[model_label], markeredgecolor=\'white\', markeredgewidth=0.8,\\n            markersize=8.0, elinewidth=1.6, capsize=4, capthick=1.6, linewidth=0, zorder=3,\\n        )\\n\\n        for _, row in sub.iterrows():\\n            f = float(row[\'freq\'])\\n            if np.isclose(f, 20.0):\\n                continue\\n            stat = rep_freq_posthoc[\\n                (rep_freq_posthoc[\'model_label\'] == model_label)\\n                & (rep_freq_posthoc[\'block\'] == blk)\\n                & np.isclose(rep_freq_posthoc[\'freq_a\'], 20.0)\\n                & np.isclose(rep_freq_posthoc[\'freq_b\'], f)\\n            ]\\n            if stat.empty:\\n                continue\\n            mean_diff = float(stat.iloc[0][\'mean_diff\'])\\n            if \'p_bonferroni_family\' in stat.columns and np.isfinite(stat.iloc[0][\'p_bonferroni_family\']):\\n                p = float(stat.iloc[0][\'p_bonferroni_family\'])\\n            else:\\n                p = float(stat.iloc[0].get(\'p_holm\', np.nan))\\n            if mean_diff <= 0 or not _rr_p_to_symbol(p):\\n                continue\\n            xi = x_pos_rr_combo[f] + point_offsets_rr_combo[(model_label, blk)]\\n            ax_rr_sum.text(\\n                xi,\\n                float(row[\'auc_q975\']) + 0.04,\\n                sig_symbols_rr_combo[blk],\\n                ha=\'center\',\\n                va=\'bottom\',\\n                color=model_colors_rr_combo[model_label],\\n                fontsize=14,\\n                fontweight=\'bold\',\\n                clip_on=False,\\n            )\\n\\nax_rr_sum.set_xticks(range(len(freq_order_rr_combo)))\\nax_rr_sum.set_xticklabels([str(int(f)) for f in freq_order_rr_combo])\\nax_rr_sum.set_xlabel(\'Frequency (Hz)\')\\nax_rr_sum.set_ylabel(\'Mean Effective Rate Share (AUC)\')\\nax_rr_sum.set_ylim(-0.02, 1.08)\\n# x-limits are reset after layout adjustment so tick centers align to panel-a columns.\\nax_rr_sum.set_xlim(-0.35, len(freq_order_rr_combo) - 0.65)\\nax_rr_sum.spines[\'top\'].set_visible(False)\\nax_rr_sum.spines[\'right\'].set_visible(False)\\nax_rr_sum.spines[\'left\'].set_visible(True)\\nax_rr_sum.spines[\'bottom\'].set_visible(True)\\nax_rr_sum.grid(True, alpha=0.25, linewidth=0.8)\\nshape_handles_rr_combo = [\\n    plt.Line2D([0], [0], marker=\'o\', color=\'0.25\', markerfacecolor=\'0.25\', linestyle=\'None\', markersize=8, label=\'std0\'),\\n    plt.Line2D([0], [0], marker=\'s\', color=\'0.25\', markerfacecolor=\'0.25\', linestyle=\'None\', markersize=8, label=\'std1\'),\\n]\\nshape_legend_rr_combo = ax_rr_sum.legend(\\n    handles=shape_handles_rr_combo, loc=\'upper left\', frameon=False, fontsize=10,\\n    handletextpad=0.6, borderaxespad=0.5, labelspacing=0.45,\\n)\\nax_rr_sum.add_artist(shape_legend_rr_combo)\\n\\nhandles_rr_combo = [\\n    plt.Line2D([0], [0], color=purple_rr, lw=2.8, label=\'Per-Block Hybrid\'),\\n    plt.Line2D([0], [0], color=orange_rr, lw=2.8, label=\'Per-Point Hybrid\'),\\n]\\nfig_rr_combo.subplots_adjust(left=0.055, right=0.995, top=0.92, bottom=0.13)\\n# Align panel-b frequency ticks with the visual centers of the panel-a columns.\\n_bpos_rr = ax_rr_sum.get_position()\\n_col_centers_rr = [\\n    0.5 * (axes_rr_curve[0, ci].get_position().x0 + axes_rr_curve[0, ci].get_position().x1)\\n    for ci in range(len(freq_order_rr_combo))\\n]\\n_r0_rr = (_col_centers_rr[0] - _bpos_rr.x0) / (_bpos_rr.x1 - _bpos_rr.x0)\\n_r1_rr = (_col_centers_rr[-1] - _bpos_rr.x0) / (_bpos_rr.x1 - _bpos_rr.x0)\\n_xlim_width_rr = (len(freq_order_rr_combo) - 1) / (_r1_rr - _r0_rr)\\n_xlim_left_rr = -_r0_rr * _xlim_width_rr\\nax_rr_sum.set_xlim(_xlim_left_rr, _xlim_left_rr + _xlim_width_rr)\\nlegend_x_rr_combo = 0.5 * (ax_rr_sum.get_position().x0 + ax_rr_sum.get_position().x1)\\nfig_rr_combo.legend(\\n    handles=handles_rr_combo, loc=\'lower center\', bbox_to_anchor=(legend_x_rr_combo, 0.005),\\n    ncol=2, frameon=False, fontsize=10,\\n)\\nlabel_x_rr_combo = 0.004\\nfig_rr_combo.text(\\n    label_x_rr_combo, axes_rr_curve[0, 0].get_position().y1 + 0.015, \'a\',\\n    fontsize=14, fontweight=\'bold\', ha=\'left\', va=\'top\',\\n)\\nfig_rr_combo.text(\\n    label_x_rr_combo, ax_rr_sum.get_position().y1 + 0.015, \'b\',\\n    fontsize=14, fontweight=\'bold\', ha=\'left\', va=\'top\',\\n)\\nplt.show()\\n\\nfig_rr_combo.savefig(OUT_DIR / \'suppfig03_rate_reliance.png\', dpi=300, bbox_inches=\'tight\')\\nfig_rr_combo.savefig(OUT_DIR / \'suppfig03_rate_reliance.pdf\', bbox_inches=\'tight\')\\n\\n# Supp. Fig. 03 stats helper table: 20 Hz frequency comparisons\\n# plus direction-aware std0/std1 block contrasts at each frequency.\\n_model_short_rr = {\'Per-Block Hybrid\': \'Per-Block\', \'Per-Point Hybrid\': \'Per-Point\'}\\n_model_order_rr = {\'Per-Block\': 0, \'Per-Point\': 1}\\n_block_order_rr = {\'std0\': 0, \'std1\': 1, \'\': 2}\\n\\nsuppfig03_freq20_stats = rep_freq_posthoc[\\n    (np.isclose(rep_freq_posthoc[\'freq_a\'], 20.0))\\n    & (rep_freq_posthoc[\'freq_b\'].isin([50.0, 100.0, 200.0]))\\n].copy()\\nsuppfig03_freq20_stats[\'Section\'] = \'Frequency vs 20 Hz\'\\nsuppfig03_freq20_stats[\'Model\'] = suppfig03_freq20_stats[\'model_label\'].map(_model_short_rr).fillna(suppfig03_freq20_stats[\'model_label\'])\\nsuppfig03_freq20_stats[\'Block\'] = suppfig03_freq20_stats[\'block\']\\nsuppfig03_freq20_stats[\'Comparison\'] = \'20 vs \' + suppfig03_freq20_stats[\'freq_b\'].astype(int).astype(str) + \' Hz\'\\nsuppfig03_freq20_stats[\'delta_auc\'] = suppfig03_freq20_stats[\'mean_diff\']\\nsuppfig03_freq20_stats[\'p_raw\'] = suppfig03_freq20_stats[\'p_perm\']\\nsuppfig03_freq20_stats[\'p_bonferroni\'] = suppfig03_freq20_stats[\'p_bonferroni_family\']\\nsuppfig03_freq20_stats[\'stars\'] = suppfig03_freq20_stats[\'stars_bonferroni_family\']\\n\\nsuppfig03_table_parts = [suppfig03_freq20_stats[[\\n    \'Section\', \'Model\', \'Block\', \'Comparison\', \'delta_auc\', \'ci_low\', \'ci_high\',\\n    \'p_raw\', \'p_bonferroni\', \'stars\', \'freq_b\'\\n]]]\\n\\nif not rep_block_by_freq_posthoc.empty:\\n    _block_freq = rep_block_by_freq_posthoc.copy()\\n    _block_freq[\'Section\'] = \'Block difference by frequency\'\\n    _block_freq[\'Model\'] = _block_freq[\'model_label\'].map(_model_short_rr).fillna(_block_freq[\'model_label\'])\\n    _block_freq[\'Block\'] = \'\'\\n    _block_freq[\'Comparison\'] = _block_freq[\'freq\'].astype(int).astype(str) + \' Hz \' + _block_freq[\'direction\'].astype(str)\\n    _block_freq[\'delta_auc\'] = _block_freq[\'mean_diff\']\\n    _block_freq[\'p_raw\'] = np.where(\\n        _block_freq[\'mean_diff\'] >= 0,\\n        _block_freq[\'p_perm_std0_gt_std1\'],\\n        _block_freq[\'p_perm_std1_gt_std0\'],\\n    )\\n    _block_freq[\'p_bonferroni\'] = np.where(\\n        _block_freq[\'mean_diff\'] >= 0,\\n        _block_freq[\'p_bonferroni_std0_gt_std1\'],\\n        _block_freq[\'p_bonferroni_std1_gt_std0\'],\\n    )\\n    _block_freq[\'stars\'] = _block_freq[\'p_bonferroni\'].map(_rr_stars)\\n    _block_freq[\'freq_b\'] = _block_freq[\'freq\'] + 500.0\\n    suppfig03_table_parts.append(_block_freq[[\\n        \'Section\', \'Model\', \'Block\', \'Comparison\', \'delta_auc\', \'ci_low\', \'ci_high\',\\n        \'p_raw\', \'p_bonferroni\', \'stars\', \'freq_b\'\\n    ]])\\n\\nsuppfig03_rate_reliance_stats_summary = pd.concat(suppfig03_table_parts, ignore_index=True)\\nsuppfig03_rate_reliance_stats_summary[\'_model_order\'] = suppfig03_rate_reliance_stats_summary[\'Model\'].map(_model_order_rr).fillna(99)\\nsuppfig03_rate_reliance_stats_summary[\'_block_order\'] = suppfig03_rate_reliance_stats_summary[\'Block\'].map(_block_order_rr).fillna(99)\\nsuppfig03_rate_reliance_stats_summary[\'_section_order\'] = np.where(\\n    suppfig03_rate_reliance_stats_summary[\'Section\'].eq(\'Frequency vs 20 Hz\'), 0, 1\\n)\\nsuppfig03_rate_reliance_stats_summary = (\\n    suppfig03_rate_reliance_stats_summary\\n    .sort_values([\'_model_order\', \'_section_order\', \'_block_order\', \'freq_b\'])\\n    .drop(columns=[\'_model_order\', \'_block_order\', \'_section_order\', \'freq_b\'])\\n    .reset_index(drop=True)\\n)\\nsuppfig03_rate_reliance_stats_summary.to_csv(OUT_DIR / \'suppfig03_rate_reliance_stats_summary.csv\', index=False)\\nprint(\'Supp. Fig. 03 rate-reliance stats summary:\')\\ndisplay(suppfig03_rate_reliance_stats_summary)\\n"}'


def manuscript_figure_specs() -> tuple[ManuscriptFigureSpec, ...]:
    """Return manuscript figures in the order they are discussed in the manuscript."""

    return MANUSCRIPT_FIGURE_SPECS


def get_manuscript_figure_spec(key: str) -> ManuscriptFigureSpec:
    """Return a figure spec by key, label, or title."""

    needle = str(key).strip().lower().replace(" ", "_")
    for spec in MANUSCRIPT_FIGURE_SPECS:
        aliases = {
            spec.key.lower(),
            spec.label.lower().replace(" ", "_"),
            spec.title.lower().replace(" ", "_"),
        }
        if needle in aliases:
            return spec
    known = ", ".join(spec.key for spec in MANUSCRIPT_FIGURE_SPECS)
    raise KeyError(f"Unknown manuscript figure {key!r}. Known keys: {known}")


def _legacy_cell_sources() -> dict[int, str]:
    return {int(k): str(v) for k, v in json.loads(_LEGACY_CELL_SOURCES_JSON).items()}


def _quiet_display(*_args: Any, **_kwargs: Any) -> None:
    return None


def _normalize_legacy_source(
    source: str,
    *,
    rebuild_rate_reliance: bool,
    use_notebook_cache: bool,
) -> str:
    """Patch notebook-era source so it writes to manuscript paths and stays quiet."""

    source = source.replace(
        "SLACK_DIR = REPO_ROOT / 'data' / 'discrimination' / 'slack'",
        "SLACK_DIR = FIGURE_PATHS.discrimination_slack_dir",
    )
    source = source.replace(
        "from IPython.display import display",
        "from IPython.display import display as _ipython_display\ndisplay = _quiet_display",
    )
    source = source.replace(
        "from IPython.display import HTML, clear_output, display",
        "from IPython.display import HTML, clear_output, display as _ipython_display\ndisplay = _quiet_display",
    )
    source = source.replace("plt.show()", "pass  # display suppressed by figure_generation.py")
    source = source.replace(
        "RR_REBUILD_REP_RATE_RELIANCE = False",
        f"RR_REBUILD_REP_RATE_RELIANCE = {bool(rebuild_rate_reliance)}",
    )
    source = source.replace(
        "merge_keys_pcv_eff = [\n    'partial_path', 'scope', 'model_bank', 'freq', 'block', 'bin_ms', 'mc', 'drop_thresh', 'is_loto'\n]",
        "merge_keys_pcv_eff = [\n    c for c in ['partial_path', 'scope', 'model_bank', 'freq', 'block', 'bin_ms', 'mc', 'drop_thresh', 'is_loto']\n    if c in runs_pcv.columns and c in curves_pcv.columns\n]",
    )
    return source


def _apply_paths_to_namespace(ns: dict[str, Any], paths: FigureGenerationPaths) -> None:
    ns.update(
        {
            "FIGURE_PATHS": paths,
            "REPO_ROOT": paths.repo_root,
            "DATA_DIR": paths.data_dir,
            "RESULTS_DIR_MANUSCRIPT": paths.discrimination_per_block_dir,
            "MANUSCRIPT_OUT_DIR": paths.output_dir,
            "SRC_DIR": paths.repo_root / "src",
            "DT_CSV": paths.detection_threshold_csv,
            "CV0_RESP_CSV": paths.cv0_responses_csv,
            "DETECTION_XLSX": paths.detection_behavior_xlsx,
            "TIME_PATTERNS_DIR": paths.time_patterns_dir,
            "FEAT_SIM_CSV": paths.cv0_responses_csv,
            "RESULTS_DIR": paths.discrimination_per_block_dir,
            "DEFAULT_SLACK_DIR": paths.discrimination_slack_dir,
            "OUT_DIR": paths.output_dir,
            "DISC_PER_BLOCK_DIR": paths.discrimination_per_block_dir,
            "DISC_PERCV_DIR": paths.discrimination_per_point_dir,
            "SLACK_DIR": paths.discrimination_slack_dir,
            "CV_ORDER": CV_ORDER,
            "FREQ_ORDER": FREQ_ORDER,
            "THRESH_ORDER": THRESH_ORDER,
        }
    )


def _copy_rate_reliance_cache(paths: FigureGenerationPaths) -> list[Path]:
    """Seed the notebook cache with existing rate-reliance intermediates."""

    written: list[Path] = []
    if not paths.legacy_figure_dir.exists():
        return written
    patterns = (
        "exploratory_rate_reliance_*",
        "rate_reliance_*",
        "suppfig03_rate_reliance_stats_summary.csv",
    )
    for pattern in patterns:
        for src in paths.legacy_figure_dir.glob(pattern):
            if not src.is_file():
                continue
            name = src.name.replace("exploratory_rate_reliance_", "rate_reliance_")
            dst = paths.notebook_cache_dir / name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
                written.append(dst)
    return written


def _publish_legacy_output_aliases(paths: FigureGenerationPaths) -> list[Path]:
    """Copy legacy notebook output names to manuscript-numbered output names."""

    written: list[Path] = []
    for legacy_stem, manuscript_stem in _LEGACY_OUTPUT_STEM_ALIASES.items():
        for ext in ("png", "pdf", "svg"):
            legacy_path = paths.output_dir / f"{legacy_stem}.{ext}"
            if not legacy_path.exists():
                continue
            manuscript_path = paths.output_dir / f"{manuscript_stem}.{ext}"
            if (
                not manuscript_path.exists()
                or legacy_path.stat().st_mtime > manuscript_path.stat().st_mtime
                or legacy_path.stat().st_size != manuscript_path.stat().st_size
            ):
                shutil.copy2(legacy_path, manuscript_path)
                written.append(manuscript_path)
            legacy_path.unlink()

    for legacy_name, manuscript_name in _LEGACY_OUTPUT_FILE_ALIASES.items():
        legacy_path = paths.output_dir / legacy_name
        if not legacy_path.exists():
            continue
        manuscript_path = paths.notebook_cache_dir / manuscript_name
        if (
            not manuscript_path.exists()
            or legacy_path.stat().st_mtime > manuscript_path.stat().st_mtime
            or legacy_path.stat().st_size != manuscript_path.stat().st_size
        ):
            shutil.copy2(legacy_path, manuscript_path)
            written.append(manuscript_path)
        legacy_path.unlink()
    return written


def extract_static_manuscript_figures(
    manuscript_docx: str | Path,
    paths: FigureGenerationPaths,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Extract static schematic figures from the manuscript DOCX into the figure folder."""

    docx = Path(manuscript_docx)
    if not docx.exists():
        raise FileNotFoundError(f"Manuscript DOCX not found: {docx}")
    written: list[Path] = []
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.static_figure_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx) as zf:
        members = set(zf.namelist())
        for spec in MANUSCRIPT_FIGURE_SPECS:
            if spec.static_media is None:
                continue
            if spec.static_media not in members:
                continue
            suffix = Path(spec.static_media).suffix.lower() or ".png"
            payload = zf.read(spec.static_media)
            for stem in spec.stems:
                for folder in (paths.static_figure_dir, paths.output_dir):
                    out = folder / f"{stem}{suffix}"
                    if overwrite or not out.exists():
                        out.write_bytes(payload)
                        written.append(out)
    return written


def copy_static_figure_assets(paths: FigureGenerationPaths, *, overwrite: bool = False) -> list[Path]:
    """Copy cached static schematic figures into the manuscript figure output folder."""

    written: list[Path] = []
    if not paths.static_figure_dir.exists():
        return written
    for spec in MANUSCRIPT_FIGURE_SPECS:
        if spec.static_media is None:
            continue
        for output_stem in spec.stems:
            for source_stem in spec.static_source_stems:
                matches = sorted(paths.static_figure_dir.glob(f"{source_stem}.*"))
                for src in matches:
                    dst = paths.output_dir / f"{output_stem}{src.suffix}"
                    if overwrite or not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(src, dst)
                        written.append(dst)
    return written


def _static_source_candidates(spec: ManuscriptFigureSpec, paths: FigureGenerationPaths) -> list[Path]:
    candidates: list[Path] = []
    for stem in spec.static_source_stems:
        for ext in ("png", "jpg", "jpeg", "tif", "tiff", "webp", "pdf", "svg"):
            path = paths.static_figure_dir / f"{stem}.{ext}"
            if path.exists():
                candidates.append(path)
    return candidates


def collect_manuscript_figure_outputs(
    paths: FigureGenerationPaths | None = None,
) -> dict[str, list[Path]]:
    """Collect generated manuscript figure files keyed by manuscript figure key."""

    paths = paths or setup_figure_generation()
    outputs: dict[str, list[Path]] = {}
    for spec in MANUSCRIPT_FIGURE_SPECS:
        found: list[Path] = []
        for stem in spec.stems:
            for ext in ("png", "pdf", "svg"):
                path = paths.output_dir / f"{stem}.{ext}"
                if path.exists():
                    found.append(path)
        outputs[spec.key] = found
    return outputs


def run_legacy_manuscript_figure_cells(
    paths: FigureGenerationPaths | None = None,
    *,
    rebuild_rate_reliance: bool = False,
    quiet: bool = True,
) -> dict[str, list[Path]]:
    """Run the old figure-generation notebook code from a clean Python entry point."""

    paths = paths or setup_figure_generation()
    ensure_src_on_path(paths.repo_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.notebook_cache_dir.mkdir(parents=True, exist_ok=True)
    paths.static_figure_dir.mkdir(parents=True, exist_ok=True)
    if not rebuild_rate_reliance:
        _copy_rate_reliance_cache(paths)

    sources = _legacy_cell_sources()
    ns: dict[str, Any] = {
        "__name__": "__figure_generation_notebook__",
        "_quiet_display": _quiet_display,
    }
    _apply_paths_to_namespace(ns, paths)

    old_cwd = Path.cwd()
    os.chdir(paths.repo_root)
    try:
        for cell_idx in _LEGACY_CELL_ORDER:
            source = _normalize_legacy_source(
                sources[cell_idx],
                rebuild_rate_reliance=rebuild_rate_reliance,
                use_notebook_cache=True,
            )
            _apply_paths_to_namespace(ns, paths)
            code = compile(source, f"<06_FigureGeneration cell {cell_idx}>", "exec")
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(code, ns)
            else:
                exec(code, ns)
            _apply_paths_to_namespace(ns, paths)
    finally:
        os.chdir(old_cwd)
    return collect_manuscript_figure_outputs(paths)


def generate_all_manuscript_figures(
    paths: FigureGenerationPaths | None = None,
    *,
    manuscript_docx: str | Path | None = None,
    run_dynamic: bool = True,
    rebuild_rate_reliance: bool = False,
    quiet: bool = True,
) -> dict[str, list[Path]]:
    """Generate/copy all manuscript figures into ``results/figures``."""

    paths = paths or setup_figure_generation()
    if manuscript_docx is not None:
        extract_static_manuscript_figures(manuscript_docx, paths)
    else:
        copy_static_figure_assets(paths)
    if run_dynamic:
        run_legacy_manuscript_figure_cells(
            paths,
            rebuild_rate_reliance=rebuild_rate_reliance,
            quiet=quiet,
        )
    return collect_manuscript_figure_outputs(paths)


def display_manuscript_figure(
    key: str,
    outputs: Mapping[str, list[Path]] | None = None,
    paths: FigureGenerationPaths | None = None,
    *,
    width: int | None = None,
) -> None:
    """Display one manuscript figure in a notebook and return the displayed path."""

    from IPython.display import Image, Markdown, display

    paths = paths or setup_figure_generation()
    spec = get_manuscript_figure_spec(key)
    outputs = outputs or collect_manuscript_figure_outputs(paths)
    candidates = [p for p in outputs.get(spec.key, []) if p.suffix.lower() == ".png"]
    if not candidates:
        display(Markdown(f"**Missing output for {spec.label}.** Expected one of: {', '.join(spec.stems)}"))
        return None
    path = candidates[0]
    display(Image(filename=str(path), width=width))
    return path


def caption_markdown(spec: ManuscriptFigureSpec) -> str:
    """Return notebook markdown containing the manuscript figure caption."""

    return f"## {spec.label}: {spec.title}\n\n{spec.caption}"


# ---------------------------------------------------------------------------
# Clean, item-by-item API used by ``06_FigureGeneration.ipynb``.
#
# The historical figure notebook is still embedded above so we can reuse the
# exact manuscript plotting logic while the cleaned notebook stays readable.
# These wrappers keep generated manuscript outputs separated from reusable
# notebook caches and avoid copying old figures forward.

EXTERNAL_DESIGN_ONLY_KEYS: tuple[str, ...] = (
    "figure_1",
    "figure_2",
    "supplementary_figure_1",
)

_DETECTION_GROUP_KEYS = {
    "figure_3",
    "supplementary_figure_2",
    "supplementary_figure_3",
    "supplementary_figure_4",
    "supplementary_figure_7",
    "supplementary_figure_8",
}
_FEATURE_GROUP_KEYS = {
    "supplementary_figure_5",
    "supplementary_figure_6",
}
_RATE_RELIANCE_GROUP_KEYS = {"figure_5"}
_DIRECT_DISCRIMINATION_KEYS = {"figure_4", "supplementary_figure_9"}
_GENERATED_GROUPS: set[str] = set()

_DYNAMIC_MANUSCRIPT_STEMS: tuple[str, ...] = (
    "fig03_detection_threshold_behavioral_matching",
    "fig04_behavioral_psychometric_curve_matching",
    "fig05_rate_reliance",
    "suppfig02_spike_count_threshold_amplitudes",
    "suppfig03_threshold_spike_counts",
    "suppfig04_cv0_based_threshold_spike_counts",
    "suppfig05_single_repetition_feature_engineering",
    "suppfig06_combined_normalized_evidence_curves",
    "suppfig07_sub_refractory_inter_pulse_intervals",
    "suppfig08_detection_mixture_weights",
    "suppfig09_periodicity_discrimination_mixture_weights",
)
_STATIC_MANUSCRIPT_STEMS: tuple[str, ...] = tuple(
    stem
    for spec in MANUSCRIPT_FIGURE_SPECS
    if spec.static_media is not None
    for stem in spec.stems
)
_MANUSCRIPT_TABLE_NAMES: tuple[str, ...] = (
    "table_fig03_detection_threshold_matching_rmse.csv",
    "table_fig04_periodicity_discrimination_model_rmse.csv",
    "fig05_rate_reliance_stats_summary.csv",
    "suppfig08_detection_mixture_weights.csv",
    "suppfig08_detection_mixture_weights_individual_regimes.csv",
)
_ALLOWED_MANUSCRIPT_OUTPUT_NAMES: frozenset[str] = frozenset(
    [f"{stem}.{ext}" for stem in _DYNAMIC_MANUSCRIPT_STEMS for ext in ("png", "pdf", "svg")]
    + [f"{stem}.{ext}" for stem in _STATIC_MANUSCRIPT_STEMS for ext in ("png", "jpg", "jpeg", "tif", "tiff", "webp", "pdf", "svg")]
)

_LEGACY_OUTPUT_STEM_ALIASES: Mapping[str, str] = {
    "fig09_detection_threshold_behavioral_matching": "fig03_detection_threshold_behavioral_matching",
    "fig10_behavioral_psychometric_curve_matching": "fig04_behavioral_psychometric_curve_matching",
    "suppfig03_rate_reliance": "fig05_rate_reliance",
    "fig03_spike_count_threshold_amplitudes": "suppfig02_spike_count_threshold_amplitudes",
    "fig04_threshold_spike_counts": "suppfig03_threshold_spike_counts",
    "fig06_cv0_based_threshold_spike_counts": "suppfig04_cv0_based_threshold_spike_counts",
    "fig07_single_repetition_feature_engineering": "suppfig05_single_repetition_feature_engineering",
    "fig08_combined_normalized_evidence_curves": "suppfig06_combined_normalized_evidence_curves",
    "suppfig02_sub_refractory_inter_pulse_intervals": "suppfig07_sub_refractory_inter_pulse_intervals",
    "suppfig05_detection_mixture_weights": "suppfig08_detection_mixture_weights",
    "fig11_block_shared_regime_weights": "suppfig09_periodicity_discrimination_mixture_weights",
}

_LEGACY_OUTPUT_FILE_ALIASES: Mapping[str, str] = {
    "table_fig09_detection_threshold_matching_rmse.csv": "table_fig03_detection_threshold_matching_rmse.csv",
    "table_periodicity_discrimination_model_rmse.csv": "table_fig04_periodicity_discrimination_model_rmse.csv",
    "suppfig03_rate_reliance_stats_summary.csv": "fig05_rate_reliance_stats_summary.csv",
    "suppfig05_detection_mixture_weights.csv": "suppfig08_detection_mixture_weights.csv",
    "suppfig05_detection_mixture_weights_individual_regimes.csv": "suppfig08_detection_mixture_weights_individual_regimes.csv",
}


def reset_manuscript_generation_state() -> None:
    """Clear in-kernel group-generation cache used by the figure notebook."""

    _GENERATED_GROUPS.clear()


@dataclass(frozen=True)
class ManuscriptGenerationResult:
    """Return value for one generated manuscript item."""

    key: str
    files: tuple[Path, ...]
    message: str = ""


def _normalize_legacy_source(
    source: str,
    *,
    rebuild_rate_reliance: bool,
    use_notebook_cache: bool,
) -> str:
    """Patch notebook-era source so it uses cleaned manuscript paths and caches."""

    source = source.replace(
        "from IPython.display import display",
        "from IPython.display import display as _ipython_display\ndisplay = _quiet_display",
    )
    source = source.replace(
        "from IPython.display import HTML, clear_output, display",
        "from IPython.display import HTML, clear_output, display as _ipython_display\ndisplay = _quiet_display",
    )
    source = source.replace("plt.show()", "pass  # display suppressed by figure_generation.py")
    source = source.replace(
        "RR_REBUILD_REP_RATE_RELIANCE = False",
        f"RR_REBUILD_REP_RATE_RELIANCE = {bool(rebuild_rate_reliance or not use_notebook_cache)}",
    )
    source = f"USE_NB_CACHE = {bool(use_notebook_cache)}\n" + source
    source = source.replace("ABS_REF_MS = 1.4", "ABS_REF_MS = 1.10")
    source = source.replace("n_short = int(np.sum(isi_ms < abs_ref_ms))", "n_short = int(np.sum(isi_ms <= abs_ref_ms))")
    source = source.replace(
        "isi_cache = DATA_CACHE_DIR / 'cache_detection_isi_rep_stats.csv'",
        "isi_cache = DATA_CACHE_DIR / f\"cache_detection_isi_rep_stats_absref_{ABS_REF_MS:.2f}ms.csv\"",
    )
    source = source.replace(
        "isi_cache = FIGURE_PATHS.notebook_cache_dir / 'cache_detection_isi_rep_stats.csv'",
        "isi_cache = FIGURE_PATHS.notebook_cache_dir / f\"cache_detection_isi_rep_stats_absref_{ABS_REF_MS:.2f}ms.csv\"",
    )
    source = source.replace(
        "ax.set_xlabel('Fraction of Intervals < 1.4 ms')",
        "ax.set_xlabel(f'Fraction of Intervals <= {ABS_REF_MS:.2f} ms')",
    )
    source = source.replace(
        "ax_f_rate.set_ylim(s_min - 0.06 * s_span, s_max + 0.10 * s_span)",
        "ax_f_rate.set_ylim(s_min - 0.08 * s_span, s_max + 0.36 * s_span)",
    )
    source = source.replace(
        "b_hi = b_max + 0.75 * b_span",
        "b_hi = b_max + 1.05 * b_span",
    )
    source = source.replace(
        "ax_f_time.set_ylim(li_min - 0.06 * li_span, li_max + 0.06 * li_span)",
        "ax_f_time.set_ylim(li_min - 0.08 * li_span, li_max + 0.30 * li_span)",
    )
    source = source.replace(
        "ax_f_time_b.set_ylim(lb_min - 0.15 * lb_span, lb_max + 0.75 * lb_span)",
        "ax_f_time_b.set_ylim(lb_min - 0.15 * lb_span, lb_max + 1.05 * lb_span)",
    )
    source = source.replace(
        "if isi_cache.exists():\n    isi_rep_stats = pd.read_csv(isi_cache)\nelse:",
        "if USE_NB_CACHE and isi_cache.exists():\n    isi_rep_stats = pd.read_csv(isi_cache)\nelse:",
    )
    source = source.replace(
        "if CACHE_SINGLE_JSON.exists() and CACHE_SINGLE_META_JSON.exists():",
        "if USE_NB_CACHE and CACHE_SINGLE_JSON.exists() and CACHE_SINGLE_META_JSON.exists():",
    )
    source = source.replace(
        "if cache_csv.exists() and cache_meta_json.exists():",
        "if USE_NB_CACHE and cache_csv.exists() and cache_meta_json.exists():",
    )
    detection_fit_block_old = """single_regime_fits = _fit_single_regime_detection_models(behavior_targets, regime_targets, THRESH_ORDER)
best_single_thresh = min(THRESH_ORDER, key=lambda t: single_regime_fits[t]['mean_rmse'])
best_single_fit = single_regime_fits[best_single_thresh]

mixture_fit_per_cv_reg = _fit_detection_mixture(
    behavior_targets,
    regime_targets,
    THRESH_ORDER,
    x_by_freq,
    mode='per_cv',
    use_entmax=DETECTION_EXPLORATORY_REG_USE_ENTMAX,
    entmax_alpha=DETECTION_EXPLORATORY_REG_ENTMAX_ALPHA,
    weight_kl=DETECTION_EXPLORATORY_REG_WEIGHT_KL,
    weight_entropy_floor_k=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K,
    weight_entropy_floor_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA,
    weight_max_cap=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP,
    weight_max_cap_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA,
    weight_smooth_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA,
)

mixture_fit_points_per_cv_reg = _fit_detection_mixture(
    behavior_targets_points,
    regime_targets_points,
    THRESH_ORDER,
    x_by_freq_points,
    mode='per_cv',
    use_entmax=DETECTION_EXPLORATORY_REG_USE_ENTMAX,
    entmax_alpha=DETECTION_EXPLORATORY_REG_ENTMAX_ALPHA,
    weight_kl=DETECTION_EXPLORATORY_REG_WEIGHT_KL,
    weight_entropy_floor_k=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K,
    weight_entropy_floor_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA,
    weight_max_cap=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP,
    weight_max_cap_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA,
    weight_smooth_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA,
)
"""
    detection_fit_block_new = """import pickle

def _detection_cache_file_meta(path):
    path = Path(path)
    return {
        'name': path.name,
        'size': int(path.stat().st_size),
        'mtime_ns': int(path.stat().st_mtime_ns),
    }

detection_fit_cache_path = FIGURE_PATHS.notebook_cache_dir / 'detection_behavioral_matching_fit_cache.pkl'
detection_fit_cache_meta = {
    'cache_version': 1,
    'kind': 'detection_behavioral_matching_fit_cache',
    'detection_threshold_csv': _detection_cache_file_meta(DT_CSV),
    'cv0_responses_csv': _detection_cache_file_meta(CV0_RESP_CSV),
    'behavioral_xlsx': _detection_cache_file_meta(DETECTION_XLSX),
    'cutline': int(cutline),
    'plot_freqs': [float(f) for f in plot_freqs],
    'curve_grid_n': int(len(curve_grid)),
    'datapoint_cvs': [float(x) for x in DETECTION_DATAPOINT_CVS],
    'reg_params': {
        'use_entmax': bool(DETECTION_EXPLORATORY_REG_USE_ENTMAX),
        'entmax_alpha': float(DETECTION_EXPLORATORY_REG_ENTMAX_ALPHA),
        'weight_kl': float(DETECTION_EXPLORATORY_REG_WEIGHT_KL),
        'entropy_floor_k': float(DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K),
        'entropy_floor_lambda': float(DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA),
        'weight_max_cap': float(DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP),
        'weight_max_cap_lambda': float(DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA),
        'weight_smooth_lambda': float(DETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA),
    },
}

use_detection_fit_cache = False
if USE_NB_CACHE and detection_fit_cache_path.exists():
    try:
        with open(detection_fit_cache_path, 'rb') as _f:
            detection_fit_payload = pickle.load(_f)
        use_detection_fit_cache = detection_fit_payload.get('meta') == detection_fit_cache_meta
    except Exception:
        use_detection_fit_cache = False

if use_detection_fit_cache:
    single_regime_fits = detection_fit_payload['single_regime_fits']
    best_single_thresh = detection_fit_payload['best_single_thresh']
    best_single_fit = detection_fit_payload['best_single_fit']
    mixture_fit_per_cv_reg = detection_fit_payload['mixture_fit_per_cv_reg']
    mixture_fit_points_per_cv_reg = detection_fit_payload['mixture_fit_points_per_cv_reg']
else:
    single_regime_fits = _fit_single_regime_detection_models(behavior_targets, regime_targets, THRESH_ORDER)
    best_single_thresh = min(THRESH_ORDER, key=lambda t: single_regime_fits[t]['mean_rmse'])
    best_single_fit = single_regime_fits[best_single_thresh]

    mixture_fit_per_cv_reg = _fit_detection_mixture(
        behavior_targets,
        regime_targets,
        THRESH_ORDER,
        x_by_freq,
        mode='per_cv',
        use_entmax=DETECTION_EXPLORATORY_REG_USE_ENTMAX,
        entmax_alpha=DETECTION_EXPLORATORY_REG_ENTMAX_ALPHA,
        weight_kl=DETECTION_EXPLORATORY_REG_WEIGHT_KL,
        weight_entropy_floor_k=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K,
        weight_entropy_floor_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA,
        weight_max_cap=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP,
        weight_max_cap_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA,
        weight_smooth_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA,
    )

    mixture_fit_points_per_cv_reg = _fit_detection_mixture(
        behavior_targets_points,
        regime_targets_points,
        THRESH_ORDER,
        x_by_freq_points,
        mode='per_cv',
        use_entmax=DETECTION_EXPLORATORY_REG_USE_ENTMAX,
        entmax_alpha=DETECTION_EXPLORATORY_REG_ENTMAX_ALPHA,
        weight_kl=DETECTION_EXPLORATORY_REG_WEIGHT_KL,
        weight_entropy_floor_k=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_K,
        weight_entropy_floor_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_ENTROPY_FLOOR_LAMBDA,
        weight_max_cap=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP,
        weight_max_cap_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_MAX_CAP_LAMBDA,
        weight_smooth_lambda=DETECTION_EXPLORATORY_REG_WEIGHT_SMOOTH_LAMBDA,
    )

    try:
        detection_fit_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detection_fit_cache_path, 'wb') as _f:
            pickle.dump({
                'meta': detection_fit_cache_meta,
                'single_regime_fits': single_regime_fits,
                'best_single_thresh': best_single_thresh,
                'best_single_fit': best_single_fit,
                'mixture_fit_per_cv_reg': mixture_fit_per_cv_reg,
                'mixture_fit_points_per_cv_reg': mixture_fit_points_per_cv_reg,
            }, _f)
    except Exception:
        pass
"""
    source = source.replace(detection_fit_block_old, detection_fit_block_new)
    source = source.replace(
        'loc="upper left",\n        handlelength=4.6,',
        'loc="upper left",\n        bbox_to_anchor=(0.02, 1.06),\n        borderaxespad=0.0,\n        handlelength=4.6,',
    )
    source = source.replace(
        'loc="upper right",\n        handlelength=4.6,',
        'loc="upper right",\n        bbox_to_anchor=(0.98, 1.06),\n        borderaxespad=0.0,\n        handlelength=4.6,',
    )
    source = source.replace(
        "loc='upper left',\n    handlelength=4.6,",
        "loc='upper left',\n    bbox_to_anchor=(0.02, 1.06),\n    borderaxespad=0.0,\n    handlelength=4.6,",
    )
    source = source.replace(
        "loc='upper right',\n    handlelength=4.6,",
        "loc='upper right',\n    bbox_to_anchor=(0.98, 1.06),\n    borderaxespad=0.0,\n    handlelength=4.6,",
    )

    old_data_cache_dir = "DATA_CACHE_DIR = REPO_ROOT / 'data' / '" + "notebook-cache'"
    replacements = {
        old_data_cache_dir:
            "DATA_CACHE_DIR = FIGURE_PATHS.notebook_cache_dir",
        "SLACK_DIR = REPO_ROOT / 'data' / 'discrimination' / 'slack'":
            "SLACK_DIR = FIGURE_PATHS.discrimination_slack_dir",
        "DEFAULT_SLACK_DIR = Path('data/discrimination/slack')":
            "DEFAULT_SLACK_DIR = FIGURE_PATHS.discrimination_slack_dir",
        "DEFAULT_SLACK_DIR = Path('../data/discrimination/slack')":
            "DEFAULT_SLACK_DIR = FIGURE_PATHS.discrimination_slack_dir",
        "RESULTS_DIR = Path('data/discrimination/per-block')":
            "RESULTS_DIR = FIGURE_PATHS.discrimination_per_block_dir",
        "RESULTS_DIR = Path('../data/discrimination/per-block')":
            "RESULTS_DIR = FIGURE_PATHS.discrimination_per_block_dir",
        "CACHE_DIR = REPO_ROOT / 'data' / 'discrimination' / 'per-block'":
            "CACHE_DIR = FIGURE_PATHS.discrimination_per_block_dir",
        "DISC_PER_BLOCK_DIR = REPO_ROOT / 'data' / 'discrimination' / 'manuscript' / 'per-block'":
            "DISC_PER_BLOCK_DIR = FIGURE_PATHS.discrimination_per_block_dir",
        "DISC_PERCV_DIR = REPO_ROOT / 'data' / 'discrimination' / 'manuscript' / 'per-point'":
            "DISC_PERCV_DIR = FIGURE_PATHS.discrimination_per_point_dir",
        "th = 'I_10' if 'I_10' in ths else (ths[0] if ths else None)":
            "th = 'I_25' if 'I_25' in ths else (ths[0] if ths else None)",
        "rep = 1 if 1 in reps_all else int(reps_all[0])":
            "rep = 0 if 0 in reps_all else int(reps_all[0])",
        "fig.suptitle('I_10 Single Repetition Feature Engineering', y=1.04)":
            "fig.suptitle('I_25 Single Repetition Feature Engineering', y=1.04)",
        "RATE_COLOR = '#E30613'  # I_10 rate color":
            "RATE_COLOR = '#9F040E'  # I_25 rate color",
        "TIMING_COLOR = '#0575AB'  # I_10 timing color":
            "TIMING_COLOR = '#074680'  # I_25 timing color",
        "fig_exp10.savefig(OUT_DIR / 'fig10_behavioral_psychometric_curve_matching.png', bbox_inches='tight')":
            "pass  # Figure 4 is regenerated directly from manuscript helpers.",
        "fig_exp10.savefig(OUT_DIR / 'fig10_behavioral_psychometric_curve_matching.pdf', bbox_inches='tight')":
            "pass  # Figure 4 is regenerated directly from manuscript helpers.",
        "isi_cache = MANUSCRIPT_OUT_DIR / 'cache_detection_isi_rep_stats.csv'":
            "isi_cache = FIGURE_PATHS.notebook_cache_dir / f\"cache_detection_isi_rep_stats_absref_{ABS_REF_MS:.2f}ms.csv\"",
        "OUT_DIR = Path('results/figures')":
            "OUT_DIR = FIGURE_PATHS.output_dir",
        "OUT_DIR = Path('../results/figures')":
            "OUT_DIR = FIGURE_PATHS.output_dir",
        "pd.read_csv(DATA_DIR / 'discrimination' / 'per-block' / 'merged_results.csv')":
            "pd.read_csv(FIGURE_PATHS.discrimination_per_block_dir / 'merged_results.csv')",
        "merge_keys_pcv_eff = [\n    'partial_path', 'scope', 'model_bank', 'freq', 'block', 'bin_ms', 'mc', 'drop_thresh', 'is_loto'\n]":
            "merge_keys_pcv_eff = [\n    c for c in ['partial_path', 'scope', 'model_bank', 'freq', 'block', 'bin_ms', 'mc', 'drop_thresh', 'is_loto']\n    if c in runs_pcv.columns and c in curves_pcv.columns\n]",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    source = source.replace(
        "    if 'is_fixed_thresh' not in pcv_results.columns:\n"
        "        pcv_results['is_fixed_thresh'] = False\n",
        "    if 'is_fixed_thresh' not in pcv_results.columns:\n"
        "        pcv_results['is_fixed_thresh'] = False\n"
        "    for _df, _scope, _flex in ((pb_results, 'per_block', 'shared'), (pcv_results, 'per_freq_block', 'per_point')):\n"
        "        if 'scope' not in _df.columns:\n"
        "            _df['scope'] = _scope\n"
        "        if 'drop_thresh' not in _df.columns:\n"
        "            _df['drop_thresh'] = 'NONE'\n"
        "        if 'is_loto' not in _df.columns:\n"
        "            _df['is_loto'] = False\n"
        "        if 'thresh_flex_mode' not in _df.columns:\n"
        "            _df['thresh_flex_mode'] = _flex\n"
        "        if 'rate_gate_mode' not in _df.columns:\n"
        "            _df['rate_gate_mode'] = 'delta_cv'\n",
    )

    # Rate-reliance intermediates are reusable notebook caches, not manuscript
    # outputs. Final figures/tables still go to OUT_DIR.
    source = source.replace(
        "OUT_DIR / 'exploratory_rate_reliance_",
        "FIGURE_PATHS.notebook_cache_dir / 'exploratory_rate_reliance_",
    )
    source = source.replace(
        'OUT_DIR / "exploratory_rate_reliance_',
        'FIGURE_PATHS.notebook_cache_dir / "exploratory_rate_reliance_',
    )
    source = source.replace("exploratory_rate_reliance_", "rate_reliance_")
    return source


def clean_manuscript_output_dir(
    paths: FigureGenerationPaths | None = None,
    *,
    dry_run: bool = False,
) -> list[Path]:
    """Remove non-manuscript generated files from ``results/figures``."""

    paths = paths or setup_figure_generation()
    if not paths.output_dir.exists():
        return []
    removed: list[Path] = []
    for path in paths.output_dir.iterdir():
        if not path.is_file() or path.name in _ALLOWED_MANUSCRIPT_OUTPUT_NAMES:
            continue
        removed.append(path)
        if not dry_run:
            path.unlink()
    return removed


def _run_legacy_cell_subset(
    cell_indices: Iterable[int],
    paths: FigureGenerationPaths,
    *,
    rebuild_rate_reliance: bool = False,
    use_notebook_cache: bool = True,
    quiet: bool = True,
) -> None:
    """Run selected embedded legacy cells in a clean namespace."""

    ensure_src_on_path(paths.repo_root)
    sources = _legacy_cell_sources()
    ns: dict[str, Any] = {
        "__name__": "__figure_generation_item__",
        "_quiet_display": _quiet_display,
    }
    _apply_paths_to_namespace(ns, paths)

    old_cwd = Path.cwd()
    os.chdir(paths.repo_root)
    try:
        for cell_idx in cell_indices:
            source = _normalize_legacy_source(
                sources[int(cell_idx)],
                rebuild_rate_reliance=rebuild_rate_reliance,
                use_notebook_cache=use_notebook_cache,
            )
            _apply_paths_to_namespace(ns, paths)
            code = compile(source, f"<06_FigureGeneration cell {cell_idx}>", "exec")
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(code, ns)
            else:
                exec(code, ns)
            _apply_paths_to_namespace(ns, paths)
    finally:
        os.chdir(old_cwd)
        plt.close("all")
    _publish_legacy_output_aliases(paths)


def _load_manuscript_discrimination(paths: FigureGenerationPaths):
    from discrimination.discrimination_core import (
        DiscriminationDataConfig,
        load_discrimination_data,
    )

    return load_discrimination_data(
        DiscriminationDataConfig(repo_root=paths.repo_root)
    )


def _write_periodicity_rmse_table(paths: FigureGenerationPaths) -> Path:
    from discrimination.discrimination_core import model_comparison_pivot

    data = _load_manuscript_discrimination(paths)
    table = model_comparison_pivot(data, include_block_means=True).round(4)
    flat = table.copy()
    flat.columns = [f"{analysis} | {model}" for analysis, model in flat.columns]
    out = flat.reset_index()
    path = paths.notebook_cache_dir / "table_fig04_periodicity_discrimination_model_rmse.csv"
    out.to_csv(path, index=False)
    return path


def _generate_direct_discrimination_item(key: str, paths: FigureGenerationPaths) -> list[Path]:
    from discrimination.discrimination_plot import (
        plot_psychometric_fits,
        plot_regime_mixture_comparison,
    )

    data = _load_manuscript_discrimination(paths)
    written: list[Path] = []
    if key == "figure_4":
        fig = plot_psychometric_fits(
            data,
            include_per_point=True,
        )
        written.extend(save_manuscript_figure(fig, "fig04_behavioral_psychometric_curve_matching", paths, dpi=300))
        plt.close(fig)
        written.append(_write_periodicity_rmse_table(paths))
    elif key == "supplementary_figure_9":
        fig = plot_regime_mixture_comparison(data)
        written.extend(save_manuscript_figure(fig, "suppfig09_periodicity_discrimination_mixture_weights", paths, dpi=300))
        plt.close(fig)
    else:
        raise KeyError(key)
    return written


def _collect_key_outputs(key: str, paths: FigureGenerationPaths) -> list[Path]:
    spec = get_manuscript_figure_spec(key)
    found: list[Path] = []
    for stem in spec.stems:
        for ext in ("png", "pdf", "svg"):
            path = paths.output_dir / f"{stem}.{ext}"
            if path.exists():
                found.append(path)
    return found


def _collect_static_key_outputs(key: str, paths: FigureGenerationPaths) -> list[Path]:
    spec = get_manuscript_figure_spec(key)
    found: list[Path] = []
    for stem in spec.static_source_stems:
        for ext in ("png", "jpg", "jpeg", "tif", "tiff", "webp"):
            path = paths.static_figure_dir / f"{stem}.{ext}"
            if path.exists():
                found.append(path)
    return found


def generate_manuscript_item(
    key: str,
    paths: FigureGenerationPaths | None = None,
    *,
    rebuild_rate_reliance: bool = False,
    force: bool = False,
    use_notebook_cache: bool = False,
    quiet: bool = True,
) -> ManuscriptGenerationResult:
    """Generate one manuscript figure/table group and return its files.

    ``use_notebook_cache`` is reserved for intermediate calculation caches. It
    never skips figure rendering because an output image already exists.
    """

    paths = paths or setup_figure_generation()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.notebook_cache_dir.mkdir(parents=True, exist_ok=True)
    spec = get_manuscript_figure_spec(key)

    if spec.key in EXTERNAL_DESIGN_ONLY_KEYS:
        copy_static_figure_assets(paths)
        static_files = _collect_key_outputs(spec.key, paths)
        if not static_files:
            static_files = _collect_static_key_outputs(spec.key, paths)
        return ManuscriptGenerationResult(
            key=spec.key,
            files=tuple(static_files),
            message=(
                f"{spec.label} is an external design composite. The final static image "
                "is read from notebooks/cache/static-figures rather than regenerated here."
            ),
        )

    if spec.key in _DETECTION_GROUP_KEYS:
        if force or "detection" not in _GENERATED_GROUPS:
            _run_legacy_cell_subset((2, 4), paths, use_notebook_cache=use_notebook_cache, quiet=quiet)
            _GENERATED_GROUPS.add("detection")
    elif spec.key in _FEATURE_GROUP_KEYS:
        if force or "feature" not in _GENERATED_GROUPS:
            _run_legacy_cell_subset((2, 6), paths, use_notebook_cache=use_notebook_cache, quiet=quiet)
            _GENERATED_GROUPS.add("feature")
    elif spec.key in _RATE_RELIANCE_GROUP_KEYS:
        if force or rebuild_rate_reliance or "rate_reliance" not in _GENERATED_GROUPS:
            if not rebuild_rate_reliance:
                _copy_rate_reliance_cache(paths)
            _run_legacy_cell_subset(
                (8, 9, 10, 13, 15, 16, 17),
                paths,
                rebuild_rate_reliance=rebuild_rate_reliance,
                use_notebook_cache=use_notebook_cache,
                quiet=quiet,
            )
            _GENERATED_GROUPS.add("rate_reliance")
    elif spec.key in _DIRECT_DISCRIMINATION_KEYS:
        _generate_direct_discrimination_item(spec.key, paths)
    else:
        raise KeyError(f"No generator registered for {spec.key!r}.")

    clean_manuscript_output_dir(paths)
    return ManuscriptGenerationResult(spec.key, tuple(_collect_key_outputs(spec.key, paths)))


def run_legacy_manuscript_figure_cells(
    paths: FigureGenerationPaths | None = None,
    *,
    rebuild_rate_reliance: bool = False,
    quiet: bool = True,
) -> dict[str, list[Path]]:
    """Compatibility wrapper that regenerates dynamic manuscript items only."""

    paths = paths or setup_figure_generation()
    for spec in MANUSCRIPT_FIGURE_SPECS:
        if spec.key in EXTERNAL_DESIGN_ONLY_KEYS:
            continue
        generate_manuscript_item(
            spec.key,
            paths,
            rebuild_rate_reliance=rebuild_rate_reliance,
            force=False,
            quiet=quiet,
        )
    return collect_manuscript_figure_outputs(paths)


def generate_all_manuscript_figures(
    paths: FigureGenerationPaths | None = None,
    *,
    manuscript_docx: str | Path | None = None,
    run_dynamic: bool = True,
    rebuild_rate_reliance: bool = False,
    quiet: bool = True,
    include_external_design: bool = False,
) -> dict[str, list[Path]]:
    """Generate all code-derived manuscript figures without copying stale composites."""

    paths = paths or setup_figure_generation()
    clean_manuscript_output_dir(paths)
    if include_external_design:
        if manuscript_docx is not None:
            extract_static_manuscript_figures(manuscript_docx, paths)
        else:
            copy_static_figure_assets(paths)
    if run_dynamic:
        run_legacy_manuscript_figure_cells(
            paths,
            rebuild_rate_reliance=rebuild_rate_reliance,
            quiet=quiet,
        )
    return collect_manuscript_figure_outputs(paths)


def display_manuscript_figure(
    key: str,
    outputs: Mapping[str, list[Path]] | None = None,
    paths: FigureGenerationPaths | None = None,
    *,
    width: int | None = None,
) -> Path | None:
    """Display a generated manuscript figure or a note for external composites."""

    from IPython.display import Image, Markdown, display

    paths = paths or setup_figure_generation()
    spec = get_manuscript_figure_spec(key)
    if spec.key in EXTERNAL_DESIGN_ONLY_KEYS:
        candidates = [p for p in _collect_static_key_outputs(spec.key, paths) if p.suffix.lower() == ".png"]
        if not candidates:
            display(Markdown(
                f"**Missing cached static image for {spec.label}.** Expected one of: {', '.join(spec.static_source_stems)} "
                "in `notebooks/cache/static-figures/`."
            ))
            return None
        path = candidates[0]
        display(Image(filename=str(path), width=width))
        return path

    outputs = outputs or collect_manuscript_figure_outputs(paths)
    candidates = [p for p in outputs.get(spec.key, []) if p.suffix.lower() == ".png"]
    if not candidates:
        candidates = [p for p in _collect_key_outputs(spec.key, paths) if p.suffix.lower() == ".png"]
    if not candidates:
        display(Markdown(f"**Missing output for {spec.label}.** Expected one of: {', '.join(spec.stems)}"))
        return
    path = candidates[0]
    display(Image(filename=str(path), width=width))
    return


def display_manuscript_item(
    key: str,
    result: ManuscriptGenerationResult | None = None,
    paths: FigureGenerationPaths | None = None,
    *,
    width: int | None = None,
) -> None:
    """Generate-notebook-friendly alias for ``display_manuscript_figure``."""

    if result is not None and result.message:
        from IPython.display import Markdown, display

        display(Markdown(result.message))
    display_manuscript_figure(key, paths=paths, width=width)


def display_table(
    table_name: str,
    paths: FigureGenerationPaths | None = None,
    *,
    max_rows: int = 30,
) -> None:
    """Display a reviewer-facing CSV table with plain pandas formatting."""

    import pandas as pd
    from IPython.display import Markdown, display

    paths = paths or setup_figure_generation()
    aliases = {
        "detection_threshold_matching": "table_fig03_detection_threshold_matching_rmse.csv",
        "periodicity_model_rmse": "table_fig04_periodicity_discrimination_model_rmse.csv",
        "rate_reliance_stats": "fig05_rate_reliance_stats_summary.csv",
        "detection_mixture_weights": "suppfig08_detection_mixture_weights.csv",
        "detection_mixture_individual_regimes": "suppfig08_detection_mixture_weights_individual_regimes.csv",
    }
    filename = aliases.get(str(table_name), str(table_name))
    path = paths.notebook_cache_dir / filename
    if not path.exists():
        display(Markdown(f"**Missing table:** `{filename}`"))
        return
    df = pd.read_csv(path)
    if max_rows is not None and len(df) > int(max_rows):
        display(df.head(int(max_rows)))
        display(Markdown(f"Showing first {int(max_rows)} of {len(df)} rows."))
    else:
        display(df)
    return


display_manuscript_table = display_table
