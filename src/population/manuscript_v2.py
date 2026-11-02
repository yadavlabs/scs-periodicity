"""Reproducible manuscript v2 figures and descriptive comparisons.

No model fitting is performed. Independently optimized cue controls are kept
distinct from the primary hybrid's cue-removal projections.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, FormatStrFormatter

REGIMES = ["I_first", "I_volley5", "I_volley10", "I_5", "I_10", "I_15", "I_25", "I_50", "I_all"]
FREQS = [20, 50, 100, 200]
COLORS = dict(zip(REGIMES, plt.cm.viridis(np.linspace(.02, .98, 9))))
CUES = {"hybrid": "#6a3d9a", "rate": "#e30613", "timing": "#0575ab"}
SCOPES = {
    "primary": ("per_frequency", "per_block"),
    "crossed": ("per_block", "per_frequency"),
    "per_panel": ("per_panel", "per_panel"),
}
LABELS = {
    "primary": "Per-frequency mixtures + per-block evidence weighting",
    "crossed": "Per-block mixtures + per-frequency evidence weighting",
    "per_panel": "Per-panel mixtures + per-panel evidence weighting",
}


def label(regime):
    return r"$I_{" + ("max" if regime == "I_all" else regime[2:]) + "}$"


def select(frame, scope="primary", family="hybrid", split=None):
    ws, rs = SCOPES[scope]
    selected = frame[(frame.weight_scope == ws) & (frame.readout_scope == rs)
                     & (frame.state_mode == "mixture") & (frame.model_family == family)
                     & (frame.rate_gate_mode == "none") & (frame.mixture_semantics == "latent_trial")]
    if split is not None:
        selected = selected[selected.split == split]
    if selected.empty:
        raise ValueError(f"Missing {scope}/{family}/{split} results")
    return selected.copy()


def choice(cv, p, block):
    return np.where(np.isclose(cv, 0 if block == "std0" else 1), 1 - np.asarray(p), p)


def cv_ticks(ax):
    """Eleven tick marks, with labels only at 0.0, 0.2, ..., 1.0."""
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f"{value:.1f}" if np.isclose(value * 5, round(value * 5)) else ""))


def cue_line_style(family):
    return dict(lw=3.0 if family == "hybrid" else 1.8,
                alpha=1.0 if family == "hybrid" else .75,
                zorder=5 if family == "hybrid" else 4)


def tv(a, b):
    """Total variation: proportion of mixture mass that must be reassigned."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.allclose(a.sum(axis=-1), 1) or not np.allclose(b.sum(axis=-1), 1):
        raise ValueError("Mixture weights must sum to one")
    return .5 * np.abs(a - b).sum(axis=-1)


def integrated_share(frame, keys, value):
    rows = []
    for key, sub in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        sub = sub.sort_values("cv")
        if sub.cv.duplicated().any() or not np.allclose(sub.cv.to_numpy(), np.linspace(0, 1, 11)):
            raise ValueError("Expected exactly eleven CVs per integration group")
        integral = np.trapezoid(sub[value], sub.cv) if hasattr(np, "trapezoid") else np.trapz(sub[value], sub.cv)
        rows.append(dict(zip(keys, key), rate_share_auc=float(integral)))
    return pd.DataFrame(rows)


class ManuscriptFigures:
    def __init__(self, root, output):
        self.root, self.output = Path(root), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []
        focus = self.root / "results/population/discrimination_v5_scope_hypothesis_full"
        control = self.root / "results/population/discrimination"
        self.focus = {name: pd.read_csv(focus / f"population_{stem}.csv", low_memory=False)
                      for name, stem in [("curves", "discrimination_curves"), ("weights", "observer_weights"),
                                         ("rmse", "observer_rmse")]}
        self.control = {name: pd.read_csv(control / f"population_{stem}.csv", low_memory=False)
                        for name, stem in [("curves", "discrimination_curves"), ("weights", "observer_weights"),
                                           ("rmse", "observer_rmse")]}
        self.mc = pd.read_csv(focus / "merged_curves.csv", low_memory=False)
        self.weights_mc = pd.read_csv(focus / "merged_weights.csv", low_memory=False)
        self.params = pd.read_csv(focus / "merged_params.csv", low_memory=False)
        self.index = pd.read_csv(self.root / "data/detection/population/index_regime_amplitudes.csv")
        self.equiv = pd.read_csv(self.root / "data/features/population/population_regime_response_equivalence.csv")
        det = self.root / "results/population/detection/behavioral_matching"
        self.det = {name: pd.read_csv(det / f"detection_{name}.csv") for name in ["curves", "weights", "metrics"]}
        from population.detection.detection_core import load_detection_behavior
        from population.discrimination.behavioral_data import load_slack_dir
        self.det_animals = load_detection_behavior(self.root / "data/detection/slack/DetectionThresholds.xlsx")
        self.animals, self.behavior = load_slack_dir(str(self.root / "data/discrimination/slack"))
        self.table(self.det_animals, "source_detection_individual_animals")
        self.table(self.animals, "source_discrimination_individual_animals")
        for folder in [focus, control]:
            valid = json.loads((folder / "validation.json").read_text())
            if valid.get("status") != "ready" or valid.get("failed_observer_fits", 0):
                raise ValueError(f"Unvalidated fit source: {folder}")
        # Compare shared hybrid fits as a numerical bridge, not a claim of
        # byte-identical historical inputs (those hashes were not recorded).
        a, b = select(self.focus["curves"], split="test"), select(self.control["curves"], split="test")
        columns = ["loss_reg_l2", "weight_kl", "weight_entropy_floor_k", "weight_entropy_floor_lambda",
                   "weight_max_cap", "weight_max_cap_lambda", "n_mc"]
        for col in columns:
            if set(a[col]) != set(b[col]):
                raise ValueError(f"Focused/control fit settings differ: {col}")
        paired = a.merge(b, on=["freq", "block", "cv"], validate="one_to_one")
        self.bridge_gap = float(abs(paired.p_predicted_mean_x - paired.p_predicted_mean_y).max())
        if len(paired) != 88 or self.bridge_gap > .001:
            raise ValueError("Historical cue-control bridge requires a fresh fit audit")

    def figure(self, fig, stem):
        for suffix in ["png", "pdf"]:
            fig.savefig(self.output / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
        self.records.append(dict(kind="figure", stem=stem,
                                 path=(self.output / f"{stem}.png").relative_to(self.root).as_posix()))
        return fig

    def table(self, frame, stem):
        frame.to_csv(self.output / f"{stem}.csv", index=False)
        frame.to_html(self.output / f"{stem}.html", index=False, float_format=lambda x: f"{x:.4f}")
        self.records.append(dict(kind="table", stem=stem,
                                 path=(self.output / f"{stem}.csv").relative_to(self.root).as_posix()))
        return frame

    def mixture(self, scope="primary", family="hybrid", block="std0", controls=False):
        frame = select((self.control if controls else self.focus)["weights"], scope, family)
        if "block" in frame and scope == "per_panel":
            frame = frame[frame.block == block]
        frame = frame[frame.regime_active.astype(str).str.lower().isin(["true", "1"])]
        mapping = self.equiv[["freq", "canonical_regime", "representative_regime"]].drop_duplicates()
        frame = frame.merge(mapping, left_on=["freq", "canonical_thresh_type"],
                            right_on=["freq", "canonical_regime"], validate="many_to_one")
        result = frame.pivot_table(index="freq", columns="representative_regime", values="weight_mean", aggfunc="sum")
        result = result.reindex(index=FREQS, columns=REGIMES).fillna(0)
        if not np.allclose(result.sum(axis=1), 1):
            raise ValueError("Incomplete nominal mixture mapping")
        return result

    def animal_panel(self, ax, freq, block, means=True):
        animals = self.animals[(self.animals.freq == freq) & (self.animals.block == block)]
        for _, animal in animals.groupby("rat_name"):
            animal = animal.sort_values("cv")
            ax.plot(animal.cv, choice(animal.cv, animal.p_correct, block), color=".72", lw=.8, alpha=.65, zorder=1)
        if means:
            b = self.behavior[(self.behavior.freq == freq) & (self.behavior.block == block)].sort_values("cv")
            ax.errorbar(b.cv, choice(b.cv, b.p_mean, block), yerr=b.p_sem,
                        fmt="o", color="black", ms=3.5, capsize=2, zorder=8)
        ax.set(title=f"{block} | {freq} Hz", ylim=(-.02, 1.03), xlim=(-.02, 1.02))

    def observer_panel(self, ax, freq, block, scopes=False, families=None,
                       emphasize=False, locked_curves=None):
        if scopes and locked_curves is not None:
            raise ValueError("Locked cue controls apply only to the primary scope")
        self.animal_panel(ax, freq, block)
        if scopes:
            items = [(s, "hybrid", self.focus, c) for s, c in
                     [("crossed", "#b2182b"), ("per_panel", "#762a83"), ("primary", "#2166ac")]]
        else:
            items = [("primary", f, self.focus if f == "hybrid" else self.control, CUES[f])
                     for f in (["rate", "timing", "hybrid"] if families is None else families)]
        for scope, family, source, color in items:
            if locked_curves is not None and family != "hybrid":
                p = locked_curves[locked_curves.model == family]
            else:
                p = select(source["curves"], scope, family, "test")
            p = p[(p.freq == freq) & (p.block == block)].sort_values("cv")
            primary = (scope == "primary" and family == "hybrid")
            style = cue_line_style(family) if emphasize and not scopes else dict(
                lw=2.6 if primary else 1.8, alpha=1 if primary else .8, zorder=5 if primary else 4)
            ax.plot(p.cv, choice(p.cv, p.p_predicted_mean, block), color=color, **style)

    def locked_cue_curves(self):
        """Load validated conditional RR/TT fits, retaining the original hybrid reference."""
        if hasattr(self, "_locked_cue_curves"):
            return self._locked_cue_curves
        from population.discrimination.locked_mixture_followup import STEMS, validate_frames, verify
        folder = self.root / "results/population/discrimination_locked_mixture_followup"
        bundle = self.root / "results/population/discrimination_locked_mixture_inputs"
        validation = json.loads((folder / "validation.json").read_text())
        if validation.get("status") != "ready" or validation.get("conditional_fits") != 1000:
            raise ValueError("Locked-mixture figure requires the validated full run")
        if json.loads((folder / "input_manifest.json").read_text()) != verify(self.root, bundle):
            raise ValueError("Locked-mixture input manifest mismatch")
        frames = {s: pd.read_csv(folder / f"merged_{s}.csv") for s in STEMS}
        validate_frames(frames, bundle, 200)
        current = select(self.mc, split="test")
        curves = frames["curves"]
        saved = curves[(curves.model == "primary_hybrid_reference") & (curves.split == "test")]
        paired = current.merge(saved, on=["mc", "freq", "block", "cv"],
                               validate="one_to_one", suffixes=("_current", "_locked"))
        if len(paired) != 17600 or not np.allclose(
                paired.p_predicted_current, paired.p_predicted_locked, rtol=0, atol=1e-10):
            raise ValueError("Locked controls do not share the displayed primary hybrid")
        weights = frames["weights"].pivot_table(index="freq", columns="thresh_type", values="weight")
        weights = weights.reindex(index=FREQS, columns=REGIMES)
        if not np.allclose(weights, self.mixture(), rtol=0, atol=1e-12):
            raise ValueError("Displayed hybrid mixtures differ from the locked controls")
        selected = curves[(curves.split == "test") & curves.model.isin(["rate", "timing", "primary_hybrid_reference"])].copy()
        selected["model"] = selected.model.replace({"primary_hybrid_reference": "hybrid"})
        result = selected.groupby(["model", "freq", "block", "cv"]).p_predicted.mean().reset_index(name="p_predicted_mean")
        self.table(result, "fig4_locked_mixture_cue_curves_source")
        self.table(weights.rename_axis(index="freq", columns="regime").stack().rename("weight").reset_index(),
                   "fig4_locked_hybrid_mixtures_source")
        self._locked_cue_curves = result
        return result

    def single_mixture_bars(self, ax, freq, mixtures):
        """One shared hybrid mixture, with response-equivalent labels combined."""
        regimes = self.retained_regimes(freq)
        values = mixtures.loc[freq, regimes]
        ax.bar(range(len(regimes)), values, color=[COLORS[r] for r in regimes], width=.8)
        upper = max(.05, float(np.ceil(mixtures.to_numpy().max() * 1.1 / .05) * .05))
        ax.set(title=f"{freq} Hz", ylim=(0, upper), xticks=range(len(regimes)),
               xticklabels=[self.collapsed_label(freq, r) for r in regimes])
        ax.tick_params(axis="x", labelrotation=60, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    def mixture_bars(self, ax, freq):
        bottom = np.zeros(3)
        for regime in REGIMES:
            y = [self.mixture(family=family, controls=family != "hybrid").loc[freq, regime]
                 for family in ["rate", "hybrid", "timing"]]
            ax.bar(range(3), y, bottom=bottom, color=COLORS[regime], width=.72, edgecolor="white", linewidth=.3)
            bottom += y
        ax.set(xticks=range(3), xticklabels=["Rate", "Hybrid", "Timing"], ylim=(0, 1), title=f"{freq} Hz")
        for tick, family in zip(ax.get_xticklabels(), ["rate", "hybrid", "timing"]):
            tick.set_color(CUES[family])
        ax.get_xticklabels()[1].set_fontweight("bold")

    def independent_mixture_panel(self, ax, freq, mixtures, style="bars"):
        """Identical independent mixtures in stacked, grouped, or categorical-line form."""
        if style not in {"bars", "grouped", "lines"}:
            raise ValueError("Unknown independent mixture display")
        regimes = self.retained_regimes(freq)
        families = ["rate", "hybrid", "timing"]
        x = np.arange(len(regimes))
        if style == "bars":
            bottom = np.zeros(3)
            for regime in regimes:
                values = np.array([mixtures[f].loc[freq, regime] for f in families])
                ax.bar(range(3), values, bottom=bottom, color=COLORS[regime],
                       width=.72, edgecolor="white", linewidth=.3)
                bottom += values
            if not np.allclose(bottom, 1, atol=1e-9, rtol=0):
                raise ValueError("Incomplete retained mixture mass")
            ax.set(xticks=range(3), xticklabels=[f.title() for f in families], ylim=(0, 1))
            for tick, family in zip(ax.get_xticklabels(), families):
                tick.set_color(CUES[family])
                tick.set_fontweight("bold" if family == "hybrid" else "normal")
        else:
            upper = max(.1, float(np.ceil(max(m.to_numpy().max() for m in mixtures.values()) * 1.08 / .1) * .1))
            for offset, family in zip([-.25, 0, .25], families):
                values = mixtures[family].loc[freq, regimes].to_numpy(float)
                if style == "grouped":
                    ax.bar(x+offset, values, width=.23, color=CUES[family],
                           alpha=1 if family == "hybrid" else .75,
                           edgecolor=CUES[family], linewidth=.6, label=family.title())
                else:
                    ax.plot(x, values, color=CUES[family], marker="o", ms=3,
                            label=family.title(), **cue_line_style(family))
            ax.set(xticks=x, xticklabels=[self.collapsed_label(freq, r) for r in regimes],
                   ylim=(0, upper), xlim=(-.55, len(regimes)-.45))
            ax.tick_params(axis="x", labelrotation=60, labelsize=8)
            for tick in ax.get_xticklabels():
                tick.set_ha("right")
        ax.set_title(f"{freq} Hz")
        ax.grid(False, axis="x")

    def independent_mixture_source(self, mixtures):
        return pd.DataFrame([
            dict(freq=freq, model=family, representative=regime,
                 collapsed_label=self.collapsed_label(freq, regime),
                 weight=float(mixtures[family].loc[freq, regime]))
            for freq in FREQS for family in ["rate", "hybrid", "timing"]
            for regime in self.retained_regimes(freq)])

    def retained_regimes(self, freq):
        """Omit collapsed classes, not genuine zero-weight retained classes."""
        representatives = set(self.equiv.loc[self.equiv.freq == freq, "representative_regime"])
        return [r for r in REGIMES if r in representatives]

    def collapsed_label(self, freq, representative):
        rows = self.equiv[(self.equiv.freq == freq) &
                          (self.equiv.representative_regime == representative)]
        members = set("|".join(rows.member_regimes.dropna().unique()).split("|"))
        ordered = [representative] + [r for r in REGIMES if r in members and r != representative]
        return " / ".join(label(r) for r in ordered)

    def mixture_lines(self):
        """Inspection alternative; weights are frequency-specific, not CV-specific."""
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
        mixtures = {f: self.mixture(family=f, controls=f != "hybrid") for f in CUES}
        rows = []
        for ax, freq in zip(axes, FREQS):
            regimes = self.retained_regimes(freq)
            for family in ["rate", "timing", "hybrid"]:
                y = mixtures[family].loc[freq, regimes]
                ax.plot(range(len(regimes)), y, color=CUES[family], marker="o",
                        ms=4, lw=2.3 if family == "hybrid" else 1.6, label=family.title())
                rows.extend(dict(freq=freq, model=family, regime=r, weight=float(y[r])) for r in regimes)
            ax.set(title=f"{freq} Hz", xticks=range(len(regimes)),
                   xticklabels=[self.collapsed_label(freq, r) for r in regimes], ylim=(0, .6))
            ax.tick_params(axis="x", labelrotation=60)
        axes[0].set_ylabel("Mixture weight (shared across blocks)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
        fig.text(.5, .01, "Retained spike-count regimes; lines are visual guides between categories, not CV trajectories.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, .92))
        self.table(pd.DataFrame(rows), "fig4c_alternative_mixture_lines_source")
        return self.figure(fig, "fig4c_alternative_mixture_lines")

    def fig4(self, scopes=False):
        fig, axes = plt.subplots(2 if scopes else 3, 4, figsize=(15, 7 if scopes else 10),
                                 gridspec_kw=None if scopes else {"height_ratios": [1, 1, .65]})
        for row, block in enumerate(["std0", "std1"]):
            for col, freq in enumerate(FREQS):
                self.observer_panel(axes[row, col], freq, block, scopes)
                axes[row, col].set_xlabel("Coefficient of Variation (CV)" if row == 1 else "", fontsize=9)
                cv_ticks(axes[row, col])
                axes[row, col].tick_params(axis="x", labelbottom=row == 1, labelsize=9)
                if col == 0:
                    axes[row, col].set_ylabel("P(choose CV1)" if row == 0 else "P(choose CV0)")
        if not scopes:
            for col, freq in enumerate(FREQS):
                self.mixture_bars(axes[2, col], freq)
            axes[2, 0].set_ylabel("Mixture weight\n(shared across blocks)")
            handles = [Line2D([], [], color=CUES[f], lw=2.5, label=t) for f, t in
                       [("rate", "Rate"), ("timing", "Timing"), ("hybrid", "Hybrid")]]
            fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.5, .985), frameon=False, fontsize=9)
            fig.legend(handles=[Patch(color=COLORS[r], label=label(r)) for r in REGIMES],
                       ncol=9, loc="lower center", bbox_to_anchor=(.5, .025), frameon=False)
            fig.text(.5, .009, "Response-equivalent regimes are combined under the retained nominal representative.",
                     ha="center", fontsize=8)
            fig.tight_layout(rect=(.015, .065, 1, .92))
            # Make room for one common CV label between behavior and mixtures.
            for ax in axes[2]:
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0, pos.width, pos.height * .86])
            stem = "fig4_discrimination_independent_cue_models"
        else:
            handles = [Line2D([], [], color=c, lw=2, label=LABELS[s].replace(" + ", "\n+ ")) for s, c in
                       [("primary", "#2166ac"), ("crossed", "#b2182b"), ("per_panel", "#762a83")]]
            fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False, fontsize=8)
            fig.tight_layout(rect=(.015, .11, 1, 1))
            stem = "suppfig8_competing_scope_models"
        for row, letter in enumerate("ab" if scopes else "abc"):
            axes[row, 0].text(-.18, 1.08, letter, transform=axes[row, 0].transAxes, weight="bold", fontsize=13)
        return self.figure(fig, stem)

    def fig4_reliance_preview(self, mixture_style="bars", *, locked_mixtures=False):
        """Combined layout: behavior, reliance and a choice of mixture displays."""
        if mixture_style not in ({"single"} if locked_mixtures else {"bars", "grouped", "lines"}):
            raise ValueError("Use bars/grouped/lines for independent fits, or single for locked mixtures")
        locked_curves = self.locked_cue_curves() if locked_mixtures else None
        summary = integrated_share(select(self.mc, split="test"),
                                   ["mc", "freq", "block"], "effective_rate_share")
        summary = summary.groupby(["freq", "block"]).rate_share_auc.agg(
            mean="mean", q025=lambda x: x.quantile(.025), q975=lambda x: x.quantile(.975)).reset_index()
        self.table(summary, "fig4_reliance_preview_source")
        from population.figure_generation.layout import observer_grid, panel_label, observer_headers, reliance_background
        fig, axes = observer_grid(mixture_row=True)
        for row, block in enumerate(["std0", "std1"]):
            for col, freq in enumerate(FREQS):
                ax = axes[row, col]
                self.observer_panel(ax, freq, block, emphasize=True, locked_curves=locked_curves)
                cv_ticks(ax)
                ax.tick_params(axis="x", labelbottom=row == 1, labelsize=8)
                ax.set_xlabel("Coefficient of Variation (CV)" if row == 1 else "", fontsize=8)
                if col:
                    ax.tick_params(axis="y", labelleft=False)
            axes[row, 0].set_ylabel("P(choose CV1)" if row == 0 else "P(choose CV0)")
            ax = axes[row, 4]
            p = summary[summary.block == block].set_index("freq").loc[FREQS]
            # Draw quantile endpoints directly; this remains valid if a skewed
            # MC distribution has its mean outside the central interval.
            x = np.arange(4)
            ax.vlines(x, p.q025, p.q975, color=CUES["hybrid"], lw=1.5)
            ax.hlines(p.q025, x - .07, x + .07, color=CUES["hybrid"], lw=1.5)
            ax.hlines(p.q975, x - .07, x + .07, color=CUES["hybrid"], lw=1.5)
            ax.plot(x, p["mean"], "o", color=CUES["hybrid"], ms=5)
            ax.axhline(.5, color=".7", ls=":", lw=1)
            ax.set(xticks=x, xticklabels=FREQS, ylim=(0, 1), xlim=(-.4, 3.4),
                   ylabel="CV-averaged rate reliance", title=f"Hybrid | {block}")
            ax.set_xlabel("Frequency (Hz)" if row else "")
            reliance_background(ax)
        mixtures = {f: self.mixture(family=f, controls=f != "hybrid")
                    for f in (["hybrid"] if locked_mixtures else CUES)}
        if not locked_mixtures:
            self.table(self.independent_mixture_source(mixtures), "fig4_independent_mixtures_source")
        for col, freq in enumerate(FREQS):
            ax = axes[2, col]
            if mixture_style == "single":
                self.single_mixture_bars(ax, freq, mixtures["hybrid"])
            else:
                self.independent_mixture_panel(ax, freq, mixtures, mixture_style)
            if col == 0:
                ax.set_ylabel("Hybrid mixture weight\n(shared by all readouts)" if locked_mixtures
                              else "Mixture weight\n(shared across blocks)")
        legend_ax = axes[2, 4]
        legend_ax.set_axis_off()
        # Center in the entire blank region, including the inter-panel gutter.
        legend_x = (axes[2, 3].get_position().x1 + axes[0, 4].get_position().x1) / 2
        legend_y = (legend_ax.get_position().y0 + legend_ax.get_position().y1) / 2
        if mixture_style == "bars":
            legend_y = legend_ax.get_position().y1 - .15 / fig.get_figheight()
        cue_legend = legend_ax.legend(
            handles=[Line2D([], [], color=CUES[f], label=f.title(), **cue_line_style(f))
                     for f in ["rate", "timing", "hybrid"]],
            loc="center", bbox_to_anchor=(legend_x, legend_y), bbox_transform=fig.transFigure,
            ncol=3, frameon=False,
            fontsize=8, handlelength=1.4, handletextpad=.4, columnspacing=.7,
            borderaxespad=0, title="Observer Model Components", title_fontsize=9)
        cue_legend.get_texts()[2].set_fontweight("bold")
        if mixture_style == "bars":
            legend_ax.add_artist(cue_legend)
            legend_ax.legend(handles=[Patch(color=COLORS[r], label=label(r)) for r in REGIMES],
                             ncol=3, loc="upper center", bbox_to_anchor=(legend_x,
                                 legend_ax.get_position().y1 - .49 / fig.get_figheight()),
                             bbox_transform=fig.transFigure, frameon=False,
                             fontsize=8, handlelength=1.2, handletextpad=.4, columnspacing=.8,
                             borderaxespad=0, labelspacing=.5,
                             title="Operating Regimes", title_fontsize=9)
        # One label for both behavior rows; the reliance column is panel b.
        observer_headers(fig, axes, FREQS)
        for letter, ax in [("a", axes[0, 0]), ("b", axes[0, 4]), ("c", axes[2, 0])]:
            panel_label(fig, ax, letter)
        stem = ("fig4_combined_behavior_locked_mixtures_reliance" if locked_mixtures else
                f"fig4_combined_behavior_mixtures_reliance_{mixture_style}")
        return self.figure(fig, stem)

    def detection_panel(self, ax, freq, fits=True, shared_mixture=False):
        for _, animal in self.det_animals[self.det_animals.freq == freq].groupby("rat"):
            animal = animal.sort_values("cv")
            ax.plot(animal.cv, animal.norm_threshold, color=".72", lw=.8, alpha=.65)
        curves = self.det["curves"]
        curves = curves[(curves.target_kind == "datapoints") & (curves.freq == freq)]
        fixed = curves[curves["mode"] == "best_single"].sort_values("cv")
        shared = curves[curves["mode"] == "per_frequency"].sort_values("cv")
        points = curves[(curves["mode"] == "per_point") & curves.behavior_mean.notna()].sort_values("cv")
        single = self.det["metrics"].query("target_kind == 'datapoints' and mode == 'best_single' and freq == 'ALL'")
        regime = single.selected_single_regime.iloc[0]
        if fits:
            ax.plot(fixed.cv, fixed.prediction, color="#2ca25f", lw=2, label=f"Single Regime ({label(regime)})")
            if shared_mixture:
                if shared.empty or shared.cv.duplicated().any():
                    raise ValueError("Expected one per-frequency detection prediction per CV")
                ax.plot(shared.cv, shared.prediction, color="#756bb1", ls="--", lw=1.7,
                        label="Per-Frequency Mixture", zorder=3)
            ax.scatter(points.cv, points.prediction, color="#f4a261", marker="s", s=50,
                       label="Per-CV Mixture", zorder=4)
        ax.errorbar(points.cv, points.behavior_mean, yerr=points.behavior_sem, fmt="o", color="black",
                    ms=3, capsize=2, label="Rat mean ± SEM", zorder=5)
        # Adapt each frequency without clipping faint individual animals or SEM.
        values = np.r_[self.det_animals.loc[self.det_animals.freq == freq, "norm_threshold"],
                       points.behavior_mean - points.behavior_sem,
                       points.behavior_mean + points.behavior_sem]
        if fits:
            values = np.r_[values, fixed.prediction, points.prediction]
            if shared_mixture:
                values = np.r_[values, shared.prediction]
        low = np.floor((np.nanmin(values) - .01) * 100) / 100
        high = np.ceil((max(1., np.nanmax(values)) + .01) * 100) / 100
        ax.set(xlabel="CV", xlim=(-.02, 1.02), ylim=(low, high), title=f"{freq} Hz")
        ax.set_yticks(np.arange(np.ceil(low * 20) / 20, high + 1e-9, .05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    def fig3(self):
        fig, axes = plt.subplots(3, 5, figsize=(15, 7.6), gridspec_kw={"height_ratios": [1, 1, .65]})
        for col, freq in enumerate([10] + FREQS):
            p = self.index[self.index.freq == freq].groupby(["cv", "thresh_type"]).amp_uA.median().unstack()
            for regime in REGIMES:
                axes[0, col].plot(p.index, p[regime], color=COLORS[regime], lw=1.3, marker="o", ms=2)
            axes[0, col].set(title=f"{freq} Hz", xlim=(0, 1))
            self.detection_panel(axes[1, col], freq, shared_mixture=True)
            w = self.det["weights"]
            w = w[(w.target_kind == "datapoints") & (w["mode"] == "per_point") & (w.freq == freq)]
            w = w.pivot(index="cv", columns="thresh_type", values="weight").reindex(columns=REGIMES).fillna(0)
            if len(w) != 6 or not np.allclose(w.sum(axis=1), 1):
                raise ValueError("Detection mixture figure requires six measured CVs")
            bottom = np.zeros(len(w))
            for regime in REGIMES:
                axes[2, col].bar(w.index, w[regime], bottom=bottom, width=.2, color=COLORS[regime])
                bottom += w[regime].to_numpy()
            # Contiguous bins centered on the measured CVs; the same x mapping
            # on all rows keeps the endpoints and intermediate ticks aligned.
            for boundary in np.arange(.1, 1, .2):
                axes[2, col].axvline(boundary, color=".65", lw=.6)
            axes[2, col].set(ylim=(0, 1))
            axes[2, col].grid(False, which="both")
            for row in range(3):
                axes[row, col].set(xlim=(-.1, 1.1), xlabel="")
                if row == 2:
                    axes[row, col].set_xlabel("Coefficient of Variation (CV)", fontsize=9)
                cv_ticks(axes[row, col])
                axes[row, col].tick_params(axis="x", labelbottom=row == 2)
                axes[row, col].tick_params(axis="both", direction="out", labelsize=9)
            axes[1, col].set_title("")
        for row, ylabel in enumerate(["Index threshold\n(µA; median)", "Normalized\nthreshold", "Per-CV\nmixture weight"]):
            axes[row, 0].set_ylabel(ylabel)
        handles, labels_ = axes[1, 0].get_legend_handles_labels()
        pairs = [(h, l) for h, l in zip(handles, labels_) if l != "Rat mean ± SEM"]
        axes[1, 0].legend(*zip(*pairs), loc="lower left", frameon=False, fontsize=7.5,
                          handlelength=1.7, borderaxespad=.3, labelspacing=.25)
        fig.legend(handles=[Patch(color=COLORS[r], label=label(r)) for r in REGIMES], ncol=9,
                   loc="lower center", bbox_to_anchor=(.5, .025), frameon=False)
        fig.tight_layout(rect=(.01, .10, 1, 1), h_pad=.6, w_pad=.7)
        for row, letter in enumerate("abc"):
            fig.text(.018, axes[row, 0].get_position().y1 + .008, letter, weight="bold", fontsize=13)
        return self.figure(fig, "fig3_detection_capacity")

    def partials(self, freq=100, rep=0):
        figures = []
        fig, ax = plt.subplots(figsize=(5, 3.5))
        self.detection_panel(ax, freq, fits=False)
        ax.set_ylabel("Normalized detection threshold")
        fig.tight_layout()
        figures.append(self.figure(fig, "fig1d_detection_behavior"))
        fig, ax = plt.subplots(figsize=(10, 5.6))
        subset = self.index[(self.index.freq == freq) & (self.index.cv == 0) & (self.index.rep_index == rep)].set_index("thresh_type")
        patterns = self.root / "data/time-patterns/population"
        meta = json.loads((patterns / f"cv0.0_f{freq}_meta.json").read_text())
        with np.load(patterns / f"cv0.0_f{freq}_waveforms.npz") as bundle:
            key = str(subset.iloc[0].rep_label)
            wave = np.asarray(bundle[key], float)
        active = wave < 0
        onsets = np.flatnonzero(active & ~np.r_[False, active[:-1]]) / float(meta["fs"])
        positions = np.arange(10)[::-1] * 1.05
        ax.vlines(onsets, positions[0] - .44, positions[0] + .44, color=".15", lw=1.1)
        left_labels, right_labels = ["Input pulses"], [f"{len(onsets)} pulses"]
        table = []
        for y, regime in enumerate(REGIMES):
            row = subset.loc[regime]
            spikes = np.asarray(json.loads(row.index_spike_times_ms), float)
            # Saved NEURON timestamps include initialization; display same origin
            # as population feature traces (initialization removed).
            spikes = spikes - 100.
            pos = positions[y + 1]
            ax.vlines(spikes / 1000., pos - .44, pos + .44, color=COLORS[regime], lw=1.1)
            left_labels.append(f"{label(regime)}\n({row.amp_uA:.2f} µA)")
            right_labels.append(f"{len(spikes)} spike" + ("s" if len(spikes) != 1 else ""))
            table.append(dict(regime=regime, amp_uA=row.amp_uA, spikes=row.n_index_spikes, freq=freq, cv=0, rep=rep))
        ax.set(yticks=positions, yticklabels=left_labels, xlabel="Time (sec)",
               xticks=np.arange(0, 2.01, .25), xlim=(-.01, 2.0),
               ylim=(-.95, positions.max() + .95), title=f"{freq} Hz | CV=0")
        for tick in ax.get_yticklabels():
            tick.set_multialignment("center")
        ax.grid(False)
        ax.grid(True, axis="x", alpha=.22)
        ax.tick_params(axis="both", direction="in", top=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
        right = ax.twinx()
        right.set(ylim=ax.get_ylim(), yticks=positions, yticklabels=right_labels)
        right.tick_params(axis="y", length=0, pad=6)
        right.grid(False)
        fig.tight_layout()
        figures.append(self.figure(fig, "fig1c_index_regime_raster_100Hz"))
        self.table(pd.DataFrame(table), "fig1c_index_regime_values")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        y = self.mixture().loc[freq]
        regimes = self.retained_regimes(freq)
        ax.bar(range(len(regimes)), y[regimes], color=[COLORS[r] for r in regimes])
        ax.set(xticks=range(len(regimes)), xticklabels=[self.collapsed_label(freq, r) for r in regimes], ylabel="Mixture weight", title=f"Primary hybrid | {freq} Hz")
        ax.tick_params(axis="x", labelrotation=60)
        fig.tight_layout()
        figures.append(self.figure(fig, "fig1d_primary_mixture_100Hz"))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        self.animal_panel(ax, freq, "std1")
        b = self.behavior[(self.behavior.freq == freq) & (self.behavior.block == "std1")].sort_values("cv")
        ax.plot(b.cv, choice(b.cv, b.p_mean, "std1"), color="black", lw=1.4)
        cv_ticks(ax)
        ax.set(xlabel="Coefficient of Variation (CV)", ylabel="P(choose CV0)")
        fig.tight_layout()
        figures.append(self.figure(fig, "fig1d_discrimination_behavior_100Hz"))
        return figures

    def fig2(self, freq=100):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), sharex=True)
        for ax, block in zip(axes, ["std0", "std1"]):
            self.observer_panel(ax, freq, block, families=["hybrid"])
            ax.set_ylabel("P(choose CV1)" if block == "std0" else "P(choose CV0)")
            ax.set_xlabel("CV")
        fig.legend(handles=[Line2D([], [], color=CUES["hybrid"], lw=2.5, label="Primary hybrid")],
                   loc="lower center", frameon=False)
        fig.tight_layout(rect=(0, .10, 1, 1))
        return self.figure(fig, "fig2_psychometrics_100Hz")

    def comparisons(self):
        from population.discrimination.discrimination_core import load_behavior
        behavior = load_behavior(str(self.root / "data/discrimination/slack"), FREQS,
                                 {(float(f), round(c, 1)) for f in FREQS for c in np.linspace(0, 1, 11)})
        trials = int(behavior.n.sum())
        rows, panel_rows, shares, block_diagnostics = [], [], [], []
        for scope, family, source in [(s, "hybrid", self.focus) for s in SCOPES] + [("primary", f, self.control) for f in ["rate", "timing"]]:
            test = select(source["rmse"], scope, family, "test")
            panels = test[pd.to_numeric(test.freq, errors="coerce").notna() & test.block.isin(["std0", "std1"])]
            overall = test[(test.freq == "ALL") & (test.block == "ALL")].iloc[0]
            train = select(source["rmse"], scope, family, "train")
            train_panels = train[pd.to_numeric(train.freq, errors="coerce").notna() & train.block.isin(["std0", "std1"])].copy()
            train = train[(train.freq == "ALL") & (train.block == "ALL")].iloc[0]
            k = int(overall.n_parameters) if "n_parameters" in overall else 29
            auc = integrated_share(select(source["curves"], scope, family, "test"), ["freq", "block"], "effective_rate_share_mean")
            auc["model"] = scope + "_" + family
            shares.append(auc)
            rows.append(dict(model=scope + "_" + family, mixtures=SCOPES[scope][0], evidence=SCOPES[scope][1],
                             parameters=k, simulation_test_RMSE=overall.rmse_mean,
                             frequency_mean_RMSE=panels.rmse_mean.mean(),
                             std0_frequency_mean_RMSE=panels[panels.block == "std0"].rmse_mean.mean(),
                             std1_frequency_mean_RMSE=panels[panels.block == "std1"].rmse_mean.mean(),
                             std0_rate_share=auc[auc.block == "std0"].rate_share_auc.mean(),
                             std1_rate_share=auc[auc.block == "std1"].rate_share_auc.mean(),
                             AIC=2*k + 2*trials*train.data_nll_train_mean,
                             BIC=np.log(trials)*k + 2*trials*train.data_nll_train_mean,
                             information_criterion_basis="Heuristic: mean train NLL at regularized fits; nominal parameter count; pooled behavioral trials",
                             source="focused_scope_run" if source is self.focus else "comprehensive_control_run"))
            # Likelihood contributions of the existing JOINT fits, not blockwise
            # AIC/BIC: shared mixture parameters cannot be assigned to one block.
            counts = behavior.groupby(["freq", "block"], as_index=False).n.sum()
            train_panels["freq"] = pd.to_numeric(train_panels.freq)
            likelihood = train_panels.merge(counts, on=["freq", "block"], validate="one_to_one")
            predictions = select(source["curves"], scope, family, "test")
            for block_name in ["std0", "std1"]:
                l = likelihood[likelihood.block == block_name]
                b = predictions[predictions.block == block_name]
                block_diagnostics.append(dict(
                    model=scope + "_" + family, block=block_name,
                    frequency_mean_RMSE=panels[panels.block == block_name].rmse_mean.mean(),
                    ensemble_prediction_pooled_RMSE=float(np.sqrt(np.mean((b.p_predicted_mean - b.p_observed_mean)**2))),
                    behavioral_trials=int(l.n.sum()),
                    train_NLL_joint_fit=float((l.n * l.nll_per_trial_mean).sum()),
                    interpretation="Block contribution of joint regularized fit; not block-specific model selection"))
            panels = panels.copy()
            panels["model"] = scope + "_" + family
            panel_rows.append(panels)
        metrics = pd.DataFrame(rows)
        metrics["delta_AIC"] = metrics.AIC - metrics.AIC.min()
        metrics["delta_BIC"] = metrics.BIC - metrics.BIC.min()
        self.table(metrics, "table1_model_performance_and_reliance")
        self.table(pd.DataFrame(block_diagnostics), "table1_block_fit_diagnostics")
        self.table(metrics[["model", "parameters", "simulation_test_RMSE", "std0_frequency_mean_RMSE",
                            "std1_frequency_mean_RMSE", "delta_AIC", "delta_BIC"]], "table1a_performance")
        self.table(metrics[["model", "std0_rate_share", "std1_rate_share"]], "table1b_rate_reliance")
        self.table(pd.concat(panel_rows), "supptable2_panel_performance")
        self.table(pd.concat(shares), "supptable2_frequency_rate_reliance")
        self.table(self.params, "supptable2_fitted_evidence_parameters")
        reliance = []
        for scope in ["primary", "per_panel"]:
            auc = integrated_share(select(self.mc, scope, split="test"), ["mc", "freq", "block"], "effective_rate_share")
            auc["model"] = scope
            reliance.append(auc)
        reliance = pd.concat(reliance)
        summary = reliance.groupby(["model", "freq", "block"]).rate_share_auc.agg(
            mean="mean", q025=lambda x: x.quantile(.025), q975=lambda x: x.quantile(.975)).reset_index()
        self.table(summary, "supptable2_reliance_MC_descriptive_intervals")
        block = reliance.groupby(["model", "mc", "block"]).rate_share_auc.mean().unstack("block")
        block["std0_minus_std1"] = block.std0 - block.std1
        self.table(block.reset_index(), "source_AIC_BIC_block_reliance_comparison")
        similarities = []
        primary = self.mixture()
        for block_name in ["std0", "std1"]:
            other = self.mixture("per_panel", block=block_name)
            for freq in FREQS:
                similarities.append(dict(comparison="primary_vs_per_panel", freq=freq, block=block_name,
                                         total_variation=tv(primary.loc[freq], other.loc[freq])))
        for family in ["rate", "timing"]:
            other = self.mixture(family=family, controls=True)
            for freq in FREQS:
                similarities.append(dict(comparison="hybrid_vs_" + family, freq=freq, block="shared",
                                         total_variation=tv(primary.loc[freq], other.loc[freq])))
        rate, timing = [self.mixture(family=f, controls=True) for f in ["rate", "timing"]]
        for freq in FREQS:
            similarities.append(dict(comparison="rate_vs_timing", freq=freq, block="shared",
                                     total_variation=tv(rate.loc[freq], timing.loc[freq])))
        self.table(pd.DataFrame(similarities), "supptable2_mixture_similarity")
        curve_gaps = []
        h = select(self.focus["curves"], split="test")
        for family in ["rate", "timing"]:
            p = h.merge(select(self.control["curves"], family=family, split="test"), on=["freq", "block", "cv"], validate="one_to_one")
            p["squared_prediction_difference"] = (p.p_predicted_mean_x - p.p_predicted_mean_y)**2
            for (freq, block_name), sub in p.groupby(["freq", "block"]):
                curve_gaps.append(dict(family=family, freq=freq, block=block_name,
                                       hybrid_prediction_RMS_gap=np.sqrt(sub.squared_prediction_difference.mean())))
        self.table(pd.DataFrame(curve_gaps), "supptable2_independent_cue_prediction_similarity")
        self.table(self.det["metrics"][self.det["metrics"].target_kind == "datapoints"],
                   "supptable2_detection_scope_comparison")
        for family in CUES:
            self.table(self.mixture(family=family, controls=family != "hybrid").reset_index(), "source_" + family + "_mixture")
        self.table(pd.DataFrame([dict(shared_hybrid_max_probability_difference=self.bridge_gap,
                                     compared_curve_points=88, behavioral_trials=trials,
                                     caveat="Settings match; historical input hashes unavailable. Not held-out animals.")]), "source_control_fit_bridge")
        return metrics

    def evidence_heatmaps(self):
        evidence = pd.read_csv(self.root / "data/features/population/population_cv0_template_evidence.csv")
        names = ["rate_evidence_1ms", "rate_evidence_10ms", "timing_evidence_1ms", "timing_evidence_10ms"]
        fig, axes = plt.subplots(4, 4, figsize=(14, 11), constrained_layout=True)
        for row, name in enumerate(names):
            for col, freq in enumerate(FREQS):
                p = evidence[evidence.freq == freq].groupby(["thresh_type", "cv"])[name].mean().unstack().reindex(REGIMES)
                image = axes[row, col].imshow(p, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto", interpolation="nearest")
                axes[row, col].set(yticks=range(9), yticklabels=[label(r) for r in REGIMES] if col == 0 else [],
                                   xticks=[0, 5, 10], xticklabels=[0, .5, 1], title=f"{freq} Hz" if row == 0 else "")
                if col == 0:
                    axes[row, col].set_ylabel(name.replace("_evidence_", "\n"))
                if row == 3:
                    axes[row, col].set_xlabel("CV")
        fig.colorbar(image, ax=axes, shrink=.7, label="Mean bounded evidence (all nominal regimes)")
        return self.figure(fig, "suppfig6_population_evidence_inventory")

    def population_summary(self, mode="cv0", include_index=True, stem=None):
        if mode == "true":
            refresh = self.root / "results/population/detection/imax_population_refresh"
            manifest = refresh / "merge_manifest.json"
            if not manifest.exists():
                return None  # Never silently publish old-I_max population summaries.
            from population.detection.refresh_imax_population import digest
            if json.loads(manifest.read_text())["index_sha256"] != digest(self.root / "data/detection/population/index_regime_amplitudes.csv"):
                raise ValueError("True-response summary belongs to a different index table")
            frame = pd.read_csv(refresh / "true_population_trial_summary.csv").merge(
                self.index[["freq", "cv", "rep_index", "thresh_type", "n_index_spikes", "n_input_pulses"]],
                on=["freq", "cv", "rep_index", "thresh_type"], validate="one_to_one")
        else:
            frame = pd.read_csv(self.root / "data/features/population/population_cv0_based_features.csv")
        frame["index_fraction"] = frame.n_index_spikes / frame.n_input_pulses
        metrics = ["index_fraction", "active_axon_count", "total_population_spikes"]
        titles = ["Index spikes / input pulses", "Recruited modeled axons (of 420)", "Population spikes"]
        if not include_index:
            metrics, titles = metrics[1:], titles[1:]
        # Match scales between S3/S4 using their observed ranges rather than the
        # entire cohort count, which hides recruitment differences below ~60.
        scale_frames = [frame]
        if mode == "true":
            scale_frames.append(pd.read_csv(self.root / "data/features/population/population_cv0_based_features.csv",
                                           usecols=["active_axon_count", "total_population_spikes"]))
        else:
            refresh = self.root / "results/population/detection/imax_population_refresh"
            manifest = refresh / "merge_manifest.json"
            if manifest.exists():
                from population.detection.refresh_imax_population import digest
                if json.loads(manifest.read_text())["index_sha256"] == digest(self.root / "data/detection/population/index_regime_amplitudes.csv"):
                    scale_frames.append(pd.read_csv(refresh / "true_population_trial_summary.csv"))
        recruitment_max = max(10, 10 * np.ceil(max(x.active_axon_count.max() for x in scale_frames) / 10))
        spikes_max = max(10, 10 ** np.ceil(np.log10(max(x.total_population_spikes.max() for x in scale_frames))))
        fig, axes = plt.subplots(len(metrics), 5, figsize=(16, 3 * len(metrics)), constrained_layout=True)
        from matplotlib.colors import SymLogNorm
        for row, (metric, title) in enumerate(zip(metrics, titles)):
            for col, freq in enumerate([10] + FREQS):
                p = frame[frame.freq == freq].groupby(["thresh_type", "cv"])[metric].mean().unstack().reindex(REGIMES)
                # Keep zero visible: linear from 0 to 1, logarithmic above 1.
                # Identical count scales in the true-CV and CV0-anchored figures.
                kwargs = {"vmin": 0, "vmax": 1} if metric == "index_fraction" else {
                    "norm": SymLogNorm(linthresh=1, linscale=1, base=10, vmin=0,
                                       vmax=recruitment_max if metric == "active_axon_count" else spikes_max)}
                image = axes[row, col].imshow(p, aspect="auto", interpolation="nearest", cmap="viridis", **kwargs)
                axes[row, col].grid(False)
                axes[row, col].set(yticks=range(9), yticklabels=[label(r) for r in REGIMES] if col == 0 else [],
                                   xticks=[0, 5, 10], xticklabels=[0, .5, 1], title=f"{freq} Hz" if row == 0 else "")
                if col == 0:
                    axes[row, col].set_ylabel(title)
                if row == len(metrics) - 1:
                    axes[row, col].set_xlabel("CV")
            colorbar = fig.colorbar(image, ax=axes[row, :], shrink=.75,
                                   label=title + (" (log above 1)" if metric != "index_fraction" else ""))
            if metric != "index_fraction":
                maximum = recruitment_max if metric == "active_axon_count" else spikes_max
                ticks = [0, 1] + [10 ** k for k in range(1, int(np.floor(np.log10(maximum))) + 1)]
                if maximum not in ticks:
                    ticks.append(maximum)
                colorbar.set_ticks(ticks)
                colorbar.set_ticklabels([f"{value:g}" for value in ticks])
        fig.suptitle("Responses at each CV's regime amplitudes" if mode == "true" else "Responses at CV0-anchored regime amplitudes")
        self.table(frame.groupby(["freq", "cv", "thresh_type"])[metrics].mean().reset_index(), f"source_{mode}_population_summary")
        return self.figure(fig, stem or ("suppfig3_true_responses" if mode == "true" else "suppfig4_CV0_responses"))
