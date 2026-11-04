"""Promoted compact S5 and retained exploratory population layouts."""
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.ticker import LogLocator, FuncFormatter, LogFormatterSciNotation

from figure_generation.figure_model import COLORS, REGIMES, label
from .conditional_spatial import DISPLAY_FREQUENCIES, conditional_faces, prepare_data
from .spatial_exploration import grid_panel
from .spatial_prototype import summarize_trials


def load_trials(root):
    true = pd.read_csv(root / 'results/detection/true_population_trial_summary.csv')
    true['population_spikes'] = true.total_population_spikes
    cv0 = pd.read_csv(root / 'data/features/population_cv0_based_features.csv')
    cv0['population_spikes'] = cv0.source_total_population_spikes
    columns = ['freq', 'cv', 'rep_index', 'thresh_type', 'active_axon_count', 'population_spikes']
    trials = pd.concat([true[columns].assign(mode='true'), cv0[columns].assign(mode='cv0')], ignore_index=True)
    summarize_trials(trials)  # Validate balanced conditions and all 100 repetitions.
    return trials[trials.freq.isin(DISPLAY_FREQUENCIES)].copy()


def choose_cvs(trials):
    """Maximize mean recruitment, not a favorable individual example or site coverage."""
    scores = trials[trials.thresh_type.eq('I_all')].groupby(['mode', 'freq', 'cv']).agg(
        mean_recruited_axons=('active_axon_count', 'mean'),
        median_recruited_axons=('active_axon_count', 'median'),
        max_recruited_axons=('active_axon_count', 'max'), n=('rep_index', 'nunique')).reset_index()
    ranked = scores[scores['mode'].eq('true')].sort_values(
        ['freq', 'mean_recruited_axons', 'cv'], ascending=[True, False, True])
    chosen = ranked.drop_duplicates('freq').set_index('freq').cv.to_dict()
    scores['selected_matched_cv'] = scores.cv.eq(scores.freq.map(chosen))
    return chosen, scores


def validate_cv0_maps(stats):
    columns = ['target_axon_id', 'target_cutline_id', 'target_diameter_um', 'n_repetitions',
               'n_active_repetitions', 'sum_spikes', 'conditional_mean_spikes',
               'recruitment_probability', 'equivalent_regime']
    zero = stats[stats.cv.eq(0)].set_index(['mode', 'freq', 'target_axon_id'], drop=False)
    pd.testing.assert_frame_equal(zero.loc['true', columns].sort_index(),
                                  zero.loc['cv0', columns].sort_index())


def fill_spatial_column(ax):
    """Use the full assigned column width without stretching circular rings.

    Equal aspect otherwise shrinks these axes horizontally to fit their short
    subplot slots. Expand only their height, about the existing row center.
    """
    box = ax.get_position(original=True)
    fig = ax.figure
    data_ratio = np.ptp(ax.get_ylim()) / np.ptp(ax.get_xlim())
    height = box.width * fig.get_figwidth() * data_ratio / fig.get_figheight()
    ax.set_position([box.x0, (box.y0 + box.y1 - height) / 2, box.width, height])


def upper_count_tick(maximum):
    """Next 1/2/5-decade tick above every observation (not just the median)."""
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('A positive finite maximum is required')
    target = maximum * 1.03
    decade = 10. ** np.floor(np.log10(target))
    return next(multiplier * decade for multiplier in [1, 2, 5, 10]
                if multiplier * decade >= target)


def draw_median_segments(ax, zero, medians, color, *, connect_transitions=False):
    """Plot measured medians, with optional visual guides across the zero/log break."""
    positive = medians.gt(0)
    ax.plot(medians.index, medians.where(positive), color=color, lw=1.25)
    # An isolated positive median otherwise has no visible line segment.
    isolated = positive & ~positive.shift(1, fill_value=False) & ~positive.shift(-1, fill_value=False)
    if isolated.any():
        ax.plot(medians.index[isolated], medians[isolated], linestyle='none', marker='_',
                color=color, ms=5)
    zero.plot(medians.index, medians.where(medians.eq(0)), color=color,
              lw=1.25, marker='_', ms=5)
    if connect_transitions:
        points = list(medians.items())
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            if not np.isfinite([y0, y1]).all() or not ((y0 == 0 < y1) or (y1 == 0 < y0)):
                continue
            # Keep zero on its own linear strip. A straight display-space guide
            # joins adjacent sampled CVs without replacing zero by a pseudocount.
            guide = ConnectionPatch((x0, y0), (x1, y1), coordsA='data', coordsB='data',
                                    axesA=zero if y0 == 0 else ax,
                                    axesB=zero if y1 == 0 else ax,
                                    color=color, lw=1.25, arrowstyle='-', shrinkA=0, shrinkB=0,
                                    clip_on=False, zorder=3)
            ax.figure.add_artist(guide)


def repetition_panel(fig, slot, trials, freq, mode, metric, show_x, first_column, *, separate_ranges=False):
    """Unfiltered observations on log axes; zeros get a separate linear strip."""
    pair = slot.subgridspec(2, 1, height_ratios=[1, .085], hspace=.02)
    ax = fig.add_subplot(pair[0]); zero = fig.add_subplot(pair[1], sharex=ax)
    both = trials[trials.freq.eq(freq)]
    source = both[both['mode'].eq(mode)]
    limit_source = source if separate_ranges else both
    positive = limit_source.loc[limit_source[metric].gt(0), metric]
    ax.set_yscale('log')
    lower = max(.75, positive.min() * .8)
    top_tick = upper_count_tick(positive.max()) if separate_ranges else None
    ax.set_ylim(lower, top_tick * 1.08 if separate_ranges else positive.max() * 1.20)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,)))
    if separate_ranges:
        decades = 10. ** np.arange(np.ceil(np.log10(lower)), np.floor(np.log10(top_tick)) + 1)
        ax.set_yticks(decades)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        ax.tick_params(axis='y', which='minor', left=True, length=2.5, width=.6)
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True) if separate_ranges else
                                FuncFormatter(lambda value, _: f'{value:g}'))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    for k, regime in enumerate(REGIMES):
        p = source[source.thresh_type.eq(regime)].sort_values(['cv', 'rep_index'])
        jitter = ((p.rep_index.to_numpy() * .61803398875) % 1 - .5) * .004
        x = p.cv.to_numpy() + (k - 4) * .005 + jitter
        positive_mask = p[metric].gt(0).to_numpy()
        ax.scatter(x[positive_mask], p.loc[positive_mask, metric], s=2.5,
                   color=COLORS[regime], alpha=.16, linewidths=0, rasterized=True)
        zero.scatter(x[~positive_mask], np.zeros((~positive_mask).sum()), s=2.5,
                     color=COLORS[regime], alpha=.16, linewidths=0, rasterized=True)
        medians = p.groupby('cv')[metric].median()
        draw_median_segments(ax, zero, medians, COLORS[regime], connect_transitions=separate_ranges)
    for target in [ax, zero]:
        target.set_xlim(-.04, 1.04)
        target.set_xticks(np.arange(0, 1.01, .1))
        target.tick_params(axis='both', labelsize=8, pad=2)
        target.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.grid(axis='y', which='major', alpha=.16)
    zero.set_ylim(-.6, .6)
    zero.set_yticks([0], ['0']); zero.set_facecolor('.97')
    zero.tick_params(axis='y', length=0)
    zero.tick_params(axis='x', labelbottom=show_x)
    zero.set_xticklabels([f'{cv:.1f}' if i % 2 == 0 else '' for i, cv in enumerate(np.arange(0, 1.01, .1))])
    if show_x:
        zero.set_xlabel('Coefficient of Variation (CV)', fontsize=9)
    if first_column:
        ax.set_ylabel('Recruited axons' if metric == 'active_axon_count' else 'Population spikes', fontsize=10)
        if separate_ranges:
            # Keep the axis title in a fixed lane beside the amplitude-basis label.
            ax.yaxis.set_label_coords(-.023 / ax.get_position().width, .5)
    return ax, zero


@plt.rc_context(plt.rcParamsDefault)
def prototype(stats, trials, chosen, compact=False):
    validate_cv0_maps(stats)
    map_rows = [('true', None, 'True-CV'), ('cv0', None, 'CV0-based')] if compact else [
        ('true', 0., 'True-CV / CV0-based'), ('true', .5, 'True-CV'), ('cv0', .5, 'CV0-based'),
        ('true', 1., 'True-CV'), ('cv0', 1., 'CV0-based')]
    fig = plt.figure(figsize=(17, 10.5 if compact else 13))
    outer = fig.add_gridspec(6, 4, left=.09, right=.99, bottom=.095, top=.925 if compact else .94,
                            height_ratios=[1.45 if compact else 3.9, .28, 1.2, 1.2, 1.2, 1.2],
                            hspace=.13, wspace=.12 if compact else .20)
    map_axes = {}
    for col, freq in enumerate(DISPLAY_FREQUENCIES):
        maps = outer[0, col].subgridspec(len(map_rows), 1, hspace=.16)
        for row, (mode, fixed_cv, mode_label) in enumerate(map_rows):
            cv = chosen[freq] if compact else fixed_cv
            p = stats[stats.freq.eq(freq) & stats.cv.eq(cv) & stats['mode'].eq(mode)]
            if len(p) != 420:
                raise ValueError(f'Missing validated spatial cache for {freq} Hz, CV {cv}')
            ax = fig.add_subplot(maps[row])
            grid_panel(ax, p, conditional_faces(p), '', dense=True, left_phase='ph1',
                       radius=.48, depth_spacing=1.10, tight_bounds=True)
            if compact:
                fill_spatial_column(ax)
            ax.set_title((f'CV = {cv:g}' if row == 0 else '') if compact else mode_label,
                         fontsize=10 if compact else 9, pad=3 if compact else 1)
            ax.tick_params(labelbottom=row == len(map_rows)-1, labelleft=compact or col == 0, labelsize=7)
            if compact or col == 0:
                ax.set_yticklabels(['S', 'M', 'D'])
            ax.set_xlabel('Column number' if row == len(map_rows)-1 else '', fontsize=8)
            map_axes[row, col] = ax
        box = outer[0, col].get_position(fig)
        header = f'{freq} Hz'
        fig.text((box.x0+box.x1)/2, .965, header, ha='center', weight='bold', fontsize=13)
        for row, (metric, mode) in enumerate([
                ('active_axon_count', 'true'), ('active_axon_count', 'cv0'),
                ('population_spikes', 'true'), ('population_spikes', 'cv0')], 2):
            repetition_panel(fig, outer[row, col], trials, freq, mode, metric, row == 5, col == 0,
                             separate_ranges=compact)
    panel_label_x = .030 if compact else .018
    row_label_x = .048 if compact else .035
    fig.text(panel_label_x, .965, 'a', fontsize=15, weight='bold')
    if compact:
        for row, text in enumerate(['True-CV', 'CV0-based']):
            box = map_axes[row, 0].get_position()
            fig.text(row_label_x, (box.y0 + box.y1) / 2, text, rotation=90, ha='center', va='center',
                     fontsize=10, weight='bold')
    if not compact:
        for cv, start, end in [(0, 0, 0), (.5, 1, 2), (1, 3, 4)]:
            top = map_axes[start, 0].get_position().y1
            bottom = map_axes[end, 0].get_position().y0
            fig.text(.054, (top+bottom)/2, f'CV = {cv:g}', rotation=90,
                     ha='center', va='center', weight='bold', fontsize=10)
    for row, text in [(2, 'True-CV'), (3, 'CV0-based'), (4, 'True-CV'), (5, 'CV0-based')]:
        box = outer[row, 0].get_position(fig)
        fig.text(row_label_x, (box.y0+box.y1)/2, text, rotation=90, ha='center', va='center', fontsize=10, weight='bold')
        if row in [2, 4]:
            fig.text(panel_label_x, box.y1, 'b' if row == 2 else 'c', fontsize=15, weight='bold')
    fig.legend(handles=[Patch(facecolor=COLORS[r], label=label(r)) for r in REGIMES],
               loc='lower center', bbox_to_anchor=(.53, .012), frameon=False, ncol=9, fontsize=10)
    return fig


def manuscript_figure(v2):
    """Promote the accepted compact layout as the notebook's sole S5 output."""
    stem = 'suppfig5_population_recruitment'
    output = v2.output / f'{stem}_data'
    stats, _ = prepare_data(v2.root, output)
    trials = load_trials(v2.root)
    chosen, scores = choose_cvs(trials)
    displayed_stats = stats[stats.cv.eq(stats.freq.map(chosen))].copy()
    data_dir = output / '_data'
    displayed_stats.to_csv(data_dir / 'compact_spatial_plot_data.csv', index=False)
    trials.to_csv(data_dir / 'compact_repetition_plot_data.csv', index=False)
    scores.to_csv(data_dir / 'compact_cv_selection_audit.csv', index=False)
    provenance_path = data_dir / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    provenance.update(layout='compact_repetition_spread', cached_source_cvs=provenance['map_cvs'],
                      map_cvs=sorted(set(chosen.values())), selected_cv_by_frequency=chosen,
                      selection='Maximum mean true-CV Imax active-axon count across all 100 repetitions; same CV for both amplitude bases')
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    (data_dir / 'caption_notes.txt').write_text(
        'Panel a: Imax responses at the CV maximizing mean true-CV recruited-axon count across all 100 '
        'repetitions (currently CV1 at 20/50/100/200 Hz); CV0-based maps use the same CV. This is not '
        'maximum occupied-location coverage or a maximal individual repetition. True-CV amplitudes '
        'are defined at the displayed CV; CV0-based amplitudes are anchored at CV0.\n\n'
        'Ring color: mean spikes conditional on at least one spike, mapped to nearest count-equivalent '
        'regime anchors, not independently calibrated axonal thresholds. Imax reference: mean index-axon '
        'count at the amplitude-defining CV. Opacity: active repetitions / 100. Never-active halves '
        'are unfilled and have undefined conditional means. Ensemble summary, not one simultaneous population. '
        'Six rings inner-to-outer: 5.7, 8.7, 10, 11.5, 12.8, 14 micrometers; left/right halves: phases 1/0. '
        'Black outline: index axon. S/M/D: superficial/middle/deep. Spatial positions and ring sizes are schematic.\n\n'
        'Panels b/c: all repetition-level recruited-axon counts and total population spike counts, with '
        'median lines, across all CVs and index regimes. True-CV/CV0-based occupy separate rows. Points '
        'have display-only horizontal offsets; medians use actual CVs. Frequency- and basis-specific '
        'log limits include every positive observation and an upper anchoring tick. Only powers of ten '
        'are labeled; intermediate ticks are unlabeled. Exact zeros use a separate linear strip. '
        'Straight guides join adjacent zero/positive medians across the axis break; these are not log '
        'interpolation through zero, and no pseudocounts are used. Individual repetitions are unconnected.',
        encoding='utf-8')
    fig = prototype(stats, trials, chosen, compact=True)
    return v2.figure(fig, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact-only', action='store_true')
    args = parser.parse_args()
    plt.switch_backend('Agg')
    root = Path(__file__).resolve().parents[2]
    out = root / 'results/figures/exploratory_s5/repetition_layouts'
    out.mkdir(parents=True, exist_ok=True)
    # Reuse cached aggregates without overwriting the promoted caption/provenance.
    cache_source = root / 'results/figures/suppfig5_population_recruitment_data/_data'
    (out / '_data').mkdir(exist_ok=True)
    for name in ['conditional_axon_statistics.csv', 'response_cache_manifest.json', 'trial_validation.csv']:
        if (cache_source / name).exists() and not (out / '_data' / name).exists():
            shutil.copy2(cache_source / name, out / '_data' / name)
    stats, _ = prepare_data(root, out)
    trials = load_trials(root)
    chosen, scores = choose_cvs(trials)
    scores.to_csv(out / 'cv_selection_audit.csv', index=False)
    trials.to_csv(out / 'repetition_plot_data.csv', index=False)
    stats.to_csv(out / 'spatial_plot_data.csv', index=False)
    for compact, stem in [(False, 'S5_merged_CV0_repetition_spread'), (True, 'S5_max_mean_recruitment_repetition_spread')]:
        if args.compact_only and not compact:
            continue
        fig = prototype(stats, trials, chosen, compact=compact)
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'{stem}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved {stem}', flush=True)
    (out / 'selection_and_caption_notes.json').write_text(json.dumps(dict(
        selected_cvs=chosen, criterion='Maximal mean recruited-axon count over all 100 true-CV repetitions at Imax',
        paired_comparison='CV0-based maps use the same selected CV, not separately selected maxima',
        cv0_merge='True-CV and CV0-based aggregate maps verified equal at CV=0',
        interpretation='Selection is not maximal occupied-site coverage or maximal single-repetition recruitment.',
        maps='Conditional positive spike-count mean colors; recruitment probability opacity; existing shared palette.',
        curves='All 100 repetitions per condition; median lines at true CV; horizontal display-only offsets.',
        y_axes='All positive points retained; separate zero strips. Compact: per-frequency/per-basis limits include a 1/2/5-decade upper tick above all observations, but only powers of ten are labeled; intermediate ticks are visible and unlabeled. Merged-CV0: shared limits across bases.',
        median_connections='Adjacent zero medians connect in the zero strip. Compact: straight display-space guides join adjacent zero/positive medians across the separate axes; they do not imply log interpolation through zero. No pseudocounts. Isolated positive medians have short tick markers. Individual repetitions are unconnected.'
    ), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
