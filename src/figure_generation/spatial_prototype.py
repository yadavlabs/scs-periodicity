"""S5 layout prototype: selected spatial examples plus all-repetition summaries."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from figure_generation.figure_model import REGIMES, COLORS, label
from figure_generation.spatial_exploration import (
    digest, extract, grid_panel, rank_recruitment_repetitions, recruitment_onset,
)

FREQUENCIES = [10, 20, 50, 100, 200]
CVS = np.round(np.linspace(0, 1, 11), 1).tolist()
MODES = [('true', 'True-CV'), ('cv0', 'CV0-based')]


def count_equivalent_regimes(counts, n_input_pulses, imax_reference):
    """Nearest count anchors, NOT inferred axon-specific amplitude thresholds.

    Fixed targets follow detection_core._compute_regimes. The final anchor is
    the observed INDEX-axon Imax count at the defining CV (CV0 for anchored maps).
    Equal targets retain the first REGIMES label; midpoint ties go to the lower
    count. Preserve aliases and errors for an auditable approximate color key.
    """
    counts = np.asarray(counts, dtype=float)
    if (not np.isfinite(counts).all() or (counts < 0).any() or
            not np.isfinite(n_input_pulses) or n_input_pulses < 1 or
            not np.isfinite(imax_reference) or imax_reference < 1):
        raise ValueError('Invalid counts for regime-equivalent colors')
    targets = dict(I_first=1, I_volley5=5, I_volley10=10)
    targets.update({f'I_{pct}': max(1, int(np.ceil(pct / 100 * n_input_pulses)))
                    for pct in [5, 10, 15, 25, 50]})
    targets['I_all'] = float(imax_reference)
    if imax_reference <= max(targets[r] for r in REGIMES if r != 'I_all'):
        raise ValueError('Imax count reference must exceed the fixed-count targets')
    anchors = sorted(set(targets.values()))
    chosen = np.abs(counts[:, None] - np.asarray(anchors)[None, :]).argmin(axis=1)
    records = []
    for count, idx in zip(counts, chosen):
        target = anchors[idx]
        aliases = [r for r in REGIMES if targets[r] == target]
        records.append(dict(n_spikes=count, equivalent_regime=aliases[0] if count > 0 else 'none',
                            equivalent_aliases=' / '.join(aliases) if count > 0 else '',
                            reference_count=target if count > 0 else 0,
                            absolute_count_error=abs(count - target) if count > 0 else 0,
                            imax_reference_count=imax_reference,
                            above_imax_reference=bool(count > imax_reference)))
    return pd.DataFrame(records)


def summarize_trials(trials, frequencies=FREQUENCIES, cvs=CVS, regimes=REGIMES, n_reps=100):
    """Require a complete balanced grid; summarize trials, not pooled axon rows."""
    keys = ['mode', 'freq', 'cv', 'thresh_type']
    expected = pd.MultiIndex.from_product(
        [['true', 'cv0'], frequencies, cvs, regimes], names=keys)
    counts = trials.groupby(keys).size().reindex(expected)
    if (len(trials) != len(expected) * n_reps or counts.isna().any() or
            not counts.eq(n_reps).all() or trials.duplicated(keys + ['rep_index']).any()):
        raise ValueError('Incomplete or duplicate population summary conditions')
    if not trials.groupby(keys).rep_index.apply(lambda x: set(x) == set(range(n_reps))).all():
        raise ValueError('Population summary repetition identities differ across conditions')
    if (not trials.active_axon_count.between(0, 420).all() or
            not np.isfinite(trials.population_spikes).all() or
            (trials.population_spikes < trials.active_axon_count).any()):
        raise ValueError('Invalid population summary counts')
    return trials.groupby(keys, as_index=False).agg(
        median_active_axons=('active_axon_count', 'median'),
        median_population_spikes=('population_spikes', 'median'),
        n_repetitions=('rep_index', 'size'))


def draw_curve_panel(fig, slot, summaries, freq, metric, title, first_column, show_x_labels=True,
                     label_peak=False):
    """Log-count curves with a distinct non-log strip for exact zero medians."""
    grid = slot.subgridspec(2, 1, height_ratios=[1, .08], hspace=.04)
    ax = fig.add_subplot(grid[0])
    zero_ax = fig.add_subplot(grid[1], sharex=ax)
    for regime in REGIMES:
        for mode, _ in MODES:
            subset = summaries[summaries.freq.eq(freq) & summaries['mode'].eq(mode) &
                               summaries.thresh_type.eq(regime)].set_index('cv').reindex(CVS)
            values = subset[metric].to_numpy()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Invalid or missing values in population curves')
            marker = 'o' if mode == 'true' else 's'
            style = dict(color=COLORS[regime], marker=marker, markersize=2.6,
                         markerfacecolor=COLORS[regime] if mode == 'true' else 'white',
                         markeredgewidth=.7)
            # NaNs break curves at zero; never replace zero with a pseudocount.
            ax.plot(CVS, np.where(values > 0, values, np.nan),
                    linestyle='-' if mode == 'true' else '--', linewidth=1.4, **style)
            zeros = np.asarray(CVS)[values == 0]
            zero_ax.plot(zeros, np.full(len(zeros), .5), linestyle='none', **style)
    # Use all regimes and both amplitude conventions to set this frequency's
    # limits. Zero medians remain in the separate strip, not in the log limits.
    local = summaries.loc[summaries.freq.eq(freq), metric].to_numpy()
    positive = local[local > 0]
    lower, upper = (positive.min() * .8, positive.max() * 1.2) if len(positive) else (.8, 1.2)
    ax.set(yscale='log', ylim=(lower, upper), xlim=(-.03, 1.03))
    if metric == 'median_active_axons':
        ticks = [tick for tick in [1, 3, 10, 30, 100, 300, 420] if lower <= tick <= upper]
        if label_peak and len(positive):
            peak = float(positive.max())
            ax.set_ylim(lower, peak * 1.06)
            # Avoid nearly coincident ordinary and peak tick labels on log axes.
            ticks = [tick for tick in ticks if tick < peak / 1.25] + [peak]
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}') if label_peak else ScalarFormatter())
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=8 if label_peak and metric == 'median_active_axons' else 9, labelleft=True,
                   pad=1 if label_peak and metric == 'median_active_axons' else 3.5)
    ax.grid(axis='y', which='major', alpha=.22, linewidth=.6)
    if first_column:
        ax.set_ylabel(f'{title}\n(log scale)', fontsize=11)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    zero_ax.set(ylim=(0, 1), yticks=[.5], yticklabels=['0'])
    zero_ax.set_facecolor('#f0f0f0')
    zero_ax.set_xticks(np.arange(0, 1.01, .2))
    zero_ax.set_xticks(CVS, minor=True)
    zero_ax.tick_params(labelsize=9, axis='x', length=3, labelbottom=show_x_labels)
    zero_ax.tick_params(axis='y', length=0, labelsize=8)
    zero_ax.set_xlabel('Coefficient of Variation (CV)' if show_x_labels else '', fontsize=10)
    zero_ax.grid(False)
    for spine in ['top', 'right', 'left']:
        zero_ax.spines[spine].set_visible(False)
    return ax, zero_ax


def draw_prototype(data, summaries, output, summary_style='heatmaps', ring_colors='intensity'):
    """Five frequency columns; three paired map rows, then two paired heatmap rows."""
    compact = summary_style == 'curves'
    categorical = ring_colors == 'count-equivalent'
    if categorical and not compact:
        raise ValueError('Count-equivalent prototype uses the curve layout')
    fig = plt.figure(figsize=(24, 13 if categorical else 15.5) if compact else (27, 20))
    outer = fig.add_gridspec(3, 5, left=.065 if compact else .075, right=.855 if compact else .92,
                            bottom=.065 if compact else .12, top=.95 if compact else .93,
                            height_ratios=[3.2 if categorical else 3.9, 1, 1], hspace=.10 if compact else .35,
                            wspace=.09 if compact else .19)
    intensity_norm = Normalize(0, 1)
    intensity_cmap = plt.get_cmap('viridis').copy()
    intensity_cmap.set_over('#d62728')
    heat_cmap = plt.get_cmap('viridis').copy()
    heat_cmap.set_bad('#eeeeee')
    spatial_axes, heat_axes, curve_axes, panel_records, color_records = {}, {}, {}, [], []
    for col, freq in enumerate(FREQUENCIES):
        reps = data.loc[data.freq.eq(freq), 'rep_index'].unique()
        if len(reps) != 1:
            raise ValueError(f'Expected one selected repetition at {freq} Hz')
        rep = int(reps[0])
        groups = outer[0, col].subgridspec(3, 1, hspace=.10 if categorical else .26)
        for row, cv in enumerate([0., .5, 1.]):
            paired = groups[row].subgridspec(2, 1, hspace=.18 if categorical else .23)
            for mode_row, (mode, mode_label) in enumerate(MODES):
                p = data[data.freq.eq(freq) & np.isclose(data.cv, cv) &
                         data.thresh_type.eq('I_all') & data['mode'].eq(mode)]
                if len(p) != 420 or p.target_axon_id.nunique() != 420:
                    raise ValueError(f'Incomplete Imax map: {freq}/{cv}/{mode}')
                active, spikes = int((p.n_spikes > 0).sum()), int(p.n_spikes.sum())
                colors = {r.target_axon_id: 'white' if r.n_spikes == 0 else
                          intensity_cmap(intensity_norm(r.n_spikes / r.n_input_pulses))
                          for r in p.itertuples()}
                if categorical:
                    reference_cv = cv if mode == 'true' else 0.
                    reference = data[data.freq.eq(freq) & np.isclose(data.cv, reference_cv) &
                                     data.thresh_type.eq('I_all') & data['mode'].eq(mode) &
                                     data.target_axon_id.eq('S01_140um_ph1')]
                    if len(reference) != 1 or p.n_input_pulses.nunique() != 1:
                        raise ValueError('Ambiguous index Imax color reference')
                    classes = count_equivalent_regimes(p.n_spikes.to_numpy(), p.n_input_pulses.iloc[0],
                                                       reference.n_spikes.iloc[0])
                    classes['target_axon_id'] = p.target_axon_id.to_numpy()
                    colors = {r.target_axon_id: 'white' if r.equivalent_regime == 'none' else COLORS[r.equivalent_regime]
                              for r in classes.itertuples()}
                    color_records.append(classes.assign(freq=freq, cv=cv, mode=mode, rep_index=rep,
                                                        reference_cv=reference_cv, n_input_pulses=p.n_input_pulses.iloc[0]))
                ax = fig.add_subplot(paired[mode_row])
                title = f'{mode_label} | {p.amp_uA.iloc[0]:.1f} µA | {active} axons | {spikes:,} spikes'
                grid_panel(ax, p, colors, title, dense=True, left_phase='ph1' if compact else 'ph0',
                           radius=.48 if categorical else .43, depth_spacing=1.10 if categorical else 1.3,
                           tight_bounds=categorical)
                if categorical:
                    # Extend maps into the otherwise unused y-label gutter of
                    # the plots below, retaining their common right boundary.
                    slot = ax.get_position(original=True)
                    ax.set_position([slot.x0 - .004, slot.y0, slot.width + .004, slot.height])
                ax.set_title(title, fontsize=9, pad=2)
                ax.tick_params(labelbottom=row == 2 and mode_row == 1, labelleft=col == 0,
                               labelsize=7)
                if col == 0:
                    ax.set_yticklabels(['S', 'M', 'D'])
                ax.set_xlabel('Grid column (schematic)' if row == 2 and mode_row == 1 else '', fontsize=8)
                spatial_axes[row, col, mode_row] = ax
                panel_records.append(dict(freq=freq, cv=cv, mode=mode, rep_index=rep,
                                          amplitude_uA=p.amp_uA.iloc[0], active_axons=active,
                                          population_spikes=spikes))
        box = outer[0, col].get_position(fig)
        fig.text(box.x0 + box.width / 2, .975 if compact else .955,
                 f'{freq} Hz', ha='center', fontsize=17, weight='bold')
        if not compact:
            fig.text(box.x0 + box.width / 2, .942, f'Selected rep index {rep}', ha='center', fontsize=10)

    metrics = [('median_active_axons', 'Recruited axons', LogNorm(1, 420), [1, 3, 10, 30, 100, 420]),
               ('median_population_spikes', 'Population spikes',
                LogNorm(1, 10 ** np.ceil(np.log10(max(10, summaries.median_population_spikes.max())))),
                [1, 10, 100, 1000, 10000, 100000])]
    for row, (metric, title, norm, ticks) in enumerate(metrics, start=1):
        for col, freq in enumerate(FREQUENCIES):
            if summary_style == 'curves':
                curve_axes[row, col] = draw_curve_panel(
                    fig, outer[row, col], summaries, freq, metric, title,
                    first_column=col == 0, show_x_labels=row == 2)
                continue
            pair = outer[row, col].subgridspec(1, 2, wspace=.10)
            for j, (mode, mode_label) in enumerate(MODES):
                ax = fig.add_subplot(pair[j])
                frame = summaries[summaries.freq.eq(freq) & summaries['mode'].eq(mode)]
                values = frame.pivot(index='thresh_type', columns='cv', values=metric).reindex(
                    index=REGIMES, columns=CVS).to_numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Missing values in population heatmap')
                ax.imshow(np.ma.masked_equal(values, 0), norm=norm, cmap=heat_cmap,
                          aspect='auto', interpolation='nearest', extent=[-.05, 1.05, 8.5, -.5])
                ax.set_yticks(range(9), [label(r) for r in REGIMES])
                ax.tick_params(axis='y', labelleft=col == 0 and j == 0, length=0, labelsize=10)
                ax.set_xticks([0, .5, 1], ['0', '0.5', '1'])
                ax.set_xticks(CVS, minor=True)
                ax.tick_params(axis='x', labelsize=9)
                ax.set_xlabel('CV', fontsize=10)
                ax.set_title(mode_label, fontsize=10, pad=6)
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_linewidth(.5)
                    spine.set_color('.55')
                heat_axes[row, col, j] = ax
        pos = outer[row, 0].get_position(fig)
        fig.text(.025, pos.y1 + (.008 if compact else .033),
                 'b' if row == 1 else 'c', fontsize=20, weight='bold')
        if summary_style == 'curves':
            continue
        fig.text(.075, pos.y1 + .033, f'{title} — median across all 100 repetitions', fontsize=13, weight='bold')
        color_ax = fig.add_axes([.94, pos.y0, .009, pos.height])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=heat_cmap), cax=color_ax,
                          ticks=[t for t in ticks if t <= norm.vmax])
        if metric == 'median_active_axons':
            cb.set_ticklabels([str(t) for t in ticks])
        cb.set_label(f'{title} (log color scale)', fontsize=11)
        cb.ax.tick_params(labelsize=9)

    fig.text(.025, .975 if compact else .955, 'a', fontsize=20, weight='bold')
    for row, cv in enumerate([0., .5, 1.]):
        upper = spatial_axes[row, 0, 0].get_position()
        lower = spatial_axes[row, 0, 1].get_position()
        fig.text(.029 if categorical else (.042 if compact else .018),
                 (upper.y1 + lower.y0) / 2, f'CV = {cv:g}', rotation=90,
                 ha='center', va='center', fontsize=14, weight='bold')
    if not categorical:
        spatial_color_ax = fig.add_axes([.885, .82, .10, .012] if compact else [.94, .63, .009, .18])
        cb = fig.colorbar(ScalarMappable(norm=intensity_norm, cmap=intensity_cmap),
                          cax=spatial_color_ax, extend='max', orientation='horizontal' if compact else 'vertical')
        if compact:
            spatial_color_ax.set_title('Axonal spikes /\nnominal input pulses', loc='left', fontsize=11, pad=12)
            cb.set_ticks([0, .5, 1])
        else:
            cb.set_label('Axonal spikes / nominal input pulses', fontsize=11)
        cb.ax.tick_params(labelsize=9)
        fig.text(.885 if compact else .934, .76 if compact else .59,
                 'White: no spikes\nRed: >1 spike/pulse', fontsize=10 if compact else 9)
    if not compact:
        fig.suptitle('S5 prototype: spatial responses at $I_{max}$ and population-wide recruitment/output',
                     y=.99, fontsize=19)
    if summary_style == 'curves':
        # Equal-aspect ring maps may occupy less than their GridSpec slot width.
        # Align the actual b/c plotting boxes (and zero strips) to those maps.
        for (row, col), axes in curve_axes.items():
            if categorical:
                # Here maps deliberately extend left into the y-label gutter;
                # b/c retain the narrower curve boxes so tick labels stay clear.
                continue
            spatial = spatial_axes[0, col, 0].get_position()
            for ax in axes:
                box = ax.get_position()
                ax.set_position([spatial.x0, box.y0, spatial.width, box.height])
        handles = [Line2D([], [], color=COLORS[r], lw=2, label=label(r)) for r in REGIMES]
        if categorical:
            handles.append(Patch(facecolor='white', edgecolor='.6', label='No spikes (a)'))
        fig.legend(handles=handles,
                   loc='upper left', bbox_to_anchor=(.88, .85 if categorical else .68), ncol=1, frameon=False,
                   title='Operating regime\n(a: count-equivalent)' if categorical else 'Operating regime',
                   fontsize=11, title_fontsize=11)
        fig.legend(handles=[Line2D([], [], color='.25', lw=1.5, marker='o', ms=3, label='True-CV'),
                            Line2D([], [], color='.25', lw=1.5, linestyle='--', marker='s',
                                   markerfacecolor='white', ms=3, label='CV0-based')],
                   loc='upper left', bbox_to_anchor=(.88, .38), frameon=False, fontsize=11,
                   title='Amplitude basis')
    notes = [
        'a: High-recruitment examples, not typical responses. One rep per frequency selected by maximum mean CV0-based recruitment across 11 CVs and I₁₀/I₅₀/Imax; same rep used for both amplitude conventions.',
        'Rings inner→outer: 5.7, 8.7, 10, 11.5, 12.8, 14 µm. Left/right halves: nodal phases 0/1. Black outline: index axon. S/M/D: superficial/middle/deep; schematic locations, not anatomical coordinates.',
        'b–c: All repetitions and all nine index-axon regimes; no example-selection filter. Shared logarithmic scale within each quantity; gray denotes zero. Recruitment is the number of modeled axons with ≥1 spike (of 420).',
        'True-CV: amplitudes defined separately at each CV. CV0-based: amplitudes anchored at CV=0. Population spike counts are summed over the modeled axons before taking the median across repetitions.'
    ]
    if summary_style == 'curves':
        notes[1] = notes[1].replace('nodal phases 0/1', 'nodal phases 1/0')
        notes[2] = ('b–c: Medians across all 100 repetitions and all nine regimes; no example-selection filter. '
                    'Frequency-specific log y-ranges. Exact zeros appear only in the separate gray strips; lines break at zero. No pseudocounts added.')
    if categorical:
        notes.append('Panel a uses APPROXIMATE nearest spike-count-equivalent categories, not each axon\'s calibrated operating regime. '
                     'Fixed targets: 1, 5, 10 spikes and ceil(5/10/15/25/50% of nominal input pulses). '
                     'Imax reference: observed index-axon count at its Imax amplitude, at the displayed CV for true-CV maps and at CV0 for anchored maps. '
                     'Equal-count aliases use the first canonical label; equal distances use the lower count. '
                     'Counts above the reference remain in the top category and are flagged in the per-axon export; zero is white. '
                     'This is a count-only visualization; it does not establish an individual axon\'s plateau or temporal firing pattern.')
    if not compact:
        for i, note in enumerate(notes):
            fig.text(.075, .072 - i * .014, note, fontsize=9)
    stem = 'S5_prototype_spatial_and_population_summary' + ('_curves' if summary_style == 'curves' else '')
    if categorical:
        stem += '_regime_colors'
    (output / '_data').mkdir(parents=True, exist_ok=True)
    (output / '_data' / f'{stem}_caption_notes.txt').write_text('\n\n'.join(notes), encoding='utf-8')
    if categorical:
        pd.concat(color_records, ignore_index=True).to_csv(output / '_data' / f'{stem}_axon_classes.csv', index=False)
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'{stem}.{ext}', dpi=220, bbox_inches='tight')
        print(f'Saved {stem}.{ext}', flush=True)
    plt.close(fig)
    return pd.DataFrame(panel_records)


def onset_cv_tag(cvs):
    """Stable distinct artifact names for single- and multiple-CV maps."""
    if not cvs or len(set(cvs)) != len(cvs) or not all(cv in CVS for cv in cvs):
        raise ValueError('Choose distinct CVs from the simulated CV grid')
    return 'CV' + '_'.join(f'{cv:g}' for cv in cvs).replace('.', 'p')


def draw_recruitment_onset(data, summaries, output, cv=1., cvs=None):
    """Paired spatial rows: regime at the lowest tested recruiting amplitude."""
    cvs = [cv] if cvs is None else list(cvs)
    tag = onset_cv_tag(cvs)
    selected = data[data.cv.isin(cvs)]
    onset = recruitment_onset(selected)
    stem = f'S5_prototype_recruitment_onset_{tag}'
    onset.to_csv(output / '_data' / f'{stem}_axon_onsets.csv', index=False)
    fig = plt.figure(figsize=(24, 7.9 + 2.55 * (len(cvs) - 1)))
    outer = fig.add_gridspec(3, 5, left=.065, right=.855, bottom=.065, top=.95,
                            height_ratios=[1.15 + 1.025 * (len(cvs) - 1), 1, 1], hspace=.10, wspace=.09)
    map_axes, records = {}, []
    for col, freq in enumerate(FREQUENCIES):
        cv_groups = outer[0, col].subgridspec(len(cvs), 1, hspace=.10)
        for cv_row, displayed_cv in enumerate(cvs):
            paired = cv_groups[cv_row].subgridspec(2, 1, hspace=.18)
            for row, (mode, mode_label) in enumerate(MODES):
                p = onset[onset.freq.eq(freq) & onset['mode'].eq(mode) & np.isclose(onset.cv, displayed_cv)]
                if len(p) != 420 or p.target_axon_id.nunique() != 420 or p.rep_index.nunique() != 1:
                    raise ValueError('Incomplete or mixed-repetition onset map')
                active = int(p.first_regime.ne('none').sum())
                lost = int(p.loses_recruitment_at_higher_tested_amplitude.sum())
                colors = {r.target_axon_id: 'white' if r.first_regime == 'none' else COLORS[r.first_regime]
                          for r in p.itertuples()}
                ax = fig.add_subplot(paired[row])
                title = f'{mode_label} | {active}/420 recruited'
                grid_panel(ax, p, colors, title, dense=True, left_phase='ph1', radius=.48,
                           depth_spacing=1.10, tight_bounds=True)
                slot = ax.get_position(original=True)
                ax.set_position([slot.x0 - .004, slot.y0, slot.width + .004, slot.height])
                ax.set_title(title, fontsize=9, pad=2)
                show_grid_labels = cv_row == len(cvs) - 1 and row == 1
                ax.tick_params(labelbottom=show_grid_labels, labelleft=col == 0, labelsize=7)
                if col == 0:
                    ax.set_yticklabels(['S', 'M', 'D'])
                ax.set_xlabel('Grid column (schematic)' if show_grid_labels else '', fontsize=8)
                map_axes[cv_row, row, col] = ax
                records.append(dict(freq=freq, cv=displayed_cv, mode=mode, rep_index=int(p.rep_index.iloc[0]),
                                    recruited_in_tested_set=active, later_lose_recruitment=lost))
        box = outer[0, col].get_position(fig)
        fig.text(box.x0 + box.width / 2, .975, f'{freq} Hz', ha='center', fontsize=17, weight='bold')
        for row, (metric, title) in enumerate([
                ('median_active_axons', 'Recruited axons'), ('median_population_spikes', 'Population spikes')], 1):
            draw_curve_panel(fig, outer[row, col], summaries, freq, metric, title,
                             first_column=col == 0, show_x_labels=row == 2)
    fig.text(.025, .975, 'a', fontsize=20, weight='bold')
    for cv_row, displayed_cv in enumerate(cvs):
        top, bottom = map_axes[cv_row, 0, 0].get_position(), map_axes[cv_row, 1, 0].get_position()
        fig.text(.029, (top.y1 + bottom.y0) / 2, f'CV = {displayed_cv:g}', rotation=90,
                 ha='center', va='center', fontsize=14, weight='bold')
    for row, letter in [(1, 'b'), (2, 'c')]:
        box = outer[row, 0].get_position(fig)
        fig.text(.025, box.y1 + .008, letter, fontsize=20, weight='bold')
    handles = [Line2D([], [], color=COLORS[r], lw=2, label=label(r)) for r in REGIMES]
    handles.append(Patch(facecolor='white', edgecolor='.6', label='Not recruited (a)'))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.88, .93), frameon=False,
               title='Index-axon regime\n(a: first recruitment)', fontsize=11, title_fontsize=11)
    fig.legend(handles=[Line2D([], [], color='.25', lw=1.5, marker='o', ms=3, label='True-CV'),
                        Line2D([], [], color='.25', lw=1.5, linestyle='--', marker='s',
                               markerfacecolor='white', ms=3, label='CV0-based')],
               loc='upper left', bbox_to_anchor=(.88, .36), frameon=False, fontsize=11,
               title='Amplitude basis')
    notes = (
        f'Panel a: CVs={cvs}, one selected high-CV0-recruitment repetition per frequency, matched across modes and CVs. '
        'Each half-ring is colored by the index-axon regime at the lowest tested amplitude producing at least one spike in that axon. '
        'All nine regimes are searched in increasing ACTUAL amplitude, not nominal label order. '
        'Equal-amplitude activating aliases use the first canonical regime label; all aliases are exported. '
        'White denotes no spikes at any tested regime amplitude. These are discrete tested-amplitude recruitment onsets, '
        'not continuous activation thresholds, firing-intensity classes, or proof of sustained recruitment at higher amplitudes. '
        'Any later loss of recruitment is flagged in the export.\n\n'
        'True-CV uses amplitudes defined at the displayed CV; CV0-based uses the CV0-defined amplitudes with the displayed CV input. '
        'Rings inner to outer: 5.7, 8.7, 10, 11.5, 12.8, 14 µm. Left/right: phases 1/0; black outline is the index axon. '
        'Schematic superficial/middle/deep grid, not anatomical coordinates.\n\n'
        'Panels b/c are unchanged: median recruited axons and total population spikes across all 100 repetitions, '
        'all CVs and regimes, with frequency-specific logarithmic ranges and separate zero strips. '
        'Solid/filled circles: true-CV; dashed/open squares: CV0-based. No pseudocounts.'
    )
    (output / '_data' / f'{stem}_caption_notes.txt').write_text(notes, encoding='utf-8')
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'{stem}.{ext}', dpi=220, bbox_inches='tight')
        print(f'Saved {stem}.{ext}', flush=True)
    plt.close(fig)
    return pd.DataFrame(records)


def main():
    plt.switch_backend('Agg')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-style', choices=['heatmaps', 'curves'], default='heatmaps')
    parser.add_argument('--ring-colors', choices=['intensity', 'count-equivalent'], default='intensity')
    parser.add_argument('--recruitment-onset', action='store_true', help='Alternative with paired onset map rows')
    cv_options = parser.add_mutually_exclusive_group()
    cv_options.add_argument('--map-cv', type=float, default=1., choices=CVS, help='CV for the single onset row (default: 1)')
    cv_options.add_argument('--map-cvs', type=float, nargs='+', choices=CVS, help='CV rows, e.g. 0 0.5 1')
    parser.add_argument('--lock-count-equivalent', action='store_true', help='Save a frozen count-equivalent layout in its own folder')
    args = parser.parse_args()
    map_cvs = args.map_cvs if args.map_cvs is not None else [args.map_cv]
    map_tag = onset_cv_tag(map_cvs)
    if args.ring_colors == 'count-equivalent' and args.summary_style != 'curves':
        parser.error('--ring-colors count-equivalent requires --summary-style curves')
    if args.lock_count_equivalent and (args.recruitment_onset or args.ring_colors != 'count-equivalent'):
        parser.error('--lock-count-equivalent requires count-equivalent colors and no onset alternative')
    root = Path(__file__).resolve().parents[2]
    base = root / 'results/figures/exploratory_s5'
    output = base / 'consolidated_prototype'
    if args.lock_count_equivalent:
        output = output / 'locked_count_equivalent'
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite locked-version directory: {output}')
    elif args.recruitment_onset:
        output = output / 'recruitment_onset'
    (output / '_data').mkdir(parents=True, exist_ok=True)
    true_path = root / 'results/detection/true_population_trial_summary.csv'
    cv0_path = root / 'data/features/population_cv0_based_features.csv'
    cv0 = pd.read_csv(cv0_path)
    ranking = rank_recruitment_repetitions(cv0, FREQUENCIES, CVS)
    winners = ranking[ranking.selected]
    repetitions = {int(r.freq): int(r.rep_index) for r in winners.itertuples()}
    data = extract(root, base / 'max_cv0_recruitment/_data', FREQUENCIES, CVS, repetitions)
    true = pd.read_csv(true_path).rename(columns={'total_population_spikes': 'population_spikes'})
    cv0 = cv0.rename(columns={'source_total_population_spikes': 'population_spikes'})
    cols = ['freq', 'cv', 'rep_index', 'thresh_type', 'active_axon_count', 'population_spikes']
    trials = pd.concat([true[cols].assign(mode='true'), cv0[cols].assign(mode='cv0')], ignore_index=True)
    summaries = summarize_trials(trials)
    summaries.to_csv(output / '_data/all_repetition_medians.csv', index=False)
    ranking.to_csv(output / '_data/repetition_ranking.csv', index=False)
    print('Validated 99,000 population trials; rendering prototype.', flush=True)
    if args.recruitment_onset:
        panels = draw_recruitment_onset(data, summaries, output, cvs=map_cvs)
        panels.to_csv(output / '_data' / f'spatial_panels_{map_tag}.csv', index=False)
    else:
        draw_prototype(data, summaries, output, summary_style=args.summary_style,
                       ring_colors=args.ring_colors).to_csv(output / '_data/spatial_panels.csv', index=False)
    provenance_name = f'provenance_{map_tag}.json' if args.recruitment_onset else 'provenance.json'
    (output / '_data' / provenance_name).write_text(json.dumps(dict(
        repetitions=repetitions, spatial_cvs=map_cvs if args.recruitment_onset else [0, .5, 1],
        spatial_regime='lowest tested recruiting amplitude across all regimes' if args.recruitment_onset else 'I_all',
        selection='Maximum mean CV0-based active axons over all CVs and I_10/I_50/I_all',
        summaries='Median of trial totals across all 100 repetitions per condition',
        sources={str(p): digest(p) for p in [true_path, cv0_path]},
        index_sha256=digest(root / 'data/detection/index_regime_amplitudes.csv')),
        indent=2), encoding='utf-8')
    if args.lock_count_equivalent:
        files = [p for p in output.rglob('*') if p.is_file()]
        lock = dict(status='Locked for review; not promoted into manuscript notebook',
                    ring_left_extension_figure_fraction=.004, arguments=vars(args),
                    artifacts={str(p.relative_to(output)): digest(p) for p in files},
                    renderer_sha256=digest(Path(__file__)),
                    spatial_helper_sha256=digest(Path(__file__).with_name('spatial_exploration.py')))
        (output / 'LOCKED.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
        print(f'Locked snapshot: {output}', flush=True)


if __name__ == '__main__':
    main()
