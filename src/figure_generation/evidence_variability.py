"""S7 mean display and explicitly distinguished repetition-variability previews."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_generation.figure_model import REGIMES, FREQS, cv_ticks

RATE = dict(zip(REGIMES, ['#FF9F8A', '#FF7968', '#FF5A48', '#FF3B2F', '#E30613',
                          '#C70512', '#9F040E', '#77030B', '#500207']))
TIMING = dict(zip(REGIMES, ['#00D4FF', '#01BCEA', '#02A5D5', '#038DC0', '#0575AB',
                            '#065D95', '#074680', '#082E6B', '#091656']))
SPECS = [('rate_evidence_1ms', 'Rate evidence\n(spikes, 1 ms)', RATE, ':'),
         ('rate_evidence_10ms', 'Rate evidence\n(bursts, 10 ms)', RATE, '--'),
         ('timing_evidence_1ms', 'Timing evidence\n(spikes, 1 ms)', TIMING, ':'),
         ('timing_evidence_10ms', 'Timing evidence\n(bursts, 10 ms)', TIMING, '--')]


def evidence_summary(active):
    keys = ['freq', 'representative_regime', 'cv']
    if active.duplicated(keys + ['rep_index']).any():
        raise ValueError('Duplicate active representative/rep: would overweight aliases')
    parts = []
    for column, *_ in SPECS:
        part = active.groupby(keys)[column].agg(
            mean='mean', median='median', sem='sem', q25=lambda x: x.quantile(.25),
            q75=lambda x: x.quantile(.75), minimum='min', maximum='max', n='count').reset_index()
        parts.append(part.assign(evidence_metric=column))
    return pd.concat(parts, ignore_index=True)


def evidence_curves(v2, style='mean', stem=None):
    if style not in {'mean', 'median_iqr', 'mean_sem', 'individual_reps'}:
        raise ValueError(style)
    source = pd.read_csv(v2.root / 'data/features/population_cv0_template_evidence.csv')
    active = source[source.regime_active.astype(str).str.lower().isin(['true', '1', 'yes']) &
                    source.freq.isin(FREQS)].copy()
    summary = evidence_summary(active)
    if not summary.n.eq(100).all():
        raise ValueError('Expected all 100 repetitions per evidence condition')
    fig, axes = plt.subplots(4, 4, figsize=(17.5, 12), sharex=True, sharey=True)
    for row, (column, row_label, colors, ls) in enumerate(SPECS):
        for col, freq in enumerate(FREQS):
            ax = axes[row, col]
            panel = summary[summary.freq.eq(freq) & summary.evidence_metric.eq(column)]
            for k, regime in enumerate(REGIMES):
                p = panel[panel.representative_regime.eq(regime)].sort_values('cv')
                if p.empty:
                    continue
                center = p['median'] if style in ['median_iqr', 'individual_reps'] else p['mean']
                if style == 'median_iqr':
                    ax.fill_between(p.cv, p.q25, p.q75, color=colors[regime], alpha=.13, lw=0)
                elif style == 'mean_sem':
                    # An uncertainty-of-mean display, not repetition spread. No clipping.
                    ax.fill_between(p.cv, p['mean'] - p['sem'], p['mean'] + p['sem'],
                                    color=colors[regime], alpha=.18, lw=0)
                elif style == 'individual_reps':
                    raw = active[active.freq.eq(freq) & active.representative_regime.eq(regime)]
                    jitter = ((raw.rep_index.to_numpy() * .61803398875) % 1 - .5) * .006
                    ax.scatter(raw.cv + (k-4)*.002 + jitter, raw[column], s=3,
                               color=colors[regime], alpha=.10, linewidths=0, rasterized=True)
                ax.plot(p.cv, center, marker='o', ms=3, lw=1.7, ls=ls,
                        color=colors[regime], alpha=.96, zorder=3)
            ax.axhline(0, color='black', lw=.7, alpha=.55)
            ax.set_ylim(-1.12 if style == 'mean_sem' else -1.05, 1.12 if style == 'mean_sem' else 1.05)
            ax.set_yticks([-1, -.5, 0, .5, 1])
            cv_ticks(ax); ax.grid(False); ax.grid(axis='y', alpha=.12)
            if row == 0:
                ax.set_title(f'{freq} Hz', fontsize=12)
            if col == 0:
                ax.set_ylabel(row_label)
            if row == 3:
                ax.set_xlabel('Coefficient of Variation (CV)')
    fig.subplots_adjust(left=.06, right=.985, top=.97, bottom=.16, wspace=.04, hspace=.14)
    for row, letter in enumerate('abcd'):
        pos = axes[row, 0].get_position()
        fig.text(pos.x0-.045, pos.y1, letter, va='top', weight='bold', fontsize=13)
    fig.legend(handles=[Line2D([], [], color='black', ls=ls, lw=2, label=text)
                        for ls, text in [(':', 'Spikes (1 ms)'), ('--', 'Bursts (10 ms)')]],
               loc='lower center', bbox_to_anchor=(.5, .066), ncol=2, frameon=False,
               handlelength=2.5, columnspacing=1.2)
    # Crisp paired color swatches, no translucent halo in the legend.
    yr, yt, yl = .060, .041, .0505
    fig.text(.17, yr, 'Rate', ha='right', va='center', fontsize=10, weight='bold')
    fig.text(.17, yt, 'Timing', ha='right', va='center', fontsize=10, weight='bold')
    for x, regime in zip(np.linspace(.21, .86, len(REGIMES)), REGIMES):
        for y, palette in [(yr, RATE), (yt, TIMING)]:
            fig.add_artist(Line2D([x-.029, x-.009], [y, y], transform=fig.transFigure,
                                  color=palette[regime], lw=2.4, solid_capstyle='butt'))
        suffix = 'max' if regime == 'I_all' else regime[2:]
        fig.text(x, yl, rf'$\hat{{E}}_{{\mathrm{{{suffix}}}}}$', va='center', fontsize=12)
    stem = stem or ('suppfig7_population_evidence_curves' if style == 'mean' else f'S7_{style}')
    v2.table(summary, stem + '_summary')
    return v2.figure(fig, stem)
