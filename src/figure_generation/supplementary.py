"""Supplementary manuscript plots; no fitting or simulation is performed."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_generation.figure_model import REGIMES, FREQS, label, cv_ticks, select, choice, integrated_share

FREQUENCIES = [10, 20, 50, 100, 200]
FREQUENCY_COLORS = dict(zip(FREQUENCIES, plt.cm.viridis(np.linspace(.02, .98, 5))))
FREQUENCY_MARKERS = dict(zip(FREQUENCIES, ['o', 's', 'D', '^', 'v']))


def count_source(v2, mode):
    if mode == 'true':
        frame = v2.index.copy()
    elif mode == 'cv0':
        frame = pd.read_csv(v2.root / 'data/features/population_cv0_based_features.csv')
    else:
        raise ValueError(mode)
    frame['display_count'] = frame.n_index_spikes.astype(float)
    mask = frame.thresh_type.isin(REGIMES[3:])
    frame.loc[mask, 'display_count'] *= 100 / frame.loc[mask, 'n_input_pulses']
    return frame


def central_point_mask(frame, value='display_count', interval=None):
    """Pointwise display subset, never a filtered source for computing medians."""
    if interval is None:
        return pd.Series(True, index=frame.index)
    lo, hi = interval
    if not 0 <= lo < hi <= 1:
        raise ValueError('Invalid display quantiles')
    grouped = frame.groupby(['thresh_type', 'freq', 'cv'])[value]
    return frame[value].between(grouped.transform(lambda x: x.quantile(lo)),
                                grouped.transform(lambda x: x.quantile(hi)))


def index_counts(v2, mode='true', *, amplitudes=False, point_interval=None):
    """Median plus reps; optional central pointwise interval is an explicit preview."""
    frame = v2.index.copy() if amplitudes else count_source(v2, mode)
    if amplitudes:
        frame['display_count'] = frame.amp_uA
    frame['displayed'] = central_point_mask(frame, interval=point_interval)
    summary = frame.groupby(['thresh_type', 'freq', 'cv']).display_count.agg(
        median='median', minimum='min', maximum='max', q05=lambda x: x.quantile(.05),
        q10=lambda x: x.quantile(.10), q90=lambda x: x.quantile(.90), q95=lambda x: x.quantile(.95),
        n='count').reset_index()
    shown = frame.groupby(['thresh_type', 'freq', 'cv']).displayed.sum().rename('n_displayed').reset_index()
    summary = summary.merge(shown, on=['thresh_type', 'freq', 'cv'], validate='one_to_one')
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10), sharex=True)
    for ax, regime in zip(axes.flat, REGIMES):
        for k, freq in enumerate(FREQUENCIES):
            raw = frame[(frame.thresh_type == regime) & (frame.freq == freq) & frame.displayed].sort_values(['cv', 'rep_index'])
            p = summary[(summary.thresh_type == regime) & (summary.freq == freq)].sort_values('cv')
            # Deterministic jitter changes x only, never spike counts. Independent
            # offsets reveal frequency-specific distributions at the same CV.
            jitter = ((raw.rep_index.to_numpy() * .61803398875) % 1 - .5) * .006
            ax.scatter(raw.cv.to_numpy() + (k - 2) * .012 + jitter, raw.display_count,
                       s=6, alpha=.24, color=FREQUENCY_COLORS[freq], linewidths=0,
                       rasterized=True, zorder=1)
            ax.plot(p.cv, p['median'], color=FREQUENCY_COLORS[freq],
                    lw=1.4, label=f'{freq} Hz', zorder=3)
        ax.set_title(label(regime))
        cv_ticks(ax)
        ax.set_xlim(-.045, 1.045)
        ax.grid(False)
        ax.grid(axis='y', alpha=.15)
        ax.margins(y=.06)
        if ax in axes[-1]:
            ax.set_xlabel('Coefficient of Variation (CV)', fontsize=9)
        if ax in axes[:, 0]:
            ax.set_ylabel('Amplitude (µA)' if amplitudes else
                          ('Spikes (#)' if regime in REGIMES[:3] else 'Spikes (% of input pulses)'))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, ncol=5,
               loc='lower center', bbox_to_anchor=(.5, .005), title='Frequency')
    fig.tight_layout(rect=(0, .065, 1, 1), h_pad=1.1, w_pad=1.0)
    stem = ('suppfig2_index_regime_amplitudes' if amplitudes else
            ('suppfig3_true_index_spike_counts' if mode == 'true' else 'suppfig4_CV0_index_spike_counts'))
    if point_interval is not None:
        stem += '_central80_preview'
    v2.table(summary, stem + '_summary')
    return v2.figure(fig, stem)


def feature_example(v2, freq=200, regime='I_50', rep=0, stem='suppfig6_population_feature_engineering'):
    """Population example with the single-axon figure's axes/line conventions."""
    folder = v2.root / 'data/features'
    with np.load(folder / 'population_cv0_time_representations_v5.npz', allow_pickle=True) as z:
        indices = np.flatnonzero((z['freq'] == freq) & (z['thresh_type'].astype(str) == regime) & (z['rep_index'] == rep))
        indices = indices[np.argsort(z['cv'][indices])]
        cvs = z['cv'][indices].astype(float)
        spikes = z['population_spike_counts_1ms'][indices]
        bursts = z['population_burst_occupancy_counts_10ms'][indices]
    features = pd.read_csv(folder / 'population_cv0_based_features.csv')
    # Keep source selection explicit; evidence uses the same repetition/regime.
    evidence = pd.read_csv(folder / 'population_cv0_template_evidence.csv')
    def selected(frame):
        return frame[(frame.freq == freq) & (frame.thresh_type == regime) & (frame.rep_index == rep)].sort_values('cv')
    rows, ev = selected(features), selected(evidence)
    if len(cvs) != 11 or not np.allclose(cvs, rows.cv) or not np.allclose(cvs, ev.cv):
        raise ValueError('Incomplete or mismatched feature-example CV grid')
    red, blue = {'I_10': ('#E30613', '#0575AB'), 'I_25': ('#9F040E', '#074680'),
                 'I_50': ('#77030B', '#082E6B'), 'I_all': ('#500207', '#091656')}[regime]
    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(7, 2, height_ratios=[1.15, .20, 1.15, .48, 1, .24, 1],
                            hspace=0, wspace=.26)
    a = fig.add_subplot(grid[0, :]); b = fig.add_subplot(grid[2, :], sharex=a)
    c = fig.add_subplot(grid[4, 0]); d = fig.add_subplot(grid[4, 1])
    e = fig.add_subplot(grid[6, 0], sharex=c); f = fig.add_subplot(grid[6, 1], sharex=d)
    colors = plt.cm.viridis(np.linspace(.02, .98, 11))
    for y, color in enumerate(colors):
        occupied = np.flatnonzero(spikes[y])
        width = .35 + 1.8 * np.sqrt(spikes[y, occupied] / max(1, spikes.max()))
        a.vlines(occupied + .5, y - .32, y + .32, color=color, linewidth=width, alpha=.9)
        support = np.r_[0., (bursts[y] > 0).astype(float), 0.]
        time = np.arange(-1, bursts.shape[1] + 1) * 10.
        envelope = y - .72 * support
        b.fill_between(time, y, envelope, step='post', color=color, alpha=.18, linewidth=0)
        b.step(time, envelope, where='post', color=color, lw=1.35)
    for ax in (a, b):
        ax.set_yticks(range(11), [f'{cv:.1f}' for cv in cvs])
        ax.set_ylim(10.5, -1)
        ax.set_ylabel('Coefficient of Variation (CV)')
        # Keep symmetric nominal-window padding, extending both sides only if
        # necessary to retain late spikes for a different requested example.
        occupied_bins = np.nonzero(spikes)[1]
        padding = max(12., float(occupied_bins.max() + 1 - 2000) if occupied_bins.size else 12.)
        ax.set_xlim(-padding, 2000 + padding)
        ax.set_xticks(np.arange(0, 2001, 250))
        ax.grid(False)
    a.set_title('Spikes (1 ms bins)'); a.tick_params(labelbottom=False)
    b.set_title('Bursts (10 ms bins)'); b.set_xlabel('Time (ms)')
    cr = c.twinx(); dr = d.twinx()
    c.plot(cvs, rows.n_spikes_1ms, ':o', color=red, label='n_spikes', ms=4)
    cr.plot(cvs, rows.n_events_10ms, '--o', color=red, label='n_bursts', ms=4)
    cr.plot(cvs, rows.spikes_per_event_10ms, marker='o', ls=(0, (3, 1, 1, 1)),
            color=red, label='spikes_per_burst', ms=4)
    d.plot(cvs, rows.silence_lv_1ms, ':o', color=blue, label='LV_ISI', ms=4)
    dr.plot(cvs, rows.silence_lv_10ms, '--o', color=blue, label='LV_IBI', ms=4)
    # Same independent-axis headroom strategy as the old single-axon Fig. 7.
    for ax, values, low, high in [
            (c, rows.n_spikes_1ms, .06, .40),
            (cr, np.r_[rows.n_events_10ms, rows.spikes_per_event_10ms], .15, .75),
            (d, rows.silence_lv_1ms, .06, .35),
            (dr, rows.silence_lv_10ms, .15, .75)]:
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        span = max(float(np.ptp(values)), abs(float(values.max())) * .05, .01)
        ax.set_ylim(max(0., values.min() - low * span), values.max() + high * span)
    for ax, right in [(c, cr), (d, dr)]:
        ax.set_ylabel('Spike Metric'); right.set_ylabel('Burst Metric')
        for target, title, loc in [(ax, 'Primary Axis', 'upper left'), (right, 'Secondary Axis', 'upper right')]:
            handles = [Line2D([], [], color=line.get_color(),
                              ls=(0, (3, 1, 1, 1)) if line.get_label() == 'spikes_per_burst' else line.get_linestyle(), lw=2.4,
                              label=line.get_label()) for line in target.lines]
            target.legend(handles=handles, title=title, loc=loc, frameon=False,
                          fontsize=8, title_fontsize=8, handlelength=4.6, handletextpad=.6,
                          bbox_to_anchor=(0 if loc == 'upper left' else 1, 1.04), borderaxespad=.35)
        right.spines['right'].set_visible(True)
        right.spines['right'].set_linewidth(1.)
        right.spines['right'].set_color('.35')
        right.tick_params(axis='y', right=True, labelright=True, direction='out', length=4, pad=4)
        right.grid(False); ax.grid(False); ax.tick_params(labelbottom=False)
    c.set_title(r'Rate Features ($F_{rate}$)'); d.set_title(r'Timing Features ($F_{timing}$)')
    for ax, kind, color in [(e, 'rate', red), (f, 'timing', blue)]:
        for window, ls, text in [(1, ':', 'Spikes (1 ms)'), (10, '--', 'Bursts (10 ms)')]:
            ax.plot(cvs, ev[f'{kind}_evidence_{window}ms'], ls=ls, marker='o', ms=4, color=color, label=text)
        handles = [Line2D([], [], color=color, ls=ls, lw=2.4, label=text)
                   for ls, text in [(':', 'Spikes (1 ms)'), ('--', 'Bursts (10 ms)')]]
        ax.axhline(0, color='.65', lw=.7, zorder=0)
        ax.set_ylabel('Normalized Evidence')
        ax.set_xlabel('Coefficient of Variation (CV)')
        ax.set_ylim(-1.08, 1.08)
        ax.legend(handles=handles, frameon=False, loc='upper left' if kind == 'rate' else 'lower right',
                  fontsize=8, handlelength=4.6, handletextpad=.6)
        ax.grid(alpha=.16)
    e.set_title(r'Rate Evidence ($E_{rate}$)'); f.set_title(r'Timing Evidence ($E_{timing}$)')
    for ax in [c, d, e, f]:
        cv_ticks(ax)
    fig.subplots_adjust(left=.07, right=.945, bottom=.075, top=.96)
    # Figure coordinates align labels regardless of full-width/half-width axes.
    for ax, letter in zip([a, b, c, d, e, f], 'abcdef'):
        x = .018 if letter in 'abce' else d.get_position().x0 - .045
        fig.text(x, ax.get_position().y1 + .01, letter, weight='bold', fontsize=13)
    return v2.figure(fig, stem)


def model_comparison(v2):
    """Constrained sharing hypotheses and the less-constrained hybrid benchmark."""
    models = [('primary', 'hybrid', 'Per-frequency hybrid', '#6a3d9a', '-'),
              ('crossed', 'hybrid', 'Per-block hybrid', '#c88715', '--'),
              ('per_panel', 'hybrid', 'Per-panel hybrid', '#27996d', '-.')]
    from figure_generation.layout import observer_grid, panel_label, observer_headers, reliance_background
    fig, axes = observer_grid()
    curve_rows, reliance_rows, performance = [], [], []
    for scope, family, name, color, ls in models:
        source = v2.focus if family == 'hybrid' else v2.control
        curves = select(source['curves'], scope, family, 'test')
        scores = select(source['rmse'], scope, family, 'test')
        overall = scores[scores.freq.eq('ALL') & scores.block.eq('ALL')]
        if len(overall) != 1:
            raise ValueError('Expected one pooled test RMSE per model')
        performance.append(dict(display_model=name, model=f'{scope}_{family}',
                                simulation_test_RMSE=float(overall.rmse_mean.iloc[0])))
        curve_rows.append(curves.assign(display_model=name))
        for row, block in enumerate(['std0', 'std1']):
            for col, freq in enumerate(FREQS):
                p = curves[(curves.freq == freq) & (curves.block == block)].sort_values('cv')
                axes[row, col].plot(p.cv, choice(p.cv, p.p_predicted_mean, block), color=color,
                                   ls=ls, lw=2.3 if scope == 'primary' else 1.6, alpha=.95)
        if family == 'hybrid':
            values = integrated_share(select(v2.mc, scope, split='test'), ['mc', 'freq', 'block'], 'effective_rate_share')
            summary = values.groupby(['freq', 'block']).rate_share_auc.agg(
                mean='mean', low=lambda x: x.quantile(.025), high=lambda x: x.quantile(.975)).reset_index()
            reliance_rows.append(summary.assign(display_model=name))
            offset = {'primary': -.12, 'crossed': 0, 'per_panel': .12}[scope]
            for row, block in enumerate(['std0', 'std1']):
                p = summary[summary.block == block].set_index('freq').loc[FREQS]
                # Quantile intervals need not contain the mean: draw endpoints directly.
                x = np.arange(4) + offset
                axes[row, 4].vlines(x, p.low, p.high, color=color, lw=1.2)
                axes[row, 4].plot(x, p['mean'], 'o', color=color, ms=4)
    for row, block in enumerate(['std0', 'std1']):
        for col, freq in enumerate(FREQS):
            ax = axes[row, col]
            v2.animal_panel(ax, freq, block)
            cv_ticks(ax)
            ax.tick_params(labelbottom=row == 1, labelsize=9)
            if col:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.set_xlabel('Coefficient of Variation (CV)', fontsize=8)
            if col == 0:
                ax.set_ylabel('P(choose CV1)' if row == 0 else 'P(choose CV0)')
        ax = axes[row, 4]
        ax.set(title=f'Rate reliance | {block}', xticks=range(4), xticklabels=FREQS,
               ylim=(-.03, 1.03), ylabel='CV-averaged rate reliance')
        ax.axhline(.5, color='.65', ls=':', lw=1)
        ax.set_xlabel('Frequency (Hz)' if row == 1 else '')
        reliance_background(ax)
    observer_headers(fig, axes, FREQS)
    fig.legend(handles=[Line2D([], [], color=c, ls=ls, lw=2,
                               label=f"{n}\n(RMSE = {p['simulation_test_RMSE']:.3f})")
                        for (_, _, n, c, ls), p in zip(models, performance)],
               loc='lower center', bbox_to_anchor=(.5, .01), ncol=3, frameon=False,
               fontsize=9, columnspacing=2.5, title='Hybrid Model Structure', title_fontsize=10)
    panel_label(fig, axes[0, 0], 'a')
    panel_label(fig, axes[0, 4], 'b')
    v2.table(pd.concat(curve_rows), 'suppfig9_model_prediction_source')
    v2.table(pd.concat(reliance_rows), 'suppfig9_model_reliance_source')
    v2.table(pd.DataFrame(performance), 'suppfig9_model_performance_source')
    return v2.figure(fig, 'suppfig9_model_comparison')
