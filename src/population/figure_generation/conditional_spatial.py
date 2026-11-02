"""Promoted S5: conditional mean count colors and recruitment-probability opacity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from population.manuscript_v2 import COLORS, REGIMES, label
from population.figure_generation.spatial_exploration import digest, grid_panel
from population.figure_generation.spatial_prototype import (
    FREQUENCIES, count_equivalent_regimes, draw_curve_panel, summarize_trials,
)

MAP_CVS = [0., .5, 1.]
DISPLAY_FREQUENCIES = [20, 50, 100, 200]
MODES = [('true', 'True-CV'), ('cv0', 'CV0-based')]
STEM = 'S5_conditional_mean_recruitment_opacity'
MANUSCRIPT_STEM = 'suppfig5_population_recruitment'
AXON_KEYS = ['mode', 'freq', 'cv', 'target_axon_id', 'target_cutline_id', 'target_diameter_um']


def conditional_statistics(data, n_reps=100):
    """Separate positive-count mean from recruitment fraction; never condition the denominator."""
    if data.duplicated(AXON_KEYS + ['rep_index']).any():
        raise ValueError('Duplicate per-repetition axon responses')
    if not data.n_spikes.ge(0).all() or not np.isfinite(data.n_spikes).all():
        raise ValueError('Invalid spike counts')
    if not data.rep_index.isin(range(n_reps)).all():
        raise ValueError('Unexpected repetition index')
    result = data.groupby(AXON_KEYS, as_index=False).agg(
        n_repetitions=('rep_index', 'nunique'),
        n_active_repetitions=('n_spikes', lambda x: int(x.gt(0).sum())),
        sum_spikes=('n_spikes', 'sum'), n_input_pulses=('n_input_pulses', 'first'))
    if not result.n_repetitions.eq(n_reps).all():
        raise ValueError('Missing repetitions; opacity requires the full denominator')
    if not data.groupby(AXON_KEYS).n_input_pulses.nunique().eq(1).all():
        raise ValueError('Input pulse counts differ within a condition')
    result['recruitment_probability'] = result.n_active_repetitions / result.n_repetitions
    result['conditional_mean_spikes'] = result.sum_spikes / result.n_active_repetitions.replace(0, np.nan)
    result['unconditional_mean_spikes'] = result.sum_spikes / result.n_repetitions
    return result


def response_paths(root):
    return {
        'true': root / 'data/detection/population/population_responses.csv',
        'cv0': root / 'data/detection/population/population_cv0_based_responses.csv',
    }


def extract_statistics(root, output, trials):
    """Read all 100 reps at Imax, audit against independent trial totals, cache only aggregates."""
    paths = response_paths(root)
    index_path = root / 'data/detection/population/index_regime_amplitudes.csv'
    refresh = root / 'results/population/detection/imax_population_refresh'
    if json.loads((refresh / 'merge_manifest.json').read_text())['index_sha256'] != digest(index_path):
        raise ValueError('True-CV responses do not match current index amplitudes')
    fingerprints = dict(version=1, index_sha256=digest(index_path), cvs=MAP_CVS,
                        frequencies=FREQUENCIES, n_repetitions=100, regime='I_all',
                        sources={m: dict(path=p.relative_to(root).as_posix(), size=p.stat().st_size,
                                         mtime_ns=p.stat().st_mtime_ns)
                                 for m, p in paths.items()})
    cache = output / '_data/conditional_axon_statistics.csv'
    manifest = output / '_data/response_cache_manifest.json'
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text()) == fingerprints:
        print('Using validated all-repetition aggregate cache.', flush=True)
        return pd.read_csv(cache, dtype={'mode': str})
    pieces = []
    cols = AXON_KEYS[1:] + ['rep_index', 'thresh_type', 'amp_uA', 'n_spikes', 'n_input_pulses']
    for mode, path in paths.items():
        print(f'Scanning all repetitions: {mode} ({path.stat().st_size / 1e9:.2f} GB)', flush=True)
        for i, chunk in enumerate(pd.read_csv(path, usecols=cols, chunksize=500000)):
            selected = chunk[chunk.freq.isin(FREQUENCIES) & chunk.cv.isin(MAP_CVS) & chunk.thresh_type.eq('I_all')]
            if not selected.empty:
                pieces.append(selected.assign(mode=mode))
            if i % 10 == 0:
                print(f'  scanned {(i+1)*500000:,} rows', flush=True)
    data = pd.concat(pieces, ignore_index=True)
    if len(data) != 2 * 5 * 3 * 100 * 420:
        raise ValueError('Incomplete all-repetition spatial subset')
    trial_keys = ['mode', 'freq', 'cv', 'rep_index', 'thresh_type']
    if data.duplicated(trial_keys + ['target_axon_id']).any() or not data.groupby(trial_keys).size().eq(420).all():
        raise ValueError('Invalid per-trial axon coverage')
    basis = pd.read_csv(root / 'data/space-time-V/population/axon_basis_manifest.csv')
    identities = data[['target_axon_id', 'target_cutline_id', 'target_diameter_um']].drop_duplicates()
    if set(identities.target_axon_id) != set(basis.axon_id):
        raise ValueError('Response axon identities differ from the 420-axon basis')
    paired = identities.merge(basis[['axon_id', 'cutline_id', 'diameter_um']],
                              left_on='target_axon_id', right_on='axon_id', validate='one_to_one')
    if not paired.target_cutline_id.eq(paired.cutline_id).all() or not np.allclose(paired.target_diameter_um, paired.diameter_um):
        raise ValueError('Incorrect spatial locations or diameters')
    index = pd.read_csv(index_path)
    lookup = index[index.thresh_type.eq('I_all')].set_index(['freq', 'cv', 'rep_index']).amp_uA
    for (mode, freq, cv, rep, _), group in data.groupby(trial_keys):
        expected = lookup.loc[(freq, cv if mode == 'true' else 0., rep)]
        if not np.allclose(group.amp_uA, expected, atol=.01, rtol=0):
            raise ValueError('Stale Imax amplitude in conditional spatial data')
    observed = data.groupby(trial_keys, as_index=False).agg(
        population_spikes=('n_spikes', 'sum'), active_axon_count=('n_spikes', lambda x: x.gt(0).sum()))
    audit = observed.merge(trials, on=trial_keys, suffixes=('_raw', '_summary'), validate='one_to_one')
    if len(audit) != 3000:
        raise ValueError('Incomplete independent trial-summary audit')
    for metric in ['population_spikes', 'active_axon_count']:
        if not np.allclose(audit[metric + '_raw'], audit[metric + '_summary']):
            raise ValueError(f'Trial summary mismatch: {metric}')
    result = conditional_statistics(data)
    audit.to_csv(output / '_data/trial_validation.csv', index=False)
    result.to_csv(cache, index=False)
    manifest.write_text(json.dumps(fingerprints, indent=2), encoding='utf-8')
    print(f'Validated {len(data):,} responses from 3,000 trials.', flush=True)
    return result


def color_statistics(stats):
    """Color the conditional mean using a panel-level mean index Imax reference."""
    output = []
    for (mode, freq, cv), group in stats.groupby(['mode', 'freq', 'cv']):
        reference_cv = cv if mode == 'true' else 0.
        index = stats[stats['mode'].eq(mode) & stats.freq.eq(freq) & stats.cv.eq(reference_cv) &
                      stats.target_axon_id.eq('S01_140um_ph1')]
        if len(index) != 1 or not index.n_active_repetitions.eq(100).all():
            raise ValueError('Expected reliably active index axon for Imax reference')
        reference = index.unconditional_mean_spikes.iloc[0]
        classes = count_equivalent_regimes(group.conditional_mean_spikes.fillna(0).to_numpy(),
                                           group.n_input_pulses.iloc[0], reference)
        classes = classes.drop(columns='n_spikes')
        output.append(pd.concat([group.reset_index(drop=True), classes], axis=1).assign(reference_cv=reference_cv))
    return pd.concat(output, ignore_index=True)


def conditional_faces(frame):
    """Per-face alpha only: outline alpha is never changed."""
    return {r.target_axon_id: (1., 1., 1., 0.) if r.n_active_repetitions == 0 else
            tuple(COLORS[r.equivalent_regime][:3]) + (float(r.recruitment_probability),)
            for r in frame.itertuples()}


@plt.rc_context(plt.rcParamsDefault)
def draw_figure(stats, summaries):
    """Return the accepted layout, independent of notebook-wide style settings."""
    fig = plt.figure(figsize=(20, 14))
    # Dedicated spacer between a and b; reserve bottom margin for the shared legend.
    outer = fig.add_gridspec(4, len(DISPLAY_FREQUENCIES), left=.065, right=.855, bottom=.10, top=.95,
                            height_ratios=[3.2, .10, 1, 1], hspace=.09, wspace=.09)
    map_axes = {}
    for col, freq in enumerate(DISPLAY_FREQUENCIES):
        rows = outer[0, col].subgridspec(3, 1, hspace=.10)
        for cv_row, cv in enumerate(MAP_CVS):
            pair = rows[cv_row].subgridspec(2, 1, hspace=.18)
            for j, (mode, mode_label) in enumerate(MODES):
                p = stats[stats.freq.eq(freq) & stats.cv.eq(cv) & stats['mode'].eq(mode)]
                if len(p) != 420:
                    raise ValueError('Incomplete conditional map')
                ax = fig.add_subplot(pair[j])
                grid_panel(ax, p, conditional_faces(p), mode_label, dense=True,
                           left_phase='ph1', radius=.48, depth_spacing=1.10, tight_bounds=True)
                box = ax.get_position(original=True)
                ax.set_position([box.x0 - .004, box.y0, box.width + .004, box.height])
                ax.set_title(mode_label, fontsize=10, pad=2)
                show_grid = cv_row == 2 and j == 1
                ax.tick_params(labelbottom=show_grid, labelleft=col == 0, labelsize=7)
                if col == 0:
                    ax.set_yticklabels(['S', 'M', 'D'])
                ax.set_xlabel('Column number' if show_grid else '', fontsize=8)
                map_axes[cv_row, j, col] = ax
        box = outer[0, col].get_position(fig)
        fig.text(box.x0 + box.width / 2, .975, f'{freq} Hz', ha='center', fontsize=17, weight='bold')
        for row, (metric, title) in enumerate([
                ('median_active_axons', 'Recruited axons'), ('median_population_spikes', 'Population spikes')], 2):
            draw_curve_panel(fig, outer[row, col], summaries, freq, metric, title,
                             first_column=col == 0, show_x_labels=row == 3, label_peak=True)
    fig.text(.025, .975, 'a', fontsize=20, weight='bold')
    for row, cv in enumerate(MAP_CVS):
        top, bottom = map_axes[row, 0, 0].get_position(), map_axes[row, 1, 0].get_position()
        fig.text(.040, (top.y1 + bottom.y0) / 2, f'CV = {cv:g}', rotation=90,
                 ha='center', va='center', fontsize=14, weight='bold')
    for row, letter in [(2, 'b'), (3, 'c')]:
        box = outer[row, 0].get_position(fig)
        fig.text(.025, box.y1 + .008, letter, fontsize=20, weight='bold')
    handles = [Patch(facecolor=COLORS[r], edgecolor='none', label=label(r)) for r in REGIMES]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.46, .012), frameon=False,
               ncol=len(REGIMES), title='Operating regime', fontsize=11, title_fontsize=10)
    # No opacity key, suptitle, or on-figure caption text.
    return fig


def prepare_data(root, output):
    """Validate current sources and export plot data and interpretation notes."""
    (output / '_data').mkdir(parents=True, exist_ok=True)
    true_path = root / 'results/population/detection/imax_population_refresh/true_population_trial_summary.csv'
    cv0_path = root / 'data/features/population/population_cv0_based_features.csv'
    true = pd.read_csv(true_path).rename(columns={'total_population_spikes': 'population_spikes'})
    cv0 = pd.read_csv(cv0_path)
    cv0['population_spikes'] = cv0.source_total_population_spikes
    cols = ['freq', 'cv', 'rep_index', 'thresh_type', 'active_axon_count', 'population_spikes']
    trials = pd.concat([true[cols].assign(mode='true'), cv0[cols].assign(mode='cv0')], ignore_index=True)
    summaries = summarize_trials(trials)
    stats = extract_statistics(root, output, trials)
    if len(stats) != 12600 or not stats.n_repetitions.eq(100).all():
        raise ValueError('Incomplete conditional summary')
    # Retain the validated five-frequency source cache; export only displayed data.
    stats = stats[stats.freq.isin(DISPLAY_FREQUENCIES)].copy()
    summaries = summaries[summaries.freq.isin(DISPLAY_FREQUENCIES)].copy()
    stats = color_statistics(stats)
    stats.to_csv(output / '_data/conditional_axon_colors.csv', index=False)
    summaries.to_csv(output / '_data/all_repetition_medians.csv', index=False)
    notes = (
        'Panel a: responses at each repetition\'s index-defined Imax amplitude, pooled over all 100 repetitions; '
        'CV 0/0.5/1, observer frequencies 20/50/100/200 Hz, true-CV and CV0-anchored amplitudes. No exemplar selection. '
        'Color represents the arithmetic mean spike count CONDITIONAL ON at least one spike. '
        'Fill opacity is exactly the number of active repetitions divided by 100; no opacity floor or nonlinear scaling. '
        'Never-active axons have undefined conditional means (NaN in data) and unfilled halves. All outlines remain opaque. '
        'Unconditional mean = recruitment probability multiplied by conditional mean.\n\n'
        'Colors use nearest spike-count-equivalent regime anchors, not independently calibrated axonal regimes. '
        'Fixed anchors are 1/5/10 spikes and ceil(5/10/15/25/50% of nominal pulse count). '
        'The Imax anchor is the mean observed INDEX-axon Imax count across all 100 repetitions at the defining CV '
        '(displayed CV for true-CV; CV0 for anchored). Equal-count aliases retain the first canonical label; '
        'midpoint ties choose the lower count. Values above the Imax reference use the top category and are flagged. '
        'No individual axon plateau or timing pattern is inferred.\n\n'
        'Rings inner to outer: 5.7, 8.7, 10, 11.5, 12.8, 14 micrometers. Left/right: phases 1/0. '
        'Black outline: index axon. Schematic superficial/middle/deep sampling grid, not anatomical coordinates. '
        'Opacity, never-recruited, and amplitude-basis keys omitted by request; these definitions belong in the caption.\n\n'
        'Panels b/c unchanged: medians of recruited-axon counts and total population spikes over all 100 repetitions, '
        'all CVs and regimes. Frequency-specific logarithmic y ranges, separate zero strips, no pseudocounts. '
        'Panel b upper labeled ticks mark each frequency\'s largest plotted median recruitment, not single-repetition maxima. '
        'Solid/filled circles: true-CV; dashed/open squares: CV0-based.'
    )
    (output / '_data/caption_notes.txt').write_text(notes, encoding='utf-8')
    (output / '_data/provenance.json').write_text(json.dumps(dict(
        all_repetitions=list(range(100)), map_cvs=MAP_CVS, displayed_frequencies=DISPLAY_FREQUENCIES, regime='I_all',
        conditional_mean='sum of spikes / count of repetitions with spikes > 0; NaN if none',
        opacity='active repetitions / 100', imax_color_anchor='mean index Imax spikes at defining CV',
                        summary_source_hashes={p.relative_to(root).as_posix(): digest(p)
                                               for p in [true_path, cv0_path]}), indent=2), encoding='utf-8')
    return stats, summaries


def manuscript_figure(v2):
    """Notebook S5 entry point; one figure plus its own audited supporting data."""
    stats, summaries = prepare_data(v2.root, v2.output / f'{MANUSCRIPT_STEM}_data')
    fig = draw_figure(stats, summaries)
    return v2.figure(fig, MANUSCRIPT_STEM)


def main():
    plt.switch_backend('Agg')
    root = Path(__file__).resolve().parents[3]
    output = root / 'results/population/figures_v3/exploratory_s5/consolidated_prototype/conditional_mean_opacity'
    stats, summaries = prepare_data(root, output)
    fig = draw_figure(stats, summaries)
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'{STEM}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
        print(f'Saved {STEM}.{ext}', flush=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
