"""Exploratory split-ring S5 maps from measured per-axon response counts.

Coordinates are schematic grid indices, not anatomical coordinates. No fitting,
simulation, or interpolation of per-axon operating thresholds is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Wedge, Patch
from matplotlib.collections import PatchCollection
from figure_generation.figure_model import REGIMES, COLORS, label

FREQUENCIES = [20, 50, 100, 200]
DIAMETERS = [5.7, 8.7, 10., 11.5, 12.8, 14.]
MODE_NAMES = {'true': 'True-CV amplitudes', 'cv0': 'CV0-anchored amplitudes'}
DENSE_REGIMES = ['I_10', 'I_50', 'I_all']


def rank_recruitment_repetitions(features, frequencies, cvs):
    """Rank complete reps by mean active axons over the displayed CV/regime cells.

    One rep is chosen per frequency, not independently for each panel. Ties use
    the lowest zero-based rep index. This is an extreme-example selection, not
    an estimate of typical population recruitment.
    """
    selected = features[features.freq.isin(frequencies) & features.cv.isin(cvs) &
                        features.thresh_type.isin(DENSE_REGIMES)].copy()
    keys = ['freq', 'rep_index', 'cv', 'thresh_type']
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate recruitment selection conditions')
    if (set(selected.freq) != set(frequencies) or selected.empty or
            not selected.groupby(['freq', 'rep_index']).size().eq(len(cvs) * len(DENSE_REGIMES)).all()):
        raise ValueError('Incomplete recruitment selection conditions')
    if not selected.active_axon_count.between(0, 420).all():
        raise ValueError('Invalid recruitment counts')
    ranking = selected.groupby(['freq', 'rep_index'], as_index=False).agg(
        mean_active_axons=('active_axon_count', 'mean'),
        peak_active_axons=('active_axon_count', 'max'),
        n_conditions=('active_axon_count', 'size'))
    imax = selected[selected.thresh_type.eq('I_all')].groupby(['freq', 'rep_index']).active_axon_count.mean()
    ranking = ranking.merge(imax.rename('imax_mean_active_axons'), on=['freq', 'rep_index'], validate='one_to_one')
    ranking = ranking.sort_values(['freq', 'mean_active_axons', 'rep_index'], ascending=[True, False, True])
    ranking['rank'] = ranking.groupby('freq').cumcount() + 1
    ranking['selected'] = ranking['rank'].eq(1)
    return ranking


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(root, output, frequencies=None, cvs=None, repetitions=None):
    """Cache the small selected subset, validating amplitudes and aggregate counts."""
    root, output = Path(root), Path(output)
    frequencies = FREQUENCIES if frequencies is None else list(frequencies)
    cvs = [0., 1.] if cvs is None else list(cvs)
    repetitions = {freq: 0 for freq in frequencies} if repetitions is None else repetitions
    if set(repetitions) != set(frequencies):
        raise ValueError('Provide exactly one repetition for each requested frequency')
    output.mkdir(parents=True, exist_ok=True)
    refresh = root / 'data/detection'
    index_path = root / 'data/detection/index_regime_amplitudes.csv'
    manifest = json.loads((refresh / 'true_population_trial_summary_manifest.json').read_text())
    if manifest['index_sha256'] != digest(index_path):
        raise ValueError('True-response refresh does not match the current index amplitudes')
    paths = {'true': root / 'data/detection/population_responses.csv',
             'cv0': root / 'data/detection/population_cv0_based_responses.csv'}
    fingerprint = {'version': 1, 'index_sha256': digest(index_path),
                   'sources': {mode: dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                               for mode, p in paths.items()},
                   'freqs': frequencies, 'cvs': cvs,
                   'repetitions': {str(freq): int(repetitions[freq]) for freq in frequencies}}
    cached = output / 'selected_axon_responses.csv'
    cache_manifest = output / 'selected_axon_responses_manifest.json'
    if cached.exists() and cache_manifest.exists() and json.loads(cache_manifest.read_text()) == fingerprint:
        # Prevent pandas from inferring the literal mode 'true' as boolean True
        # in chunks that happen not to contain any 'cv0' rows.
        data = pd.read_csv(cached, dtype={'mode': str})
        print('Using matching selected-response cache.', flush=True)
    else:
        chunks = []
        columns = ['target_axon_id', 'target_cutline_id', 'target_diameter_um', 'freq', 'cv',
                   'rep_index', 'rep_label', 'thresh_type', 'amp_uA', 'n_spikes', 'n_input_pulses']
        for mode, path in paths.items():
            print(f'Scanning {mode}: {path.name} ({path.stat().st_size / 1e9:.2f} GB)', flush=True)
            for i, frame in enumerate(pd.read_csv(path, usecols=columns, chunksize=400000)):
                selected = frame[frame.freq.isin(frequencies) & frame.cv.isin(cvs) &
                                 frame.rep_index.eq(frame.freq.map(repetitions))]
                if not selected.empty:
                    chunks.append(selected.assign(mode=mode))
                if i % 10 == 0:
                    print(f'  {mode}: scanned {(i + 1) * 400000:,} rows', flush=True)
        data = pd.concat(chunks, ignore_index=True)
    keys = ['mode', 'freq', 'cv', 'rep_index', 'thresh_type']
    if data.duplicated(keys + ['target_axon_id']).any():
        raise ValueError('Duplicate axon responses')
    if not data.rep_index.eq(data.freq.map(repetitions)).all():
        raise ValueError('Cached responses do not match selected repetitions')
    if len(data) != 2 * len(frequencies) * len(cvs) * 9 * 420 or not data.groupby(keys).size().eq(420).all():
        raise ValueError('Incomplete spatial response subset')
    basis = pd.read_csv(root / 'data/space-time-V/axon_basis_manifest.csv')
    if set(data.target_axon_id) != set(basis.axon_id) or len(basis) != 420:
        raise ValueError('Spatial axon identities do not match the 420-axon basis')
    identities = data[['target_axon_id', 'target_cutline_id', 'target_diameter_um']].drop_duplicates().merge(
        basis[['axon_id', 'cutline_id', 'diameter_um']], left_on='target_axon_id', right_on='axon_id', validate='one_to_one')
    if not identities.target_cutline_id.eq(identities.cutline_id).all() or not np.allclose(
            identities.target_diameter_um, identities.diameter_um):
        raise ValueError('Spatial axon diameter/location mismatch')
    index = pd.read_csv(index_path)
    lookup = index.set_index(['freq', 'cv', 'rep_index', 'thresh_type']).amp_uA
    for key, group in data.groupby(keys):
        mode, freq, cv, rep, regime = key
        expected = lookup.loc[(freq, cv if mode == 'true' else 0., rep, regime)]
        if not np.allclose(group.amp_uA, expected, rtol=0, atol=.01):
            raise ValueError(f'Stale regime amplitude: {key}')
    # Compare raw per-axon sums to independently retained trial summaries.
    audits = []
    for mode in paths:
        observed = data[data['mode'] == mode].groupby(['freq', 'cv', 'rep_index', 'thresh_type']).agg(
            total_spikes=('n_spikes', 'sum'), active_axons=('n_spikes', lambda x: (x > 0).sum()))
        if mode == 'true':
            source = pd.read_csv(refresh / 'true_population_trial_summary.csv')
            spike_column = 'total_population_spikes'
        else:
            source = pd.read_csv(root / 'data/features/population_cv0_based_features.csv')
            spike_column = 'source_total_population_spikes'
        paired = observed.reset_index().merge(source, on=['freq', 'cv', 'rep_index', 'thresh_type'], validate='one_to_one')
        if len(paired) != len(frequencies) * len(cvs) * 9 or not np.allclose(paired.total_spikes, paired[spike_column]):
            raise ValueError(f'Population total mismatch for {mode}')
        if not np.allclose(paired.active_axons, paired.active_axon_count):
            raise ValueError(f'Recruited-axon count mismatch for {mode}')
        audits.append(paired.assign(mode=mode)[['mode', 'freq', 'cv', 'rep_index', 'thresh_type', 'total_spikes', 'active_axons']])
    pd.concat(audits).to_csv(output / 'selected_condition_audit.csv', index=False)
    data.to_csv(cached, index=False)
    cache_manifest.write_text(json.dumps(fingerprint, indent=2), encoding='utf-8')
    return data


def recruitment_onset(data):
    """Lowest tested activating amplitude, not first nominal category or permanent recruitment."""
    records = []
    for key, group in data.groupby(['mode', 'freq', 'cv', 'rep_index', 'target_axon_id']):
        group = group.copy()
        group['regime_order'] = group.thresh_type.map({r: i for i, r in enumerate(REGIMES)})
        group = group.sort_values(['amp_uA', 'regime_order'])
        active = group[group.n_spikes > 0]
        record = dict(zip(['mode', 'freq', 'cv', 'rep_index', 'target_axon_id'], key))
        record.update(target_cutline_id=group.iloc[0].target_cutline_id,
                      target_diameter_um=group.iloc[0].target_diameter_um)
        if active.empty:
            record.update(first_regime='none', first_amp_uA=np.nan, tied_regimes='',
                          loses_recruitment_at_higher_tested_amplitude=False)
        else:
            first_amp = active.iloc[0].amp_uA
            tied = active[np.isclose(active.amp_uA, first_amp, rtol=0, atol=1e-8)].sort_values('regime_order')
            record.update(first_regime=tied.iloc[0].thresh_type, first_amp_uA=first_amp,
                          tied_regimes=' / '.join(tied.thresh_type),
                          loses_recruitment_at_higher_tested_amplitude=bool(
                              ((group.amp_uA > first_amp + 1e-8) & (group.n_spikes == 0)).any()))
        records.append(record)
    return pd.DataFrame(records)


def grid_panel(ax, frame, colors, title, dense=False, left_phase='ph0', radius=.43, depth_spacing=1.3,
               tight_bounds=False):
    """Six equal-width concentric annuli, with an explicit display orientation."""
    if left_phase not in ('ph0', 'ph1'):
        raise ValueError('left_phase must be ph0 or ph1')
    if not 0 < radius < .5 or depth_spacing <= 2 * radius:
        raise ValueError('Rings must not overlap adjacent sites')
    positions = {'S': 2 * depth_spacing, 'M': depth_spacing, 'D': 0.}
    patches = []
    for row in frame.itertuples():
        site = row.target_cutline_id
        x, y = int(site[1:]), positions[site[0]]
        j = DIAMETERS.index(float(row.target_diameter_um))
        phase = row.target_axon_id.rsplit('_', 1)[-1]
        angle = (90, 270) if phase == left_phase else (-90, 90)
        patch = Wedge((x, y), radius * (j + 1) / 6, *angle, width=radius / 6,
                      facecolor=colors[row.target_axon_id], edgecolor='.60', linewidth=.18)
        if dense:
            patches.append(patch)
        else:
            ax.add_patch(patch)
        if row.target_axon_id == 'S01_140um_ph1':
            ax.add_patch(Wedge((x, y), radius, *angle, width=radius / 6,
                               fill=False, edgecolor='black', linewidth=1.1))
    if dense:
        ax.add_collection(PatchCollection(patches, match_original=True, rasterized=True, zorder=1))
    margin = radius + .06 if tight_bounds else .65
    ax.set(xlim=(1 - margin, 13 + margin) if tight_bounds else (.3, 13.7),
           ylim=(-margin, 2 * depth_spacing + margin), yticks=[2 * depth_spacing, depth_spacing, 0],
           yticklabels=['Superficial', 'Middle', 'Deep'], xticks=range(1, 14), title=title)
    ax.set_aspect('equal')
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlabel('Grid column (schematic)', fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def legend_key(fig, y=.075):
    fig.text(.5, y, 'Rings, inner to outer: 5.7, 8.7, 10, 11.5, 12.8, 14 µm. Left half: phase 0; right half: phase 1.',
             ha='center', fontsize=9)
    fig.text(.5, y - .022, 'Schematic sampling grid, not anatomical coordinates or axon sizes. Black outline: index axon (S01, 14 µm, phase 1).',
             ha='center', fontsize=9)


def save(fig, output, stem):
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'{stem}.{suffix}', dpi=260, bbox_inches='tight')
    plt.close(fig)


def intensity_maps(data, output):
    subset = data[(data.freq == 100) & (data.cv == 1)]
    fig, axes = plt.subplots(2, 3, figsize=(23, 8))
    norm = Normalize(0, 1, clip=False)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_over('#d62728')
    for i, mode in enumerate(['true', 'cv0']):
        for j, regime in enumerate(['I_10', 'I_50', 'I_all']):
            p = subset[(subset['mode'] == mode) & (subset.thresh_type == regime)]
            colors = {r.target_axon_id: 'white' if r.n_spikes == 0 else cmap(norm(r.n_spikes / r.n_input_pulses))
                      for r in p.itertuples()}
            grid_panel(axes[i, j], p, colors,
                       f'{MODE_NAMES[mode]} | {label(regime)} | {p.amp_uA.iloc[0]:.2f} µA\n'
                       f'{(p.n_spikes > 0).sum()}/420 recruited; {p.n_spikes.sum():,} spikes')
    fig.subplots_adjust(left=.055, right=.94, top=.89, bottom=.23, hspace=.26, wspace=.17)
    color_ax = fig.add_axes([.955, .31, .008, .45])
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, extend='max',
                 label='Axonal spikes / nominal input pulses (red: >1)')
    fig.suptitle('S5 exploration A: firing intensity | 100 Hz, CV1, repetition 0', y=.97)
    legend_key(fig, .105)
    fig.text(.5, .055, 'White: no spikes. Colors encode recorded output, not each axon’s independently calibrated I_max.', ha='center', fontsize=9)
    save(fig, output, 'S5A_spatial_firing_intensity')


def dense_intensity_maps(data, output, frequencies=None, selection_note=None):
    """One 11-CV by 6-condition selection sheet per frequency, same color scale."""
    frequencies = [10, 20, 50, 100, 200] if frequencies is None else frequencies
    cvs = np.round(np.linspace(0, 1, 11), 1)
    columns = [(regime, mode) for regime in DENSE_REGIMES for mode in ['true', 'cv0']]
    norm = Normalize(0, 1, clip=False)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_over('#d62728')
    rows = []
    for freq in frequencies:
        reps = data.loc[data.freq.eq(freq), 'rep_index'].unique()
        if len(reps) != 1:
            raise ValueError(f'Expected one matched repetition at {freq} Hz')
        rep = int(reps[0])
        print(f'Rendering dense intensity sheet: {freq} Hz', flush=True)
        fig, axes = plt.subplots(11, 6, figsize=(36, 25))
        for i, cv in enumerate(cvs):
            for j, (regime, mode) in enumerate(columns):
                p = data[(data.freq == freq) & np.isclose(data.cv, cv) &
                         (data.thresh_type == regime) & (data['mode'] == mode)]
                if len(p) != 420 or p.target_axon_id.nunique() != 420:
                    raise ValueError(f'Incomplete dense panel: {freq}, {cv}, {regime}, {mode}')
                colors = {r.target_axon_id: 'white' if r.n_spikes == 0 else cmap(norm(r.n_spikes / r.n_input_pulses))
                          for r in p.itertuples()}
                active = int((p.n_spikes > 0).sum()); spikes = int(p.n_spikes.sum())
                title = f'{p.amp_uA.iloc[0]:.2f} µA  |  {active}/420 active  |  {spikes:,} spikes'
                ax = axes[i, j]
                grid_panel(ax, p, colors, title, dense=True)
                ax.set_title(title, fontsize=8, pad=2)
                ax.set_xlabel('Grid column' if i == 10 else '', fontsize=8)
                ax.tick_params(labelbottom=i == 10, labelleft=j == 0, labelsize=7)
                rows.append(dict(freq=freq, cv=cv, rep_index=rep, rep_label=p.rep_label.iloc[0], regime=regime, mode=mode,
                                 amplitude_uA=float(p.amp_uA.iloc[0]), active_axons=active, total_spikes=spikes,
                                 over_one=int((p.n_spikes > p.n_input_pulses).sum())))
        fig.subplots_adjust(left=.06, right=.985, bottom=.075, top=.945, hspace=.28, wspace=.10)
        for j, (regime, mode) in enumerate(columns):
            pos = axes[0, j].get_position()
            fig.text(pos.x0 + pos.width / 2, .96, f'{label(regime)} — {MODE_NAMES[mode]}',
                     ha='center', fontsize=13, weight='bold')
        for i, cv in enumerate(cvs):
            pos = axes[i, 0].get_position()
            fig.text(.012, pos.y0 + pos.height / 2, f'CV {cv:.1f}', va='center', fontsize=13, weight='bold')
        fig.suptitle(f'S5A firing-intensity selection sheet | {freq} Hz | CV 0–1 | rep index {rep}', y=.99, fontsize=20)
        if selection_note:
            fig.text(.5, .976, selection_note, ha='center', fontsize=11)
        color_ax = fig.add_axes([.35, .038, .3, .009])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation='horizontal', extend='max')
        cb.set_label('Axonal spikes / nominal input pulses; white = no spikes; red = >1', fontsize=10)
        fig.text(.5, .011, 'Rings inner→outer: 5.7, 8.7, 10, 11.5, 12.8, 14 µm. Left/right = phases 0/1. '
                 'Black outline = index axon. Schematic grid, not anatomical coordinates. Same scale in every panel.',
                 ha='center', fontsize=10)
        stem = f'S5A_dense_{freq:03d}Hz_all_CV'
        # Batched glyph collections are rasterized in PDF; labels remain vector.
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'{stem}.{suffix}', dpi=220, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {stem}.png/.pdf', flush=True)
    return pd.DataFrame(rows)


def categorical_maps(onset, output, cross_frequency=False):
    columns = FREQUENCIES if cross_frequency else [0., 1.]
    fig, axes = plt.subplots(2, len(columns), figsize=(26, 8.3) if cross_frequency else (20, 10.4))
    summary = []
    for i, mode in enumerate(['true', 'cv0']):
        for j, value in enumerate(columns):
            freq, cv = (value, 1.) if cross_frequency else (100, value)
            p = onset[(onset['mode'] == mode) & (onset.freq == freq) & (onset.cv == cv)]
            colors = {r.target_axon_id: 'white' if r.first_regime == 'none' else COLORS[r.first_regime]
                      for r in p.itertuples()}
            recruited = int((p.first_regime != 'none').sum())
            lost = int(p.loses_recruitment_at_higher_tested_amplitude.sum())
            grid_panel(axes[i, j], p, colors, f'{MODE_NAMES[mode]} | {freq} Hz, CV{cv:g}\n'
                       f'{recruited}/420 recruited at any tested regime; {lost} later lose recruitment')
            summary.append(dict(mode=mode, freq=freq, cv=cv, ever_recruited=recruited, later_lose_recruitment=lost))
    handles = [Patch(facecolor=COLORS[r], label=label(r)) for r in REGIMES]
    handles.append(Patch(facecolor='white', edgecolor='.6', label='Not recruited in tested set'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .145), ncol=10, frameon=False,
               title='Index-axon regime at lowest tested amplitude producing ≥1 axonal spike', fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, bottom=.28, top=.88, wspace=.19, hspace=.45)
    fig.suptitle('S5 exploration ' + ('C: recruitment onset across frequencies | CV1' if cross_frequency else
                                     'B: recruitment onset at CV0 versus CV1 | 100 Hz') + ', repetition 0', y=.97)
    legend_key(fig, .105)
    fig.text(.5, .045, 'Tested-amplitude onset, not a continuous activation threshold. Equal-amplitude ties use the first listed regime; all tied labels are exported.',
             ha='center', fontsize=9)
    stem = 'S5C_recruitment_regime_across_frequencies' if cross_frequency else 'S5B_recruitment_regime_CV_comparison'
    pd.DataFrame(summary).to_csv(output / f'{stem}_summary.csv', index=False)
    save(fig, output, stem)


def main():
    plt.switch_backend('Agg')  # File-only rendering; no desktop GUI required.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--max-cv0-recruitment', action='store_true',
                        help='One rep per frequency maximizing mean CV0-based recruitment over displayed CVs/regimes')
    layouts = parser.add_mutually_exclusive_group()
    layouts.add_argument('--dense', action='store_true', help='Default: intensity-only sheets, all CVs and five frequencies')
    layouts.add_argument('--legacy', action='store_true', help='Explicitly regenerate the original A/B/C exploration layouts')
    args = parser.parse_args()
    if args.legacy and args.max_cv0_recruitment:
        parser.error('--max-cv0-recruitment applies to dense sheets, not --legacy')
    root = Path(__file__).resolve().parents[2]
    output = root / 'results/figures/exploratory_s5'
    if not args.legacy:
        frequencies = [10, 20, 50, 100, 200]
        cvs = np.round(np.linspace(0, 1, 11), 1).tolist()
        repetitions, selection_note = None, None
        if args.max_cv0_recruitment:
            output = output / 'max_cv0_recruitment'
            (output / '_data').mkdir(parents=True, exist_ok=True)
            feature_path = root / 'data/features/population_cv0_based_features.csv'
            ranking = rank_recruitment_repetitions(pd.read_csv(feature_path), frequencies, cvs)
            winners = ranking[ranking.selected]
            repetitions = {int(r.freq): int(r.rep_index) for r in winners.itertuples()}
            ranking.to_csv(output / '_data/repetition_ranking.csv', index=False)
            (output / '_data/repetition_selection.json').write_text(json.dumps(dict(
                criterion='Maximum mean active axons across 11 CVs and I_10/I_50/I_all; CV0-based responses only',
                tie_break='Lowest zero-based rep_index', frequencies=frequencies, cvs=cvs,
                regimes=DENSE_REGIMES, repetitions=repetitions,
                feature_source=str(feature_path), feature_sha256=digest(feature_path)), indent=2), encoding='utf-8')
            print(winners.to_string(index=False), flush=True)
            selection_note = 'High-recruitment example: rep selected by mean CV0-based active axons over all 11 CVs and three displayed regimes.'
        data = extract(root, output / '_data', frequencies=frequencies, cvs=cvs, repetitions=repetitions)
        if args.max_cv0_recruitment:
            observed = data[data['mode'].eq('cv0') & data.thresh_type.isin(DENSE_REGIMES)].groupby(
                ['freq', 'cv', 'thresh_type']).n_spikes.apply(lambda x: (x > 0).sum()).groupby('freq').mean()
            expected = winners.set_index('freq').mean_active_axons
            if not np.allclose(observed.sort_index(), expected.sort_index()):
                raise ValueError('Selected raw recruitment differs from ranking')
        if not args.extract_only:
            dense_intensity_maps(data, output, selection_note=selection_note).to_csv(output / '_data/dense_panel_summary.csv', index=False)
        print(f'Validated {len(data):,} responses for all-CV intensity sheets.', flush=True)
        return
    data = extract(root, output)
    onset = recruitment_onset(data)
    onset.to_csv(output / 'first_tested_recruitment_regime.csv', index=False)
    if not args.extract_only:
        intensity_maps(data, output)
        categorical_maps(onset, output)
        categorical_maps(onset, output, cross_frequency=True)
    print(f'Validated {len(data):,} responses; exported to {output}', flush=True)


if __name__ == '__main__':
    main()
