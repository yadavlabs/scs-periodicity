"""Fig. 2 assembly rasters using the S6 plotting conventions and CV palette."""
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def example_spikes(root, freq=100, regime='I_10', rep=0):
    source = root / 'data/features/population/population_cv0_time_representations_v5.npz'
    with np.load(source, allow_pickle=True) as z:
        indices = np.flatnonzero((z['freq'] == freq)
                                & (z['thresh_type'].astype(str) == regime)
                                & (z['rep_index'] == rep))
        indices = indices[np.argsort(z['cv'][indices])]
        cvs = z['cv'][indices].astype(float)
        spikes = z['population_spike_counts_1ms'][indices].copy()
    if len(cvs) != 11 or not np.allclose(cvs, np.linspace(0, 1, 11)):
        raise ValueError('Expected complete example CV grid')
    return source, cvs, spikes


def raster_components(v2, freq=100, regime='I_10', rep=0):
    """Three transparent, unlabeled strips for manual Fig. 2 assembly.

    Bin centers, viridis colors, alpha, global linewidth normalization and
    symmetric time padding match feature_example panel a. These are population
    responses, not the electrical input pulses or an individual axon's spikes.
    """
    source, cvs, spikes = example_spikes(v2.root, freq, regime, rep)
    colors = plt.cm.viridis(np.linspace(.02, .98, 11))
    occupied_bins = np.nonzero(spikes)[1]
    padding = max(12., float(occupied_bins.max() + 1 - 2000)
                  if occupied_bins.size else 12.)
    figures, records = [], []
    for cv in (0., .9, 1.):
        index = int(np.flatnonzero(np.isclose(cvs, cv))[0])
        occupied = np.flatnonzero(spikes[index])
        widths = .35 + 1.8 * np.sqrt(spikes[index, occupied] / max(1, spikes.max()))
        fig, ax = plt.subplots(figsize=(9, .65))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.vlines(occupied + .5, -.32, .32, color=colors[index], linewidth=widths, alpha=.9)
        ax.set(xlim=(-padding, 2000 + padding), ylim=(-.5, .5))
        ax.set_axis_off()
        stem = f'fig2_population_raster_CV{cv:.1f}'.replace('.', 'p')
        for suffix in ('png', 'pdf', 'svg'):
            with plt.rc_context({'svg.fonttype': 'none'}):
                fig.savefig(v2.output / f'{stem}.{suffix}', dpi=300, transparent=True,
                            bbox_inches='tight', pad_inches=0)
        v2.records.append(dict(kind='figure', stem=stem))
        records.extend(dict(cv=cv, time_ms=float(t + .5), population_spikes=int(spikes[index, t]),
                            linewidth=float(w), red=float(colors[index, 0]),
                            green=float(colors[index, 1]), blue=float(colors[index, 2]))
                       for t, w in zip(occupied, widths))
        figures.append(fig)
    v2.table(pd.DataFrame(records), 'fig2_population_rasters_source')
    with source.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = dict(frequency_hz=freq, regime=regime, repetition=rep, cvs=[0, .9, 1],
                    source=str(source.relative_to(v2.root)),
                    source_sha256=source_hash,
                    time_limits_ms=[-padding, 2000 + padding],
                    linewidth_normalization='Maximum bin count over all 11 CVs of this example, using the S6 panel-a convention',
                    interpretation='CV0-based population spike counts in 1-ms bins; line width encodes count')
    (v2.output / 'fig2_population_rasters_provenance.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    return figures
