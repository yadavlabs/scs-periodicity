"""Vector-native Fig. 1b population sampling key for manual Adobe assembly."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch

from .spatial_exploration import grid_panel


def sampling_source(v2):
    basis = pd.read_csv(v2.root / 'data/space-time-V/axon_basis_manifest.csv')
    index_ids = v2.index.index_axon_id.dropna().unique()
    if len(index_ids) != 1 or basis.axon_id.duplicated().any():
        raise ValueError('Expected one index axon and unique population identities')
    index_id = index_ids[0]
    index = basis[basis.axon_id.eq(index_id)]
    if len(index) != 1:
        raise ValueError('Index axon missing from the population manifest')
    diameters = sorted(basis.diameter_um.unique())
    if len(diameters) != 6 or set(basis.phase_id) != {'ph0', 'ph1'}:
        raise ValueError('Expected six diameters and two nodal phases')
    if index_id != 'S01_140um_ph1' or index.iloc[0].diameter_um != max(diameters):
        raise ValueError('Index identity changed: review the shared S5 highlight and Fig. 1b annotations')
    expected = {(diameter, phase) for diameter in diameters for phase in ['ph0', 'ph1']}
    for site, group in basis.groupby('cutline_id'):
        if len(group) != 12 or set(zip(group.diameter_um, group.phase_id)) != expected:
            raise ValueError(f'Incomplete diameter/phase coverage at {site}')
    if basis.cutline_id.nunique() != 35 or len(basis) != 420:
        raise ValueError('Unexpected sampling design: review schematic layout')
    frame = basis[['axon_id', 'cutline_id', 'diameter_um', 'phase_id', 'phase_fraction']].copy()
    frame['is_index_axon'] = frame.axon_id.eq(index_id)
    frame['ring_order_inner_to_outer'] = frame.diameter_um.map({d: i+1 for i, d in enumerate(diameters)})
    frame['display_half'] = np.where(frame.phase_id.eq('ph1'), 'left', 'right')
    return frame, index.iloc[0], diameters


@plt.rc_context({**plt.rcParamsDefault, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
def sampling_component(v2):
    frame, index, diameters = sampling_source(v2)
    fig = plt.figure(figsize=(11.5, 3.8))
    grid = fig.add_axes([.085, .27, .49, .49])
    source = frame.rename(columns={'axon_id': 'target_axon_id', 'cutline_id': 'target_cutline_id',
                                   'diameter_um': 'target_diameter_um'})
    colors = {row.axon_id: ('.20' if row.is_index_axon else '.96') for row in frame.itertuples()}
    grid_panel(grid, source, colors, '', dense=False, left_phase='ph1',
               radius=.43, depth_spacing=1.2, tight_bounds=True)
    grid.set_yticklabels(['Superficial', 'Middle', 'Deep'], fontsize=9)
    grid.set_xlabel('Column number (schematic)', fontsize=9, labelpad=7)
    grid.tick_params(axis='x', labelsize=8, pad=4)
    fig.text(.33, .91, '35 sampling locations', ha='center', fontsize=13, weight='bold')
    fig.text(.33, .845, '6 diameters × 2 nodal phases per location', ha='center', fontsize=10)
    fig.text(.33, .12, '35 × 6 × 2 = 420 modeled axons', ha='center', fontsize=11)
    site_x = int(index.cutline_id[1:])
    site_y = {'S': 2.4, 'M': 1.2, 'D': 0}[index.cutline_id[0]]
    grid.annotate(index.cutline_id, xy=(site_x, site_y+.44), xytext=(site_x+.7, site_y+.88),
                  fontsize=8, ha='center', arrowprops=dict(arrowstyle='-', color='.2', lw=.8))
    # An enlarged exemplar at the index location, not a second spatial sample.
    key = fig.add_axes([.65, .22, .32, .64])
    radius = 1.1
    for i, diameter in enumerate(diameters):
        for phase, angles in [('ph1', (90, 270)), ('ph0', (-90, 90))]:
            is_index = diameter == index.diameter_um and phase == index.phase_id
            key.add_patch(Wedge((0, 0), radius*(i+1)/6, *angles, width=radius/6,
                                facecolor='.20' if is_index else '.96',
                                edgecolor='black' if is_index else '.55', linewidth=1.1 if is_index else .7))
        r = radius*(i+.5)/6
        x, y = r*np.cos(np.pi/4), r*np.sin(np.pi/4)
        label_y = -.15 + i*.27
        key.plot([x, 1.26, 1.40], [y, label_y, label_y], color='.45', lw=.65)
        key.text(1.48, label_y, f'{diameter:g} µm', va='center', fontsize=9)
    key.text(1.48, 1.45, 'Diameter', fontsize=9, weight='bold')
    key.text(-.59, -1.34, 'Phase 1', ha='center', fontsize=9)
    key.text(.59, -1.34, 'Phase 0', ha='center', fontsize=9)
    key.set(xlim=(-1.30, 2.45), ylim=(-1.5, 1.65), aspect='equal')
    key.set_axis_off()
    fig.text(.805, .91, f'Index location ({index.cutline_id})', ha='center', fontsize=13, weight='bold')
    fig.text(.805, .12, f'Index axon: {index.diameter_um:g} µm, phase {str(index.phase_id)[-1]}',
             ha='center', fontsize=10, weight='bold')
    fig.add_artist(FancyArrowPatch((.59,.53), (.642,.53), transform=fig.transFigure,
                                  arrowstyle='->', mutation_scale=13, color='.4', lw=1.2))
    stem = 'fig1b_population_sampling_key'
    v2.table(frame, stem + '_source')
    (v2.output / f'{stem}_caption_notes.txt').write_text(
        'Schematic sampling design from axon_basis_manifest.csv. Each concentric glyph represents '
        'six alternative modeled diameters at one sampling location, not nested axons or myelin layers. '
        'Rings inner-to-outer: '+', '.join(f'{d:g}' for d in diameters)+' micrometers. Equal ring widths '
        'are not diameter-scaled. Left/right halves represent nodal phases 1/0, respectively; these '
        'are spatial nodal placements, not stimulus polarity. Dark outer-left half identifies '+str(index.axon_id)+
        '. Neutral fills do not encode recruitment, activity, or current density. S/M/D sampling locations '
        'are schematic, not anatomical coordinates. The enlarged glyph illustrates the index location. '
        'Updated anatomy and any field-map background must be assembled from verified geometry/field exports.',
        encoding='utf-8')
    # Editable text and vector rings; transparent background for Adobe insertion.
    fig.savefig(v2.output / f'{stem}.svg', bbox_inches='tight', transparent=True)
    return v2.figure(fig, stem)
