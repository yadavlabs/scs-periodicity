import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from population.figure_generation.spatial_exploration import recruitment_onset, grid_panel, rank_recruitment_repetitions


class SpatialExplorationTests(unittest.TestCase):
    def selection_rows(self):
        return pd.DataFrame([dict(freq=f, cv=cv, rep_index=rep, thresh_type=regime,
                                  active_axon_count=(8 if (f == 10 and rep == 2) else 2))
                             for f in [10, 20] for rep in [0, 2] for cv in [0., 1.]
                             for regime in ['I_10', 'I_50', 'I_all']])

    def test_selection_uses_mean_recruitment_and_deterministic_ties(self):
        rows = self.selection_rows()
        # A single higher peak must not beat consistently larger recruitment.
        rows.loc[(rows.freq == 10) & (rows.rep_index == 0) &
                 (rows.cv == 1) & (rows.thresh_type == 'I_all'), 'active_axon_count'] = 20
        ranked = rank_recruitment_repetitions(rows, [10, 20], [0., 1.])
        winners = ranked[ranked.selected].set_index('freq')
        self.assertEqual(winners.loc[10, 'rep_index'], 2)
        self.assertEqual(winners.loc[20, 'rep_index'], 0)
        self.assertEqual(winners.loc[10, 'mean_active_axons'], 8)

    def test_selection_rejects_missing_and_duplicate_conditions(self):
        rows = self.selection_rows()
        for invalid in [rows.iloc[1:], pd.concat([rows, rows.iloc[:1]])]:
            with self.assertRaises(ValueError):
                rank_recruitment_repetitions(invalid, [10, 20], [0., 1.])

    def rows(self, amplitudes, counts, regimes):
        return pd.DataFrame([dict(mode='true', freq=100, cv=1., rep_index=0,
                                  target_axon_id='S01_140um_ph1', target_cutline_id='S01',
                                  target_diameter_um=14., amp_uA=a, n_spikes=n, thresh_type=r)
                             for a, n, r in zip(amplitudes, counts, regimes)])

    def test_onset_uses_actual_amplitude_and_flags_later_loss(self):
        rows = self.rows([300, 250, 400], [2, 1, 0], ['I_volley5', 'I_5', 'I_all'])
        onset = recruitment_onset(rows).iloc[0]
        self.assertEqual(onset.first_regime, 'I_5')
        self.assertEqual(onset.first_amp_uA, 250)
        self.assertTrue(onset.loses_recruitment_at_higher_tested_amplitude)

    def test_equal_amplitude_ties_are_preserved(self):
        rows = self.rows([250, 250, 400], [1, 1, 2], ['I_5', 'I_volley10', 'I_all'])
        onset = recruitment_onset(rows).iloc[0]
        self.assertEqual(onset.first_regime, 'I_volley10')
        self.assertEqual(onset.tied_regimes, 'I_volley10 / I_5')
        self.assertFalse(onset.loses_recruitment_at_higher_tested_amplitude)

    def test_never_recruited_remains_distinct(self):
        onset = recruitment_onset(self.rows([250, 400], [0, 0], ['I_first', 'I_all'])).iloc[0]
        self.assertEqual(onset.first_regime, 'none')
        self.assertTrue(np.isnan(onset.first_amp_uA))

    def test_split_rings_encode_both_phases_and_index_outline(self):
        frame = self.rows([250], [1], ['I_first'])
        other = frame.copy(); other.target_axon_id = 'S01_140um_ph0'
        frame = pd.concat([frame, other])
        fig, ax = plt.subplots()
        grid_panel(ax, frame, {'S01_140um_ph0': 'white', 'S01_140um_ph1': 'yellow'}, 'Test')
        self.assertEqual(len(ax.patches), 3)  # Two phase halves plus index outline.
        self.assertEqual((ax.patches[0].theta1, ax.patches[0].theta2), (-90, 90))
        self.assertEqual((ax.patches[2].theta1, ax.patches[2].theta2), (90, 270))
        plt.close(fig)

    def test_dense_ring_collection_retains_segments_and_index_outline(self):
        frame = self.rows([250], [1], ['I_first'])
        other = frame.copy(); other.target_axon_id = 'S01_140um_ph0'
        fig, ax = plt.subplots()
        grid_panel(ax, pd.concat([frame, other]),
                   {'S01_140um_ph0': 'white', 'S01_140um_ph1': 'yellow'}, 'Dense', dense=True)
        self.assertEqual(len(ax.collections), 1)
        self.assertEqual(len(ax.collections[0].get_paths()), 2)
        self.assertTrue(ax.collections[0].get_rasterized())
        self.assertEqual(len(ax.patches), 1)
        plt.close(fig)

    def test_phase_swap_places_index_axon_and_outline_on_left(self):
        frame = self.rows([250], [1], ['I_first'])
        other = frame.copy(); other.target_axon_id = 'S01_140um_ph0'
        fig, ax = plt.subplots()
        grid_panel(ax, pd.concat([frame, other]),
                   {'S01_140um_ph0': 'white', 'S01_140um_ph1': 'yellow'}, 'Swapped', left_phase='ph1')
        self.assertEqual((ax.patches[0].theta1, ax.patches[0].theta2), (90, 270))
        self.assertEqual((ax.patches[1].theta1, ax.patches[1].theta2), (90, 270))
        self.assertEqual((ax.patches[2].theta1, ax.patches[2].theta2), (-90, 90))
        plt.close(fig)

    def test_larger_rings_and_compact_grid_preserve_circles(self):
        fig, ax = plt.subplots()
        grid_panel(ax, self.rows([250], [1], ['I_first']), {'S01_140um_ph1': 'yellow'},
                   'Larger', radius=.48, depth_spacing=1.10, tight_bounds=True, left_phase='ph1')
        self.assertAlmostEqual(ax.patches[0].r, .48)
        self.assertAlmostEqual(ax.patches[0].center[1], 2.2)
        self.assertEqual(ax.get_aspect(), 1)
        np.testing.assert_allclose(ax.get_xlim(), [.46, 13.54])
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
