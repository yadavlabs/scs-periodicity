import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_generation.conditional_spatial import conditional_statistics, conditional_faces, manuscript_figure, MANUSCRIPT_STEM
from figure_generation.spatial_exploration import grid_panel


class ConditionalSpatialTests(unittest.TestCase):
    def test_notebook_entry_exports_one_named_figure(self):
        v2 = Mock(root=Path('repo'), output=Path('exports'))
        with patch('figure_generation.conditional_spatial.prepare_data', return_value=('stats', 'summaries')) as prepare, \
             patch('figure_generation.conditional_spatial.draw_figure', return_value='figure') as draw:
            result = manuscript_figure(v2)
        prepare.assert_called_once_with(v2.root, v2.output / f'{MANUSCRIPT_STEM}_data')
        draw.assert_called_once_with('stats', 'summaries')
        v2.figure.assert_called_once_with('figure', MANUSCRIPT_STEM)
        self.assertIs(result, v2.figure.return_value)

    def rows(self, counts):
        return pd.DataFrame([dict(mode='true', freq=100, cv=1., rep_index=i,
                                  target_axon_id='S01_140um_ph1', target_cutline_id='S01',
                                  target_diameter_um=14., n_spikes=n, n_input_pulses=200)
                             for i, n in enumerate(counts)])

    def test_conditional_mean_and_full_denominator(self):
        row = conditional_statistics(self.rows([0, 10, 0, 30, 0]), n_reps=5).iloc[0]
        self.assertEqual(row.conditional_mean_spikes, 20)
        self.assertEqual(row.recruitment_probability, .4)
        self.assertEqual(row.unconditional_mean_spikes, 8)
        self.assertEqual(row.unconditional_mean_spikes,
                         row.conditional_mean_spikes * row.recruitment_probability)

    def test_never_active_is_undefined(self):
        row = conditional_statistics(self.rows([0, 0, 0]), n_reps=3).iloc[0]
        self.assertTrue(np.isnan(row.conditional_mean_spikes))
        self.assertEqual(row.recruitment_probability, 0)

    def test_missing_duplicate_and_invalid_counts_rejected(self):
        data = self.rows([0, 10, 30])
        for invalid in [data.iloc[1:], pd.concat([data, data.iloc[:1]]), self.rows([0, -1, 2])]:
            with self.assertRaises(ValueError):
                conditional_statistics(invalid, n_reps=3)

    def test_only_fill_is_transparent(self):
        frame = conditional_statistics(self.rows([0, 10, 0, 30, 0]), n_reps=5)
        frame['equivalent_regime'] = 'I_10'
        faces = conditional_faces(frame)
        self.assertEqual(faces['S01_140um_ph1'][3], .4)
        fig, ax = plt.subplots()
        grid_panel(ax, frame, faces, 'Test', dense=True, left_phase='ph1')
        self.assertAlmostEqual(ax.collections[0].get_facecolors()[0, 3], .4)
        self.assertEqual(ax.collections[0].get_edgecolors()[0, 3], 1.)
        self.assertEqual(ax.patches[-1].theta1, 90)
        self.assertEqual(ax.patches[-1].get_edgecolor()[3], 1.)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
