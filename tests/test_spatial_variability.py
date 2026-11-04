import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import LogFormatterMathtext

from figure_generation.figure_model import REGIMES
from figure_generation.spatial_variability import (
    choose_cvs, repetition_panel, fill_spatial_column, upper_count_tick, draw_median_segments, manuscript_figure)


class SpatialVariabilityTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_manuscript_entry_promotes_one_compact_figure_and_selected_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'suppfig5_population_recruitment_data/_data'
            data.mkdir(parents=True)
            (data / 'provenance.json').write_text(json.dumps({'map_cvs': [0, .5, 1]}))
            stats = pd.DataFrame({'freq': [20,20,20], 'cv': [0,.5,1]})
            trials = pd.DataFrame({'test': [1]})
            scores = pd.DataFrame({'cv': [1]})
            records = []
            v2 = SimpleNamespace(root=root, output=root, figure=lambda fig, stem: records.append((fig,stem)))
            with patch('figure_generation.spatial_variability.prepare_data', return_value=(stats,None)), \
                 patch('figure_generation.spatial_variability.load_trials', return_value=trials), \
                 patch('figure_generation.spatial_variability.choose_cvs', return_value=({20:1.},scores)), \
                 patch('figure_generation.spatial_variability.prototype', return_value='figure') as draw:
                manuscript_figure(v2)
                self.assertTrue(draw.call_args.kwargs['compact'])
            self.assertEqual(records, [('figure','suppfig5_population_recruitment')])
            self.assertEqual(pd.read_csv(data/'compact_spatial_plot_data.csv').cv.tolist(), [1])
            provenance = json.loads((data/'provenance.json').read_text())
            self.assertEqual(provenance['map_cvs'], [1])
            self.assertEqual(provenance['cached_source_cvs'], [0,.5,1])

    def test_spatial_rings_fill_column_without_distortion(self):
        fig = plt.figure(figsize=(17, 10.5))
        ax = fig.add_axes([.1, .7, .2, .05])
        ax.set(xlim=(.46, 13.54), ylim=(-.54, 2.74), aspect='equal')
        original = ax.get_position(original=True)
        self.assertLess(ax.get_position().width, original.width)
        fill_spatial_column(ax)
        actual = ax.get_position()
        self.assertAlmostEqual(actual.x0, original.x0)
        self.assertAlmostEqual(actual.width, original.width)
        self.assertAlmostEqual((actual.y0 + actual.y1) / 2, (original.y0 + original.y1) / 2)
        self.assertAlmostEqual(actual.width * 17 / 13.08, actual.height * 10.5 / 3.28)

    def test_selection_uses_true_mean_not_extreme_or_cv0_maximum(self):
        trials = pd.DataFrame([
            dict(mode=mode, freq=20, cv=cv, rep_index=rep, thresh_type='I_all', active_axon_count=value)
            for mode, cv, values in [('true', 0., [1, 9]), ('true', .5, [6, 6]),
                                      ('true', 1., [6, 6]), ('cv0', 1., [8, 8])]
            for rep, value in enumerate(values)])
        chosen, scores = choose_cvs(trials)
        self.assertEqual(chosen, {20: .5})  # Tie goes to smaller CV.
        self.assertTrue(scores.loc[scores.selected_matched_cv, 'cv'].eq(.5).all())

    def test_full_raw_range_and_separate_zero_strip(self):
        trials = pd.DataFrame([
            dict(mode=mode, freq=20, cv=cv, rep_index=rep, thresh_type=r, active_axon_count=value)
            for mode in ['true', 'cv0'] for r in REGIMES for cv in [0., 1.]
            for rep, value in enumerate([0, 1, 2, 300] if mode == 'true' else [0, 1, 1, 5])])
        fig = plt.figure(); grid = fig.add_gridspec(2, 1)
        first = repetition_panel(fig, grid[0], trials, 20, 'true', 'active_axon_count', False, True)
        second = repetition_panel(fig, grid[1], trials, 20, 'cv0', 'active_axon_count', True, True)
        self.assertEqual(first[0].get_ylim(), second[0].get_ylim())
        self.assertGreater(first[0].get_ylim()[1], 300)
        for axes in [first, second]:
            self.assertEqual(sum(len(c.get_offsets()) for ax in axes for c in ax.collections), 72)
            self.assertTrue(all(np.all(c.get_offsets()[:, 1] > 0) for c in axes[0].collections))
            self.assertTrue(all(np.all(c.get_offsets()[:, 1] == 0) for c in axes[1].collections))
        compact = plt.figure(); compact_grid = compact.add_gridspec(2, 1)
        true = repetition_panel(compact, compact_grid[0], trials, 20, 'true', 'active_axon_count',
                                False, True, separate_ranges=True)[0]
        cv0 = repetition_panel(compact, compact_grid[1], trials, 20, 'cv0', 'active_axon_count',
                               True, True, separate_ranges=True)[0]
        self.assertGreater(true.get_ylim()[1], 300)
        self.assertGreater(cv0.get_ylim()[1], 5)
        self.assertLess(cv0.get_ylim()[1], 11)
        self.assertGreater(max(cv0.get_yticks()), 5)
        self.assertLess(max(cv0.get_yticks()), cv0.get_ylim()[1])
        self.assertIsInstance(cv0.yaxis.get_major_formatter(), LogFormatterMathtext)
        self.assertEqual(cv0.yaxis.get_major_formatter()(2000), '')
        self.assertIn('10^{3}', cv0.yaxis.get_major_formatter()(1000))
        self.assertIn(200, true.yaxis.get_minorticklocs())
        self.assertIn(500, true.yaxis.get_minorticklocs())
        self.assertLess(500, true.get_ylim()[1])

    def test_upper_ticks_are_nice_and_strictly_above_observations(self):
        self.assertEqual(upper_count_tick(1271), 2000)
        self.assertEqual(upper_count_tick(300), 500)
        self.assertEqual(upper_count_tick(1000), 2000)
        for maximum in [1, 5, 15, 59, 300, 1271, 62000]:
            self.assertGreater(upper_count_tick(maximum), maximum)

    def test_zero_segments_do_not_connect_across_positive_medians(self):
        fig, (ax, zero) = plt.subplots(2)
        medians = pd.Series([6, 0, 0, 0, 0, 3, 15, 0], index=np.arange(8)/10)
        draw_median_segments(ax, zero, medians, 'purple')
        np.testing.assert_allclose(zero.lines[0].get_ydata(), [np.nan, 0, 0, 0, 0, np.nan, np.nan, 0], equal_nan=True)
        self.assertEqual(zero.lines[0].get_linestyle(), '-')
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [6, np.nan, np.nan, np.nan, np.nan, 3, 15, np.nan], equal_nan=True)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [6])

    def test_transition_guides_connect_adjacent_medians_without_pseudocounts(self):
        fig, (ax, zero) = plt.subplots(2)
        medians = pd.Series([6, 0, 0, 3, 15, 0], index=np.arange(6)/10)
        draw_median_segments(ax, zero, medians, 'purple', connect_transitions=True)
        guides = [artist for artist in fig.artists if isinstance(artist, ConnectionPatch)]
        self.assertEqual(len(guides), 3)
        self.assertEqual(guides[0].xy1, (0., 6))
        self.assertEqual(guides[0].xy2, (.1, 0))
        self.assertIs(guides[0].axesA, ax)
        self.assertIs(guides[0].axesB, zero)
        self.assertEqual(guides[1].xy1, (.2, 0))
        self.assertEqual(guides[1].xy2, (.3, 3))


if __name__ == '__main__':
    unittest.main()
