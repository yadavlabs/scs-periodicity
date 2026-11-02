import unittest

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from population.figure_generation.spatial_prototype import (
    summarize_trials, draw_curve_panel, count_equivalent_regimes, onset_cv_tag, CVS, REGIMES,
)


class SpatialPrototypeTests(unittest.TestCase):
    def test_onset_cv_tags_keep_single_and_multiple_exports_distinct(self):
        self.assertEqual(onset_cv_tag([1.]), 'CV1')
        self.assertEqual(onset_cv_tag([0., .5, 1.]), 'CV0_0p5_1')
        for invalid in [[], [0., 0.], [.55]]:
            with self.assertRaises(ValueError):
                onset_cv_tag(invalid)

    def test_count_equivalent_categories_preserve_aliases_and_zero(self):
        result = count_equivalent_regimes([0, 1, 4, 5, 10, 20, 22], 20, 20)
        self.assertEqual(result.equivalent_regime.tolist(),
                         ['none', 'I_first', 'I_15', 'I_volley5', 'I_volley10', 'I_all', 'I_all'])
        self.assertEqual(result.iloc[1].equivalent_aliases, 'I_first / I_5')
        self.assertEqual(result.iloc[4].equivalent_aliases, 'I_volley10 / I_50')
        self.assertEqual(result.iloc[2].reference_count, 3)  # Midpoint chooses lower count.
        self.assertTrue(result.iloc[-1].above_imax_reference)

    def test_imax_color_anchor_is_observed_reference_not_pulse_count(self):
        result = count_equivalent_regimes([200, 320], 400, 320)
        self.assertEqual(result.equivalent_regime.tolist(), ['I_50', 'I_all'])
        self.assertEqual(result.iloc[-1].reference_count, 320)
        self.assertEqual(result.iloc[-1].absolute_count_error, 0)
        with self.assertRaises(ValueError):
            count_equivalent_regimes([1], 400, 100)

    def trials(self):
        return pd.DataFrame([
            dict(mode=mode, freq=100, cv=cv, thresh_type=regime, rep_index=rep,
                 active_axon_count=rep + 1, population_spikes=spikes)
            for mode in ['true', 'cv0'] for cv in [0., 1.]
            for regime in ['I_10', 'I_all'] for rep, spikes in enumerate([2, 4, 90])])

    def summary(self, frame):
        return summarize_trials(frame, [100], [0., 1.], ['I_10', 'I_all'], n_reps=3)

    def test_medians_use_all_repetitions_and_preserve_conditions(self):
        result = self.summary(self.trials())
        self.assertEqual(len(result), 8)
        self.assertTrue(result.median_active_axons.eq(2).all())
        self.assertTrue(result.median_population_spikes.eq(4).all())
        self.assertTrue(result.n_repetitions.eq(3).all())

    def test_missing_duplicate_and_mismatched_repetitions_fail(self):
        rows = self.trials()
        mismatch = rows.copy()
        mismatch.loc[0, 'rep_index'] = 9
        for invalid in [rows.iloc[1:], pd.concat([rows, rows.iloc[:1]]), mismatch]:
            with self.assertRaises(ValueError):
                self.summary(invalid)

    def test_zeros_are_preserved(self):
        rows = self.trials()
        rows.loc[rows['mode'].eq('cv0'), ['active_axon_count', 'population_spikes']] = 0
        result = self.summary(rows)
        self.assertTrue(result[result['mode'].eq('cv0')].median_active_axons.eq(0).all())

    def test_log_curves_break_at_zero_and_mark_separate_strip(self):
        summary = pd.DataFrame([
            dict(freq=100, mode=mode, cv=cv, thresh_type=regime,
                 median_active_axons=0 if mode == 'cv0' and cv == .5 and regime == 'I_first' else 3)
            for mode in ['true', 'cv0'] for cv in CVS for regime in REGIMES])
        fig = plt.figure()
        ax, zero_ax = draw_curve_panel(fig, fig.add_gridspec(1, 1)[0], summary,
                                      100, 'median_active_axons', 'Recruited axons', True)
        self.assertEqual(ax.get_yscale(), 'log')
        self.assertEqual(zero_ax.get_yscale(), 'linear')
        self.assertEqual(len(ax.lines), 18)
        self.assertTrue(np.isnan(ax.lines[1].get_ydata()[5]))
        self.assertEqual(sum(len(line.get_xdata()) for line in zero_ax.lines), 1)
        self.assertEqual(zero_ax.lines[1].get_xdata()[0], .5)
        np.testing.assert_allclose(ax.get_ylim(), [2.4, 3.6])
        plt.close(fig)

    def test_curve_limits_are_frequency_specific_and_show_tick_labels(self):
        summary = pd.DataFrame([
            dict(freq=freq, mode=mode, cv=cv, thresh_type=regime,
                 median_active_axons=(3 if freq == 10 else 200))
            for freq in [10, 100] for mode in ['true', 'cv0'] for cv in CVS for regime in REGIMES])
        fig = plt.figure()
        grid = fig.add_gridspec(1, 2)
        first, _ = draw_curve_panel(fig, grid[0], summary, 10, 'median_active_axons', 'Axons', True)
        second, strip = draw_curve_panel(fig, grid[1], summary, 100, 'median_active_axons', 'Axons', False,
                                       show_x_labels=False)
        self.assertLess(first.get_ylim()[1], second.get_ylim()[0])
        self.assertTrue(all(tick.get_visible() for tick in second.get_yticklabels()))
        self.assertEqual(strip.get_xlabel(), '')
        self.assertFalse(strip.get_xticklabels())
        self.assertGreater(len(strip.get_xticks()), 0)
        self.assertTrue(strip.xaxis.get_major_ticks()[0].tick1line.get_visible())
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
