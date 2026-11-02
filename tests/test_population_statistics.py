"""Numerics, pairing, correction families, and non-destructive notebook insertion."""
from copy import deepcopy
from dataclasses import replace
import unittest

import nbformat
import numpy as np
import pandas as pd
from scipy.stats import t

from population.discrimination import discrimination_core as core
from population.figure_generation.statistics import (
    holm, corrected_resampled, resampling_contrast, predict_saved, panel_parameters,
    rmse_tests, check_grid, reliance_tests)
from population.figure_generation.refresh_statistics import install, PREFIX
from test_population_observer_optimization import _spec, _design


class PopulationStatisticsTests(unittest.TestCase):
    def test_holm_preserves_order(self):
        np.testing.assert_allclose(holm([.03, .001, .04]), [.06, .003, .06])
        with self.assertRaises(ValueError):
            holm([np.nan])

    def test_corrected_resampling_formula(self):
        d = np.array([.1, .3, .2, -.1])
        result = corrected_resampled(d)
        se = np.sqrt((1/4+.25)*d.var(ddof=1))
        self.assertAlmostEqual(result['corrected_se'], se)
        self.assertAlmostEqual(result['corrected_t_p'], 2*t.sf(abs(d.mean()/se), 3))
        self.assertEqual(corrected_resampled(np.zeros(5))['corrected_t_p'], 1)
        self.assertTrue(np.isnan(corrected_resampled(np.ones(5))['corrected_t_p']))

    def test_signflip_and_bootstrap_are_deterministic_and_nonzero(self):
        a = np.arange(10.)/10
        kwargs = dict(paired=True, n_perm=999, n_boot=200)
        r1 = resampling_contrast(a+1, a, rng=np.random.default_rng(1), **kwargs)
        r2 = resampling_contrast(a+1, a, rng=np.random.default_rng(1), **kwargs)
        self.assertEqual(r1, r2)
        self.assertGreaterEqual(r1['p_raw'], .001)
        self.assertLess(r1['p_raw'], .03)
        self.assertAlmostEqual(r1['ci_low'], 1)
        equal = resampling_contrast(a, a, rng=np.random.default_rng(1), **kwargs)
        self.assertEqual(equal['p_raw'], 1)
        with self.assertRaises(ValueError):
            resampling_contrast(a, a[:-1], rng=np.random.default_rng(1), **kwargs)

    def test_unpaired_permutation_null(self):
        r = resampling_contrast(np.zeros(5), np.zeros(7), paired=False,
            rng=np.random.default_rng(9), n_perm=99, n_boot=50)
        self.assertEqual(r['p_raw'], 1)
        self.assertEqual(r['ci_high'], 0)

    def test_vectorized_prediction_matches_production_all_scopes(self):
        design = _design()
        evidence = np.random.default_rng(7).uniform(-1, 1, design.evidence.shape)
        design = replace(design, evidence=evidence)
        behavior = pd.DataFrame(dict(freq=design.freq, cv=design.cv, block=design.block))
        for ws, rs in [('per_frequency', 'per_block'), ('per_panel', 'per_panel'), ('per_block', 'per_frequency')]:
            for family in ['hybrid', 'rate', 'timing']:
                spec = replace(_spec(weight_scope=ws, readout_scope=rs, state_mode='mixture'), model_family=family)
                model = core.PopulationObserverModel(spec)
                theta = model.initial_theta(np.random.default_rng(12))
                weights, readouts = model.unpack(theta)
                wr = [dict(weight_key=k, thresh_type=r, weight=weights[i, j])
                      for i, k in enumerate(model.weight_keys) for j, r in enumerate(spec.regimes)]
                pr = []
                for key, channels, ro in zip(model.readout_keys, model.channel_groups, readouts):
                    gains = np.zeros(4)
                    gains[list(channels)] = ro['gains']
                    pr.append(dict(readout_key=key, bias=ro['bias'], **dict(zip(
                        ['gain_rate_1ms', 'gain_rate_10ms', 'gain_timing_1ms', 'gain_timing_10ms'], gains))))
                w, g, b = panel_parameters(pd.DataFrame(wr), pd.DataFrame(pr), behavior, spec.regimes, ws, rs)
                actual = predict_saved(evidence, w, g, b, design.cv, design.block)
                expected = model.predict(theta, design)
                for key, production_key in [('p_predicted', 'p_correct'), ('effective_rate_share', 'effective_rate_share'),
                                             ('rate_drive', 'rate_drive'), ('timing_drive', 'timing_drive')]:
                    np.testing.assert_allclose(actual[key], expected[production_key], atol=1e-14, rtol=0)
                repeated = predict_saved(np.stack([evidence, evidence]), w, g, b, design.cv, design.block)
                np.testing.assert_allclose(repeated['p_predicted'], np.stack([actual['p_predicted']]*2))

    def test_zero_drive_and_template_mapping(self):
        evidence = np.zeros((4, 1, 4))
        out = predict_saved(evidence, np.ones((4, 1)), np.ones((4, 4)), np.ones(4),
                            np.array([0, 1, 0, 1]), np.array(['std0', 'std0', 'std1', 'std1']))
        np.testing.assert_array_equal(out['effective_rate_share'], 0)
        self.assertTrue(out['zero_drive'].all())
        self.assertLess(out['p_predicted'][0], .5)
        self.assertGreater(out['p_predicted'][1], .5)
        self.assertLess(out['p_predicted'][2], .5)
        self.assertGreater(out['p_predicted'][3], .5)

    def test_block_rmse_is_arithmetic_not_pooled_and_rejects_missing_pairs(self):
        frame = pd.DataFrame([dict(model=m, mc=mc, freq=f, block=b, rmse=(i+1)*.01+mc*.0001+(m=='b')*.005)
            for m in ['a', 'b', 'c'] for mc in range(5) for i, f in enumerate([20, 50, 100, 200]) for b in ['std0', 'std1']])
        means, result = rmse_tests(frame.sample(frac=1, random_state=2), {'cue': ['a', 'b', 'c']}, .25)
        self.assertAlmostEqual(means.query("model == 'a' and mc == 0").block_mean_rmse.iloc[0], .025)
        self.assertEqual(len(result), 6)
        self.assertTrue(result.family_size.eq(3).all())
        with self.assertRaises(ValueError):
            rmse_tests(frame[~((frame.model == 'b') & (frame.mc == 0))], {'cue': ['a', 'b', 'c']}, .25)

    def test_grid_allows_roundoff_but_not_missing_rows(self):
        frame = pd.MultiIndex.from_product([[0], ['train', 'test'], [20., 50., 100., 200.],
            ['std0', 'std1'], np.linspace(0, 1, 11)], names=['mc', 'split', 'freq', 'block', 'cv']).to_frame(index=False)
        frame.cv = frame.cv.round(1)
        check_grid(frame, [0])
        with self.assertRaises(ValueError):
            check_grid(frame.iloc[:-1], [0])

    def test_reliance_families_and_replication_guard(self):
        data = pd.DataFrame([dict(model=m, freq=f, block=b, rep_index=r, auc=r/200+(b=='std0')*.2)
            for m in ['primary_hybrid', 'per_panel_hybrid'] for f in [20., 50., 100., 200.]
            for b in ['std0', 'std1'] for r in range(100)])
        result = reliance_tests(data, seed=7, n_perm=99, n_boot=30)
        self.assertEqual(len(result), 32)
        self.assertTrue(result.query("family == 'block_contrasts'").family_size.eq(8).all())
        self.assertTrue(result.query("family == 'frequency_contrasts'").family_size.eq(24).all())
        with self.assertRaises(ValueError):
            reliance_tests(pd.concat([data, data.iloc[:1]]), seed=7, n_perm=10, n_boot=10)

    def test_notebook_install_preserves_other_cells_and_metadata(self):
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('custom = 1'),
            nbformat.v4.new_markdown_cell('## Supplementary Figure 1 - manual')],
            metadata={'kernelspec': {'name': 'python3', 'display_name': '.venv (3.12.10)'}})
        before = deepcopy(nb)
        install(nb)
        first = deepcopy(nb)
        install(nb)
        self.assertEqual(nb, first)
        self.assertEqual(nb.metadata, before.metadata)
        self.assertEqual([c for c in nb.cells if not c.id.startswith(PREFIX)], before.cells)


if __name__ == '__main__':
    unittest.main()
