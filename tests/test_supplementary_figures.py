import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from figure_generation.figure_model import REGIMES, SCOPES
from figure_generation.supplementary import count_source, index_counts, model_comparison, central_point_mask
from figure_generation.layout import observer_grid, panel_label, observer_headers, reliance_background


class SupplementaryFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_central_preview_selects_within_each_condition(self):
        frame = pd.DataFrame([dict(thresh_type='I_volley5', freq=f, cv=0.,
                                   display_count=x + f) for f in [20, 100] for x in range(100)])
        mask = central_point_mask(frame, interval=(.1, .9))
        self.assertEqual(mask.sum(), 160)
        self.assertTrue(central_point_mask(frame).all())
        self.assertEqual(frame.loc[mask & frame.freq.eq(20), 'display_count'].min(), 30)

    def test_shared_layout_matches_physical_spacing(self):
        main, a = observer_grid(True)
        supp, b = observer_grid(False)
        for col in range(5):
            self.assertAlmostEqual(a[0,col].get_position().width*main.get_figwidth(),
                                   b[0,col].get_position().width*supp.get_figwidth())
        gaps = [(a[0,c+1].get_position().x0-a[0,c].get_position().x1)*main.get_figwidth() for c in range(4)]
        self.assertGreater(gaps[-1], gaps[0])
        for letter, ax in [('a',a[0,0]),('b',a[0,4]),('c',a[2,0])]:
            panel_label(main,ax,letter)
        offsets = [(ax.get_position().x0-text.get_position()[0])*main.get_figwidth()
                   for ax,text in zip([a[0,0],a[0,4],a[2,0]],main.texts)]
        self.assertTrue(np.allclose(offsets,[1.04, .64, 1.04]))

    def test_observer_headers_and_background_do_not_change_data(self):
        fig, axes = observer_grid(True)
        for ax in axes.flat:
            ax.set_title('duplicate')
        observer_headers(fig, axes, [20, 50, 100, 200])
        self.assertTrue(all(not ax.get_title() for ax in axes.flat))
        self.assertEqual([t.get_text() for t in fig.texts],
                         ['20 Hz', '50 Hz', '100 Hz', '200 Hz', 'STD0', 'STD1'])
        self.assertTrue(all(t.get_fontweight() == 'bold' for t in fig.texts[:4]))
        self.assertGreater(fig.texts[4].get_position()[0], axes[0,0].get_position().x0-1.04/fig.get_figwidth())
        ax = axes[0, 4]
        ax.set_ylim(0, 1)
        reliance_background(ax)
        self.assertEqual(len(ax.patches), 2)
        self.assertEqual(ax.get_ylim(), (0, 1))
        self.assertTrue(all(p.get_alpha() == .045 for p in ax.patches))

    def test_count_units_and_repetition_points(self):
        frame = pd.DataFrame([dict(thresh_type=r, freq=f, cv=cv, rep_index=rep,
                                   n_index_spikes=10 + rep, n_input_pulses=f*2)
                              for r in REGIMES for f in [10, 20, 50, 100, 200]
                              for cv in np.linspace(0, 1, 11) for rep in range(3)])
        v2 = SimpleNamespace(index=frame, table=lambda *a: None, figure=lambda fig, stem: fig)
        source = count_source(v2, 'true')
        self.assertEqual(source.iloc[0].display_count, 10)
        row = source[(source.thresh_type == 'I_50') & (source.freq == 100)].iloc[0]
        self.assertEqual(row.display_count, 5)
        with self.assertRaises(ValueError):
            count_source(v2, 'unknown')
        fig = index_counts(v2)
        self.assertIsNone(fig._suptitle)
        self.assertEqual(len(fig.axes), 9)
        for ax in fig.axes:
            self.assertTrue(ax.get_title())
            self.assertEqual(len(ax.collections), 5)
            self.assertTrue(all(not isinstance(c, PolyCollection) for c in ax.collections))
            self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections), 165)
            self.assertEqual(len(ax.lines), 5)
            self.assertTrue(all(line.get_marker() == 'None' for line in ax.lines))

    def test_comparison_has_only_three_hybrid_structures(self):
        frames, mc, rmse = [], [], []
        for scope, family in [('primary', 'hybrid'), ('crossed', 'hybrid'), ('per_panel', 'hybrid'),
                              ('primary', 'rate'), ('primary', 'timing')]:
            ws, rs = SCOPES[scope]
            rmse.append(dict(weight_scope=ws, readout_scope=rs, state_mode='mixture',
                             model_family=family, rate_gate_mode='none', mixture_semantics='latent_trial',
                             split='test', freq='ALL', block='ALL', rmse_mean=.123))
            for block in ['std0', 'std1']:
                for freq in [20, 50, 100, 200]:
                    for cv in np.linspace(0, 1, 11):
                        base = dict(weight_scope=ws, readout_scope=rs, state_mode='mixture',
                                    model_family=family, rate_gate_mode='none', mixture_semantics='latent_trial',
                                    split='test', block=block, freq=freq, cv=cv, p_predicted_mean=.4 + .2*cv)
                        frames.append(base)
                        if family == 'hybrid':
                            for repetition in range(2):
                                mc.append(dict(base, mc=repetition, effective_rate_share=.3 + .1*repetition))
        curves = pd.DataFrame(frames)
        v2 = SimpleNamespace(focus={'curves': curves, 'rmse':pd.DataFrame(rmse)},
                             control={'curves': curves, 'rmse':pd.DataFrame(rmse)}, mc=pd.DataFrame(mc),
                             table=lambda *a: None, figure=lambda fig, stem: fig, animal_panel=lambda *a: None)
        fig = model_comparison(v2)
        self.assertEqual(len(fig.axes), 10)
        for i, ax in enumerate(fig.axes):
            self.assertEqual(len(ax.lines), 4 if i % 5 == 4 else 3)
        self.assertEqual(len(fig.legends[0].texts), 3)
        self.assertTrue(all('RMSE = 0.123' in t.get_text() for t in fig.legends[0].texts))


if __name__ == '__main__':
    unittest.main()
