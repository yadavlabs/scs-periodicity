import unittest
from copy import deepcopy
import nbformat
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from population.manuscript_v2 import tv, choice, integrated_share, ManuscriptFigures, REGIMES


class ManuscriptV2Tests(unittest.TestCase):
    def test_total_variation(self):
        self.assertAlmostEqual(float(tv([.2, .8], [.7, .3])), .5)
        with self.assertRaises(ValueError):
            tv([.2, .3], [.2, .8])

    def test_choice_only_flips_reference_endpoint(self):
        np.testing.assert_allclose(choice([0, .5, 1], [.8, .6, .9], "std0"), [.2, .6, .9])
        np.testing.assert_allclose(choice([0, .5, 1], [.8, .6, .9], "std1"), [.8, .6, .1])

    def test_integrated_share_is_not_unweighted_cv_mean(self):
        data = pd.DataFrame(dict(freq=[20]*11, cv=np.linspace(0, 1, 11), share=[1]+[0]*10))
        result = integrated_share(data, ["freq"], "share")
        self.assertAlmostEqual(result.rate_share_auc.iloc[0], .05)
        with self.assertRaises(ValueError):
            integrated_share(data.iloc[:-1], ["freq"], "share")

    def test_retained_regimes_do_not_drop_real_zero_weights(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.equiv = pd.DataFrame({"freq": [100] * 9, "representative_regime":
                                      ["I_volley10" if r == "I_5" else r for r in REGIMES]})
        self.assertEqual(figures.retained_regimes(100), [r for r in REGIMES if r != "I_5"])

    def test_collapsed_label_includes_all_class_members(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.equiv = pd.DataFrame(dict(freq=[100], representative_regime=["I_volley10"],
                                          member_regimes=["I_volley10|I_5"]))
        self.assertEqual(figures.collapsed_label(100, "I_volley10"), r"$I_{volley10}$ / $I_{5}$")

    def test_detection_behavior_asset_excludes_fits(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.det_animals = pd.DataFrame(dict(freq=[100, 100], rat=[1, 1], cv=[0, 1],
                                               norm_threshold=[.9, 1.]))
        figures.det = {
            "curves": pd.DataFrame(dict(target_kind=["datapoints"] * 4, freq=[100] * 4,
                mode=["best_single"] * 2 + ["per_point"] * 2, cv=[0, 1, 0, 1],
                prediction=[.92, .95, .9, 1.], behavior_mean=[np.nan, np.nan, .9, 1.],
                behavior_sem=[np.nan, np.nan, .02, .02])),
            "metrics": pd.DataFrame(dict(target_kind=["datapoints"], mode=["best_single"],
                                          freq=["ALL"], selected_single_regime=["I_5"]))}
        fig, ax = plt.subplots()
        figures.detection_panel(ax, 100, fits=False)
        self.assertEqual(ax.get_legend_handles_labels()[1], ["Rat mean ± SEM"])
        self.assertLess(ax.get_ylim()[0], .88)
        self.assertGreater(ax.get_ylim()[1], 1.02)
        plt.close(fig)

    def test_detection_shared_mixture_uses_saved_datapoint_curve(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.det_animals = pd.DataFrame(dict(freq=[100, 100], rat=[1, 1], cv=[0, 1],
                                               norm_threshold=[.9, 1.]))
        figures.det = {
            'curves': pd.DataFrame(dict(target_kind=['datapoints'] * 6, freq=[100] * 6,
                mode=['best_single'] * 2 + ['per_point'] * 2 + ['per_frequency'] * 2,
                cv=[0, 1] * 3, prediction=[.92, .95, .9, 1., .87, .98],
                behavior_mean=[np.nan, np.nan, .9, 1., np.nan, np.nan],
                behavior_sem=[np.nan, np.nan, .02, .02, np.nan, np.nan])),
            'metrics': pd.DataFrame(dict(target_kind=['datapoints'], mode=['best_single'],
                                          freq=['ALL'], selected_single_regime=['I_5']))}
        fig, ax = plt.subplots()
        figures.detection_panel(ax, 100, shared_mixture=True)
        shared = next(line for line in ax.lines if line.get_label() == 'Per-Frequency Mixture')
        np.testing.assert_allclose(shared.get_ydata(), [.87, .98])
        self.assertEqual(shared.get_linestyle(), '--')
        self.assertLess(ax.get_ylim()[0], .87)
        plt.close(fig)

    def test_fig4_has_eleven_ticks_and_per_column_cv_labels(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.observer_panel = lambda ax, *args, **kwargs: ax.plot([0, 1], [0, 1])
        figures.mixture_bars = lambda ax, freq: ax.bar([0, 1, 2], [.2, .4, .3])
        figures.figure = lambda fig, stem: fig
        fig = figures.fig4()
        for i, ax in enumerate(fig.axes[:8]):
            np.testing.assert_allclose(ax.get_xticks(), np.linspace(0, 1, 11))
            self.assertEqual(ax.get_xlabel(), "" if i < 4 else "Coefficient of Variation (CV)")
        self.assertEqual([t.get_text() for t in fig.legends[0].texts[:3]], ["Rate", "Timing", "Hybrid"])
        self.assertEqual(len(fig.legends[0].texts), 3)
        self.assertEqual([t.get_text() for t in fig.axes[4].get_xticklabels()],
                         ["0.0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1.0"])
        plt.close(fig)

    def test_combined_fig4_panel_groups_and_corner_legends(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.mc = pd.DataFrame([
            dict(mc=0, freq=freq, block=block, cv=cv, effective_rate_share=.5,
                 weight_scope="per_frequency", readout_scope="per_block", state_mode="mixture",
                 model_family="hybrid", rate_gate_mode="none", mixture_semantics="latent_trial",
                 split="test")
            for freq in [20, 50, 100, 200] for block in ["std0", "std1"]
            for cv in np.linspace(0, 1, 11)])
        figures.observer_panel = lambda ax, *args, **kwargs: ax.plot([0, 1], [0, 1])
        figures.mixture = lambda **kwargs: pd.DataFrame(
            1 / 9, index=[20, 50, 100, 200], columns=REGIMES)
        figures.retained_regimes = lambda freq: REGIMES
        figures.collapsed_label = lambda freq, regime: regime
        figures.table = lambda frame, stem: frame
        figures.figure = lambda fig, stem: fig
        figures.locked_cue_curves = lambda: pd.DataFrame()
        for style in ["bars", "grouped", "lines", "single"]:
            with self.subTest(style=style):
                fig = figures.fig4_reliance_preview(mixture_style=style, locked_mixtures=style == "single")
                letters = [t for t in fig.texts if t.get_text() in "abc"]
                self.assertEqual([t.get_text() for t in letters], ["a", "b", "c"])
                self.assertGreater(letters[1].get_position()[0], .75)
                self.assertEqual(letters[0].get_position()[1], letters[1].get_position()[1])
                self.assertEqual(fig.legends, [])
                legends = fig.axes[14].findobj(match=Legend)
                self.assertEqual(len(legends), 2 if style == "bars" else 1)
                if style == "bars":
                    self.assertEqual(legends[1]._ncols, 3)
                    self.assertEqual(len(legends[1].texts), 9)
                self.assertEqual([t.get_text() for t in legends[0].texts], ["Rate", "Timing", "Hybrid"])
                self.assertEqual(legends[0].texts[2].get_fontweight(), "bold")
                if style == "single":
                    for ax in fig.axes[10:14]:
                        self.assertEqual(len(ax.patches), len(REGIMES))
                        np.testing.assert_allclose([p.get_height() for p in ax.patches], 1 / 9)
                for ax in fig.axes[5:9]:
                    self.assertEqual(ax.get_xlabel(), "Coefficient of Variation (CV)")
                plt.close(fig)

    def test_three_mixture_styles_encode_identical_values(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.retained_regimes = lambda f: ["I_first", "I_volley10"]
        figures.collapsed_label = lambda f, r: r + (" / I_5" if r == "I_volley10" else "")
        mixtures = {f: pd.DataFrame([[a, 1-a]], index=[100], columns=["I_first", "I_volley10"])
                    for f, a in [("rate", .2), ("hybrid", .4), ("timing", .6)]}
        for style in ["bars", "grouped", "lines"]:
            fig, ax = plt.subplots()
            figures.independent_mixture_panel(ax, 100, mixtures, style)
            if style == "bars":
                np.testing.assert_allclose([p.get_height() for p in ax.patches], [.2, .4, .6, .8, .6, .4])
            elif style == "grouped":
                np.testing.assert_allclose([p.get_height() for p in ax.patches], [.2, .8, .4, .6, .6, .4])
            else:
                for line, expected in zip(ax.lines, [[.2, .8], [.4, .6], [.6, .4]]):
                    np.testing.assert_allclose(line.get_ydata(), expected)
            if style != "bars":
                self.assertIn("I_5", ax.get_xticklabels()[1].get_text())
                self.assertGreater(ax.get_ylim()[1], .8)
            plt.close(fig)

    def test_figure_role_install_is_selective_and_idempotent(self):
        from population.figure_generation.fig4_review import install, FIG4_CODE
        cells = [nbformat.v4.new_code_cell('user_custom_plot()'),
                 nbformat.v4.new_markdown_cell('## Figure 4 - old'),
                 nbformat.v4.new_code_cell('old_locked_plot()'),
                 nbformat.v4.new_markdown_cell('## Supplementary Figure 9 - old'),
                 nbformat.v4.new_code_cell('model_comparison(v2)')]
        notebook = nbformat.v4.new_notebook(cells=cells, metadata={'user_setting': 'keep'})
        original = deepcopy(notebook)
        install(notebook)
        self.assertEqual(notebook.cells[0], original.cells[0])
        self.assertEqual(notebook.metadata, original.metadata)
        self.assertEqual(notebook.cells[2].source, FIG4_CODE)
        self.assertIn('locked_mixtures=False', notebook.cells[2].source)
        self.assertIn("mixture_style='grouped'", notebook.cells[2].source)
        self.assertEqual(notebook.cells[4].source, original.cells[4].source)
        first = deepcopy(notebook)
        install(notebook)
        self.assertEqual(notebook, first)

    def test_locked_panel_uses_conditional_controls_but_original_hybrid(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.animal_panel = lambda *args: None
        rows = [dict(freq=100, block="std1", cv=cv, model_family=family,
                     p_predicted_mean=value, weight_scope="per_frequency", readout_scope="per_block",
                     state_mode="mixture", rate_gate_mode="none", mixture_semantics="latent_trial", split="test")
                for family, value in [("hybrid", .8), ("rate", .6), ("timing", .7)] for cv in [0., 1.]]
        figures.focus = figures.control = {"curves": pd.DataFrame(rows)}
        locked = pd.DataFrame([dict(model=family, freq=100, block="std1", cv=cv,
                                    p_predicted_mean=value)
                               for family, value in [("rate", .5), ("timing", .75), ("hybrid", .9)]
                               for cv in [0., 1.]])
        fig, ax = plt.subplots()
        figures.observer_panel(ax, 100, "std1", emphasize=True, locked_curves=locked)
        # Reference CV=1 is flipped to P(choose CV0). Hybrid is not the refitted .9 fixture.
        for line, expected in zip(ax.lines, [[.5, .5], [.75, .25], [.8, .2]]):
            np.testing.assert_allclose(line.get_ydata(), expected)
        self.assertEqual([line.get_linewidth() for line in ax.lines], [1.8, 1.8, 3.])
        self.assertEqual([line.get_alpha() for line in ax.lines], [.75, .75, 1.])
        plt.close(fig)

    def test_single_mixture_display_keeps_collapsed_labels(self):
        figures = ManuscriptFigures.__new__(ManuscriptFigures)
        figures.equiv = pd.DataFrame(dict(freq=[100, 100], representative_regime=["I_first", "I_volley10"],
                                          member_regimes=["I_first", "I_volley10|I_5"]))
        mixtures = pd.DataFrame([[.3, .7]], index=[100], columns=["I_first", "I_volley10"])
        fig, ax = plt.subplots()
        figures.single_mixture_bars(ax, 100, mixtures)
        self.assertEqual(len(ax.patches), 2)
        self.assertEqual(ax.get_xticklabels()[1].get_text(), r"$I_{volley10}$ / $I_{5}$")
        np.testing.assert_allclose([p.get_height() for p in ax.patches], [.3, .7])
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
