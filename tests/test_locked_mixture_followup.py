from dataclasses import replace
import unittest
import numpy as np
import pandas as pd

from population.discrimination import discrimination_core as core
from population.discrimination.locked_mixture_followup import (
    LockedMixtureModel, readout_start, audit_source, FAMILIES, GAINS)
from test_population_observer_optimization import _spec, _design


class LockedMixtureTests(unittest.TestCase):
    def setUp(self):
        self.spec = replace(_spec(readout_scope="per_block", state_mode="mixture"), model_family="hybrid")
        self.weights = np.array([[.2, .8], [.7, .3]])

    def test_weights_never_depend_on_optimized_parameters(self):
        for family in FAMILIES:
            model = LockedMixtureModel(replace(self.spec, model_family=family), self.weights)
            self.assertEqual(model.n_params, 10 if family == "hybrid" else 6)
            self.assertEqual(model.n_weight_params, 0)
            self.assertEqual(len(model.parameter_bounds()), model.n_params)
            for scale in [0, 1, 10]:
                theta = model.initial_theta(np.random.default_rng(1)) * scale
                np.testing.assert_array_equal(model.unpack(theta)[0], self.weights)
            with self.assertRaises(ValueError):
                model.locked_weights[0, 0] = .9

    def test_locked_predictions_reproduce_free_model(self):
        free = core.PopulationObserverModel(self.spec)
        theta = free.initial_theta(np.random.default_rng(10))
        w, _ = free.unpack(theta)
        locked = LockedMixtureModel(self.spec, w)
        design = _design()
        np.testing.assert_allclose(free.predict(theta, design)["p_correct"],
                                   locked.predict(theta[free.n_weight_params:], design)["p_correct"], atol=0, rtol=0)
        self.assertEqual(free.objective(theta, design), locked.objective(theta[free.n_weight_params:], design))

    def test_reject_invalid_or_inactive_weights_and_scope(self):
        for weights in [self.weights * 2, np.full((2, 2), np.nan), [[-.1, 1.1], [.5, .5]]]:
            with self.assertRaises(ValueError):
                LockedMixtureModel(self.spec, weights)
        with self.assertRaises(ValueError):
            LockedMixtureModel(replace(self.spec, active_regimes_by_frequency=((True, False), (True, True))), self.weights)
        with self.assertRaises(ValueError):
            LockedMixtureModel(replace(self.spec, readout_scope="per_frequency"), self.weights)

    def test_source_gain_projection_and_forbidden_cues(self):
        params = pd.DataFrame([dict(block=b, bias=.2, **dict(zip(GAINS, [.1, .2, .3, .4]))) for b in ["std0", "std1"]])
        design = _design()
        for family in FAMILIES:
            model = LockedMixtureModel(replace(self.spec, model_family=family), self.weights)
            theta = readout_start(model, params)
            _, readouts = model.unpack(theta)
            for channels, ro in zip(model.channel_groups, readouts):
                np.testing.assert_allclose(ro["gains"], np.array([.1, .2, .3, .4])[list(channels)])
            altered = replace(design, evidence=design.evidence.copy())
            for block, channels in zip(model.readout_keys, model.channel_groups):
                for channel in set(range(4)) - set(channels):
                    altered.evidence[altered.block == block, :, channel] += 1000
            np.testing.assert_array_equal(model.predict(theta, design)["p_correct"], model.predict(theta, altered)["p_correct"])

    def test_reconstruction_stops_on_changed_predictions(self):
        model = LockedMixtureModel(self.spec, self.weights)
        theta = model.initial_theta(np.random.default_rng(2))
        design = _design()
        pred = model.predict(theta, design)
        curves = pd.DataFrame(dict(split="train", freq=design.freq, block=design.block, cv=design.cv,
            p_observed=design.p_observed, p_predicted=pred["p_correct"], rate_drive=pred["rate_drive"],
            timing_drive=pred["timing_drive"], effective_rate_share=pred["effective_rate_share"]))
        result = pd.DataFrame(dict(objective_train=[model.objective(theta, design)]))
        self.assertEqual(audit_source(model, theta, {"train": design}, curves, result)["max_prediction_gap"], 0)
        curves.loc[0, "p_predicted"] += .01
        with self.assertRaises(ValueError):
            audit_source(model, theta, {"train": design}, curves, result)


if __name__ == "__main__":
    unittest.main()
