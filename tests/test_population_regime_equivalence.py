import unittest

import numpy as np
import pandas as pd

from population.discrimination.discrimination_core import ModelSpec, PopulationObserverModel
from population.feature_engineering.feature_core import (
    RESPONSE_EQUIVALENCE_COLUMNS,
    _regime_response_equivalence,
)


class PopulationRegimeEquivalenceTests(unittest.TestCase):
    def _rows(self, freq, regimes, values, cv0_counts):
        rows = []
        for cv in (0.0, 1.0):
            for rep in (0, 1):
                for regime in regimes:
                    value = float(values[regime][int(cv)])
                    row = {
                        "freq": float(freq),
                        "cv": cv,
                        "rep_index": rep,
                        "thresh_type": regime,
                        "n_input_pulses": int(round(2 * freq)),
                        "n_index_spikes": int(cv0_counts[regime] if cv == 0 else 0),
                    }
                    row.update({column: value for column in RESPONSE_EQUIVALENCE_COLUMNS})
                    rows.append(row)
        return rows

    def test_non_nominal_duplicate_keeps_label_closest_to_realized_count(self):
        rows = self._rows(
            200,
            ("I_first", "I_volley5", "I_10"),
            {
                "I_first": (-1, 1),
                "I_volley5": (-1, 1),
                "I_10": (-0.5, 0.5),
            },
            {"I_first": 6, "I_volley5": 6, "I_10": 42},
        )
        mapping = _regime_response_equivalence(pd.DataFrame(rows))
        duplicate = mapping[mapping["member_regimes"].eq("I_first|I_volley5")]
        self.assertEqual(set(duplicate["canonical_regime"]), {"I_volley5"})
        self.assertEqual(set(duplicate["representative_regime"]), {"I_volley5"})
        self.assertFalse(bool(duplicate.loc[duplicate["thresh_type"].eq("I_first"), "regime_active"].iloc[0]))
        self.assertTrue(bool(duplicate.loc[duplicate["thresh_type"].eq("I_volley5"), "regime_active"].iloc[0]))

    def test_nominal_duplicate_combines_names(self):
        rows = self._rows(
            100,
            ("I_volley10", "I_5"),
            {"I_volley10": (-1, 1), "I_5": (-1, 1)},
            {"I_volley10": 10, "I_5": 10},
        )
        mapping = _regime_response_equivalence(pd.DataFrame(rows))
        self.assertEqual(set(mapping["canonical_regime"]), {"I_volley10 + I_5"})
        self.assertEqual(int(mapping["regime_active"].sum()), 1)

    def test_per_frequency_mixture_parameterizes_only_active_classes(self):
        spec = ModelSpec(
            frequencies=(200.0,),
            regimes=("I_first", "I_volley5", "I_10"),
            weight_scope="per_frequency",
            readout_scope="per_block",
            state_mode="mixture",
            model_family="hybrid",
            rate_gate_mode="none",
            mixture_semantics="latent_trial",
            loss_reg_l2=1e-3,
            weight_kl=0.0,
            rate_gate_reg=0.0,
            rate_gate_prior_floor=1.0,
            rate_gate_prior_center=0.5,
            rate_gate_prior_slope=8.0,
            fixed_selection_n_starts=3,
            active_regimes_by_frequency=((False, True, True),),
            canonical_labels_by_frequency=(("I_volley5", "I_volley5", "I_10"),),
            member_labels_by_frequency=(("I_first|I_volley5", "I_first|I_volley5", "I_10"),),
        )
        model = PopulationObserverModel(spec)
        self.assertEqual(model.n_weight_params, 1)
        weights, _ = model.unpack(np.zeros(model.n_params))
        np.testing.assert_allclose(weights, [[0.0, 0.5, 0.5]])

    def test_per_block_weights_collapse_separately_at_each_frequency(self):
        spec = ModelSpec(
            frequencies=(20.0, 50.0),
            regimes=("I_first", "I_volley5", "I_10"),
            weight_scope="per_block",
            readout_scope="per_frequency",
            state_mode="mixture",
            model_family="hybrid",
            rate_gate_mode="none",
            mixture_semantics="latent_trial",
            loss_reg_l2=1e-3,
            weight_kl=0.0,
            rate_gate_reg=0.0,
            rate_gate_prior_floor=1.0,
            rate_gate_prior_center=0.5,
            rate_gate_prior_slope=8.0,
            fixed_selection_n_starts=3,
            active_regimes_by_frequency=(
                (False, True, True),
                (True, True, True),
            ),
            canonical_labels_by_frequency=(
                ("I_volley5", "I_volley5", "I_10"),
                ("I_first", "I_volley5", "I_10"),
            ),
            member_labels_by_frequency=(
                ("I_first|I_volley5", "I_first|I_volley5", "I_10"),
                ("I_first", "I_volley5", "I_10"),
            ),
        )
        model = PopulationObserverModel(spec)
        self.assertEqual(model.weight_keys, ("std0", "std1"))
        self.assertEqual(model.n_weight_params, 4)
        weights, _ = model.unpack(np.zeros(model.n_params))
        views = model.collapsed_weight_views(weights)
        self.assertEqual(len(views), 4)
        self.assertEqual(views[0][2], ("I_volley5", "I_10"))
        np.testing.assert_allclose(views[0][3], [2.0 / 3.0, 1.0 / 3.0])
        self.assertEqual(views[1][2], ("I_first", "I_volley5", "I_10"))
        np.testing.assert_allclose(views[1][3], [1.0 / 3.0] * 3)

    def test_per_panel_uses_frequency_specific_active_classes(self):
        spec = ModelSpec(
            frequencies=(20.0, 50.0),
            regimes=("I_first", "I_volley5", "I_10"),
            weight_scope="per_panel",
            readout_scope="per_panel",
            state_mode="mixture",
            model_family="hybrid",
            rate_gate_mode="none",
            mixture_semantics="latent_trial",
            loss_reg_l2=1e-3,
            weight_kl=0.0,
            rate_gate_reg=0.0,
            rate_gate_prior_floor=1.0,
            rate_gate_prior_center=0.5,
            rate_gate_prior_slope=8.0,
            fixed_selection_n_starts=3,
            active_regimes_by_frequency=((False, True, True), (True, True, True)),
            canonical_labels_by_frequency=(
                ("I_volley5", "I_volley5", "I_10"),
                ("I_first", "I_volley5", "I_10"),
            ),
            member_labels_by_frequency=(
                ("I_first|I_volley5", "I_first|I_volley5", "I_10"),
                ("I_first", "I_volley5", "I_10"),
            ),
        )
        model = PopulationObserverModel(spec)
        self.assertEqual(len(model.weight_keys), 4)
        self.assertEqual(model.n_weight_params, 6)
        weights, _ = model.unpack(np.zeros(model.n_params))
        np.testing.assert_allclose(weights[0], [0.0, 0.5, 0.5])
        np.testing.assert_allclose(weights[1], [0.0, 0.5, 0.5])
        np.testing.assert_allclose(weights[2], [1.0 / 3.0] * 3)


if __name__ == "__main__":
    unittest.main()
