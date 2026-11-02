import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import replace

import numpy as np
import pandas as pd

from population.discrimination.discrimination_CLI import validate_outputs
from population.discrimination.discrimination_core import (
    Design,
    ModelSpec,
    PopulationObserverModel,
    _optimize_continuous,
    fit_best_single,
    fit_fixed_single,
)


def _spec(*, weight_scope="per_frequency", readout_scope="per_panel", state_mode="best_single"):
    return ModelSpec(
        frequencies=(20.0, 50.0),
        regimes=("I_first", "I_10"),
        weight_scope=weight_scope,
        readout_scope=readout_scope,
        state_mode=state_mode,
        model_family="timing",
        rate_gate_mode="none",
        mixture_semantics="latent_trial",
        loss_reg_l2=1e-3,
        weight_kl=0.0,
        rate_gate_reg=0.0,
        rate_gate_prior_floor=1.0,
        rate_gate_prior_center=0.5,
        rate_gate_prior_slope=8.0,
        fixed_selection_n_starts=4,
        active_regimes_by_frequency=((True, True), (True, True)),
        canonical_labels_by_frequency=(("I_first", "I_10"), ("I_first", "I_10")),
        member_labels_by_frequency=(("I_first", "I_10"), ("I_first", "I_10")),
    )


def _design():
    # Each frequency favors the opposite state.  Both standard blocks are
    # included so the exact per-frequency/per-panel factorization is exercised.
    freq = np.repeat([20.0, 50.0], 4)
    block = np.tile(np.repeat(["std0", "std1"], 2), 2)
    cv = np.tile([0.25, 0.75], 4)
    evidence = np.zeros((8, 2, 4), dtype=float)
    evidence[:4, 0, 2:] = np.asarray([[-1, -0.5], [1, 0.5], [-1, -0.5], [1, 0.5]])
    evidence[:4, 1, 2:] = 0.0
    evidence[4:, 0, 2:] = 0.0
    evidence[4:, 1, 2:] = np.asarray([[-1, -0.5], [1, 0.5], [-1, -0.5], [1, 0.5]])
    # A moderate target avoids boundary-only logistic solutions.
    p = np.asarray([0.75, 0.75, 0.75, 0.75] * 2, dtype=float)
    n = np.full(8, 40.0)
    return Design(
        freq=freq, block=block, cv=cv,
        k=p * n, n=n, p_observed=p, evidence=evidence,
    )


class PopulationObserverOptimizationTests(unittest.TestCase):
    def test_opposing_cues_are_block_specific_with_no_unused_parameters(self):
        base = _spec(readout_scope="per_block", state_mode="mixture")
        design = _design()
        design.evidence[:] = np.random.default_rng(7).normal(size=design.evidence.shape)
        for family, expected in [("rate_timing", ((0, 1), (2, 3))),
                                 ("timing_rate", ((2, 3), (0, 1)))]:
            model = PopulationObserverModel(replace(base, model_family=family))
            self.assertEqual(model.channel_groups, expected)
            self.assertEqual(model.n_params, 2 + 6)
            self.assertEqual(len(model.parameter_bounds()), model.n_params)
            theta = np.random.default_rng(8).normal(size=model.n_params)
            before = model.predict(theta, design)
            changed = replace(design, evidence=design.evidence.copy())
            for b, channels in zip(("std0", "std1"), expected):
                inactive = [i for i in range(4) if i not in channels]
                for channel in inactive:
                    changed.evidence[changed.block == b, :, channel] += 1000.
            np.testing.assert_array_equal(before["p_correct"], model.predict(theta, changed)["p_correct"])
            for block, channels in zip(("std0", "std1"), expected):
                inactive_drive = "timing_drive" if channels == (0, 1) else "rate_drive"
                np.testing.assert_array_equal(before[inactive_drive][design.block == block], 0.)

    def test_block_assignment_rejects_unsupported_sharing(self):
        with self.assertRaises(ValueError):
            PopulationObserverModel(replace(_spec(state_mode="mixture"), model_family="rate_timing"))

    def test_manuscript_mixture_prior_penalizes_collapsed_weights(self):
        spec = ModelSpec(
            frequencies=(20.0,),
            regimes=("I_first", "I_5", "I_10", "I_15"),
            weight_scope="per_frequency",
            readout_scope="per_block",
            state_mode="mixture",
            model_family="timing",
            rate_gate_mode="none",
            mixture_semantics="latent_trial",
            loss_reg_l2=0.0,
            weight_kl=0.01,
            rate_gate_reg=0.0,
            rate_gate_prior_floor=1.0,
            rate_gate_prior_center=0.5,
            rate_gate_prior_slope=8.0,
            fixed_selection_n_starts=1,
            active_regimes_by_frequency=((True, True, True, True),),
            canonical_labels_by_frequency=(("I_first", "I_5", "I_10", "I_15"),),
            member_labels_by_frequency=(("I_first", "I_5", "I_10", "I_15"),),
            weight_entropy_floor_k=3.0,
            weight_entropy_floor_lambda=0.01,
            weight_max_cap=0.65,
            weight_max_cap_lambda=0.01,
        )
        model = PopulationObserverModel(spec)
        design = Design(
            freq=np.asarray([20.0]), block=np.asarray(["std0"]),
            cv=np.asarray([0.5]), k=np.asarray([20.0]), n=np.asarray([40.0]),
            p_observed=np.asarray([0.5]), evidence=np.zeros((1, 4, 4), dtype=float),
        )
        uniform = np.zeros(model.n_params, dtype=float)
        collapsed = uniform.copy()
        collapsed[:3] = np.asarray([12.0, -12.0, -12.0])

        uniform_parts = model.objective_components(uniform, design)
        collapsed_parts = model.objective_components(collapsed, design)
        self.assertAlmostEqual(uniform_parts["weight_kl"], 0.0, places=10)
        self.assertAlmostEqual(uniform_parts["weight_entropy_floor"], 0.0, places=10)
        self.assertAlmostEqual(uniform_parts["weight_max_cap"], 0.0, places=10)
        self.assertGreater(collapsed_parts["weight_kl"], 0.0)
        self.assertGreater(collapsed_parts["weight_entropy_floor"], 0.0)
        self.assertGreater(collapsed_parts["weight_max_cap"], 0.0)
        self.assertGreater(collapsed_parts["total"], uniform_parts["total"])

    def test_retry_prefers_formally_converged_solution(self):
        model = PopulationObserverModel(_spec(state_mode="mixture"))
        design = _design()
        first = SimpleNamespace(
            x=np.zeros(model.n_params), fun=0.4, success=False,
        )
        second = SimpleNamespace(
            x=np.ones(model.n_params), fun=0.401, success=True,
        )
        with patch(
            "population.discrimination.discrimination_core.minimize",
            side_effect=[first, second],
        ) as mocked:
            theta, loss, success = _optimize_continuous(
                model, design, np.zeros(model.n_params), None
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(success)
        self.assertAlmostEqual(loss, 0.401)
        np.testing.assert_allclose(theta, 1.0)

    def test_factorized_best_single_contains_every_shared_single_choice(self):
        model = PopulationObserverModel(_spec())
        design = _design()
        best = fit_best_single(model, design, seed=11, n_starts=4)
        self.assertTrue(best[3])
        for regime in range(model.n_regimes):
            fixed = fit_fixed_single(
                model, design, np.asarray([regime, regime]),
                seed=100 + regime, n_starts=2,
            )
            self.assertTrue(fixed[3])
            self.assertLessEqual(best[2], fixed[2] + 1e-7)

    def test_validation_rejects_any_failed_fit(self):
        config = {
            "weight_scope": "per_frequency", "readout_scope": "per_panel",
            "state_mode": "mixture", "model_family": "hybrid",
            "rate_gate_mode": "none", "mixture_semantics": "latent_trial",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([
                {**config, "mc": 0, "split": split, "freq": "ALL", "block": "ALL",
                 "optimization_success": False, "objective_train": 0.5,
                 "rmse": 0.1, "mae": 0.1, "nll_per_trial": 0.5}
                for split in ("train", "test")
            ]).to_csv(root / "merged_results.csv", index=False)
            pd.DataFrame([
                {**config, "mc": 0, "weight_key": "freq=20", "weight": 1.0}
            ]).to_csv(root / "merged_weights.csv", index=False)
            pd.DataFrame([
                {**config, "mc": 0, "split": "test", "freq": 20.0,
                 "block": "std0", "cv": 0.5, "p_predicted": 0.5,
                 "rate_drive": 0.0, "timing_drive": 0.0, "gate": 1.0,
                 "effective_rate_share": 0.0}
            ]).to_csv(root / "merged_curves.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                validate_outputs(str(root), expected_mc=1, expected_configurations=1)
            report = json.loads((root / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_observer_fits"], 1)


if __name__ == "__main__":
    unittest.main()
