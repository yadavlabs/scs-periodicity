import unittest

import numpy as np
import pandas as pd

from population.detection.index_operating_state_pilot import (
    PILOT_VERSION,
    analyze_local_samples,
    build_cv0_threshold_proposal,
    nonmonotonic_step_table,
    target_definition_collisions,
    target_spike_counts,
)
from population.feature_engineering.feature_core import (
    _bounded_signed_distance_evidence,
    _rate_evidence,
)


class PopulationIndexOperatingStatePilotTests(unittest.TestCase):
    def test_definition_collisions_follow_two_second_pulse_counts(self):
        targets_50 = target_spike_counts(50.0)
        self.assertEqual(targets_50["I_volley5"], targets_50["I_5"])
        self.assertEqual(targets_50["I_volley10"], targets_50["I_10"])
        collisions = target_definition_collisions()
        row = collisions[
            np.isclose(collisions["freq"], 50.0)
            & (collisions["target_spikes"] == 5)
        ].iloc[0]
        self.assertEqual(set(row["regimes"].split("|")), {"I_volley5", "I_5"})

    def test_nonmonotonic_steps_only_use_adjacent_fine_samples(self):
        samples = pd.DataFrame({
            "amp_uA": [1.00, 1.01, 1.02, 2.00, 2.01],
            "n_index_spikes": [1, 5, 3, 20, 19],
            "is_local_sweep": [True] * 5,
        })
        result = analyze_local_samples(samples, step_uA=0.01)
        self.assertEqual(result["n_nonmonotonic_steps"], 2)
        self.assertEqual(result["max_spike_drop"], 2)
        transitions = nonmonotonic_step_table(samples.assign(
            index_axon_id="index", freq=20.0, representative_rep_index=0,
            rep_label="rep0", sweep_step_uA=0.01,
        ))
        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions["spike_drop"].tolist(), [2, 1])

    def test_cv0_proposal_uses_maximum_of_two_representatives(self):
        thresholds = pd.DataFrame({
            "index_axon_id": ["index"] * 4,
            "freq": [20.0] * 4,
            "representative_rep_index": [0, 1, 0, 1],
            "thresh_type": ["I_first", "I_first", "I_all", "I_all"],
            "refined_amp_uA": [1.01, 1.02, 2.00, 2.10],
        })
        proposal = build_cv0_threshold_proposal(thresholds)
        self.assertEqual(proposal["pilot_version"].iloc[0], PILOT_VERSION)
        self.assertAlmostEqual(proposal["I_first0_uA"].iloc[0], 1.02)
        self.assertAlmostEqual(proposal["I_all0_uA"].iloc[0], 2.10)

    def test_silence_is_rate_information_only_when_endpoints_separate(self):
        table = pd.DataFrame({
            "n_spikes_1ms": [0.0, 0.0],
            "n_spikes_1ms__cv0": [0.0, 0.0],
            "n_spikes_1ms__cv1": [10.0, 0.0],
        })
        evidence, defined, reliability, available, _ = _rate_evidence(
            table, ["n_spikes_1ms"], k=2.5,
        )
        self.assertEqual(evidence.tolist(), [-1.0, 0.0])
        self.assertEqual(defined.tolist(), [True, True])
        self.assertEqual(reliability.tolist(), [1.0, 0.0])
        self.assertEqual(available.tolist(), [True, False])

    def test_silent_timing_feature_returns_zero_and_unavailable(self):
        table = pd.DataFrame({
            "silence_lv_10ms": [np.nan],
            "silence_lv_10ms__cv0": [0.2],
            "silence_lv_10ms__cv1": [0.8],
        })
        evidence, available = _bounded_signed_distance_evidence(table, "silence_lv_10ms")
        self.assertEqual(evidence.tolist(), [0.0])
        self.assertEqual(available.tolist(), [False])


if __name__ == "__main__":
    unittest.main()
