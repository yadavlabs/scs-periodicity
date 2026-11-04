import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from population.protocol import DETECTION_PROTOCOL_VERSION
from population.detection.refresh_imax_population import prepare, merge, validate_replacements


class RefreshTests(unittest.TestCase):
    def fixture(self, root):
        data = root / "data/detection/population"
        data.mkdir(parents=True)
        basis = root / "data/space-time-V"
        basis.mkdir(parents=True)
        pd.DataFrame({"axon_id": ["a", "b"]}).to_csv(basis / "axon_basis_manifest.csv", index=False)
        index = pd.DataFrame([
            dict(freq=20, cv=0, rep_index=0, thresh_type="I_first", amp_uA=250, n_index_spikes=1,
                 detection_protocol_version=DETECTION_PROTOCOL_VERSION),
            dict(freq=20, cv=0, rep_index=0, thresh_type="I_all", amp_uA=300, n_index_spikes=39,
                 detection_protocol_version=DETECTION_PROTOCOL_VERSION)])
        index.to_csv(data / "index_regime_amplitudes.csv", index=False)
        feats = root / "data/features"
        feats.mkdir(parents=True)
        index.to_csv(feats / "population_cv0_based_features.csv", index=False)
        rows = []
        for axon in ["a", "b"]:
            for row in index.to_dict("records"):
                rows.append(dict(row, target_axon_id=axon, n_spikes=1, spike_times_ms="[101.0]",
                                 detection_protocol_version="old", amp_uA=1500 if row["thresh_type"] == "I_all" else 250))
        pd.DataFrame(rows).to_csv(data / "population_responses.csv", index=False)
        return data, index

    def test_complete_scan_stage_preserves_other_responses_and_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, index = self.fixture(root)
            out = root / "refresh"
            original = (data / "population_responses.csv").read_bytes()
            audit = prepare(root, out)
            self.assertEqual(audit["changed_regime_rows"], 1)
            self.assertEqual(audit["mismatched_population_rows"], 2)
            (out / "partials").mkdir()
            for axon in ["a", "b"]:
                row = index[index.thresh_type == "I_all"].assign(target_axon_id=axon, n_spikes=2, spike_times_ms="[101.0, 105.0]")
                row.to_csv(out / "partials" / f"{axon}.csv", index=False)
            result = merge(root, out)
            self.assertEqual(result["replaced_rows"], 2)
            self.assertEqual(original, (data / "population_responses.csv").read_bytes())
            staged = pd.read_csv(out / "population_responses.csv")
            self.assertTrue(staged[staged.thresh_type == "I_all"].amp_uA.eq(300).all())
            self.assertTrue(staged[staged.thresh_type == "I_first"].spike_times_ms.eq("[101.0]").all())
            self.assertTrue(staged[staged.thresh_type == "I_all"].n_index_spikes.eq(39).all())
            summary = pd.read_csv(out / "true_population_trial_summary.csv").set_index("thresh_type")
            self.assertEqual(summary.loc["I_all", "total_population_spikes"], 4)
            self.assertEqual(summary.loc["I_all", "active_axon_count"], 2)
            cohort = pd.read_csv(out / "population_recruitment_outcomes.csv").set_index("thresh_type")
            self.assertEqual(cohort.loc["I_first", "expected_total_spikes"], 2)
            self.assertEqual(cohort.loc["I_all", "axons_always_recruited"], 2)

    def test_explicit_promotion_retains_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, index = self.fixture(root)
            out = root / "refresh"
            original = (data / "population_responses.csv").read_bytes()
            prepare(root, out)
            (out / "partials").mkdir()
            for axon in ["a", "b"]:
                index[index.thresh_type == "I_all"].assign(target_axon_id=axon, n_spikes=1,
                    spike_times_ms="[101.0]").to_csv(out / "partials" / f"{axon}.csv", index=False)
            result = merge(root, out, promote=True)
            self.assertEqual(result["status"], "promoted")
            self.assertEqual(original, (out / "population_responses_pre_refresh.csv").read_bytes())
            self.assertNotEqual(original, (data / "population_responses.csv").read_bytes())

    def test_rejects_stale_partial_and_count_mismatch(self):
        regimes = pd.DataFrame([dict(freq=20, cv=1, rep_index=0, thresh_type="I_all", amp_uA=300)])
        frame = regimes.assign(target_axon_id="a", detection_protocol_version=DETECTION_PROTOCOL_VERSION,
                               n_spikes=2, spike_times_ms="[101.0]")
        with self.assertRaisesRegex(ValueError, "spike"):
            validate_replacements(frame, regimes, "a")
        frame["n_spikes"] = 1
        frame["amp_uA"] = 1500
        with self.assertRaisesRegex(ValueError, "amplitudes"):
            validate_replacements(frame, regimes, "a")

    def test_rejects_duplicate_and_non_imax_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, _ = self.fixture(root)
            source = pd.read_csv(data / "population_responses.csv")
            source.loc[0, "amp_uA"] = 251
            source.to_csv(data / "population_responses.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Non-I_max"):
                prepare(root, root / "refresh")


if __name__ == "__main__":
    unittest.main()
