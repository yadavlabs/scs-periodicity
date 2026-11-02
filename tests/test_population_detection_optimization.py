import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from population.detection.detection_core import (
    DETECTION_PROTOCOL_VERSION,
    POPULATION_THRESHOLD_TOL_UA,
    POPULATION_RESPONSE_COLUMNS,
    _diameter_screen_failures,
    _equivalent_regime_repetition_groups,
    _evaluate_regime_amplitudes,
    _find_cached_plateau_threshold,
    _find_cached_peak_first_threshold,
    _population_basis_paths,
    _population_partial_path,
    _refine_cached_targets,
    _repetition_equivalence_groups,
    _replicate_rows_for_repetitions,
    _stable_audit_sample,
    _select_peak_first_imax_coarse,
    merge_population_recruitment_partials,
)


class FakeRunner:
    def __init__(self, counts):
        self.counts = counts
        self.calls = []

    def run_mrg_for_amp(self, amplitude):
        amplitude = float(amplitude)
        self.calls.append(amplitude)
        count = int(self.counts[amplitude])
        return np.arange(count, dtype=float), count


class PopulationDetectionOptimizationTests(unittest.TestCase):
    def test_peak_first_imax_prefers_clear_initial_peak(self):
        amplitudes = np.arange(300.0, 825.0, 25.0)
        counts = np.array([
            200, 280, 315, 320, 319, 318, 318, 318, 318, 318, 318,
            318, 318, 318, 318, 318, 325, 330, 330, 330, 330,
        ])
        selected = _select_peak_first_imax_coarse(
            amplitudes, counts, n_input_pulses=400,
        )
        self.assertEqual(selected["selection_method"], "early_response_peak")
        self.assertEqual(selected["coarse_selected_amplitude_uA"], 375.0)
        self.assertEqual(selected["target_spikes"], 320)

    def test_peak_first_imax_plateau_onset_must_attain_representative_count(self):
        amplitudes = np.arange(300.0, 825.0, 25.0)
        counts = np.array([
            200, 300, 380, 391, 394, 394, 394, 393, 394, 394, 393,
            394, 394, 393, 394, 394, 393, 394, 394, 393, 394,
        ])
        selected = _select_peak_first_imax_coarse(
            amplitudes, counts, n_input_pulses=400,
        )
        self.assertEqual(selected["selection_method"], "stable_saturation_plateau")
        self.assertEqual(selected["target_spikes"], 394)
        self.assertEqual(selected["coarse_selected_amplitude_uA"], 400.0)
        self.assertGreaterEqual(selected["coarse_selected_spikes"], selected["target_spikes"])

    def test_peak_first_imax_plateau_uses_first_attainment_before_brief_dip(self):
        amplitudes = np.arange(300.0, 825.0, 25.0)
        counts = np.array([
            107, 153, 172, 175, 176, 176, 176, 175, 175, 176, 176,
            176, 176, 176, 176, 176, 176, 176, 176, 176, 176,
        ])
        selected = _select_peak_first_imax_coarse(
            amplitudes, counts, n_input_pulses=200,
        )
        self.assertEqual(selected["selection_method"], "stable_saturation_plateau")
        self.assertEqual(selected["target_spikes"], 176)
        self.assertEqual(selected["coarse_selected_amplitude_uA"], 400.0)

    def test_cached_peak_first_refines_local_rising_edge_without_endpoint(self):
        calls = []
        cache = {}

        def counter(amplitude):
            amplitude = float(amplitude)
            calls.append(amplitude)
            if amplitude not in cache:
                if amplitude < 368.0:
                    count = int(200 + max(0.0, amplitude - 250.0))
                elif amplitude <= 375.0:
                    count = 320
                else:
                    count = 318
                cache[amplitude] = (np.arange(count, dtype=float), count)
            return cache[amplitude][1]

        threshold, count, metadata = _find_cached_peak_first_threshold(
            counter, cache, scan_start_uA=250.0, n_input_pulses=400,
            tol_uA=0.01, search_width_uA=500.0, safety_bound_uA=2000.0,
        )
        self.assertEqual(metadata["selection_method"], "early_response_peak")
        self.assertGreaterEqual(threshold, 368.0)
        self.assertLessEqual(threshold, 368.01)
        self.assertEqual(count, 320)
        self.assertNotIn(2000.0, calls)

    def test_peak_first_uses_i50_when_it_already_attains_first_plateau(self):
        cache = {}

        def counter(amplitude):
            amplitude = float(amplitude)
            count = 20 if amplitude < 263.4 else 40
            cache.setdefault(amplitude, (np.arange(count, dtype=float), count))
            return cache[amplitude][1]

        threshold, count, metadata = _find_cached_peak_first_threshold(
            counter, cache, scan_start_uA=263.4, n_input_pulses=40,
            tol_uA=0.01, search_width_uA=500.0, safety_bound_uA=2000.0,
        )
        self.assertEqual(metadata["selection_method"], "stable_saturation_plateau")
        self.assertEqual(threshold, 263.4)
        self.assertEqual(count, 40)
        self.assertFalse(metadata["refinement_performed"])

    def test_plateau_search_selects_first_stable_near_maximum(self):
        calls = []
        cache = {250.0: (np.arange(1, dtype=float), 1)}

        def counter(amplitude):
            amplitude = float(amplitude)
            calls.append(amplitude)
            if amplitude not in cache:
                if amplitude < 300.0:
                    count = 1
                elif amplitude < 1500.0:
                    count = 38
                else:
                    count = 39
                cache[amplitude] = (np.arange(count, dtype=float), count)
            return cache[amplitude][1]

        threshold, maximum = _find_cached_plateau_threshold(
            counter, cache, 250.0, scan_start_uA=250.0,
            amp_max_uA=2000.0, tol_uA=0.01, coarse_step_uA=25.0,
            plateau_width_uA=100.0, min_sampled_max_fraction=0.95,
        )
        self.assertIn(2000.0, calls)
        self.assertEqual(maximum, 38)
        self.assertGreaterEqual(threshold, 300.0)
        self.assertLessEqual(threshold, 300.01)

    def test_plateau_search_uses_sampled_maximum_not_only_endpoint(self):
        cache = {250.0: (np.arange(1, dtype=float), 1)}

        def counter(amplitude):
            amplitude = float(amplitude)
            if amplitude not in cache:
                if 600.0 <= amplitude < 1000.0:
                    count = 100
                elif amplitude >= 300.0:
                    count = 96
                else:
                    count = 1
                cache[amplitude] = (np.arange(count, dtype=float), count)
            return cache[amplitude][1]

        threshold, maximum = _find_cached_plateau_threshold(
            counter, cache, 250.0, scan_start_uA=250.0,
            amp_max_uA=2000.0, tol_uA=0.01, coarse_step_uA=25.0,
            plateau_width_uA=100.0, min_sampled_max_fraction=0.95,
        )
        self.assertEqual(maximum, 96)
        self.assertGreaterEqual(threshold, 300.0)
        self.assertLessEqual(threshold, 300.01)

    def test_joint_target_refinement_shares_cache_and_tolerance(self):
        cache = {
            1.0: (np.empty(0), 1),
            9.0: (np.empty(0), 9),
        }
        calls = []

        def counter(amplitude):
            amplitude = float(amplitude)
            if amplitude not in cache:
                calls.append(amplitude)
                count = int(np.floor(amplitude))
                cache[amplitude] = (np.arange(count, dtype=float), count)
            return cache[amplitude][1]

        refined = _refine_cached_targets(
            counter, cache, [3, 3, 5, 7], lower_bound=1.0, tol_uA=0.1,
        )
        for target in (3, 5, 7):
            amplitude, count = refined[target]
            self.assertGreaterEqual(amplitude, float(target))
            self.assertLessEqual(amplitude, float(target) + 0.1)
            self.assertGreaterEqual(count, target)
        self.assertEqual(len(calls), len(set(calls)))

    def test_exact_waveform_groups_are_replicated_with_original_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waveforms.npz"
            np.savez_compressed(
                path,
                current=np.asarray([[0, 1, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float32),
                rep_labels=np.asarray(["rep001", "rep002", "rep003"]),
            )
            groups = _repetition_equivalence_groups(path)
            self.assertEqual(groups, [[(0, "rep001"), (1, "rep002")], [(2, "rep003")]])
            rows = [{"rep_index": 0, "rep_label": "rep001", "amp_uA": 3.0}]
            replicated = _replicate_rows_for_repetitions(rows, groups[0])
            self.assertEqual([row["rep_index"] for row in replicated], [0, 1])
            self.assertEqual([row["rep_label"] for row in replicated], ["rep001", "rep002"])

    def test_cv0_legacy_labels_share_one_canonical_periodic_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S01_140um_ph1_1uA_cv0.0_f50_SpaceTimeV.npz"
            np.savez_compressed(
                path,
                current=np.asarray(
                    [[0, -1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32,
                ),
                rep_labels=np.asarray(
                    ["rep001_slack", "rep002_regular", "rep003_regular"],
                ),
            )
            self.assertEqual(
                _repetition_equivalence_groups(path),
                [[
                    (0, "rep001_slack"),
                    (1, "rep002_regular"),
                    (2, "rep003_regular"),
                ]],
            )

    def test_population_protocol_uses_fine_threshold_tolerance(self):
        self.assertEqual(POPULATION_THRESHOLD_TOL_UA, 0.01)
        self.assertIn("canonical-cv0", DETECTION_PROTOCOL_VERSION)
        self.assertIn("tol0p01", DETECTION_PROTOCOL_VERSION)

    def test_response_deduplication_requires_matching_regime_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "waveforms.npz"
            np.savez_compressed(
                path,
                current=np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32),
                rep_labels=np.asarray(["rep001", "rep002", "rep003"]),
            )
            base = {
                "index_axon_id": "S01_140um", "index_cutline_id": "S01",
                "index_diameter_um": 14.0, "file": "x.npz", "cv": 0.0,
                "freq": 10.0, "n_input_pulses": 20, "thresh_type": "I_first",
                "n_index_spikes": 1, "index_spike_times_ms": "[]", "N_max_spikes": 20,
            }
            regimes = {
                0: [{**base, "rep_index": 0, "rep_label": "rep001", "amp_uA": 3.0}],
                1: [{**base, "rep_index": 1, "rep_label": "rep002", "amp_uA": 3.0}],
                2: [{**base, "rep_index": 2, "rep_label": "rep003", "amp_uA": 4.0}],
            }
            groups = _equivalent_regime_repetition_groups(path, regimes)
            self.assertEqual(groups, [[(0, "rep001"), (1, "rep002")], [(2, "rep003")]])

    def test_exact_duplicate_amplitudes_run_once_and_highest_first(self):
        runner = FakeRunner({3.0: 1, 2.0: 2, 1.0: 1})
        rows = [{"amp_uA": value} for value in (1.0, 2.0, 2.0, 3.0)]
        results, stats = _evaluate_regime_amplitudes(
            runner, rows, prune_on_max_failure=True,
            audit_max_failure=False, expected_max_failure=False,
        )
        self.assertEqual(runner.calls, [3.0, 2.0, 1.0])
        self.assertEqual(results[2.0][1], 2)
        self.assertEqual(stats["duplicate_runs_saved"], 1)

    def test_maximum_failure_prunes_lower_amplitudes(self):
        runner = FakeRunner({3.0: 0, 2.0: 1, 1.0: 1})
        rows = [{"amp_uA": value} for value in (1.0, 2.0, 3.0)]
        results, stats = _evaluate_regime_amplitudes(
            runner, rows, prune_on_max_failure=True,
            audit_max_failure=False, expected_max_failure=False,
        )
        self.assertEqual(runner.calls, [3.0])
        self.assertTrue(all(result[1] == 0 for result in results.values()))
        self.assertEqual(stats["max_failure_pruned"], 1)

    def test_maximum_failure_audit_detects_violation(self):
        runner = FakeRunner({3.0: 0, 2.0: 1})
        with self.assertRaisesRegex(RuntimeError, "Maximum-amplitude pruning audit failed"):
            _evaluate_regime_amplitudes(
                runner, [{"amp_uA": 2.0}, {"amp_uA": 3.0}],
                prune_on_max_failure=True, audit_max_failure=True,
                expected_max_failure=False,
            )

    def test_diameter_audit_detects_smaller_diameter_response(self):
        runner = FakeRunner({3.0: 1})
        with self.assertRaisesRegex(RuntimeError, "Diameter-pruning audit failed"):
            _evaluate_regime_amplitudes(
                runner, [{"amp_uA": 3.0}], prune_on_max_failure=True,
                audit_max_failure=False, expected_max_failure=True,
            )

    def test_screen_uses_each_repetitions_largest_amplitude(self):
        table = pd.DataFrame([
            {"rep_index": 0, "amp_uA": 1.0, "n_spikes": 0},
            {"rep_index": 0, "amp_uA": 3.0, "n_spikes": 1},
            {"rep_index": 1, "amp_uA": 2.0, "n_spikes": 0},
            {"rep_index": 1, "amp_uA": 4.0, "n_spikes": 0},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.csv"
            table.to_csv(path, index=False)
            self.assertEqual(_diameter_screen_failures(path), {0: False, 1: True})

    def test_audit_sampling_is_deterministic_and_validates_rate(self):
        first = _stable_audit_sample(0.5, "axon", 12)
        self.assertEqual(first, _stable_audit_sample(0.5, "axon", 12))
        self.assertFalse(_stable_audit_sample(0.0, "axon", 12))
        self.assertTrue(_stable_audit_sample(1.0, "axon", 12))
        with self.assertRaises(ValueError):
            _stable_audit_sample(1.1, "axon", 12)

    def test_population_manifest_selects_cohort_without_deleting_old_bases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            basis_dir = root / "data" / "space-time-V" / "population"
            basis_dir.mkdir(parents=True)
            for axon_id in ("C01_57um", "C02_57um", "C03_57um"):
                (basis_dir / f"{axon_id}_1uA_VeBasis.npz").touch()
            manifest = root / "cohort.csv"
            pd.DataFrame([
                {"axon_id": "C03_57um", "include": True},
                {"axon_id": "C01_57um", "include": False},
                {"axon_id": "C02_57um", "include": True},
            ]).to_csv(manifest, index=False)
            selected = _population_basis_paths(root, manifest)
            self.assertEqual([path.name for path in selected], [
                "C03_57um_1uA_VeBasis.npz", "C02_57um_1uA_VeBasis.npz",
            ])

    def test_cv0_merge_propagates_realized_index_count_to_every_axon_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial_dir = root / "partials"
            partial_dir.mkdir(parents=True)
            jobs = [
                {"target_axon_id": "index", "cv": 0.3, "freq": 200.0},
                {"target_axon_id": "other", "cv": 0.3, "freq": 200.0},
            ]
            base = {
                "detection_protocol_version": DETECTION_PROTOCOL_VERSION,
                "index_axon_id": "index", "index_cutline_id": "S01",
                "index_diameter_um": 14.0, "target_cutline_id": "S01",
                "target_diameter_um": 14.0, "target_n_valid_nodes": 51,
                "target_readout_node_number": 25, "file": "x.npz",
                "cv": 0.3, "freq": 200.0, "rep_index": 0,
                "rep_label": "rep001", "n_input_pulses": 400,
                "thresh_type": "I_volley5", "amp_uA": 500.0,
                "spike_times_ms": "[]",
            }
            for job, n_spikes in zip(jobs, (6, 0)):
                path = _population_partial_path(partial_dir, "cv0", job)
                row = {
                    **base,
                    "target_axon_id": job["target_axon_id"],
                    "n_index_spikes": 6 if job["target_axon_id"] == "index" else np.nan,
                    "n_spikes": n_spikes,
                }
                pd.DataFrame([row], columns=POPULATION_RESPONSE_COLUMNS).to_csv(path, index=False)
            output = root / "merged.csv"
            with patch(
                "population.detection.detection_core.population_recruitment_jobs",
                return_value=jobs,
            ), patch(
                "population.detection.detection_core.identify_index_axon",
                return_value={"axon_id": "index"},
            ):
                merge_population_recruitment_partials(
                    root=root, mode="cv0", partial_dir=partial_dir, output_csv=output,
                )
            merged = pd.read_csv(output)
            self.assertEqual(merged["n_index_spikes"].tolist(), [6, 6])


if __name__ == "__main__":
    unittest.main()
