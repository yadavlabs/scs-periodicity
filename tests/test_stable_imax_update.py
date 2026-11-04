import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from detection.detection_core import REGIME_COLUMNS
from detection.stable_imax_update import (
    UPDATE_COLUMNS,
    UPDATE_VERSION,
    _partial_path,
    merge,
)
from protocol import DETECTION_PROTOCOL_VERSION


class StableImaxUpdateTests(unittest.TestCase):
    def test_merge_replaces_only_imax_values_and_shared_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "index_regime_amplitudes.csv"
            base = {
                "detection_protocol_version": "old_protocol",
                "index_axon_id": "index", "index_cutline_id": "S01",
                "index_diameter_um": 14.0, "file": "waveform.npz",
                "cv": 0.0, "freq": 20.0, "rep_index": 0,
                "rep_label": "rep001", "n_input_pulses": 40,
                "index_spike_times_ms": "[1.0]", "N_max_spikes": 40,
            }
            source = pd.DataFrame([
                {**base, "thresh_type": "I_first", "amp_uA": 250.0, "n_index_spikes": 1},
                {**base, "thresh_type": "I_all", "amp_uA": 1500.0, "n_index_spikes": 40},
            ], columns=REGIME_COLUMNS)
            source.to_csv(source_path, index=False)

            partial_dir = root / "partials"
            partial_dir.mkdir()
            update = {
                "update_version": UPDATE_VERSION,
                "detection_protocol_version": DETECTION_PROTOCOL_VERSION,
                "index_axon_id": "index", "index_cutline_id": "S01",
                "index_diameter_um": 14.0, "file": "waveform.npz",
                "cv": 0.0, "freq": 20.0, "rep_index": 0,
                "rep_label": "rep001", "n_input_pulses": 40,
                "thresh_type": "I_all", "old_amp_uA": 1500.0,
                "old_n_index_spikes": 40, "old_N_max_spikes": 40,
                "I_first_uA": 250.0, "I_50_uA": 275.0,
                "amp_uA": 300.0, "n_index_spikes": 39,
                "index_spike_times_ms": "[2.0]", "N_max_spikes": 39,
                "selection_method": "stable_saturation_plateau", "target_spikes": 39,
                "coarse_selected_amp_uA": 300.0, "coarse_selected_spikes": 39,
                "early_peak_detected": False, "early_peak_amp_uA": 300.0,
                "early_peak_spikes": 39, "early_peak_rise_spikes": 19,
                "early_peak_terminal_drop_spikes": 0.0,
                "plateau_tolerance_fraction": 0.0, "plateau_tolerance_spikes": 0,
                "plateau_start_uA": 300.0, "plateau_confirmed_through_uA": 500.0,
                "first_sustained_decline_uA": float("nan"),
                "grid_start_uA": 275.0, "grid_end_uA": 775.0,
                "coarse_grid_samples": 21, "refinement_performed": True,
                "sampled_amplitudes_uA": "[275.0, 300.0]",
                "sampled_spike_counts": "[20, 39]", "neuron_runs": 21,
            }
            pd.DataFrame([update], columns=UPDATE_COLUMNS).to_csv(
                _partial_path(partial_dir, 0.0, 20.0), index=False,
            )
            output_dir = root / "output"
            with patch(
                "detection.stable_imax_update.index_regime_conditions",
                return_value=[(0.0, 20.0)],
            ):
                outputs = merge(
                    root=root, source_csv=source_path,
                    partial_dir=partial_dir, output_dir=output_dir,
                )
            result = pd.read_csv(outputs["staged_index_regimes"])
            fixed = result[result["thresh_type"].eq("I_first")].iloc[0]
            imax = result[result["thresh_type"].eq("I_all")].iloc[0]
            self.assertEqual(fixed["amp_uA"], 250.0)
            self.assertEqual(fixed["n_index_spikes"], 1)
            self.assertEqual(fixed["index_spike_times_ms"], "[1.0]")
            self.assertEqual(fixed["N_max_spikes"], 39)
            self.assertEqual(imax["amp_uA"], 300.0)
            self.assertEqual(imax["n_index_spikes"], 39)
            self.assertEqual(imax["index_spike_times_ms"], "[2.0]")
            self.assertTrue(result["detection_protocol_version"].eq(DETECTION_PROTOCOL_VERSION).all())


if __name__ == "__main__":
    unittest.main()
