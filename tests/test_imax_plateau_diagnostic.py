import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

import detection.imax_plateau_diagnostic as diagnostic
from detection.imax_plateau_diagnostic import (
    fine_transition_steps,
    first_stable_plateau,
    plateau_segments,
    pulse_response_summary,
    select_conditions,
    transition_brackets,
)


def _condition_table():
    rows = []
    for frequency in (20.0, 50.0):
        for rep_index, amplitude, target in [
            (0, 300.0, 39), (1, 305.0, 40), (2, 310.0, 38),
            (3, 1400.0, 39), (4, 1600.0, 40), (5, 1900.0, 41),
        ]:
            rows.append({
                "freq": frequency, "cv": 1.0, "rep_index": rep_index,
                "rep_label": f"rep{rep_index:03d}", "thresh_type": "I_all",
                "amp_uA": amplitude, "N_max_spikes": target,
                "n_index_spikes": target, "index_axon_id": "axon",
                "index_cutline_id": "S01", "index_diameter_um": 14.0,
                "file": "input.npz", "n_input_pulses": int(2 * frequency),
            })
    return pd.DataFrame(rows)


def test_condition_selection_spans_high_branch_and_matches_ordinary_controls():
    selected = select_conditions(
        _condition_table(), frequencies=(20.0, 50.0), cv=1.0,
        branch_cutoff_uA=800.0, repetitions_per_branch=2,
    )
    assert len(selected) == 8
    counts = selected.groupby(["freq", "branch"]).size()
    assert set(counts.index) == {(20.0, "high"), (20.0, "ordinary"), (50.0, "high"), (50.0, "ordinary")}
    assert (counts == 2).all()
    for frequency in (20.0, 50.0):
        high = selected[(selected["freq"] == frequency) & (selected["branch"] == "high")]
        assert set(high["production_Imax_uA"]) == {1400.0, 1900.0}
        ordinary = selected[(selected["freq"] == frequency) & (selected["branch"] == "ordinary")]
        target_distance = [
            np.min(np.abs(high["production_target_spikes"].to_numpy() - target))
            for target in ordinary["production_target_spikes"]
        ]
        assert max(target_distance) <= 1


def test_plateau_and_late_transition_detection():
    amplitudes = np.arange(200.0, 2000.1, 25.0)
    counts = np.select(
        [amplitudes < 300.0, amplitudes < 1500.0],
        [0, 99], default=100,
    )
    samples = pd.DataFrame({
        "amp_uA": amplitudes,
        "n_index_spikes": counts,
        "sample_role": "coarse",
    })
    segments = plateau_segments(samples, coarse_step_uA=25.0)
    stable = first_stable_plateau(
        segments, min_width_uA=100.0, min_fraction_of_max=0.90,
    )
    assert stable is not None
    assert stable["start_uA"] == 300.0
    assert stable["n_index_spikes"] == 99
    transitions = transition_brackets(samples, late_min_uA=500.0)
    late = transitions[transitions["is_late_transition"]]
    assert len(late) == 1
    assert late.iloc[0]["amp_high_uA"] == 1500.0
    assert late.iloc[0]["spike_change"] == 1

    fine = pd.DataFrame({
        "amp_uA": [300.0, 300.5, 301.0, 1499.0, 1499.5, 1500.0],
        "n_index_spikes": [99, 99, 99, 99, 99, 100],
        "sample_role": "fine",
    })
    fine_changes = fine_transition_steps(fine, fine_step_uA=0.5, late_min_uA=500.0)
    assert len(fine_changes) == 1
    assert fine_changes.iloc[0]["amp_low_uA"] == 1499.5
    assert fine_changes.iloc[0]["amp_high_uA"] == 1500.0


def test_pulse_response_summary_distinguishes_missing_and_multiple_responses():
    summary = pulse_response_summary(
        [100.5, 200.5, 200.8, 400.5], [100.0, 200.0, 300.0, 400.0]
    )
    assert summary["n_pulses_with_response"] == 3
    assert summary["n_pulses_without_response"] == 1
    assert summary["n_pulses_with_multiple_spikes"] == 1
    assert summary["max_spikes_per_pulse"] == 2
    assert np.isclose(summary["min_interspike_interval_ms"], 0.3)


def test_run_condition_writes_complete_outputs_with_synthetic_runner():
    class FakeRunner:
        def __init__(self):
            self.t_rec = np.arange(0.0, 10.0, 0.1)
            self.v_rec = np.zeros_like(self.t_rec)

        def run_mrg_for_amp(self, amplitude):
            count = 0 if amplitude < 300 else (99 if amplitude < 1500 else 100)
            self.v_rec = np.linspace(-80.0, 20.0 if count else -60.0, len(self.t_rec))
            return np.arange(count, dtype=float) + 100.5, count

    fake_rep = {
        "time_ms": np.arange(0.0, 5.0, 0.1),
        "current": np.asarray([-1.0] + [0.0] * 49),
    }

    def fake_make_runner(*args, **kwargs):
        return fake_rep, FakeRunner()

    def fake_parallel(root, index, cv, frequency, rep_index, amplitudes, n_jobs):
        runner = FakeRunner()
        output = {}
        for amplitude in amplitudes:
            spikes, count = runner.run_mrg_for_amp(amplitude)
            output[float(amplitude)] = {
                "spike_times_ms": spikes,
                "n_index_spikes": count,
                "peak_voltage_mV": float(np.max(runner.v_rec)),
                "minimum_voltage_mV": float(np.min(runner.v_rec)),
            }
        return output

    with TemporaryDirectory() as directory:
        root = Path(directory)
        output = root / "diagnostic"
        manifest = output / "diagnostic_conditions.csv"
        output.mkdir(parents=True)
        pd.DataFrame([{
            "diagnostic_version": diagnostic.DIAGNOSTIC_VERSION,
            "condition_index": 0, "freq": 50.0, "cv": 1.0,
            "rep_index": 1, "rep_label": "rep002_gamma", "branch": "high",
            "production_Imax_uA": 1500.0, "production_target_spikes": 100,
        }]).to_csv(manifest, index=False)
        originals = (
            diagnostic.identify_index_axon,
            diagnostic._make_runner,
            diagnostic._run_parallel_amplitudes,
        )
        diagnostic.identify_index_axon = lambda **kwargs: {
            "cutline_id": "S01", "diameter_um": 14.0, "axon_id": "axon"
        }
        diagnostic._make_runner = fake_make_runner
        diagnostic._run_parallel_amplitudes = fake_parallel
        try:
            paths = diagnostic.run_condition(
                root=root, condition_index=0, manifest=manifest,
                partial_dir=output / "partials", coarse_min_uA=200,
                coarse_max_uA=2000, coarse_step_uA=100,
                fine_step_uA=5, fine_half_width_uA=20,
                late_transition_min_uA=500, plateau_min_width_uA=100,
                n_jobs=1,
            )
        finally:
            diagnostic.identify_index_axon, diagnostic._make_runner, diagnostic._run_parallel_amplitudes = originals
        assert all(path.is_file() for path in paths.values())
        summary = pd.read_csv(paths["summary"]).iloc[0]
        assert summary["sampled_max_spikes"] == 100
        assert summary["first_stable_plateau_count"] == 99
        assert summary["late_upward_transition_count"] == 1
        with np.load(paths["traces"], allow_pickle=False) as traces:
            assert 1500.0 in traces["amplitude_uA"]
            assert 2000.0 in traces["amplitude_uA"]
        merged = diagnostic.merge(
            root=root, manifest=manifest, partial_dir=output / "partials",
            output_dir=output,
        )
        assert all(path.is_file() for path in merged.values())
        assert (output / "imax_amplitude_response_sweeps.png").is_file()
