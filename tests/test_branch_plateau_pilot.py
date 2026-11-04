import numpy as np
import pandas as pd

from detection.branch_plateau_pilot import (
    classify_branch,
    coarse_grid,
    early_recruitment_peak,
    select_cross_frequency_conditions,
    select_frequency_cv_validation_conditions,
    validation_amplitude_grid,
)


def test_coarse_grid_starts_at_next_absolute_step_and_keeps_safety_bound():
    grid = coarse_grid(263.4, 401.0, 25.0)
    assert np.array_equal(grid, np.array([275.0, 300.0, 325.0, 350.0, 375.0, 400.0, 401.0]))


def test_first_sustained_predecline_plateau_is_selected():
    amplitudes = np.arange(250.0, 701.0, 25.0)
    counts = np.array([200, 240, 280, 310, 319, 320, 321, 320, 320, 319, 320, 320, 320, 300, 295, 294, 294, 294, 294])
    result = classify_branch(
        amplitudes,
        counts,
        n_input_pulses=400,
        tolerance_fraction=0.01,
        candidate_width_uA=100.0,
        confirmation_width_uA=100.0,
        decline_width_uA=50.0,
    )
    assert result["classification"] == "stable_predecline_plateau"
    assert result["plateau_start_uA"] == 350.0
    assert result["plateau_count"] == 320
    assert result["first_sustained_decline_uA"] == 575.0


def test_short_shoulder_then_decline_does_not_select_later_plateau():
    amplitudes = np.arange(250.0, 851.0, 25.0)
    counts = np.array([
        200, 250, 295, 318, 320, 320, 319, 308, 300, 298, 299, 300, 300,
        300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
    ])
    result = classify_branch(
        amplitudes,
        counts,
        n_input_pulses=400,
        tolerance_fraction=0.01,
        candidate_width_uA=100.0,
        confirmation_width_uA=100.0,
        decline_width_uA=50.0,
    )
    assert result["classification"] == "blocked_before_plateau"
    assert np.isnan(result["plateau_start_uA"])
    assert result["later_plateau_start_uA"] == 500.0
    assert result["later_plateau_count"] == 300


def test_endpoint_and_late_global_max_do_not_change_selected_plateau():
    amplitudes = np.arange(250.0, 1001.0, 25.0)
    counts = np.array([
        200, 250, 290, 315, 320, 320, 321, 320, 320, 320, 321, 320, 318,
        300, 298, 298, 298, 298, 298, 298, 298, 330, 331, 331, 331, 331,
        331, 331, 331, 331, 335,
    ])
    result = classify_branch(
        amplitudes,
        counts,
        n_input_pulses=400,
        tolerance_fraction=0.01,
        candidate_width_uA=100.0,
        confirmation_width_uA=100.0,
        decline_width_uA=50.0,
    )
    assert result["classification"] == "stable_predecline_plateau"
    assert result["plateau_start_uA"] == 350.0
    assert result["plateau_count"] == 320


def test_cross_frequency_selection_keeps_all_sparse_high_cases_and_matched_controls():
    rows = []
    for cv, rep, imax, count in (
        (0.8, 0, 1500.0, 34),
        (0.9, 1, 1600.0, 36),
        (1.0, 2, 1700.0, 32),
        (0.8, 3, 400.0, 33),
        (0.9, 4, 400.0, 35),
        (1.0, 5, 400.0, 32),
        (1.0, 6, 395.0, 30),
    ):
        common = {
            "cv": cv, "freq": 20.0, "rep_index": rep, "rep_label": f"rep{rep}",
            "n_input_pulses": 40, "file": "synthetic.npz", "index_axon_id": "a",
            "index_cutline_id": "c", "index_diameter_um": 10.0,
        }
        rows.extend([
            {**common, "thresh_type": "I_first", "amp_uA": 250.0,
             "n_index_spikes": 1, "N_max_spikes": count},
            {**common, "thresh_type": "I_50", "amp_uA": 300.0,
             "n_index_spikes": 20, "N_max_spikes": count},
            {**common, "thresh_type": "I_all", "amp_uA": imax,
             "n_index_spikes": count, "N_max_spikes": count},
        ])
    selected = select_cross_frequency_conditions(
        pd.DataFrame(rows), frequencies_hz=[20.0], high_branch_cutoff_uA=800.0,
        max_high_per_frequency=8, controls_per_frequency=2,
    )
    assert len(selected) == 5
    assert selected["historical_high_branch"].sum() == 3
    assert (selected["selection_group"] == "matched_ordinary_control").sum() == 2
    assert set(selected.loc[selected["historical_high_branch"], "rep_index"]) == {0, 1, 2}
    assert np.allclose(selected["primary_tolerance_fraction"], 0.02)


def test_early_recruitment_peak_accepts_initial_peak_and_rejects_flat_plateau():
    amplitudes = np.arange(300.0, 901.0, 25.0)
    peaked = np.array([100, 160, 180, 185, 184, 183, 182, 182, 181, 181, 181, 181, 181,
                       181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181, 181])
    flat = np.array([100, 160, 180, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185,
                     185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185, 185])
    peak = early_recruitment_peak(amplitudes, peaked, n_input_pulses=200)
    plateau = early_recruitment_peak(amplitudes, flat, n_input_pulses=200)
    assert peak["is_clear_peak"]
    assert peak["peak_amplitude_uA"] == 375.0
    assert peak["peak_count"] == 185
    assert not plateau["is_clear_peak"]


def test_validation_grid_separates_contiguous_search_from_late_sentinels():
    amplitudes, is_early = validation_amplitude_grid(
        263.4,
        step_uA=25.0,
        sweep_width_uA=100.0,
        sentinel_amplitudes_uA=[350.0, 500.0, 800.0],
        safety_bound_uA=700.0,
    )
    assert np.array_equal(amplitudes, np.array([275.0, 300.0, 325.0, 350.0, 375.0, 500.0]))
    assert np.array_equal(is_early, np.array([True, True, True, True, True, False]))


def test_frequency_cv_validation_selection_spans_old_imax_amplitudes():
    rows = []
    for frequency in (20.0, 50.0):
        for cv in (0.0, 0.5):
            for rep, imax in enumerate((350.0, 450.0, 550.0, 650.0, 1500.0)):
                common = {
                    "cv": cv, "freq": frequency, "rep_index": rep,
                    "rep_label": f"rep{rep}", "n_input_pulses": int(2 * frequency),
                    "file": "synthetic.npz", "index_axon_id": "a",
                    "index_cutline_id": "c", "index_diameter_um": 10.0,
                }
                rows.extend([
                    {**common, "thresh_type": "I_first", "amp_uA": 250.0,
                     "n_index_spikes": 1, "N_max_spikes": int(2 * frequency)},
                    {**common, "thresh_type": "I_50", "amp_uA": 275.0,
                     "n_index_spikes": int(frequency), "N_max_spikes": int(2 * frequency)},
                    {**common, "thresh_type": "I_all", "amp_uA": imax,
                     "n_index_spikes": int(2 * frequency), "N_max_spikes": int(2 * frequency)},
                ])
    selected = select_frequency_cv_validation_conditions(
        pd.DataFrame(rows), frequencies_hz=[20.0, 50.0], cvs=[0.0, 0.5],
        repetitions_cv0=1, repetitions_nonzero=3,
    )
    assert len(selected) == 8
    assert (selected[np.isclose(selected["cv"], 0.0)].groupby("freq").size() == 1).all()
    assert (selected[np.isclose(selected["cv"], 0.5)].groupby("freq").size() == 3).all()
    for frequency in (20.0, 50.0):
        panel = selected[np.isclose(selected["freq"], frequency) & np.isclose(selected["cv"], 0.5)]
        assert set(panel["old_Imax_uA"]) == {350.0, 550.0, 1500.0}
    assert selected["pilot_version"].eq("peak_first_imax_frequency_cv_validation_v1").all()
