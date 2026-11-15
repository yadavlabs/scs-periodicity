"""Presentation tables derived from the fitted population observer."""

import pandas as pd


MODEL_NAMES = {
    "primary_hybrid": "Per-frequency hybrid",
    "crossed_hybrid": "Per-block hybrid",
    "per_panel_hybrid": "Per-panel hybrid",
    "primary_rate": "Per-frequency rate-only",
    "primary_timing": "Per-frequency timing-only",
}
COLUMNS = [
    "model", "parameters", "simulation_test_RMSE", "std0_frequency_mean_RMSE",
    "std1_frequency_mean_RMSE", "delta_AIC", "delta_BIC", "std0_rate_share",
    "std1_rate_share",
]


def consolidated_table(figures):
    metrics = figures.comparisons()
    frame = metrics[COLUMNS].copy()
    frame["model"] = frame.model.map(MODEL_NAMES)
    if frame.model.isna().any() or len(frame) != 5:
        raise ValueError("Expected the five independent/scope models")
    figures.table(frame, "table1_consolidated")
    frame.columns = pd.MultiIndex.from_tuples([
        ("Model", ""), ("Complexity", "k"), ("RMSE", "Overall"), ("RMSE", "Std0"),
        ("RMSE", "Std1"), ("Criterion scores", "ΔAIC*"),
        ("Criterion scores", "ΔBIC*"), ("Rate reliance", "Std0"),
        ("Rate reliance", "Std1"),
    ])
    return frame.style.hide(axis="index").format({
        ("Complexity", "k"): "{:.0f}",
        **{("RMSE", name): "{:.4f}" for name in ("Overall", "Std0", "Std1")},
        **{("Criterion scores", name): "{:.1f}" for name in ("ΔAIC*", "ΔBIC*")},
        **{("Rate reliance", name): "{:.3f}" for name in ("Std0", "Std1")},
    })
