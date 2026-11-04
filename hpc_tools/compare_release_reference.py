"""Compare manuscript-facing rerun tables with the accepted data extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DETECTION_FIT = (
    "detection_behavior_points.csv",
    "detection_curves.csv",
    "detection_frozen_global_weights.csv",
    "detection_frozen_global_weights_datapoints.csv",
    "detection_frozen_global_weights_quadratic.csv",
    "detection_frozen_per_frequency_best_single.csv",
    "detection_frozen_per_frequency_best_single_datapoints.csv",
    "detection_frozen_per_frequency_best_single_quadratic.csv",
    "detection_metrics.csv",
    "detection_mixture_population_summary.csv",
    "detection_population_states.csv",
    "detection_weights.csv",
)
FIGURE_STATISTICS = (
    "availability.csv",
    "reconstruction_audit.csv",
    "reliance_estimand_comparison.csv",
    "reliance_repetition_auc.csv",
    "reliance_test_appearances.csv",
    "reliance_tests.csv",
    "rmse_block_observations.csv",
    "rmse_panel_observations.csv",
    "rmse_tests.csv",
)
TABLES = (
    *(f"results/detection/behavioral_matching/{name}" for name in DETECTION_FIT),
    "results/detection/true_population_trial_summary.csv",
    "results/discrimination/control/population_observer_rmse.csv",
    "results/discrimination/scope/population_observer_rmse.csv",
    "results/discrimination/locked/followup/performance_summary.csv",
    *(f"results/figures/statistics/{name}" for name in FIGURE_STATISTICS),
    "results/figures/table1_consolidated.csv",
)
LEGACY_ROOTS = {
    "results/discrimination/control/": "results/discrimination/",
    "results/discrimination/scope/": "results/discrimination_v5_scope_hypothesis_full/",
    "results/discrimination/locked/followup/": "results/discrimination_locked_mixture_followup/",
}


def reference_path(root: Path, relative: str) -> Path:
    current = root / relative
    if current.is_file():
        return current
    for new, old in LEGACY_ROOTS.items():
        if relative.startswith(new):
            return root / (old + relative[len(new):])
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Source and accepted-data ZIP extraction directory")
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Clean source extraction with downloaded results and rerun figures")
    args = parser.parse_args()
    reference = args.reference.resolve()
    candidate = args.candidate.resolve()
    failures = []
    for relative in TABLES:
        left = reference_path(reference, relative)
        right = candidate / relative
        if not left.is_file() or not right.is_file():
            failures.append(f"{relative}: missing in {'reference' if not left.is_file() else 'candidate'}")
            continue
        try:
            pd.testing.assert_frame_equal(
                pd.read_csv(left), pd.read_csv(right), check_dtype=False,
                check_exact=False, rtol=1e-10, atol=1e-10,
            )
        except AssertionError as error:
            failures.append(f"{relative}: {str(error).splitlines()[0]}")
    for failure in failures:
        print(f"MISMATCH {failure}")
    if failures:
        raise SystemExit(f"{len(failures)} of {len(TABLES)} manuscript-facing tables differ")
    print(f"Matched {len(TABLES)} manuscript-facing tables against the accepted release data")


if __name__ == "__main__":
    main()
