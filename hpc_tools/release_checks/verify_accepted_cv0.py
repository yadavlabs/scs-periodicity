"""Compare regenerated CV0 features and traces with the accepted archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FILES = (
    "population_cv0_based_features.csv",
    "population_cv0_template_evidence.csv",
    "population_collapsed_regime_diagnostic.csv",
    "population_regime_response_equivalence.csv",
)
TRACE = "population_cv0_time_representations_v5.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="Directory containing the regenerated CV0 feature files")
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/features")
    args = parser.parse_args()
    failures = []
    for name in FILES:
        expected = pd.read_csv(args.reference / name, low_memory=False)
        observed = pd.read_csv(args.candidate_dir / name, low_memory=False)
        try:
            pd.testing.assert_frame_equal(
                expected, observed, check_dtype=False, check_exact=False,
                rtol=1e-10, atol=1e-10,
            )
        except AssertionError as error:
            failures.append(f"{name}: {str(error).splitlines()[0]}")
    with np.load(args.reference / TRACE, allow_pickle=False) as expected, np.load(
        args.candidate_dir / TRACE, allow_pickle=False
    ) as observed:
        if expected.files != observed.files:
            failures.append(f"{TRACE}: array names differ")
        else:
            for name in expected.files:
                try:
                    np.testing.assert_array_equal(expected[name], observed[name])
                except AssertionError as error:
                    failures.append(f"{TRACE}/{name}: {str(error).splitlines()[0]}")
    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        raise SystemExit(f"{len(failures)} accepted CV0 comparisons differ")
    print(f"Matched all {len(FILES)} accepted CV0 tables and the full trace archive")


if __name__ == "__main__":
    main()
