"""Compare independently regenerated CV0 features with the accepted archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FILES = (
    "population_cv0_based_features.csv",
    "population_cv0_template_evidence.csv",
    "population_collapsed_regime_diagnostic.csv",
    "population_regime_response_equivalence.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="Downloaded logs/accepted-cv0-10hz directory")
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
    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        raise SystemExit(f"{len(failures)} of {len(FILES)} accepted CV0 tables differ")
    print(f"Matched all {len(FILES)} accepted CV0 feature/evidence tables")


if __name__ == "__main__":
    main()
