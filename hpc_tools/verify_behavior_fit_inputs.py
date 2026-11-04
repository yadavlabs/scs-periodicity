"""Verify the archived pre-Imax inputs used for the accepted detection fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/detection/behavior_fit_inputs"
KEY = ["freq", "cv", "rep_index", "thresh_type"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((FOLDER / "provenance.json").read_text(encoding="utf-8"))
    for name, expected in manifest["inputs_sha256"].items():
        if digest(FOLDER / name) != expected:
            raise ValueError(f"Historical fit input hash mismatch: {name}")
    earlier = pd.read_csv(FOLDER / "index_regime_amplitudes_pre_imax.csv")
    audit = pd.read_csv(FOLDER / "peak_first_imax_updates.csv",
                        usecols=KEY + ["old_amp_uA", "old_n_index_spikes"])
    if len(earlier) != 49_500 or len(audit) != 5_500:
        raise ValueError("Incomplete historical regime inputs")
    joined = earlier.merge(audit, on=KEY, how="left", validate="one_to_one")
    if len(joined) != 49_500:
        raise ValueError("Historical regime key mismatch")
    imax = joined.thresh_type.eq("I_all")
    if (imax.sum() != 5_500 or not np.allclose(joined.loc[imax, "amp_uA"],
            joined.loc[imax, "old_amp_uA"], rtol=0, atol=1e-9)
            or not np.array_equal(joined.loc[imax, "n_index_spikes"],
                                  joined.loc[imax, "old_n_index_spikes"])):
        raise ValueError("Pre-Imax table does not match the retained update audit")
    current_path = ROOT / "data/detection/population/index_regime_amplitudes.csv"
    if current_path.is_file():
        current = pd.read_csv(current_path, usecols=KEY + ["amp_uA", "n_index_spikes"])
        if len(current) != 49_500:
            raise ValueError("Incomplete current index regimes")
        comparison = current.merge(earlier, on=KEY, validate="one_to_one",
                                   suffixes=("_current", "_old"))
        other = comparison.thresh_type.ne("I_all")
        if (len(comparison) != 49_500 or not np.allclose(
                comparison.loc[other, "amp_uA_current"],
                comparison.loc[other, "amp_uA_old"], rtol=0, atol=0.01)
                or not np.array_equal(comparison.loc[other, "n_index_spikes_current"],
                                      comparison.loc[other, "n_index_spikes_old"])):
            raise ValueError("Non-Imax regimes changed between accepted and current analyses")
    outcomes = pd.read_csv(FOLDER / "population_recruitment_outcomes_pre_imax.csv", dtype={"mode": str})
    if len(outcomes) != 990 or outcomes.duplicated(["mode", "freq", "cv", "thresh_type"]).any():
        raise ValueError("Incomplete historical recruitment summary")
    print("Verified 49,500 historical regimes, 5,500 pre-Imax rows, and 990 recruitment summaries")


if __name__ == "__main__":
    main()
