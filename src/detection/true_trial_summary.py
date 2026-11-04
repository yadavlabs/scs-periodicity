"""Summarize the completed true-CV population response table for manuscript figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


KEY = ["freq", "cv", "rep_index", "thresh_type"]
SOURCE = Path("data/detection/population_responses.csv")
OUTPUT = Path("results/detection/true_population_trial_summary.csv")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(root: Path) -> dict[str, object]:
    source = root / SOURCE
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = None
    counts = None
    source_rows = 0
    for frame in pd.read_csv(source, usecols=KEY + ["n_spikes"], chunksize=200_000):
        source_rows += len(frame)
        frame["active_axon_count"] = frame.n_spikes.gt(0).astype(int)
        grouped = frame.groupby(KEY, sort=False)[["n_spikes", "active_axon_count"]].agg(["sum", "count"])
        totals = grouped if totals is None else totals.add(grouped, fill_value=0)
    if totals is None:
        raise ValueError("Empty true-CV population response table")
    counts = totals[("n_spikes", "count")]
    if source_rows != 20_790_000 or len(totals) != 49_500 or not counts.eq(420).all():
        raise ValueError(f"Incomplete population response table: {source_rows} rows, {len(totals)} trials")
    summary = totals[[ ("n_spikes", "sum"), ("active_axon_count", "sum") ]].astype("int64")
    summary.columns = ["total_population_spikes", "active_axon_count"]
    summary.sort_index().reset_index().to_csv(output, index=False)
    report = {
        "status": "complete",
        "source": SOURCE.as_posix(),
        "source_rows": source_rows,
        "trials": len(summary),
        "axons_per_trial": 420,
        "source_sha256": digest(source),
        "index_sha256": digest(root / "data/detection/index_regime_amplitudes.csv"),
        "summary_sha256": digest(output),
    }
    output.with_name("true_population_trial_summary_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(summarize(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
