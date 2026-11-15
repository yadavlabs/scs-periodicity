"""Restore the four accepted-fit inputs from the paired release-data ZIP.

Only these small inputs are needed before a full HPC regeneration. The other
generated files in the data ZIP can remain unextracted until comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "index_regime_amplitudes_pre_imax.csv",
    "peak_first_imax_updates.csv",
    "population_recruitment_outcomes_pre_imax.csv",
    "provenance.json",
)
PREFIX = "data/detection/behavior_fit_inputs"


def restore(archive: Path, root: Path) -> None:
    with ZipFile(archive) as bundle:
        contents = {
            name: bundle.read(f"{PREFIX}/{name}")
            for name in INPUTS
        }
    destination = root / PREFIX
    for name, contents_bytes in contents.items():
        target = destination / name
        if target.exists() and target.read_bytes() != contents_bytes:
            raise ValueError(f"Existing accepted-fit input differs: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents_bytes in contents.items():
        target = destination / name
        if not target.exists():
            target.write_bytes(contents_bytes)
    print(f"Restored or verified {len(INPUTS)} accepted-fit inputs in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    restore(args.archive.resolve(), args.root.resolve())


if __name__ == "__main__":
    main()
