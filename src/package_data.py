"""Build the generated population data archive paired with the source ZIP.

Run after the full population HPC outputs have been retrieved and the final
figure notebook has been rerun. This script intentionally refuses to package
an incomplete dataset or any historical single-axon output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path


ARCHIVE_NAME = "scs-periodicity-population-release-data.zip"
SOURCE_INPUT_DIRS = (
    "data/comsol",
    "data/time-patterns/population",
    "data/time-patterns/slack",
    "data/detection/slack",
    "data/discrimination/slack",
    "data/detection/reference",
)
GENERATED_DIRS = (
    "data/detection/behavior_fit_inputs",
    "data/features",
    "results/detection/behavioral_matching",
    "results/discrimination/control",
    "results/discrimination/scope",
    "results/discrimination/locked/followup",
    "results/discrimination/locked/inputs",
    "results/figures",
)
NOTICE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff")
EXTRA_FILES = (
    "results/detection/imax_refresh_manifest.json",
    "results/detection/true_population_trial_summary.csv",
    "results/detection/true_population_trial_summary_manifest.json",
)
EXCLUDED_PARTS = {"partials", "__pycache__", "old", "exploratory", "submitted_jobs.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[Path]:
    missing = [name for name in (*SOURCE_INPUT_DIRS, *GENERATED_DIRS, *NOTICE_FILES,
                                 *EXTRA_FILES, "data/detection")
               if not (root / name).exists()]
    if missing:
        raise FileNotFoundError("Missing release inputs:\n" + "\n".join(missing))
    # Source inputs, Ve bases, and notices already live in the source ZIP.
    # Repeating them here would overwrite Git-normalized copies on extraction.
    files = [root / name for name in EXTRA_FILES]
    files.extend((root / "data/detection").glob("*.csv"))
    for directory in GENERATED_DIRS:
        files.extend(path for path in (root / directory).rglob("*")
                     if path.is_file() and not any(
                         part in EXCLUDED_PARTS or part.startswith("exploratory_")
                         for part in path.relative_to(root).parts
                     ))

    basis_dir = root / "data/space-time-V"
    manifest = basis_dir / "axon_basis_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 420 or len({row["axon_id"] for row in rows}) != 420:
        raise ValueError("Release requires the exact 420-axon basis manifest")
    for row in rows:
        basis = basis_dir / f"{row['axon_id']}_1uA_VeBasis.npz"
        if not basis.is_file():
            raise FileNotFoundError(basis)

    needed = (
        "data/detection/index_regime_amplitudes.csv",
        "data/detection/population_responses.csv",
        "data/detection/population_cv0_based_responses.csv",
        "data/features/population_cv0_template_evidence.csv",
        "results/discrimination/scope/validation.json",
        "results/discrimination/locked/followup/validation.json",
    )
    for name in needed:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required production output is missing: {name}")
    for path in files:
        relative = path.relative_to(root).as_posix().lower()
        if "/single/" in f"/{relative}/" or "mrgaxon.hoc" in relative or "mrgaxon_population_template.hoc" in relative:
            raise ValueError(f"Non-release file selected: {relative}")
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(ARCHIVE_NAME))
    parser.add_argument("--dry-run", action="store_true", help="List selected size without hashing or writing an archive")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        files = collect_files(root)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    total = sum(path.stat().st_size for path in files)
    print(f"Selected {len(files):,} population files, {total / 1024**3:.2f} GiB uncompressed")
    if args.dry_run:
        return 0

    archive = (root / args.out).resolve()
    manifest = archive.with_name(archive.stem + "-manifest.csv")
    checksum = archive.with_suffix(".sha256")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "size_bytes", "sha256"))
        for path in files:
            writer.writerow((path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as output:
        for path in files:
            output.write(path, path.relative_to(root).as_posix())
    digest = sha256(archive)
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Wrote {archive}, {manifest}, and {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
