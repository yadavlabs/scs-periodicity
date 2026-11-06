"""Quarantine incompatible merged population-detection outputs.

Population response CSVs are valid only for the index axon and versioned
protocol whose regime amplitudes they contain. Production partials are already
namespaced by both values; this utility checks the small set of merged outputs
and moves incompatible files out of active paths. Legacy unversioned partials
are retained in place. It is a dry run unless ``--apply`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PARTIAL_FAMILIES = (
    "index-regimes",
    "population-responses",
    "population-cv0-responses",
)
DERIVED_OUTPUTS = (
    "index_regime_amplitudes.csv",
    "index_cv0_threshold_max.csv",
    "population_responses.csv",
    "population_cv0_based_responses.csv",
)


def _cache_metadata(path: Path) -> tuple[str, str | None]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "index_axon_id" not in reader.fieldnames:
            raise RuntimeError(f"Cached detection CSV lacks index_axon_id: {path}")
        first = next(reader, None)
    if first is None or not first.get("index_axon_id"):
        raise RuntimeError(f"Cached detection CSV is empty or lacks index metadata: {path}")
    protocol = first.get("detection_protocol_version")
    return str(first["index_axon_id"]), (None if not protocol else str(protocol))


def incompatible_cache_files(
    root: Path,
    expected_index_axon_id: str,
    expected_protocol_version: str | None = None,
) -> list[tuple[Path, str, str | None]]:
    root = root.resolve()
    candidates: list[Path] = []
    # Protocol-aware production partials are namespaced by both protocol and
    # expected index axon. Their path is the cache key, so a restart need not
    # open tens of thousands of matching CSVs merely to rediscover metadata.
    if expected_protocol_version is None:
        partial_root = root / "data" / "detection" / "partials"
        for family in PARTIAL_FAMILIES:
            candidates.extend(sorted((partial_root / family).glob("*.csv")))
    data_root = root / "data" / "detection"
    candidates.extend(data_root / name for name in DERIVED_OUTPUTS if (data_root / name).is_file())
    incompatible: list[tuple[Path, str, str | None]] = []
    for path in candidates:
        observed_index, observed_protocol = _cache_metadata(path)
        if (
            observed_index != expected_index_axon_id
            or (
                expected_protocol_version is not None
                and observed_protocol != expected_protocol_version
            )
        ):
            incompatible.append((path, observed_index, observed_protocol))
    return incompatible


def prepare_cache(
    *, root: Path, expected_index_axon_id: str, apply: bool,
    archive_tag: str | None = None, expected_protocol_version: str | None = None,
) -> dict[str, object]:
    root = root.resolve()
    incompatible = incompatible_cache_files(
        root, expected_index_axon_id, expected_protocol_version,
    )
    tag = archive_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = (
        root / "logs" / "index-cache-archive" /
        f"for-{expected_index_axon_id}" / tag
    )
    moved: list[dict[str, str]] = []
    for source, observed, observed_protocol in incompatible:
        related = [source]
        audit = source.with_suffix(".audit.json")
        if audit.is_file():
            related.append(audit)
        for item in related:
            destination = archive_root / item.relative_to(root)
            if apply:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(f"Archive destination already exists: {destination}")
                shutil.move(str(item), str(destination))
            moved.append({
                "source": str(item), "archive": str(destination),
                "observed_index_axon_id": observed,
                "observed_protocol_version": observed_protocol,
            })
    return {
        "applied": apply,
        "expected_index_axon_id": expected_index_axon_id,
        "expected_protocol_version": expected_protocol_version,
        "incompatible_csv_count": len(incompatible),
        "files_moved_or_planned": len(moved),
        "archive_root": str(archive_root),
        "files": moved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-index-axon-id", required=True)
    parser.add_argument("--expected-protocol-version")
    parser.add_argument("--archive-tag")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = prepare_cache(
        root=args.root, expected_index_axon_id=args.expected_index_axon_id,
        apply=bool(args.apply), archive_tag=args.archive_tag,
        expected_protocol_version=args.expected_protocol_version,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.apply:
        print("Dry run only; rerun with --apply after reviewing the files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
