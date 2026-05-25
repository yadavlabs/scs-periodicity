"""Build the public release-data archive from local data and notebook caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


EXCLUDE_PARTS = {"partials", "__pycache__"}
DEFAULT_ARCHIVE = "scs-periodicity-release-data.zip"
DEFAULT_CHECKSUM = "scs-periodicity-release-data.sha256"
DEFAULT_MANIFEST = "scs-periodicity-release-data-manifest.csv"
RELEASE_NOTICE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(root: Path, paths: list[Path], output: Path) -> None:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": str(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    with output.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def normalize_time_pattern_metadata(root: Path) -> int:
    """Make Stage 01 metadata paths portable before distribution."""

    meta_dir = root / "data" / "time-patterns" / "generated"
    if not meta_dir.exists():
        return 0

    changed = 0
    for meta_path in sorted(meta_dir.glob("*_meta.json")):
        with meta_path.open("r", encoding="utf-8") as file_obj:
            meta = json.load(file_obj)
        npz_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_waveforms.npz"))
        repo_relative = npz_path.relative_to(root).as_posix()
        if meta.get("npz_path") == repo_relative:
            continue
        meta["npz_path"] = repo_relative
        with meta_path.open("w", encoding="utf-8") as file_obj:
            json.dump(meta, file_obj, indent=2)
            file_obj.write("\n")
        changed += 1
    return changed


def normalize_time_pattern_manifest(root: Path) -> int:
    """Make Stage 01 manifest paths portable before distribution."""

    manifest_path = root / "data" / "time-patterns" / "generated" / "manifest.csv"
    if not manifest_path.exists():
        return 0

    with manifest_path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        return 0

    changed = 0
    for row in rows:
        cv = float(row["cv"])
        freq = float(row["freq"])
        base = f"cv{cv:.1f}_f{int(freq)}"
        expected = {
            "npz": f"data/time-patterns/generated/{base}_waveforms.npz",
            "meta": f"data/time-patterns/generated/{base}_meta.json",
        }
        for key, value in expected.items():
            if row.get(key) != value:
                row[key] = value
                changed += 1

    if changed:
        with manifest_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=["cv", "freq", "npz", "meta"])
            writer.writeheader()
            writer.writerows(rows)
    return changed


def _collect_files(root: Path, folder: str) -> list[Path]:
    source = root / folder
    if not source.exists():
        raise FileNotFoundError(f"Required release directory not found: {source}")
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file()
        and not (set(path.relative_to(source).parts) & EXCLUDE_PARTS)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create one release archive containing generated data and reviewer notebook caches."
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--out", default=DEFAULT_ARCHIVE, help="Output ZIP path.")
    parser.add_argument("--checksum-out", default=DEFAULT_CHECKSUM, help="Output SHA-256 file.")
    parser.add_argument("--manifest-out", default=DEFAULT_MANIFEST, help="Output manifest CSV path.")
    parser.add_argument(
        "--allow-partials",
        action="store_true",
        help="Package despite data/partials being present; transient partials are always omitted.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    partials_dir = root / "data" / "partials"
    if partials_dir.exists() and not args.allow_partials:
        raise SystemExit(
            f"Transient partials remain at {partials_dir}. "
            "Wait for the HPC finalizer or remove this folder before packaging."
        )

    normalized_meta = normalize_time_pattern_metadata(root)
    normalized_manifest = normalize_time_pattern_manifest(root)
    if normalized_meta:
        print(f"Normalized {normalized_meta} time-pattern metadata paths.")
    if normalized_manifest:
        print(f"Normalized {normalized_manifest} time-pattern manifest paths.")

    paths = _collect_files(root, "data") + _collect_files(root, "notebooks/cache")
    for name in RELEASE_NOTICE_FILES:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Required release notice file not found: {path}")
        paths.append(path)
    paths = sorted(paths)
    if not paths:
        raise SystemExit("No files found to package.")

    archive = (root / args.out).resolve()
    checksum = (root / args.checksum_out).resolve()
    manifest = (root / args.manifest_out).resolve()
    write_manifest(root, paths, manifest)

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
        for path in paths:
            zip_file.write(path, path.relative_to(root).as_posix())

    digest = sha256_file(archive)
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Wrote {archive}")
    print(f"Wrote {manifest}")
    print(f"Wrote {checksum}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
