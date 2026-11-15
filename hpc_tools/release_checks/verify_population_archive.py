"""Verify every population release archive entry against the SHA-256 manifest."""

from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from package_data import ARCHIVE_NAME, collect_files, sha256 as file_sha256  # noqa: E402


def main() -> None:
    archive = ROOT / ARCHIVE_NAME
    manifest = archive.with_name(archive.stem + "-manifest.csv")
    checksum = archive.with_suffix(".sha256")
    expected_checksum, expected_name = checksum.read_text(encoding="utf-8").split()
    if expected_name != archive.name or file_sha256(archive) != expected_checksum:
        raise ValueError("Archive checksum mismatch")

    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["path"] for row in rows]
    selected = [path.relative_to(ROOT).as_posix() for path in collect_files(ROOT)]
    if len(names) != len(set(names)) or names != selected:
        raise ValueError("File manifest differs from the release selector")

    with zipfile.ZipFile(archive) as package:
        if package.namelist() != names:
            raise ValueError("ZIP entry list differs from the file manifest")
        total = 0
        for row in rows:
            result = sha256()
            length = 0
            with package.open(row["path"]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    length += len(chunk)
                    result.update(chunk)
            if length != int(row["size_bytes"]) or result.hexdigest() != row["sha256"]:
                raise ValueError(f"ZIP entry does not match manifest: {row['path']}")
            total += length
    source = ROOT / "scs-periodicity-1.1.0.zip"
    if source.is_file():
        with zipfile.ZipFile(source) as package:
            source_names = {
                name.removeprefix("scs-periodicity-1.1.0/")
                for name in package.namelist() if not name.endswith("/")
            }
        if source_names.intersection(names):
            raise ValueError("Source and data ZIPs contain overlapping paths")
    print(f"Verified archive SHA-256 and {len(rows)} ZIP entries ({total / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()
