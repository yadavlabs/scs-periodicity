"""Download regenerated pipeline outputs from an HPC workspace into this repository."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_PATHS = (
    "data/detection/generated",
    "data/discrimination/data_manifest.json",
    "data/discrimination/run_config.json",
    "data/discrimination/per-block",
    "data/discrimination/per-point",
    "data/space-time-V/generated",
    "data/time-patterns/generated",
)


def _remote_archive_command(remote_project: str) -> str:
    quoted_project = shlex.quote(remote_project.rstrip("/"))
    paths = " ".join(shlex.quote(path) for path in GENERATED_PATHS)
    return f"set -eu; cd {quoted_project}; tar -czf - {paths}"


def _extract_checked(archive: Path, staging: Path) -> None:
    allowed = tuple(PurePosixPath(path) for path in GENERATED_PATHS)
    with tarfile.open(archive, "r:gz") as tar_file:
        for member in tar_file.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe archive member received from HPC: {member.name}")
            if not any(name == path or path in name.parents for path in allowed):
                raise RuntimeError(f"Unexpected archive member received from HPC: {member.name}")
        tar_file.extractall(staging, filter="data")


def _replace_generated_outputs(staging: Path, root: Path) -> None:
    missing = [path for path in GENERATED_PATHS if not (staging / path).exists()]
    if missing:
        raise FileNotFoundError(
            "HPC download is missing required completed-pipeline outputs:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    for relative in GENERATED_PATHS:
        source = staging / relative
        destination = root / relative
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve completed HPC-generated data outputs and install them in the local repository."
    )
    parser.add_argument("--host", required=True, help="SSH host, for example ONYEN@longleaf.unc.edu.")
    parser.add_argument("--remote-project", required=True, help="Remote scs-periodicity repository path.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Local repository root.")
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded temporary TAR archive at the repository root.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not (root / "data").exists():
        raise FileNotFoundError(f"Local data directory not found: {root / 'data'}")

    archive = root / "scs-periodicity-hpc-data.tar.gz"
    command = _remote_archive_command(args.remote_project)
    print(f"Downloading generated pipeline outputs from {args.host}:{args.remote_project}")
    with archive.open("wb") as output:
        process = subprocess.run(
            ["ssh", args.host, command],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    if process.returncode != 0:
        archive.unlink(missing_ok=True)
        message = process.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Remote data download failed:\n{message}")

    with tempfile.TemporaryDirectory(prefix="scs-hpc-data-", dir=root) as temp_dir:
        staging = Path(temp_dir)
        _extract_checked(archive, staging)
        _replace_generated_outputs(staging, root)

    if args.keep_archive:
        print(f"Kept downloaded archive at {archive}")
    else:
        archive.unlink(missing_ok=True)
    print("Installed regenerated outputs under local data/.")
    print("Regenerate notebook caches locally, then run: python src/package_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
