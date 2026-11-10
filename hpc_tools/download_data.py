"""Download completed population detection or discrimination outputs safely."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATHS = (
    "data/detection/index_axon_thresholds.csv",
    "data/detection/index_axon.csv",
    "data/detection/index_regime_amplitudes.csv",
    "data/detection/index_cv0_threshold_max.csv",
    "data/detection/population_responses.csv",
    "data/detection/population_cv0_based_responses.csv",
)
DISCRIMINATION_OUTPUT_PATHS = (
    "data/discrimination/control/population_observer_rmse.csv",
    "data/discrimination/control/population_discrimination_curves.csv",
    "data/discrimination/control/population_observer_weights.csv",
    "data/discrimination/control/population_observer_readout.csv",
    "data/discrimination/control/population_effective_rate_share.csv",
    "data/discrimination/control/validation.json",
)
OUTPUTS_BY_KIND = {
    "detection": OUTPUT_PATHS,
    "discrimination": DISCRIMINATION_OUTPUT_PATHS,
    "release": tuple(line.strip() for line in
                     (Path(__file__).with_name("release_output_paths.txt")).read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.startswith("#")),
}


def _remote_command(remote_project: str, paths_to_download: tuple[str, ...]) -> str:
    project = shlex.quote(remote_project.rstrip("/"))
    paths = " ".join(shlex.quote(path) for path in paths_to_download)
    return f"set -eu; cd {project}; test -f {shlex.quote(paths_to_download[-1])}; tar -czf - {paths}"


def _install_checked(archive: Path, root: Path, paths_to_download: tuple[str, ...],
                     reserve_gib: float) -> None:
    allowed = {PurePosixPath(path) for path in paths_to_download}
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        files = {}
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe archive member received from Longleaf: {member.name}")
            if member.isfile():
                if name not in allowed or name in files:
                    raise RuntimeError(f"Unexpected or duplicate output received from Longleaf: {member.name}")
                files[name] = member
            elif not member.isdir():
                raise RuntimeError(f"Unexpected archive member received from Longleaf: {member.name}")
        missing = [path for path in paths_to_download if PurePosixPath(path) not in files]
        if missing:
            raise FileNotFoundError(
                "Longleaf download is incomplete:\n" + "\n".join(f"  - {path}" for path in missing)
            )
        for relative in paths_to_download:
            if not relative.endswith("validation.json"):
                continue
            with bundle.extractfile(files[PurePosixPath(relative)]) as source:
                report = json.load(source)
            if report.get("status") != "ready":
                raise ValueError(f"Longleaf output is not validated: {relative}")
        # Each member is written directly to its final filesystem. The old
        # staging-then-copy path required two uncompressed copies at once.
        required = sum(member.size for member in files.values())
        reserve = reserve_gib * 1024**3
        free = shutil.disk_usage(root).free
        if free < required + reserve:
            shortfall = (required + reserve - free) / 1024**3
            raise OSError(
                f"Need {shortfall:.2f} GiB more free space to install the "
                f"{required / 1024**3:.2f} GiB release outputs with a {reserve_gib:g} GiB reserve. "
                "The downloaded archive is intact; free space and rerun with --from-archive."
            )
        for relative in paths_to_download:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".hpc-new")
            try:
                with bundle.extractfile(files[PurePosixPath(relative)]) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="SSH destination, e.g. ONYEN@longleaf.unc.edu.")
    parser.add_argument("--remote-project")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--kind", choices=sorted(OUTPUTS_BY_KIND), default="detection")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--from-archive", action="store_true",
                        help="Resume installation from a previously downloaded archive without SSH")
    parser.add_argument("--archive", type=Path,
                        help="Path to a saved archive (defaults to one under --root)")
    parser.add_argument("--reserve-gib", type=float, default=2,
                        help="Free disk space to retain during installation (default: 2 GiB)")
    args = parser.parse_args()
    if not args.from_archive and (not args.host or not args.remote_project):
        parser.error("--host and --remote-project are required unless --from-archive is used")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be nonnegative")

    root = args.root.expanduser().resolve()
    paths_to_download = OUTPUTS_BY_KIND[args.kind]
    archive = (args.archive or root / f"population-{args.kind}-hpc-data.tar.gz").expanduser().resolve()
    if args.from_archive:
        if not archive.is_file():
            raise FileNotFoundError(f"Saved download archive is missing: {archive}")
        print(f"Installing completed population {args.kind} outputs from {archive}")
    else:
        print(f"Downloading completed population {args.kind} outputs from {args.host}")
        with archive.open("wb") as output:
            result = subprocess.run(
                ["ssh", args.host, _remote_command(args.remote_project, paths_to_download)],
                stdout=output, stderr=subprocess.PIPE, check=False,
            )
        if result.returncode:
            archive.unlink(missing_ok=True)
            raise RuntimeError(result.stderr.decode(errors="replace").strip() or "Remote download failed")

    _install_checked(archive, root, paths_to_download, args.reserve_gib)
    if args.keep_archive:
        print(f"Kept archive: {archive}")
    else:
        archive.unlink(missing_ok=True)
    destination = {
        "detection": "data/detection/",
        "discrimination": "data/discrimination/control/",
        "release": "data/ and results/",
    }[args.kind]
    print(f"Installed {len(paths_to_download)} completed population {args.kind} outputs under {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
