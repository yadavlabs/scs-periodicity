"""Download completed population detection or discrimination outputs safely."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import tarfile
import tempfile
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
PEAK_FIRST_IMAX_OUTPUT_PATHS = (
    "data/detection/peak_first_imax_update/conditions.csv",
    "data/detection/peak_first_imax_update/preflight.json",
    "data/detection/peak_first_imax_update/index_regime_amplitudes.csv",
    "data/detection/peak_first_imax_update/peak_first_imax_updates.csv",
    "data/detection/peak_first_imax_update/peak_first_imax_audit_summary.csv",
    "data/detection/peak_first_imax_update/merge_manifest.json",
)
OUTPUTS_BY_KIND = {
    "detection": OUTPUT_PATHS,
    "discrimination": DISCRIMINATION_OUTPUT_PATHS,
    "peak-first-imax": PEAK_FIRST_IMAX_OUTPUT_PATHS,
    "release": tuple(line.strip() for line in
                     (Path(__file__).with_name("release_output_paths.txt")).read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.startswith("#")),
}


def _remote_command(remote_project: str, paths_to_download: tuple[str, ...]) -> str:
    project = shlex.quote(remote_project.rstrip("/"))
    paths = " ".join(shlex.quote(path) for path in paths_to_download)
    return f"set -eu; cd {project}; test -f {shlex.quote(paths_to_download[-1])}; tar -czf - {paths}"


def _extract_checked(archive: Path, staging: Path, paths_to_download: tuple[str, ...]) -> None:
    allowed = {PurePosixPath(path) for path in paths_to_download}
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe archive member received from Longleaf: {member.name}")
            if member.isfile() and name not in allowed:
                raise RuntimeError(f"Unexpected output received from Longleaf: {member.name}")
        bundle.extractall(staging, filter="data")


def _install_outputs(staging: Path, root: Path, paths_to_download: tuple[str, ...]) -> None:
    missing = [path for path in paths_to_download if not (staging / path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Longleaf download is incomplete:\n" + "\n".join(f"  - {path}" for path in missing)
        )
    for relative in paths_to_download:
        source = staging / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".hpc-new")
        shutil.copy2(source, temporary)
        temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH destination, e.g. ONYEN@longleaf.unc.edu.")
    parser.add_argument("--remote-project", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--kind", choices=sorted(OUTPUTS_BY_KIND), default="detection")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    paths_to_download = OUTPUTS_BY_KIND[args.kind]
    archive = root / f"population-{args.kind}-hpc-data.tar.gz"
    print(f"Downloading completed population {args.kind} outputs from {args.host}")
    with archive.open("wb") as output:
        result = subprocess.run(
            ["ssh", args.host, _remote_command(args.remote_project, paths_to_download)],
            stdout=output, stderr=subprocess.PIPE, check=False,
        )
    if result.returncode:
        archive.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or "Remote download failed")

    with tempfile.TemporaryDirectory(prefix="population-hpc-download-", dir=root) as temp_dir:
        staging = Path(temp_dir)
        _extract_checked(archive, staging, paths_to_download)
        _install_outputs(staging, root, paths_to_download)
    if args.keep_archive:
        print(f"Kept archive: {archive}")
    else:
        archive.unlink(missing_ok=True)
    destination = {
        "detection": "data/detection/",
        "discrimination": "data/discrimination/control/",
        "peak-first-imax": "data/detection/peak_first_imax_update/",
        "release": "data/ and results/",
    }[args.kind]
    print(f"Installed {len(paths_to_download)} completed population {args.kind} outputs under {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
