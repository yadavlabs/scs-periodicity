"""Package or stream the population workspace to Longleaf.

The default upload includes code, SLURM files, population input bases,
and behavioral source inputs. The MRG model is installed remotely.
Complete remote environments and large generated outputs are excluded.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import shlex
import socket
import subprocess
import tarfile
from pathlib import Path
from typing import BinaryIO, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "scs-periodicity-population"

CODE_PATHS = (
    ".gitattributes",
    ".gitignore",
    "README.md",
    "requirements.txt",
    "hpc_tools",
    "src",
    "notebooks",
)
UPSTREAM_INPUT_PATHS = (
    "data/comsol",
    "data/time-patterns/slack",
    "data/time-patterns/generated",
    "data/detection/reference",
    "data/detection/behavior_fit_inputs",
)
DETECTION_INPUT_PATHS = (
    "data/space-time-V",
)
OBSERVER_INPUT_PATHS = (
    "data/detection/slack",
    "data/discrimination/slack",
)
EXCLUDE_PATTERNS = (
    ".git", ".git/*", ".venv", ".venv/*", "venv", "venv/*",
    "__pycache__", "*/__pycache__", "*/__pycache__/*", "*.pyc",
    "logs", "logs/*", "results", "results/*", "*.tar", "*.tar.gz",
    "notebooks/cache/cache_*.csv",
    "data/space-time-V/*SpaceTimeV.npz",
    "hpc_tools/release_checks/*",
)
LF_SUFFIXES = (".sh", ".sbatch")


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _excluded(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    if normalized.startswith("hpc_tools/MRG_model/"):
        return normalized != "hpc_tools/MRG_model/README.md"
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in EXCLUDE_PATTERNS)


def collect_files(
    *, include_upstream: bool = True, include_detection_inputs: bool = True,
    include_observer_inputs: bool = True,
) -> list[Path]:
    include_paths = list(CODE_PATHS)
    if include_upstream:
        include_paths.extend(UPSTREAM_INPUT_PATHS)
    if include_detection_inputs:
        include_paths.extend(DETECTION_INPUT_PATHS)
    if include_observer_inputs:
        include_paths.extend(OBSERVER_INPUT_PATHS)
    files: list[Path] = []
    missing: list[str] = []
    for relative in include_paths:
        path = REPO_ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        if path.is_file():
            if not _excluded(relative):
                files.append(path)
            continue
        files.extend(
            child for child in sorted(path.rglob("*"))
            if child.is_file() and not _excluded(_relative(child))
        )
    if missing:
        raise FileNotFoundError(
            "Missing required upload paths:\n" + "\n".join(f"  - {item}" for item in missing)
        )
    return sorted(set(files), key=_relative)


def _add_file(archive: tarfile.TarFile, path: Path) -> None:
    relative = _relative(path)
    if path.suffix.lower() in LF_SUFFIXES:
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        info = archive.gettarinfo(str(path), arcname=relative)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    else:
        archive.add(path, arcname=relative, recursive=False)


def create_archive(output: Path, files: Iterable[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if output.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(output, mode) as archive:
        for path in files:
            _add_file(archive, path)


class _ProgressWriter:
    def __init__(self, stream: BinaryIO, total_bytes: int) -> None:
        self.stream = stream
        self.total_bytes = total_bytes
        self.written = 0
        self.next_report = 512 * 1024 * 1024
        self.error: OSError | None = None

    def write(self, data: bytes) -> int:
        # If the SSH pipe failed, TarFile.__exit__ may make one final buffered
        # write while unwinding. Pretend that cleanup write succeeded so it
        # cannot mask the original, useful BrokenPipeError.
        if self.error is not None:
            return len(data)
        try:
            count = self.stream.write(data)
        except OSError as error:
            self.error = error
            raise
        self.written += count
        if self.written >= self.next_report:
            print(
                f"Streamed {self.written / (1024 ** 3):.1f} GiB "
                f"of approximately {self.total_bytes / (1024 ** 3):.1f} GiB",
                flush=True,
            )
            self.next_report += 512 * 1024 * 1024
        return count

    def flush(self) -> None:
        if self.error is None:
            self.stream.flush()


def remote_project_dir(remote_dir: str, project_name: str) -> str:
    return f"{remote_dir.rstrip('/')}/{project_name.strip('/')}"


def remote_command(
    *, project_dir: str, setup_env: bool, submit: bool, python_module: str,
    venv_name: str, force_compile: bool, detection_only: bool,
    submit_discrimination: bool, submit_pilot: bool,
) -> str:
    commands = [
        "set -euo pipefail",
        f"mkdir -p {_quote(project_dir)}",
        f"tar -xf - -C {_quote(project_dir)}",
        f"cd {_quote(project_dir)}",
        "mkdir -p logs",
    ]
    if setup_env or submit or submit_discrimination or submit_pilot:
        commands.extend(["module purge", f"module load {_quote(python_module)}"])
    if setup_env:
        commands.extend([
            f"python -m venv {_quote(venv_name)}",
            f". {_quote(f'{venv_name}/bin/activate')}",
            "python -m pip install --upgrade pip",
            "python -m pip install -r hpc_tools/hpc_requirements.txt",
            "python hpc_tools/install_mrg_model.py" + (" --force" if force_compile else ""),
            f"export PYTHONPATH={_quote(project_dir + '/src')}",
            "python -c 'import neuron; print(\"NEURON\", neuron.__version__)'",
        ])
    elif submit:
        commands.extend([
            f". {_quote(f'{venv_name}/bin/activate')}",
            f"export PYTHONPATH={_quote(project_dir + '/src')}",
            "python hpc_tools/install_mrg_model.py",
        ])
    elif submit_discrimination or submit_pilot:
        commands.extend([
            f". {_quote(f'{venv_name}/bin/activate')}",
            f"export PYTHONPATH={_quote(project_dir + '/src')}",
        ])
    if submit_pilot:
        commands.extend([
            f"export PROJECT={_quote(project_dir)}",
            "sbatch --parsable hpc_tools/slurm/pilot/smoke.sbatch",
        ])
    elif submit:
        submit_script = (
            "hpc_tools/slurm/detection/run_detection_pipeline.sh"
            if detection_only else "hpc_tools/slurm/run_population_pipeline.sh"
        )
        commands.extend([
            f"export PROJECT={_quote(project_dir)}",
            f"bash {submit_script}",
        ])
    elif submit_discrimination:
        commands.extend([
            f"export PROJECT={_quote(project_dir)}",
            "bash hpc_tools/slurm/discrimination/run_population_discrimination_pipeline.sh",
        ])
    return "; ".join(commands)


def _ssh_endpoint(host: str) -> tuple[str, int]:
    """Resolve the effective OpenSSH hostname/port, including SSH config aliases."""
    result = subprocess.run(
        ["ssh", "-G", host], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    hostname = host.rsplit("@", 1)[-1]
    port = 22
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            key, _, value = line.partition(" ")
            if key == "hostname" and value.strip():
                hostname = value.strip()
            elif key == "port" and value.strip().isdigit():
                port = int(value.strip())
    return hostname, port


def check_ssh_reachable(host: str, *, timeout_seconds: float) -> None:
    hostname, port = _ssh_endpoint(host)
    try:
        with socket.create_connection((hostname, port), timeout=timeout_seconds):
            pass
    except OSError as error:
        raise ConnectionError(
            f"Cannot reach {hostname} on TCP port {port} within {timeout_seconds:g} seconds. "
            "No upload was attempted. If you are off campus, connect to the UNC VPN, "
            f"then confirm an interactive login with: ssh {host}"
        ) from error


def stream_upload(
    files: list[Path], *, host: str, command: str, dry_run: bool,
    connect_timeout_seconds: float,
) -> None:
    print(f"ssh {_quote(host)} {_quote(command)}", flush=True)
    if dry_run:
        return
    check_ssh_reachable(host, timeout_seconds=connect_timeout_seconds)
    process = subprocess.Popen(["ssh", host, command], stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("Could not open SSH standard input")
    writer = _ProgressWriter(process.stdin, sum(path.stat().st_size for path in files))
    try:
        with tarfile.open(fileobj=writer, mode="w|") as archive:
            for path in files:
                _add_file(archive, path)
        process.stdin.close()
        status = process.wait()
    except (BrokenPipeError, OSError) as error:
        if process.poll() is None:
            process.kill()
        status = process.wait()
        raise RuntimeError(
            f"SSH closed the upload stream before extraction completed (exit status {status}). "
            "No local data were changed; rerun after confirming an interactive SSH login."
        ) from error
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    if status != 0:
        raise subprocess.CalledProcessError(status, ["ssh", host, command])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="SSH destination, e.g. ONYEN@longleaf.unc.edu.")
    parser.add_argument("--remote-dir", help="Remote parent, e.g. /work/users/t/v/ONYEN.")
    parser.add_argument("--project-name", default=PROJECT_NAME)
    parser.add_argument("--archive", type=Path, help="Create a local TAR/TAR.GZ instead of uploading.")
    parser.add_argument("--skip-upstream-inputs", action="store_true", help="Exclude COMSOL cutlines and time patterns.")
    parser.add_argument("--skip-detection-inputs", action="store_true", help="Exclude population Ve bases and manifest for a code-only update.")
    parser.add_argument("--skip-observer-inputs", action="store_true", help="Exclude population evidence and Slack behavioral inputs.")
    parser.add_argument("--setup-env", action="store_true", help="Create/update venv, install NEURON, and compile AXNODE.mod.")
    parser.add_argument("--submit", action="store_true", help="Submit the full population generation/detection workflow after upload/setup.")
    parser.add_argument("--submit-discrimination", action="store_true", help="Submit the population discrimination observer after upload.")
    parser.add_argument("--submit-pilot", action="store_true", help="Submit the isolated three-condition population smoke job.")
    parser.add_argument(
        "--detection-only", action="store_true",
        help="With --submit, skip Longleaf SpaceTimeV generation and validation.",
    )
    parser.add_argument("--python-module", default="python/3.12.4")
    parser.add_argument("--venv-name", default="venv")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--connect-timeout", type=float, default=15.0, help="TCP preflight timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-files", action="store_true")
    args = parser.parse_args()
    if args.detection_only and not args.submit:
        parser.error("--detection-only requires --submit")
    if sum((bool(args.submit), bool(args.submit_discrimination), bool(args.submit_pilot))) > 1:
        parser.error("Choose only one of --submit, --submit-discrimination, or --submit-pilot")

    files = collect_files(
        include_upstream=not args.skip_upstream_inputs,
        include_detection_inputs=not args.skip_detection_inputs,
        include_observer_inputs=not args.skip_observer_inputs,
    )
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Local source: {REPO_ROOT}")
    print(f"Selected {len(files):,} files totaling {total_bytes / (1024 ** 3):.2f} GiB")
    if args.list_files:
        for path in files:
            print(_relative(path))

    if args.archive:
        if args.host or args.remote_dir:
            parser.error("--archive cannot be combined with --host/--remote-dir; use streaming upload or archive creation.")
        output = args.archive.expanduser().resolve()
        if args.dry_run:
            print(f"Would create {output}")
        else:
            create_archive(output, files)
            print(f"Created {output} ({output.stat().st_size / (1024 ** 3):.2f} GiB)")
        return 0

    if not args.host or not args.remote_dir:
        parser.error("Pass --host and --remote-dir to upload, or --archive to create a local package.")
    project_dir = remote_project_dir(args.remote_dir, args.project_name)
    command = remote_command(
        project_dir=project_dir, setup_env=bool(args.setup_env), submit=bool(args.submit),
        python_module=args.python_module, venv_name=args.venv_name,
        force_compile=bool(args.force_compile), detection_only=bool(args.detection_only),
        submit_discrimination=bool(args.submit_discrimination), submit_pilot=bool(args.submit_pilot),
    )
    stream_upload(
        files, host=args.host, command=command, dry_run=bool(args.dry_run),
        connect_timeout_seconds=float(args.connect_timeout),
    )
    print(f"Longleaf project: {args.host}:{project_dir}")
    if not args.submit and not args.submit_discrimination and not args.submit_pilot:
        if args.skip_upstream_inputs and args.skip_detection_inputs and not args.skip_observer_inputs:
            submit_script = "hpc_tools/slurm/discrimination/run_population_discrimination_pipeline.sh"
        else:
            submit_script = "hpc_tools/slurm/run_population_pipeline.sh"
        print(
            f"Submit later: ssh {args.host} "
            f"'cd {project_dir} && PROJECT={project_dir} bash {submit_script}'"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
