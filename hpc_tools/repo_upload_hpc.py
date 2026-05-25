"""Create and optionally upload a minimal HPC regeneration archive."""

from __future__ import annotations

import argparse
import fnmatch
import io
import shlex
import subprocess
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_INCLUDE_PATHS = (
    ".gitattributes",
    ".gitignore",
    "README.md",
    "data/README.md",
    "hpc_tools/hpc_requirements.txt",
    "src",
    "hpc_tools",
    "data/comsol",
    "data/time-patterns/slack",
    "data/detection/slack",
    "data/discrimination/slack",
)

PILOT_INCLUDE_PATHS = (
    "data/detection/generated/cv0_based_responses.csv",
)

EXCLUDE_PATTERNS = (
    ".git",
    ".git/*",
    ".venv",
    ".venv/*",
    "venv",
    "venv/*",
    "__pycache__",
    "*/__pycache__",
    "*/__pycache__/*",
    "*.pyc",
    "logs",
    "logs/*",
    "scs-periodicity-release-data.zip",
    "scs-periodicity-release-data.sha256",
    "scs-periodicity-release-data-manifest.csv",
    "scs-periodicity-hpc-data.tar.gz",
    "data_old",
    "data_old/*",
    "results",
    "results/*",
    "*.tar.gz",
    "notebooks/cache",
    "notebooks/cache/*",
    "data/partials",
    "data/partials/*",
    "data/time-patterns/generated",
    "data/time-patterns/generated/*",
    "data/space-time-V/generated",
    "data/space-time-V/generated/*",
    "data/discrimination/per-block",
    "data/discrimination/per-block/*",
    "data/discrimination/per-point",
    "data/discrimination/per-point/*",
    "hpc_tools/MRG_model/*.o",
    "hpc_tools/MRG_model/*.c",
    "hpc_tools/MRG_model/*.cpp",
    "hpc_tools/MRG_model/*.dll",
    "hpc_tools/MRG_model/AXNODE.mod",
    "hpc_tools/MRG_model/MRGaxon.hoc",
    "hpc_tools/MRG_model/MRGaxon.ses",
    "hpc_tools/MRG_model/mosinit.hoc",
    "hpc_tools/MRG_model/README",
    "hpc_tools/MRG_model/MODELDB_SOURCE.txt",
    "hpc_tools/MRG_model/x86_64",
    "hpc_tools/MRG_model/x86_64/*",
)

LF_NORMALIZED_SUFFIXES = (".sbatch", ".sh")


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _is_excluded(rel_path: str) -> bool:
    rel_path = rel_path.replace("\\", "/")
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in EXCLUDE_PATTERNS)


def _iter_files(include_paths: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []
    for item in include_paths:
        path = REPO_ROOT / item
        if not path.exists():
            missing.append(item)
            continue
        if path.is_file():
            rel_path = item.replace("\\", "/")
            if not _is_excluded(rel_path):
                files.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and not _is_excluded(_rel(child)):
                files.append(child)
    if missing:
        raise FileNotFoundError("Missing required upload inputs:\n" + "\n".join(f"  - {m}" for m in missing))
    return sorted(set(files), key=lambda p: _rel(p))


def create_archive(out_path: Path, *, pilot_inputs: bool) -> list[Path]:
    include_paths = tuple(BASE_INCLUDE_PATHS + (PILOT_INCLUDE_PATHS if pilot_inputs else ()))
    files = _iter_files(include_paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tf:
        for path in files:
            rel_path = _rel(path)
            if path.suffix.lower() in LF_NORMALIZED_SUFFIXES:
                data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                info = tf.gettarinfo(str(path), arcname=rel_path)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                tf.add(path, arcname=rel_path)
    return files


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _run_with_stdin(cmd: list[str], stdin_path: Path, *, dry_run: bool) -> None:
    print(" ".join(shlex.quote(str(c)) for c in cmd) + f" < {stdin_path}", flush=True)
    if dry_run:
        return
    with stdin_path.open("rb") as f:
        subprocess.run(cmd, stdin=f, check=True)


def _q(value: str) -> str:
    return shlex.quote(str(value))


def upload_archive(archive: Path, *, host: str, remote_dir: str, dry_run: bool) -> None:
    _run(["ssh", host, "mkdir", "-p", remote_dir], dry_run=dry_run)
    _run(["scp", str(archive), f"{host}:{remote_dir.rstrip('/')}/{archive.name}"], dry_run=dry_run)


def remote_project_dir(remote_dir: str, project_name: str) -> str:
    return f"{remote_dir.rstrip('/')}/{project_name.strip('/')}"


def extract_remote_archive(
    archive: Path,
    *,
    host: str,
    remote_dir: str,
    project_name: str,
    keep_remote_archive: bool,
    dry_run: bool,
) -> str:
    remote_archive = f"{remote_dir.rstrip('/')}/{archive.name}"
    project_dir = remote_project_dir(remote_dir, project_name)
    command = (
        "set -eu; "
        f"mkdir -p {_q(project_dir)}; "
        f"tar -xzf {_q(remote_archive)} -C {_q(project_dir)}"
    )
    if not keep_remote_archive:
        command += f"; rm -f {_q(remote_archive)}"
    _run(["ssh", host, command], dry_run=dry_run)
    return project_dir


def setup_remote_environment(
    *,
    host: str,
    project_dir: str,
    python_module: str,
    venv_name: str,
    module_purge: bool,
    module_load: bool,
    install_neuron: bool,
    compile_neuron: bool,
    dry_run: bool,
) -> None:
    commands = ["set -eu", f"cd {_q(project_dir)}"]
    if module_load:
        if module_purge:
            commands.append("module purge")
        commands.append(f"module load {_q(python_module)}")
    commands.extend(
        [
            f"python -m venv {_q(venv_name)}",
            f". {_q(f'{venv_name}/bin/activate')}",
            "python -m pip install --upgrade pip",
            "python -m pip install -r hpc_tools/hpc_requirements.txt",
        ]
    )
    if install_neuron:
        commands.append("python -m pip install neuron")
    if compile_neuron:
        commands.append("python hpc_tools/install_mrg_model.py --force")
        commands.append("cd hpc_tools/MRG_model && nrnivmodl && cd ../..")
    _run(["ssh", host, "; ".join(commands)], dry_run=dry_run)


def upload_extract_setup_stream(
    archive: Path,
    *,
    host: str,
    remote_dir: str,
    project_name: str,
    setup_env: bool,
    python_module: str,
    venv_name: str,
    module_purge: bool,
    module_load: bool,
    install_neuron: bool,
    compile_neuron: bool,
    keep_remote_archive: bool,
    dry_run: bool,
) -> str:
    """Stream the archive to the remote host and run all requested setup in one SSH session."""

    remote_archive = f"{remote_dir.rstrip('/')}/{archive.name}"
    project_dir = remote_project_dir(remote_dir, project_name)
    commands = [
        "set -eu",
        f"mkdir -p {_q(remote_dir)}",
        f"cat > {_q(remote_archive)}",
        f"mkdir -p {_q(project_dir)}",
        f"tar -xzf {_q(remote_archive)} -C {_q(project_dir)}",
    ]
    if not keep_remote_archive:
        commands.append(f"rm -f {_q(remote_archive)}")
    if setup_env:
        commands.append(f"cd {_q(project_dir)}")
        if module_load:
            if module_purge:
                commands.append("module purge")
            commands.append(f"module load {_q(python_module)}")
        commands.extend(
            [
                f"python -m venv {_q(venv_name)}",
                f". {_q(f'{venv_name}/bin/activate')}",
                "python -m pip install --upgrade pip",
                "python -m pip install -r hpc_tools/hpc_requirements.txt",
            ]
        )
        if install_neuron:
            commands.append("python -m pip install neuron")
        if compile_neuron:
            commands.append("python hpc_tools/install_mrg_model.py --force")
            commands.append("cd hpc_tools/MRG_model && nrnivmodl && cd ../..")
    _run_with_stdin(["ssh", host, "; ".join(commands)], archive, dry_run=dry_run)
    return project_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Package and optionally upload a minimal HPC repo archive.")
    parser.add_argument("--out", default="scs-periodicity-hpc.tar.gz", help="Local archive path.")
    parser.add_argument("--pilot-inputs", action="store_true", help="Include archived stage-03 CSV needed for pilot runs.")
    parser.add_argument("--host", default=None, help="SSH host, e.g. ONYEN@longleaf.unc.edu.")
    parser.add_argument("--remote-dir", default=None, help="Remote parent directory for the archive and extracted project.")
    parser.add_argument("--project-name", default="scs-periodicity", help="Project directory created under --remote-dir.")
    parser.add_argument("--no-extract", action="store_true", help="Upload only; do not extract the archive remotely.")
    parser.add_argument("--keep-local-archive", action="store_true", help="Keep the local tarball after a successful upload.")
    parser.add_argument("--keep-remote-archive", action="store_true", help="Keep the uploaded tarball on the remote host after extraction.")
    parser.add_argument("--scp", action="store_true", help="Use separate ssh/scp commands instead of the default one-connection streamed upload.")
    parser.add_argument("--setup-env", action="store_true", help="After extraction, create the venv, install requirements, install NEURON, and compile mechanisms.")
    parser.add_argument("--python-module", default="python/3.12.4", help="Environment module to load during --setup-env.")
    parser.add_argument("--skip-module-load", action="store_true", help="Do not run module purge/load during --setup-env.")
    parser.add_argument("--no-module-purge", action="store_true", help="Do not run module purge before module load.")
    parser.add_argument("--venv-name", default="venv", help="Remote virtual environment directory name.")
    parser.add_argument("--skip-neuron", action="store_true", help="Do not pip install neuron during --setup-env.")
    parser.add_argument("--skip-nrnivmodl", action="store_true", help="Do not download or compile the MRG mechanisms during --setup-env.")
    parser.add_argument("--dry-run", action="store_true", help="Print upload commands without running ssh/scp.")
    args = parser.parse_args()

    archive = Path(args.out).expanduser().resolve()
    files = create_archive(archive, pilot_inputs=bool(args.pilot_inputs))
    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"Wrote {archive}")
    print(f"Included {len(files)} files ({size_mb:.1f} MB archive).")

    if args.host or args.remote_dir:
        if not args.host or not args.remote_dir:
            raise SystemExit("Pass both --host and --remote-dir to upload.")
        project_dir = remote_project_dir(str(args.remote_dir), str(args.project_name))
        if args.setup_env and args.no_extract:
            raise SystemExit("--setup-env requires remote extraction; omit --no-extract.")

        if args.scp:
            upload_archive(archive, host=str(args.host), remote_dir=str(args.remote_dir), dry_run=bool(args.dry_run))
            if not args.no_extract:
                project_dir = extract_remote_archive(
                    archive,
                    host=str(args.host),
                    remote_dir=str(args.remote_dir),
                    project_name=str(args.project_name),
                    keep_remote_archive=bool(args.keep_remote_archive),
                    dry_run=bool(args.dry_run),
                )
            if args.setup_env:
                setup_remote_environment(
                    host=str(args.host),
                    project_dir=project_dir,
                    python_module=str(args.python_module),
                    venv_name=str(args.venv_name),
                    module_purge=not bool(args.no_module_purge),
                    module_load=not bool(args.skip_module_load),
                    install_neuron=not bool(args.skip_neuron),
                    compile_neuron=not bool(args.skip_nrnivmodl),
                    dry_run=bool(args.dry_run),
                )
        elif args.no_extract:
            upload_archive(archive, host=str(args.host), remote_dir=str(args.remote_dir), dry_run=bool(args.dry_run))
        else:
            if args.no_extract:
                raise SystemExit("--setup-env requires remote extraction; omit --no-extract.")
            project_dir = upload_extract_setup_stream(
                archive,
                host=str(args.host),
                remote_dir=str(args.remote_dir),
                project_name=str(args.project_name),
                setup_env=bool(args.setup_env),
                python_module=str(args.python_module),
                venv_name=str(args.venv_name),
                module_purge=not bool(args.no_module_purge),
                module_load=not bool(args.skip_module_load),
                install_neuron=not bool(args.skip_neuron),
                compile_neuron=not bool(args.skip_nrnivmodl),
                keep_remote_archive=bool(args.keep_remote_archive),
                dry_run=bool(args.dry_run),
            )

        print("\nOn the HPC system:")
        print(f"  cd {project_dir}")
        if args.setup_env:
            if not args.skip_module_load:
                if not args.no_module_purge:
                    print("  module purge")
                print(f"  module load {args.python_module}")
            print(f"  source {args.venv_name}/bin/activate")
        else:
            print("  module purge")
            print(f"  module load {args.python_module}")
            print(f"  python -m venv {args.venv_name}")
            print(f"  source {args.venv_name}/bin/activate")
            print("  python -m pip install --upgrade pip")
            print("  python -m pip install -r hpc_tools/hpc_requirements.txt")
            print("  python -m pip install neuron")
            print("  python hpc_tools/install_mrg_model.py --force")
            print("  cd hpc_tools/MRG_model && nrnivmodl && cd ../..")
        print("  python hpc_tools/query_hpc.py")

        if not args.keep_local_archive and not args.dry_run:
            archive.unlink(missing_ok=True)
            print(f"\nRemoved local archive {archive}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
