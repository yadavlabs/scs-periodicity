"""Print SLURM resource information useful for choosing pipeline flags."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess


def _run_probe(label: str, cmd: list[str]) -> int:
    print(f"\n[{label}]")
    if shutil.which(cmd[0]) is None:
        print(f"{cmd[0]} not found on PATH.")
        return 127
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if proc.returncode != 0:
        print(f"Command exited with status {proc.returncode}.")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query SLURM limits for HPC pipeline submission.")
    parser.add_argument("--user", default=os.environ.get("USER") or os.environ.get("USERNAME"), help="Cluster username.")
    args = parser.parse_args()

    user = args.user or "$USER"
    print("SLURM resource query")
    print("Use these values to choose --partition, --account, --qos, CPU, memory, and wall-time flags.")
    statuses = [
        _run_probe("sbatch", ["sbatch", "--version"]),
        _run_probe("partitions", ["sinfo", "-s", "-o", "%P %c %m %l %a"]),
        _run_probe(
            "user associations",
            [
                "sacctmgr",
                "show",
                "assoc",
                f"user={user}",
                "format=Account,Partition,QOS,MaxTRESPerJob,MaxWall",
                "-P",
            ],
        ),
    ]
    return 0 if any(status == 0 for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
