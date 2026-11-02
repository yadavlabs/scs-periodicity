"""Print Longleaf/SLURM and NEURON information relevant to this pipeline."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _probe(label: str, command: list[str]) -> int:
    print(f"\n[{label}]")
    if shutil.which(command[0]) is None:
        print(f"{command[0]} not found on PATH")
        return 127
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode:
        print(f"Command exited with status {result.returncode}")
    return int(result.returncode)


def _slurm_limits() -> int:
    print("\n[SLURM configured limits]")
    if shutil.which("scontrol") is None:
        print("scontrol not found on PATH")
        return 127
    result = subprocess.run(
        ["scontrol", "show", "config"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    wanted = ("MaxArraySize", "MaxJobCount", "MaxStepCount")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(wanted)]
    print("\n".join(lines) if lines else "No array/job limit lines returned")
    if result.stderr.strip():
        print(result.stderr.strip())
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=os.environ.get("USER") or os.environ.get("USERNAME"))
    args = parser.parse_args()
    user = args.user or "$USER"
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    statuses = [
        _probe("SLURM", ["sbatch", "--version"]),
        _slurm_limits(),
        _probe("partitions", ["sinfo", "-s", "-o", "%P %c %m %l %a"]),
        _probe(
            "user associations",
            ["sacctmgr", "show", "assoc", f"user={user}", "format=Account,Partition,QOS,MaxTRESPerJob,MaxWall", "-P"],
        ),
        _probe("NEURON", [sys.executable, "-c", "import neuron; print(neuron.__version__)"]),
    ]
    return 0 if any(status == 0 for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
