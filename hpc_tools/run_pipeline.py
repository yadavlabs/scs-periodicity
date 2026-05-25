"""Submit or run the full release data-generation pipeline.

The cleaned notebooks are intentionally easy-mode: they inspect archived stage
outputs and do not regenerate data locally. This script is the canonical full
regeneration entrypoint for an HPC environment with SLURM, NEURON, and compiled
MRG mechanisms.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


@dataclass(frozen=True)
class SlurmProfile:
    cpus: int
    mem: str
    time: str


def _quote_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _clean_job_id(raw: str) -> str:
    job_id = str(raw).strip().splitlines()[-1].strip()
    job_id = job_id.split(";")[0].strip()
    if not job_id:
        raise RuntimeError(f"Could not parse SLURM job id from output: {raw!r}")
    return job_id


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> str:
    print(_quote_command(cmd), flush=True)
    if dry_run:
        return "DRY_RUN"
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr, flush=True)
    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd,
            output=proc.stdout,
            stderr=proc.stderr,
        )
    return proc.stdout.strip()


def _export_arg(values: dict[str, str]) -> str:
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def _dependency_arg(job_ids: str | list[str] | tuple[str, ...] | None) -> str | None:
    if not job_ids:
        return None
    if isinstance(job_ids, str):
        joined = job_ids
    else:
        joined = ":".join(str(job_id) for job_id in job_ids if str(job_id))
    if not joined:
        return None
    return f"afterok:{joined}"


def _array_spec_from_cpu_cap(
    total_tasks: int,
    *,
    cpus_per_task: int,
    max_user_cpus: int,
    cpu_buffer: int,
) -> str:
    total = int(total_tasks)
    cpus = max(1, int(cpus_per_task))
    cap = max(1, int(max_user_cpus))
    usable_cpus = max(cpus, cap - max(0, int(cpu_buffer)))
    if total <= 0:
        raise ValueError("Array task count must be > 0")
    max_parallel = max(1, usable_cpus // cpus)
    parallel = min(total, max_parallel)
    return f"0-{total - 1}%{parallel}"


def _cluster_args(args: argparse.Namespace) -> list[str]:
    opts: list[str] = []
    for flag, value in [
        ("--partition", args.partition),
        ("--account", args.account),
        ("--qos", args.qos),
        ("--constraint", args.constraint),
    ]:
        if value:
            opts.extend([flag, str(value)])
    return opts


def submit_sbatch(
    script: str,
    *,
    project: Path,
    export: dict[str, str],
    profile: SlurmProfile,
    job_name: str,
    output: str,
    error: str,
    cluster_args: list[str],
    dependency: str | list[str] | tuple[str, ...] | None = None,
    array: str | None = None,
    dry_run: bool = False,
) -> str:
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--cpus-per-task",
        str(profile.cpus),
        "--mem",
        str(profile.mem),
        "--time",
        str(profile.time),
        "--output",
        output,
        "--error",
        error,
    ]
    cmd.extend(cluster_args)
    dep = _dependency_arg(dependency)
    if dep:
        cmd.extend(["--dependency", dep])
    if array:
        cmd.extend(["--array", array])
    cmd.extend(["--export", _export_arg(export), script])
    output_text = _run_command(cmd, cwd=project, dry_run=dry_run)
    if dry_run:
        label = re.sub(r"[^A-Za-z0-9]+", "_", job_name).strip("_").upper()
        return f"DRY_{label}"
    return _clean_job_id(output_text)


def _log_path(log_dir: str, filename: str) -> str:
    return f"{log_dir.rstrip('/')}/{filename}"


def _submit_discrimination_run(
    *,
    run_kind: str,
    project: Path,
    base_export: dict[str, str],
    log_dir: str,
    cluster_args: list[str],
    worker_profile: SlurmProfile,
    merge_profile: SlurmProfile,
    dependency: str | None,
    mc_total: int,
    mc_shards: int,
    max_parallel: int,
    dry_run: bool,
) -> str:
    if run_kind == "per_block":
        total = 2 * 3 * mc_shards
    elif run_kind == "per_point":
        total = 8 * 1 * mc_shards
    else:
        raise ValueError(f"Unknown discrimination run kind: {run_kind}")

    parallel = max(1, min(int(max_parallel), total))
    array = f"0-{total - 1}%{parallel}"
    partial_root = f"{base_export['PROJECT']}/data/partials/discrimination/{run_kind}"
    out_folder = "per-block" if run_kind == "per_block" else "per-point"
    out_root = f"{base_export['PROJECT']}/data/discrimination/{out_folder}"
    safe_kind = run_kind.replace("_", "-")

    array_job = submit_sbatch(
        "hpc_tools/slurm/discrimination/01_discrimination_array.sbatch",
        project=project,
        export={
            **base_export,
            "RUN_KIND": run_kind,
            "PARTIAL_ROOT": partial_root,
            "MC_TOTAL": str(mc_total),
            "MC_SHARDS": str(mc_shards),
        },
        profile=SlurmProfile(cpus=1, mem=worker_profile.mem, time=worker_profile.time),
        job_name=f"disc-release-{safe_kind}",
        output=_log_path(log_dir, f"disc_release_{safe_kind}_%A_%a.out"),
        error=_log_path(log_dir, f"disc_release_{safe_kind}_%A_%a.err"),
        cluster_args=cluster_args,
        dependency=dependency,
        array=array,
        dry_run=dry_run,
    )
    merge_job = submit_sbatch(
        "hpc_tools/slurm/discrimination/02_discrimination_merge.sbatch",
        project=project,
        export={
            **base_export,
            "RUN_KIND": run_kind,
            "PARTIAL_ROOT": partial_root,
            "OUT_ROOT": out_root,
        },
        profile=merge_profile,
        job_name=f"disc-release-merge-{safe_kind}",
        output=_log_path(log_dir, f"disc_release_merge_{safe_kind}_%j.out"),
        error=_log_path(log_dir, f"disc_release_merge_{safe_kind}_%j.err"),
        cluster_args=cluster_args,
        dependency=array_job,
        dry_run=dry_run,
    )
    return merge_job


def _run_probe(label: str, cmd: list[str]) -> None:
    print(f"\n[{label}]")
    if shutil.which(cmd[0]) is None:
        print(f"{cmd[0]} not found on PATH.")
        return
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        print(f"Could not run {_quote_command(cmd)}: {exc}")
        return
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if proc.returncode != 0:
        print(f"Command exited with status {proc.returncode}.")


def check_slurm(args: argparse.Namespace) -> int:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "$USER"
    print("SLURM preflight for the release pipeline")
    print("Use these outputs to choose --partition, --account, --qos, and resource values.")
    _run_probe("sbatch", ["sbatch", "--version"])
    _run_probe("partitions", ["sinfo", "-s", "-o", "%P %c %m %l %a"])
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
    )
    print("\nCopy/paste template after replacing YOUR_* values:")
    print(
        "python hpc_tools/run_pipeline.py submit --pilot \\\n"
        '  --project "$PWD" \\\n'
        "  --python-module python/3.12.4 \\\n"
        '  --venv-activate "$PWD/venv/bin/activate" \\\n'
        "  --partition YOUR_PARTITION \\\n"
        "  --account YOUR_ACCOUNT \\\n"
        "  --qos YOUR_QOS \\\n"
        f"  --worker-cpus {args.worker_cpus} \\\n"
        f"  --worker-mem {args.worker_mem} \\\n"
        f"  --worker-time {args.worker_time}"
    )
    print("\nOmit --account, --qos, or --partition if your cluster does not require them.")
    print("Remove --pilot for the full NEURON plus 200-MC regeneration.")
    return 0


def submit_full_pipeline(args: argparse.Namespace) -> int:
    project_text = str(args.project).rstrip("/\\")
    project = Path(project_text).expanduser().resolve() if not args.dry_run else REPO_ROOT
    venv_activate = args.venv_activate or f"{project_text}/venv/bin/activate"
    worker_profile = SlurmProfile(
        cpus=int(args.worker_cpus),
        mem=str(args.worker_mem),
        time=str(args.worker_time),
    )
    discrimination_worker_profile = SlurmProfile(
        cpus=1,
        mem=str(args.discrimination_worker_mem),
        time=str(args.discrimination_worker_time),
    )
    discrimination_merge_profile = SlurmProfile(
        cpus=int(args.discrimination_merge_cpus),
        mem=str(args.discrimination_merge_mem),
        time=str(args.discrimination_merge_time),
    )
    coord_profile = SlurmProfile(
        cpus=1,
        mem=str(args.coord_mem),
        time=str(args.coord_time),
    )
    cluster_args = _cluster_args(args)
    pre_n_jobs = int(args.pre_n_jobs or args.worker_cpus)
    detection_array = str(args.detection_array) if args.detection_array else _array_spec_from_cpu_cap(
        int(args.detection_total_tasks),
        cpus_per_task=int(args.worker_cpus),
        max_user_cpus=int(args.max_user_cpus),
        cpu_buffer=int(args.array_cpu_buffer),
    )
    pilot = bool(args.pilot)
    skip_pre_neuron = bool(args.skip_pre_neuron or pilot)
    skip_detection = bool(args.skip_detection or pilot)
    discrimination_mc_total = int(
        args.discrimination_mc_total
        if args.discrimination_mc_total is not None
        else (2 if pilot else 200)
    )
    discrimination_mc_shards = int(
        args.discrimination_mc_shards
        if args.discrimination_mc_shards is not None
        else (1 if pilot else 10)
    )
    discrimination_max_parallel = int(
        args.discrimination_max_parallel
        if args.discrimination_max_parallel is not None
        else (16 if pilot else 500)
    )
    if discrimination_mc_total <= 0:
        raise ValueError("--discrimination-mc-total must be > 0")
    if discrimination_mc_shards <= 0 or discrimination_mc_total % discrimination_mc_shards != 0:
        raise ValueError("--discrimination-mc-total must be divisible by --discrimination-mc-shards")
    log_dir = f"{project_text}/logs"
    if not args.dry_run:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    base_export = {
        "PROJECT": project_text,
        "PYTHON_MODULE": str(args.python_module),
        "VENV_ACTIVATE": str(venv_activate),
    }

    print("Submitting release pipeline to SLURM")
    print(f"  mode: {'pilot' if pilot else 'full'}")
    print(f"  project: {project_text}")
    print(f"  python module: {args.python_module}")
    print(f"  venv activate: {venv_activate}")
    print(f"  NEURON/preprocessing worker profile: cpus={worker_profile.cpus} mem={worker_profile.mem} time={worker_profile.time}")
    print(f"  discrimination worker profile: cpus={discrimination_worker_profile.cpus} mem={discrimination_worker_profile.mem} time={discrimination_worker_profile.time}")
    print(f"  discrimination merge profile: cpus={discrimination_merge_profile.cpus} mem={discrimination_merge_profile.mem} time={discrimination_merge_profile.time}")
    print(f"  coordinator profile: cpus=1 mem={coord_profile.mem} time={coord_profile.time}")
    if cluster_args:
        print(f"  cluster options: {_quote_command(cluster_args)}")
    else:
        print("  cluster options: cluster defaults")
    print(f"  skip pre-neuron: {skip_pre_neuron}")
    print(f"  skip detection: {skip_detection}")
    if not skip_detection:
        print(
            "  detection array: "
            f"{detection_array} (tasks={args.detection_total_tasks}, "
            f"max_user_cpus={args.max_user_cpus}, buffer={args.array_cpu_buffer})"
        )
    print(f"  discrimination MC: total={discrimination_mc_total} shards={discrimination_mc_shards} max_parallel={discrimination_max_parallel}")
    print(f"  dry run: {args.dry_run}")

    submitted_jobs: list[tuple[str, str]] = []
    pre_job: str | None = None
    if not skip_pre_neuron:
        pre_job = submit_sbatch(
            "hpc_tools/slurm/pipeline/01_pre_neuron.sbatch",
            project=project,
            export={**base_export, "PRE_N_JOBS": str(pre_n_jobs), "FORCE_PRE_NEURON": "1"},
            profile=worker_profile,
            job_name="scs-pre-neuron",
            output=_log_path(log_dir, "scs_pre_neuron_%j.out"),
            error=_log_path(log_dir, "scs_pre_neuron_%j.err"),
            cluster_args=cluster_args,
            dry_run=args.dry_run,
        )
        submitted_jobs.append(("pre_neuron", pre_job))

    if skip_detection:
        response_merge_job = pre_job
    else:
        filelist_job = submit_sbatch(
            "hpc_tools/slurm/detection/01_make_filelist.sbatch",
            project=project,
            export=base_export,
            profile=coord_profile,
            job_name="scs-filelist",
            output=_log_path(log_dir, "scs_filelist_%j.out"),
            error=_log_path(log_dir, "scs_filelist_%j.err"),
            cluster_args=cluster_args,
            dependency=pre_job,
            dry_run=args.dry_run,
        )
        threshold_job = submit_sbatch(
            "hpc_tools/slurm/detection/02_thresholds_array.sbatch",
            project=project,
            export=base_export,
            profile=worker_profile,
            job_name="mrg-thresholds",
            output=_log_path(log_dir, "mrg_thresh_%A_%a.out"),
            error=_log_path(log_dir, "mrg_thresh_%A_%a.err"),
            cluster_args=cluster_args,
            dependency=filelist_job,
            array=detection_array,
            dry_run=args.dry_run,
        )
        threshold_merge_job = submit_sbatch(
            "hpc_tools/slurm/detection/03_merge_thresholds_and_cv0.sbatch",
            project=project,
            export=base_export,
            profile=coord_profile,
            job_name="mrg-merge-cv0",
            output=_log_path(log_dir, "mrg_merge_cv0_%j.out"),
            error=_log_path(log_dir, "mrg_merge_cv0_%j.err"),
            cluster_args=cluster_args,
            dependency=threshold_job,
            dry_run=args.dry_run,
        )
        response_job = submit_sbatch(
            "hpc_tools/slurm/detection/04_responses_array.sbatch",
            project=project,
            export=base_export,
            profile=worker_profile,
            job_name="mrg-cv0-responses",
            output=_log_path(log_dir, "mrg_resp_%A_%a.out"),
            error=_log_path(log_dir, "mrg_resp_%A_%a.err"),
            cluster_args=cluster_args,
            dependency=threshold_merge_job,
            array=detection_array,
            dry_run=args.dry_run,
        )
        response_merge_job = submit_sbatch(
            "hpc_tools/slurm/detection/05_merge_responses.sbatch",
            project=project,
            export=base_export,
            profile=coord_profile,
            job_name="mrg-merge-cv0-resp",
            output=_log_path(log_dir, "mrg_merge_resp_%j.out"),
            error=_log_path(log_dir, "mrg_merge_resp_%j.err"),
            cluster_args=cluster_args,
            dependency=response_job,
            dry_run=args.dry_run,
        )
        submitted_jobs.extend([
            ("detection_filelist", filelist_job),
            ("detection_thresholds_and_true_responses", threshold_job),
            ("detection_threshold_true_response_merge", threshold_merge_job),
            ("cv0_response_table", response_job),
            ("cv0_response_table_merge", response_merge_job),
        ])

    per_block_merge_job = _submit_discrimination_run(
        run_kind="per_block",
        project=project,
        base_export=base_export,
        log_dir=log_dir,
        cluster_args=cluster_args,
        worker_profile=discrimination_worker_profile,
        merge_profile=discrimination_merge_profile,
        dependency=response_merge_job,
        mc_total=discrimination_mc_total,
        mc_shards=discrimination_mc_shards,
        max_parallel=discrimination_max_parallel,
        dry_run=args.dry_run,
    )
    per_point_merge_job = _submit_discrimination_run(
        run_kind="per_point",
        project=project,
        base_export=base_export,
        log_dir=log_dir,
        cluster_args=cluster_args,
        worker_profile=discrimination_worker_profile,
        merge_profile=discrimination_merge_profile,
        dependency=response_merge_job,
        mc_total=discrimination_mc_total,
        mc_shards=discrimination_mc_shards,
        max_parallel=discrimination_max_parallel,
        dry_run=args.dry_run,
    )
    finalize_job = submit_sbatch(
        "hpc_tools/slurm/pipeline/06_finalize_discrimination_data.sbatch",
        project=project,
        export={
            **base_export,
            "DISCRIMINATION_MC_TOTAL": str(discrimination_mc_total),
            "RUN_LABEL": "pilot" if pilot else "release",
        },
        profile=coord_profile,
        job_name="scs-finalize-data",
        output=_log_path(log_dir, "scs_finalize_data_%j.out"),
        error=_log_path(log_dir, "scs_finalize_data_%j.err"),
        cluster_args=cluster_args,
        dependency=[per_block_merge_job, per_point_merge_job],
        dry_run=args.dry_run,
    )
    submitted_jobs.extend([
        ("discrimination_per_block_merge", per_block_merge_job),
        ("discrimination_per_point_merge", per_point_merge_job),
        ("finalize_discrimination_metadata", finalize_job),
    ])

    print("\nSubmitted pipeline jobs:")
    for name, job_id in submitted_jobs:
        print(f"  {name}: {job_id}")

    print("\nAfter a full run, retrieve generated outputs from a local checkout with:")
    print("  python hpc_tools/download_data.py --host USER@HOST --remote-project REMOTE_PROJECT")
    print("Then regenerate notebook caches locally and build the release archive with:")
    print("  python src/package_data.py")
    return 0


def worker_pre_neuron(args: argparse.Namespace) -> int:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from space_time import SpaceTimeConfig, run_stage as run_space_time_stage
    from time_patterns import TimePatternConfig, run_stage as run_time_pattern_stage

    n_jobs = int(args.n_jobs)
    force = bool(args.force)
    print(f"[pre-neuron] running stage 01 with force={force}")
    result_01 = run_time_pattern_stage(
        config=TimePatternConfig(parallel=n_jobs != 1),
        force=force,
    )
    print(result_01.summary())

    print(f"[pre-neuron] running stage 02 with n_jobs={n_jobs} force={force}")
    result_02 = run_space_time_stage(
        config=SpaceTimeConfig(n_jobs=n_jobs),
        force=force,
    )
    print(result_02.summary())
    return 0


def _add_resource_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--worker-cpus", type=int, default=24, help="CPUs for heavy worker jobs.")
    parser.add_argument("--worker-mem", default="120G", help="Memory for heavy worker jobs.")
    parser.add_argument("--worker-time", default="10:00:00", help="Wall time for heavy worker jobs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full release data pipeline entrypoint. Notebooks are easy-mode only."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check-slurm", help="Print SLURM resources and a submit-command template.")
    _add_resource_args(check)
    check.set_defaults(func=check_slurm)

    submit = sub.add_parser("submit", help="Submit the full release data pipeline to SLURM.")
    submit.add_argument("--project", default=str(REPO_ROOT), help="Project root on the HPC filesystem.")
    submit.add_argument("--python-module", default="python/3.12.4", help="Environment module loaded by SLURM jobs.")
    submit.add_argument("--venv-activate", default=None, help="Path to the HPC virtualenv activate script.")
    submit.add_argument("--partition", default=None, help="SLURM partition to pass to sbatch.")
    submit.add_argument("--account", default=None, help="SLURM account to pass to sbatch.")
    submit.add_argument("--qos", default=None, help="SLURM QOS to pass to sbatch.")
    submit.add_argument("--constraint", default=None, help="Optional SLURM node constraint.")
    submit.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    submit.add_argument(
        "--pilot",
        action="store_true",
        help="Run a small discrimination-only pilot using existing data/detection outputs.",
    )
    submit.add_argument("--skip-pre-neuron", action="store_true", help="Skip stage 01/02 generation.")
    submit.add_argument("--skip-detection", action="store_true", help="Skip stage 03 NEURON detection jobs.")
    _add_resource_args(submit)
    submit.add_argument("--coord-mem", default="16G", help="Memory for lightweight merge/publish jobs.")
    submit.add_argument("--coord-time", default="02:00:00", help="Wall time for lightweight merge/publish jobs.")
    submit.add_argument("--pre-n-jobs", type=int, default=None, help="Workers for the stage 01/02 pre-NEURON job; defaults to --worker-cpus.")
    submit.add_argument("--max-user-cpus", type=int, default=1000, help="CPU cap used to derive array concurrency defaults.")
    submit.add_argument("--array-cpu-buffer", type=int, default=40, help="CPU headroom subtracted from --max-user-cpus when deriving array concurrency; default reproduces the old 0-54%%40 detection array with 24-cpu workers and a 1000-cpu cap.")
    submit.add_argument("--detection-total-tasks", type=int, default=55, help="Number of stage-03 detection array tasks; defaults to 11 CVs x 5 frequencies.")
    submit.add_argument("--detection-array", default=None, help="Optional explicit SLURM array spec for detection jobs; overrides --detection-total-tasks and --max-user-cpus.")
    submit.add_argument("--discrimination-worker-mem", default="16G", help="Memory for discrimination array jobs.")
    submit.add_argument("--discrimination-worker-time", default="10:00:00", help="Wall time for discrimination array jobs.")
    submit.add_argument("--discrimination-merge-cpus", type=int, default=8, help="CPUs for discrimination merge jobs.")
    submit.add_argument("--discrimination-merge-mem", default="64G", help="Memory for discrimination merge jobs.")
    submit.add_argument("--discrimination-merge-time", default="04:00:00", help="Wall time for discrimination merge jobs.")
    submit.add_argument("--discrimination-mc-total", type=int, default=None, help="Total MC splits for discrimination fits; defaults to 200, or 2 with --pilot.")
    submit.add_argument("--discrimination-max-parallel", type=int, default=None, help="Max concurrent discrimination array tasks; defaults to 500, or 16 with --pilot.")
    submit.add_argument("--discrimination-mc-shards", type=int, default=None, help="MC shards for discrimination fits; defaults to 10, or 1 with --pilot.")
    submit.set_defaults(func=submit_full_pipeline)

    worker = sub.add_parser("worker-pre-neuron", help=argparse.SUPPRESS)
    worker.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    worker.add_argument("--force", action="store_true")
    worker.set_defaults(func=worker_pre_neuron)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
