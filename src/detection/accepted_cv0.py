"""Reconstruct the submitted CV0 operating amplitudes from archived inputs.

The accepted observer analysis used the v1 CV0 branch. Its full 49,500-row
pre-Imax regime table is distributed as a fitting input, so the five CV0
operating states can be rebuilt without copying the accepted response output.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from detection.protocol import CV0_PROTOCOL_VERSION


ACCEPTED_PROTOCOL = CV0_PROTOCOL_VERSION
REGIMES = (
    "I_first", "I_volley5", "I_volley10", "I_5", "I_10", "I_15",
    "I_25", "I_50", "I_all",
)
FREQUENCIES = (10.0, 20.0, 50.0, 100.0, 200.0)
CVS = tuple(round(0.1 * index, 1) for index in range(11))
REPETITIONS = tuple(range(100))
DEFAULT_PARTIAL_DIR = Path("logs/accepted-cv0-10hz/partials")
DEFAULT_MERGED = Path("logs/accepted-cv0-10hz/population_cv0_based_responses.csv")
DEFAULT_INDEX = Path("logs/accepted-cv0-10hz/index_cv0_threshold_max_accepted.csv")
SOURCE_SHA256 = "d24c1cee7ee82280e2fe3b018b39de6affdb545b14ebddd63e5775451dc4ac2a"
INDEX_SHA256 = "4d91d6332457bb60ca2ca086b8a96dc6252769dea292777f2ebcab99410fd390"


def checked_csv(path: Path, expected_hash: str, *, normalize_newlines: bool = False) -> pd.DataFrame:
    contents = path.read_bytes()
    if normalize_newlines:
        contents = contents.replace(b"\r\n", b"\n")
    if sha256(contents).hexdigest() != expected_hash:
        raise ValueError(f"Missing or modified accepted CV0 input: {path}")
    return pd.read_csv(path)


def build_cv0_index_table(regimes: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Return the accepted five-row CV0 table after checking all source trials."""
    required = {"freq", "cv", "rep_index", "thresh_type", "amp_uA", "n_index_spikes"}
    if not required.issubset(regimes.columns):
        raise ValueError(f"Pre-Imax regime input lacks {sorted(required - set(regimes.columns))}")
    keys = ["freq", "cv", "rep_index", "thresh_type"]
    if regimes.duplicated(keys).any():
        raise ValueError("Pre-Imax regime input contains duplicate trial/regime keys")
    expected = len(FREQUENCIES) * len(CVS) * len(REPETITIONS) * len(REGIMES)
    if len(regimes) != expected:
        raise ValueError(f"Expected {expected:,} pre-Imax rows, found {len(regimes):,}")
    if (set(regimes["freq"].astype(float)) != set(FREQUENCIES)
            or set(regimes["cv"].astype(float)) != set(CVS)
            or set(regimes["rep_index"].astype(int)) != set(REPETITIONS)
            or set(regimes["thresh_type"].astype(str)) != set(REGIMES)):
        raise ValueError("Pre-Imax trial grid differs from the accepted design")
    if len(index) != 1 or str(index.iloc[0]["axon_id"]) != "S01_140um_ph1":
        raise ValueError("Expected the accepted S01_140um_ph1 index axon")
    identity = index.iloc[0]
    rows: list[dict[str, object]] = []
    cv0 = regimes.loc[np.isclose(regimes["cv"].astype(float), 0.0)]
    for freq in FREQUENCIES:
        row: dict[str, object] = {
            "detection_protocol_version": ACCEPTED_PROTOCOL,
            "index_axon_id": str(identity["axon_id"]),
            "index_cutline_id": str(identity["cutline_id"]),
            "index_diameter_um": float(identity["diameter_um"]),
            "freq": freq,
        }
        for regime in REGIMES:
            matching = cv0.loc[
                np.isclose(cv0["freq"].astype(float), freq)
                & cv0["thresh_type"].eq(regime)
            ]
            if len(matching) != len(REPETITIONS):
                raise ValueError(f"Expected 100 CV0 repetitions for {freq:g} Hz {regime}")
            amplitudes = matching["amp_uA"].astype(float).to_numpy()
            counts = matching["n_index_spikes"].astype(int).to_numpy()
            if not np.isfinite(amplitudes).all() or np.ptp(amplitudes) != 0 or np.ptp(counts) != 0:
                raise ValueError(f"CV0 repetitions disagree for {freq:g} Hz {regime}")
            stem = f"{regime}0" if regime in {"I_first", "I_all"} else f"{regime}_0"
            row[f"{stem}_uA"] = float(amplitudes[0])
            row[f"{stem}_n_index_spikes"] = int(counts[0])
        row["N_max_spikes"] = row["I_all0_n_index_spikes"]
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_accepted_index(root: Path, output: Path) -> Path:
    """Write the accepted five-row CV0 table from the archived fit input."""
    source = root / "data/detection/behavior_fit_inputs/index_regime_amplitudes_pre_imax.csv"
    index = root / "data/detection/index_axon.csv"
    table = build_cv0_index_table(checked_csv(source, SOURCE_SHA256), pd.read_csv(index))
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    if sha256(output.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != INDEX_SHA256:
        raise ValueError("Reconstructed CV0 index table differs from the accepted table")
    return output


def simulate_pilot(root: Path, *, axon_id: str, cv: float, rep_index: int) -> dict[str, object]:
    """Run one accepted 10 Hz I_all response with the existing MRG model."""
    from detection import detection_core as core
    from space_time_v.spacetime_core import spacetime_path

    source = root / "data/detection/behavior_fit_inputs/index_regime_amplitudes_pre_imax.csv"
    index = root / "data/detection/index_axon.csv"
    states = build_cv0_index_table(checked_csv(source, SOURCE_SHA256), pd.read_csv(index))
    old_amp = float(states.loc[states["freq"].eq(10.0), "I_all0_uA"].iloc[0])
    basis_path = root / "data/space-time-V" / f"{axon_id}_1uA_VeBasis.npz"
    basis = core.load_population_basis(basis_path)
    source_path = spacetime_path(
        root / "data/space-time-V", basis["cutline_id"], basis["diameter_um"],
        cv, 10.0, axon_id=axon_id,
    )
    rep = core.load_population_rep(source_path, rep_index)
    time_ms, voltage = core.trim_tail_window_ms(rep["time_ms"], rep["V_base"], rep["current"])
    time_ms, voltage = core.prepend_zeros_ms(time_ms, voltage, prepad_ms=core.PREPAD_MS)
    runner = core.PopulationMRGRunner(
        time_ms, voltage,
        config_path=core._config_path(root, rep["diameter_um"], rep["n_valid_nodes"]),
        readout_node_number=rep["readout_node_number"],
    )
    spike_times, n_spikes = runner.run_mrg_for_amp(old_amp)
    return {
        "protocol": ACCEPTED_PROTOCOL,
        "axon_id": axon_id,
        "cv": cv,
        "freq": 10.0,
        "rep_index": rep_index,
        "thresh_type": "I_all",
        "amp_uA": old_amp,
        "n_spikes": n_spikes,
        "spike_times_ms": np.asarray(spike_times, dtype=float).tolist(),
    }


def trial_regimes(root: Path, cv: float) -> pd.DataFrame:
    """Expand the recovered v1 I_all amplitude over one 10 Hz condition."""
    from detection import detection_core as core
    from space_time_v.spacetime_core import spacetime_path

    states = checked_csv(
        root / DEFAULT_INDEX,
        INDEX_SHA256, normalize_newlines=True,
    )
    if len(states) != len(FREQUENCIES) or set(states["freq"]) != set(FREQUENCIES):
        raise ValueError("Accepted CV0 index table is missing frequencies")
    state = states.loc[states["freq"].eq(10.0)].iloc[0]
    if state["detection_protocol_version"] != ACCEPTED_PROTOCOL:
        raise ValueError("Accepted CV0 index table has the wrong protocol")
    index = core.identify_index_axon(root=root, n_jobs=1, force=False)
    index_path = spacetime_path(
        root / "data/space-time-V", str(index["cutline_id"]),
        float(index["diameter_um"]), cv, 10.0, axon_id=str(index["axon_id"]),
    )
    with np.load(index_path, allow_pickle=False) as data:
        labels = [str(label) for label in data["rep_labels"]]
    if len(labels) != len(REPETITIONS):
        raise ValueError(f"Expected 100 repetitions in {index_path}")
    rows = []
    for rep_index, rep_label in enumerate(labels):
        rows.append({
            "detection_protocol_version": ACCEPTED_PROTOCOL,
            "index_axon_id": index["axon_id"],
            "index_cutline_id": index["cutline_id"],
            "index_diameter_um": index["diameter_um"],
            "file": index_path.name,
            "cv": cv,
            "freq": 10.0,
            "rep_index": rep_index,
            "rep_label": rep_label,
            "n_input_pulses": 20,
            "thresh_type": "I_all",
            "amp_uA": float(state["I_all0_uA"]),
            "n_index_spikes": np.nan,
            "index_spike_times_ms": "[]",
            "N_max_spikes": int(state["N_max_spikes"]),
        })
    return pd.DataFrame(rows, columns=core.REGIME_COLUMNS)


def run_target(root: Path, *, axon_id: str, cv: float, n_jobs: int,
               partial_dir: Path, diameter_screen: Path | None = None,
               overwrite: bool = False) -> Path:
    """Simulate the old 10 Hz I_all amplitude for one target and 100 trials."""
    from detection import detection_core as core

    basis = core.load_population_basis(
        root / "data/space-time-V" / f"{axon_id}_1uA_VeBasis.npz"
    )
    job = {
        "target_axon_id": basis["axon_id"],
        "target_cutline_id": basis["cutline_id"],
        "target_diameter_um": basis["diameter_um"],
        "basis_path": str(basis["basis_path"]),
        "cv": cv,
        "freq": 10.0,
    }
    return core._run_population_recruitment_condition(
        root=root, mode="cv0", job=job, n_jobs=n_jobs,
        partial_dir=partial_dir, overwrite=overwrite,
        prune_on_max_failure=True, max_pruning_audit_rate=0.01,
        diameter_screen_path=diameter_screen,
        diameter_pruning_audit_rate=0.01,
        regimes_override=trial_regimes(root, cv),
    )


def trajectory_jobs(root: Path, population_manifest: Path | None = None) -> list[dict]:
    """Return the 70 phase/trajectory groups by 11 CVs affected at 10 Hz."""
    from detection import detection_core as core

    jobs = [job for job in core.population_trajectory_condition_jobs(root, population_manifest)
            if float(job["freq"]) == 10.0]
    if len(jobs) != 770 or any(len(job["targets"]) != 6 for job in jobs):
        raise ValueError(f"Expected 770 six-axon 10 Hz trajectory jobs, found {len(jobs)}")
    return jobs


def run_trajectory(root: Path, *, task_index: int, n_jobs: int,
                   partial_dir: Path, population_manifest: Path | None = None,
                   overwrite: bool = False) -> list[Path]:
    """Run six geometries in fresh processes for one trajectory/CV task."""
    from detection import detection_core as core

    jobs = trajectory_jobs(root, population_manifest)
    if not 0 <= task_index < len(jobs):
        raise IndexError(f"task_index must be in [0, {len(jobs) - 1}]")
    job = jobs[task_index]
    outputs = []
    previous: Path | None = None
    for target in job["targets"]:
        command = [
            sys.executable, "-m", "detection.accepted_cv0", "target",
            "--root", str(root), "--axon-id", str(target["target_axon_id"]),
            "--cv", str(job["cv"]), "--n-jobs", str(n_jobs),
            "--partial-dir", str(partial_dir),
        ]
        if previous is not None:
            command.extend(("--diameter-screen", str(previous)))
        if overwrite:
            command.append("--overwrite")
        subprocess.run(command, cwd=root, check=True)
        previous = core._population_partial_path(
            partial_dir, "cv0", {**target, "cv": job["cv"], "freq": 10.0}
        )
        outputs.append(previous)
    return outputs


def load_replacements(root: Path, partial_dir: Path) -> dict[tuple[str, str, str], tuple[str, str, str]]:
    """Check all 462,000 old-amplitude responses before touching the baseline."""
    from detection import detection_core as core

    basis_paths = core._population_basis_paths(root)
    if len(basis_paths) != 420:
        raise ValueError(f"Expected 420 population targets, found {len(basis_paths)}")
    states = checked_csv(
        root / DEFAULT_INDEX,
        INDEX_SHA256, normalize_newlines=True,
    )
    if len(states) != 5 or states.iloc[0]["detection_protocol_version"] != ACCEPTED_PROTOCOL:
        raise ValueError("Accepted CV0 index table is incomplete or has the wrong protocol")
    old_amp = float(states.loc[states["freq"].eq(10.0), "I_all0_uA"].iloc[0])
    replacements: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for basis_path in basis_paths:
        axon_id = basis_path.name.removesuffix("_1uA_VeBasis.npz")
        for cv in CVS:
            cv_text = f"{cv:.1f}"
            part = partial_dir / f"population_cv0_based_responses_{axon_id}_cv{cv_text}_f10.csv"
            if not part.is_file():
                raise FileNotFoundError(f"Accepted CV0 response partial is missing: {part}")
            with part.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != list(core.POPULATION_RESPONSE_COLUMNS):
                    raise ValueError(f"Accepted CV0 partial has an unexpected header: {part}")
                rows = list(reader)
            if len(rows) != 100:
                raise ValueError(f"Expected 100 accepted CV0 rows in {part}, found {len(rows)}")
            if {int(row["rep_index"]) for row in rows} != set(REPETITIONS):
                raise ValueError(f"Accepted CV0 partial has an incomplete repetition grid: {part}")
            for row in rows:
                if (
                    row["detection_protocol_version"] != ACCEPTED_PROTOCOL
                    or row["target_axon_id"] != axon_id
                    or float(row["cv"]) != cv
                    or float(row["freq"]) != 10.0
                    or row["thresh_type"] != "I_all"
                    or float(row["amp_uA"]) != old_amp
                ):
                    raise ValueError(f"Accepted CV0 partial has wrong metadata: {part}")
                key = (axon_id, cv_text, row["rep_index"])
                if key in replacements:
                    raise ValueError(f"Duplicate accepted CV0 response {key}")
                replacements[key] = (
                    row["amp_uA"], row["n_spikes"], row["spike_times_ms"],
                )
    expected = 420 * len(CVS) * len(REPETITIONS)
    if len(replacements) != expected:
        raise ValueError(f"Expected {expected:,} old-amplitude responses, found {len(replacements):,}")
    return replacements


def patch_response_rows(
    reader: csv.DictReader, writer: csv.DictWriter,
    replacements: dict[tuple[str, str, str], tuple[str, str, str]],
    *, index_id: str, baseline_protocol: str,
) -> tuple[int, int]:
    """Rewrite response metadata and replace only old-amplitude 10 Hz rows."""
    seen: set[tuple[str, str, str]] = set()
    n_rows = 0
    for row in reader:
        n_rows += 1
        if row["detection_protocol_version"] != baseline_protocol:
            raise ValueError(f"Fresh baseline has a mixed protocol at row {n_rows:,}")
        if float(row["freq"]) == 10.0 and row["thresh_type"] == "I_all":
            key = (row["target_axon_id"], f"{float(row['cv']):.1f}", row["rep_index"])
            replacement = replacements.get(key)
            if replacement is None or key in seen:
                raise ValueError(f"Missing or repeated replacement for {key}")
            seen.add(key)
            row["amp_uA"], row["n_spikes"], row["spike_times_ms"] = replacement
        row["detection_protocol_version"] = ACCEPTED_PROTOCOL
        row["n_index_spikes"] = row["n_spikes"] if row["target_axon_id"] == index_id else ""
        writer.writerow(row)
    if len(seen) != len(replacements):
        raise ValueError(f"Only {len(seen):,} of {len(replacements):,} replacements were used")
    return n_rows, len(seen)


def merge_recovered_responses(root: Path, *, partial_dir: Path, output: Path) -> dict[str, object]:
    """Splice independently simulated v1 I_all rows into the fresh CV0 branch."""
    from detection import detection_core as core

    replacements = load_replacements(root, partial_dir)
    baseline = root / "data/detection/population_cv0_based_responses.csv"
    if baseline.resolve() == output.resolve():
        raise ValueError("Recovery output must differ from the fresh baseline")
    expected_protocol = core.DETECTION_PROTOCOL_VERSION
    index_id = str(pd.read_csv(root / "data/detection/index_axon.csv").iloc[0]["axon_id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    n_rows = 0
    n_replaced = 0
    try:
        with baseline.open(newline="", encoding="utf-8") as source, temporary.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(core.POPULATION_RESPONSE_COLUMNS):
                raise ValueError("Fresh CV0 response table has an unexpected header")
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            n_rows, n_replaced = patch_response_rows(
                reader, writer, replacements,
                index_id=index_id, baseline_protocol=expected_protocol,
            )
        if n_rows != 420 * len(FREQUENCIES) * len(CVS) * len(REPETITIONS) * len(REGIMES):
            raise ValueError(f"Fresh baseline has the wrong row count: {n_rows:,}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "status": "ready",
        "protocol": ACCEPTED_PROTOCOL,
        "baseline_protocol": expected_protocol,
        "baseline_rows": n_rows,
        "replaced_10hz_I_all_rows": n_replaced,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Rebuild the accepted CV0 index table")
    prepare.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    prepare.add_argument("--output", type=Path,
                         help="Defaults to logs/accepted-cv0-10hz/index_cv0_threshold_max_accepted.csv")
    pilot = commands.add_parser("pilot", help="Simulate one old-amplitude response on Longleaf")
    pilot.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    pilot.add_argument("--axon-id", default="S01_140um_ph1")
    pilot.add_argument("--cv", type=float, default=0.1)
    pilot.add_argument("--rep-index", type=int, default=15)
    pilot.add_argument("--output", type=Path)
    target = commands.add_parser("target", help="Simulate one target axon at old I_all")
    target.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    target.add_argument("--axon-id", required=True)
    target.add_argument("--cv", type=float, required=True)
    target.add_argument("--n-jobs", type=int, default=1)
    target.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    target.add_argument("--diameter-screen", type=Path)
    target.add_argument("--overwrite", action="store_true")
    trajectory = commands.add_parser("trajectory", help="Run one 10 Hz trajectory/CV task")
    trajectory.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    trajectory.add_argument("--task-index", type=int, required=True)
    trajectory.add_argument("--n-jobs", type=int, default=24)
    trajectory.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    trajectory.add_argument("--population-manifest", type=Path)
    trajectory.add_argument("--overwrite", action="store_true")
    merge = commands.add_parser("merge", help="Merge old 10 Hz responses into the fresh CV0 branch")
    merge.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    merge.add_argument("--partial-dir", type=Path, default=DEFAULT_PARTIAL_DIR)
    merge.add_argument("--output", type=Path, default=DEFAULT_MERGED)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "merge":
        partial_dir = args.partial_dir if args.partial_dir.is_absolute() else root / args.partial_dir
        output = args.output if args.output.is_absolute() else root / args.output
        print(json.dumps(merge_recovered_responses(
            root, partial_dir=partial_dir.resolve(), output=output.resolve()
        ), indent=2))
        return
    if args.command in {"target", "trajectory"}:
        partial_dir = args.partial_dir
        if not partial_dir.is_absolute():
            partial_dir = root / partial_dir
        partial_dir = partial_dir.resolve()
        partial_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "target":
            output = run_target(
                root, axon_id=args.axon_id, cv=args.cv, n_jobs=args.n_jobs,
                partial_dir=partial_dir, diameter_screen=args.diameter_screen,
                overwrite=args.overwrite,
            )
            print(output)
        else:
            outputs = run_trajectory(
                root, task_index=args.task_index, n_jobs=args.n_jobs,
                partial_dir=partial_dir, population_manifest=args.population_manifest,
                overwrite=args.overwrite,
            )
            print(f"Completed {len(outputs)} accepted-CV0 target partials")
        return
    if args.command == "pilot":
        result = simulate_pilot(root, axon_id=args.axon_id, cv=args.cv,
                                rep_index=args.rep_index)
        output = args.output or root / "logs/accepted_cv0_pilot.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    output = args.output or root / DEFAULT_INDEX
    print(f"Reconstructed 5 accepted CV0 operating states: {prepare_accepted_index(root, output)}")


if __name__ == "__main__":
    main()
