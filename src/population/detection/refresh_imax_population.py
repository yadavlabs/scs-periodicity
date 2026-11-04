"""Targeted, auditable population refresh after index I_max promotion.

prepare scans the complete true-response table. run-axon only simulates changed
amplitudes (no monotonicity pruning). merge validates every row and stages a new
table. Production replacement is explicit and retains a recoverable backup.
CV0 responses are deliberately not relabeled or modified by this utility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from population.protocol import DETECTION_PROTOCOL_VERSION

KEY = ["freq", "cv", "rep_index", "thresh_type"]
DEFAULT = "results/detection/imax_population_refresh"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paths(root):
    data = root / "data/detection/population"
    return data / "index_regime_amplitudes.csv", data / "population_responses.csv"


def aligned_positions(frame, index):
    positions = index.index.get_indexer(pd.MultiIndex.from_frame(frame[KEY]))
    if (positions < 0).any():
        raise ValueError("Response table contains unknown index-regime keys")
    return positions


def prepare(root, output):
    index_path, source = paths(root)
    index = pd.read_csv(index_path).set_index(KEY, verify_integrity=True)
    if set(index.detection_protocol_version) != {DETECTION_PROTOCOL_VERSION}:
        raise ValueError("Promote the validated peak-first index table first")
    ids = sorted(pd.read_csv(root / "data/space-time-V/axon_basis_manifest.csv").axon_id)
    id_map = {v: i for i, v in enumerate(ids)}
    seen = np.zeros((len(ids), len(index)), dtype=bool)
    changed = np.zeros(len(index), dtype=bool)
    changed_rows = 0
    with pd.read_csv(source, usecols=KEY + ["target_axon_id", "amp_uA"], chunksize=200000) as chunks:
        for chunk in chunks:
            pos = aligned_positions(chunk, index)
            target = chunk.target_axon_id.map(id_map)
            if target.isna().any():
                raise ValueError("Unknown target axon")
            target = target.to_numpy(int)
            flat = target * len(index) + pos
            if len(np.unique(flat)) != len(flat) or seen[target, pos].any():
                raise ValueError("Duplicate response keys")
            seen[target, pos] = True
            mismatch = ~np.isclose(chunk.amp_uA.to_numpy(), index.amp_uA.to_numpy()[pos], atol=1e-9, rtol=0)
            if chunk.loc[mismatch, "thresh_type"].ne("I_all").any():
                raise ValueError("Non-I_max amplitudes changed; outside targeted refresh scope")
            changed[pos[mismatch]] = True
            changed_rows += int(mismatch.sum())
    if not seen.all():
        raise ValueError("Incomplete source population responses")
    output.mkdir(parents=True, exist_ok=True)
    index.iloc[np.flatnonzero(changed)].reset_index().to_csv(output / "changed_regimes.csv", index=False)
    pd.DataFrame({"axon_index": range(len(ids)), "target_axon_id": ids}).to_csv(output / "axons.csv", index=False)
    report = dict(status="refresh_required" if changed.any() else "current", source=str(source),
                  source_sha256=digest(source), index_sha256=digest(index_path),
                  target_protocol=DETECTION_PROTOCOL_VERSION, source_rows=int(seen.sum()),
                  changed_regime_rows=int(changed.sum()), mismatched_population_rows=changed_rows,
                  axons=len(ids), maximum_simulations=int(changed.sum()) * len(ids),
                  discrimination_cv0_20_to_200_unchanged=True)
    # The CV0 assertion is checked numerically against the actual feature input.
    features = pd.read_csv(root / "data/features/population_cv0_based_features.csv",
                           usecols=KEY + ["amp_uA"])
    anchors = index.reset_index().query("cv == 0").groupby(["freq", "thresh_type"]).amp_uA.first()
    f = features[features.freq.isin([20, 50, 100, 200])]
    target_amp = anchors.reindex(pd.MultiIndex.from_frame(f[["freq", "thresh_type"]])).to_numpy()
    report["discrimination_cv0_20_to_200_unchanged"] = bool(np.allclose(f.amp_uA, target_amp, atol=1e-9, rtol=0))
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_axon(root, output, axon_index, n_jobs):
    from population.detection.detection_core import _population_response_rows_for_rep, _atomic_dataframe_csv
    audit = json.loads((output / "audit.json").read_text())
    if digest(paths(root)[0]) != audit["index_sha256"]:
        raise ValueError("Index data changed since preparation")
    axon = pd.read_csv(output / "axons.csv").set_index("axon_index").loc[axon_index, "target_axon_id"]
    part = output / "partials" / f"{axon}.csv"
    regimes = pd.read_csv(output / "changed_regimes.csv")
    if part.exists():
        existing = pd.read_csv(part)
        validate_replacements(existing, regimes, axon)
        return str(part)
    basis = root / "data/space-time-V" / f"{axon}_1uA_VeBasis.npz"
    jobs = [(dict(basis_path=str(basis), freq=row["freq"], cv=row["cv"]), [row])
            for row in regimes.to_dict("records")]
    values = Parallel(n_jobs=n_jobs)(delayed(_population_response_rows_for_rep)(
        root, job, rows, prune_on_max_failure=False) for job, rows in jobs)
    result = pd.DataFrame([row for rows, _ in values for row in rows])
    validate_replacements(result, regimes, axon)
    part.parent.mkdir(parents=True, exist_ok=True)
    _atomic_dataframe_csv(result, part)
    return str(part)


def validate_replacements(frame, regimes, axon):
    ref = regimes.set_index(KEY, verify_integrity=True)
    if len(frame) != len(ref) or frame.duplicated(KEY).any() or not frame.target_axon_id.eq(axon).all():
        raise ValueError("Incomplete/duplicate/wrong-axon replacement rows")
    pos = aligned_positions(frame, ref)
    if not np.allclose(frame.amp_uA, ref.amp_uA.to_numpy()[pos], atol=1e-9, rtol=0):
        raise ValueError("Stale replacement amplitudes")
    if not frame.detection_protocol_version.eq(DETECTION_PROTOCOL_VERSION).all():
        raise ValueError("Stale replacement protocol")
    for row in frame.itertuples():
        spikes = np.asarray(json.loads(row.spike_times_ms), float)
        if len(spikes) != row.n_spikes or not np.isfinite(spikes).all() or (np.diff(spikes) < 0).any():
            raise ValueError("Invalid spike times/count")


def merge(root, output, promote=False):
    from population.detection.detection_core import _atomic_dataframe_csv
    index_path, source = paths(root)
    audit = json.loads((output / "audit.json").read_text())
    if digest(index_path) != audit["index_sha256"] or digest(source) != audit["source_sha256"]:
        raise ValueError("Source/index changed since preparation; prepare again")
    index = pd.read_csv(index_path).set_index(KEY, verify_integrity=True)
    regimes = pd.read_csv(output / "changed_regimes.csv")
    parts = []
    for axon in pd.read_csv(output / "axons.csv").target_axon_id:
        part = pd.read_csv(output / "partials" / f"{axon}.csv")
        validate_replacements(part, regimes, axon)
        parts.append(part)
    replacements = pd.concat(parts, ignore_index=True).set_index(["target_axon_id"] + KEY, verify_integrity=True)
    staged = output / "population_responses.csv"
    temporary = staged.with_suffix(".csv.tmp")
    totals = None
    axon_summaries = []
    matched, total = 0, 0
    for chunk in pd.read_csv(source, chunksize=100000):
        keys = pd.MultiIndex.from_frame(chunk[["target_axon_id"] + KEY])
        loc = replacements.index.get_indexer(keys)
        mask = loc >= 0
        if mask.any():
            chunk.loc[mask, replacements.columns] = replacements.iloc[loc[mask]].to_numpy()
            matched += int(mask.sum())
        pos = aligned_positions(chunk, index)
        if not np.allclose(chunk.amp_uA, index.amp_uA.to_numpy()[pos], atol=1e-9, rtol=0):
            raise ValueError("Unrepaired population amplitude")
        if "index_axon_id" in chunk:
            index_rows = chunk.target_axon_id.eq(chunk.index_axon_id).to_numpy()
            if not np.array_equal(chunk.n_spikes.to_numpy()[index_rows], index.n_index_spikes.to_numpy()[pos[index_rows]]):
                raise ValueError("Actual index-axon response disagrees with promoted index counts")
        chunk["n_index_spikes"] = index.n_index_spikes.to_numpy()[pos]
        chunk["detection_protocol_version"] = DETECTION_PROTOCOL_VERSION
        chunk.to_csv(temporary, mode="w" if total == 0 else "a", header=total == 0, index=False)
        total += len(chunk)
        local = chunk.assign(active_axon_count=chunk.n_spikes.gt(0).astype(int)).groupby(KEY)[["n_spikes", "active_axon_count"]].sum()
        totals = local if totals is None else totals.add(local, fill_value=0)
        axon_summaries.append(chunk.groupby(["target_axon_id", "freq", "cv", "thresh_type"]).n_spikes.agg(["min", "max"]))
    if total != audit["source_rows"] or matched != len(replacements):
        raise ValueError("Refresh coverage mismatch")
    temporary.replace(staged)
    trial_summary = totals.rename(columns={"n_spikes": "total_population_spikes"}).astype(int).reset_index()
    _atomic_dataframe_csv(trial_summary, output / "true_population_trial_summary.csv")
    per_axon = pd.concat(axon_summaries).groupby(level=[0, 1, 2, 3]).agg({"min": "min", "max": "max"})
    per_axon["ever"] = per_axon["max"].gt(0).astype(int)
    per_axon["always"] = per_axon["min"].gt(0).astype(int)
    cohort = per_axon.groupby(level=[1, 2, 3]).agg(simulated_axons=("ever", "size"),
                axons_ever_recruited=("ever", "sum"), axons_always_recruited=("always", "sum"))
    means = trial_summary.groupby(["freq", "cv", "thresh_type"])[["total_population_spikes", "active_axon_count"]].mean()
    cohort = cohort.join(means).rename(columns={"total_population_spikes": "expected_total_spikes", "active_axon_count": "expected_active_axons"})
    cohort["expected_active_fraction"] = cohort.expected_active_axons / cohort.simulated_axons
    cohort = cohort.reset_index().assign(mode="true")
    aggregate_source = root / "data/detection/population/population_recruitment_outcomes.csv"
    if aggregate_source.exists():
        previous = pd.read_csv(aggregate_source)
        retained = previous[~previous["mode"].astype(str).str.lower().eq("true")]
        cohort = pd.concat([cohort, retained], ignore_index=True).reindex(columns=previous.columns)
    _atomic_dataframe_csv(cohort, output / "population_recruitment_outcomes.csv")
    report = dict(status="staged", rows=total, replaced_rows=matched, index_sha256=audit["index_sha256"],
                  staged_sha256=digest(staged), protocol=DETECTION_PROTOCOL_VERSION)
    if promote:
        backup = output / "population_responses_pre_refresh.csv"
        aggregate_backup = output / "population_recruitment_outcomes_pre_refresh.csv"
        if backup.exists() or aggregate_backup.exists():
            raise FileExistsError("Preserving existing production backup; refusing overwrite")
        shutil.copy2(source, backup)
        # Retain staged data as well as the old production backup.
        production_tmp = source.with_suffix(".refresh.tmp")
        shutil.copy2(staged, production_tmp)
        production_tmp.replace(source)
        if aggregate_source.exists():
            shutil.copy2(aggregate_source, aggregate_backup)
            _atomic_dataframe_csv(cohort, aggregate_source)
        report.update(status="promoted", backup=str(backup))
    (output / "merge_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run-axon", "merge"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--axon-index", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir or root / DEFAULT
    result = prepare(root, output) if args.command == "prepare" else (
        run_axon(root, output, args.axon_index, args.n_jobs) if args.command == "run-axon"
        else merge(root, output, args.promote))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
