"""Validate the phase-aware population cohort before a Longleaf upload."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from population.detection.detection_core import population_trajectory_condition_jobs
from population.space_time_v.spacetime_core import (
    POPULATION_CUTLINE_IDS,
    POPULATION_CV_VALS,
    POPULATION_DIAMETERS_UM,
    POPULATION_FREQ_VALS_HZ,
    SPACETIME_NPZ_FIELDS,
    discover_comsol_cutline_ids,
    population_axon_jobs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INDEX_AXON_ID = "S01_140um_ph1"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_preflight(root: Path, expected_index: str) -> dict[str, object]:
    root = root.resolve()
    comsol_dir = root / "data" / "comsol"
    observed_cutlines = discover_comsol_cutline_ids(comsol_dir)
    if set(observed_cutlines) != set(POPULATION_CUTLINE_IDS):
        missing = sorted(set(POPULATION_CUTLINE_IDS) - set(observed_cutlines))
        extra = sorted(set(observed_cutlines) - set(POPULATION_CUTLINE_IDS))
        raise RuntimeError(f"COMSOL cohort mismatch; missing={missing}, extra={extra}")

    time_dir = root / "data" / "time-patterns" / "population"
    time_paths = [
        time_dir / f"cv{cv:.1f}_f{int(freq)}_waveforms.npz"
        for cv in POPULATION_CV_VALS for freq in POPULATION_FREQ_VALS_HZ
    ]
    missing_time = [str(path) for path in time_paths if not path.is_file()]
    if missing_time:
        raise FileNotFoundError(f"Missing {len(missing_time)} population time-pattern files; first={missing_time[0]}")
    for path in time_paths:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise RuntimeError(f"Time-pattern archive is empty: {path}")

    basis_dir = root / "data" / "space-time-V"
    manifest = basis_dir / "axon_basis_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Population basis manifest is missing: {manifest}")
    rows = _read_rows(manifest)
    expected_jobs = population_axon_jobs(root)
    expected_ids = {str(job["axon_id"]) for job in expected_jobs}
    observed_ids = {row["axon_id"] for row in rows}
    if len(rows) != 420 or len(observed_ids) != 420 or observed_ids != expected_ids:
        raise RuntimeError(
            f"Expected exactly 420 phase-aware bases; rows={len(rows)}, unique={len(observed_ids)}, "
            f"missing={len(expected_ids - observed_ids)}, extra={len(observed_ids - expected_ids)}"
        )
    for row in rows:
        path = basis_dir / f"{row['axon_id']}_1uA_VeBasis.npz"
        reference = Path(row["basis_npz"])
        if reference.is_absolute() or root / reference != path:
            raise RuntimeError(f"Nonportable or mismatched basis path for {row['axon_id']}: {reference}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "axon_id", "cutline_id", "diameter_um", "phase_id", "phase_fraction",
                "phase_offset_mm", "phase_scheme", "node_numbers", "V_neg_mV", "V_pos_mV",
                "n_valid_nodes", "readout_node_number",
            }
            if not required.issubset(archive.files):
                raise RuntimeError(f"Basis metadata is incomplete: {path}")

    index_path = root / "data" / "reference" / "index-screen" / "index_axon.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Preflight index screen is missing: {index_path}")
    index_rows = _read_rows(index_path)
    if len(index_rows) != 1:
        raise RuntimeError(f"Expected exactly one preflight index row: {index_path}")
    observed_index = index_rows[0]["axon_id"]
    if observed_index != expected_index:
        raise RuntimeError(f"Expected index {expected_index!r}, observed {observed_index!r}")
    thresholds_path = index_path.with_name("index_axon_thresholds.csv")
    threshold_rows = _read_rows(thresholds_path)
    if len(threshold_rows) != 420:
        raise RuntimeError(f"Expected 420 reference thresholds, found {len(threshold_rows)}")
    thresholds = np.asarray([float(row["I_first_uA"]) for row in threshold_rows], dtype=float)
    reference_counts = {
        f"{amplitude:g}": int(np.count_nonzero(thresholds <= amplitude + 1e-9))
        for amplitude in (300.0, 400.0, 500.0, 600.0, 750.0, 1000.0, 2000.0)
    }

    trajectory_jobs = population_trajectory_condition_jobs(root, manifest)
    if len(trajectory_jobs) != 3850:
        raise RuntimeError(f"Expected 3,850 recruitment tasks per branch, found {len(trajectory_jobs)}")
    return {
        "status": "ready",
        "cutlines": len(observed_cutlines),
        "diameters_um": list(POPULATION_DIAMETERS_UM),
        "nodal_phases_per_trajectory": 2,
        "axons": len(rows),
        "time_pattern_files": len(time_paths),
        "spacetime_files_expected": len(rows) * len(time_paths),
        "recruitment_tasks_per_branch": len(trajectory_jobs),
        "expected_index_axon_id": expected_index,
        "reference_I_first_uA": float(index_rows[0]["I_first_uA"]),
        "reference_impulse_recruited_by_amplitude_uA": reference_counts,
        "reference_threshold_median_uA": float(np.median(thresholds)),
        "basis_manifest": str(manifest),
        "spacetime_required_fields": sorted(SPACETIME_NPZ_FIELDS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-index-axon-id", default=EXPECTED_INDEX_AXON_ID)
    args = parser.parse_args()
    print(json.dumps(validate_preflight(args.root, args.expected_index_axon_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
